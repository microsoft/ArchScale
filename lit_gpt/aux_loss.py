import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce as diff_all_reduce
from typing import Literal

def activation_budget_bce(
    router_out: torch.Tensor,
    tau: float = 0.25,
    mask: torch.Tensor | None = None,
    lambda_budget: float = 1.0,
    eps: float = 1e-6,
    loss: Literal["bce", "mse"] = "mse",
    mode: Literal["per_sequence", "per_layer"] = "per_layer",
) -> torch.Tensor:
    """
    BCE/MSE budget loss to enforce an activation fraction ~ tau.

    Args:
        router_out: (B, L, T) tensor. Probabilities in (0,1).
        tau: target fraction (e.g., 0.25).
        mask: Optional (B, T) boolean/float mask for variable-length sequences.
              1/True for valid tokens, 0/False for pad.
        lambda_budget: multiplier for the auxiliary loss.
        eps: numerical stability.
        loss: which loss to use between per-sequence mean activation and target tau.
              "bce" (default) uses binary cross entropy; "mse" uses mean squared error.
        mode: aggregation mode for enforcing the budget:
              - "per_sequence": enforce the average across layers and tokens per sequence ~ tau.
              - "per_layer": enforce each layer's average across tokens per sequence ~ tau.

    Returns:
        Scalar tensor: lambda_budget * mean_B( BCE( m_b , tau ) ),
        where m_b is the expected active fraction per sequence.
    """
    p = router_out.clamp(eps, 1 - eps)

    # Expect (B, L, T)
    B, L, T = p.shape

    if mode == "per_sequence":
        if mask is None:
            # Mean over layers and tokens per sequence
            m_b = p.mean(dim=(1, 2))  # (B,)
        else:
            # mask: (B, T) -> broadcast to (B, 1, T)
            m = mask.to(p.dtype).clamp(min=0.0, max=1.0)
            m = m.unsqueeze(1)  # (B, 1, T)
            num_tokens = m.sum(dim=(1, 2))  # (B,)
            denom = (L * num_tokens).clamp_min(eps)  # (B,)
            m_b = (p * m).sum(dim=(1, 2)) / denom    # (B,)

        target = torch.full_like(m_b, tau)
        if loss == "bce":
            m_b_stable = m_b.clamp(eps, 1 - eps)
            budget_loss = F.binary_cross_entropy(m_b_stable, target, reduction="mean")
        elif loss == "mse":
            budget_loss = F.mse_loss(m_b, target, reduction="mean")
        else:
            raise ValueError(f"Unsupported loss type '{loss}'. Expected 'bce' or 'mse'.")

    elif mode == "per_layer":
        if mask is None:
            # Mean over tokens per layer, per sequence
            m_bl = p.mean(dim=2)  # (B, L)
        else:
            # mask: (B, T) -> broadcast to (B, 1, T)
            m = mask.to(p.dtype).clamp(min=0.0, max=1.0)
            m = m.unsqueeze(1)  # (B, 1, T)
            num_tokens = m.sum(dim=(1, 2)).clamp_min(eps)  # (B,)
            m_bl = (p * m).sum(dim=2) / num_tokens.unsqueeze(1)  # (B, L)

        target = torch.full_like(m_bl, tau)
        if loss == "bce":
            m_bl_stable = m_bl.clamp(eps, 1 - eps)
            budget_loss = F.binary_cross_entropy(m_bl_stable, target, reduction="mean")
        elif loss == "mse":
            budget_loss = F.mse_loss(m_bl, target, reduction="mean")
        else:
            raise ValueError(f"Unsupported loss type '{loss}'. Expected 'bce' or 'mse'.")
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'per_sequence' or 'per_layer'.")

    return lambda_budget * budget_loss


def load_balance_loss(
    expert_weights: torch.Tensor, # T x E 
    tokens_per_expert: torch.Tensor, # E
    mask: torch.Tensor | None = None,
    gamma: float = 1.0,
    eps: float = 1e-30,
    style: Literal["entropy", "switch", "switch_seq"] = "switch",
    global_stats: bool = False,
    seq_len: int | None = None,
) -> torch.Tensor:
    """
    This encourages balanced expert usage by minimizing negative entropy
    of the average expert probabilities across the sequence.
    
    Args:
        expert_weights: (T, N_E) tensor of router weights (post-softmax, sparse),
                      where T=total_tokens (B*seq_len), N_E=num_experts.
        tokens_per_expert: (E,) for global styles, or (B, E) for "switch_seq" style.
        mask: Optional (B, T) boolean/float mask for variable-length sequences.
              1/True for valid tokens, 0/False for pad.
        gamma: coefficient γ to scale the loss.
        eps: numerical stability for log computation.
        style: Loss style - "entropy", "switch", "switch_seq".
               "switch_seq" computes switch loss marginalized over sequences.
        global_stats: Whether to all-reduce across ranks for global statistics.
        seq_len: Sequence length, required for "switch_seq" style to reshape tensors.
    
    Returns:
        Scalar tensor: gamma * loss
    """
    if style == "switch_seq":
        # Switch loss marginalized over sequences
        # expert_weights: (T, E) where T = B * seq_len
        # tokens_per_expert: (B, E) - per-sequence counts
        T, E = expert_weights.shape
        if seq_len is None:
            raise ValueError("seq_len must be provided for 'switch_seq' style")
        B = T // seq_len
        
        # Reshape to (B, seq_len, E)
        expert_weights_seq = expert_weights.view(B, seq_len, E)
        
        # Compute per-sequence average probabilities: P_i = mean router prob for expert i per sequence
        # p: (B, E)
        p = expert_weights_seq.sum(dim=1)+ eps # (B, E)
        p = p / p.sum(dim=-1, keepdim=True)
        # tokens_per_expert should be (B, E) for switch_seq
        # f_i = fraction of tokens dispatched to expert i per sequence
        # f: (B, E)
        f = tokens_per_expert.float() / (tokens_per_expert.sum(dim=-1, keepdim=True) + eps)  # (B, E)
        
        # Switch loss per sequence: L_b = E * Σ_i f_i * P_i
        # Then average over batch
        balance_loss = E * (f * p).sum(dim=-1)  # (B,)
        balance_loss = balance_loss.mean()
        
        return gamma * balance_loss
    
    # Original global styles
    # router_logits: (T, E)
    T, E = expert_weights.shape
    # Compute average probabilities across the sequence: p = (1/T) Σ softmax(...)
    # if mask is None:
    # Simple average over sequence length, calculated over global batch with all-reduce
    p = expert_weights.sum(dim=-2)  # (E)
    
    # All-reduce across all ranks to get global statistics
    if global_stats and dist.is_available() and dist.is_initialized():
        p = diff_all_reduce(p, op=dist.ReduceOp.SUM)
    # else:
    #     # Masked average for variable-length sequences
    #     m = mask.to(probs.dtype).unsqueeze(-1)  # (T, 1)
    #     num_tokens = m.sum(dim=1).clamp_min(eps)  # (1)
    #     p = (probs * m).sum(dim=1) / num_tokens  # ( E)
    p_stable = p + eps
    
    if style == "entropy":
        # MoEUT: https://arxiv.org/abs/2405.16039
        # Compute balance loss: L = Σ p[e] log p[e]
        # This is the negative entropy (we minimize this to maximize entropy)
        balance_loss = (p_stable * torch.log(p_stable)).sum(dim=-1)  # (1,)
    elif style == "switch":
        # switch transformer: https://arxiv.org/pdf/2101.03961
        balance_loss =  tokens_per_expert.shape[0] * p_stable/ p_stable.sum() * tokens_per_expert/ tokens_per_expert.sum()
        balance_loss = balance_loss.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported style '{style}'. Expected 'entropy', 'switch', 'switch_seq'.")
    # Average over batch
    balance_loss = balance_loss.mean()
    
    return gamma * balance_loss


def compute_tokens_per_expert_seq(
    expert_indices: torch.Tensor,  # (B * seq_len, top_k) or (B, seq_len, top_k)
    n_experts: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Compute tokens_per_expert at sequence level for use with switch_seq style.
    
    Args:
        expert_indices: Tensor of expert indices from top-k routing.
                       Shape: (T, top_k) where T = B * seq_len, or (B, seq_len, top_k)
        n_experts: Number of experts.
        seq_len: Sequence length for reshaping.
    
    Returns:
        tokens_per_expert: (B, E) tensor counting tokens per expert per sequence.
    """
    # Handle different input shapes
    if expert_indices.dim() == 3:
        # Already (B, seq_len, top_k)
        B, S, K = expert_indices.shape
        expert_indices_flat = expert_indices.view(B, -1)  # (B, seq_len * top_k)
    else:
        # (T, top_k) where T = B * seq_len
        T, K = expert_indices.shape
        B = T // seq_len
        expert_indices_flat = expert_indices.view(B, -1)  # (B, seq_len * top_k)
    
    # Count tokens per expert per sequence using scatter
    # Create one-hot-like counts: (B, seq_len * top_k) -> (B, E)
    tokens_per_expert = torch.zeros(
        B, n_experts, 
        dtype=torch.int64, 
        device=expert_indices.device
    )
    
    # Use scatter_add to count
    ones = torch.ones_like(expert_indices_flat, dtype=torch.int64)
    tokens_per_expert.scatter_add_(dim=1, index=expert_indices_flat, src=ones)
    
    return tokens_per_expert
