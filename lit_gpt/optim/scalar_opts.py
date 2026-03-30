# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# modified from https://github.com/microsoft/dion/blob/main/dion/scalar_opts.py
# for adam-hyperball support

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from typing import Generator, List, Optional, Tuple


@torch.compile(fullgraph=True)
def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """
    AdamW optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    if cautious_wd:
        # Compute update direction (pre-LR) for CWD mask
        update_dir = M / denom

        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (update_dir * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)


@torch.compile(fullgraph=True)
def lion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    G = G.to(M.dtype)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = M.lerp(G, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    M.lerp_(G, 1 - beta2)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (U * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)


@torch.compile(fullgraph=True)
def adamw_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
    use_hyperball: bool = False,  # Deprecated: hyperball is handled in adamw_update_foreach_async
):
    """
    AdamW optimizer algorithm (foreach implementation).

    Note: Hyperball normalization (use_hyperball=True) is handled separately in
    adamw_update_foreach_async with proper global norm computation via all-reduce.
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    M_div = torch._foreach_div(M, denom)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, M_div)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Standard weight update
    # X = X - adj_lr * M / denom
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_sub_(X, M_div)


@torch.compile(fullgraph=True)
def lion_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)

    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = torch._foreach_lerp(M, G, [1 - beta1] * batch_size)
    torch._foreach_sign_(U)

    # Update momentum in place with new gradient
    # M = beta2 * M + (1 - beta2) * G
    torch._foreach_lerp_(M, G, [1 - beta2] * batch_size)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    torch._foreach_mul_(U, lr)
    torch._foreach_sub_(X, U)


@torch.compile(fullgraph=True)
def hyperball_weight_decay_update_and_local_sq_norms(
    X: List[Tensor],
    U: List[Tensor],
    lr: Tensor,
    weight_decay: Tensor,
    epsilon: float,
    cautious_wd: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Apply weight decay for hyperball and compute local squared norms.
    U should be the bias-corrected update direction (M / denom / bias_correction1).
    Returns (x_sq_norms, u_sq_norms) as stacked tensors of shape (N, 1).
    These need to be all-reduced across ranks to get global squared norms.
    """
    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)
        decay_masks = torch._foreach_add(decay_masks, 1)
        decay_masks = torch._foreach_minimum(decay_masks, 1)

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Compute local norms and return squared values for all-reduce
    x_norms = torch._foreach_norm(X)
    u_norms = torch._foreach_norm(U)

    x_sq = torch.stack([n.reshape(1) for n in x_norms]).square()
    u_sq = torch.stack([n.reshape(1) for n in u_norms]).square()

    return x_sq, u_sq


@torch.compile(fullgraph=True)
def hyperball_apply_scaled_update_and_local_sq_norms(
    X: List[Tensor],
    U: List[Tensor],
    x_norms: Tensor,
    u_norms: Tensor,
    lr: Tensor,
    epsilon: float,
) -> Tensor:
    """
    Apply hyperball scaled update using global norms.
    x_norms and u_norms should be global norms (shape N, 1).
    Returns local squared norms of new X for subsequent all-reduce.
    """
    # Clamp u_norms to avoid division by zero
    u_norms_clamped = u_norms.clamp_min(epsilon)

    # Compute scale factors: x_norm / u_norm for each tensor
    scale_factors = x_norms / u_norms_clamped
    scale_list = [s.squeeze(0) for s in scale_factors.unbind(0)]

    # Scale updates: U * (x_norm / u_norm) * lr
    U_scaled = torch._foreach_mul(U, scale_list)
    torch._foreach_mul_(U_scaled, lr)

    # Apply update: X = X - U_scaled
    torch._foreach_sub_(X, U_scaled)

    # Compute local squared norms of new X for all-reduce
    new_norms = torch._foreach_norm(X)
    new_sq = torch.stack([n.reshape(1) for n in new_norms]).square()

    return new_sq


@torch.compile(fullgraph=True)
def hyperball_renormalize(
    X: List[Tensor],
    x_norms: Tensor,
    new_x_norms: Tensor,
    epsilon: float,
):
    """
    Renormalize X to preserve original global norm.
    x_norms: original global norms (N, 1)
    new_x_norms: new global norms after update (N, 1)
    """
    new_x_norms_clamped = new_x_norms.clamp_min(epsilon)
    scale_factors = x_norms / new_x_norms_clamped
    scale_list = [s.squeeze(0) for s in scale_factors.unbind(0)]

    torch._foreach_mul_(X, scale_list)


@torch.compile(fullgraph=True)
def hyperball_compute_update_direction(
    M: List[Tensor],
    V: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: int,
    epsilon: float,
) -> List[Tensor]:
    """
    Compute the bias-corrected AdamW update direction: M / denom / bias_correction1.
    This is separated out so it can be compiled while the all-reduce happens outside.
    """
    batch_size = len(M)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # Compute the denominator
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Compute bias-corrected update direction
    M_div = torch._foreach_div(M, denom)
    torch._foreach_div_(M_div, bias_correction1)

    return M_div


def adamw_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
    use_hyperball: bool = False,
    process_group: Optional[ProcessGroup] = None,
) -> Generator[None, None, None]:
    """
    Async wrapper for AdamW update with proper global norm handling for hyperball.
    """
    if use_hyperball:
        # Hyperball with global norms: need all-reduce for distributed training
        batch_size = len(X)
        M_dtype = M[0].dtype
        V_dtype = V[0].dtype

        # Update momentum and variance (same as standard AdamW)
        G_typed = [g.to(dtype=M_dtype) for g in G]
        torch._foreach_lerp_(M, G_typed, [1 - beta1] * batch_size)

        G_square = torch._foreach_mul(G_typed, G_typed)
        G_square = [g.to(dtype=V_dtype) for g in G_square]
        torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

        # Compute bias-corrected update direction
        U = hyperball_compute_update_direction(M, V, beta1, beta2, step, epsilon)

        # Step 1: Apply weight decay and compute local squared norms
        x_sq, u_sq = hyperball_weight_decay_update_and_local_sq_norms(
            X, U, lr, weight_decay, epsilon, cautious_wd
        )

        # Step 2: All-reduce to get global squared norms, then sqrt
        if process_group is not None:
            dist.all_reduce(x_sq, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(u_sq, op=dist.ReduceOp.SUM, group=process_group)
        x_norms = x_sq.sqrt()
        u_norms = u_sq.sqrt()

        # Step 3: Apply scaled update and compute new local squared norms
        new_x_sq = hyperball_apply_scaled_update_and_local_sq_norms(
            X, U, x_norms, u_norms, lr, epsilon
        )

        # Step 4: All-reduce for new norms
        if process_group is not None:
            dist.all_reduce(new_x_sq, op=dist.ReduceOp.SUM, group=process_group)
        new_x_norms = new_x_sq.sqrt()

        # Step 5: Renormalize to preserve original global norm
        hyperball_renormalize(X, x_norms, new_x_norms, epsilon)
    else:
        adamw_update_foreach(
            X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd, use_hyperball=False
        )
    yield


def lion_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay, cautious_wd)
    yield