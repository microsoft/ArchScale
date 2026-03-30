# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
from lit_gpt.config import Config
import torch.nn.functional as F

import utils

from .fused_rotary_embedding import apply_rotary_emb_func
from .moe_utils import _permute, _unpermute
from .gated_memory_unit import swiglu, glu
from .attention import build_rope_cache
from .aux_loss import load_balance_loss, compute_tokens_per_expert_seq
import torch.distributed as dist

# Optional SonicMoE import
try:
    from .smoe_functional import moe_TC_softmax_topk_layer
    from sonicmoe.enums import ActivationType, is_glu
    SONICMOE_AVAILABLE = True
except:
    SONICMOE_AVAILABLE = False
        
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]

def stable_topk(scores, k, dim=-1):
    # scores: [..., E]
    scores32 = scores.float() 
    order = torch.argsort(scores32, dim=dim, descending=True, stable=True)
    idx = order.narrow(dim, 0, k)
    val = scores32.gather(dim, idx)
    return idx.contiguous(), val.to(scores.dtype).contiguous()


class Router(nn.Module):
    def __init__(self, config: Config, route_size: int) -> None:
        super().__init__()
        self.config = config
        if config.share_expert:
            assert config.top_k > 1, "shared expert requires top_k > 1"
            route_top_k = config.top_k-1
        else:
            route_top_k = config.top_k
        self.top_k = route_top_k
        self.softmax_after_topk = True # https://arxiv.org/abs/1701.06538, also gpt-oss 
        self.lc_proj= nn.Linear(config.n_embd, route_size, bias=config.bias)
        self.stable_topk = True
        # mup scaling: https://arxiv.org/abs/2508.09752
        if config.mup and not config.use_muon:
            self.logit_scale = config.mup_d0 / config.n_layer 
        else:
            self.logit_scale = 1.0
    
    def reset_parameters(self) -> None:
        """Initialize router parameters."""
        if self.config.use_muon:
            utils.torch_default_init(self.lc_proj.weight)
        else:
            torch.nn.init.normal_(self.lc_proj.weight, std=0.02)
        if self.lc_proj.bias is not None:
            nn.init.zeros_(self.lc_proj.bias)

    #@torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x =  x.float() @ self.lc_proj.weight.t().float()
        if self.lc_proj.bias is not None:
            x = x + self.lc_proj.bias.float()
        x = x * self.logit_scale # T,E 
        router_logits = x
        if not self.softmax_after_topk:
            x = torch.nn.functional.softmax(x.to(torch.float32), dim=-1)
        if self.stable_topk:
            expert_indices, expert_weights = stable_topk(x, k=self.top_k, dim=-1)
        else:
            experts = torch.topk(x, k=self.top_k, dim=-1, sorted=True)
            expert_indices = experts.indices
            expert_weights = experts.values # T, K
        if self.softmax_after_topk:
            expert_weights = torch.nn.functional.softmax(expert_weights.to(torch.float32), dim=-1)
        return expert_indices, expert_weights, router_logits


def to_group_sequence(counts: torch.Tensor) -> torch.Tensor:
    # counts: shape [L], integer counts for labels 1..L
    counts = counts.to(dtype=torch.int32)
    labels = torch.arange(1, counts.numel() + 1, device=counts.device, dtype=torch.int32)
    return torch.repeat_interleave(labels, counts)

def mlp_func(x: torch.Tensor) -> torch.Tensor:
    x_gate, x_inp = x.chunk(2, dim=-1)
    x_mlp = swiglu(x_gate, x_inp)
    return x_mlp

class MoE(nn.Module):
    def __init__(self, config: Config, is_attn = False, is_mlp = True) -> None:
        super().__init__()
        import transformer_engine.pytorch as te
        # Mixture of Experts Layer:
        # We support both attention and mlp experts,
        # attention is computed on tokens allocated to each expert through packing and unpacking, 
        # similar to seqboat: https://arxiv.org/abs/2306.11197
        self.config = config
        self.global_aux = config.global_aux
        self.aux_gamma = config.aux_gamma
        self.aux_style = config.aux_style  # Options: "entropy", "switch", "switch_seq"
        self.sqrt_gate = config.sqrt_gate # sqrt router gate  # Var(ax) = a^2 * Var(x)
        self.top_k = config.top_k # maximum activated mods per loop
        self.n_head = config.n_head // config.top_k
        self.head_size = config.head_size
        self.share_expert = config.share_expert
        self.is_attn = is_attn
        self.is_mlp = is_mlp
        self.n_query_groups = config.n_query_groups // config.top_k
        n_mods = config.sparsity * self.top_k
        self.total_experts = n_mods
        if self.share_expert:
            self.top_k = self.top_k-1 
            n_mods = n_mods-1
        self.route_size = n_mods
        self.n_mods = n_mods
        if self.is_attn:
            mod_in_size = (self.n_head + 2 * self.n_query_groups) * config.head_size
            mod_out_size = self.n_head * config.head_size
            if self.config.gated_attn:
                mod_in_size = mod_in_size + config.head_size * self.n_head
            self.n_attn=n_mods

        elif self.is_mlp:
            mod_in_size = int(config.intermediate_size*2) // config.top_k
            mod_out_size = config.intermediate_size // config.top_k
            self.n_attn = 0
        if self.share_expert:
            assert not self.is_attn, "shared expert is currently not supported for attention"
            self.shared_mod_in = nn.Linear(config.n_embd, mod_in_size, bias=config.bias)
            self.shared_mod_out = nn.Linear(mod_out_size, config.n_embd, bias=config.bias)

        self.mod_out_size = mod_out_size
        self.router = Router(config, self.route_size)
        self.mods_in = te.GroupedLinear(n_mods, config.n_embd, mod_in_size , bias= config.bias, device="meta",
                        init_method = partial(nn.init.kaiming_uniform_, a=math.sqrt(5)))

        self.mods_out = te.GroupedLinear(n_mods, mod_out_size , config.n_embd, bias= config.bias, device="meta",
                        init_method = partial(nn.init.kaiming_uniform_, a=math.sqrt(5))) 
                        # partial(nn.init.kaiming_uniform_, a=math.sqrt(6 * config.top_k - 1))
                        # scale up initialization for fine-grained moe coz mod_out_size is 
                        # smaller than the actual mixing size, a is calculated as:
                        # solving x for 2/(1+x^2) *3 =1/self.top_k
        if self.is_attn:
            # attn ops config
            # self.n_head = config.n_head
            self.scale = config.attn_scale if config.attn_scale is not None else 1.0 / math.sqrt(self.head_size) 
            if self.config.mup:
                self.scale = self.scale * math.sqrt(self.config.mup_hd0)/ math.sqrt(self.head_size) 
            if self.config.local_window > -1:
                self.win_tuple = (self.config.local_window-1, 0)
            else:
                self.win_tuple = (-1,-1)
            self.use_cu_seqlen = config.use_cu_seqlen
        
            compute_capability = torch.cuda.get_device_capability()
            sm_version = compute_capability[0] * 10 + compute_capability[1]
            if sm_version == 100:
                import flash_attn.cute as flash_attn_interface
                print("Using flash_attn 4 for gb200!")
            elif sm_version == 90:
                try:
                    import flash_attn_interface
                    print("Using flash_attn 3.0.0 for hopper!")
                except:
                    print("Using flash_attn 2.8.1 for hopper!")
                    from flash_attn import flash_attn_interface
            else:
                from flash_attn import flash_attn_interface
            self.attn_func = partial(flash_attn_interface.flash_attn_varlen_func, causal=True, softmax_scale= self.scale, window_size = self.win_tuple)

    def reset_parameters(self) -> None:
        """Initialize MoE parameters."""
        # Reset router
        self.router.reset_parameters()
        
        # Reset mods_in and mods_out (te.GroupedLinear)
        # GroupedLinear from transformer_engine has its own reset_parameters
        self.mods_in.reset_parameters()
        self.mods_out.reset_parameters()
        
        # Reset shared expert if present
        if self.share_expert:
            utils.torch_default_init(self.shared_mod_in.weight)
            if self.shared_mod_in.bias is not None:
                nn.init.zeros_(self.shared_mod_in.bias)
            utils.torch_default_init(self.shared_mod_out.weight)
            #nn.init.kaiming_uniform_(self.shared_mod_out.weight, a=math.sqrt(6 * self.total_experts - 1))
            if self.shared_mod_out.bias is not None:
                nn.init.zeros_(self.shared_mod_out.bias)

    @torch.compile(dynamic=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:   
       
        B, seq_len, C = x.shape
        x = x.view(-1, C)
        if self.share_expert:
            x_smlp = self.shared_mod_in(x)
            x_smlp = mlp_func(x_smlp)
            x_smlp = self.shared_mod_out(x_smlp)
        ox = x
        expert_indices, expert_weights, router_logits = self.router(x) # T, K
        self.router_z_mean = (torch.logsumexp(router_logits.float(), dim=-1) ** 2).mean()
        x, sorted_weights, token_indices, tokens_per_expert = \
                self.sort_tokens(x.bfloat16(), expert_indices, expert_weights, self.route_size)
        tokens_per_expert_global = tokens_per_expert
        if self.aux_gamma > 0:
            # Convert to sparse T x E matrix
            T = expert_weights.shape[0]
            E = self.route_size
            expert_weights_sparse = F.softmax(router_logits.float(), dim=-1)
            # expert_weights_sparse = torch.zeros(T, E, dtype=expert_weights.dtype, device=expert_weights.device)
            # expert_weights_sparse.scatter_(dim=1, index=expert_indices, src=expert_weights)
            
            if self.aux_style == "switch_seq":
                # Compute tokens_per_expert at sequence level for switch_seq
                tokens_per_expert_seq = compute_tokens_per_expert_seq(
                    expert_indices, n_experts=E, seq_len=seq_len
                )
                self.aux_loss = load_balance_loss(
                    expert_weights_sparse, tokens_per_expert_seq, 
                    gamma=self.aux_gamma, global_stats=False,
                    style=self.aux_style, seq_len=seq_len
                )
            else:
                if self.global_aux and dist.is_available() and dist.is_initialized():
                    tokens_per_expert_global = tokens_per_expert.clone()
                    dist.all_reduce(tokens_per_expert_global, op=dist.ReduceOp.SUM)
                self.aux_loss = load_balance_loss(
                    expert_weights_sparse, tokens_per_expert_global, 
                    gamma=self.aux_gamma, global_stats=self.global_aux,
                    style=self.aux_style
                )         
        self.token_indices = token_indices.detach().cpu()
        self.tokens_per_expert = tokens_per_expert_global.tolist() # E # for logging
        # padding for bf16/fp8 grouped linear, padding idx = -1
        permuted_indices, num_tokens_per_expert = _permute(
            tokens_per_expert, token_indices.shape[0], ep_degree = 1, num_local_experts = self.n_mods
        )
        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        if self.config.fuse_sort_and_padding:
            input_shape = (token_indices.shape[0] + 1, x.shape[-1])
            sorted_weights = F.pad(sorted_weights, pad=(0,1), value=0.0)
            sorted_weights = sorted_weights[permuted_indices]
        else:
            input_shape = x.shape
        # do sort tokens and padding for x together coz we don't do EP
        # we also fuse sort and padding for unpermute with padded gates
        padded_token_indices = F.pad(token_indices, pad=(0,1), value=x.shape[0]-1).to(torch.int32) 
        padded_token_indices = padded_token_indices[permuted_indices]
        if self.config.fuse_sort_and_padding:
            x = x[padded_token_indices, :]
        else:
            x = x[permuted_indices, :]
        # final padding tokens mapped to last expert
        num_tokens_per_expert[-1] = num_tokens_per_expert[-1] + x.shape[0] - num_tokens_per_expert.sum().item()  
        tokens_per_expert_list = num_tokens_per_expert.tolist()
        
        # torch compile is unhappy with dynamic shape :-(
        if self.is_attn:
            n_attn_tokens = x.shape[0]
        else:
            n_attn_tokens = 0

        if self.is_attn:
            with torch.no_grad():
                pad_per_expert = num_tokens_per_expert - tokens_per_expert # E
                num_tokens_padded = torch.stack((tokens_per_expert[:self.n_attn], pad_per_expert[:self.n_attn]), dim=-1).flatten()
                tokens_padded_group = to_group_sequence(num_tokens_padded)
                # print(tokens_padded_group)
                token_group = padded_token_indices // seq_len
                # combine groups
                token_group = (token_group.max()+1) * tokens_padded_group + token_group
                # print(token_group)
                n = token_group.numel()
                # 1. Find the start indices of each run
                #    - 0 is always a start
                #    - plus any position where the value changes from the previous
                change_points = torch.nonzero(token_group[1:] != token_group[:-1]).flatten() + 1
                starts = torch.cat((torch.tensor([0], device=token_group.device), change_points))
                # 2. Add a sentinel index at the end (n) and take differences
                ends = torch.cat((starts[1:], torch.tensor([n], device=token_group.device)))
                tokens_per_expert_attn = ends - starts
                cu_seqlens = F.pad(tokens_per_expert_attn.cumsum(0), pad=(1,0), value=0).to(torch.int32)
                if self.config.nope:
                    rope = None
                else:
                    # todo: we can also use permuted_indices so the relative distance doesn't go beyond trained dsistance
                    rope = self.build_rope_cache(padded_token_indices[:n_attn_tokens])

        all_in = self.mods_in(x, tokens_per_expert_list) # T * top_k, C
        if self.is_mlp:
            # mlp split
            x_mlp = all_in[n_attn_tokens:, :]
            x_mlp = mlp_func(x_mlp)

        if self.is_attn:
            ## qkv split
            qkv = all_in[:n_attn_tokens, :]
            if self.config.gated_attn:
                gate_dim = self.head_size * self.n_head
                qkv, attn_gate = qkv.split((qkv.shape[-1] - gate_dim, gate_dim), dim=-1)
            # if self.config.use_cu_seqlen:
            #     sorted_seq_ids = sequence_ids[token_indices]
            #     mask, _ = build_segment_cu_seqlen(sorted_seq_ids, tokens_per_expert)
            # else:
            #     mask = F.pad(tokens_per_expert.cumsum(0), pad=(1,0), value=0).to(torch.int32)
            q_per_kv = self.n_head // self.n_query_groups
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.reshape(-1, self.n_query_groups, total_qkv, self.head_size) # ( T * top_k, n_query_groups, total_qkv, hs)
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
            q = q.reshape(1, n_attn_tokens, q_per_kv*self.n_query_groups, self.head_size)  # (B, T, nh_ q, hs)
            k = k.reshape(1, n_attn_tokens, self.n_query_groups, self.head_size)  
            v = v.reshape(n_attn_tokens, self.n_query_groups, self.head_size)  
            if rope is not None:         
                cos, sin = rope
                # fused rope expect (batch_size, seqlen, nheads, headdim)
                q = apply_rotary_emb_func(q, cos, sin, False, True)
                k = apply_rotary_emb_func(k, cos, sin, False, True)
                # q = apply_rotary_emb_torch(q, cos, sin)
                # k = apply_rotary_emb_torch(k, cos, sin)
            x_attn = self.scaled_dot_product_attention(q.squeeze(0), k.squeeze(0), v, cu_seqlens=cu_seqlens).reshape(n_attn_tokens, self.mod_out_size) 

            if self.config.gated_attn:
                x_attn = glu(attn_gate, x_attn)

        all_out = x_attn if self.is_attn else x_mlp
        ## attn out
        # attn_out_weight = self.attn_out_weight[expert_indices, ...] # T, K, o, i
        # x = torch.einsum("tkoi,kti->tko", attn_out_weight, y) 
        x = self.mods_out(all_out, tokens_per_expert_list) # T * top_k, C   
        if self.config.fuse_sort_and_padding:
            gate = sorted_weights.unsqueeze(-1)
            if self.config.sqrt_gate:
                gate = torch.sqrt(gate + 1e-30)
            x = (x.float() *  gate).type_as(x)
            out = ox.new_zeros((ox.shape[0]+1, ox.shape[1])) 
            x = out.scatter_add(dim=0, index=padded_token_indices.unsqueeze(-1).expand(-1, x.shape[-1]), src=x)
            x = x[:-1]
        else:
            x = _unpermute(x, input_shape, permuted_indices)
            gate = sorted_weights.unsqueeze(-1)
            if self.config.sqrt_gate:
                gate = torch.sqrt(gate + 1e-30)
            x = (x.float() *  gate).type_as(x)
            out = torch.zeros_like(ox) # T, C
            x = out.scatter_add(dim=0, index=token_indices.unsqueeze(-1).expand(-1, x.shape[-1]), src=x)
        if self.share_expert:
            if self.config.sqrt_gate:
                p = 0.5
                x = math.sqrt(1-p) * x + math.sqrt(p) * x_smlp
            else:
                x = x + x_smlp
        x = x.reshape(B, seq_len, -1)
        return x

    @torch.compiler.disable()
    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
    ):
              
        seqlen_q = q.shape[0]
        cu_seqlens_k = cu_seqlens_q = cu_seqlens.int()
        max_seqlen_q = max_seqlen_k = (cu_seqlens - cu_seqlens.roll(1)).max().cpu().item()

        attn_output =self.attn_func(q, k, v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, 
                                        cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,)
        attn_output = attn_output.reshape(seqlen_q, q.shape[-2], q.shape[-1])
        return attn_output

    def sort_tokens(self, x, topk_ids, topk_weights, n_experts):
        # modified from https://github.com/pytorch/torchtitan/blob/main/experiments/deepseek_v3/model.py
        # This part sorts the token indices so that tokens routed to the same expert reside consecutively.
        # An implication is that tokens to the same "expert group" (i.e., device) are also consecutive.

        # topk_ids: [seq_len, topk]
        with torch.no_grad():
            # [seq_len, n_experts]
            expert_counts = topk_ids.new_zeros(
                (topk_ids.shape[0], n_experts)
            )
            # Fill 1 to the selected experts
            expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = expert_counts.sum(dim=0)
            # Token indices for each expert
            token_indices = topk_ids.view(-1).argsort(stable=True) 
            sorted_weights = topk_weights.view(-1)[token_indices] # T*k
            token_indices = token_indices // topk_ids.shape[1] # T*k
        if self.config.fuse_sort_and_padding:
            sorted_tokens = x #fused later with padding  # T*k, C 
        else:
            sorted_tokens = x[token_indices] # T*k, C 
        return (sorted_tokens, sorted_weights, token_indices, tokens_per_expert)


    def build_rope_cache(self, seq_idx: torch.Tensor) -> RoPECache:
        initial_window = self.config.local_window if self.config.local_window > 0 else self.config.block_size
        cos, sin = build_rope_cache(
            seq_idx=seq_idx,
            n_elem=int(self.config.rotary_percentage * self.head_size),
            dtype=torch.bfloat16,
            device=seq_idx.device,
            base = 0.09 * initial_window ** 1.628, # Base of RoPE Bounds Context Length https://arxiv.org/abs/2405.14591
            scaling_factor=self.config.scaling_factor,
        )
        rope = (cos, sin)
        return rope


class SonicMoE(nn.Module):
    """
    todo: add sqrt_gate
    Wrapper for SonicMoE (https://github.com/Dao-AILab/sonic-moe) to provide
    a compatible interface with the existing MoE implementation.
    
    SonicMoE provides IO-aware and tile-aware optimizations for MoE layers,
    offering significant speedups on NVIDIA Hopper GPUs (H100, H200, etc.).
    
    Muon support:
        Expert weights are stored as 3D tensors (num_experts, out_features, in_features)
        in the Experts class. With Muon's flatten=False (default), the Newton-Schulz
        orthogonalization treats these as batches of 2D matrices, orthogonalizing each
        expert's weight matrix independently. This is handled automatically by the
        Muon optimizer for any 3D parameter in the "muon" param group.
    
    """
    
    def __init__(self, config: Config, is_attn: bool = False, is_mlp: bool = True) -> None:
        super().__init__()
        
        if not SONICMOE_AVAILABLE:
            raise ImportError(
                "SonicMoE is not installed. Please install it via:\n"
                "  pip install git+https://github.com/Dao-AILab/sonic-moe.git\n"
                "Or set use_sonicmoe=False in your config."
            )
        
        if is_attn:
            raise NotImplementedError(
                "SonicMoE wrapper currently only supports MLP experts (is_mlp=True). "
                "Attention experts are not yet supported with SonicMoE."
            )
        
        self.config = config
        self.is_attn = is_attn
        self.is_mlp = is_mlp
        self.aux_gamma = config.aux_gamma
        self.aux_style = config.aux_style  # Options: "entropy", "switch", "switch_seq"
        self.global_aux = config.global_aux
        if self.aux_style == "switch_seq":
            self.global_aux = False
        self.share_expert = config.share_expert
        
        # Calculate number of experts
        n_mods = config.sparsity * config.top_k
        self.total_experts = n_mods
        self.top_k = config.top_k
        
        if self.share_expert:
            assert config.top_k > 1, "shared expert requires top_k > 1"
            self.top_k = config.top_k - 1
            n_mods = n_mods - 1
        
        self.n_mods = n_mods
        
        self.intermediate_size = config.intermediate_size // config.top_k

        self.router = nn.Linear(in_features=config.n_embd, out_features=n_mods, bias=False)

        self.activation_function = ActivationType.SWIGLU

        mod_in_size = 2 * self.intermediate_size if is_glu(self.activation_function) else self.intermediate_size

        self.mods_in = Experts(
            num_experts=n_mods,
            in_features=config.n_embd,
            out_features=mod_in_size,
            bias=config.bias,
        )

        self.mods_out = Experts(
            num_experts=n_mods,
            in_features=self.intermediate_size,
            out_features=config.n_embd,
            bias=config.bias,
        )

        if self.share_expert:
            self.shared_mod_in = nn.Linear(config.n_embd, mod_in_size, bias=config.bias)
            self.shared_mod_out = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)

        if config.mup and not config.use_muon:
            self.logit_scale = config.mup_d0 / config.n_layer 
        else:
            self.logit_scale = 1.0
        
        self.stream_id = torch.cuda.current_stream().cuda_stream    
    
    def reset_parameters(self) -> None:
        """
        Initialize SonicMoE parameters to match the existing MoE initialization scheme.
        
        SonicMoE structure:
        - self.router: nn.Linear (hidden_size -> num_experts, no bias)
        - self.mods_in: Experts with weight shape (num_experts, out_features, in_features)
        - self.mods_out: Experts with weight shape (num_experts, out_features, in_features)
        
        Initialization:
        - Router weights: normal_(std=0.02) or torch_default_init (kaiming) if using muon
        - Expert weights (mods_in, mods_out): kaiming_uniform_(a=sqrt(5)) per expert matrix,
          matching te.GroupedLinear. GPT.reset_parameters() may further re-init to
          normal_(std=0.02) * w_init_scale for non-mup configs.
        - Biases: zeros
        
        Muon compatibility:
        - Expert weights are 3D (num_experts, out_features, in_features). With Muon's
          flatten=False, Newton-Schulz orthogonalizes each expert's 2D matrix independently.
        - Expert weights go to the "muon" param group (they contain "weight" in name
          and don't match vector_names). Router weights also go to "muon" (2D matrix).
        """
        # Initialize router weights (matches our Router class initialization)
        if self.config.use_muon:
            utils.torch_default_init(self.router.weight)
        else:
            torch.nn.init.normal_(self.router.weight, std=0.02)
        
        # Initialize expert weights with kaiming_uniform_ (matches te.GroupedLinear)
        self._init_expert_weights(self.mods_in.weight)
        if self.mods_in.bias is not None:
            nn.init.zeros_(self.mods_in.bias)
        
        self._init_expert_weights(self.mods_out.weight)
        if self.mods_out.bias is not None:
            nn.init.zeros_(self.mods_out.bias)
        
        if self.share_expert:
            utils.torch_default_init(self.shared_mod_in.weight)
            if self.shared_mod_in.bias is not None:
                nn.init.zeros_(self.shared_mod_in.bias)
            utils.torch_default_init(self.shared_mod_out.weight)
            if self.shared_mod_out.bias is not None:
                nn.init.zeros_(self.shared_mod_out.bias)
    
    @torch.no_grad()
    def _init_expert_weights(self, weight: torch.Tensor) -> None:
        # weight shape: (num_experts, out_features, in_features)
        # For each expert, apply kaiming_uniform_ to the 2D (out_features, in_features) matrix
        from torch.distributed.tensor import DTensor, Shard, Replicate

        if not isinstance(weight, DTensor):
            for i in range(weight.size(0)):
                utils.torch_default_init(weight[i])
            return

        original_placements = weight.placements
        has_shard = any(isinstance(p, Shard) for p in original_placements)

        if not has_shard:
            local = weight.to_local()
            for i in range(local.size(0)):
                utils.torch_default_init(local[i])
            return

        replicated_placements = [Replicate() for _ in original_placements]
        weight_rep = weight.redistribute(placements=replicated_placements)
        local = weight_rep.to_local()
        for i in range(local.size(0)):
            utils.torch_default_init(local[i])
        weight_sharded = weight_rep.redistribute(placements=original_placements)
        weight.to_local().copy_(weight_sharded.to_local())

    @torch.compiler.disable()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using SonicMoE.
        
        Args:
            x: Input tensor of shape (B, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (B, seq_len, hidden_size)
        """
        B, seq_len, C = x.shape
        
        # Flatten batch and sequence dimensions for SonicMoE
        hidden_states = x.view(-1, C)
        
        if self.share_expert:
            x_smlp = self.shared_mod_in(hidden_states)
            x_smlp = mlp_func(x_smlp)
            x_smlp = self.shared_mod_out(x_smlp)
        
        hidden_states, router_logits, expert_frequency, expert_indices = moe_TC_softmax_topk_layer(
            hidden_states,
            self.router.weight,
            self.mods_in.weight.permute(1, 2, 0),
            self.mods_in.bias,
            self.mods_out.weight.permute(1, 2, 0),
            self.mods_out.bias,
            self.top_k,
            self.stream_id,
            self.activation_function,
            not self.training,
            self.config.sqrt_gate,
            self.logit_scale,
        )
        self.router_z_mean = (torch.logsumexp(router_logits.float(), dim=-1) ** 2).mean()
        if self.global_aux and dist.is_available() and dist.is_initialized():
            dist.all_reduce(expert_frequency, op=dist.ReduceOp.SUM)

        self.token_indices = (expert_indices.reshape(-1).argsort(stable=True)//self.top_k).detach().cpu() # T, K # for logging
        self.tokens_per_expert = expert_frequency.tolist() # E # for logging
        if self.aux_gamma > 0:
            if self.aux_style == "switch_seq":
                # Compute tokens_per_expert at sequence level for switch_seq
                expert_frequency = compute_tokens_per_expert_seq(
                    expert_indices, n_experts=self.n_mods, seq_len=seq_len
                )
            self.aux_loss = load_balance_loss(
                        F.softmax(router_logits, dim=-1, dtype=torch.float32), expert_frequency,
                        gamma=self.aux_gamma, global_stats=self.global_aux,
                        style=self.aux_style
                    )
        
        if self.share_expert:
            if self.config.sqrt_gate:
                p = 0.5
                hidden_states = math.sqrt(1 - p) * hidden_states + math.sqrt(p) * x_smlp
            else:
                hidden_states = hidden_states + x_smlp
        
        # Reshape back to (B, seq_len, hidden_size)
        return hidden_states.view(B, seq_len, -1)

class Experts(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))


def get_moe(config: Config, is_attn: bool = False, is_mlp: bool = True):
    if config.use_sonicmoe:
        return SonicMoE(config, is_attn=is_attn, is_mlp=is_mlp)
    else:
        return MoE(config, is_attn=is_attn, is_mlp=is_mlp)
