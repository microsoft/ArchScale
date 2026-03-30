# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
from lit_gpt.config import Config
from .diff_attn import FlashDiffAttention
from .fused_rotary_embedding import apply_rotary_emb_func
import torch.nn.functional as F
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn
except:
    causal_conv1d_fn = None
from .bert_padding import pad_input, unpad_input
from .gated_memory_unit import glu
import utils

torch._dynamo.config.capture_scalar_outputs = True
unpad_input =torch.compiler.disable(unpad_input)
pad_input = torch.compiler.disable(pad_input)

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int , n_embd: int, yoco_cross = False,) -> None:
        super().__init__()
        self.config = config
        self.yoco_cross = yoco_cross
        self.local = layer_idx % config.full_per_layer < config.full_per_layer-1
            
        self.head_size = config.head_size
        self.n_head = config.n_head
        self.n_query_groups = config.n_query_groups
        if config.separate_qkv:
            self.q_proj = nn.Linear(n_embd, self.head_size * self.n_head, bias=config.attn_bias)
            if not yoco_cross:
                self.k_proj = nn.Linear(n_embd, self.head_size * self.n_query_groups, bias=False)
                self.v_proj = nn.Linear(n_embd, self.head_size * self.n_query_groups, bias=config.attn_bias)
            if config.gated_attn:
                # we use per head gate for gated attention
                self.g_proj = nn.Linear(n_embd, self.n_head, bias=config.attn_bias)
        else:
            if yoco_cross:
                shape = self.head_size * self.n_head
            else:
                shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
            if config.gated_attn:
                shape = shape + self.n_head
            self.attn_shape = shape
            # key, query, value projections for all heads, but in a batch
            self.attn = nn.Linear(n_embd, shape, bias=config.attn_bias)
        # attn ops
        self.scale = config.attn_scale if config.attn_scale is not None else 1.0 / math.sqrt(self.head_size) 
        if self.config.mup:
            self.scale = self.scale * math.sqrt(self.config.mup_hd0)/ math.sqrt(self.head_size) 
        if self.local and self.config.local_window > -1:
            self.win_tuple = (self.config.local_window-1, 0)
            self.window_size = self.win_tuple 
        else:
            self.win_tuple = (-1,-1)
            self.window_size = None
        self.use_cu_seqlen = config.use_cu_seqlen
        self.use_da = config.use_da
        self.use_sigmoid = config.use_sigmoid #and not self.local
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            from lit_gpt.rmsnorm import FusedRMSNorm
            self.q_norm = FusedRMSNorm(config.head_size, eps=config.norm_eps) #partial(F.normalize, dim=-1,eps=config.norm_eps) 
            self.k_norm = FusedRMSNorm(config.head_size, eps=config.norm_eps) #partial(F.normalize, dim=-1,eps=config.norm_eps)
        if self.use_sigmoid:
            from flash_sigmoid import flash_attn_func as flash_sigmoid_func
            self.attn_func = flash_sigmoid_func
            self.sub_norm = config.norm_class(config.head_size, eps=config.norm_eps)  
            # norm over all heads is slightly better than group norm but we still use group norm
        elif self.use_da:       
            depth = 10000 if config.da_const_lamb else layer_idx
            self.da = FlashDiffAttention(self.head_size, depth, causal=True, softmax_scale= self.scale, window_size = self.win_tuple)
        elif config.sink_attn:
            # deprecated, use flash_attn instead
            from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention
            self.attn_func = DotProductAttention(
                self.n_head,
                self.head_size,
                num_gqa_groups=self.n_query_groups,
                attention_dropout=0.0,
                tp_size=1,
                get_rng_state_tracker=None, # need to set if dp is enabled
                sequence_parallel=False,
                tp_group=None,
                layer_number=layer_idx,
                attention_type="self",
                softmax_type="learnable",
                softmax_scale = self.scale,
                attn_mask_type = "causal",
            )
            nn.init.zeros_(self.attn_func.softmax_offset)
        else:
            self.use_fa4 = False
            compute_capability = torch.cuda.get_device_capability()
            sm_version = compute_capability[0] * 10 + compute_capability[1]
            if sm_version == 100:
                import flash_attn.cute as flash_attn_interface
                print("Using flash_attn 4 for gb200!")
                self.use_fa4 = True
            elif sm_version == 90:
                try:
                    import flash_attn_interface
                    print("Using flash_attn 3.0.0 for hopper!")
                except:
                    print("Using flash_attn 2.8.1 for hopper!")
                    from flash_attn import flash_attn_interface
            else:
                from flash_attn import flash_attn_interface
            self.attn_func = partial(flash_attn_interface.flash_attn_varlen_func, causal=True, deterministic=True, softmax_scale= self.scale, window_size = self.win_tuple)
        if self.config.ada_rope:
            self.rope_cache = None
        if self.config.attn_norm:
            self.attn_norm = config.norm_class(self.head_size * self.n_head, eps=config.norm_eps)
        # output projection
        self.proj = nn.Linear(self.head_size * self.n_head, n_embd, bias=config.attn_out_bias)
        self.sc = config.sc_attn
        self.max_len = self.config.block_size
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels= self.kv_dim,
                out_channels= self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups= self.kv_dim,
                padding=d_conv - 1,
            ) 

    def reset_parameters(self) -> None:
        """Initialize attention parameters using PyTorch default initialization."""
        # Initialize attn linear
        if not self.config.separate_qkv:
            utils.torch_default_init(self.attn.weight)
            if self.attn.bias is not None:
                nn.init.zeros_(self.attn.bias)
        else:
            utils.torch_default_init(self.q_proj.weight)
            if self.q_proj.bias is not None:
                nn.init.zeros_(self.q_proj.bias)
            if not self.yoco_cross:
                utils.torch_default_init(self.k_proj.weight)
                if self.k_proj.bias is not None:
                    nn.init.zeros_(self.k_proj.bias)
                utils.torch_default_init(self.v_proj.weight)
                if self.v_proj.bias is not None:
                    nn.init.zeros_(self.v_proj.bias)
            if self.config.gated_attn:
                utils.torch_default_init(self.g_proj.weight)
                if self.g_proj.bias is not None:
                    nn.init.zeros_(self.g_proj.bias)
        
        # Initialize proj linear
        utils.torch_default_init(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        
        # Initialize conv1d for short conv attention
        if self.sc:
            for conv in [self.q_conv1d, self.k_conv1d, self.v_conv1d]:
                # PyTorch default for Conv1d is kaiming_uniform_ with a=sqrt(5)
                utils.torch_default_init(conv.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        
        # Initialize qk_norm if present
        if hasattr(self, 'q_norm') and hasattr(self.q_norm, 'reset_parameters'):
            self.q_norm.reset_parameters()
        if hasattr(self, 'k_norm') and hasattr(self.k_norm, 'reset_parameters'):
            self.k_norm.reset_parameters()
        if hasattr(self, 'sub_norm') and hasattr(self.sub_norm, 'reset_parameters'):
            self.sub_norm.reset_parameters()
        
        # Initialize differential attention if present
        if hasattr(self, 'da') and hasattr(self.da, 'reset_parameters'):
            self.da.reset_parameters()

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        if self.config.yoco_window and \
            self.config.local_window == self.config.block_size//2: # this is a long SWA layer, may be long context extended
            initial_window = 4096 # hardcoded for now
            scaling_factor = self.config.scaling_factor
        else:
            initial_window = self.config.local_window
            scaling_factor = 1.0
        return build_rope_cache(
                seq_len= seq_len,
                n_elem=int(self.config.rotary_percentage * self.config.head_size),
                dtype=torch.bfloat16,
                device=idx.device,
                base = 0.09 * initial_window ** 1.628 ,
                scaling_factor= scaling_factor, 
                )

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        gmu_mems = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        if self.config.ada_rope:
            assert self.config.local_window > 0, "local_window must be set for ada_rope"
            if self.rope_cache is None:
                self.rope_cache = self.build_rope_cache(x, self.max_len)
            if T > self.max_len:
                self.max_len = T
                self.rope_cache = self.build_rope_cache(x, self.max_len)
            cos, sin = self.rope_cache   
            cos = cos[:T]
            sin = sin[:T]
            rope = (cos, sin)
        
        if self.yoco_cross:
            if self.config.separate_qkv:
                q = self.q_proj(x)
                if self.config.gated_attn:
                    g = self.g_proj(x)   
                    g = g.reshape(B,  T, self.n_head, 1)  
            else:
                q = self.attn(x)
                if self.config.gated_attn:
                    q, g = q.split((self.attn_shape - self.n_head, self.n_head), dim=-1)
                    g = g.reshape(B,  T, self.n_head, 1)  
            q = q.reshape(B,  T, -1, self.head_size) 
            if not self.config.nope and not self.config.yoco_nope:         
                cos, sin = rope
                # apply rope in fp32 significanly stabalize training
                # fused rope expect (batch_size, seqlen, nheads, headdim)
                q = apply_rotary_emb_func(q, cos, sin, False, True)       
             
            k, v = kv_cache
            y = self.scaled_dot_product_attention(q, k, v, attention_mask=mask)
            if self.config.gated_attn:
                y = glu(g, y)
            y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side

            # output projection
            y = self.proj(y)
            return y, kv_cache, gmu_mems
        
        if self.config.separate_qkv:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            if self.config.gated_attn:
                g = self.g_proj(x)   
                g = g.reshape(B,  T, self.n_head, 1)  
        else:
            qkv = self.attn(x)
            if self.config.gated_attn:
                qkv, g = qkv.split((self.attn_shape - self.n_head, self.n_head), dim=-1)
                g = g.reshape(B,  T, self.n_head, 1)  
            # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
            q_per_kv = self.n_head // self.n_query_groups
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size) # (B, T, n_query_groups, total_qkv, hs)
            # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        if self.sc:
            q = q.reshape(B,  T, -1 )  # (B, T, nh_q, hs)
            k = k.reshape(B,  T, -1 )  
            v = v.reshape(B,  T, -1 )  
            q = causal_conv1d_fn(
                        x = q.transpose(-1,-2),
                        weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                        bias=self.q_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            k = causal_conv1d_fn(
                        x = k.transpose(-1,-2),
                        weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                        bias=self.k_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            v = causal_conv1d_fn(
                        x = v.transpose(-1,-2),
                        weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                        bias=self.v_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2) 

        q = q.reshape(B,  T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.head_size)  
        v = v.reshape(B,  T, -1, self.head_size)
        if self.qk_norm:
            q, k = self.q_norm(q).to(q), self.k_norm(k).to(q)
        if not self.config.nope and not (self.config.yoco_nope and self.win_tuple == (-1,-1)):         
            cos, sin = rope
            # apply rope in fp32 significanly stabalize training
            # fused rope expect (batch_size, seqlen, nheads, headdim)
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)


        kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, attention_mask=mask)
        if self.config.gated_attn:
            y = glu(g, y)
        y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side
        if self.config.gmu_attn:
            gmu_mems = y
            # y = y * torch.sigmoid(g)
        # output projection
        if self.config.attn_norm:
            y = self.attn_norm(y)
        y = self.proj(y)
        return y, kv_cache, gmu_mems

    @torch.compiler.disable 
    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        
        if self.config.fa2:
            if self.use_sigmoid:
                out = self.attn_func(q, k, v, dropout_p=0.0, softmax_scale=self.scale, causal=True, window_size=self.win_tuple, sigmoid_bias = 0)
                out = self.sub_norm(out)
                return out
            
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = k.shape[1]
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = None, None, None, None
            if attention_mask is not None:
                qkv_format = 'thd' # may not be supported for sink attention
                if self.use_cu_seqlen:
                    cu_seqlens_k = cu_seqlens_q = attention_mask.int()
                    max_seqlen_q = max_seqlen_k = (attention_mask - attention_mask.roll(1)).max().cpu().item()
                else:
                    k, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, attention_mask.to(k.device))
                    v, _, _, _, _ = unpad_input(v, attention_mask.to(v.device))

                    if seqlen_q == 1:
                        attention_mask = torch.ones(batch_size, 1, device=q.device)
                    elif seqlen_q != seqlen_k:
                        attention_mask = attention_mask[:, -seqlen_q:]

                    q, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, attention_mask.to(q.device))
            else:
                qkv_format = 'bshd'
    
            if self.config.full_swa_extend and self.win_tuple == (-1,-1) and not self.training:
                wintuple = (self.config.block_size -1, 0)
                if self.use_da:
                    self.da.window_size = wintuple
                elif self.config.sink_attn:
                    self.window_size = self.config.block_size
                else:   
                    self.attn_func.window_size = wintuple
            
            if self.use_da:
                attn_output =self.da(q, k, v, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, 
                                        cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,)
            elif self.config.sink_attn:
                attn_output = self.attn_func(
                    q,
                    k,
                    v,
                    qkv_format=qkv_format,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_k,
                    window_size=self.window_size,
                    
                )
            else:
                if self.use_fa4:
                    attn_output =self.attn_func(q, k, v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, 
                                            cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k, return_lse=True)
                else:
                    attn_output = self.attn_func(q, k, v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, 
                                            cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k)
            if isinstance(attn_output, tuple):
                attn_output, lse = attn_output
                self.z_mean = (lse.float() ** 2).mean()
            else:
                self.z_mean = None
            if self.use_cu_seqlen:
                attn_output = attn_output.reshape(batch_size, seqlen_q, q.shape[-2], q.shape[-1])
            else:
                attn_output = (
                    pad_input(attn_output, indices_q, batch_size, max_seqlen_q)
                    if attention_mask is not None
                    else attn_output
                )
            return attn_output

        # legacy sqrt attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)

        # y = torch.nn.functional.scaled_dot_product_attention(
        #     q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        # )
        q = q * self.scale
        mask = torch.triu(torch.ones(q.shape[-2],q.shape[-2]).to(q).float(),1)*(-10000)
        v = F.silu(v)
        y =  torch.sqrt(F.softmax(q @ k.transpose(-2, -1) + mask, dim=-1,dtype=torch.float32)+1e-5).type_as(q) @ v
        return y.transpose(1, 2)

def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, 
    base: int = 10000,
    ntk_alpha: float = 1.0,
    ntk_beta: float = 32.0,
    scaling_factor: float = 1.0,
    initial_context_length: int = 4096,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    See YaRN paper: https://arxiv.org/abs/2309.00071
    """
   
    freq = (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    if scaling_factor > 1.0:
        concentration = (
            0.1 * math.log(scaling_factor) + 1.0
        )  # YaRN concentration

        d_half = n_elem / 2
        # NTK by parts
        low = (
            d_half
            * math.log(initial_context_length / (ntk_beta * 2 * math.pi))
            / math.log(base)
        )
        high = (
            d_half
            * math.log(initial_context_length / (ntk_alpha * 2 * math.pi))
            / math.log(base)
        )
        assert 0 < low < high < d_half - 1

        interpolation = 1.0 / (scaling_factor * freq)
        extrapolation = 1.0 / freq

        ramp = (
            torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
        ) / (high - low)
        mask = 1 - ramp.clamp(0, 1)

        inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
        concentration = 1.0
        inv_freq = 1.0 / freq
            
    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, inv_freq)

    cos, sin = torch.cos(idx_theta) * concentration, torch.sin(idx_theta) * concentration

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin