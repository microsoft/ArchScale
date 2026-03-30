# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from .gated_memory_unit import swiglu
import utils
# use mamba_ssm's RMSNormGated so torch.compile can work
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated 

chunk_gated_delta_rule = torch.compiler.disable(chunk_gated_delta_rule)
fused_recurrent_gated_delta_rule = torch.compiler.disable(fused_recurrent_gated_delta_rule)
pad_input = torch.compiler.disable(pad_input)
get_unpad_data = torch.compiler.disable(get_unpad_data)
index_first_axis = torch.compiler.disable(index_first_axis)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.

    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        gmu_save: bool = False,
        **kwargs
    ) -> GatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx
        
        self.gmu_save = gmu_save
        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        # hard coded for now
        self.dt_min = 0.001
        self.dt_max = 0.1
        self.dt_init_floor = 1e-4
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # GDN-A: norm after gate
        self.norm = RMSNormGated(self.value_dim, eps = norm_eps, norm_before_gate = False) 
        
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize GatedDeltaNet parameters."""
        
        # Initialize linear projections
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.a_proj, self.b_proj, self.o_proj]:
            utils.torch_default_init(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        # Initialize gate projection if present
        if self.use_gate:
            utils.torch_default_init(self.g_proj.weight)
            if self.g_proj.bias is not None:
                nn.init.zeros_(self.g_proj.bias)
        
        # Initialize short convolutions if present
        if self.use_short_conv:
            for conv in [self.q_conv1d, self.k_conv1d, self.v_conv1d]:
                if hasattr(conv, 'reset_parameters'):
                    conv.reset_parameters()
        
        # Initialize A_log: uniform(0, 16) then log
        A = torch.empty(self.num_v_heads, dtype=torch.float32, device=self.A_log.device).uniform_(0, 16)
        self.A_log.copy_(torch.log(A))
        
        # Initialize dt_bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.num_v_heads, device=self.dt_bias.device, dtype=torch.float32) 
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=self.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt.to(dtype=self.dt_bias.dtype))
        
        # Initialize norm
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        
        gmu_mems = kwargs.get('gmu_mems', None)
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        
        q,k,v= self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        if self.use_short_conv:

            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=q,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=k,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=v,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        g= None
        o = rearrange(o, 'b t h d -> b t (h d)')  
        if self.use_gate:
            g = self.g_proj(hidden_states)
            if self.gmu_save:
                gmu_mems = o    
                if attention_mask is not None:
                    gmu_mems = pad_input(gmu_mems.squeeze(0), indices, batch_size, q_len)
        o= self.norm(o, g)
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, gmu_mems #, past_key_values