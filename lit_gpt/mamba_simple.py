# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

import utils



try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
    
from .triton_sequential_scan import triton_selective_scan_sequential
from lit_gpt.rmsnorm import FusedRMSNorm
from .gated_memory_unit import GMU, swiglu


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        jamba_norm = False,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        gmu_save=False,
        device=None,
        dtype=None,
        config=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.jamba_norm = jamba_norm
        if jamba_norm:
            self.dt_layernorm = FusedRMSNorm(self.dt_rank, eps=1e-5)
            self.b_layernorm = FusedRMSNorm(self.d_state, eps=1e-5)
            self.c_layernorm = FusedRMSNorm(self.d_state, eps=1e-5)
            
        self.gmu_save = gmu_save
        self.use_cu_seqlen = False
        self.use_triton = False  # TODO: Hard code True for GB200
        if self.use_triton:
            self.use_fast_path = False
        else:
            if config is not None and config.use_cu_seqlen:
                self.use_cu_seqlen = True
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as ssf_custom, mamba_inner_fn as mb_custom
                self.selective_scan_fn =  ssf_custom
                self.mamba_inner_fn =  mb_custom
            else:
                from .selective_scan_interface import selective_scan_fn, mamba_inner_fn
                self.selective_scan_fn =  selective_scan_fn
                self.mamba_inner_fn =  mamba_inner_fn

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_init = dt_init
        self.dt_scale = dt_scale

        self.A_log = nn.Parameter(torch.ones(self.d_inner, self.d_state, dtype=torch.float32, device=device))

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize Mamba parameters."""
        # Initialize linear layers with PyTorch default
        utils.torch_default_init(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        
        # Initialize conv1d
        utils.torch_default_init(self.conv1d.weight)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)
        
        # Initialize x_proj (no bias)
        utils.torch_default_init(self.x_proj.weight)
        
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, device=self.dt_proj.bias.device, dtype=torch.float32) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.copy_(inv_dt.to(dtype=self.dt_proj.bias.dtype))
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        
        # S4D real initialization for A_log
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.A_log.device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log.copy_(torch.log(A))
        
        # D "skip" parameter
        nn.init.ones_(self.D)
        
        # out_proj
        utils.torch_default_init(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
        # Jamba norms if present
        if self.jamba_norm:
            if hasattr(self.dt_layernorm, 'reset_parameters'):
                self.dt_layernorm.reset_parameters()
            if hasattr(self.b_layernorm, 'reset_parameters'):
                self.b_layernorm.reset_parameters()
            if hasattr(self.c_layernorm, 'reset_parameters'):
                self.c_layernorm.reset_parameters()

    def forward(self, hidden_states, seq_idx=None, inference_params=None, mask= None, support_init_states = False, gmu_mems = None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if hidden_states.shape[1] == 1: #inference_params.get_seq_length(self.layer_idx) > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (not self.jamba_norm) and (not self.gmu_save) and self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.use_cu_seqlen:
                out = self.mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    cu_seqlens=mask,
                    seq_idx=seq_idx,
                    delta_softplus=True,
                )
            else:               
                out = self.mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    mask=mask,
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            if self.gmu_save:
                z = z.transpose(-1,-2).contiguous()

            if mask is not None and not self.use_cu_seqlen and (not (self.use_triton and self.training)):
                x = x * mask.unsqueeze(1)
            # Compute short convolution
            ox = x
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                if conv_state is not None:
                    x = torch.cat([conv_state,x],dim = -1)
                #print(rearrange(self.conv1d.weight, "d 1 w -> d w").shape)
                if self.use_cu_seqlen:
                    x = causal_conv1d_fn(
                        x=x.transpose(1,2).contiguous().transpose(1,2),
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        seq_idx=seq_idx,
                        activation=self.activation,
                    )
                else:
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )
                if conv_state is not None:
                    x = x[..., -seqlen:].contiguous()
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(ox, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if mask is not None and not self.use_cu_seqlen and (not (self.use_triton and self.training)):
                x = x * mask.unsqueeze(1)
            # print(mask[0,:])
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            if self.jamba_norm:
                dt = self.dt_layernorm(dt)
                B = self.b_layernorm(B)
                C = self.c_layernorm(C)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            if self.use_triton or (support_init_states and ssm_state is not None):
                dt = F.softplus(dt + self.dt_proj.bias.float()[:, None])
                x = x.transpose(-1, -2).contiguous()
                dt = dt.transpose(-1, -2).contiguous()
                B = B.transpose(-1,-2).contiguous()
                C = C.transpose(-1,-2).contiguous()
                if not self.gmu_save:
                    z = z.transpose(-1,-2).contiguous()
                o = triton_selective_scan_sequential(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    initial_state= ssm_state,
                )       
                o, last_state = o     
                if ssm_state is not None:
                    ssm_state.copy_(last_state)
                if self.gmu_save:
                    gmu_mems = o
                y = swiglu(z, o)
            else:
                if self.use_cu_seqlen:
                    y = self.selective_scan_fn(
                        x,
                        dt,
                        A,
                        B,
                        C,
                        self.D.float(),
                        z= None if self.gmu_save else z,
                        delta_bias=self.dt_proj.bias.float(),
                        delta_softplus=True,
                        return_last_state=ssm_state is not None,
                        cu_seqlens=mask,
                    )
                else:
                    y = self.selective_scan_fn(
                        x,
                        dt,
                        A,
                        B,
                        C,
                        self.D.float(),
                        z= None if self.gmu_save else z,
                        delta_bias=self.dt_proj.bias.float(),
                        delta_softplus=True,
                        return_last_state=ssm_state is not None,
                    )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")
                if self.gmu_save:
                    gmu_mems = y
                    y = swiglu(z, y)
            out = self.out_proj(y)
        return out, gmu_mems

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
