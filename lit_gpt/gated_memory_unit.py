# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.layernorm_gated import RMSNorm as RMSNormGated

import utils

swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""
swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)
 
 
class SwiGLUFunction(torch.autograd.Function):
 
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)
 
    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_bwd(x, y, dout)
 
swiglu = SwiGLUFunction.apply


glu_fwd_codestring = """
template <typename T> T glu_fwd(T x, T y) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return x_sigmoid * float(y);
}
"""
glu_bwd_codestring = """
template <typename T> T glu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1.0f - x_sigmoid) * float(y) * float(g);
    dy = x_sigmoid * float(g);
}
"""
glu_fwd = torch.cuda.jiterator._create_jit_fn(glu_fwd_codestring)
glu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(glu_bwd_codestring, num_outputs=2)


class GLUFunction(torch.autograd.Function):
    # return sigmoid(x) * y
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return glu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return glu_bwd(x, y, dout)

glu = GLUFunction.apply

class GMU(nn.Module):
    def __init__(
        self,
        d_model,
        d_mem,
        bias = False,
        use_norm = False,
        ngroups = 1,
        device = None,
        dtype = None,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_mem, bias = bias, device = device, dtype = dtype)
        self.out_proj = nn.Linear(d_mem, d_model, bias = bias, device = device, dtype = dtype)
        if use_norm:
            self.norm = RMSNormGated(d_mem, eps = 1e-5, norm_before_gate = False, group_size = d_mem // ngroups)
        self.use_norm = use_norm
    
    def reset_parameters(self) -> None:
        """Initialize GMU parameters using PyTorch default initialization."""
        utils.torch_default_init(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        
        utils.torch_default_init(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
        if self.use_norm and hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()
        
    def forward(self, hidden_states, memory):
        out = self.in_proj(hidden_states)
        if self.use_norm:
            out = self.norm(memory, out)
        else:
            out = swiglu(out, memory)
        out = self.out_proj(out)
        return out
    
class GMUWrapper(nn.Module):
    def __init__(self, d_model, d_mem, bias = False, use_norm = False, ngroups = 1, device = None, dtype = None):
        super().__init__()
        self.gmu = GMU(d_model, d_mem, bias, use_norm, ngroups, device, dtype)
    
    def reset_parameters(self) -> None:
        """Initialize GMUWrapper parameters."""
        self.gmu.reset_parameters()
        
    def forward(self, hidden_states, memory):
        return self.gmu(hidden_states, memory), memory
        
