# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from functools import partial

import torch
import torch.nn as nn

from lit_gpt.config import Config

from einops import rearrange
import torch.nn.functional as F

import utils

try:
    from causal_conv1d import causal_conv1d_fn
except:
    causal_conv1d_fn = None
    
def eval_index(x, dim):
    strx = "x["
    if dim<0:
        xlen = len(x.shape) +dim
    else:
        xlen = dim
    for _ in range(xlen):
        strx +=":,"
    strx += "1:]"    
    return eval(strx)

@torch.compiler.disable    
def compress_seq(q,index_q, max_sl, dim = -2):
    shape = q.shape[:dim] + (max_sl+1,) + q.shape[dim+1:]
    new_q = torch.zeros(shape, device = q.device, dtype = torch.float32)
    new_q.scatter_(dim,index_q, q.float())
    if max_sl>0:
        new_q = eval_index(new_q,dim)
    else:
        new_q = 0*new_q
    return new_q.type_as(q)

@torch.compiler.disable    
def extract(h, index_q):
    #h; B, T, C
    h = F.pad(h, (0,0,1,0))
    h = torch.gather(h.float(),-2,index_q.expand(-1,-1,h.shape[-1])).type_as(h)
    return h


class LayerConfigurator(nn.Module):
    def __init__(self, config: Config,) -> None:
        super().__init__()
        self.norm = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.lc_proj = nn.Linear(config.n_embd, 2, bias=config.bias)
        self.act_bias = 2
        self.temp = 1 #math.sqrt(config.n_embd)
        self.index_q = None
        self.gate = None
        #self.act = lambda x: torch.relu(torch.tanh( (x + self.act_bias)/self.temp))
        self.act = lambda x: F.softmax(x / self.temp, dim=-1)[:,:,1:]
        #self.act = lambda x: torch.relu(x + self.act_bias)**2
        self.compress = True #config.compress
        self.sc = True #False #True #config.sc_lc
        ## TODO inter-layer sc
        if self.sc:
            self.d_inner = config.n_embd
            d_conv = 4
            conv_bias = False
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )
    
    def reset_parameters(self) -> None:
        """Initialize LayerConfigurator parameters."""
        # Initialize norm if it has reset_parameters
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()
        
        # Initialize lc_proj
        utils.torch_default_init(self.lc_proj.weight)
        if self.lc_proj.bias is not None:
            nn.init.zeros_(self.lc_proj.bias)
        
        # Initialize conv1d if present
        if self.sc:
            utils.torch_default_init(self.conv1d.weight)
            if self.conv1d.bias is not None:
                nn.init.zeros_(self.conv1d.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, C
        ox = x
        x = self.norm(x)
        if self.sc:
            x = causal_conv1d_fn(
                x = x.transpose(-1,-2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            ).transpose(-1,-2)
        x = self.act(self.lc_proj(x)) 
        mask_q = (x>0.5)
        if self.compress:
            max_q = mask_q.sum(1).max()
            index_q = mask_q * torch.cumsum(mask_q.long(),dim=1)
            ox = compress_seq(ox, index_q.expand(-1,-1,ox.shape[-1]), max_q ,dim = 1) # B x T_c x D 
            self.index_q = index_q
        self.gate = x
        self.act_mask = mask_q
        return ox
    
    def extract(self, h):
        if self.compress:
            h = extract(h, self.index_q) 
        return h