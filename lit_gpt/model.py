# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
from typing import Any, List, Optional, Tuple
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from typing_extensions import Self
from lit_gpt.config import Config

import utils
# from xformers.ops import SwiGLU
from .attention import CausalSelfAttention, RoPECache, KVCache, build_rope_cache
from torch import Tensor
from .mamba_simple import Mamba
from .gla import GatedLinearAttention
from .moe import MoE, get_moe

# Optional SonicMoE Experts import for initialization handling
try:
    from .moe import Experts as SonicMoEExperts
except:
    SonicMoEExperts = None
from .delta_net import DeltaNet
from .multiscale_retention import MultiScaleRetention
import transformer_engine.pytorch as te
import torch.nn.functional as F
try:
    from .gated_deltanet import GatedDeltaNet
except:
    GatedDeltaNet = None
    
try:
    from .mega import S6GatedAttention
    from .relu2_attn_glu import Relu2Attention, Relu2MLP
    from .mamba2 import Mamba2
except:
    S6GatedAttention, Relu2Attention, Relu2MLP, Mamba2 = None, None, None, None

from lit_gpt.config import RMSNormFunc

from .gated_memory_unit import swiglu, GMU, GMUWrapper
from .layer_configurator import LayerConfigurator
import copy
from collections import namedtuple

CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "weight"], defaults=[None, None])        

def truncated_normal_(tensor, mean=0.0, std=0.02):
   
    tensor=torch.nn.init.trunc_normal_(tensor, mean, std, -2*std, 2*std)
   
    return tensor


def get_rnn(config: Config, layer_idx: int, gmu_save: bool = False, **factory_kwargs):
    # Create the appropriate RNN module based on rnn_type
    if config.rnn_type == "mamba":
        return Mamba(config.n_embd, layer_idx=layer_idx, gmu_save=gmu_save, config=config, **factory_kwargs)
    elif config.rnn_type == "mamba2":
        mamba2_expand = 8 * math.ceil(config.n_embd * 2 / 64 / 8) * 64 / config.n_embd 
        return Mamba2(config.n_embd, expand=mamba2_expand, layer_idx=layer_idx, gmu_save=gmu_save, config=config, **factory_kwargs)
    elif config.rnn_type == "gdn":
        return GatedDeltaNet(hidden_size=config.n_embd, num_heads=math.ceil(int(config.n_embd*0.75)/256), head_dim=256, mode='chunk', gmu_save=gmu_save, use_short_conv=True, allow_neg_eigval=True)
    #TODO: add support of gmu for rnns below
    elif config.rnn_type == "retnet": 
        return MultiScaleRetention(hidden_size=config.n_embd, num_heads=config.n_head // 2, expand_k=1, expand_v=2, mode='fused_chunk', use_short_conv=False)
    elif config.rnn_type == "gla":
        return GatedLinearAttention(hidden_size=config.n_embd, num_heads=config.n_embd // 384, expand_k=0.5, expand_v=1, mode='fused_chunk', use_short_conv=False)
    elif config.rnn_type == "delta":
        return DeltaNet(hidden_size=config.n_embd, num_heads=9, expand_k=1.5, expand_v=1.5, mode='chunk', use_short_conv=True)
    else:
        raise ValueError(f"Unknown RNN type: {config.rnn_type}. Supported types: mamba, mamba2, retnet, gla, delta, gdn")


def compute_rmsnorm(x: torch.Tensor) -> torch.Tensor:
    """Compute RMSNorm (root mean square) of a tensor."""
    return torch.sqrt(torch.mean(x.float() ** 2) + 1e-30)


def compute_outlier_percentage(x: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """Compute percentage of outlier elements using the N-sigma rule, per token.
    
    For a tensor of shape (B, T, D), computes mean and std across the D dimension
    for each token, then counts elements where |x - mean| > sigma * std.
    Returns the overall outlier percentage as a scalar.
    """
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    outliers = (x - mean).abs() > sigma * std
    return outliers.float().mean() * 100.0


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.mup = config.mup
        self.shadow_init = config.shadow_init
        self.super_mup = config.super_mup
        if config.mup and not config.use_muonh:
            self.logit_scale = config.mup_d0 / config.n_layer 
        else:
            self.logit_scale = 1.0
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        if config.lc and config.sparsity > 1:
            new_config = copy.deepcopy(config)
            new_config.n_layer = config.n_layer * config.sparsity
            config = new_config
        if config.share_per_layer>0:
            num_layer = config.share_per_layer * config.share_group
        else:
            num_layer = config.n_layer 
        if config.moe:
            self.n_attn = [ 0 for i in range(num_layer)] 
            self.n_mlp = [ config.sparsity * config.top_k for i in range(num_layer)] 
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(num_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.layer_configurator = None
        if config.lc and config.lc_shared:
            self.layer_configurator = LayerConfigurator(config)
        # Unified LC gate history at model level (store raw gates with grad)
        self.lc_gate_hist = []
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.max_len = self.config.block_size
        self.scale_embed = config.scale_embed
        self.tied_embed = config.tied_embed
        if self.tied_embed:
            self.tie_weights()
                
    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight
    
    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize all parameters. Called after model is moved from meta device."""
        # Initialize embedding (RWKV-style init)
        std = 1e-4  # RWKV init
        if self.tied_embed:
            std = 0.02
        if self.scale_embed:
            std = std / math.sqrt(self.config.n_embd)
        nn.init.normal_(self.transformer.wte.weight, std=std)
        
        # Initialize lm_head
        if not self.tied_embed:
            utils.torch_default_init(self.lm_head.weight)
            if self.lm_head.bias is not None:
                nn.init.zeros_(self.lm_head.bias)
        
        # Initialize final layer norm
        if hasattr(self.transformer.ln_f, 'reset_parameters'):
            self.transformer.ln_f.reset_parameters()
        
        # Initialize all blocks
        for block in self.transformer.h:
            block.reset_parameters()
        
        # Initialize layer_configurator if present
        if self.layer_configurator is not None:
            self.layer_configurator.reset_parameters()
        
        # Apply custom initialization logic
        n_layer = self.config.n_layer
        # mup zero_readout trick for lm_head
        if self.mup and not self.tied_embed:
            if self.config.head_init_std is not None:
                nn.init.normal_(self.lm_head.weight, std=self.config.head_init_std)
            else:
                nn.init.zeros_(self.lm_head.weight)
        else:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    # Apply per-layer output projection scaling (GPT-2 style)
                    if (name.endswith("out_proj.weight") or 
                        name.endswith("o_proj.weight") or 
                        name.endswith("proj.weight") or
                        name.endswith("w3.weight")):
                        if not self.config.mlp:
                            n_residuals_per_layer = 1  
                        else:
                            n_residuals_per_layer = 2   
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
        
        # Apply w_init_scale multiplier for Linear weights
        # Include SonicMoE Experts (3D weight tensors) for consistent initialization
        weight_module_types = (nn.Linear, te.GroupedLinear)
        if SonicMoEExperts is not None:
            weight_module_types = weight_module_types + (SonicMoEExperts,)
        for name, p in self.named_parameters():
            if p.requires_grad and "weight" in name:
                parent_name = name.rsplit('.', 1)[0]
                parent_module = self.get_submodule(parent_name)
                if isinstance(parent_module, weight_module_types):
                    if not self.mup:
                        nn.init.normal_(p, std=0.02)
                    p.data *= self.config.w_init_scale
        
        # Re-tie weights if using tied embeddings
        if self.tied_embed:
            self.tie_weights()
    
    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        residual_dropout: float = 0.0,
        use_flce_loss: bool = False,
    ) -> torch.Tensor:

        B, T = idx.size()
        if self.config.use_cu_seqlen:
            assert idx.size(0) == 1, "only support batch size 1 for variable length training"
            attn_mask = (idx.flatten()==self.config.eos_token_id)
            
        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size

        if self.config.nope or self.config.ada_rope:
            rope = None
        else:
            if self.rope_cache is None:
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            if T > self.max_len:
                self.max_len = T
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            cos, sin = self.rope_cache   
            cos = cos[:T]
            sin = sin[:T]
            rope = (cos, sin)
        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.scale_embed:
            x = x * math.sqrt(self.config.n_embd)
        input_embed = x
        kv_cache = None
        gmu_mems = None
        # reset unified histories
        self.lc_gate_hist = []
        self.tokens_per_expert = []
        self.token_indices = []
        self.aux_loss = 0.0 # load balance loss
        # reset output RMSNorm, outlier, and LSE tracking
        self.attn_output_rmsnorms = []
        self.mlp_output_rmsnorms = []
        self.attn_outlier_pcts = []
        self.mlp_outlier_pcts = []
        self.attn_z_means = []
        self.router_z_means = []
        
        if self.config.share_per_layer>0:
            for i in range(self.config.share_group):
                for _ in range(self.config.n_layer // self.config.share_group // self.config.share_per_layer):
                    for block in self.transformer.h[i*self.config.share_per_layer:(i+1)*self.config.share_per_layer]:
                        x, kv_cache, gmu_mems, input_embed = block(
                            x,
                            input_embed,
                            rope,
                            max_seq_length,
                            attn_mask,
                            lc=self.layer_configurator,
                            kv_cache=kv_cache,
                            gmu_mems=gmu_mems,
                            residual_dropout=residual_dropout,
                            lc_gate_hist=self.lc_gate_hist,
                        )
                        self.attn_output_rmsnorms.append(block.attn_output_rmsnorm)
                        self.mlp_output_rmsnorms.append(block.mlp_output_rmsnorm)
                        self.attn_outlier_pcts.append(block.attn_outlier_pct)
                        self.mlp_outlier_pcts.append(block.mlp_outlier_pct)
                        self.attn_z_means.append(block.attn_z_mean)
                        self.router_z_means.append(block.router_z_mean)
                        if self.config.moe:
                            self.aux_loss = self.aux_loss + block.mlp.aux_loss
                            self.token_indices.append(block.mlp.token_indices)
                            self.tokens_per_expert.append(block.mlp.tokens_per_expert) # E # for logging
        else:
            for block in self.transformer.h:
                x, kv_cache, gmu_mems, input_embed = block(
                    x,
                    input_embed,
                    rope,
                    max_seq_length,
                    attn_mask,
                    lc=self.layer_configurator,
                    kv_cache=kv_cache,
                    gmu_mems=gmu_mems,
                    residual_dropout=residual_dropout,
                    lc_gate_hist=self.lc_gate_hist,
                )
                self.attn_output_rmsnorms.append(block.attn_output_rmsnorm)
                self.mlp_output_rmsnorms.append(block.mlp_output_rmsnorm)
                self.attn_outlier_pcts.append(block.attn_outlier_pct)
                self.mlp_outlier_pcts.append(block.mlp_outlier_pct)
                self.attn_z_means.append(block.attn_z_mean)
                self.router_z_means.append(block.router_z_mean)
                if self.config.moe:
                    self.aux_loss = self.aux_loss + block.mlp.aux_loss
                    self.token_indices.append(block.mlp.token_indices)
                    self.tokens_per_expert.append(block.mlp.tokens_per_expert) # E # for logging

        if self.config.sum_skip: 
            x = input_embed

        x = self.transformer.ln_f(x.to(dtype=self.lm_head.weight.dtype))
        x = x * self.logit_scale
        if use_flce_loss:
            return CausalLMOutput(logits=x, weight=self.lm_head.weight) # (b, t, vocab_size)
        else:
            lm_logits = self.lm_head(x).float()
            return CausalLMOutput(logits=lm_logits) # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            base = self.config.rope_base,
            scaling_factor=self.config.scaling_factor,
        )

    
class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        factory_kwargs = {"jamba_norm": config.jamba_norm, "device": "cuda", "dtype": torch.float32}

        decouple_scaler = config.decouple_postnorm
        self.decouple_scaler = decouple_scaler
        if decouple_scaler:
            self.norm_1 = RMSNormFunc(config.n_embd, eps=config.norm_eps, 
                         decouple_gain=True, skip_gain=config.skip_gain)
        else:
            self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.use_skip_proj = config.skip_weight_per_layer > 0 and \
            layer_idx % config.skip_weight_per_layer == config.skip_weight_per_layer - 1
        if self.use_skip_proj:
            self.skip_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Initialize flags
        self.use_rnn = False # use rnn for this layer
        self.rnn_type = config.rnn_type  # Store the actual RNN type being used
        self.use_gmu = False # use gmu for this layer
        self.yoco_kv = False # save kv for yoco
        self.gmu_save = False # save memory for gmu
        self.yoco_cross = False # use cross attention for yoco
        self.last_layer = layer_idx == config.n_layer - 1

        if config.yoco:
            config = copy.deepcopy(config)
            assert config.n_layer % 4 == 0, 'n_layer should be divisible by 4 for samba + yoco'
            if layer_idx < config.n_layer//2:
                self.use_rnn = config.rnn_per_layer > 0 and layer_idx % config.rnn_per_layer == 0
                self.use_full = False
            else:
                if config.gmu_yoco and not config.gmu_attn:
                    self.gmu_save = (layer_idx >= (config.n_layer//2))
                else:
                    self.gmu_save = False
                self.yoco_kv = (layer_idx >= (config.n_layer//2 +1))
                self.yoco_cross = (layer_idx >= (config.n_layer//2 +2))
                self.use_full = (layer_idx >= (config.n_layer//2 +1))
                if layer_idx == (config.n_layer//2):
                    self.use_rnn = config.rnn_per_layer > 0 
                if config.gmu_yoco and layer_idx >= (config.n_layer//2+2):
                    self.use_gmu = layer_idx % config.gmu_per_layer == 0
 
            if self.use_full:
                if config.yoco_window:
                    config.local_window = config.block_size // 2
                else:
                    config.local_window = -1

        else: 
            if config.attn_layer_pos is not None:
                # For attn_layer_pos, RNN is used when NOT in the attention layer positions
                self.use_rnn = layer_idx not in eval(config.attn_layer_pos)
            else:
                self.use_rnn = config.rnn_per_layer > 0 and layer_idx % config.rnn_per_layer == 0
                
        ### token mixer
        if self.use_gmu:
            if config.gmu_attn:
                gmu_inner = config.head_size * config.n_head
            elif config.gmu_mlp:
                gmu_inner = config.intermediate_size
            elif config.rnn_per_layer > 0 and config.rnn_type == "mamba2":
                mamba2_expand = 8 * math.ceil(config.n_embd * 2 / 64 / 8) * 64 / config.n_embd
                gmu_inner = int(config.n_embd * mamba2_expand)
            elif config.rnn_per_layer > 0 and config.rnn_type == "gdn":
                gmu_inner = math.ceil(int(config.n_embd*0.75)/256)* 256 * 2
            else:
                gmu_inner = config.n_embd * 2
            use_norm = config.rnn_per_layer > 0 and (config.rnn_type == "mamba2" or config.rnn_type == "gdn")
            self.attn = GMUWrapper(config.n_embd, gmu_inner, bias=config.bias, use_norm=use_norm)
        elif self.use_rnn:
            self.attn = get_rnn(config, layer_idx, gmu_save=self.gmu_save, **factory_kwargs)
        else:
            if config.relu2:
                self.attn = Relu2Attention(config, n_embd= config.n_embd, layer_idx= layer_idx, )
            else:
                self.attn = CausalSelfAttention(config, n_embd= config.n_embd, layer_idx= layer_idx, yoco_cross=self.yoco_cross)
            
        # mlp
        if config.mlp:
            if decouple_scaler:
                self.norm_2 = RMSNormFunc(config.n_embd, eps=config.norm_eps, 
                         decouple_gain=True, skip_gain=config.skip_gain)
            else:
                self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
            if config.moe:
                self.mlp = get_moe(config, is_attn=False, is_mlp=True)
            elif config.relu2:
                self.mlp = Relu2MLP(config, )
            else:
                self.mlp = LLaMAMLP(config,)
                
        self.config = config
        if config.lc and not config.lc_shared:
            self.layer_configurator = LayerConfigurator(config)
            if config.mlp:
                self.lc_mlp= LayerConfigurator(config)
    
    def reset_parameters(self) -> None:
        """Initialize all parameters for this block."""
        self.attn.reset_parameters()
        if hasattr(self, 'skip_proj'):
            utils.torch_default_init(self.skip_proj.weight)
            if self.skip_proj.bias is not None:
                nn.init.zeros_(self.skip_proj.bias)
        if self.config.mlp:
            self.mlp.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        input_embed: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        lc = None,
        gmu_mems = None,
        residual_dropout: float = 0.0,
        lc_gate_hist = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        ox = x
        if hasattr(self, "layer_configurator"):
            lc = self.layer_configurator
        if lc is not None:
            x = lc(x)
            if hasattr(self, "layer_configurator"):
                lc_gate_hist.append(lc.gate)
            else:
                lc_gate_hist.append(lc.gate.clone())
        # ox = x
        if self.config.post_norm:
            if hasattr(self.norm_1, "weight"):
                ox = ox.to(dtype=self.norm_1.weight.dtype)
            ox =self.norm_1(ox).bfloat16()
            n_1 = ox
            if self.decouple_scaler:
                n_1 = n_1 * self.norm_1.weight
            if self.config.skip_gain:
                ox = ox * self.norm_1.weight1
            if self.use_skip_proj:
                ox = self.skip_proj(ox)
        else:
            ox = ox.to(torch.float32) # reduce error accumulation across layers during inference
            if hasattr(self.norm_1, "weight"):
                x = x.to(dtype=self.norm_1.weight.dtype)
            n_1 = self.norm_1(x).bfloat16()
        seq_idx = None
        if self.config.use_cu_seqlen:
            if self.use_rnn and self.rnn_type == "mamba2":
                new_seq_pos = F.pad(mask[:-1].to(torch.int32), (1, 0)).unsqueeze(0)
                seq_idx = torch.cumsum(new_seq_pos, dim=-1).to(torch.int32)
            else:
                def get_cu_seqlen(a,):
                    return torch.cat([torch.zeros(1).to(a.device).long(), 
                              (a).nonzero().flatten()+1, 
                              torch.tensor([a.shape[0]]).to(a.device).long() ],dim=-1)
                mask = get_cu_seqlen(mask)
                if self.use_rnn and self.rnn_type == "mamba":
                    seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=mask.device) 
                                for i, s in enumerate(mask[1:]-mask[:-1])], dim=0).unsqueeze(0)
        else:
            if self.use_rnn and self.rnn_type == "mamba2" and mask is not None:
                seq_idx  = mask.to(torch.int32)
 
        if self.use_rnn:
            if self.rnn_type in ["mamba", "mamba2"]:
                h, gmu_mems = self.attn(n_1, seq_idx=seq_idx, mask=mask, gmu_mems=gmu_mems)
                new_kv_cache = kv_cache
            elif self.rnn_type in ["gdn"]:
                cu_seqlens = mask if self.config.use_cu_seqlen else None
                attn_mask = None if self.config.use_cu_seqlen else mask
                h, gmu_mems = self.attn(n_1, attention_mask=attn_mask, cu_seqlens=cu_seqlens, gmu_mems=gmu_mems)
                new_kv_cache = kv_cache
            else: # self.rnn_type in ["retnet", "gla", "delta"]:
                h, _, new_kv_cache = self.attn(n_1)
        elif self.use_gmu:
            h, gmu_mems = self.attn(n_1, gmu_mems)
            new_kv_cache = kv_cache 
        else:
            # attention
            h, new_kv_cache, gmu_mems = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache, gmu_mems)
        x, h = self.post_process_residual(h, ox, lc, residual_dropout)    
        # Log attention/RNN output RMSNorm, outlier percentage, and attention LSE
        with torch.no_grad():
            self.attn_output_rmsnorm = compute_rmsnorm(h)
            self.attn_outlier_pct = compute_outlier_percentage(h)
            self.attn_z_mean = getattr(self.attn, 'z_mean', None)
        # if lc is not None:
        #     x = lc.extract(x)
        #x = (x * lc.gate )* lc.act_mask + bx * (~lc.act_mask)
        if self.config.mlp:
            if hasattr(self, "lc_mlp"):
                lc = self.lc_mlp
            ox = x
            if lc is not None:
                x = lc(x)
                if hasattr(self, "lc_mlp"):
                    lc_gate_hist.append(lc.gate)
                else:
                    lc_gate_hist.append(lc.gate.clone())
            if self.config.post_norm:
                if hasattr(self.norm_2, "weight"):
                    ox = ox.to(dtype=self.norm_2.weight.dtype)  
                ox =self.norm_2(ox).bfloat16()
                n_2 = ox
                if self.decouple_scaler:
                    n_2 = n_2 * self.norm_2.weight
                if self.config.skip_gain:
                    ox = ox * self.norm_2.weight1
            else:
                ox = ox.to(torch.float32) # reduce error accumulation across layers during inference
                if hasattr(self.norm_2, "weight"):
                    x = x.to(dtype=self.norm_2.weight.dtype)
                n_2 = self.norm_2(x).bfloat16()
            if self.config.moe:
                h = self.mlp(n_2)
            else:
                h, gmu_mems = self.mlp(n_2, self.layer_idx,gmu_mems)
            x, h = self.post_process_residual(h, ox, lc, residual_dropout)
            # Log MLP/MoE output RMSNorm, outlier percentage, and router LSE
            with torch.no_grad():
                self.mlp_output_rmsnorm = compute_rmsnorm(h)
                self.mlp_outlier_pct = compute_outlier_percentage(h)
                self.router_z_mean = getattr(self.mlp, 'router_z_mean', None)
            # if lc is not None:
            #     x = lc.extract(x)
            #x = (x * lc.gate)* lc.act_mask + bx * (~lc.act_mask)
            #x = x * lc.gate * lc.act_mask + ox * (~lc.act_mask)  
        return x, new_kv_cache, gmu_mems, input_embed

    def post_process_residual(self, h, ox, lc, residual_dropout: float = 0.0) -> torch.Tensor:
        if lc is not None:
            h = lc.extract(h) * torch.sqrt((2 * lc.gate - 1) * lc.act_mask + 1e-5)
            #h = lc.extract(h) * lc.gate * lc.act_mask / math.sqrt(2 * self.config.n_layer / self.config.sparsity)
        elif self.config.depth_scale:
            h = h / math.sqrt(2 * self.config.n_layer)
        if self.training and residual_dropout > 0:
            h = F.dropout(h, p=residual_dropout, training=True)
        if self.config.no_skip:
            x = h
        elif self.config.sum_skip:
            x = h
            input_embed = input_embed.to(torch.float32) # reduce error accumulation across layers during inference
            input_embed = input_embed + h.to(torch.float32)
        else:
            x = ox + h
        return x, h

# GPT-OSS SwiGLU
@torch.compile
def oss_swiglu(x_glu: torch.Tensor, x_linear: torch.Tensor, 
            alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    # chunk is faster than interleave on torch
    # # x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)

class LLaMAMLP(nn.Module):
    def __init__(self, config: Config,) -> None:
        super().__init__()
        self.relu2 = config.mlp_relu2
        self.config = config
        self.legacy_swiglu = False
        self.oss_swiglu = config.oss_swiglu
        if self.relu2:
            in_size = int(config.intermediate_size*1.5)
            out_size = int(config.intermediate_size*1.5)
        else:
            in_size = int(config.intermediate_size*2)
            out_size = config.intermediate_size
        if self.legacy_swiglu:
            self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False)
        else:
            self.w1 = nn.Linear(config.n_embd, in_size, bias=config.bias)
            self.w3 = nn.Linear(out_size, config.n_embd, bias=config.bias)
            if self.config.ffn_norm:
                self.norm = config.norm_class(out_size, eps=config.norm_eps)

    def reset_parameters(self) -> None:
        """Initialize MLP parameters using PyTorch default initialization."""
        if hasattr(self, 'w1'):
            utils.torch_default_init(self.w1.weight)
            if self.w1.bias is not None:
                nn.init.zeros_(self.w1.bias)
        if hasattr(self, 'w3'):
            utils.torch_default_init(self.w3.weight)
            if self.w3.bias is not None:
                nn.init.zeros_(self.w3.bias)
        if hasattr(self, 'swiglu') and self.legacy_swiglu:
            self.swiglu.reset_parameters()

    @torch.compile
    def forward(self, x: torch.Tensor, layer_idx: int, gmu_mems: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.relu2:
            x = self.w1(x)
            x = F.relu(x).square()
            if self.config.ffn_norm:
                x = self.norm(x)
            x = self.w3(x)
        else:
            # SwiGLU implementation: split input into two parts, apply SwiGLU activation
            if self.legacy_swiglu:
                x = self.swiglu(x)
            else:
                x_gate, x_inp = self.w1(x).chunk(2, dim=-1)
                if self.config.gmu_mlp and layer_idx == self.config.n_layer//2+1:
                    gmu_mems = x_inp
                if self.oss_swiglu:
                    x = oss_swiglu(x_gate, x_inp)
                else:   
                    x = swiglu(x_gate, x_inp)
                if self.config.ffn_norm:
                    x = self.norm(x)
                x = self.w3(x)
        return x, gmu_mems

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, bias=False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)
        # from xformers.ops import SwiGLU
        # if config.no_mlp_bias:
        #     self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False) 
        # else:
        #     self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=config.bias, _pack_weights=False)

    def reset_parameters(self) -> None:
        """Initialize SwiGLU parameters using PyTorch default initialization."""
        for linear in [self.w1, self.w2, self.w3]:
            utils.torch_default_init(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.w3(x)
        return x




    