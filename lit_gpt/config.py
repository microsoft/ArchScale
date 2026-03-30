# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple
import torch.nn.functional as F

@dataclass
class Config:
    org: str = "Microsoft"
    name: str = "transformer_d8"
    block_size: int = 4096 # sequence length for training 
    vocab_size: int = 50254 # vocab size for training
    padding_multiple: int = 512 # vocab size should be at least muliple of 8 to be efficient on hardware. compute the closest value
    padded_vocab_size: Optional[int] = None # vocab size after padding. will overide padding_multiple
    n_layer: int = 16 # number of layers
    n_head: int = 32 # number of attention heads
    n_embd: int = 4096 # embedding dimension
    ar: int = None # Aspect ratio: n_embed = ar * n_layer
    mlp_expand: int = 4 # MLP expand ratio: intermediate_size = mlp_expand * n_embd
    scale_embed: bool = False # scale embedding with 1/sqrt(n_embd)
    eos_token_id: int = 2 # llama2 token id for eos
    head_dim: int = None # head dimension. will overide head_size
    n_query_groups: Optional[int] = None # equal to n_kv_heads
    tied_embed: bool = False # tie embedding
    intermediate_size: Optional[int] = None # intermediate size for MLP
    ## MoE ##
    moe: bool = False # use MoE
    sparsity: int = 1 # width sparsity
    top_k: int = 1 # top k for mod selection
    aux_gamma: float = 0.001 # weight for auxiliary loss
    aux_style: str = "switch" # Options: "entropy", "switch", "switch_seq", "both"
    share_expert: bool = False # one shared expert for MoE
    zero_mod: bool = False # zero experts might save per layer flops. Deprecated.
    sqrt_gate: bool = False # sqrt gate for MoE
    fuse_sort_and_padding: bool = True # fuse sort and padding for MoE
    use_sonicmoe: bool = True # use SonicMoE for optimized MoE kernels (requires sonicmoe package)
    global_aux: bool = False # use global auxiliary loss
    ## attention ##
    rotary_percentage: float = 1.0 # percentage of rotary embedding
    ada_rope: bool = False # use base = 0.09 * win**1.628.
    separate_qkv: bool = True # use separate qkv for attention
    fa2: bool = True 
    local_window: int = -1 # window size for sliding window attention
    use_cu_seqlen: bool = False # use cu_seqlen for variable length training
    qk_norm: bool = False # use qk normalization
    attn_norm: bool = False # use norm after attention
    nope: bool = False # not use position embedding
    sc_attn: bool = False # use short convolution with attention
    use_da: bool = False # use differential attention
    sink_attn: bool = False # use sink attention # need cudnn 9.13+
    rope_base: int = 10000  # base frequency for rope
    relu2: bool = False
    full_swa_extend: bool = False # extrapolate the full attention with swa
    da_const_lamb: bool = False # use constant lambda for differential attention
    use_sigmoid: bool = False # use sigmoid attention
    gated_attn: bool = False # use gated attention
    scaling_factor: float = 1.0 # scaling factor for yarn
    ## bias ##
    bias: bool = False # use bias for linear layers except attention linear layers
    attn_bias: bool = False # use bias for attention qkv linear layers
    attn_out_bias: bool = False # use bias for attention output linear layers
    no_mlp_bias: bool = False # not use bias in MLP
    ## MLP ##
    mlp: bool = True # use MLP
    oss_swiglu: bool = False # use GPT-OSS SwiGLU
    ffn_norm: bool = False # use norm after ffn
    mlp_relu2: bool = False # use relu2 in MLP
    ## Hybrid ##
    full_per_layer: int = 1000000 # use full attention at the end of every x layers
    rnn_per_layer: int = -1  # use rnn at the beginning of every x layers
    rnn_type: str = "mamba"  # Options: "mamba", "mamba2", "retnet", "gla", "delta", "gdn"
    attn_layer_pos: str = None # For attn_layer_pos, RNN is used when NOT in the attention layer positions
    yoco: bool = False # use YOCO: you only cache once decoder-decoder architecture
    gmu_yoco: bool = False # use deocder-hybrid-decoder architecture with GMU
    gmu_per_layer: int = 2 # use GMU every x layers
    gmu_attn: bool = False # use GMU in attention
    gmu_mlp: bool = False # use GMU in MLP
    jamba_norm: bool = False # use Jamba-style normalization in mamba layer
    yoco_nope: bool = False # only use nope in YOCO
    yoco_window: bool = False  # use half block size SWAfor YOCO
    ## skip connection ##
    post_norm: bool = False # use post-norm for residual connection
    decouple_postnorm: bool = False # decouple scaler post-norm for residual connection
    skip_gain: bool = False # add skip scaler for post-norm for residual connection
    skip_weight_per_layer: int = -1 # add weight projection every x layers for post-norm for residual connection
    residual_in_fp32: bool = True # use residual in fp32. Deprecated.
    no_skip: bool = False # not use skip connection
    lc: bool = False 
    lc_shared: bool = False # if True, share one LayerConfigurator across layers; else per-layer
    sc_lc: bool = False
    sum_skip: bool = False # sum skip connection
    ## loop ##
    share_per_layer: int = -1
    share_group: int = 1
    compress: bool = False
    ## scalers ##
    attn_scale: float = None # attention logits multiplier
    w_init_scale: float = 1.0 # weight initialization multiplier
    ## scaling ##
    head_init_std: float = None # lm head initialization std
    mup_d0: int = 16 # base depth for muP++
    mup: bool = False # use muP++
    depth_scale: bool = False # scale down residual by 1/sqrt(L)
    use_muon: bool = False # use muon optimizer
    use_muonh: bool = False # use muon-hyperball optimizer
    super_mup: bool = False
    mup_hd0: int = 128 # base head dimension for muP++
    shadow_init: bool = False
    original_mup: bool = False  # original muP with proper width scaling
    _norm_class: Literal["LayerNorm", "RMSNorm","FusedRMSNorm","RMSNormFunc","LayerNormFunc","NoNorm"] = "FusedRMSNorm"
    norm_eps: float = 1e-5 # epsilon for normalization
    # Auxiliary losses
    lc_budget_tau: float = 0.25  # target active fraction for LC router
    lc_budget_lambda: float = 0.01  # weight for LC activation budget loss

    def __post_init__(self):
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # default intermediate size for MLP if not set
        if self.intermediate_size is None:
            self.intermediate_size = self.mlp_expand * self.n_embd
        
        if self.ar is not None:
            self.n_embd = self.ar * self.n_layer
            self.intermediate_size = self.mlp_expand * self.n_embd
        
        if self.sparsity > 1:
            self.lc_budget_tau = 1 / self.sparsity
        # error checking
        assert self.n_embd % self.n_head == 0

    @property
    def head_size(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        else:
            return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)


    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm
            return FusedRMSNorm
        elif self._norm_class == "RMSNormFunc":
            return RMSNormFunc
        elif self._norm_class == "LayerNormFunc":
            return LayerNormFunc
        elif self._norm_class == "NoNorm":
            return NoNorm
        return getattr(torch.nn, self._norm_class)

# no normalization
class NoNorm(torch.nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

# much faster than F.rms_norm on b200
class RMSNormFunc(torch.nn.Module):
    def __init__(self, size = None, eps=1e-5, decouple_gain=False, skip_gain=False):
        super().__init__()
        self.eps = eps
        self.skip_gain = skip_gain
        self.decouple_gain = decouple_gain
        if decouple_gain:
            self.weight = torch.nn.Parameter(torch.ones(size))
        if skip_gain:
            self.weight1 = torch.nn.Parameter(torch.ones(size))

    @torch.compile 
    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return x_normed

    def reset_parameters(self):
        if self.decouple_gain:
            torch.nn.init.ones_(self.weight)
        if self.skip_gain:
            torch.nn.init.constant_(self.weight1, 1e-4) # RWKV init

# much faster than F.layer_norm on b200
class LayerNormFunc(torch.nn.Module):
    def __init__(self, size=None, eps=1e-5):
        super().__init__()
        self.eps = eps

    @torch.compile
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, correction=0, dim=-1, keepdim=True)
        x_normed = (x - mean) * torch.rsqrt(var + self.eps)
        return x_normed


def get_parameters_count(model_name, depth, model_config, train_config):
    ar = model_config.ar
    n_mult = 14.5 * (ar ** 2) # 237568 # default transformer
    if "samba" in model_name:
        n_mult = 15 * (ar ** 2) + 160 * ar
    if "sambay" in model_name or "sambay2" in model_name or "phi4miniflash" in model_name:
        n_mult = 14.5 * (ar ** 2) + 144 * ar
    if "sambayoco" in model_name or "sambayattn" in model_name:
        n_mult = 13.5 * (ar ** 2) + 208 * ar
    if "mambay" in model_name or "gdny" in model_name or "mambay2" in model_name:
        n_mult = 16 * (ar ** 2) + 64 * ar 
    if "sambaymlp" in model_name:
        n_mult = 15.5 * (ar ** 2) + 144 * ar    
    if "swayoco" in model_name:
        n_mult = 12 * (ar ** 2) + 288 * ar 
    if "transformer" in model_name:
        n_mult = 12 * (ar ** 2)
    n_base = n_mult * (depth ** 3)        
    if "v2scale" in train_config:
        # Chichilla law also counts embeddings and lm_head
        n_head_base = depth * model_config.ar * model_config.vocab_size
        tied_embed = "_tie" in train_config or model_config.tied_embed
        if not tied_embed:
            n_head_base = n_head_base * 2
        n_base = n_base + n_head_base
    if model_config.skip_weight_per_layer > 0:
        n_base = n_base + model_config.n_embd ** 2 * depth // model_config.skip_weight_per_layer
    # only transformer for now
    if "transformer" in model_name:
        # qo proj
        n_base = n_base + ar * depth ** 2 * model_config.n_head * model_config.head_size * 2
        # kv proj
        n_base = n_base + ar * depth ** 2 * model_config.n_query_groups * model_config.head_size * 2
    return n_base

configs=[]

phi4_mini_flash_configs = [
    dict(
        org="Microsoft",
        name="phi4miniflash_d32", 
        block_size=8192,
        vocab_size=200064,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        attn_bias = True,
        attn_out_bias = True,
        n_layer=32,
        n_head=40,
        use_da = True,
        head_dim=64,
        ar = 80,
        tied_embed = True,
        _norm_class = "LayerNorm",
        n_query_groups= 20, 
        mlp_expand= 4, 
        local_window = 512, 
    )
]
configs.extend(phi4_mini_flash_configs)



scaling_xformersa_configs = [
        dict(
        org="Microsoft",
        name="transformersa_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        sink_attn = True,
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformersa_configs)

scaling_xformer_configs = [
        dict(
        org="Microsoft",
        name="transformer_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= d//4, 
        mlp_expand= 4, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_configs)


scaling_xformer_mha_configs = [
        dict(
        org="Microsoft",
        name="transformer_mha_d"+str(d),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= d, 
        mlp_expand= 4, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_mha_configs)

scaling_xformer_gqa4_configs = [
        dict(
        org="Microsoft",
        name="transformer_gqa4_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= 4, 
        mlp_expand= 4, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_gqa4_configs)

scaling_xformer_gqa4_configs = [
        dict(
        org="Microsoft",
        name="transformer_gqa4_h2_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= 2*d, # need to be divisible by 8 for muon with gated attention, otherwise nccl will hang
        head_dim= 128,
        ar = 128,
        n_query_groups= 4, 
        mlp_expand= 4, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_gqa4_configs)

scaling_xformer_gqa4_configs = [
        dict(
        org="Microsoft",
        name="transformer_gqa4_h2_skipw_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        skip_weight_per_layer= 4,
        n_layer= d,
        n_head= 2*d, # need to be divisible by 8 for muon with gated attention, otherwise nccl will hang
        head_dim= 128,
        ar = 128,
        n_query_groups= 4, 
        mlp_expand= 4, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_gqa4_configs)

scaling_moe_gqa4_configs = [
        dict(
        org="Microsoft",
        name="transformer_gqa4_h2_moe_s"+str(s)+"_k"+str(k)+"_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= 2*d,
        head_dim= 128,
        ar = 128,
        n_query_groups= 4, 
        mlp_expand= 4, 
        moe = True,
        top_k = k,
        sparsity = s,
    )
    for k in [1,2,4,8,16,32,64]
    for s in [1,2,4,8,16,32,64]
    for d in [8,12,16,20,24]
]
configs.extend(scaling_moe_gqa4_configs)

scaling_moe_gqa4_configs = [
        dict(
        org="Microsoft",
        name="transformer_gqa4_h2_skipw_moe_s"+str(s)+"_k"+str(k)+"_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        skip_weight_per_layer= 4,
        n_layer= d,
        n_head= 2*d,
        head_dim= 128,
        ar = 128,
        n_query_groups= 4, 
        mlp_expand= 4, 
        moe = True,
        top_k = k,
        sparsity = s,
    )
    for k in [1,2,4,8,16,32,64]
    for s in [1,2,4,8,16,32,64]
    for d in [8,12,16,20,24]
]
configs.extend(scaling_moe_gqa4_configs)

scaling_xformer_gqa4_configs = [
        dict(
        org="Microsoft",
        name="transformer_gqa4_w8_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= 4, 
        mlp_expand= 32, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_gqa4_configs)

scaling_moe_gqa4_configs = [
        dict(
        org="Microsoft",
        name="sambay_gqa4_moe_s"+str(s)+"_k"+str(k)+"_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 124,
        n_query_groups= 4, 
        mlp_expand= 4, 
        local_window = 128, 
        moe = True,
        top_k = k,
        sparsity = s,
    )
    for k in [1,2,4,8]
    for s in [1,2,4,8,16,32,64]
    for d in [8,12,16,20,24]
]
configs.extend(scaling_moe_gqa4_configs)

scaling_moe_gqa4_configs = [
        dict(
        org="Microsoft",
        name="transformer_gqa4_moe_s"+str(s)+"_k"+str(k)+"_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= 4, 
        mlp_expand= 4, 
        moe = True,
        top_k = k,
        sparsity = s,
    )
    for k in [1,2,4,8]
    for s in [1,2,4,8,16,32,64]
    for d in [8,12,16,20,24]
]
configs.extend(scaling_moe_gqa4_configs)

scaling_xformer_gqa4_ga_configs = [
        dict(
        org="Microsoft",
        name="transformerga_gqa4_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 124,
        gated_attn = True,
        n_query_groups= 4, 
        mlp_expand= 4, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_gqa4_ga_configs)


scaling_xformerls_configs = [
        dict(
        org="Microsoft",
        name="transformerls_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        full_per_layer = 4,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformerls_configs)

scaling_samba_configs = [
    dict(
        org="Microsoft",
        name="samba_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 122,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 2048, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_samba_configs)

abaltion_configs = [
    dict(
        org="Microsoft",
        name="mambay_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=1,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="mambay2_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=1,
        rnn_type="mamba2",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="gdny_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=1,
        rnn_type="gdn",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sgdny_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="gdn",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 126,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambay2_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba2",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 124,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambayattn_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        gmu_attn = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 126,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambayattnall_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        gmu_attn = True,
        gmu_per_layer = 1,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 126,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambaymlp_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        gmu_mlp = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
]
configs.extend(abaltion_configs)

scaling_swayoco_configs = [
    dict(
        org="Microsoft",
        name="swayoco_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        yoco = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 130,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_swayoco_configs)


scaling_sambaywr_configs = [
    dict(
        org="Microsoft",
        name="sambaywr_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        yoco_window = True,
        gmu_yoco = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambaywr_configs)

scaling_sambaywdr_configs = [
    dict(
        org="Microsoft",
        name="sambaywdr_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        yoco_window = True,
        gmu_yoco = True,
        ada_rope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambaywdr_configs)


scaling_sambayw_configs = [
    dict(
        org="Microsoft",
        name="sambayw_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        yoco_window = True,
        gmu_yoco = True,
        nope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambayw_configs)


scaling_sambaywsa_configs = [
    dict(
        org="Microsoft",
        name="sambaywsa_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        yoco_window = True,
        sink_attn = True,
        gmu_yoco = True,
        nope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambaywsa_configs)


scaling_sambay_configs = [
    dict(
        org="Microsoft",
        name="sambay_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambay_configs)

scaling_sambay_da_configs = [
    dict(
        org="Microsoft",
        name="sambayda_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=d,
        n_head=d,
        use_da = True,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambay_da_configs)

scaling_sambayoco_configs = [
    dict(
        org="Microsoft",
        name="sambayoco_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = False,
        nope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 126,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambayoco_configs)

name_to_config = {config["name"]: config for config in configs}
