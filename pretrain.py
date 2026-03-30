# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Literal, Dict, List, Callable
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
import torch.nn as nn
from lightning.fabric.strategies import ModelParallelStrategy
from torch.utils.data import DataLoader
import json
from dataclasses import dataclass, field
import re
from logging_utils import (
    submit_logging_task,
    drain_logging_queue,
    wait_for_logging_tasks,
    build_logging_payload,
    configure_logging_executor,
    save_weights_for_update_logging,
    capture_weight_update_data,
)
# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import num_parameters
from lit_gpt.aux_loss import activation_budget_bce
from lit_gpt.config import get_parameters_count
try:
    from lit_gpt.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
except:
    FusedLinearCrossEntropyLoss = None
from lightning.pytorch.loggers import WandbLogger
import utils
import blobfile as bf
from lit_gpt.optim.muon import Muon
from lit_gpt import FusedCrossEntropyLoss
import random
import os

from emerging_optimizers.orthogonalized_optimizers.muon_ball import MuonBall
from dataclasses import dataclass as dc_dataclass
from torch.distributed._composable import checkpoint
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
import itertools

from precision import FP8FSDPPrecision
torch._dynamo.config.capture_scalar_outputs = True
# # Workaround for https://github.com/pytorch/pytorch/issues/166926
# torch._C._dynamo.eval_frame._set_lru_cache(False)

@dc_dataclass
class ProcessGroupCollection:
    """Simple wrapper to provide process groups for MuonBall optimizer.
    
    MuonBall expects a pg_collection object with .tp attribute for tensor parallel
    and optionally .expt_tp for expert tensor parallel (MoE).
    """
    tp: torch.distributed.ProcessGroup = None
    expt_tp: torch.distributed.ProcessGroup = None

rng_seed = 3407

vector_names = ["conv1d", "wte", "lm_head", "norm", "ln", "layernorm"]

@dataclass
class BaseHyperparameters:
    """Base optimization hyperparameters for MuP (mu-parametrization) transformation"""
    d0: int = 8                   # Base depth/layers
    eta0: float = 6e-4             # Base learning rate 
    b0: int = 2**21                # Base batch size (2M tokens)
    t0: Union[int,float] = int(9.5e9) # Base tokens (9.5B)
    hd0: int = 128                 # Base head dimension
    w_init_scale: float = 1.0      # Base weight initialization multiplier
    n0_mult: float = 237568        # Base parameter count multiplier (n0 = n0_mult * d0^3)
    weight_decay: float = 1e-1    # Base weight decay
    eps: float = 1e-8             # Base optimizer epsilon
    min_lr_mult: float = 0.0      # Base minimum learning rate multiplier
    warmup_tokens: Union[int,float] = int(1e9) # Base warmup tokens
    beta1: float = 0.9           # Base Adam beta1
    beta2: float = 0.95          # Base Adam beta2
    muon_beta: float = 0.95      # Base Muon beta
    weight_lr_scale: float = 1  # scale up learning rate for weight parameters for hybrid optimizers 
    

# Global base hyperparameters object
base_hps = BaseHyperparameters()

model_name = "transformer"
train_config = "scaling_mup"
name = train_config +"_" + model_name

out_dir = os.getenv("LIGHTNING_ARTIFACTS_DIR", "out") + "/" + name
ckpt_dir = None
devices = torch.cuda.device_count() or 1

label_smoothing = 0.1 if "_ls_" in name else 0.0

mup = False
super_mup = False
original_mup = False
use_cu_seqlen = False

use_model_parallel = False

nodes = int(os.getenv("WORLD_SIZE", 1)) // int(os.getenv("LOCAL_WORLD_SIZE", 1))

code_dropout = 0.0

# Default Hyperparameters, will be overriden by setup()
train_tokens = int(1e11) # 100 billion
global_batch_size = 512 // nodes
micro_batch_size = 8 
use_flce_loss = False
depth_global = base_hps.d0 # record depth in global scope
seq_len = 4096
local_window = None
learning_rate = base_hps.eta0
beta1 = base_hps.beta1
beta2 = base_hps.beta2
weight_decay = base_hps.weight_decay
warmup_tokens = base_hps.warmup_tokens
muon_beta = base_hps.muon_beta
weight_lr_scale = base_hps.weight_lr_scale
eps = base_hps.eps
min_lr = base_hps.min_lr_mult
grad_clip = 1.0
decay_lr = True
total_evals = 400
bsz_scaling = None  # Batch size scaling strategy: "sde", "sdew", or "default"
bsz_rampup_ratio = 1
log_step_interval = 10
eval_iters = total_evals // micro_batch_size # 50 # 25 # eval is invariant to microbatch size
save_step_interval = 1000
eval_step_interval = 1000

cpu_offload = False
activation_checkpointing = False

data_chunks = 8
num_extrapol = 4

batch_size = global_batch_size // devices
gradient_accumulation_steps = max(batch_size // micro_batch_size, 1)

log_iter_interval = log_step_interval * gradient_accumulation_steps


train_data_config = [ ("train", 1.0) ]
val_data_config = [ ("validation", 1.0) ]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

wandb_logger = None

def setup(
    train_data_dir = "data/redpajama_sample",
    val_data_dir: Optional[str] = None,
    load_dir: str = None,
    resume: Union[bool, Literal["auto"], str] = False,
    train_model: str = None,
    train_name: str = "scaling_mup_tie_rbase_prolong_varlen",
    #"scaling_mup_tie_rbase_prolong_varlen", "scaling_mup", 
    depth: int = 16,
    max_tokens: float = None,
    ctx_len: int = None,
    swa_len: int = None,
    data_mixture: str = None,
    fsdp_save_mem: bool = False,
    use_mlflow: bool = False,
    save_step: int = None,
    base_hps: BaseHyperparameters = None,
    micro_bsz: int = 2,
    total_bsz: int = None,
    yarn_scale: float = 1.0,  # scaling factor for yarn, = ctx len / pretrain ctx len
    sparsity: int = 1,
    top_k: int = 2,
    use_flce: bool = None,
    norm_class: str = None,
    fp8: bool = False,
    fp4: bool = False,
    post_norm: bool = None,
    bramp_ratio: int = None,
    aux_style: str = None,
    aux_gamma: float = None,
    fsdp2: bool = False,
    num_extrp: int = None,
    offloading: bool = False,
    ortho_init: bool = False,
    diag_init: bool = False,
    maxfan_init: bool = False,
    act_ckpt: bool = False,
    share_expert: bool = None,
    global_aux: bool = None,
    sqrt_gate: bool = None,
    head_init_std: float = 0.02, # turn to None for zero lm head initialization
    logging_multiprocessing: bool = True,  # Use multiprocessing for faster logging with long sequences
    ar: int = None,  # Aspect ratio override: n_embd = ar * n_layer
    wandb_project: str = "discrete-llm",
) -> None:
    global model_name, train_config, name, out_dir, ckpt_dir, devices, learning_rate, nodes, train_tokens, \
        global_batch_size, micro_batch_size, total_evals, warmup_tokens, log_step_interval, \
        eval_iters, min_lr, batch_size, gradient_accumulation_steps, save_step_interval, log_iter_interval, hparams, \
        depth_global, weight_decay, eps, beta1, beta2, muon_beta, weight_lr_scale, label_smoothing, grad_clip, \
        mup, wandb_logger, super_mup, seq_len, local_window, num_extrapol, use_cu_seqlen, train_data_config, \
        original_mup, use_flce_loss, cpu_offload, activation_checkpointing, bsz_scaling, bsz_rampup_ratio, data_chunks, use_model_parallel
    
    # Configure logging executor (multiprocessing is faster for long sequences)
    configure_logging_executor(use_multiprocessing=logging_multiprocessing, max_workers=1)
    if ortho_init:
        utils.torch_default_init = utils.global_orthogonal_init_dtensor_
    if diag_init:
        utils.torch_default_init = utils.global_diagonal_init_dtensor_
    if maxfan_init:
        utils.torch_default_init = utils.global_max_fan_normal_init_dtensor_
    # overrides
    # Update global base_hps if provided
    if base_hps is not None:
        globals()['base_hps'] = base_hps
    else:
        base_hps = globals()['base_hps']
    seq_len = ctx_len if ctx_len is not None else seq_len
    num_extrapol = num_extrp if num_extrp is not None else num_extrapol
    save_step_interval = save_step if save_step is not None else save_step_interval
    model_name = train_model if train_model is not None else model_name
    train_config = train_name if train_name is not None else train_config
    local_window = swa_len if swa_len is not None else local_window
    ckpt_dir = load_dir if load_dir is not None else ckpt_dir
    bsz_rampup_ratio = bramp_ratio if bramp_ratio is not None else bsz_rampup_ratio
    data_chunks = 128 if "v2scale" in train_config else data_chunks
    if fsdp_save_mem:
        offloading = True
        act_ckpt = True
    fp8 = True if fp4 else fp8

    # config data mixture
    if data_mixture is not None:
        # Load dataset weights from JSON file
        with open(data_mixture, 'r') as f:
            dataset_weights = json.load(f)

        # Convert dataset weights to train_data_config format
        train_data_config = [(dir_path, weight) for dir_path, weight in dataset_weights.items()]

    if "v2scale" in train_config:
        print(f"v2scale uses independent weight decay! Default weight decay is 4e-5!")
    # parsing configs
    assert depth is not None
    # have mup_muonh in train_config name to use hyperp.
    mup= "mup" in train_config or "muonh" in train_config # use mup++
    original_mup = "_ori_mup" in train_config # use original mup
    rope_base = 640_000 if "_rbase_" in train_config else 10_000 # rope base for 32k ctx len
    super_mup = "super_mup" in train_config
    use_cu_seqlen = "_varlen" in train_config # use cu_seqlen for variable length training
    label_smoothing = 0.1 if "_ls_" in train_config else 0.0
    if "_moe" in model_name:
        model_name = model_name+"_s"+str(sparsity)+"_k"+str(top_k)
    base_model_name = model_name+"_d"+str(base_hps.d0)
    model_name = model_name+"_d"+str(depth)
    model_config = Config.from_name(model_name)
    base_model_config = Config.from_name(base_model_name)
    use_flce_loss = use_flce if use_flce is not None else \
                model_config.vocab_size > 100_000 or \
                "validation" in train_config or fp8
    
    name = train_config +"_" + model_name
    if ar is not None:
        name = name+ "_ar" + str(ar)
    name = name+ "_ctx" + str(seq_len)
    if local_window is not None:
        name = name+ "_swa" + str(local_window)
    if bsz_rampup_ratio != 1:
        name = name+ "_bramp" + str(bsz_rampup_ratio)
        
    wandb_logger = WandbLogger(project=wandb_project, name=name)
    out_dir = os.getenv("LIGHTNING_ARTIFACTS_DIR", "out") + "/" + name
            
    if seq_len == 4096: # hardcoded for now
        micro_batch_size = micro_bsz
    else:
        # long context mid training
        micro_batch_size = 1
        #num_extrapol = 2
                  
    eos_token_id = 2
    
    depth_global = depth

    if max_tokens is not None:
        train_tokens = max_tokens
    else: 
        if ar is not None:
            model_config.ar = ar
            base_model_config.ar = ar
        n_base = get_parameters_count(base_model_name, base_hps.d0, base_model_config, train_config)
        n_target = get_parameters_count(model_name, depth, model_config, train_config)
        # Scale tokens based on parameter count (Chinchilla scaling)
        train_tokens = int(base_hps.t0 * n_target / n_base)
    
    # scaling law for critical batch size
    # -0.5 is not correct empirically for muonh
    # empirically trivial to find and leave for future work
    scaled_bsz = base_hps.b0 
    raw_b = scaled_bsz if total_bsz is None else total_bsz
    multiple = nodes * devices * micro_batch_size * seq_len
    b = ((raw_b + multiple // 2) // multiple) * multiple # nearest multiple
    
    if "muonball" in train_config or "muonh" in train_config:
        base_hps.weight_decay = 0.0
    # Calculate batch ratio and apply scaling
    batch_ratio = b / base_hps.b0

    # the sqrt scaling backed by sde paper
    # we also empirically verified it on muonh
    bsz_scaling = "default" 
    scaled_params = apply_batch_size_scaling(batch_ratio, base_hps, bsz_scaling)
    
    learning_rate = scaled_params['learning_rate']
    eps = scaled_params['eps']
    beta1 = scaled_params['beta1']
    beta2 = scaled_params['beta2']
    weight_decay = scaled_params['weight_decay']

    depth_mup = mup and not original_mup
    depth_scale = depth_mup or "_dsca_" in train_config
    if depth_mup:
        learning_rate = learning_rate * math.sqrt(base_hps.d0 / depth)

    # scaling law for optimal learning rate empirically observed for muonh
    # -0.32 also found in https://arxiv.org/abs/2409.19913 for adamw
    if "muonh" in train_config:
        learning_rate = learning_rate * (train_tokens/base_hps.t0)**(-0.32)

    # Use base hyperparameters for other parameters
    min_lr = base_hps.min_lr_mult
    if "midtrain" in train_config or "muon" in train_config:
        warmup_tokens = 0
    else:
        if super_mup:
            warmup_tokens = int(base_hps.warmup_tokens / base_hps.t0 * train_tokens )
        else:
            warmup_tokens = int(base_hps.warmup_tokens)
    muon_beta = base_hps.muon_beta
    weight_lr_scale = base_hps.weight_lr_scale
    
    # weight decay scaling for second order optimizers 
    # https://arxiv.org/abs/2512.05620
    if "muon" in train_config:
        weight_decay = weight_decay * base_hps.d0 / depth
    global_batch_size =  b // (seq_len * nodes)

    if micro_batch_size == 1: # because eval batch size is ceil devided by 2
        eval_iters = total_evals // micro_batch_size // 2 # 50 # 25
    else:
        eval_iters = total_evals // micro_batch_size # 50 # 25


    batch_size = global_batch_size // devices
    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps > 0

    log_iter_interval = log_step_interval * gradient_accumulation_steps

    # setup strategy
    use_model_parallel = "muon" in train_config or fp8 or fsdp2    
    cpu_offload = offloading # fsdp_save_mem
    activation_checkpointing = act_ckpt # fsdp_save_mem 
    if use_model_parallel:
        # we are using hsdp with tensor parallel size, upto 72 for nvl72
        strategy = ModelParallelStrategy(data_parallel_size=nodes, tensor_parallel_size=devices, 
                    parallelize_fn=configure_model, save_distributed_checkpoint=False)
    else:
        ac_policy = {Block} if activation_checkpointing else None
        strategy = FSDPStrategy(auto_wrap_policy={Block}, activation_checkpointing_policy=ac_policy,
                    sharding_strategy = "HYBRID_SHARD", cpu_offload=cpu_offload, state_dict_type="full")
        
    if use_mlflow:
        from azureml.core import Run
        from lightning.pytorch.utilities.rank_zero import rank_zero_only
        class MLFlowLogger:
            def __init__(self):
                self.run = Run.get_context()
                
            @rank_zero_only
            def log_metrics(self, metrics, step):
                for key, value in metrics.items():
                    self.run.log(key, value, step = step)

        mlf_logger = MLFlowLogger()
    else:
        mlf_logger = None
    
    if fp8:
        fabric = L.Fabric(devices=devices, num_nodes=nodes, strategy=strategy,
                        plugins= FP8FSDPPrecision(precision="bf16-mixed", fp4=fp4), 
                       loggers=[ mlf_logger if use_mlflow else wandb_logger])
    else:
        fabric = L.Fabric(devices=devices, num_nodes=nodes, strategy=strategy, 
                    precision= None if use_model_parallel else "bf16-mixed",
                       loggers=[ mlf_logger if use_mlflow else wandb_logger])
    fabric.launch()

    # overrides for model config
    if "midtrain" in train_config or "validation" in train_config:
        scaling_factor = yarn_scale
    else:
        scaling_factor = 1.0
        
    overides = {"mup": mup, "depth_scale": depth_scale, "super_mup": super_mup, "mup_d0": base_hps.d0, 
            "mup_hd0": base_hps.hd0, "w_init_scale": base_hps.w_init_scale, "block_size": seq_len,
            "use_cu_seqlen": use_cu_seqlen,
            "head_init_std": head_init_std,
            "original_mup": original_mup,
            "rope_base": rope_base,
            "eos_token_id": eos_token_id,
            "scaling_factor": scaling_factor,
            }
    
    if "validation" in train_config:
        assert ckpt_dir is not None
        m = re.search(r"_swa(\d+)", ckpt_dir)
        if m:
            swa_size = int(m.group(1)) 
            fabric.print(f"{swa_size=}") 
            overides["local_window"] = swa_size 
        m = re.search(r"_ctx(\d+)", ckpt_dir)
        if m:
            ctx_size = int(m.group(1)) 
            fabric.print(f"{ctx_size=}") 
            overides["block_size"] = ctx_size 
    
    if swa_len is not None:
        overides["local_window"] = swa_len
    if ar is not None:
        overides["ar"] = ar
    if aux_style is not None:
        overides["aux_style"] = aux_style
    if aux_gamma is not None:
        overides["aux_gamma"] = aux_gamma
    if norm_class is not None:
        overides["_norm_class"] = norm_class
    if post_norm is not None:
        overides["post_norm"] = post_norm
    if sqrt_gate is not None:
        overides["sqrt_gate"] = sqrt_gate
    if share_expert is not None:
        overides["share_expert"] = share_expert
    if global_aux is not None:
        overides["global_aux"] = global_aux
    if "_bias" in train_config:
        overides["attn_bias"] = True
        overides["attn_out_bias"] = True
        overides["bias"] = True
    if "muon" in train_config:
        overides["use_muon"] = True
    if "muonh" in train_config:
        overides["use_muonh"] = True
    if "_tie" in train_config: # use tied embedding
        overides["tied_embed"] = True
    if "_lc" in train_config:
        overides["lc"] = True
    if "postnorm_decouple" in train_config:
        overides["decouple_postnorm"] = True
    if "_skipgain" in train_config:
        overides["skip_gain"] = True
    if "_qknorm" in train_config:
        overides["qk_norm"] = True
    if "_anorm" in train_config:
        overides["attn_norm"] = True
    if "_fnorm" in train_config:
        overides["ffn_norm"] = True
    if "_nope" in train_config:
        overides["nope"] = True
    if "_ga_" in train_config:
        overides["gated_attn"] = True
    if "_noskip" in train_config:
        overides["no_skip"] = True
    if "_noskip_sum" in train_config:
        overides["sum_skip"] = True
        overides["no_skip"] = False
    # log hparams
    # Include both local and global variables in hparams
    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    # Add relevant global variables
    global_vars = {k: v for k, v in globals().items() 
                  if isinstance(v, (int, float, str)) and not k.startswith("_") 
                  and k not in hparams}
    hparams.update(global_vars)
    # Add base hyperparameters to logged hparams
    base_hps_dict = {f"base_hps.{k}": v for k, v in base_hps.__dict__.items()}
    hparams.update(base_hps_dict)
    fabric.print(hparams)
    if not use_mlflow:
        fabric.logger.log_hyperparams(hparams)
    main(fabric, train_data_dir, val_data_dir, resume, fsdp_save_mem, **overides)

#### Methods to save the custom attributes of QuantizedTensors before sharding
#### them with FSDP2, and restore them after sharding.
def save_custom_attrs(module):
    from transformer_engine.pytorch.tensor import QuantizedTensor
    custom_attrs = {}
    for name, param in module.named_parameters():
        if isinstance(param, QuantizedTensor):
            # Ignore FP8 metadata attributes. Otherwise we will save duplicate copies
            # for data/transpose FP8 tensors on top of FP8 tensors that FSDP2 will save.
            ignore_keys = [key for key in param.__dict__.keys() if key.startswith("_")]
        else:
            ignore_keys = []
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items() if k not in ignore_keys}
    return custom_attrs


def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)

def configure_model(model: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    
    print("device_mesh:", device_mesh)
    # use bf16 param can be problematic for rnn A_log matrix, need to use fp32 param 
    # need also to set use_flce=True for memory efficient fp32 logits computation
    # todo: do fp8 gather for te linear layers
    torch_compile = True
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    cpu_offload_policy = CPUOffloadPolicy() if cpu_offload else None
    fp32policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32)
    custom_attrs = save_custom_attrs(model)
    
    for module in model.modules():
        if isinstance(module, Block):
            if activation_checkpointing:
                # end-to-end compile is buggy with checkpointing
                # see: https://github.com/pytorch/pytorch/issues/166926
                # https://github.com/pytorch/torchtitan/commit/b6b2c2de777ae393b73940c7d884da8fe365bdef
                torch_compile = False  
                checkpoint(module)
            fully_shard(module, mesh=device_mesh, mp_policy=mp_policy, offload_policy=cpu_offload_policy)
    fully_shard(model, mesh=device_mesh, mp_policy=mp_policy, offload_policy=cpu_offload_policy)
    
    restore_custom_attrs(model, custom_attrs)
    
    if torch_compile:
        model = torch.compile(model) 
    return model

def get_param_groups(model):
    """Group parameters by their types to apply different learning rate multipliers"""

    vector_decay = weight_decay if original_mup else 0.0
    # no need to scale lr for muon as we preserve rms norm with multiplier inside optimizer
    weight_lr_mult = 1.0 if "muon" in train_config else base_hps.d0 / depth_global 
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not "weight" in n.lower() or any( nd in n.lower() for nd in vector_names)
            ],
            "weight_decay": vector_decay,
            "lr_mult": 1.0,  # Base multiplier for no-decay parameters (vectors)
            "algorithm": "adamw"
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "weight" in n.lower() and not any( nd in n.lower() for nd in vector_names)
            ],
            "weight_decay": weight_decay,
            "lr_mult": weight_lr_mult,  # Base multiplier for weights
            "algorithm": "muon"
        }
    ]

    return param_groups

def main(fabric, train_data_dir, val_data_dir, resume, fsdp_save_mem, **overides):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        utils._make_dirs(out_dir)

    config = Config.from_name(model_name, **overides)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=rng_seed,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(rng_seed)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    
    # Use distributed initialization for FSDP2 to avoid OOM on large models
    # Create model on meta device, shard, then materialize and initialize
    if use_model_parallel:
        fabric.print("Using distributed initialization (meta device) for FSDP2...")
        with torch.device("meta"):
            model = GPT(config)
        # Note: _init_weights and reset_parameters will be called after sharding in configure_model
    else:
        # Standard initialization for non-FSDP2 paths
        with fabric.init_module(empty_init=False):
            model = GPT(config)  
 

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")
    fabric.print(model)

    if not use_model_parallel and not fsdp_save_mem:
        model = torch.compile(model) # comment this out for TP 
    model = fabric.setup(model)
    model.reset_parameters()
    
    if mup:
        param_groups = get_param_groups(model)
        # fabric.print(param_groups)
        
        if "hadam" in train_config:
            optimizer1 = torch.optim.AdamW(
                [param_groups[0]], lr=learning_rate, betas=(0.8, 0.95), eps=1e-10, fused=True
            )
            optimizer2 = torch.optim.NAdam(
                [param_groups[1]], lr=learning_rate * weight_lr_scale, betas=(beta1, beta2), eps=eps, decoupled_weight_decay=True
            )  
            #optimizer2 = torch.optim.SGD([param_groups[1]], momentum=0.95, lr=learning_rate, nesterov=True, fused=True)
            optimizer = [fabric.setup_optimizers(optimizer1), fabric.setup_optimizers(optimizer2)]
        elif "muonball" in train_config:
            # MuonBall: Spectral Ball with λ=0 (simplified version)
            # Separate optimizers: AdamW for vectors, MuonBall for 2D weight matrices
            vector_params = param_groups[0]  # vectors (1D params, embeddings, norms)
            weight_params = param_groups[1]  # weights (2D matrices)
            
            # AdamW for vector parameters
            optimizer1 = torch.optim.AdamW(
                [vector_params], lr=learning_rate, betas=(beta1, beta2), eps=eps, fused=True
            )
            
            # Create process group collection for tensor parallel support
            tp_mesh = fabric.strategy.device_mesh["tensor_parallel"]
            pg_collection = ProcessGroupCollection(tp=tp_mesh.get_group())
            
            # MuonBall for weight matrices
            optimizer2 = MuonBall(
                [weight_params],
                lr=learning_rate,
                momentum_beta=muon_beta,
                weight_decay=0.0,
                use_nesterov=True,
                pg_collection=pg_collection,
                radius_mode="spectral_mup",
                scale_mode="spectral_mup",
                power_iteration_steps=50,
                msign_steps=5,
            )
            optimizer = [fabric.setup_optimizers(optimizer1), fabric.setup_optimizers(optimizer2)]
        elif "muon" in train_config:
            optimizer = Muon(param_groups, distributed_mesh=fabric.strategy.device_mesh["tensor_parallel"], lr=learning_rate, mu=muon_beta, betas=(beta1, beta2),
                adjust_lr = "spectral_norm", use_triton=True, nesterov=True, use_muonh = "muonh" in train_config)
            optimizer.exclude_from_hyperball(model.transformer.wte.weight)
            optimizer = fabric.setup_optimizers(optimizer)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(beta1, beta2), eps=eps, fused=True)
            optimizer = fabric.setup_optimizers(optimizer)

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps, fused=True
        )
        optimizer = fabric.setup_optimizers(optimizer)

    if ckpt_dir is not None:
        state = {"model": model}
        if utils._exists_anywhere(ckpt_dir):
            fabric.print(f"Loading ckpt from {ckpt_dir}")
            utils._fabric_load_anywhere(fabric, ckpt_dir, state)
            state.update({"optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0})  # not resuming optimizer
            resume = False # not resume dataloader     
        else:
            raise ValueError(f"No ckpt found in {ckpt_dir}")
    else:   
        state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
        resume = find_resume_path(resume, out_dir)
        if resume :
            fabric.print(f"Resuming training from {resume}")
            utils._fabric_load_anywhere(fabric, resume, state)

    # Total local parameter bytes that actually allocate memory on this rank
    live_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.numel() > 0)
    print(f"{fabric.global_rank=} live_param_MB={live_bytes/1e6:.1f}")
    for n, p in model.named_parameters():
        fabric.print(n, p.shape)
        
    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume) # resume for dataloader
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    global learning_rate
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        # sanity check
        validate(fabric, model, val_dataloader, state, monitor, sanity_check= False if "validation" in train_config else True)
        
    if "validation" in train_config:
        return
    
    with torch.device("meta"):
        meta_model = GPT(model.config)
        # Use get_parameters_count for more accurate architecture-specific FLOPs estimation
        # This accounts for different model architectures (transformer, samba, sambay, etc.)
        n_params = get_parameters_count(model_name, depth_global, model.config, train_config)
        fabric.print(f"Estimated parameters for FLOPs (get_parameters_count): {n_params:,}")
        fabric.print(f"Actual model parameters (num_parameters): {num_parameters(model):,}")
        
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model, n_params=n_params) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    train_tokens_per_device = train_tokens // fabric.world_size
    tokens_per_iter = micro_batch_size * model.config.block_size
    max_iters = train_tokens_per_device // tokens_per_iter
    warmup_iters = warmup_tokens // fabric.world_size // tokens_per_iter
    initial_iter = state["iter_num"]
    curr_iter = 0
    if use_flce_loss:
        loss_func = FusedLinearCrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        loss_func = FusedCrossEntropyLoss(label_smoothing=label_smoothing)
        

    batch_rampup = bsz_rampup_ratio != 1
    if batch_rampup:
        final_grad_accum_steps = gradient_accumulation_steps * bsz_rampup_ratio
        fabric.print(f"Gradient accumulation steps: {gradient_accumulation_steps} -> {final_grad_accum_steps}")
    
    current_grad_accum_steps = gradient_accumulation_steps
    for train_data, is_code in train_dataloader:
        if isinstance(train_data, (list, tuple)) and len(train_data) == 2:
            train_data, is_code = train_data
            is_code = is_code[0]
        else:
            is_code = False
        drain_logging_queue(fabric)

        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            if val_dataloader is not None:
                validate(fabric, model, val_dataloader, state, monitor)
            checkpoint_path = out_dir + f"/iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving final checkpoint to {str(checkpoint_path)!r}")
            utils._fabric_save_anywhere(fabric, state, checkpoint_path)
            break
        
        # training
        if use_model_parallel:
            model.unshard()
        is_accumulating = (state["iter_num"] + 1) % current_grad_accum_steps != 0
        # Calculate dynamic gradient accumulation steps for batch size rampup
        batch_multiplier = None
        current_batch_ratio = None
        if batch_rampup and not is_accumulating:
            batch_multiplier = batch_size_rampup_func(
                state["iter_num"], 
                max_iters, 
                batch_size_ratio=bsz_rampup_ratio, 
            )
            # Linearly increase gradient accumulation steps from initial to final
            current_grad_accum_steps = int(gradient_accumulation_steps * batch_multiplier)
            # Calculate current batch size ratio relative to base (for optimizer scaling)
            # todo: consider tp size
            current_batch_size = micro_batch_size * current_grad_accum_steps * devices * nodes * seq_len
            current_batch_ratio = current_batch_size / base_hps.b0

            scaled_params = apply_batch_size_scaling(current_batch_ratio, base_hps, bsz_scaling)
            learning_rate = scaled_params['learning_rate']  # Use the scaled LR

        # Determine and set learning rate + batch size scaled parameters for this iteration
        def update_param(param_group, lr_mult):
            param_group["lr"] = lr * lr_mult
            # Apply batch size scaled optimizer params if rampup is enabled
            if batch_rampup and not is_accumulating:
                param_group["eps"] = scaled_params['eps']
                param_group["betas"] = (scaled_params['beta1'], scaled_params['beta2'])
                if "weight_decay" in param_group and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = scaled_params['weight_decay']
                    if "v2scale" in train_config:
                        # independent wd:
                        # https://arxiv.org/abs/2309.14322
                        # original base learning rate is 4e-4
                        param_group["weight_decay"] = param_group["weight_decay"] / learning_rate / lr_mult
            elif "weight_decay" in param_group and param_group["weight_decay"] > 0:
                if "v2scale" in train_config:
                    # independent wd: 
                    # https://arxiv.org/abs/2309.14322
                    # original base learning rate is 4e-4
                    param_group["weight_decay"] = weight_decay / learning_rate / lr_mult          
                if super_mup:
                    # todo: scheduler beta 1 to be between 0.95 -> 0.85 following learning rate schedule
                    #param_group["betas"] = (0.95 - (0.95 - 0.85) * (lr / learning_rate), param_group["betas"][1])
                    param_group["betas"] = (0.95 - (0.95 - 0.85) * (lr / learning_rate), 0.975 - (0.975 - 0.925) * (lr / learning_rate))
                    # max_eps = base_hps.eta0 / 4e-4 * base_hps.eps
                    # param_group["eps"] = 1e-16 + (max_eps - 1e-16) * (lr / learning_rate)
                    # todo: fix for batch rampup
                    # param_group["weight_decay"] = weight_decay * base_hps.eta0 / param_group["lr"]
                    # param_group["weight_decay"] = weight_decay * lr / learning_rate
                    # param_group["eps"] = eps / (lr / learning_rate )
                    # param_group["betas"][0] = 1- (1 - beta1) * (lr / learning_rate)
                    # param_group["betas"][1] = 1- (1 - beta2) * (lr / learning_rate)
                    # fabric.print(f"weight_decay: {param_group['weight_decay']}")
        
        iter_t0 = time.perf_counter()
        hybrid_optimizer = type(optimizer) == list or type(optimizer) == tuple
        lr = get_lr(state["iter_num"], warmup_iters, max_iters, learning_rate) if decay_lr else learning_rate
        if hybrid_optimizer:
            for i, opt in enumerate(optimizer):
                if i == 0: #vectors
                    mult = 1.0
                else: #weights
                    mult = weight_lr_scale
                for param_group in opt.param_groups:
                    lr_mult = mult * param_group["lr_mult"] if "lr_mult" in param_group else mult
                    update_param(param_group, lr_mult)
        else:
            for param_group in optimizer.param_groups:
                lr_mult = param_group["lr_mult"] if "lr_mult" in param_group else 1.0
                update_param(param_group, lr_mult)
        
        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        budget_loss_value = None
        lc_mean_activation = None
        bal_loss = None
        acc_seq = None
        layout_bias = layout_bias_decay_func(
            state["iter_num"], 
            max_iters, 
            max_layout_bias = 1.0,
            ratio = 0.05 
        )
        is_moe = "_moe" in model_name
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            if use_flce_loss:
                output = model(input_ids, residual_dropout=code_dropout if is_code else 0.0, use_flce_loss=True)
                loss = loss_func(output.logits, output.weight, targets)
                # no accuracy metrics for large vocab
            else:
                logits = model(input_ids, residual_dropout=code_dropout if is_code else 0.0).logits
                loss = loss_func(logits, targets)
                # mean accuracy over full sequence (small vocab only)
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc_seq = (preds == targets).float().mean()
            if model.aux_loss != 0:
                back_loss = loss + model.aux_loss
                bal_loss = model.aux_loss.detach()
            else:
                back_loss = loss
            # Auxiliary LC activation budget loss
            # Use unified LC gate history on model (works for shared and per-layer)
            if len(model.lc_gate_hist) > 0:
                gates = model.lc_gate_hist  # list of (B, T, 1)
                probs = torch.stack([g.squeeze(-1) for g in gates], dim=1)  # (B, L, T)
                aux_loss = activation_budget_bce(
                    probs,
                    tau=model.config.lc_budget_tau,
                    mask=None,
                    lambda_budget=model.config.lc_budget_lambda,
                )
                back_loss = loss + aux_loss
                budget_loss_value = aux_loss.detach()
                lc_mean_activation = probs.mean().detach()
            if use_model_parallel:
                model.set_is_last_backward(not is_accumulating or is_moe)
                model.set_reshard_after_backward(not is_accumulating or is_moe)
                model.set_requires_gradient_sync(not is_accumulating or is_moe)
            fabric.backward(back_loss / current_grad_accum_steps)
        
        state["iter_num"] += 1
        if not is_accumulating:
            state["step_count"] += 1
        if state["iter_num"] % log_iter_interval == 0 and not is_accumulating:
            payload = build_logging_payload(
                model=model,
                state=state,
                budget_loss_value=budget_loss_value,
                lc_mean_activation=lc_mean_activation,
                acc_seq=acc_seq,
                bal_loss=bal_loss,
                optimizer=optimizer,
                current_grad_accum_steps=current_grad_accum_steps,
                batch_multiplier=batch_multiplier,
                batch_rampup=batch_rampup,
                current_batch_ratio=current_batch_ratio,
                is_accumulating=is_accumulating,
                hybrid_optimizer=hybrid_optimizer,
                micro_batch_size=micro_batch_size,
                devices=devices,
                nodes=nodes,
                seq_len=seq_len,
                grad_clip=grad_clip,
                vector_names=vector_names,
                eval_step_interval=eval_step_interval,
            )
            submit_logging_task(payload)
        if not is_accumulating:
            log_weight_update_rms = False # currently slow, so disabled
            if log_weight_update_rms:
                # Save old weights before optimizer.step() (non-blocking CPU transfer)
                saved_weights = save_weights_for_update_logging(model, vector_names)
            
            if hybrid_optimizer:
                for opt in optimizer:
                    if grad_clip > 0:
                        fabric.clip_gradients(model, opt, max_norm=grad_clip)
                    opt.step()
                    opt.zero_grad()
            else:
                if grad_clip > 0:
                    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            if log_weight_update_rms:
                # Capture new weights after step (non-blocking CPU transfer) for async RMS computation
                model._weight_update_data = capture_weight_update_data(
                    model, saved_weights, optimizer, hybrid_optimizer
                )

        # input_id: B L 
        total_lengths += input_ids.size(1) * input_ids.size(0) // micro_batch_size
        t1 = time.perf_counter()

        if state["iter_num"] % log_iter_interval == 0 and not is_accumulating:
            # Build print message with optional batch size info
            print_msg = (
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" input_length: {input_ids.size(1)} total_length: {total_lengths} "
            )
            if batch_rampup:
                effective_batch_tokens = micro_batch_size * current_grad_accum_steps * devices * nodes * seq_len
                print_msg += f" batch_tokens: {effective_batch_tokens} (grad_accum: {current_grad_accum_steps}/{final_grad_accum_steps}) "
            print_msg += (
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
            fabric.print(print_msg)
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item(),
            model = model,
        )

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            validate(fabric, model, val_dataloader, state, monitor, layout_bias)
        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir + f"/iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            utils._fabric_save_anywhere(fabric, state, checkpoint_path)
            # fabric.save(checkpoint_path, state)
    wait_for_logging_tasks(fabric)

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, 
    state: dict, monitor: Monitor, layout_bias: float = 0.0, sanity_check=False) -> torch.Tensor:
    t0 = time.perf_counter()
    fabric.print("Validating ...")
    model.eval()
    global num_extrapol
    losses = torch.zeros(eval_iters, num_extrapol, device=fabric.device)
    if use_flce_loss:
        loss_func = FusedLinearCrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        loss_func = FusedCrossEntropyLoss(label_smoothing=label_smoothing)
        accs = torch.zeros(eval_iters, num_extrapol, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if isinstance(val_data, (list, tuple)) and len(val_data) == 2:
            val_data, is_code = val_data
            is_code = is_code[0]
        else:
            is_code = False
        if k >= eval_iters:
            break
        
        extrapol_list = [(i + 1) * seq_len  for i in range(num_extrapol)]   
        for i, length in enumerate(extrapol_list):   #[2048, 4096, 8192, 16384]
            input_ids = val_data[:, 0 : length].contiguous()
            targets = val_data[:, 1 : length + 1].contiguous()
            if use_flce_loss:
                output = model(input_ids, use_flce_loss=True)
                loss = loss_func(output.logits, output.weight, targets)
                # no accuracy metrics for large vocab
            else:
                logits = model(input_ids).logits
                loss = loss_func(logits, targets)
                # mean accuracy over full sequence (small vocab only)
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == targets).float().mean()
                    accs[k,i] = acc.item()
            losses[k,i] = loss.item()
    out = losses.mean(0)
    if "v2scale" in train_config:
        # Synchronize results across all GPUs
        out = fabric.all_reduce(out, reduce_op="mean")
    if "accs" in locals():
        out_acc = accs.mean(0)
        if "v2scale" in train_config:
            out_acc = fabric.all_reduce(out_acc, reduce_op="mean")
    model.train()
    t1 = time.perf_counter() - t0
    monitor.eval_end(t1)
    if sanity_check:
        return out
    for i in range(num_extrapol):
        fabric.print(f"step {state['iter_num']}: val loss@{str(i+1)}x {out[i]:.4f},  val time: {t1 * 1000:.2f}ms")
        log_dict = {
            "metric/val_loss@"+str(i+1)+"x": out[i].item(),
            "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size
        }
        if "accs" in locals():
            fabric.print(f"step {state['iter_num']}: val acc_seq@{str(i+1)}x {out_acc[i]:.4f}")
            log_dict["metric/val_acc_seq@"+str(i+1)+"x"] = out_acc[i].item()
        fabric.log_dict(log_dict, state["step_count"])
        fabric.log_dict({"metric/val_ppl@"+str(i+1)+"x": math.exp(out[i].item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
    fabric.barrier()
    return out


def list_folders_with_bin_files(root_dir: str):
    """Return folders (local or cloud) that contain at least one .bin file."""
    # For remote URIs (az://, gs://, s3://) use bf.walk; for local, bf.walk also works.
    folders_with_bin = []
    for dirpath, dirnames, filenames in bf.walk(root_dir):
        if any(fname.endswith(".bin") for fname in filenames):
            folders_with_bin.append(dirpath)
    return folders_with_bin


def create_dataloader(
    batch_size: int, block_size: int, data_dir: str, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    # slow, only for debugging
    # folders = list_folders_with_bin_files(data_dir)
    # print(folders)


    data_config = train_data_config if split == "train" else val_data_config

    for prefix, _ in data_config:
        # blobfile-aware glob (works for local and az://, gs://, s3://)
        pattern = os.path.join(data_dir, f"{prefix}*.bin")
        filenames = sorted(bf.glob(pattern))
        random.seed(seed)
        random.shuffle(filenames)
        # n_chunks control the prefetch buffer size. increase n_chunks for better sample-level randomness
        if split != "train":
            n_chunks = math.ceil(8 / nodes)
        else:
            n_chunks = data_chunks

        dataset = PackedDataset(
            filenames=filenames,
            n_chunks=n_chunks,     # prefetch buffer
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            drop_last=False if n_chunks > 8 else True,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No data found at {data_dir}. Did you create the dataset?")

    weights = [w for _, w in data_config]
    s = sum(weights)
    weights = [w / s for w in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir = "data/redpajama_sample",   # can be az://..., gs://..., s3://..., or local
    val_data_dir: Optional[str] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    # if len_rampup or len_sam:
    #     effective_block_size = block_size* batch_size + 1
    #     train_batch_size = 1
    # else:
    effective_block_size = block_size + 1
    #     train_batch_size = batch_size
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size= - (batch_size // -2), # ceil division
            block_size= num_extrapol * block_size + 1, # val 4* extrapolation
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader

def apply_batch_size_scaling(batch_ratio: float, base_hps: BaseHyperparameters, scaling_mode: str = "sdew"):
    """
    Apply batch size scaling rules to optimizer hyperparameters.
    
    Args:
        batch_ratio: Current batch size / base batch size (b / b0)
        base_hps: Base hyperparameters object
        scaling_mode: "sde" (SDE scaling for Adam), "sdew" (SDE scaling for AdamW), or None
        
    Returns:
        dict with scaled optimizer parameters:
        - learning_rate: scaled learning rate
        - eps: scaled epsilon
        - beta1: scaled beta1
        - beta2: scaled beta2
        - weight_decay: scaled weight decay
    """
    base_lr = base_hps.eta0
    
    if scaling_mode == "sde":
        # SDE scaling rules for Adam: https://arxiv.org/abs/2205.10287
        # Scale the current LR by sqrt(batch_ratio)
        learning_rate = base_lr * math.sqrt(batch_ratio)
        eps = base_hps.eps / math.sqrt(batch_ratio)
        # constant beta1 in https://arxiv.org/abs/2507.07101
        beta1 = base_hps.beta1
        beta2 = 1 - (1 - base_hps.beta2) * batch_ratio
        weight_decay = base_hps.weight_decay
    elif scaling_mode == "sdew":
        # SDE scaling rules for AdamW: https://arxiv.org/abs/2411.15958
        # constant LR, empirically better than sqrt(batch_ratio) 
        # also in https://arxiv.org/abs/2507.07101
        learning_rate = base_lr  # Keep current LR as-is
        eps = base_hps.eps / math.sqrt(batch_ratio)
        beta1 = base_hps.beta1
        beta2 = 1 - (1 - base_hps.beta2) * math.sqrt(batch_ratio)
        weight_decay = base_hps.weight_decay * math.sqrt(batch_ratio)
    else:
        # Default scaling: scale LR by sqrt(batch_ratio)
        learning_rate = base_lr * math.sqrt(batch_ratio) if scaling_mode == "default" else base_lr
        eps = base_hps.eps
        beta1 = base_hps.beta1
        beta2 = base_hps.beta2
        weight_decay = base_hps.weight_decay
    
    return {
        'learning_rate': learning_rate,
        'eps': eps,
        'beta1': beta1,
        'beta2': beta2,
        'weight_decay': weight_decay
    }

def rampup_func(k, step_width = 16, max_len = 4096, warmup_step = 10000 ):
    # k: global step
    assert step_width * warmup_step >= max_len
    rampup_step = k // (warmup_step// (max_len // step_width) ) # (2*2k)/(64k/2k)
    if rampup_step < max_len//step_width:
        x = step_width  * (rampup_step + 1)
    else:
        x = max_len
    return x

def layout_bias_decay_func(iters: int, max_iters: int, max_layout_bias: float = 1.0, ratio: float = 0.05):
    """
    Calculate the layout bias decay multiplier with linear decay to 0.
    
    """
    decay_iters = max(int(ratio * max_iters), 1)
    
    if iters >= decay_iters:
        return 0.0  # After decay period, return 0
    
    # Linear decay from 1.0 to 0.0
    progress = iters / decay_iters
    decay_multiplier = max_layout_bias - progress * max_layout_bias
    
    return decay_multiplier

def batch_size_rampup_func(iters: int,  max_iters: int,  batch_size_ratio: float = 4.0):
    """
    Calculate the current batch size multiplier for batch size rampup.
    
    """
    rampup_iters = max(int(0.05 * max_iters), 1)  # Default: 5% of total tokens, deeepseek v3 schedule
    
    if iters >= rampup_iters:
        return batch_size_ratio  # Maximum batch size
    
    # Linear rampup from 1.0 to batch_size_ratio
    progress = iters / rampup_iters
    batch_multiplier = 1.0 + (batch_size_ratio - 1.0) * progress
    
    return batch_multiplier


# learning rate scheduler with warmup, stable period, and decay
def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr * it / warmup_iters
    
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr * lr
    
    if "wsd" in train_config:
        # 3) stable period for 5/7 of training after warmup (deepseekv3 schedule) 
        # empirically, 5/7 is better than 9/10
        stable_iters = int(5/7 * (max_iters - warmup_iters))
        if it < warmup_iters + stable_iters:
            return lr
            
        # 4) decay period for remaining iterations
        decay_iters = max_iters - warmup_iters - stable_iters
        decay_ratio = (it - warmup_iters - stable_iters) / decay_iters
        assert 0 <= decay_ratio <= 1
    else:
        # 3) in between, use linear or cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    if "cosine" in train_config:
        # Cosine decay
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr * lr + coeff * (lr - min_lr * lr)
    else:
        # Linear decay
        return lr + decay_ratio * (min_lr * lr - lr)



def find_resume_path(resume: Union[bool, Literal["auto"], str], out_dir: str) -> Optional[str]:
    if not resume or (isinstance(resume, str) and resume != "auto"):
        return resume
    resume_path = utils._latest_ckpt_in_dir_anywhere(out_dir)
    if resume == "auto":
        return resume_path
    if resume is True and resume_path is None:
        raise FileNotFoundError(
            f"You passed `--resume=True`, but no checkpoint file was found in `--out_dir={out_dir}`."
        )
    return resume_path


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)