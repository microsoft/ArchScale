# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import AbstractContextManager, ExitStack
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import get_args, override

from lightning.fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning.fabric.plugins.precision.utils import (
    _ClassReplacementContextManager,
    _convert_fp_tensor,
    _DtypeContextManager,
)
from lightning_utilities.core.imports import RequirementCache
from lightning.fabric.utilities.types import Optimizable
from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.fabric.plugins.precision import FSDPPrecision
if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from lit_gpt.moe_utils import set_token_group_alignment_size_m

import logging
import math
import torch.nn as nn
from functools import partial
log = logging.getLogger(__name__)

_PRECISION_INPUT = Literal["bf16-mixed"]


class FP8FSDPPrecision(FSDPPrecision):
    """FP8 Precision plugin for training with Fully Sharded Data Parallel (FSDP).

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        precision: Full precision (32-true), half precision (16-true, bf16-true) or
            mixed precision (16-mixed, bf16-mixed).
        scaler: An optional :class:`torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler` to use.

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(self, precision: _PRECISION_INPUT, fp4: bool = False, scaler: Optional["ShardedGradScaler"] = None) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        from transformer_engine.common.recipe import Format, DelayedScaling, Float8BlockScaling, MXFP8BlockScaling, NVFP4BlockScaling, Float8CurrentScaling
        _TRANSFORMER_ENGINE_AVAILABLE = RequirementCache("transformer_engine>=0.11.0")
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in FSDP."
                f" `precision` must be one of: {supported_precision}."
            )
        if not _TRANSFORMER_ENGINE_AVAILABLE:
            raise ModuleNotFoundError(str(_TRANSFORMER_ENGINE_AVAILABLE))
        self.scaler = None
        self.precision = precision
        self._desired_input_dtype = torch.float32 
    
        # Check GPU compute capability to determine FP8 recipe
        # H100: SM 90, GB200: SM 100+
        compute_capability = torch.cuda.get_device_capability()
        sm_version = compute_capability[0] * 10 + compute_capability[1]
        rank_zero_info(f"Detected GPU compute capability: SM {compute_capability[0]}.{compute_capability[1]} (SM {sm_version})")
        
        #fp8_recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)
        # Use MXFP8BlockScaling for H100 (SM 90) or GB200 (SM 100+)
        self.fp4 = fp4
        if sm_version >= 100:
            rank_zero_info("Using MXFP8BlockScaling for GB200")
            if fp4: 
                fp8_recipe = NVFP4BlockScaling()
            else:
                fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
                set_token_group_alignment_size_m(32)
        else:
            rank_zero_warn("MXFP8BlockScaling not available, falling back to Float8BlockScaling (requires cublas 12.9+)")
            fp8_recipe = Float8BlockScaling(fp8_format=Format.E4M3)
            set_token_group_alignment_size_m(16)

        self.recipe = fp8_recipe
        self.replace_layers = True

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.replace_layers in (None, True):
            # Find the last transformer layer to exclude from FP8 conversion
            _convert_layers(module, exclude_layers=False) #self.fp4)
        if "true" in self.precision:
            return module.to(dtype=self._desired_input_dtype)
        return module

    @property
    def mixed_precision_config(self) -> "TorchMixedPrecision":
        # not used by fsdp2
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision

        # using float32 for param can give much better train and val loss near the end of training,
        # comparing to bfloat16 for param
        param_dtype = torch.float32  
        reduce_dtype = buffer_dtype = torch.bfloat16  
        return TorchMixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def module_init_context(self) -> AbstractContextManager:
        dtype_ctx = _DtypeContextManager(torch.float32)
        stack = ExitStack()
        stack.enter_context(dtype_ctx)
        return stack
    

    @override
    def forward_context(self) -> AbstractContextManager:
        dtype_ctx = self.tensor_init_context()
        #if "mixed" in self.precision:
        #fallback_autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import transformer_engine.pytorch as te

        autocast_ctx = te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        stack = ExitStack()
        stack.enter_context(dtype_ctx)
        # enable an outer fallback autocast for operations that do not support fp8
        #if "mixed" in self.precision:
        #stack.enter_context(fallback_autocast_ctx)
        stack.enter_context(autocast_ctx)
        return stack

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())

    @override
    def backward(self, tensor: Tensor, model: Optional[Module], *args: Any, **kwargs: Any) -> None:
        super().backward(tensor, model, *args, **kwargs)

    @override
    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        return super().optimizer_step(optimizer, **kwargs)

    @override
    def unscale_gradients(self, optimizer: Optimizer) -> None:
        pass

    @override
    def state_dict(self) -> dict[str, Any]:
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

def _convert_layers(module: torch.nn.Module, exclude_layers: bool = False, 
                     last_layer_idx: int = -1, current_path: str = "", parent_idx: int = -1) -> None:
    import transformer_engine.pytorch as te

    for name, child in module.named_children():
        # Track if this child is a layer in the main transformer stack
        child_idx = parent_idx
        child_path = f"{current_path}.{name}" if current_path else name
        
        # Check if this is a layer/block in the main stack
        if name in ['layers', 'blocks', 'h', 'transformer_blocks']:
            # Don't convert this module itself, recurse into its children with indices
            if isinstance(child, torch.nn.ModuleList):
                layer_list = list(child.children())  
                last_layer_idx = len(layer_list) - 1
                print(f"Found {len(layer_list)} transformer layers in '{name}', last index: {last_layer_idx}")
            else:
                layer_list = []
            for idx, layer in enumerate(layer_list):
                _convert_layers(layer, exclude_layers, last_layer_idx, f"{child_path}[{idx}]", parent_idx=idx)
            continue
        
        # If this is a numeric index from a ModuleList, update child_idx
        try:
            if name.isdigit():
                child_idx = int(name)
        except:
            pass
        
        #print(name, child, type(child))
        #replace_list = ["w1", "w3", "lm_head"] # only replace mlp and lm_head layers
        no_replace_list = ["lm_head", "lc_proj"]
        no_replace_layers= [0, last_layer_idx]
        if isinstance(child, torch.nn.Linear) and (not name in no_replace_list):
            # Skip conversion if this is in the last layer and exclude_layers is True
            if exclude_layers and child_idx in no_replace_layers:
                print(f"Skipping FP8 conversion for layer {child_path!r} (last layer)")
                _convert_layers(child, exclude_layers, last_layer_idx, child_path, child_idx)
                continue
                
            if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#FP8-autocasting
                print(
                    "Support for FP8 in the linear layers with this plugin is currently limited to"
                    " tensors with shapes where the dimensions are divisible by 16 and 16 respectively."
                    f" The layer {name!r} does not fit this criteria. You might want to add padding to your inputs."
                )
                continue
            has_bias = child.bias is not None
            replacement = te.Linear(child.in_features, child.out_features, bias=has_bias)
                                #init_method=partial(nn.init.kaiming_uniform_, a=math.sqrt(5)))
            replacement.weight.data = child.weight.data.clone()
            if has_bias:
                replacement.bias.data = child.bias.data.clone()
            print(f"Replacing layer {child_path!r} with Transformer Engine equivalent")
            module.__setattr__(name, replacement)
        else:
            # there are other transformer engine layers that we could convert but require fusion. full list at:
            # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html
            _convert_layers(child, exclude_layers, last_layer_idx, child_path, child_idx)
