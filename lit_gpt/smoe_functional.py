# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# modified from https://github.com/Dao-AILab/sonic-moe/blob/main/sonicmoe/functional/__init__.py

# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
import torch.nn.functional as F
from sonicmoe.enums import ActivationType
from sonicmoe.functional import _UpProjection, _DownProjection, TC_Softmax_Topk_Router_Function
from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton

def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    sqrt_gate: bool = False,
    logit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    E = router_w.size(0)
    router_logits = F.linear(x, router_w) * logit_scale
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, E, K)

    if sqrt_gate:
        topk_scores = torch.sqrt(topk_scores + 1e-30)

    (expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx) = (
        TC_topk_router_metadata_triton(topk_indices, E)
    )

    T = x.size(0)

    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        T * K,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_varlen_K
        activation_type,
        is_inference_mode_enabled,
    )

    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        topk_scores,
        expert_frequency_offset,
        T,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_varlen_K
        activation_type,
    )

    return o, router_logits, expert_frequency, topk_indices