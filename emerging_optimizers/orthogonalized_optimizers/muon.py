# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import triton_kernels
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import msign
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc


class Muon(OrthogonalizedOptimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz

    Muon runs standard SGD-momentum with Nesterov momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, Newton-Schulz iteration is used, which has the
    advantage that it may be stably run on tensor cores on GPUs.

    Orthogonalization can be viewed as steepest descent in the spectral norm. The theoretical foundation
    is based on modular duality and norm-constrained optimization.

    This implementation incorporates decoupled weight decay, refer to Scion which views weight decay as constrained
    optimization via Frank-Wolfe.

    References:
        - Jordan, K. *Muon Optimizer Implementation.*
          [`GitHub <https://github.com/KellerJordan/Muon/blob/master/muon.py>`_]
        - *Modular Duality in Deep Learning.* arXiv:2410.21265 (2024).
          [`arXiv:2410.21265 <https://arxiv.org/abs/2410.21265>`_]
        - *Training Deep Learning Models with Norm-Constrained LMOs.* arXiv:2502.07529 (2025).
          [`arXiv:2502.07529 <https://arxiv.org/abs/2502.07529>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer, or any 1-D
          parameters; those should all be optimized by a standard method (e.g., AdamW).

    Args:
        {_args_doc}
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express"].
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update. Defaults to "align_adamw_rms" style scaling.
        extra_scale_factor: The additional scale factor to use for the update.
        use_syrk: Whether to use the Triton kernel for the Newton-Schulz iteration.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        weight_decay: float = 0.01,
        *,
        use_nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "align_adamw_rms",
        extra_scale_factor: float = 1.0,
        use_syrk: bool = False,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        if use_syrk:
            if torch.cuda.is_available():
                sm_version = torch.cuda.get_device_capability()
            else:
                sm_version = (0, 0)
            if not triton_kernels.HAS_TRITON_340:  # type: ignore[attr-defined]
                logging.error("Triton 3.4.0 or higher is required for use_syrk to be True.")
                use_syrk = False
            elif sm_version not in ((8, 0), (9, 0), (10, 0), (10, 3)):
                logging.error(
                    f"Correctness of Triton kernel on SM {sm_version} cannot be guaranteed. Setting use_syrk to False."
                )
                use_syrk = False

        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            logging.debug(
                f"Orthogonalizing grad with {num_ns_steps} steps, "
                f"{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}"
            )
            # Use msign from spectral_ball_utils for more accurate Newton-Schulz iteration
            orth_grad = msign(grad, steps=num_ns_steps)
            scale_factor = get_muon_scale_factor(grad.size(-2), grad.size(-1), mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )


Muon.__doc__ = Muon.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]


def get_muon_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Get the scale for the update.

    Default mode is "align_adamw_rms", which is the mode that allows for learning rate transferability from AdamW.
    An extra scale factor is used to match the update RMS norm of AdamW, so that we can transfer hyperparameters
    from AdamW to Muon. An extra scale factor of sqrt((1-β₁)/(1+β₁)), where β₁ is AdamW's momentum EMA coefficient,
    analytically gives the update RMS norm of AdamW (https://kexue.fm/archives/11267).

    Args:
        size_out: The size of the output tensor.
        size_in: The size of the input tensor.
        mode: The mode to use for the scale.
    Returns:
        The scale factor for the update.
    """
    if mode == "shape_scaling":
        # Suggested by Muon (https://kellerjordan.github.io/posts/muon/)
        return max(1, size_out / size_in) ** 0.5
    elif mode == "align_adamw_rms":
        # Suggested by K. Jordan and Kimi (https://arxiv.org/abs/2502.16982)
        return 0.2*max(size_out, size_in) ** 0.5
    elif mode == "spectral_mup":
        # Suggested by Scion (https://arxiv.org/abs/2502.07529) and Bernstein et al.
        # (https://jeremybernste.in/writing/deriving-muon)
        return (size_out / size_in) ** 0.5
    else:
        raise ValueError(f"Invalid mode for Muon update scale factor: {mode}")
