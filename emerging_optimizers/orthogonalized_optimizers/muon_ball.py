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

"""MuonBall Optimizer: Spectral Ball with λ=0 (simplified version).

MuonBall combines:
- Spectral Ball's hard retraction to spectral sphere
- Muon's simple msign orthogonalization (without λ solver)

This is equivalent to Spectral Ball with λ fixed to 0, which skips the
expensive bisection solver and uses the simpler Muon update direction.

Algorithm:
1. Power iteration: σ, u, v = power_iteration(W, steps)
2. Retract W: W ← (R/σ) * W
3. Orthogonalize: Φ = msign(M)  [no λ, just like Muon]
4. Update: W ← W - lr * scale * Φ

Key difference from Spectral Ball:
- Spectral Ball solves: <Θ, msign(M + λΘ)> = 0 to find λ
- MuonBall simply uses: Φ = msign(M) with λ = 0

This makes MuonBall significantly faster than Spectral Ball while still
maintaining spectral norm constraints through retraction.
"""

from typing import Any, Callable, Optional, Tuple

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import (
    OrthogonalizedOptimizer,
    to_local,
    _args_doc,
)
from .spectral_ball_utils import (
    compute_target_radius,
    get_spectral_ball_scale_factor,
    power_iteration,
    apply_retract,
    msign,
)


def compute_muon_ball_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    *,
    tp_group: torch.distributed.ProcessGroup | None = None,
    partition_dim: int | None = None,
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Compute MuonBall update (Spectral Ball with λ=0).

    Algorithm:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Orthogonalize: Φ = msign(M)  [simplified, no λ]

    Args:
        W: Current weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations
        tp_group: Tensor parallel process group (None for single-rank)
        partition_dim: Dimension along which tensors are partitioned
        retract_mode: Retraction mode ('hard' or 'dynamic')
        retract_alpha: Alpha parameter for dynamic retraction
        current_lr: Current learning rate (for dynamic retraction)

    Returns:
        Tuple of (Phi, retract_bias, sigma_value) where:
        - Phi: Update direction to be applied as W ← W - lr * Φ
        - retract_bias: Retraction bias (0.0 for hard mode)
        - sigma_value: Spectral norm before retraction
    """
    # Handle TP if enabled
    from .spectral_ball_utils import _tp_world_and_rank, _tp_gather_along_dim, _tp_split_along_dim

    W = to_local(W)
    M = to_local(M)
    ws, _ = _tp_world_and_rank(tp_group)
    tp_enabled = tp_group is not None and partition_dim is not None and ws > 1

    if tp_enabled:
        # Gather full tensors for global operations
        W_full = _tp_gather_along_dim(W, tp_group, partition_dim)
        M_full = _tp_gather_along_dim(M, tp_group, partition_dim)
        W_work = W_full
        M_work = M_full
    else:
        W_work = W
        M_work = M

    # Convert M to fp32 and normalize
    M_fp32 = M_work.to(torch.float32)
    # no need to normalize M for MuonBall, msign already does it
    # M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2, -1), keepdim=True).clamp_min(1e-8))

    # 1. Power iteration (returns fp32)
    sigma, u, v = power_iteration(W_work, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # 2. Retract W to spectral sphere
    retract_bias = apply_retract(
        W_work, sigma_value, target_radius,
        mode=retract_mode, alpha=retract_alpha, current_lr=current_lr
    )

    # 3. MuonBall: Simply msign(M) without λ solver
    # This is the key simplification: λ = 0
    Phi = msign(M_fp32, steps=msign_steps)

    # Handle TP: split result and update local W
    if tp_enabled:
        W_local = _tp_split_along_dim(W_work, tp_group, partition_dim)
        W.copy_(W_local)
        Phi_local = _tp_split_along_dim(Phi, tp_group, partition_dim)
        return Phi_local, retract_bias, sigma_value
    else:
        return Phi, retract_bias, sigma_value


class MuonBall(OrthogonalizedOptimizer):
    """MuonBall Optimizer: Spectral Ball with λ=0.

    MuonBall simplifies Spectral Ball by fixing λ=0, which eliminates the need for
    the expensive bisection solver while still maintaining spectral norm constraints.

    The algorithm:
    1. Power iteration to compute spectral norm σ and top singular vectors (u, v)
    2. Retraction to spectral sphere: W ← (R/σ) * W
    3. Orthogonalize momentum: Φ = msign(M)  [no λ, unlike Spectral Ball]
    4. Update: W ← W - lr * Φ

    Comparison to Spectral Ball:
    - Spectral Ball: Solves for λ such that <Θ, msign(M + λΘ)> = 0
    - MuonBall: Uses λ = 0, so Φ = msign(M) directly

    Comparison to Muon:
    - Muon: No spectral constraint, weights can drift
    - MuonBall: Hard constraint via retraction, weights stay on spectral sphere

    This optimizer is useful for:
    - Faster training than Spectral Ball (no λ solver overhead)
    - More constrained than Muon (spectral norm control)
    - Validating whether λ solver is necessary in practice

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer,
          or any 1-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Args:
        {_args_doc}
        power_iteration_steps: Number of power iteration steps for spectral norm computation.
        msign_steps: Number of Newton-Schulz iterations for msign (uses Polar-Express).
        radius_mode: Target radius mode ("spectral_mup", "identity", "initialize").
        scale_mode: Scale factor mode for updates ("align_adamw_rms", "shape_scaling", "spectral_mup").
        retract_mode: Retraction mode ("hard" or "dynamic").
        retract_alpha: Alpha parameter for dynamic retraction.
        split_qkv: Whether to split QKV parameters and process Q/K/V independently.
        is_qkv_fn: Function to identify QKV parameters.
        qkv_split_shapes: Tuple of (q_dim, k_dim, v_dim) per query group.
        qkv_split_mode: QKV split mode ("component", "group", or "head").
        split_fc1: Whether to split FC1 (gate and up) for gated linear units.
        is_fc1_fn: Function to identify FC1 parameters.
        fc1_split_shapes: Tuple of (gate_dim, up_dim).
        split_moe_experts: Whether to split GroupedMLP experts and process independently.
        is_grouped_moe_fn: Function to identify GroupedMLP parameters (weight1/weight2).
        pg_collection: ProcessGroupCollection for tensor parallel support.
        tp_mode: Tensor parallel mode ("duplicated", "blockwise", or "distributed").
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.9,
        weight_decay: float = 0.01,
        *,
        use_nesterov: bool = True,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: str = "medium",
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        radius_mode: str = "spectral_mup",
        scale_mode: str = "align_adamw_rms",
        retract_mode: str = "hard",
        retract_alpha: float = 0.05,
        # QKV / TP support (optional)
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[Tuple[int, int, int]] = None,
        qkv_split_mode: str = "component",  # "component", "group", or "head"
        # FC1 split support for gated linear units (SwiGLU)
        split_fc1: bool = False,
        is_fc1_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        fc1_split_shapes: Optional[Tuple[int, int]] = None,  # (gate_dim, up_dim)
        # MoE expert split support for GroupedMLP
        split_moe_experts: bool = False,
        is_grouped_moe_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        pg_collection: Any | None = None,
        tp_mode: str = "duplicated",
    ) -> None:
        if power_iteration_steps < 1:
            raise ValueError(f"power_iteration_steps must be at least 1, got {power_iteration_steps}")
        if msign_steps < 1:
            raise ValueError(f"msign_steps must be at least 1, got {msign_steps}")
        if radius_mode not in ("spectral_mup", "identity", "initialize"):
            raise ValueError(f"Invalid radius_mode: {radius_mode}, must be one of: spectral_mup, identity, initialize")
        if retract_mode not in ("hard", "dynamic"):
            raise ValueError(f"Invalid retract_mode: {retract_mode}, must be one of: hard, dynamic")
        if qkv_split_mode not in ("component", "group", "head"):
            raise ValueError(f"Invalid qkv_split_mode: {qkv_split_mode}, must be one of: component, group, head")

        # Store MuonBall specific parameters
        self.power_iteration_steps = power_iteration_steps
        self.msign_steps = msign_steps
        self.radius_mode = radius_mode
        self.scale_mode = scale_mode
        self.retract_mode = retract_mode
        self.retract_alpha = retract_alpha
        self.retract_bias_dict = {}  # For logging retract bias (only in dynamic mode)
        self.spectral_norm_dict = {}  # For logging spectral norms
        # QKV / TP
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_split_mode = qkv_split_mode
        # FC1 split for gated linear units
        self.split_fc1 = split_fc1
        self.is_fc1_fn = is_fc1_fn
        self.fc1_split_shapes = fc1_split_shapes
        # MoE expert split for GroupedMLP
        self.split_moe_experts = split_moe_experts
        self.is_grouped_moe_fn = is_grouped_moe_fn
        self.pg_collection = pg_collection
        self.tp_mode = tp_mode

        # Placeholder for scaled_orthogonalize_fn
        # MuonBall uses custom orthogonalize() method instead
        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError(
                "MuonBall uses custom orthogonalize() method. "
                "scaled_orthogonalize_fn should not be called directly."
            )

        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
            log_per_module_update_rms=False,  # Will be set later via config
        )

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step.

        Override parent's step to clear retract_bias_dict at the beginning.
        This ensures the dict is cleared once per step, not once per parameter.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        # Clear retract bias dict at the start of each step
        self.retract_bias_dict.clear()
        self.spectral_norm_dict.clear()

        # Call parent's step method
        return super().step(closure)

    def _compute_component_update(
        self,
        W: torch.Tensor,
        M: torch.Tensor,
        tp_group: Any,
        partition_dim: Optional[int],
        current_lr: Optional[float] = None,
        param_name: Optional[str] = None,
        component_label: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute MuonBall update for a single Q/K/V component.

        Args:
            W: Weight tensor for this component
            M: Momentum tensor for this component
            tp_group: Tensor parallel group
            partition_dim: Partition dimension for TP
            current_lr: Current learning rate (for dynamic retraction)
            param_name: Parameter name for logging
            component_label: Label like 'q', 'k', 'v' or 'g0.q' for logging

        Returns:
            Update direction tensor
        """
        R = compute_target_radius(shape=W.shape, radius_mode=self.radius_mode)

        u, bias, sigma = compute_muon_ball_update(
            W=W,
            M=M,
            target_radius=R,
            power_iteration_steps=self.power_iteration_steps,
            msign_steps=self.msign_steps,
            tp_group=tp_group,
            partition_dim=partition_dim,
            retract_mode=self.retract_mode,
            retract_alpha=self.retract_alpha,
            current_lr=current_lr,
        )

        # Record bias for logging
        if self.retract_mode == 'dynamic' and bias != 0.0 and param_name and component_label:
            self.retract_bias_dict[f"{param_name}.{component_label}"] = bias
            self.spectral_norm_dict[f"{param_name}.{component_label}"] = sigma

        # Apply scale factor
        scale_factor = get_spectral_ball_scale_factor(W.shape[0], W.shape[1], mode=self.scale_mode)
        return u * scale_factor

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute MuonBall update direction.

        This method overrides the base class orthogonalize() to implement the MuonBall
        algorithm. The input 'grad' is actually the momentum M (potentially with
        Nesterov momentum applied by the base class).

        The core algorithm:
        1. Power iteration: σ, u, v = power_iteration(W, steps)
        2. Retract W: W ← (R/σ) * W  [in-place modification]
        3. Orthogonalize: Φ = msign(M)  [λ = 0, no solver needed]
        4. Return: Φ

        Args:
            p: Parameter tensor (current weight matrix W)
            grad: Momentum tensor M (after Nesterov if applicable)
            **kwargs: Additional parameters from param_group (includes 'lr')

        Returns:
            Update direction Φ to be applied as: W ← W - lr * Φ
        """
        # Extract current learning rate from kwargs (passed from param_group)
        current_lr = kwargs.get('lr', None)

        # Compute target radius
        target_radius = compute_target_radius(
            shape=p.shape,
            radius_mode=self.radius_mode,
        )

        # Resolve TP group and partition dim if available
        tp_group = None
        partition_dim = 0 # assuming FSDP2
        if self.pg_collection is not None:
            try:
                tp_group = (
                    self.pg_collection.expt_tp if getattr(p, "expert_tp", False) else self.pg_collection.tp
                )
            except Exception:
                tp_group = None
        if hasattr(p, "partition_dim"):
            partition_dim = getattr(p, "partition_dim")
            if partition_dim == -1:
                partition_dim = None

        # MoE expert splitting path for GroupedMLP
        if self.split_moe_experts and self.is_grouped_moe_fn is not None and self.is_grouped_moe_fn(p):
            num_local_experts = getattr(p, 'num_local_experts', None)
            if num_local_experts is None or num_local_experts <= 1:
                # If num_local_experts is not set or is 1, fall back to default behavior
                pass  # Continue to QKV/FC1/standard paths below
            else:
                param_name = getattr(p, 'param_name', '')

                if 'weight1' in param_name:
                    # weight1: [hidden_size, num_experts * ffn_per_expert]
                    # Need to account for gated linear units which double the output size
                    is_gated = getattr(p, 'is_gated', False)
                    ffn_multiplier = 2 if is_gated else 1
                    out_dim, in_dim = p.shape
                    ffn_dim_per_expert = in_dim // (num_local_experts * ffn_multiplier)

                    # Reshape: [hidden_size, num_experts, ffn_per_expert * multiplier]
                    # Correct PyTorch view for Row-Major [Hidden, Experts*FFN] layout
                    W_reshaped = p.data.view(out_dim, num_local_experts, ffn_dim_per_expert * ffn_multiplier)
                    M_reshaped = grad.view(out_dim, num_local_experts, ffn_dim_per_expert * ffn_multiplier)

                    # Process each expert independently
                    expert_updates = []
                    for expert_idx in range(num_local_experts):
                        # Slice: [hidden_size, ffn_per_expert * multiplier] = [In, Out]
                        W_expert = W_reshaped[:, expert_idx, :]
                        M_expert = M_reshaped[:, expert_idx, :]

                        # Further split gate and up if split_fc1 is enabled and this is a gated layer
                        if self.split_fc1 and is_gated:
                            # Split into gate and up: each is [hidden_size, ffn_per_expert]
                            W_gate, W_up = torch.split(W_expert, [ffn_dim_per_expert, ffn_dim_per_expert], dim=1)
                            M_gate, M_up = torch.split(M_expert, [ffn_dim_per_expert, ffn_dim_per_expert], dim=1)

                            # Process gate and up independently
                            # Transpose to [Out, In] for spectral update
                            U_gate = self._compute_component_update(
                                W_gate.t(), M_gate.t(), tp_group, partition_dim,
                                current_lr, param_name, f'expert{expert_idx}.gate'
                            ).t()
                            U_up = self._compute_component_update(
                                W_up.t(), M_up.t(), tp_group, partition_dim,
                                current_lr, param_name, f'expert{expert_idx}.up'
                            ).t()

                            # Concatenate gate and up back together
                            U_expert = torch.cat([U_gate, U_up], dim=1)
                        else:
                            # Process the entire expert weight as a single matrix
                            # Transpose to [Out, In] for spectral update
                            U_expert = self._compute_component_update(
                                W_expert.t(), M_expert.t(), tp_group, partition_dim,
                                current_lr, param_name, f'expert{expert_idx}'
                            ).t()
                        expert_updates.append(U_expert)

                    # Merge back to original shape
                    # Stack dim=1 results in [hidden, experts, ffn] -> view -> [hidden, experts*ffn]
                    update = torch.stack(expert_updates, dim=1).view(out_dim, in_dim)
                    return update

                elif 'weight2' in param_name:
                    # weight2 physical layout: [num_experts * ffn_per_expert, hidden_size]
                    # experts.py interprets as: [num_experts, ffn_per_expert, hidden_size]
                    # Each expert slice is [ffn_per_expert, hidden_size] = [In, Out]
                    # Need to transpose to [Out, In] for correct radius calculation
                    out_dim, in_dim = p.shape
                    ffn_dim_per_expert = out_dim // num_local_experts

                    # Reshape matching experts.py: [num_experts, ffn_per_expert, hidden_size]
                    W_reshaped = p.data.view(num_local_experts, ffn_dim_per_expert, in_dim)
                    M_reshaped = grad.view(num_local_experts, ffn_dim_per_expert, in_dim)

                    # Process each expert independently
                    expert_updates = []
                    for expert_idx in range(num_local_experts):
                        W_expert = W_reshaped[expert_idx, :, :]  # [ffn_per_expert, hidden_size] = [In, Out]
                        M_expert = M_reshaped[expert_idx, :, :]

                        # Transpose to [Out, In] for correct radius calculation, then transpose back
                        U_expert = self._compute_component_update(
                            W_expert.t(), M_expert.t(), tp_group, partition_dim,
                            current_lr, param_name, f'expert{expert_idx}'
                        ).t()
                        expert_updates.append(U_expert)

                    # Merge back to original shape
                    update = torch.stack(expert_updates, dim=0).view(out_dim, in_dim)
                    return update

        # QKV splitting path
        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            assert self.qkv_split_shapes is not None, "qkv_split_shapes must be provided when split_qkv=True"
            out_dim, in_dim = p.shape
            split_sum = sum(self.qkv_split_shapes)
            assert (
                out_dim % split_sum == 0
            ), f"QKV split shapes {self.qkv_split_shapes} do not divide output dim {out_dim}"
            num_groups = out_dim // split_sum
            param_name = getattr(p, 'param_name', None)
            component_names = ['q', 'k', 'v']

            # Compute heads_per_group from qkv_split_shapes
            q_dim_per_group, kv_channels, _ = self.qkv_split_shapes  # v_dim not used (same as k_dim/kv_channels)
            heads_per_group = q_dim_per_group // kv_channels

            # reshape: [num_groups, split_sum, in_dim]
            W_view = p.data.view(num_groups, split_sum, in_dim)
            M_view = grad.view(num_groups, split_sum, in_dim)

            if self.qkv_split_mode == "group":
                # Group mode: process each query group independently
                group_updates = []
                for g in range(num_groups):
                    # Split this group into Q/K/V: each is [component_dim, in_dim]
                    Wg_comps = torch.split(W_view[g], list(self.qkv_split_shapes), dim=0)
                    Mg_comps = torch.split(M_view[g], list(self.qkv_split_shapes), dim=0)

                    comp_updates = []
                    for idx, (Wi, Mi) in enumerate(zip(Wg_comps, Mg_comps)):
                        label = f"g{g}.{component_names[idx]}"
                        ui = self._compute_component_update(Wi, Mi, tp_group, partition_dim, current_lr, param_name, label)
                        comp_updates.append(ui)

                    # Concatenate Q/K/V updates within this group: [split_sum, in_dim]
                    group_updates.append(torch.cat(comp_updates, dim=0))

                # Stack all groups and reshape to original fused shape
                update = torch.stack(group_updates, dim=0).reshape(out_dim, in_dim)
                return update

            elif self.qkv_split_mode == "head":
                # Head mode: process each attention head independently for Q/K/V

                group_updates = []
                for g in range(num_groups):
                    # Split this group into Q/K/V
                    Wg_comps = torch.split(W_view[g], list(self.qkv_split_shapes), dim=0)
                    Mg_comps = torch.split(M_view[g], list(self.qkv_split_shapes), dim=0)

                    W_q, W_k, W_v = Wg_comps
                    M_q, M_k, M_v = Mg_comps

                    # Q: split into individual heads and process each
                    # W_q shape: [heads_per_group * kv_channels, in_dim]
                    W_q_heads = W_q.view(heads_per_group, kv_channels, in_dim)
                    M_q_heads = M_q.view(heads_per_group, kv_channels, in_dim)

                    q_head_updates = []
                    for h in range(heads_per_group):
                        label = f"g{g}.Q.h{h}"
                        uh = self._compute_component_update(
                            W_q_heads[h], M_q_heads[h], tp_group, partition_dim,
                            current_lr, param_name, label
                        )
                        q_head_updates.append(uh)

                    # Merge Q head updates: [heads_per_group, kv_channels, in_dim] -> [q_dim, in_dim]
                    U_q = torch.stack(q_head_updates, dim=0).reshape(-1, in_dim)

                    # K and V: single head per group, process directly
                    U_k = self._compute_component_update(W_k, M_k, tp_group, partition_dim,
                                                        current_lr, param_name, f"g{g}.K")
                    U_v = self._compute_component_update(W_v, M_v, tp_group, partition_dim,
                                                        current_lr, param_name, f"g{g}.V")

                    # Concatenate Q/K/V updates within this group
                    group_updates.append(torch.cat([U_q, U_k, U_v], dim=0))

                # Stack all groups and reshape to original fused shape
                update = torch.stack(group_updates, dim=0).reshape(out_dim, in_dim)
                return update

            else:  # component mode (original logic)
                # Component mode: merge all groups' Q together, all K together, all V together
                W_q, W_k, W_v = torch.split(W_view, list(self.qkv_split_shapes), dim=1)
                M_q, M_k, M_v = torch.split(M_view, list(self.qkv_split_shapes), dim=1)

                # flatten per component to 2D matrices (merging all groups)
                comps_W = [W_q.reshape(-1, in_dim), W_k.reshape(-1, in_dim), W_v.reshape(-1, in_dim)]
                comps_M = [M_q.reshape(-1, in_dim), M_k.reshape(-1, in_dim), M_v.reshape(-1, in_dim)]

                updates = []
                for idx, (Wi, Mi) in enumerate(zip(comps_W, comps_M)):
                    ui = self._compute_component_update(Wi, Mi, tp_group, partition_dim, current_lr, param_name, component_names[idx])
                    # reshape back to [num_groups, part, in_dim]
                    part_out = self.qkv_split_shapes[idx]
                    updates.append(ui.view(num_groups, part_out, in_dim))

                # stitch back into fused shape
                U_q, U_k, U_v = updates
                update = torch.cat([U_q, U_k, U_v], dim=1).reshape(out_dim, in_dim)
                return update

        # FC1 splitting path for gated linear units (SwiGLU)
        if self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            assert self.fc1_split_shapes is not None, "fc1_split_shapes must be provided when split_fc1=True"
            out_dim, in_dim = p.shape
            gate_dim, up_dim = self.fc1_split_shapes
            assert (
                out_dim == gate_dim + up_dim
            ), f"FC1 split shapes {self.fc1_split_shapes} do not match output dim {out_dim}"
            param_name = getattr(p, 'param_name', None)

            # Split gate and up along dim=0
            W_gate, W_up = torch.split(p.data, [gate_dim, up_dim], dim=0)
            M_gate, M_up = torch.split(grad, [gate_dim, up_dim], dim=0)

            # Compute MuonBall update for each component
            U_gate = self._compute_component_update(W_gate, M_gate, tp_group, partition_dim, current_lr, param_name, "gate")
            U_up = self._compute_component_update(W_up, M_up, tp_group, partition_dim, current_lr, param_name, "up")

            # Concatenate back
            update = torch.cat([U_gate, U_up], dim=0)
            return update

        # Standard 2D matrix path
        update, bias, sigma = compute_muon_ball_update(
            W=p.data,
            M=grad,
            target_radius=target_radius,
            power_iteration_steps=self.power_iteration_steps,
            msign_steps=self.msign_steps,
            tp_group=tp_group,
            partition_dim=partition_dim,
            retract_mode=self.retract_mode,
            retract_alpha=self.retract_alpha,
            current_lr=current_lr,
        )

        # Record bias (only if dynamic mode and bias != 0)
        if self.retract_mode == 'dynamic' and bias != 0.0:
            param_name = getattr(p, 'param_name', None)
            if param_name:
                self.retract_bias_dict[param_name] = bias
                self.spectral_norm_dict[param_name] = sigma

        # Apply scale factor
        scale_factor = get_spectral_ball_scale_factor(p.shape[0], p.shape[1], mode=self.scale_mode)
        update = update * scale_factor

        return update

    def get_retract_bias_dict(self):
        """Get retract bias dictionary for logging.

        Returns:
            Dictionary mapping module names to their retract bias values (-1 or +1),
            or None if retract_mode is 'hard' or dict is empty.
        """
        if self.retract_mode == 'hard' or not self.retract_bias_dict:
            return None
        return self.retract_bias_dict

    def get_spectral_norm_dict(self):
        """Get spectral norm dictionary for logging.

        Returns:
            Dictionary mapping module names to their spectral norm values,
            or None if dict is empty.
        """
        if not self.spectral_norm_dict:
            return None
        return self.spectral_norm_dict


MuonBall.__doc__ = MuonBall.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
