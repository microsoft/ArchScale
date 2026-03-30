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
from functools import partial
from itertools import chain
from typing import Callable, List, Optional, Tuple, Union


# TODO(@boxiangw): remove this once bump to python 3.12
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch
from absl import logging
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import scalar_optimizers, utils
from emerging_optimizers.soap import soap_utils


__all__ = [
    "SOAP",
    "precondition",
    "init_kronecker_factors",
    "update_kronecker_factors",
    "update_eigenbasis_and_momentum",
]


class SOAP(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Implements a variant of SOAP (ShampoO with Adam in the Preconditioner eigenbasis) algorithm.

    SOAP (https://arxiv.org/abs/2409.11321) is a preconditioned optimizer that combines the benefits of Shampoo's
    non-diagonal preconditioning with Adam's adaptive learning rates. It uses
    gradient correlation matrix eigenbasis-based preconditioning to adapt to the local geometry of the
    optimization landscape.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate to use
        betas: Inner Adam's betas parameters (b1, b2)
        shampoo_beta: Beta for the kronecker factor matrices (L and R in paper) moving average
            instead of betas[1] if >= 0
        eps: Inner Adam's epsilon for numerical stability
        weight_decay: Weight decay coefficient
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`
            for more details.
        use_nesterov: uses Nesterov momentum in Adam (https://cs229.stanford.edu/proj2015/054_report.pdf,
            https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)
        precondition_frequency: How often to update the preconditioner. Can be an integer for fixed frequency
            or a callable function that takes the current step as input and returns the frequency.
        adam_warmup_steps: How many steps to skip preconditioning in the beginning (i.e. use standard AdamW updates)
        precondition_1d: Whether to precondition 1D gradients (like biases).
        correct_bias: Whether to use bias correction in Inner Adam and Kronecker factor matrices EMA
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis.
        qr_fp32_matmul_prec: Precision of the matmul operations in QR decomposition.
        use_adaptive_criteria: Whether to use criteria to determine if eigenbasis update is needed
        adaptive_update_tolerance: Tolerance threshold for the update criteria.
            Only used if use_adaptive_criteria is True.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
        use_kl_shampoo: Whether to use KL-Shampoo correction.
        correct_shampoo_beta_bias: Whether to correct shampoo beta bias. Decoupled it from correct_bias for
            testability because reference implementation of Soap doesn't bias correct shampoo beta.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        use_nesterov: bool = False,
        precondition_frequency: Union[int, Callable[[int], int]] = 1,
        adam_warmup_steps: int = 0,
        precondition_1d: bool = False,
        correct_bias: bool = True,
        fp32_matmul_prec: str = "high",
        use_eigh: bool = False,
        qr_fp32_matmul_prec: str = "high",
        use_adaptive_criteria: bool = False,
        adaptive_update_tolerance: float = 1e-7,
        power_iter_steps: int = 1,
        max_update_rms: float = 0.0,
        use_kl_shampoo: bool = False,
        correct_shampoo_beta_bias: bool | None = None,
    ) -> None:
        self.precondition_frequency = precondition_frequency
        self.adam_warmup_steps = adam_warmup_steps
        self.precondition_1d = precondition_1d
        self.use_nesterov = use_nesterov
        self.correct_bias = correct_bias
        self.weight_decay_method = weight_decay_method
        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_eigh = use_eigh
        self.qr_fp32_matmul_prec = qr_fp32_matmul_prec
        self.use_adaptive_criteria = use_adaptive_criteria
        self.adaptive_update_tolerance = adaptive_update_tolerance
        self.power_iter_steps = power_iter_steps
        self.max_update_rms = max_update_rms
        self.use_kl_shampoo = use_kl_shampoo
        if correct_shampoo_beta_bias is not None:
            self.correct_shampoo_beta_bias = correct_shampoo_beta_bias
        else:
            self.correct_shampoo_beta_bias = correct_bias

        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # NOTE: The upstream PyTorch implementations increment the step counter in the middle of the loop
                # to be used in bias correction. But this is confusing and error prone if anything else needs to use
                # the step counter.
                # We decided to follow Python and C convention to increment the step counter at the end of the loop.
                # An explicitly named 1-based iteration/step counter is created for bias correction and other terms
                # in the math equation that needs 1-based iteration count.
                curr_iter_1_based = state["step"] + 1

                # TODO(Mkhona): Improve initialization handling.
                # - More protective checks can be added to avoid potential issues with checkpointing.
                # - Initializing zero buffers can also be avoided.
                if state["step"] == 0:
                    assert all(key not in state for key in ["exp_avg", "exp_avg_sq", "GG"]), (
                        "exp_avg and exp_avg_sq and GG should not be initialized at step 0. "
                        "Some mismatch has been created likely in checkpointing"
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    # Initialize kronecker factor matrices
                    state["GG"] = init_kronecker_factors(
                        grad,
                        precondition_1d=self.precondition_1d,
                    )

                # Define kronecker_factor_update_fn based on whether to use KL-Shampoo here
                # because it needs access to state and group
                if not self.use_kl_shampoo:
                    kronecker_factor_update_fn = partial(
                        update_kronecker_factors,
                        precondition_1d=self.precondition_1d,
                    )
                else:
                    if "Q" not in state:
                        assert state["step"] == 0, (
                            f"Q should already be initialized at step {state['step']}, Some mismatch has been created "
                            "likely in checkpointing"
                        )
                        state["Q"] = [torch.eye(shape, device=grad.device) for shape in grad.shape]
                    kronecker_factor_update_fn = partial(
                        update_kronecker_factors_kl_shampoo,
                        eigenbasis_list=state["Q"],
                        eps=group["eps"],
                    )

                shampoo_beta = group["shampoo_beta"]
                if self.correct_shampoo_beta_bias:
                    shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**curr_iter_1_based)

                torch.cuda.nvtx.range_push("update_kronecker_factors")
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    kronecker_factor_update_fn(kronecker_factor_list=state["GG"], grad=grad, shampoo_beta=shampoo_beta)
                torch.cuda.nvtx.range_pop()

                # After the adam_warmup_steps are completed , update eigenbases at precondition_frequency steps
                torch.cuda.nvtx.range_push("Update eigen basis")
                if _is_eigenbasis_update_step(
                    state["step"],
                    self.adam_warmup_steps,
                    self.precondition_frequency,
                ):
                    # Always use eigh for the first eigenbasis update
                    use_eigh = self.use_eigh if state["step"] != self.adam_warmup_steps else True

                    with utils.fp32_matmul_precision(self.qr_fp32_matmul_prec):
                        state["Q"], state["exp_avg"], state["exp_avg_sq"] = update_eigenbasis_and_momentum(
                            kronecker_factor_list=state["GG"],
                            eigenbasis_list=state.get("Q", None),
                            exp_avg_sq=state["exp_avg_sq"],
                            momentum=state["exp_avg"],
                            use_eigh=use_eigh,
                            use_adaptive_criteria=self.use_adaptive_criteria,
                            adaptive_update_tolerance=self.adaptive_update_tolerance,
                            power_iter_steps=self.power_iter_steps,
                        )
                torch.cuda.nvtx.range_pop()

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                grad_projected = grad
                # Project gradients to the eigenbases of Shampoo's preconditioner
                torch.cuda.nvtx.range_push("precondition")
                if state["step"] >= self.adam_warmup_steps:
                    with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        grad_projected = precondition(
                            grad=grad,
                            eigenbasis_list=state["Q"],
                            dims=[[0], [0]],
                        )
                torch.cuda.nvtx.range_pop()

                # Calculate the Adam update for the projected gradient tensor
                adam_update = scalar_optimizers.calculate_adam_update(
                    grad_projected,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    group["betas"],
                    self.correct_bias,
                    self.use_nesterov,
                    curr_iter_1_based,  # 1-based iteration index is used for bias correction
                    group["eps"],
                )

                # Projecting back the preconditioned (by ADAM) exponential moving average of gradients
                torch.cuda.nvtx.range_push("precondition")
                if state["step"] >= self.adam_warmup_steps:
                    with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        precond_update = precondition(
                            grad=adam_update,
                            eigenbasis_list=state.get("Q", None),
                            dims=[[0], [1]],
                        )
                else:
                    precond_update = adam_update
                torch.cuda.nvtx.range_pop()

                _clip_update_rms_in_place(precond_update, self.max_update_rms)
                p.add_(precond_update, alpha=-group["lr"])

                state["step"] += 1

        return loss


@torch.no_grad()  # type: ignore[misc]
def init_kronecker_factors(
    grad: torch.Tensor,
    precondition_1d: bool = False,
) -> List[torch.Tensor]:
    """Initializes the kronecker factor matrices for the SOAP optimizer.

    This function creates the initial Kronecker factor matrices (L and R) used for
    preconditioning. For 1D tensors (like biases), it can either skip preconditioning
    or create a single square kronecker factor matrix. For higher dimensional tensors,
    it creates a square kronecker factor matrix for each dimension.

    When precondition_1d is:
        * False (default):
            - 1D tensors (like biases) will skip SOAP preconditioning entirely
            - These parameters will use standard Adam-style updates
            - This is often desirable as biases typically have fewer parameters and simpler optimization landscapes
            - Can improve performance and reduce memory usage
        * True:
            - All parameters, including 1D tensors, will use SOAP preconditioning
            - May be beneficial for certain architectures or training scenarios

    Args:
        grad: Gradient tensor used to initialize the kronecker factor matrices.
            The shape of this tensor determines the size of the kronecker factor matrices.
        precondition_1d: Whether to create kronecker factor matrices for 1D tensors
            (like biases). If False, 1D tensors will skip preconditioning.

    Returns:
        List[torch.Tensor]: List of kronecker factor matrices (L and R in paper).
            - For 1D tensors with precondition_1d=False: List containing an empty tensor
            - For 1D tensors with precondition_1d=True: List containing a square matrix
            - For higher dimensional tensors: List of square matrices, one per dimension

    Example:
        >>> # For a 1D tensor (bias)
        >>> grad_1d = torch.randn(10)
        >>> precond_1d = init_kronecker_factors(grad_1d, precondition_1d=True)
        >>> print(len(precond_1d))  # 1
        >>> print(precond_1d[0].shape)  # (10, 10)

        >>> # For a 2D tensor (weight matrix)
        >>> grad_2d = torch.randn(10, 20)
        >>> precond_2d = init_kronecker_factors(grad_2d)
        >>> print(len(precond_2d))  # 2
        >>> print(precond_2d[0].shape)  # (10, 10)
        >>> print(precond_2d[1].shape)  # (20, 20)

    """
    kronecker_factor_list: List[torch.Tensor] = []

    if grad.dim() == 1:
        if not precondition_1d:
            # Skip preconditioning for 1D tensors
            kronecker_factor_list.append(torch.empty(0, device=grad.device))
        else:
            # Create a square preconditioner matrix for 1D tensors
            size = grad.shape[0]
            kronecker_factor_list.append(torch.zeros(size, size, device=grad.device))
    else:
        # Create a square kronecker factor matrix for each dimension
        for size in grad.shape:
            kronecker_factor_list.append(torch.zeros(size, size, device=grad.device))

    return kronecker_factor_list


@torch.no_grad()  # type: ignore[misc]
def update_kronecker_factors(
    kronecker_factor_list: List[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
    precondition_1d: bool = False,
) -> None:
    """Updates the preconditioner matrices using gradient outer products.

    This function updates the Kronecker factor matrices (L and R) used for preconditioning
    by computing and accumulating gradient outer products. For 1D tensors (like biases),
    it can optionally skip preconditioning or use a special 1D preconditioning strategy.
    It modifies the kronecker_factor_list in place.

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) to update.
            Each matrix should be square and match the corresponding dimension of grad.
        grad: Gradient tensor of the parameter being optimized
        shampoo_beta: Momentum coefficient for updating preconditioners.
            Controls how much weight to give to new vs old gradient statistics.
        precondition_1d: Whether to apply preconditioning to 1D tensors (like biases).
            If False, 1D tensors will skip preconditioning.

    Example:
        >>> grad = torch.randn(10, 20)
        >>> L = torch.zeros(10, 10)
        >>> R = torch.zeros(20, 20)
        >>> update_preconditioner([L, R], grad, shampoo_beta=0.95)

    """
    if grad.dim() == 1:
        if precondition_1d:
            # For 1D tensors, compute outer product directly
            outer_product = grad.unsqueeze(1) @ grad.unsqueeze(0)
            kronecker_factor_list[0].lerp_(outer_product, 1 - shampoo_beta)
        else:
            # For 1D tensors, skip preconditioning
            logging.error(
                "1D tensor is passed to update_kronecker_factors, "
                "but precondition_1d is not set to True, skipping preconditioning."
            )
            return
    else:
        # For higher dimensional tensors, compute outer products for each dimension
        for idx, dim_size in enumerate(grad.shape):
            # Compute outer product by contracting all dimensions except idx
            contract_dims = [*chain(range(idx), range(idx + 1, grad.dim()))]
            outer_product = torch.tensordot(
                grad,
                grad,
                dims=[contract_dims] * 2,
            )
            # Update the corresponding Kronecker factor
            kronecker_factor_list[idx].lerp_(outer_product, 1 - shampoo_beta)


@torch.no_grad()  # type: ignore[misc]
def update_kronecker_factors_kl_shampoo(
    kronecker_factor_list: List[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
    eigenbasis_list: List[torch.Tensor],
    eps: float,
    eigval_exp: float = -1.0,
) -> None:
    """Updates the kronecker factor matrices in place using KL-Shampoo correction.

    Implement Kullback–Leibler Minimization from https://arxiv.org/pdf/2509.03378

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) to update.
        grad: Gradient tensor of the parameter being optimized
        shampoo_beta: Momentum coefficient for updating preconditioners.
        eigenbasis_list: List of orthonormal eigenbases of the kronecker factor matrices
        eps: Small offset for numerical stability.
        eigenval_exp: Exponent of the eigenvalues.
    """
    if grad.dim() != 2:
        raise TypeError("KL-Shampoo mathematical correction is only supported for 2D tensors")

    # Scale the gradient matrix by the approximate eigenvalues and the eigenbasis
    # G@Q_R@λ_R^(−1)@Q_R.T@G.T/dim(GG.T) and G.T@Q_L@λ_L^(−1)@Q_L.T@G/dim(G.TG)
    updates = []
    for idx, (kronecker_factor, eigenbasis) in enumerate(zip(kronecker_factor_list, eigenbasis_list, strict=True)):
        approx_eigvals = utils.eig.conjugate(kronecker_factor, eigenbasis, diag=True)
        scale_factor = 1 / grad.shape[idx] * approx_eigvals.clamp_min(eps) ** eigval_exp

        logging.debug(f"scale_factor[{idx}]: {scale_factor}")

        correction = (eigenbasis * scale_factor[None, :]) @ eigenbasis.T

        maybe_transpose_grad = grad.T if idx == 1 else grad
        updates.append(utils.eig.conjugate(correction, maybe_transpose_grad))

    # Note that updates caculated in previous loop are in reverse order of the kronecker factor list they apply to
    for kronecker_factor, update in zip(kronecker_factor_list, updates[::-1], strict=True):
        kronecker_factor.lerp_(update, 1 - shampoo_beta)


@torch.no_grad()  # type: ignore[misc]
def update_eigenbasis_and_momentum(
    kronecker_factor_list: List[torch.Tensor],
    eigenbasis_list: List[torch.Tensor],
    exp_avg_sq: torch.Tensor,
    momentum: torch.Tensor,
    use_eigh: bool = False,
    use_adaptive_criteria: bool = False,
    adaptive_update_tolerance: Optional[float] = None,
    power_iter_steps: int = 1,
    convert_to_float: bool = True,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Updates the eigenbases using QR decomposition and power iteration or eigh.

    This function performs an update of the eigenbases (QL and QR)
    used for preconditioning. It follows these steps:

    1. Projects momentum back to the original basis
    2. Updates the eigenbases using QR decomposition and power iteration (orthogonal iteration)
    3. Projects momentum back to the new eigenbasis

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) that define
            the optimization landscape. These are updated with gradient statistics.
        eigenbasis_list: List of current eigenbases (QL and QR)
            used for preconditioning. These will be updated by this function.
        exp_avg_sq: Inner Adam's second moment tensor, used for scaling the preconditioner updates.
            This tensor is modified in-place.
        momentum: Inner Adam's first moment tensor, used for tracking gradient momentum.
            This tensor is modified in-place.
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis.
        use_adaptive_criteria: Whether to use criteria to determine if eigenbasis update is needed
        adaptive_update_tolerance: Tolerance threshold for the update criteria.
            Only used if use_adaptive_criteria is True.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.
        convert_to_float: Whether to convert the preconditioner matrices and their corresponding
            orthonormal matrices to float for amortized computation. Otherwise, they are left in their original type.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: A tuple containing:
            - List[torch.Tensor]: Updated list of eigenbases (QL and QR)
            - torch.Tensor: Updated momentum tensor projected to the new eigenbasis

    Example:
        >>> L = torch.randn(10, 10)
        >>> R = torch.randn(20, 20)
        >>> QL = torch.randn(10, 10)
        >>> QR = torch.randn(20, 20)
        >>> exp_avg_sq = torch.randn(10, 20)
        >>> momentum = torch.randn(10, 20)
        >>> updated_eigenbases = update_eigenbasis(
        ...     [L, R], [QL, QR], exp_avg_sq, momentum)

    """
    # Step 1: Project momentum back to the original basis
    torch.cuda.nvtx.range_push("eigenbasis update step 1: precondition")
    momentum = precondition(
        momentum,
        eigenbasis_list,
        dims=[[0], [1]],  # Project back to original space
    )
    torch.cuda.nvtx.range_pop()

    # Step 2: Update eigenbases
    torch.cuda.nvtx.range_push("eigenbasis update step 2: update Q")
    if use_eigh:
        updated_eigenbasis_list = soap_utils.get_eigenbasis_eigh(
            kronecker_factor_list,
            convert_to_float,
            eigenbasis_list,
            use_adaptive_criteria,
            adaptive_update_tolerance,
        )
    else:
        # Use QR decomposition and power iteration (orthogonal iteration)
        updated_eigenbasis_list, exp_avg_sq = soap_utils.get_eigenbasis_qr(
            kronecker_factor_list,
            eigenbasis_list,
            exp_avg_sq,
            convert_to_float,
            use_adaptive_criteria,
            adaptive_update_tolerance,
            power_iter_steps,
        )
    torch.cuda.nvtx.range_pop()

    # Step 3: Project momentum to the new eigenbasis using the updated eigenbases
    torch.cuda.nvtx.range_push("eigenbasis update step 3: project momentum")
    momentum = precondition(
        momentum,
        updated_eigenbasis_list,  # Use the new eigenbases
        dims=[[0], [0]],  # Project to new eigenbasis
    )
    torch.cuda.nvtx.range_pop()

    return updated_eigenbasis_list, momentum, exp_avg_sq


@torch.no_grad()  # type: ignore[misc]
@torch.compile  # type: ignore[misc]
def precondition(
    grad: torch.Tensor,
    eigenbasis_list: Optional[List[torch.Tensor]] = None,
    dims: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Projects the gradient to and from the eigenbases of the kronecker factor matrices.

    This function performs tensor contractions between the input gradient
    and kronecker factor eigenbases.


    Args:
        grad: Input tensor to be preconditioned
        eigenbasis_list: List of eigenbases for preconditioning.
            Each matrix should be a square matrix of eigenvectors.
        dims: Dimensions for tensor contraction. Default is [[0], [0]] which contracts
            the first dimension of grad with the first dimension of each eigenbasis matrix,
            for projecting into the eigenbasis. Use [[0], [1]] for projecting back to original space.

    Example:
        >>> grad = torch.randn(10, 20)
        >>> Q = torch.randn(10, 10)
        >>> precondition(grad, [Q], dims=[[0], [0]])
    """
    if dims is None:
        # Pick contraction dims to project to the eigenbasis
        dims = [[0], [0]]

    if eigenbasis_list is None:
        # If eigenbases are not provided, return the gradient without any preconditioning
        return grad

    for Q in eigenbasis_list:
        if Q.numel() > 0:
            # Perform in-place contraction
            grad = torch.tensordot(
                grad,
                Q,
                dims=dims,
            )
        else:
            # Permute gradient dimensions to process the next dimension in the following iteration
            # when preconditioning for the current dimension is skipped (Q is empty), in the case of
            # one-sided preconditioning.
            permute_order = list(range(1, grad.dim())) + [0]
            grad = grad.permute(permute_order)

    return grad


def _is_eigenbasis_update_step(
    step: int,
    adam_warmup_steps: int,
    precondition_frequency: Union[int, Callable[[int], int]],
) -> bool:
    """Checks if amortized computation of the eigenbasis should be recomputed.

    Args:
        step: Current step of the optimizer
        adam_warmup_steps: Number of steps to skip preconditioning in the beginning (i.e. use standard AdamW updates)
        precondition_frequency: How often to update the preconditioner. Can be an integer for fixed frequency
            or a callable function that takes the current step as input and returns the frequency.
    """
    if step < adam_warmup_steps:
        return False

    current_frequency = (
        precondition_frequency if not callable(precondition_frequency) else precondition_frequency(step)
    )

    return step % current_frequency == 0


@torch.compile  # type: ignore[misc]
def _clip_update_rms_in_place(u: torch.Tensor, max_rms: float, eps: float = 1e-7) -> None:
    """Clip the update root mean square (RMS) to a maximum value, in place.

    Do not clip if max_rms is 0.
    Inspired by Adafactor (https://arxiv.org/abs/1804.04235) and RMS_t (https://arxiv.org/abs/2304.13013)

    Args:
        u: The update tensor.
        max_rms: The maximum RMS value.
        eps: The epsilon value to prevent division by zero.
    """
    if max_rms == 0:
        return
    # compute current update RMS
    rms = u.square().mean().sqrt()
    # compute scale factor = min(1.0, max_rms/(rms + eps))
    scale = (max_rms / (rms + eps)).clamp(max=1.0)
    # in‐place scale
    u.mul_(scale)
