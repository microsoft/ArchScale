"""Utility functions for Spectral Ball optimizer."""

import math
from typing import Optional, Tuple

import torch
from absl import logging

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
from emerging_optimizers import utils

DEBUG_CONVERGED = False 
DEBUG_NOT_CONVERGED = False 

__all__ = [
    "compute_target_radius",
    "compute_spectral_ball_update",
    "solve_lambda_with_bisection",
    # "solve_lambda_with_bisection_gpu",
]

# for newton_schulz_step_tsyrk 
# torch.set_float32_matmul_precision("medium")

@torch.no_grad()
def _muon_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X

# enable dependents on hardware
# low performance (+200ms) on a100 after enable torch.compile
@torch.compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def _small_msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        raise ValueError(f"Input tensor G must be in float32")
 
    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    """
    WARNING: DO NOT run `msign` in bfloat16! The matrix-sign Newton–Schulz iteration is extremely sensitive to
    rounding; computing it in bf16 (or casting inputs to bf16 inside `msign`) will severely distort the update
    direction, break the intended spectral geometry, and can easily degrade or destabilize training. Always keep
    `msign` computations in full fp32.
    """
    # cast to bfloat16 to improve performance 
    X = X.to(torch.bfloat16)
    
    coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]
 
    for i in range(steps):
        if i < 8:
            a, b, c = coeffs[i]
        else:
            a, b, c = coeffs[-1]
        X = _muon_newton_schulz_step(X, a, b, c)
 
    return X.mT if transpose else X

# # @torch.compile  # type: ignore[misc]
# @torch.no_grad()
# def _small_msign(G: torch.Tensor, steps: int) -> torch.Tensor:
#     """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
#     if G.ndim < 2:
#         raise ValueError("Input tensor must have at least 2 dimensions.")
#     if G.dtype != torch.float32:
#         raise ValueError(f"Input tensor G must be in float32")
# 
#     transpose = G.size(-2) > G.size(-1)
#     X = G.mT if transpose else G
#     X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
#     """
#     WARNING: DO NOT run `msign` in bfloat16! The matrix-sign Newton–Schulz iteration is extremely sensitive to
#     rounding; computing it in bf16 (or casting inputs to bf16 inside `msign`) will severely distort the update
#     direction, break the intended spectral geometry, and can easily degrade or destabilize training. Always keep
#     `msign` computations in full fp32.
#     """
#     # cast to bfloat16 to improve performance 
#     # X = X.to(torch.bfloat16)
#     
#     if steps == 8:
#         coeffs = [
#             (8.2051, -22.9019, 16.4607),
#             (4.0664, -2.8612, 0.5184),
#             (3.9096, -2.8234, 0.5250),
#             (3.2856, -2.4153, 0.4853),
#             (2.2779, -1.6198, 0.3985),
#             (1.8726, -1.2307, 0.3585),
#             (1.8564, -1.2132, 0.3568),
#             (1.8750, -1.2500, 0.3750),
#         ]
#     else:
#         coeffs = [
#             (4.0848, -6.8946, 2.9270),
#             (3.9505, -6.3029, 2.6377),
#             (3.7418, -5.5913, 2.3037),
#             (2.8769, -3.1427, 1.2046),
#             (2.8366, -3.0525, 1.2012),
#         ]
#     for i in range(steps):
#         if i < 8:
#             a, b, c = coeffs[i]
#         else:
#             a, b, c = coeffs[-1]
#         X = _muon_newton_schulz_step(X, a, b, c)
# 
#     return X.mT if transpose else X

@torch.compile  # type: ignore[misc]
@torch.no_grad()
def _large_msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]
    with utils.fp32_matmul_precision("medium"):
        return newton_schulz(G, steps=steps, coefficient_type="custom", \
            custom_coefficient_sets=coeffs, use_syrk=True)

@torch.no_grad()
def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    # if G.shape[0] <= 512 or G.shape[1] <= 512:
    # force disable triton syrk branch
    if True:
        return _small_msign(G, steps)
    else:
        return _large_msign(G, steps)

@torch.compile
@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 50, eps: float = 1e-20):
    """Leading singular triplet (σ, u, v) via bilateral power iteration (fp32/bf16)."""
    if w.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    # w = w.to(torch.float32)
    w = w.to(torch.bfloat16)
    v = torch.ones_like(w[..., :1, :].transpose(-2, -1))
    for _ in range(steps):
        v = torch.nn.functional.normalize(w.transpose(-2, -1) @ (w @ v), dim=-2)
    u = torch.nn.functional.normalize(w @ v, dim=-2)
    s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)

    return s, u, v


@torch.no_grad()
def apply_retract(
    W: torch.Tensor,
    sigma: float,
    target_radius: float,
    mode: str = 'hard',
    alpha: float = 0.05,
    current_lr: Optional[float] = None,
) -> float:
    """Apply retraction to spectral sphere.

    Args:
        W: Weight matrix (modified in-place)
        sigma: Current spectral norm
        target_radius: Target radius R
        mode: 'hard' or 'dynamic'
        alpha: Step size for dynamic mode (ignored for hard mode)
        current_lr: Current learning rate (only used in dynamic mode to scale alpha)

    Returns:
        bias: The bias value used (only relevant for dynamic mode, 0.0 for hard mode)
    """
    if mode == 'hard':
        # Hard retraction: if sigma != R, scale W to have norm R
        if max(sigma, 0.0) + 1e-8 != target_radius:
            scale_factor = target_radius / (max(sigma, 0.0) + 1e-8)
            W.mul_(scale_factor)
        return 0.0

    elif mode == 'dynamic':
        # Dynamic retraction: bias = -sign(sigma - R), W *= (1 + alpha * current_lr * bias)
        # This aligns the retraction strength with weight decay: both scale with lr
        bias = -1.0 if sigma > target_radius else 1.0

        # If current_lr is provided, scale alpha by lr (to align with weight decay)
        # Otherwise, use alpha directly (backward compatibility)
        if current_lr is not None:
            effective_alpha = alpha * current_lr
        else:
            effective_alpha = alpha

        W.mul_(1.0 + effective_alpha * bias)
        return bias

    else:
        raise ValueError(f"Unknown retract mode: {mode}")


@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>."""
    return (a * b).sum()


@torch.no_grad()
def compute_phi(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 5) -> torch.Tensor:
    """Φ(λ) = msign(G + λΘ)."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    return Phi

@torch.no_grad()
def compute_f(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 8) -> float:
    """f(λ) = <Θ, msign(G + λΘ)>. Returns scalar float (triggers GPU sync)."""
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)
    f_value = float(inner_product(Theta, Phi).item())
    return f_value

@torch.compile
@torch.no_grad()
def compute_f_tensor(G: torch.Tensor, Theta: torch.Tensor, lambda_value: torch.Tensor, msign_steps: int = 8) -> torch.Tensor:
    """f(λ) = <Θ, msign(G + λΘ)>. Returns 0-d tensor (no GPU sync)."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    return inner_product(Theta, Phi)


@torch.no_grad()
def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    max_expansions: int = 10,
    msign_steps: int = 8,
    tolerance_f: float = 1e-8,
) -> Tuple[float, float, float, float]:
    """
    Find λ_L < λ_R such that:
        f(λ_L) <= 0 <= f(λ_R)
    with f monotone increasing.

    If f(initial_guess) is already near zero, returns a degenerate bracket.
    Otherwise expands exponentially in the direction indicated by f0.
    """

    # Function handle
    f = compute_f_tensor

    # Initial λ and f
    λ0 = initial_guess
    f0 = f(G, Theta, λ0, msign_steps)

    # If already close to zero → return degenerate bracket
    if abs(f0) < tolerance_f:
        return λ0, λ0, f0, f0

    # Decide direction:
    #   f0 < 0 → root is to the right  → step > 0
    #   f0 > 0 → root is to the left   → step < 0
    step = initial_step if f0 < 0 else -initial_step

    λ_prev = λ0
    f_prev = f0

    for _ in range(max_expansions):

        λ_new = λ_prev + step
        f_new = f(G, Theta, λ_new, msign_steps)

        # ---------------------------
        # Check sign change:
        # f_prev ≤ 0 ≤ f_new OR f_new ≤ 0 ≤ f_prev
        # ---------------------------
        sign_prev = f_prev <= 0.0
        sign_new  = f_new  <= 0.0

        if sign_prev != sign_new:  # sign change occurred
            # ------------------------------------------------
            # Choose λ_L, λ_R based *on f*, NOT λ ordering.
            # Always enforce: f_L <= 0 <= f_R
            # ------------------------------------------------
            if f_prev <= 0 and f_new >= 0:
                λ_L, f_L = λ_prev, f_prev
                λ_R, f_R = λ_new, f_new
            elif f_new <= 0 and f_prev >= 0:
                λ_L, f_L = λ_new, f_new
                λ_R, f_R = λ_prev, f_prev
            else:
                # One point is extremely close to zero
                if abs(f_prev) <= abs(f_new):
                    λ_L = λ_R = λ_prev
                    f_L = f_R = f_prev
                else:
                    λ_L = λ_R = λ_new
                    f_L = f_R = f_new
            if DEBUG_CONVERGED:
                logging.warning(
                    f"[find_bracket] CONVERGED after {_ + 1} expansions. "
                    f"λ_L={λ_L:.6e}, f_L={f_L:.6e}, λ_R={λ_R:.6e}, f_R={f_R:.6e}."
                )
            return λ_L, λ_R, f_L, f_R

        # ------------------------------------------------
        # No sign change → expand search region
        # ------------------------------------------------
        step *= 2.0
        λ_prev, f_prev = λ_new, f_new

    # Failsafe
    logging.warning(
        f"[find_bracket] Could not bracket the root after {max_expansions} expansions. "
        f"Last λ={λ_prev:.6e}, f={f_prev:.6e}, w shape={G.shape}"
    )

    return None, None, f0, f0 #没找到，则区间返回none,直接返回f0




@torch.no_grad()
def solve_lambda_with_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 8,
) -> Tuple[float, bool, float, int]:
    """
    Solve λ such that f(λ) = <Θ, msign(G + λΘ)> = 0 using bisection.
    Assumes f is strictly monotone increasing.

    Returns:
        (lambda_star, converged_bool, |f(lambda_star)|, iterations_used)
    """

    # ----------------------------------------------------------------------
    # 1. Bracket the root: must satisfy f_L <= 0 <= f_R
    # ----------------------------------------------------------------------
    λ_L, λ_R, f_L, f_R = find_bracket(
        G, Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )

    # Bracketing failed
    if λ_L is None:
        logging.error("[bisect] find_bracket failed: cannot continue bisection.")
        return 0.0, False, f_L , 0 #其实就是直接返回lambda=0,退化为muon更新

    # ----------------------------------------------------------------------
    # 2. Pick best endpoint first 
    # ----------------------------------------------------------------------
    if abs(f_L) < abs(f_R):
        best_λ, best_f = λ_L, f_L
    else:
        best_λ, best_f = λ_R, f_R

    # If best endpoint already satisfies tolerance → done
    if abs(best_f) <= tolerance_f:
        if DEBUG_CONVERGED:
            logging.warning(
                f"[bisect] CONVERGED after bracketing search. "
                f"best λ={best_λ:.6e}, |f|={abs(best_f):.6e}."
            )
        return best_λ, True, abs(best_f), 0

    # ----------------------------------------------------------------------
    # 3. Standard monotone bisection
    # ----------------------------------------------------------------------
    for it in range(1, max_iterations + 1):

        λ_mid = 0.5 * (λ_L + λ_R)
        f_mid = compute_f_tensor(G, Theta, λ_mid, msign_steps)

        # Track best point (fallback)
        if abs(f_mid) < abs(best_f):
            best_λ, best_f = λ_mid, f_mid

        # Converged
        if abs(f_mid) <= tolerance_f:
            if DEBUG_CONVERGED:
                logging.warning(
                    f"[bisect] CONVERGED after {it} iterations. "
                    f"λ_mid={λ_mid:.6e}, |f|={abs(f_mid):.6e}."
                )
            return λ_mid, True, abs(f_mid), it

        # f is strictly increasing:
        # f_mid < 0 → root is in (mid, R)
        # f_mid > 0 → root is in (L, mid)
        if f_mid < 0:
            λ_L, f_L = λ_mid, f_mid
        else:
            λ_R, f_R = λ_mid, f_mid

    # ----------------------------------------------------------------------
    # 4. Not converged: return best-so-far
    # ----------------------------------------------------------------------
    if DEBUG_NOT_CONVERGED:
        logging.warning(
            f"[bisect] NOT CONVERGED after bisection search. "
            f"λ_L={λ_L:.6e}, f_L={f_L:.6e}, λ_R={λ_R:.6e}, f_R={f_R:.6e}."
        )
    return best_λ, False, abs(best_f), max_iterations


def compute_target_radius(shape: tuple, radius_mode: str, current_weight: Optional[torch.Tensor] = None, radius_scaler: float = 1.0) -> float:
    """Compute target radius R: 'spectral_mup' → sqrt(n_out/n_in) * scaler, 'identity' → 1.0 * scaler."""
    if radius_mode == "spectral_mup":
        n_out, n_in = shape
        return radius_scaler * math.sqrt(n_out / n_in)
    elif radius_mode == "identity":
        return radius_scaler * 1.0
    else:
        raise ValueError(f"Invalid radius_mode: {radius_mode}. Must be 'spectral_mup' or 'identity'.")

def get_spectral_ball_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Get the scale factor for the spectral ball update.

    This function mirrors Muon's scale factor to enable learning rate transferability.
    The default "align_adamw_rms" mode uses the same scaling as Muon for consistency.

    Args:
        size_out: The size of the output dimension (rows).
        size_in: The size of the input dimension (columns).
        mode: The mode to use for the scale.
            - "align_adamw_rms": 0.2 * max(size_out, size_in) ** 0.5 (default, matches Muon)
            - "shape_scaling": max(1, size_out / size_in) ** 0.5
            - "spectral_mup": (size_out / size_in) ** 0.5

    Returns:
        The scale factor for the update.
    """
    if mode == "shape_scaling":
        return max(1, size_out / size_in) ** 0.5
    elif mode == "align_adamw_rms":
        return 0.2 * max(size_out, size_in) ** 0.5
    elif mode == "spectral_mup":
        return (size_out / size_in) ** 0.5
    else:
        raise ValueError(f"Invalid mode for SpectralBall update scale factor: {mode}")

@torch.no_grad()
def _tp_world_and_rank(tp_group: torch.distributed.ProcessGroup | None) -> tuple[int, int]:
    """Return (world_size, rank) from tp_group."""
    if tp_group is None:
        return 1, 0
    return tp_group.size(), tp_group.rank()


@torch.no_grad()
def _tp_gather_along_dim(x: torch.Tensor, tp_group: torch.distributed.ProcessGroup, dim: int) -> torch.Tensor:
    """All-gather shards along dim."""
    ws, _ = _tp_world_and_rank(tp_group)
    if ws == 1:
        return x
    shards = [torch.empty_like(x) for _ in range(ws)]
    torch.distributed.all_gather(shards, x, group=tp_group)
    return torch.cat(shards, dim=dim)


@torch.no_grad()
def _tp_split_along_dim(x_full: torch.Tensor, tp_group: torch.distributed.ProcessGroup, dim: int) -> torch.Tensor:
    """Split global tensor along dim, return local shard."""
    ws, rk = _tp_world_and_rank(tp_group)
    if ws == 1:
        return x_full
    parts = x_full.chunk(ws, dim=dim)
    return parts[rk].contiguous()


def _compute_single_rank(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    solver: str,
    solver_tolerance_f: float,
    solver_max_iterations: int,
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
    use_gpu_bisection: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Compute spectral ball update for single-rank (non-TP) case.

    This implements the core algorithm:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Form Θ = uv^T
    4. Solve for λ: <Θ, msign(M + λΘ)> = 0
    5. Return Φ = msign(M + λΘ)

    Returns:
        Tuple of (Phi, retract_bias, sigma_value) where retract_bias is 0.0 for hard mode
    """

    # Convert M to fp32 once at the beginning
    M_fp32 = M.to(torch.float32)
    M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2,-1), keepdim=True).clamp_min(1e-8))  # 归一化梯度

    # 1. Power iteration (returns fp32)
    sigma, u, v = power_iteration(W, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # 2. Retract W to spectral sphere
    retract_bias = apply_retract(W, sigma_value, target_radius, mode=retract_mode, alpha=retract_alpha, current_lr=current_lr)


    # 3. Form Theta (fp32)
    Theta = u @ v.transpose(-2, -1)


    # 4. Solve for lambda using selected solver
    if solver == "bisection":
        bisection_fn = solve_lambda_with_bisection
        # bisection_fn = solve_lambda_with_bisection_gpu if use_gpu_bisection else solve_lambda_with_bisection
        lambda_value, converged, residual, iterations = bisection_fn(
            G=M_fp32,
            Theta=Theta,
            initial_guess=0.0,
            initial_step=1e-3,
            tolerance_f=solver_tolerance_f,
            max_iterations=solver_max_iterations,
            max_expansions=10,
            msign_steps=msign_steps,
        )

    # 5. Compute final update direction
    Z = M_fp32 + lambda_value * Theta

    Phi = msign(Z, steps=msign_steps)

    return Phi, retract_bias, sigma_value


def _compute_tp_duplicated(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    solver: str,
    solver_tolerance_f: float,
    solver_max_iterations: int,
    tp_group: torch.distributed.ProcessGroup,
    partition_dim: int,
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
    use_gpu_bisection: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Compute spectral ball update for TP duplicated mode.

    Communication pattern (optimal):
    1. all_gather(W_shard) → W_full
    2. all_gather(M_shard) → M_full
    3. Compute on full tensors (no communication)
    4. Split Φ_full → Φ_local (local operation)

    Total: 2 all_gather operations

    Args:
        W: Weight matrix shard (modified in-place for retraction)
        M: Momentum tensor shard
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations
        solver: Solver method ('bisection')
        solver_tolerance_f: Function tolerance for solver
        solver_max_iterations: Maximum solver iterations
        tp_group: Tensor parallel process group
        partition_dim: Dimension along which tensors are partitioned

    Returns:
        Update direction Φ_local (fp32 shard)
    """
    # Gather shards to global matrices
    W_full = _tp_gather_along_dim(W, tp_group, partition_dim)
    M_full = _tp_gather_along_dim(M, tp_group, partition_dim)

    # Convert M to fp32 once
    M_full_fp32 = M_full.to(torch.float32)
    M_full_fp32 = M_full_fp32 / (torch.linalg.norm(M_full_fp32, dim=(-2,-1), keepdim=True).clamp_min(1e-8))  # 归一化梯度

    # 1. Power iteration on global W (returns fp32)
    sigma, u, v = power_iteration(W_full, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # 2. Retract global W and update local shard
    retract_bias = apply_retract(W_full, sigma_value, target_radius, mode=retract_mode, alpha=retract_alpha, current_lr=current_lr)
    # Split back to local shard and update original W
    W_local = _tp_split_along_dim(W_full, tp_group, partition_dim)
    W.copy_(W_local)

    # 3. Form Theta (fp32)
    Theta_full = u @ v.transpose(-2, -1)

    # 4. Solve for lambda on global tensors using selected solver
    if solver == "bisection":
        bisection_fn = solve_lambda_with_bisection
        lambda_value, converged, residual, iterations = bisection_fn(
            G=M_full_fp32,
            Theta=Theta_full,
            initial_guess=0.0,
            initial_step=1e-3,
            tolerance_f=solver_tolerance_f,
            max_iterations=solver_max_iterations,
            max_expansions=10,
            msign_steps=msign_steps,
        )
    if not converged:
        logging.warning(
            f"[TP] {solver.capitalize()} solver did not converge: residual={residual:.2e} "
            f"after {iterations} iterations"
        )

    # 5. Compute Φ on global tensor (no communication)
    Z_full = M_full_fp32 + lambda_value * Theta_full
    Phi_full = msign(Z_full, steps=msign_steps)

    # 6. Split back to local shard
    Phi_local = _tp_split_along_dim(Phi_full, tp_group, partition_dim)
    return Phi_local, retract_bias, sigma_value


def compute_spectral_ball_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    solver: str,
    solver_tolerance_f: float,
    solver_max_iterations: int,
    *,
    tp_group: torch.distributed.ProcessGroup | None = None,
    partition_dim: int | None = None,
    tp_mode: str = "duplicated",
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
    use_gpu_bisection: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Compute spectral ball constrained update direction (dispatcher).

    This is the main entry point that dispatches to either single-rank or
    tensor-parallel implementations based on the TP configuration.

    Algorithm overview:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Form Θ = uv^T
    4. Solve for λ: <Θ, msign(M + λΘ)> = 0
    5. Return Φ = msign(M + λΘ)

    The msign function uses Polar-Express coefficients for fast convergence.
.

    Args:
        W: Current weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations (uses Polar-Express coefficients)
        solver: Solver method ('bisection')
        solver_tolerance_f: Function tolerance for solver
        solver_max_iterations: Maximum solver iterations
        tp_group: Tensor parallel process group (None for single-rank)
        partition_dim: Dimension along which tensors are partitioned
        tp_mode: TP mode (only "duplicated" is currently supported)
        current_lr: Current learning rate (for dynamic retraction)

    Returns:
        Update direction Φ to be applied as W ← W - lr * Φ, retraction bias, and current spectral norm σ.

    Note:
        W is modified in-place during the retraction step.
    """
    # Determine if TP is enabled
    ws, _ = _tp_world_and_rank(tp_group)
    tp_enabled = tp_group is not None and partition_dim is not None and ws > 1

    if not tp_enabled:
        # Single-rank path
        return _compute_single_rank(
            W=W,
            M=M,
            target_radius=target_radius,
            power_iteration_steps=power_iteration_steps,
            msign_steps=msign_steps,
            solver=solver,
            solver_tolerance_f=solver_tolerance_f,
            solver_max_iterations=solver_max_iterations,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
            current_lr=current_lr,
            use_gpu_bisection=use_gpu_bisection,
        )
    else:
        # TP enabled: duplicated mode only
        if tp_mode != "duplicated":
            raise NotImplementedError(
                f"SpectralBall TP mode '{tp_mode}' not implemented; use 'duplicated' for now."
            )
        return _compute_tp_duplicated(
            W=W,
            M=M,
            target_radius=target_radius,
            power_iteration_steps=power_iteration_steps,
            msign_steps=msign_steps,
            solver=solver,
            solver_tolerance_f=solver_tolerance_f,
            solver_max_iterations=solver_max_iterations,
            tp_group=tp_group,
            partition_dim=partition_dim,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
            current_lr=current_lr,
            use_gpu_bisection=use_gpu_bisection,
        )