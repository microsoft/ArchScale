# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Logging utilities for distributed training.
Provides asynchronous logging infrastructure with hooks for custom metrics.
"""

import threading
import time
import atexit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Union
import multiprocessing

import torch
import lightning as L


# ============================================================================
# Configuration
# ============================================================================

# Set to True to use multiprocessing (faster for CPU-bound tasks like calc_avg_attn_span)
# Set to False to use threading (simpler, but limited by GIL)
USE_MULTIPROCESSING = True
MAX_WORKERS = 1  # Number of parallel workers

# Set to False to disable per-layer logging (reduces log volume significantly)
# When disabled, only aggregated/mean metrics are logged, not per-layer breakdowns
ENABLE_PER_LAYER_LOGGING = True
ENABLE_PER_LAYER_EXPERT_LOGGING = False

def maxvio_batch(
    tokens_per_expert: torch.Tensor,  # (E,) or (B, E)
) -> torch.Tensor:
    """
    Compute MaxVio (maximal violation) metric for load balance measurement.
    
    MaxVio quantifies the degree of load imbalance by measuring how much
    the most loaded expert exceeds the expected balanced load:
    
        MaxVio = (max_i Load_i - Load̄) / Load̄
    
    where Load_i is the number of tokens assigned to expert i, and Load̄ is
    the expected load under perfect balance (total_tokens * top_k / n_experts).
    
    For MaxVio_batch, we count Load_i on each training batch, so it reflects
    training efficiency. A value of 0 means perfect balance.
    
    Args:
        tokens_per_expert: (E,) tensor of token counts per expert for the batch,
                          or (B, E) for per-sequence counts.
        top_k: Number of experts activated per token (for computing expected load).
    
    Returns:
        Scalar tensor: MaxVio value for the batch.
    
    Example:
        >>> tokens_per_expert = torch.tensor([120, 100, 80, 100])  # 4 experts
        >>> # Total tokens dispatched = 400, with top_k=1: expected = 400/4 = 100
        >>> # max load = 120, so MaxVio = (120 - 100) / 100 = 0.2
        >>> maxvio = maxvio_batch(tokens_per_expert, top_k=1)
    """
    if tokens_per_expert.dim() == 2:
        # (B, E) - average over batch dimension first
        tokens_per_expert = tokens_per_expert.float().mean(dim=0)
    
    tokens_per_expert = tokens_per_expert.float()
    n_experts = tokens_per_expert.shape[0]
    
    # Total tokens dispatched (each token is routed to top_k experts)
    total_dispatched = tokens_per_expert.sum()
    
    # Expected load under perfect balance
    # Each expert should receive total_dispatched / n_experts tokens
    expected_load = total_dispatched / n_experts
    
    # Max load among all experts
    max_load = tokens_per_expert.max()
    
    # MaxVio = (max_load - expected_load) / expected_load
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    maxvio = (max_load - expected_load) / (expected_load + eps)
    
    return maxvio
    

def configure_logging_executor(use_multiprocessing: bool = True, max_workers: int = 1):
    """
    Configure the logging executor.
    
    Args:
        use_multiprocessing: If True, use ProcessPoolExecutor for true parallelism.
                           If False, use ThreadPoolExecutor (limited by GIL).
        max_workers: Number of parallel workers.
    
    Note: Must be called before any logging tasks are submitted.
          For long sequences, multiprocessing can be 2-4x faster for calc_avg_attn_span.
    """
    global USE_MULTIPROCESSING, MAX_WORKERS, _logging_executor
    USE_MULTIPROCESSING = use_multiprocessing
    MAX_WORKERS = max_workers
    
    # Shutdown existing executor if any
    if '_logging_executor' in globals():
        _logging_executor.shutdown(wait=False)
    
    # Create new executor
    if USE_MULTIPROCESSING:
        _logging_executor = ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            mp_context=multiprocessing.get_context('spawn')
        )
    else:
        _logging_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# ============================================================================
# Dataclasses and Type Definitions
# ============================================================================

@dataclass
class LoggingPayload:
    """Payload containing all data needed for asynchronous logging."""
    iter_num: int
    step_count: int
    grad_norm: float
    param_norms: Dict[str, float]
    budget_loss_value: Optional[float]
    lc_mean_activation: Optional[float]
    acc_seq: Optional[float]
    bal_loss: Optional[float]
    optimizer_stats: Dict[str, Union[float, Tuple[float, float]]]
    hybrid_optimizer: bool
    # Training configuration
    micro_batch_size: int
    devices: int
    nodes: int
    seq_len: int
    grad_clip: float
    vector_names: List[str]
    eval_step_interval: int
    # Model-specific data
    tokens_per_expert: Optional[List[List[int]]] = None
    token_indices: Optional[List[torch.Tensor]] = None
    n_attn: Optional[List[int]] = None
    n_mlp: Optional[List[int]] = None
    local_window: Optional[int] = None
    lc_gate_hist: Optional[List[torch.Tensor]] = None
    batch_info: Dict[str, Union[int, float, bool, None]] = field(default_factory=dict)
    # Attention and MLP/MoE output RMSNorms per layer
    attn_output_rmsnorms: Optional[List[float]] = None
    mlp_output_rmsnorms: Optional[List[float]] = None
    # Outlier percentages (5-sigma) per layer
    attn_outlier_pcts: Optional[List[float]] = None
    mlp_outlier_pcts: Optional[List[float]] = None
    # Attention and Router z-values (LSE²) per layer
    attn_z_means: Optional[List[Optional[float]]] = None
    router_z_means: Optional[List[Optional[float]]] = None
    # Weight update data for async RMS computation in hook
    # Dict of param_name -> {old_weight, new_weight, category, lr, weight_decay}
    weight_update_data: Optional[Dict[str, Dict]] = None


@dataclass
class LoggingResult:
    """Result from asynchronous logging task."""
    step_count: int
    iter_num: int
    log_dict: Dict[str, float]
    print_lines: List[str] = field(default_factory=list)


LoggingHook = Callable[[LoggingPayload, Dict[str, float], List[str]], None]


# ============================================================================
# Global Logging Infrastructure
# ============================================================================

logging_hooks: List[LoggingHook] = []

# Choose executor based on configuration
if USE_MULTIPROCESSING:
    # For multiprocessing, we need to ensure tensors are on CPU and properly serialized
    # This is better for CPU-bound tasks like calc_avg_attn_span
    _logging_executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=multiprocessing.get_context('spawn')  # Use 'spawn' for CUDA safety
    )
else:
    # Threading is simpler but limited by GIL for CPU-bound tasks
    _logging_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

_logging_results: Queue = Queue()
_pending_logging_tasks = 0
_pending_logging_lock = threading.Lock()


def register_logging_hook(hook: LoggingHook) -> LoggingHook:
    """Register a logging hook that can augment the metric dictionary."""
    logging_hooks.append(hook)
    return hook


def _logging_callback(future):
    """Callback to handle completed logging tasks."""
    global _pending_logging_tasks
    try:
        result = future.result()
    except Exception as exc:
        result = exc
    _logging_results.put(result)
    with _pending_logging_lock:
        _pending_logging_tasks -= 1


atexit.register(_logging_executor.shutdown, wait=True)


# ============================================================================
# Task Submission and Queue Management
# ============================================================================

def submit_logging_task(payload: LoggingPayload):
    """Submit a logging payload to the background executor."""
    global _pending_logging_tasks
    with _pending_logging_lock:
        _pending_logging_tasks += 1
    future = _logging_executor.submit(logging_train, payload)
    future.add_done_callback(_logging_callback)


def drain_logging_queue(fabric: L.Fabric):
    """Flush completed logging tasks and emit their outputs."""
    while True:
        try:
            result = _logging_results.get_nowait()
        except Empty:
            break
        if isinstance(result, Exception):
            raise result
        for line in result.print_lines:
            fabric.print(line)
        fabric.log_dict(result.log_dict, result.step_count)


def wait_for_logging_tasks(fabric: L.Fabric):
    """Block until all pending logging tasks are drained."""
    while True:
        drain_logging_queue(fabric)
        with _pending_logging_lock:
            remaining = _pending_logging_tasks
        if remaining == 0 and _logging_results.empty():
            break
        time.sleep(0.01)


# ============================================================================
# Helper Functions
# ============================================================================

def get_grad_norm(model):
    """Compute gradient norm for logging"""
    total_norm = 0.0
    param_norms = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            # Store individual parameter gradient norms
            param_norms[n] = param_norm
    total_norm = total_norm ** 0.5
    return total_norm, param_norms


# Parameter name patterns for categorizing weights
ATTN_PARAM_PATTERNS = ["attn", ".proj.",]
MLP_MOE_PARAM_PATTERNS = ["w1", "w2", "w3", "mods_in", "mods_out", "shared_mod"]
ROUTER_PARAM_PATTERNS = ["router", "lc_proj"]


def save_weights_for_update_logging(model, vector_names: List[str]) -> Dict[str, Dict]:
    """
    Save 2D attention, MLP/MoE, and router weights to CPU (non-blocking) before optimizer.step().
    Call this BEFORE optimizer.step().
    
    Returns dict of param_name -> {old_weight_cpu, category}
    """
    saved_weights = {}
    
    for n, p in model.named_parameters():
        name_lower = n.lower()
        
        # Only save 2D weight parameters (matrices, not vectors)
        if p.dim() != 2:
            continue
        
        # Skip biases, norms, etc.
        if "weight" not in name_lower or any(vn in name_lower for vn in vector_names):
            continue
        
        # Categorize: router, attention, or MLP/MoE
        is_router = any(pattern in name_lower for pattern in ROUTER_PARAM_PATTERNS)
        is_attn = any(pattern in name_lower for pattern in ATTN_PARAM_PATTERNS)
        is_mlp = any(pattern in name_lower for pattern in MLP_MOE_PARAM_PATTERNS)
        
        if not is_attn and not is_mlp and not is_router:
            continue
        
        # Determine category (router takes priority)
        if is_router:
            category = "router"
        elif is_attn:
            category = "attn"
        else:
            category = "mlp"
        
        # Save to CPU with non_blocking=True (async transfer)
        saved_weights[n] = {
            "old_weight": p.data.detach().to('cpu', non_blocking=True),
            "category": category,
        }
    
    return saved_weights


def capture_weight_update_data(
    model,
    saved_weights: Dict[str, Dict],
    optimizer,
    hybrid_optimizer: bool,
) -> Dict[str, Dict]:
    """
    Capture new weights and optimizer info after optimizer.step() for async RMS computation.
    Call this AFTER optimizer.step().
    
    Returns dict of param_name -> {old_weight, new_weight, category, lr, weight_decay} (all on CPU, non-blocking)
    """
    # Handle hybrid optimizer (list of optimizers)
    if hybrid_optimizer:
        all_param_groups = []
        for opt in optimizer:
            all_param_groups.extend(opt.param_groups)
    else:
        all_param_groups = optimizer.param_groups
    
    # Build param -> group mapping for lr and weight_decay
    param_to_group = {}
    for group in all_param_groups:
        for p in group["params"]:
            param_to_group[id(p)] = group
    
    weight_update_data = {}
    
    for n, p in model.named_parameters():
        if n not in saved_weights:
            continue
        
        # Get optimizer hyperparameters for this param
        group = param_to_group.get(id(p))
        if group is None:
            continue
        
        lr = group.get("lr", 1e-4)
        weight_decay = group.get("weight_decay", 0.0)
        
        # Copy new weight to CPU with non_blocking=True (async transfer)
        weight_update_data[n] = {
            "old_weight": saved_weights[n]["old_weight"],
            "new_weight": p.data.detach().to('cpu', non_blocking=True),
            "category": saved_weights[n]["category"],
            "lr": lr,
            "weight_decay": weight_decay,
        }
    
    return weight_update_data


def _compute_rmsnorm_from_norms(norms: List[float]) -> float:
    """Compute RMSNorm from a list of values (sqrt of mean of squared values)."""
    if not norms:
        return 0.0
    import math
    squared_sum = sum(n ** 2 for n in norms)
    return math.sqrt(squared_sum / len(norms))


def calc_avg_attn_span(a, win, causal=False):
    """Calculate average attention span for token indices."""
    seq_len = a.shape[-1]
    pad_mask = (a != -1)
    wind_mask = torch.ones([seq_len, seq_len]).to(a).bool()
    wind_mask = ~(torch.triu(wind_mask, diagonal=win+1) + torch.tril(wind_mask, diagonal=-win-1))
    q = a.unsqueeze(-1)
    k = a.unsqueeze(-2)
    dist = (q - k).abs()
    dist *= pad_mask.unsqueeze(-1)
    dist *= pad_mask.unsqueeze(-2)
    if causal:
        dist = torch.tril(dist)
    dist *= wind_mask
    avg_attn_span = dist.sum(-1) / ((dist != 0).sum(-1) + 1e-8)
    return avg_attn_span.mean()


# ============================================================================
# Logging Payload Building
# ============================================================================

def build_logging_payload(
    model,
    state,
    budget_loss_value,
    lc_mean_activation,
    acc_seq,
    bal_loss,
    optimizer,
    current_grad_accum_steps,
    batch_multiplier,
    batch_rampup,
    current_batch_ratio,
    is_accumulating,
    hybrid_optimizer,
    micro_batch_size: int,
    devices: int,
    nodes: int,
    seq_len: int,
    grad_clip: float,
    vector_names: List[str],
    eval_step_interval: int,
):
    """Capture the model state required for asynchronous logging."""
    grad_norm, param_norms = get_grad_norm(model)
    # Get weight update data for async RMS computation (captured after optimizer.step() in previous iteration)
    weight_update_data = getattr(model, "_weight_update_data", None)
    budget = budget_loss_value.item() if budget_loss_value is not None else None
    lc_activation = lc_mean_activation.item() if lc_mean_activation is not None else None
    acc_value = acc_seq.item() if acc_seq is not None else None
    bal_value = bal_loss.item() if bal_loss is not None else None

    if hybrid_optimizer:
        vector_group = optimizer[0].param_groups[0]
        weight_group = optimizer[1].param_groups[0]
        optimizer_stats = {
            "weight_lr": weight_group["lr"],
            "weight_decay": weight_group.get("weight_decay", 0.0),
            "vector_lr": vector_group["lr"],
            "betas": vector_group.get("betas"),
            "eps": vector_group.get("eps"),
        }
        # Non-embed vector lr (depth-scaled) when 3 param groups in the vectors optimizer
        if len(optimizer[0].param_groups) == 2:
            optimizer_stats["non_embed_vector_lr"] = optimizer[0].param_groups[1]["lr"]
        rampup_group = weight_group
    else:
        if len(optimizer.param_groups) > 1:
            vector_group = optimizer.param_groups[0]
            weight_group = optimizer.param_groups[-1]  # Last group is always weights
        else:
            vector_group = optimizer.param_groups[0]
            weight_group = optimizer.param_groups[0]
        optimizer_stats = {
            "weight_lr": weight_group["lr"],
            "weight_decay": weight_group.get("weight_decay", 0.0),
            "vector_lr": vector_group["lr"],
            "betas": weight_group.get("betas"),
            "eps": weight_group.get("eps"),
        }
        # Non-embed vector lr (depth-scaled) when 3 param groups
        if len(optimizer.param_groups) == 3:
            optimizer_stats["non_embed_vector_lr"] = optimizer.param_groups[1]["lr"]
        rampup_group = weight_group

    batch_info = {
        "batch_rampup": batch_rampup,
        "is_accumulating": is_accumulating,
        "current_grad_accum_steps": current_grad_accum_steps,
        "batch_multiplier": batch_multiplier,
        "current_batch_ratio": current_batch_ratio,
    }
    if batch_rampup and not is_accumulating:
        batch_info.update({
            "current_eps": rampup_group.get("eps"),
            "current_beta2": rampup_group.get("betas", (None, None))[1] if rampup_group.get("betas") else None,
            "current_weight_decay": rampup_group.get("weight_decay", 0.0),
        })

    tokens_per_expert = None
    token_indices = None
    n_attn = getattr(model, "n_attn", None)
    n_mlp = getattr(model, "n_mlp", None)
    local_window = getattr(model.config, "local_window", None)
    if hasattr(model, "tokens_per_expert"):
        tokens_per_expert = []
        for tokens in model.tokens_per_expert:
            if torch.is_tensor(tokens):
                tokens_list = [int(x) for x in tokens.detach().cpu().tolist()]
            else:
                tokens_list = [int(x) for x in tokens]
            tokens_per_expert.append(tokens_list)
        if hasattr(model, "token_indices"):
            token_indices = []
            for indices in model.token_indices:
                # Move to CPU for multiprocessing compatibility
                token_indices.append(indices.detach().cpu())

    lc_gate_hist = None
    if hasattr(model, "lc_gate_hist") and len(model.lc_gate_hist) > 0:
        lc_gate_hist = [gate.detach().cpu() for gate in model.lc_gate_hist]

    # Capture attention and MLP/MoE output RMSNorms
    attn_output_rmsnorms = None
    mlp_output_rmsnorms = None
    if hasattr(model, "attn_output_rmsnorms") and len(model.attn_output_rmsnorms) > 0:
        attn_output_rmsnorms = list(float(x.detach().cpu().item()) for x in model.attn_output_rmsnorms)
    if hasattr(model, "mlp_output_rmsnorms") and len(model.mlp_output_rmsnorms) > 0:
        mlp_output_rmsnorms = list(float(x.detach().cpu().item()) for x in model.mlp_output_rmsnorms)

    # Capture outlier percentages (5-sigma)
    attn_outlier_pcts = None
    mlp_outlier_pcts = None
    if hasattr(model, "attn_outlier_pcts") and len(model.attn_outlier_pcts) > 0:
        attn_outlier_pcts = list(float(x.detach().cpu().item()) for x in model.attn_outlier_pcts)
    if hasattr(model, "mlp_outlier_pcts") and len(model.mlp_outlier_pcts) > 0:
        mlp_outlier_pcts = list(float(x.detach().cpu().item()) for x in model.mlp_outlier_pcts)

    # Capture attention and router z-values (LSE²)
    attn_z_means = None
    router_z_means = None
    if hasattr(model, "attn_z_means") and len(model.attn_z_means) > 0:
        attn_z_means = [float(x.detach().cpu().item()) if x is not None else None for x in model.attn_z_means]
    if hasattr(model, "router_z_means") and len(model.router_z_means) > 0:
        router_z_means = [float(x.detach().cpu().item()) if x is not None else None for x in model.router_z_means]

    return LoggingPayload(
        iter_num=state["iter_num"],
        step_count=state["step_count"],
        grad_norm=grad_norm,
        param_norms=param_norms,
        budget_loss_value=budget,
        lc_mean_activation=lc_activation,
        acc_seq=acc_value,
        bal_loss=bal_value,
        optimizer_stats=optimizer_stats,
        hybrid_optimizer=hybrid_optimizer,
        micro_batch_size=micro_batch_size,
        devices=devices,
        nodes=nodes,
        seq_len=seq_len,
        grad_clip=grad_clip,
        vector_names=vector_names,
        eval_step_interval=eval_step_interval,
        tokens_per_expert=tokens_per_expert,
        token_indices=token_indices,
        n_attn=n_attn,
        n_mlp=n_mlp,
        local_window=local_window,
        lc_gate_hist=lc_gate_hist,
        batch_info=batch_info,
        attn_output_rmsnorms=attn_output_rmsnorms,
        mlp_output_rmsnorms=mlp_output_rmsnorms,
        attn_outlier_pcts=attn_outlier_pcts,
        mlp_outlier_pcts=mlp_outlier_pcts,
        attn_z_means=attn_z_means,
        router_z_means=router_z_means,
        weight_update_data=weight_update_data,
    )


# ============================================================================
# Main Logging Function
# ============================================================================

def logging_train(payload: LoggingPayload) -> LoggingResult:
    """Build the logging dictionary off the training thread."""
    log_dict = {
        "metric/grad_norm": payload.grad_norm,
        "metric/grad_norm_clip_ratio": payload.grad_norm / payload.grad_clip if payload.grad_clip > 0 else float("inf"),
    }
    print_lines: List[str] = []

    for hook in logging_hooks:
        hook(payload, log_dict, print_lines)

    return LoggingResult(
        step_count=payload.step_count,
        iter_num=payload.iter_num,
        log_dict=log_dict,
        print_lines=print_lines,
    )


# ============================================================================
# Logging Hooks
# ============================================================================

@register_logging_hook
def _core_metric_hook(payload: LoggingPayload, log_dict: Dict[str, float], print_lines: List[str]) -> None:
    """Log core metrics like gradients, learning rates, and optimizer stats."""
    if payload.budget_loss_value is not None:
        log_dict.update({
            "metric/lc_budget_loss": payload.budget_loss_value,
            "metric/lc_mean_activation": payload.lc_mean_activation,
        })
    if payload.bal_loss is not None:
        log_dict["metric/bal_loss"] = payload.bal_loss
    if payload.acc_seq is not None:
        log_dict["metric/acc_seq"] = payload.acc_seq

    weight_grad_norm = 0.0
    vector_grad_norm = 0.0
    for name, norm in payload.param_norms.items():
        if "weight" in name.lower() and not any(nd in name.lower() for nd in payload.vector_names):
            weight_grad_norm += norm ** 2
        else:
            vector_grad_norm += norm ** 2

    optimizer_stats = payload.optimizer_stats or {}
    betas = optimizer_stats.get("betas")
    eps = optimizer_stats.get("eps")
    if betas is not None:
        log_dict["metric/optimizer_beta1"] = betas[0]
        log_dict["metric/optimizer_beta2"] = betas[1]
    if eps is not None:
        log_dict["metric/optimizer_eps"] = eps

    weight_lr = optimizer_stats.get("weight_lr", 0.0)
    vector_lr = optimizer_stats.get("vector_lr", 0.0)
    weight_decay = optimizer_stats.get("weight_decay", 0.0)

    log_dict.update({
        "metric/weight_grad_norm": weight_grad_norm ** 0.5,
        "metric/vector_grad_norm": vector_grad_norm ** 0.5,
        "metric/intrinsic_learning_rate": weight_decay * weight_lr,
        "metric/weight_learning_rate": weight_lr,
        "metric/vector_learning_rate": vector_lr,
    })

    non_embed_vector_lr = optimizer_stats.get("non_embed_vector_lr")
    if non_embed_vector_lr is not None:
        log_dict["metric/non_embed_vector_learning_rate"] = non_embed_vector_lr


@register_logging_hook
def _adagpt_metric_hook(payload: LoggingPayload, log_dict: Dict[str, float], print_lines: List[str]) -> None:
    """Log AdaGPT-specific metrics like expert load and attention span."""
    if payload.tokens_per_expert is None or payload.n_attn is None or payload.n_mlp is None:
        return
    
    all_maxvio = []
    for i, n_tokens in enumerate(payload.tokens_per_expert):
        total_tokens = sum(n_tokens)
        if total_tokens == 0:
            continue
        attn_tokens = sum(n_tokens[:payload.n_attn[i]])
        mlp_tokens = sum(n_tokens[payload.n_attn[i]: payload.n_attn[i] + payload.n_mlp[i]])
        skip_tokens = sum(n_tokens[payload.n_attn[i] + payload.n_mlp[i]:])
        
        if ENABLE_PER_LAYER_EXPERT_LOGGING:
            log_dict.update({
                f"metric/loop_{i}_attn_load": attn_tokens / total_tokens,
                f"metric/loop_{i}_mlp_load": mlp_tokens / total_tokens,
                f"metric/loop_{i}_skip_load": skip_tokens / total_tokens,
            })
            # Only do expensive attention span logging at validation intervals
            if payload.token_indices is not None and payload.local_window is not None:
                offset = 0
                if i == len(payload.tokens_per_expert) // 2:  # only do middle layer for faster logging
                    for k in range(payload.n_attn[i]):
                        if k > 0:  # only log the first attention expert so it is faster
                            break
                        attn_t = n_tokens[k]
                        if attn_t == 0:
                            continue
                        attn_indices = payload.token_indices[i][offset: offset + attn_t]
                        avg_attn_span = calc_avg_attn_span(attn_indices, payload.local_window, causal=True)
                        log_dict[f"metric/loop_{i}_avg_attn_span_{k}"] = avg_attn_span.cpu().item()
                        offset += attn_t
            for j, exp_tokens in enumerate(n_tokens):
                log_dict[f"metric/loop_{i}_expert_{j}_load"] = exp_tokens / total_tokens
        
        # Compute MaxVio_batch for this layer
        tokens_tensor = torch.tensor(n_tokens, dtype=torch.float32)
        layer_maxvio = maxvio_batch(tokens_tensor).item()
        if ENABLE_PER_LAYER_LOGGING:
            log_dict[f"metric/loop_{i}_maxvio_batch"] = layer_maxvio
        all_maxvio.append(layer_maxvio)
    
    # Log average MaxVio across all layers
    if all_maxvio:
        log_dict["metric/maxvio_batch_avg"] = sum(all_maxvio) / len(all_maxvio)


@register_logging_hook
def _lc_sparsity_hook(payload: LoggingPayload, log_dict: Dict[str, float], print_lines: List[str]) -> None:
    """Log layer-wise sparsity metrics for LC (learned compression) models."""
    if not payload.lc_gate_hist:
        return
    all_spars = []
    for i, gate in enumerate(payload.lc_gate_hist):
        spar = (gate < 0.5).float().mean().item()
        all_spars.append(spar)
        if ENABLE_PER_LAYER_LOGGING:
            metric_name = f"metric/train_sparsity_layer_{i}"
            log_dict[metric_name] = spar
            print_lines.append(f"step {payload.iter_num}: {metric_name} {spar:.4f}")
    mean_sparsity = sum(all_spars) / len(all_spars)
    log_dict["metric/train_sparsity_mean"] = mean_sparsity
    print_lines.append(f"step {payload.iter_num}: metric/train_sparsity_mean {mean_sparsity:.4f}")


@register_logging_hook
def _batch_ramp_metrics_hook(payload: LoggingPayload, log_dict: Dict[str, float], print_lines: List[str]) -> None:
    """Log batch size rampup metrics."""
    batch_info = payload.batch_info
    if not batch_info.get("batch_rampup") or batch_info.get("is_accumulating"):
        return
    
    current_grad_accum_steps = int(batch_info.get("current_grad_accum_steps", 1))
    effective_batch_size = payload.micro_batch_size * current_grad_accum_steps * payload.devices * payload.nodes
    effective_batch_tokens = effective_batch_size * payload.seq_len
    log_dict.update({
        "metric/grad_accum_steps": current_grad_accum_steps,
        "metric/effective_batch_size": effective_batch_tokens,
        "metric/batch_multiplier": batch_info.get("batch_multiplier"),
        "metric/batch_ratio": batch_info.get("current_batch_ratio"),
        "metric/optimizer_eps": batch_info.get("current_eps"),
        "metric/optimizer_beta2": batch_info.get("current_beta2"),
        "metric/optimizer_weight_decay": batch_info.get("current_weight_decay"),
    })


@register_logging_hook
def _attn_mlp_update_output_rmsnorm_hook(payload: LoggingPayload, log_dict: Dict[str, float], print_lines: List[str]) -> None:
    """Log attention, MLP/MoE, and router weight update RMSNorms (before LR) and output RMSNorms."""
    
    # Compute weight update RMSNorms from captured data (async - CPU transfers already complete)
    if payload.weight_update_data:
        attn_update_rmsnorms = []
        mlp_update_rmsnorms = []
        router_update_rmsnorms = []
        
        for name, data in payload.weight_update_data.items():
            old_weight = data["old_weight"]
            new_weight = data["new_weight"]
            category = data["category"]
            lr = data["lr"]
            weight_decay = data["weight_decay"]
            
            # delta = new_weight - old_weight
            delta = new_weight - old_weight
            
            # For AdamW: delta = -lr * (adam_update + weight_decay * old_weight)
            # So: adam_update = -delta/lr - weight_decay * old_weight
            adam_update = -delta / lr - weight_decay * old_weight
            
            # Compute RMSNorm of this update
            update_rmsnorm = torch.sqrt(torch.mean(adam_update.to_local().float() ** 2) + 1e-30).item()
            
            if category == "attn":
                attn_update_rmsnorms.append(update_rmsnorm)
            elif category == "router":
                router_update_rmsnorms.append(update_rmsnorm)
            else:  # mlp
                mlp_update_rmsnorms.append(update_rmsnorm)
        
        # Log weight update RMSNorms
        if attn_update_rmsnorms:
            log_dict["metric/attn_weight_update_rmsnorm"] = _compute_rmsnorm_from_norms(attn_update_rmsnorms)
        if mlp_update_rmsnorms:
            log_dict["metric/mlp_weight_update_rmsnorm"] = _compute_rmsnorm_from_norms(mlp_update_rmsnorms)
        if router_update_rmsnorms:
            log_dict["metric/router_weight_update_rmsnorm"] = _compute_rmsnorm_from_norms(router_update_rmsnorms)
    
    # Log attention output RMSNorms per layer and RMS mean
    if payload.attn_output_rmsnorms:
        if ENABLE_PER_LAYER_LOGGING:
            for i, rmsnorm in enumerate(payload.attn_output_rmsnorms):
                log_dict[f"metric/attn_output_rmsnorm_layer_{i}"] = rmsnorm
        rms_attn_rmsnorm = _compute_rmsnorm_from_norms(payload.attn_output_rmsnorms)
        log_dict["metric/attn_output_rmsnorm_mean"] = rms_attn_rmsnorm
    
    # Log MLP/MoE output RMSNorms per layer and RMS mean
    if payload.mlp_output_rmsnorms:
        if ENABLE_PER_LAYER_LOGGING:
            for i, rmsnorm in enumerate(payload.mlp_output_rmsnorms):
                log_dict[f"metric/mlp_output_rmsnorm_layer_{i}"] = rmsnorm
        rms_mlp_rmsnorm = _compute_rmsnorm_from_norms(payload.mlp_output_rmsnorms)
        log_dict["metric/mlp_output_rmsnorm_mean"] = rms_mlp_rmsnorm

    # Log attention output outlier percentages (5-sigma) per layer and mean
    if payload.attn_outlier_pcts:
        if ENABLE_PER_LAYER_LOGGING:
            for i, pct in enumerate(payload.attn_outlier_pcts):
                log_dict[f"metric/attn_outlier_pct_layer_{i}"] = pct
        log_dict["metric/attn_outlier_pct_mean"] = sum(payload.attn_outlier_pcts) / len(payload.attn_outlier_pcts)

    # Log MLP/MoE output outlier percentages (5-sigma) per layer and mean
    if payload.mlp_outlier_pcts:
        if ENABLE_PER_LAYER_LOGGING:
            for i, pct in enumerate(payload.mlp_outlier_pcts):
                log_dict[f"metric/mlp_outlier_pct_layer_{i}"] = pct
        log_dict["metric/mlp_outlier_pct_mean"] = sum(payload.mlp_outlier_pcts) / len(payload.mlp_outlier_pcts)

    # Log attention z-value (LSE²) per layer and mean (None entries for non-attention layers are skipped)
    if payload.attn_z_means:
        valid_z = [v for v in payload.attn_z_means if v is not None]
        if valid_z:
            if ENABLE_PER_LAYER_LOGGING:
                for i, z in enumerate(payload.attn_z_means):
                    if z is not None:
                        log_dict[f"metric/attn_z_layer_{i}"] = z
            log_dict["metric/attn_z_mean"] = sum(valid_z) / len(valid_z)

    # Log router z-value (LSE²) per layer and mean (None entries for non-MoE layers are skipped)
    if payload.router_z_means:
        valid_z = [v for v in payload.router_z_means if v is not None]
        if valid_z:
            if ENABLE_PER_LAYER_LOGGING:
                for i, z in enumerate(payload.router_z_means):
                    if z is not None:
                        log_dict[f"metric/router_z_layer_{i}"] = z
            log_dict["metric/router_z_mean"] = sum(valid_z) / len(valid_z)

