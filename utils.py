import blobfile as bf
from io import BytesIO
from pathlib import Path
import lightning as L
import tempfile
import subprocess
from typing import Optional
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import torch.nn as nn
import math
import torch
import torch.distributed as dist
import torch.nn.init as init
from torch.distributed.tensor import DTensor, Shard, Replicate

# PyTorch default initialization
torch_default_init = partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


@torch.no_grad()
def diagonal_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Initialize a 2D tensor as a (scaled) identity/diagonal matrix.
    
    For square matrices, this creates an identity matrix scaled by `gain`.
    For non-square matrices, the diagonal of the smaller dimension is set to `gain`,
    and all other elements are zero.
    
    This initialization helps preserve information flow at the start of training,
    making the initial network behave closer to a residual identity mapping.
    
    Args:
        tensor: The 2D tensor to initialize (must have ndim == 2).
        gain: Multiplicative factor to apply to the diagonal elements.
        
    Returns:
        The initialized tensor (modified in-place).
    """
    if tensor.ndim != 2:
        raise ValueError(f"diagonal_init_ expects a 2D tensor, got {tensor.ndim}D")
    
    rows, cols = tensor.shape
    nn.init.zeros_(tensor)
    
    # Fill the diagonal with gain
    min_dim = min(rows, cols)
    tensor.diagonal()[:min_dim].fill_(gain)
    
    return tensor


@torch.no_grad()
def global_diagonal_init_dtensor_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Perform global diagonal initialization for DTensors.
    
    For sharded DTensors, this redistributes to replicated, performs diagonal
    initialization on the full tensor, then redistributes back to the original sharding.
    For regular tensors or replicated DTensors, uses standard diagonal initialization.
    
    Args:
        tensor: The tensor to initialize (can be a regular Tensor or DTensor).
        gain: Multiplicative factor to apply to the diagonal elements.
        
    Returns:
        The initialized tensor (modified in-place).
    """
    if not isinstance(tensor, DTensor):
        # Regular tensor: use standard diagonal initialization
        return diagonal_init_(tensor, gain=gain)
    
    # Get DTensor metadata
    original_placements = tensor.placements
    
    # Check if any dimension is sharded
    has_shard = any(isinstance(p, Shard) for p in original_placements)
    
    if not has_shard:
        # Already replicated: use standard diagonal initialization on local tensor
        diagonal_init_(tensor.to_local(), gain=gain)
        return tensor
    
    # Create replicated placements (same length as original)
    replicated_placements = [Replicate() for _ in original_placements]
    
    # Redistribute to replicated so every rank holds the full tensor
    tensor_rep = tensor.redistribute(placements=replicated_placements)
    
    # Perform diagonal initialization on the full local tensor
    diagonal_init_(tensor_rep.to_local(), gain=gain)
    
    # Redistribute back to original sharded placements
    tensor_sharded = tensor_rep.redistribute(placements=original_placements)
    
    # Copy back to original tensor's local storage
    tensor.to_local().copy_(tensor_sharded.to_local())
    
    return tensor


@torch.no_grad()
def global_orthogonal_init_dtensor_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Perform global orthogonal initialization for DTensors.
    
    For sharded DTensors, this redistributes to replicated, performs orthogonal
    initialization on the full tensor, then redistributes back to the original sharding.
    For regular tensors or replicated DTensors, uses standard orthogonal initialization.
    
    Args:
        tensor: The tensor to initialize (can be a regular Tensor or DTensor).
        gain: Multiplicative factor to apply to the orthogonal matrix.
        
    Returns:
        The initialized tensor (modified in-place).
    """
    if not isinstance(tensor, DTensor):
        # Regular tensor: use standard orthogonal initialization
        return init.orthogonal_(tensor, gain=gain)
    
    # Get DTensor metadata
    device_mesh = tensor.device_mesh
    original_placements = tensor.placements
    
    # Check if any dimension is sharded
    has_shard = any(isinstance(p, Shard) for p in original_placements)
    
    if not has_shard:
        # Already replicated: use standard orthogonal initialization on local tensor
        init.orthogonal_(tensor.to_local(), gain=gain)
        return tensor
    
    # Create replicated placements (same length as original)
    replicated_placements = [Replicate() for _ in original_placements]
    
    # Redistribute to replicated so every rank holds the full tensor
    tensor_rep = tensor.redistribute(placements=replicated_placements)
    
    # Perform orthogonal initialization on the full local tensor
    init.orthogonal_(tensor_rep.to_local(), gain=gain)
    
    # Redistribute back to original sharded placements
    tensor_sharded = tensor_rep.redistribute(placements=original_placements)
    
    # Copy back to original tensor's local storage
    tensor.to_local().copy_(tensor_sharded.to_local())
    
    return tensor


@torch.no_grad()
def max_fan_normal_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Initialize a tensor with normal distribution using std = gain / sqrt(max(fan_in, fan_out)).
    
    This initialization scheme scales inversely with the square root of the larger dimension,
    which can help maintain stable gradients in both forward and backward passes.
    
    Args:
        tensor: The tensor to initialize (must have at least 2D for fan calculation).
        gain: Multiplicative factor for the standard deviation.
        
    Returns:
        The initialized tensor (modified in-place).
    """
    if tensor.ndim < 2:
        raise ValueError(f"max_fan_normal_init_ expects at least a 2D tensor, got {tensor.ndim}D")
    
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain / math.sqrt(max(fan_in, fan_out))
    
    return nn.init.normal_(tensor, mean=0.0, std=std)


@torch.no_grad()
def global_max_fan_normal_init_dtensor_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Perform global max-fan normal initialization for DTensors.
    
    For sharded DTensors, this redistributes to replicated, performs max-fan normal
    initialization on the full tensor, then redistributes back to the original sharding.
    For regular tensors or replicated DTensors, uses standard max-fan normal initialization.
    
    Args:
        tensor: The tensor to initialize (can be a regular Tensor or DTensor).
        gain: Multiplicative factor for the standard deviation.
        
    Returns:
        The initialized tensor (modified in-place).
    """
    if not isinstance(tensor, DTensor):
        # Regular tensor: use standard max-fan normal initialization
        return max_fan_normal_init_(tensor, gain=gain)
    
    # Get DTensor metadata
    original_placements = tensor.placements
    
    # Check if any dimension is sharded
    has_shard = any(isinstance(p, Shard) for p in original_placements)
    
    if not has_shard:
        # Already replicated: use standard max-fan normal initialization on local tensor
        max_fan_normal_init_(tensor.to_local(), gain=gain)
        return tensor
    
    # Create replicated placements (same length as original)
    replicated_placements = [Replicate() for _ in original_placements]
    
    # Redistribute to replicated so every rank holds the full tensor
    tensor_rep = tensor.redistribute(placements=replicated_placements)
    
    # Perform max-fan normal initialization on the full local tensor
    max_fan_normal_init_(tensor_rep.to_local(), gain=gain)
    
    # Redistribute back to original sharded placements
    tensor_sharded = tensor_rep.redistribute(placements=original_placements)
    
    # Copy back to original tensor's local storage
    tensor.to_local().copy_(tensor_sharded.to_local())
    
    return tensor


REMOTE_SCHEMES = ("az://", "gs://", "s3://")

CACHE_DIR = "/tmp/data_cache"

def _ensure_local_path(src_path, concurrency=8):
    """Ensure a local copy exists for remote paths; return local path as string."""
    if _is_remote(src_path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        base = os.path.basename(src_path.rstrip("/"))
        digest = hashlib.md5(src_path.encode()).hexdigest()[:16]
        local_path = os.path.join(CACHE_DIR, f"{digest}-{base}")
        if not os.path.exists(local_path):
            _copy_to_remote(src_path, local_path, concurrency=concurrency)
        return local_path
    return src_path

def _parallel_ensure_local_paths(paths, max_workers):
    if not paths:
        return []

    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_ensure_local_path, p): i for i, p in enumerate(paths)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
    return results

def _copy_to_remote(src_path: str, dst_path: str, quiet: bool = True, concurrency: int = 128):
    remote = src_path if _is_remote(src_path) else dst_path
    if remote.startswith("az://"):
        cmd = f"azcopy copy '{src_path}' '{dst_path}' --cap-mbps=0"
    elif remote.startswith("gs://"):
        cmd = f"gsutil -m -q cp '{src_path}' '{dst_path}'"
    elif remote.startswith("s3://"):
        cmd = f"aws s3 cp '{src_path}' '{dst_path}'"
    else:
        raise ValueError(f"Unsupported remote scheme in: {remote}")
    if not quiet:
        print(f"Running: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Failed to sync {str(src_path)} to {dst_path}")            

def _is_remote(p) -> bool:
    return isinstance(p, str) and p.startswith(REMOTE_SCHEMES)

def _make_dirs(path: str):
    if _is_remote(path):
        if not bf.exists(path):
            bf.makedirs(path)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)

def _fabric_save_anywhere(fabric: L.Fabric, state: dict, dst_path: str):
    """Use Fabric to write a correct (FSDP-aware) checkpoint, then upload if remote."""
    if _is_remote(dst_path):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "ckpt.pth"
            fabric.save(tmp, state)  # lets Fabric extract model/optimizer state_dict safely
            if fabric.global_rank == 0:
                _copy_to_remote(str(tmp), str(dst_path))
                # bf.copy(str(tmp), str(dst_path), overwrite=True)
        fabric.barrier()
    else:
        fabric.save(dst_path, state)

def _fabric_load_anywhere(fabric: L.Fabric, src_path: str, state: dict):
    """Download remote to temp then let Fabric load; or load local directly."""
    if _is_remote(src_path):
        tmp = Path("/tmp/ckpt-cache/ckpt.pth")  # or a UUID under a shared FS
        if fabric.local_rank == 0:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            _copy_to_remote(str(src_path), str(tmp))
            # bf.copy(str(src_path), str(tmp), overwrite=True)
        fabric.barrier()
        fabric.load(tmp, state)
    else:
        fabric.load(src_path, state)

def _exists_anywhere(p: str) -> bool:
    return bf.exists(p) if _is_remote(p) else Path(p).exists()

def _latest_ckpt_in_dir_anywhere(root: str) -> Optional[str]:
    """Find highest iter-XXXXXX-ckpt.pth in a local or remote dir."""
    pattern = root + "/iter-*-ckpt.pth"
    if _is_remote(root):
        files = list(bf.glob(pattern))
    else:
        files = [str(p) for p in Path(root).glob("iter-*-ckpt.pth")]
    if not files:
        return None
    def _iter_num(s: str) -> int:
        base = s.rsplit("/", 1)[-1]
        # name format: iter-XXXXXX-ckpt.pth
        return int(base.split("-")[1])
    return max(files, key=_iter_num)
