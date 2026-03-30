# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# modified from https://github.com/microsoft/dion/blob/main/dion/muon.py 
# for uneven sharding, clip norm, and muon-hyperball support
import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


class Muon(Optimizer):
    """
    Distributed Muon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and parameter signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        use_muonh: bool = False,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", "clip_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', 'clip_norm', or None."
            )

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

        # Use Muon-Hyperball
        self.use_muonh = use_muonh
        # Set of parameters to exclude from hyperball normalization (e.g., embeddings)
        self._hyperball_excluded: set = set()

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

    def exclude_from_hyperball(self, *params):
        """
        Exclude parameters from hyperball normalization.
        Useful for embedding layers which are 2D but shouldn't be normalized.

        Args:
            *params: Parameter tensors to exclude from hyperball normalization.

        Example:
            optimizer = Muon(model.parameters(), use_muonh=True)
            optimizer.exclude_from_hyperball(model.embedding.weight)
        """
        for p in params:
            self._hyperball_excluded.add(p)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            muon_update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                cautious_wd=group["cautious_wd"],
                use_muonh=self.use_muonh,
            )

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Muon does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m in zip(params, gradients, momentums):
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                shard_dim=None,  # No sharded matrix dim
                                **muon_update_args,
                            )
                        )
                # Otherwise, we parallelize the Muon update across devices
                else:
                    yield AsyncTask(
                        muon_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **muon_update_args,
                        )
                    )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            cautious_wd = group["cautious_wd"]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    cautious_wd=cautious_wd,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            cautious_wd = group["cautious_wd"]
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            # Separate hyperball-eligible params (2D, not excluded) from others
            # This separation happens here (not in compiled function) for torch.compile compatibility
            if self.use_muonh:
                # Hyperball only applies to 2D params that are not excluded
                hyperball_params = [
                    p for p in params
                    if p.ndim == 2 and p not in self._hyperball_excluded
                ]
                standard_params = [
                    p for p in params
                    if p.ndim != 2 or p in self._hyperball_excluded
                ]
            else:
                hyperball_params = []
                standard_params = params

            # Process hyperball params (2D, not excluded)
            if hyperball_params:
                gradients = [p.grad for p in hyperball_params]
                states = [self._get_or_initialize_state(p, algo_name) for p in hyperball_params]
                momentums = [s["momentum"] for s in states]
                variances = [s["variance"] for s in states]

                yield AsyncTask(
                    adamw_update_foreach_async(
                        X=to_local(hyperball_params),
                        G=to_local(gradients),
                        M=to_local(momentums),
                        V=to_local(variances),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon,
                        cautious_wd=cautious_wd,
                        use_hyperball=True,
                        process_group=self._process_group,
                    )
                )

            # Process standard params (1D, 3D+, or excluded from hyperball)
            if standard_params:
                gradients = [p.grad for p in standard_params]
                states = [self._get_or_initialize_state(p, algo_name) for p in standard_params]
                momentums = [s["momentum"] for s in states]
                variances = [s["variance"] for s in states]

                yield AsyncTask(
                    adamw_update_foreach_async(
                        X=to_local(standard_params),
                        G=to_local(gradients),
                        M=to_local(momentums),
                        V=to_local(variances),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon,
                        cautious_wd=cautious_wd,
                        use_hyperball=False,
                    )
                )


def muon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    cautious_wd: bool = False,
    use_muonh: bool = False,  # Whether to use Muon-Hyperball variant
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"

        # Compute shard sizes for each rank (supports uneven sharding)
        # This follows torch.tensor_split semantics where first (full_size % world_size)
        # ranks get one extra element
        full_size = X[0].size(shard_dim)
        base_size, remainder = divmod(full_size, world_size)
        shard_sizes = [base_size + (1 if i < remainder else 0) for i in range(world_size)]

        # Allocate buffers to receive shards of one whole matrix from other devices
        # Each buffer has the size of the shard from that rank
        local_shape = list(U[0].shape)
        single_matrix_shards = []
        for i in range(world_size):
            shape = local_shape.copy()
            shape[shard_dim] = shard_sizes[i]
            single_matrix_shards.append(
                torch.empty(shape, dtype=U[0].dtype, device=U[0].device)
            )

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U) > 1:
        assert len(U) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Allocate empty tensors to receive updates from other devices
        U = [torch.empty_like(u) for u in U]

        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(
            U, single_matrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U) == 1
        U[0] = muon_update_newton_schulz(
            U[0],
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "clip_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten, clip_norm=True)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Update model parameters with orthogonalized output
    X_local = to_local(X)

    if use_muonh:
        # Muon-Hyperball: need global norms across all ranks for correct scaling
        # Step 1: Apply weight decay and compute local squared norms
        x_sq, u_sq = muonh_weight_decay_and_local_sq_norms(
            X_local, U, lr, weight_decay, cautious_wd
        )

        # Step 2: All-reduce to get global squared norms, then sqrt
        if shard_dim is not None and process_group is not None:
            dist.all_reduce(x_sq, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(u_sq, op=dist.ReduceOp.SUM, group=process_group)
        x_norms = x_sq.sqrt()
        u_norms = u_sq.sqrt()

        # Step 3: Apply scaled update and compute new local squared norms
        new_x_sq = muonh_apply_scaled_update_and_local_sq_norms(
            X_local, U, x_norms, u_norms, lr, epsilon
        )

        # Step 4: All-reduce for new norms
        if shard_dim is not None and process_group is not None:
            dist.all_reduce(new_x_sq, op=dist.ReduceOp.SUM, group=process_group)
        new_x_norms = new_x_sq.sqrt()

        # Step 5: Renormalize to preserve original global norm
        muonh_renormalize(X_local, x_norms, new_x_norms, epsilon)
    else:
        muon_update_post_orthogonalize(
            X=X_local,
            U=U,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            use_muonh=False,
            epsilon=epsilon,
        )


@torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
    use_muonh: bool = False,
    epsilon: Tensor = None,
):
    """
    Apply weight decay and standard Muon weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().

    Note: Muon-Hyperball (use_muonh=True) is handled separately in muon_update_batched
    with proper global norm computation via all-reduce.
    """
    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = base_lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        decay_terms = torch._foreach_mul(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Standard Muon weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


@torch.compile(fullgraph=True)
def muonh_weight_decay_and_local_sq_norms(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Apply weight decay and compute local squared norms for Muon-Hyperball.
    Returns (x_sq_norms, u_sq_norms) as stacked tensors of shape (N, 1).
    These need to be all-reduced across ranks to get global squared norms.
    """
    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        coeff = base_lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)
        decay_masks = torch._foreach_add(decay_masks, 1)
        decay_masks = torch._foreach_minimum(decay_masks, 1)

        decay_terms = torch._foreach_mul(X, decay_masks)
        decay_terms = torch._foreach_mul(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Compute local norms and return squared values for all-reduce
    x_norms = torch._foreach_norm(X)
    u_norms = torch._foreach_norm(U)

    x_sq = torch.stack([n.reshape(1) for n in x_norms]).square()
    u_sq = torch.stack([n.reshape(1) for n in u_norms]).square()

    return x_sq, u_sq


@torch.compile(fullgraph=True)
def muonh_apply_scaled_update_and_local_sq_norms(
    X: List[Tensor],
    U: List[Tensor],
    x_norms: Tensor,
    u_norms: Tensor,
    base_lr: Tensor,
    epsilon: float,
) -> Tensor:
    """
    Apply Muon-Hyperball scaled update using global norms.
    x_norms and u_norms should be global norms (shape N, 1).
    Returns local squared norms of new X for subsequent all-reduce.
    """
    eps = epsilon if epsilon is not None else 1e-10

    # Clamp u_norms to avoid division by zero
    u_norms_clamped = u_norms.clamp_min(eps)

    # Compute scale factors: x_norm / u_norm for each tensor
    scale_factors = x_norms / u_norms_clamped
    scale_list = [s.squeeze(0) for s in scale_factors.unbind(0)]

    # Scale updates: U * (x_norm / u_norm) * lr
    U_scaled = torch._foreach_mul(U, scale_list)
    torch._foreach_mul_(U_scaled, base_lr)

    # Apply update: X = X - U_scaled
    torch._foreach_sub_(X, U_scaled)

    # Compute local squared norms of new X for all-reduce
    new_norms = torch._foreach_norm(X)
    new_sq = torch.stack([n.reshape(1) for n in new_norms]).square()

    return new_sq


@torch.compile(fullgraph=True)
def muonh_renormalize(
    X: List[Tensor],
    x_norms: Tensor,
    new_x_norms: Tensor,
    epsilon: float,
):
    """
    Renormalize X to preserve original global norm.
    x_norms: original global norms (N, 1)
    new_x_norms: new global norms after update (N, 1)
    """
    eps = epsilon if epsilon is not None else 1e-10

    new_x_norms_clamped = new_x_norms.clamp_min(eps)
    scale_factors = x_norms / new_x_norms_clamped
    scale_list = [s.squeeze(0) for s in scale_factors.unbind(0)]

    torch._foreach_mul_(X, scale_list)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape, flatten):
    # Adjust learning rate for constant element-wise RMS norm
    # https://arxiv.org/abs/2502.16982
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape, flatten, clip_norm: bool = False):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    if clip_norm:
        adjusted_lr = lr * max(1.0, fan_out / fan_in) ** 0.5
    else:
        adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X