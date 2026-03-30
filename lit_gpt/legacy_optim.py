# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# kudos to Zichong Li(zli911@gatech.edu)
# credit:https://github.com/samsja/muon_fsdp_2
# credits to https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823

from typing import Generator
from collections import deque

import torch
from torch.optim.optimizer import ParamsT
try:
    from torch.distributed.tensor import DTensor, Shard
except:
    from torch.distributed._tensor import DTensor, Shard
from torch.distributed import gather, scatter
import torch.distributed as dist
from torch import Tensor
import os

@torch.compile(fullgraph=True)
def nsloop_torch(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    """
    When compiled down, inductor produces the following steps:
    1. A = matmul X with reinterpret_tensor(X)
    2. (triton) read A -> write b*A and c*A
    3. B = addmm(b*A, c*A, A)
    4. (triton) read X -> write a*X (this is stupid)
    5. X = addmm(a*X, B, X)
    """
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def zeropower_via_newtonschulz(G, steps=10, eps=1e-7, f_iter=nsloop_torch):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
        # DTensor will NaN for sharded compute on Shard(1)
        if isinstance(X, DTensor):
            p = [Shard(0) if isinstance(p, Shard) else p for p in X._spec.placements]
            X = X.redistribute(placements=p)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)  # ensure top singular value <= 1
    X = f_iter(X, steps)
    return X if G.size(-2) <= G.size(-1) else X.mT


def paramst_to_groups(params: ParamsT) -> list[dict]:
    if all(isinstance(p, dict) for p in params):
        return params
    if all(isinstance(p, torch.nn.Parameter) for p in params):
        return [dict(params=params)]
    if all(isinstance(p, list) for p in params):
        return [dict(params=p) for p in params]
    raise ValueError(f"Invalid paramst_to_groups input: {params}")


class Muon_fsdp2(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    This is a pruned implementation which uses the following hardcoded behaviors:
    * assumed used of 2D+ DTensor parameters, which will always be true if you use FSDP2.
    * nestrov momentum (on the input to NS)
    * EMA momentum (unlike the original Muon, which uses .mul_(beta))

    Arguments:
        params: Params/groups to be optimized.

    Default arguments (used for params with no specific group lr/wd/beta):
        lr: Learning rate.
        wd: Weight decay.
        beta: momentum buffer decay rate.
    """

    def __init__(
        self, params: ParamsT, *, lr: float | None = None, weight_decay: float = 0.0, beta: float = 0.95, ns_steps: int = 5, muon_mode: str = "normal",
    ):
        # setup torch optimizer
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta, ns_steps=ns_steps)
        groups = paramst_to_groups(list(params))
        super().__init__(groups, defaults)
        self.mode = muon_mode
        # init buffers ahead of time
        for pg in self.param_groups:
            for p in pg["params"]:
                assert isinstance(p, DTensor), "We only support DTensor. Use FSDP2."
                self.mesh = p._spec.device_mesh
                # TODO: figure out how to store optim step state without exploding DCP
                self.state[p] = dict(m=torch.zeros_like(p))
                if p.ndim < 2:
                    raise ValueError(f"0/1D parameters are banned from Muon; user provided {p.shape=}")
                if p.ndim > 2:
                    print(f"WARNING: muon used for {p.shape=}")
            # todo: also declare tensorlists for foreach
            ...

    def filter_group(self, group: dict) -> Generator[tuple[DTensor, DTensor, DTensor, int], None, None]:
        pg, lr, weight_decay, beta = group["params"], group["lr"], group["weight_decay"], group["beta"]
        pg = [p for p in pg if p.grad is not None]
        list_p = [p.data for p in pg]
        list_g = [p.grad.flatten(1) for p in pg]
        list_m = [self.state[p]["m"] for p in pg]
        torch._foreach_lerp_(list_m, list_g, 1 - beta)  # EMA momentum
        torch._foreach_lerp_(list_g, list_m, beta)  # nestrov momentum (for NS input)
        # torch._foreach_mul_(list_p, 1 - lr * weight_decay)  # weight decay
        yield from zip(list_p, list_g, list_m)

    @torch.no_grad()
    def step(self, *, prefetch_factor: int = 8):
        pg_last = self.mesh.get_group(-1)
        r_local = self.mesh.get_local_rank(-1)                 # 0..ws-1 within subgroup
        ws = dist.get_world_size(group=pg_last)
        dq = deque()

        def deferred_work(p, g, g_full_block, spec, lr, weight_decay, src_local, my_local):
            src_global = dist.get_global_rank(pg_last, src_local)   # <<< NEW
            if my_local == src_local:
                chunks = list(g_full_block.chunk(ws, dim=0))
                worker = scatter(
                    g.to_local(), chunks, src=src_global, async_op=True, group=pg_last
                )
            else:
                worker = scatter(
                    g.to_local(), None, src=src_global, async_op=True, group=pg_last
                )

            p.mul_(1 - lr * weight_decay)
            assert weight_decay > 0, f"Muon weight decay must be positive, got {weight_decay}"

            worker.wait()
            if self.mode == "spectral":
                lr_scale = (p.size(-2) / p.size(-1)) ** 0.5
            elif self.mode == "normal":
                lr_scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
            elif self.mode == "moonlight":
                lr_scale = 0.2 * max(p.size(-2), p.size(-1)) ** 0.5
            elif self.mode == "moonlight_large":
                lr_scale = 0.4 * max(p.size(-2), p.size(-1)) ** 0.5
            elif self.mode == "moonlight_larger":
                lr_scale = 0.8 * max(p.size(-2), p.size(-1)) ** 0.5
            elif self.mode == "liming":
                lr_scale = 0.25 * (p.size(0) * p.size(1))**0.5 / (g.norm() + 1e-8)
            p.add_(g, alpha=-lr * lr_scale)

        i = 0
        for group in self.param_groups:
            for p, g, m in self.filter_group(group):
                spec = g._spec
                dest_local = i % ws                                  # subgroup-local
                dest_global = dist.get_global_rank(pg_last, dest_local)  # <<< NEW

                if dest_local == r_local:
                    gather_lists = [torch.zeros_like(g.to_local()) for _ in range(ws)]
                    gather(
                        g.to_local(), gather_lists, dst=dest_global, async_op=False, group=pg_last
                    )
                    g_full_block = torch.cat(gather_lists, dim=0)
                    g_full_block.copy_(zeropower_via_newtonschulz(g_full_block, steps=group["ns_steps"]))
                    g_full_block = g_full_block.view_as(p).type_as(p)
                else:
                    g_local = g.to_local()
                    gather(
                        g_local, None, dst=dest_global, async_op=False, group=pg_last
                    )
                    g_full_block = None

                dq.append([p, g, g_full_block, spec, group["lr"], group["weight_decay"], dest_local, r_local])
                if len(dq) > prefetch_factor:
                    deferred_work(*dq.popleft())
                i += 1

        for ls in dq:
            deferred_work(*ls)


def zeropower_via_newtonschulz5(G: Tensor, steps=5) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonDDP(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, beta=0.95, nesterov=True, ns_steps=5):
        self.rank = int(os.environ.get("LOCAL_RANK"))
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.ns5 = zeropower_via_newtonschulz5
        print("rank", self.rank, "world_size", self.world_size)
        params = params[0]
        defaults = dict(lr=lr, weight_decay=params["weight_decay"], lr_mult=params["lr_mult"], beta=beta, nesterov=nesterov, ns_steps=ns_steps)
        params_in = params["params"]
        param_groups = []
        for size in {p.numel() for p in params_in}:
            b = torch.empty((self.world_size, size), dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params_in if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)], weight_decay=weight_decay, lr_mult=params["lr_mult"])
            param_groups.append(group)
        
        # print length and size of each group
        if self.rank == 0:
            for i, group in enumerate(param_groups):
                print(f"Group {i}: {len(group['params'])} params, size {group['params'][0].numel()}")
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["beta"])
                    g = g.lerp_(buf, group["beta"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = self.ns5(g).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()