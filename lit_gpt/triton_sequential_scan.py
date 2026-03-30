# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# credit: https://github.com/sustcsonglin/mamba-triton/blob/master/triton_sequential_scan.py

import torch
import triton
import triton.language as tl

assert triton.__version__ != '2.1.0', 'Triton 2.1.0 is missing enable_fp_fusion. Triton 2.2.0 is required for numerical stability of this implementation.'

inv_ln2 = tl.constexpr(1.44269504089)# for exp2

# credit: https://github.com/proger/accelerated-scan/blob/b9edbad65c673f9a1915efe51dc6bbf50fd7f8c4/accelerated_scan/triton.py

def reduce(H, C):
    out = H @ C.unsqueeze(-1)
    return out.squeeze(-1)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['D', 'K'],
)
@triton.jit
def fwd_recurrence(
    A,
    B,
    C,
    Dt,
    X,
    Y,
    H,
    initial_state,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)

    dt_ptr = Dt + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    u_ptr = X + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    o_ptr = Y + i_bh * T * D + i_v * BV + tl.arange(0, BV)

    h = tl.zeros([BV, K], dtype=tl.float32)

    b_ptr = B + i_bh * T * K + tl.arange(0, K)

    A = A + ((i_v * BV) + tl.arange(0, BV)
             [:, None])*K + tl.arange(0, K)[None, :]
    _A = tl.load(A)

    H_ptr = H + i_bh * T * D * K + \
        (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :]

    h += tl.load(initial_state + i_bh * D * K + (i_v * BV +
                 tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :])

    for i in range(T):
        b = tl.load(b_ptr).to(tl.float32)
        dt = tl.load(dt_ptr)
        u = tl.load(u_ptr)
        x_dt = u * dt
        x_dt_b = x_dt[:, None] * b[None, :]
        dt_a = tl.exp2(dt[:, None] * _A * inv_ln2)
        h = h * dt_a + x_dt_b
        tl.store(H_ptr, h)

        b_ptr += K
        dt_ptr += D
        u_ptr += D
        o_ptr += D
        H_ptr += D * K

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['D', 'K'],
)
@triton.jit
def bwd_recurrence(
    A,
    B,
    C,
    U,
    Dt,
    DO,
    H,
    DA,
    DB,
    DC,
    dDt,
    dU,
    batch,
    initial_state,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    NV = tl.cdiv(D, BV)

    dt_ptr = Dt + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    ddt_ptr = dDt + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    u_ptr = U + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    du_ptr = dU + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    do_ptr = DO + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D

    dh = tl.zeros([BV, K], dtype=tl.float32)
    dA = tl.zeros([BV, K], dtype=tl.float32)

    b_ptr = B + i_bh * T * K + tl.arange(0, K) + (T - 1) * K
    c_ptr = C + i_bh * T * K + tl.arange(0, K) + (T - 1) * K
    dc_ptr = DC + (i_bh + batch * i_v) * T * K + tl.arange(0, K) + (T - 1) * K
    db_ptr = DB + (i_bh + batch * i_v) * T * K + tl.arange(0, K) + (T - 1) * K

    A = A + ((i_v * BV) + tl.arange(0, BV)
             [:, None])*K + tl.arange(0, K)[None, :]
    _A = tl.load(A)
    H_ptr = H + i_bh * T * D * K + \
        (i_v * BV + tl.arange(0, BV)[:, None]) * K + \
        tl.arange(0, K)[None, :] + (T - 1) * D * K

    for i in range(T):
        h = tl.load(H_ptr)
        if i < T - 1:
            next_h = tl.load(H_ptr - D * K)
        else:
            next_h = tl.load(initial_state + i_bh * D * K + (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :])
        b = tl.load(b_ptr).to(tl.float32)
        c = tl.load(c_ptr).to(tl.float32)
        do = tl.load(do_ptr).to(tl.float32)
        u = tl.load(u_ptr).to(tl.float32)
        dt = tl.load(dt_ptr).to(tl.float32)

        # gradient wrt output proj
        dc = tl.sum(h * do[:, None], axis=0)
        tl.store(dc_ptr, dc)

        # graident wrt input
        dh += do[:, None] * c[None, :]
        dt_u = dt * u
        db = tl.sum(dh * dt_u[:, None], axis=0)
        tl.store(db_ptr, db)
        ddt_u = tl.sum(dh * b[None, :], axis=1)
        ddt = ddt_u * u
        du = ddt_u * dt
        tl.store(du_ptr, du)

        # gradient wrt decay
        dt_a = tl.exp2(dt[:, None] * _A * inv_ln2)
        dh *= dt_a

        d_decay = dh * next_h
        dA += d_decay * dt[:, None]
        ddt += tl.sum(d_decay * _A, axis=1)
        tl.store(ddt_ptr, ddt)


        # update ptr
        b_ptr -= K
        c_ptr -= K
        dc_ptr -= K
        db_ptr -= K
        dt_ptr -= D
        ddt_ptr -= D
        u_ptr -= D
        du_ptr -= D
        do_ptr -= D
        H_ptr -= D * K

    DA_ptr = DA + i_bh * D * K + \
        (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :]
    tl.store(DA_ptr, dA)


class SelectiveScan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, initial_state=None):
        b_size,  T, d = u.shape
        K = B.shape[-1]

        ctx.b_size = b_size
        ctx.T = T
        ctx.d = d
        ctx.K = K
        BV = 64

        if b_size <= 16:
            BV = 32 
        
        NV = triton.cdiv(d, BV)

        o = torch.empty_like(u)
        H = torch.empty(b_size, T, d, K, device=u.device, dtype=torch.float32)

        if initial_state is None:
            initial_state = torch.zeros(
                b_size, d, K, device=u.device, dtype=torch.float32)

        fwd_recurrence[(b_size, NV)](A, B, C, delta, u, o, H,
                                     initial_state,  T, d, K, BV)
        o = reduce(H, C)
        ctx.save_for_backward(A, B, C, delta, H, u)
        ctx.initial_state = initial_state
        return o, H[:,-1]

    @staticmethod
    def backward(ctx, grad_output, d_final_state):
        do = grad_output
        A, B, C, delta, H, u = ctx.saved_tensors
        b_size = ctx.b_size
        T = ctx.T
        d = ctx.d
        K = ctx.K

        BV = 64

        if b_size <= 16:
            BV = 32

        NV = triton.cdiv(d, BV)
        dA = A.new_empty(b_size, d, K)
        du = torch.empty_like(u)
        d_delta = torch.empty_like(delta)
        db = B.new_empty(NV, b_size, T, K)
        dc = C.new_empty(NV, b_size, T, K)

        bwd_recurrence[(b_size, NV)](A, B, C, u, delta, do, H, dA, db, dc,
                                     d_delta, du, b_size, ctx.initial_state, T, d, K, BV)
        db = db.sum(0)
        dc = dc.sum(0)

        return du, d_delta, dA.sum(0), db, dc, None


def triton_selective_scan_sequential(u, delta, A, B, C, D, initial_state=None):
    original_dtype = u.dtype
    D = D.float()
    A = A.float()
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = SelectiveScan.apply(u, delta, A, B, C.float(), initial_state)
    o = o + D * u
    return o.to(original_dtype), final_state

