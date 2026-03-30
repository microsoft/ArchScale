"""FlashAttention w/support for learned sinks and banded attention.

This is an expanded version of the Flash Attention v2 implementation (see https://tridao.me/publications/flash2/flash2.pdf)
which can be found at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.

This version has been extended to support banded attention and learned attention sinks.
"""

import torch

import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# _attn_fwd_inner
# (thanks o3 for the help + kind comment strings....)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    BANDWIDTH: tl.constexpr,
):
    # ---------------- range of kv indices for this stage ---------------------
    if STAGE == 1:
        # off-band (used only when BANDWIDTH == 0)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # on-band **plus** the preceding tokens that fall inside `BANDWIDTH`
        if BANDWIDTH == 0:  # full context → current block only
            lo = start_m * BLOCK_M
        else:  # local context
            lo = tl.maximum(0, start_m * BLOCK_M - BANDWIDTH)
        hi = (start_m + 1) * BLOCK_M
        # make the compiler aware that `lo` is a multiple of BLOCK_N so that
        # the first `tl.load` is aligned (matches what the large kernel does)
        lo = tl.multiple_of(lo, BLOCK_N)
    else:  # STAGE == 3  (non-causal)
        lo, hi = 0, N_CTX

    # advance the KV block-pointers so they point at `lo`
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # ---------------- main loop over K/V tiles -------------------------------
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # ---- Q·Kᵀ ------------------------------------------------------------
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)

        # ------------- causal + bandwidth masking (STAGE == 2) ----------------
        if STAGE == 2:
            # causal mask  (j ≤ i)
            causal_ok = offs_m[:, None] >= (start_n + offs_n[None, :])

            if BANDWIDTH == 0:  # full causal attention
                mask = causal_ok
            else:  # local causal attention
                # j ≥ i − BANDWIDTH + 1  ⟺  i < j + BANDWIDTH
                within_bw = offs_m[:, None] < (start_n + offs_n[None, :] + BANDWIDTH)
                mask = causal_ok & within_bw

            qk = qk * qk_scale + tl.where(mask, 0.0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # STAGE 1 (when BANDWIDTH == 0) or STAGE 3 (non-causal)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # ---- softmax ---------------------------------------------------------
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # ---- running numerically-stable accumulators -------------------------
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        # Ensure tl.dot inputs have matching dtypes
        p_cast = p.to(v.dtype)
        acc = tl.dot(p_cast, v, acc)

        m_i = m_ij

        # ---- advance pointers ------------------------------------------------
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    M,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = float("-inf")

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1 and (BANDWIDTH == 0):
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            BANDWIDTH,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,
            BANDWIDTH,
        )
    # epilogue
    l_i = tl.maximum(l_i, 1e-12)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_packed(
    Q, K, V, Sinks,
    sm_scale, M, Out,
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_om, stride_oh, stride_ok,
    TOTAL_Q, H,  # totals
    cu_seqlens_q, cu_seqlens_k,  # int32/int64
    seq_ids_q, seq_ids_k,        # per-token seq ids
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    BANDWIDTH: tl.constexpr,
):
    # program ids
    start_m_abs = tl.program_id(0) * BLOCK_M
    h = tl.program_id(1)

    offs_m_abs = start_m_abs + tl.arange(0, BLOCK_M)
    valid_row = offs_m_abs < TOTAL_Q
    # Find the sequence id of the first row; rows that belong to other seqs will be masked
    # (avoids binary search per row)
    seq0 = tl.load(seq_ids_q + start_m_abs, mask=valid_row, other=0)
    # Broadcast first row's seq id
    seq0 = tl.max(seq0, axis=0)

    # Load seq boundaries (q_lo,q_hi,k_lo,k_hi)
    q_lo = tl.load(cu_seqlens_q + seq0)
    q_hi = tl.load(cu_seqlens_q + seq0 + 1)
    k_lo = tl.load(cu_seqlens_k + seq0)
    k_hi = tl.load(cu_seqlens_k + seq0 + 1)

    # Mask rows that cross into the next sequence boundary
    in_seq = (offs_m_abs >= q_lo) & (offs_m_abs < q_hi) & valid_row

    # Load sink
    if Sinks is not None:
        sink = tl.load(Sinks + h).to(tl.float32)
    else:
        sink = float("-inf")

    # offsets within the current sequence
    offs_m_local = (offs_m_abs - q_lo).to(tl.int32)
    offs_n_local = tl.arange(0, BLOCK_N)

    # Initialize m/l/acc
    m_i = tl.where(in_seq, sink, float("-inf")).to(tl.float32)
    l_i = tl.where(in_seq, 1.0, 0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln(2)

    # Load Q tile (BLOCK_M, HEAD_DIM)
    q_ptrs = Q + (offs_m_abs[:, None] * stride_qm + h * stride_qh + tl.arange(0, HEAD_DIM)[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=in_seq[:, None], other=0.0)

    # Iterate over K/V tiles within this sequence
    lo = 0
    hi = (k_hi - k_lo)

    for start_n_local in range(lo, hi, BLOCK_N):
        # absolute positions for K/V
        offs_n_abs = (k_lo + start_n_local) + offs_n_local
        kv_in_seq = offs_n_abs < k_hi

        # Load K and V
        k_ptrs = K + (offs_n_abs[None, :] * stride_km + h * stride_kh + tl.arange(0, HEAD_DIM)[:, None] * stride_kk)
        v_ptrs = V + (offs_n_abs[:, None] * stride_vm + h * stride_vh + tl.arange(0, HEAD_DIM)[None, :] * stride_vk)
        k = tl.load(k_ptrs, mask=kv_in_seq[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=kv_in_seq[:, None], other=0.0)

        # QK^T
        qk = tl.dot(q, k)

        if CAUSAL:
            # causal mask: j <= i
            causal_ok = offs_m_local[:, None] >= (start_n_local + offs_n_local[None, :])
            if BANDWIDTH == 0:
                mask = causal_ok
            else:
                within_bw = offs_m_local[:, None] < (start_n_local + offs_n_local[None, :] + BANDWIDTH)
                mask = causal_ok & within_bw
            mask = mask & in_seq[:, None] & kv_in_seq[None, :]
            qk = qk * qk_scale + tl.where(mask, 0.0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            mask = in_seq[:, None] & kv_in_seq[None, :]
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # softmax update
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        # Ensure tl.dot inputs have matching dtypes
        p_cast = p.to(v.dtype)
        acc = tl.dot(p_cast, v, acc)
        m_i = m_ij

    # epilogue
    l_i = tl.maximum(l_i, 1e-12)
    m_i = m_i + tl.math.log2(l_i)
    # write m and out (guard invalid rows)
    tl.store(M + (h * TOTAL_Q + offs_m_abs), m_i, mask=in_seq)
    o = acc / l_i[:, None]
    o_ptrs = Out + (offs_m_abs[:, None] * stride_om + h * stride_oh + tl.arange(0, HEAD_DIM)[None, :] * stride_ok)
    tl.store(o_ptrs, o.to(Out.type.element_ty), mask=in_seq[:, None])


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,  #
    Sinks,
    DSinks,
    DSinkstemp,
    Atomic_counters,
    M,
    Delta,  #
    Z,
    H,
    N_CTX,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    off_z = off_hz // H
    off_h = off_hz % H
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)

    if Sinks is not None:
        m = tl.load(M + off_z * H * N_CTX + off_h * N_CTX + off_m)
        sink = tl.load(Sinks + off_h)
        dl = tl.sum(tl.math.exp2(sink - m) * delta, axis=0)

        depth = Z * (N_CTX // BLOCK_M)

        tl.store(
            DSinkstemp + (off_h * Z + off_z) * (N_CTX // BLOCK_M) + tl.program_id(0), dl
        )

        if tl.atomic_add(Atomic_counters + off_h, 1) == depth - 1:
            dl_acc = 0.0

            for i in range(0, depth, BLOCK_M):
                idxs = i + tl.arange(0, BLOCK_M)
                temps = tl.load(
                    DSinkstemp + off_h * depth + idxs, mask=(idxs < depth), other=0.0
                )
                dl_acc += tl.sum(temps, axis=0)

            tl.store(DSinks + off_h, (-0.69314718) * dl_acc)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
    BANDWIDTH: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            if BANDWIDTH == 0:  # full causal
                mask = offs_m[None, :] >= offs_n[:, None]
            else:  # local causal
                mask = (offs_m[None, :] >= offs_n[:, None]) & (
                    offs_m[None, :] < offs_n[:, None] + BANDWIDTH
                )
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        # Match dtypes for tl.dot: cast pT to do.dtype
        ppT = pT.to(do.dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        # Cast dsT to qT.dtype for tl.dot(dsT, qT^T)
        dsT = (pT * (dpT - Di[None, :])).to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    BANDWIDTH: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            if BANDWIDTH == 0:  # full causal
                mask = offs_m[:, None] >= offs_n[None, :]
            else:  # local causal
                mask = (offs_m[:, None] >= offs_n[None, :]) & (
                    offs_m[:, None] < offs_n[None, :] + BANDWIDTH
                )
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        # Cast ds to q.dtype to match tl.dot(ds, kT)
        ds = (p * (dp - Di[:, None])).to(q.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BANDWIDTH: tl.constexpr,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=True,  #
        BANDWIDTH=BANDWIDTH,
    )

    start_m += num_steps * MASK_BLOCK_M1
    # how many *additional* rows may still attend to the current key block?
    if BANDWIDTH == 0:
        rows_left = N_CTX - start_m
    else:
        rows_left = min(N_CTX - start_m, BLOCK_N1)
    num_steps = rows_left // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=BANDWIDTH != 0,  #
        BANDWIDTH=BANDWIDTH,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,  #
        BANDWIDTH,
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,  #
        MASK=True,  #
    )
    end_n -= num_steps * MASK_BLOCK_N2

    # stage-1 (rows that still fall inside the window)
    if BANDWIDTH == 0:
        cols_left = end_n
    else:
        cols_left = min(end_n, BLOCK_M2)
    num_steps = cols_left // BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        BANDWIDTH,
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,  #
        MASK=BANDWIDTH != 0,  #
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sinks,
        causal,
        sm_scale,
        bandwidth,
        warp_specialize=True,
        USE_TMA=True,
    ):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        BLOCK_M = 128
        grid = (
            triton.cdiv(q.shape[2], BLOCK_M),
            q.shape[0] * q.shape[1],
            1,
        )
        _attn_fwd[grid](
            q,
            k,
            v,
            sinks,
            sm_scale,
            M,
            o,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            q.shape[0],
            q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=64,
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, sinks, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.bandwidth = bandwidth
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, o, M = ctx.saved_tensors
        do = do.contiguous()
        # Make contiguous copies to normalize strides for kernels
        cq = q.contiguous()
        ck = k.contiguous()
        cv = v.contiguous()
        co = o.contiguous()
        cdo = do.contiguous()

        dq = torch.empty_like(cq)
        dk = torch.empty_like(ck)
        dv = torch.empty_like(cv)

        BATCH, N_HEAD, N_CTX = cq.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = ck * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        if sinks is not None:
            dsinks = torch.empty_like(sinks)
            dsinkstemp = torch.empty(pre_grid, dtype=torch.float32, device=sinks.device)
            atomic_counters = torch.zeros(
                N_HEAD, dtype=torch.int32, device=sinks.device
            )
        else:
            dsinks, dsinkstemp, atomic_counters = None, None, None
        _attn_bwd_preprocess[pre_grid](
            co,
            cdo,  #
            # Info for attention sinks.
            sinks,
            dsinks,
            dsinkstemp,
            atomic_counters,
            M,
            ######
            delta,  #
            BATCH,
            N_HEAD,
            N_CTX,  #
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            cq,
            arg_k,
            cv,
            ctx.sm_scale,
            cdo,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            cq.stride(0),
            cq.stride(1),
            cq.stride(2),
            cq.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BANDWIDTH=ctx.bandwidth,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
        )

        # Map contiguous grads back to input strides
        dq_out = torch.empty_like(q)
        dk_out = torch.empty_like(k)
        dv_out = torch.empty_like(v)
        dq_out.copy_(dq)
        dk_out.copy_(dk)
        dv_out.copy_(dv)

        return dq_out, dk_out, dv_out, dsinks, None, None, None, None, None


@triton.jit
def _attn_bwd_packed_preprocess(
    O, DO,
    Sinks, DSinks,
    M, D,  # M from fwd, D to write (delta per row)
    TOTAL_Q, H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m_abs = tl.program_id(0) * BLOCK_M
    h = tl.program_id(1)
    offs_m = start_m_abs + tl.arange(0, BLOCK_M)
    valid = offs_m < TOTAL_Q
    # load o and do
    o_ptrs = O + (offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]) + h * (TOTAL_Q * HEAD_DIM)
    do_ptrs = DO + (offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]) + h * (TOTAL_Q * HEAD_DIM)
    o = tl.load(o_ptrs, mask=valid[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(D + (h * TOTAL_Q + offs_m), delta, mask=valid)
    if Sinks is not None:
        m = tl.load(M + (h * TOTAL_Q + offs_m), mask=valid, other=float("-inf"))
        sink = tl.load(Sinks + h)
        contrib = tl.sum(tl.math.exp2(sink - m) * delta, axis=0)
        # atomic add to DSinks[h]
        tl.atomic_add(DSinks + h, (-0.69314718) * contrib)


@triton.jit
def _attn_bwd_packed_dkdv(
    Q, K, V, DO, M, D,  # inputs
    DK, DV,              # outputs (global accum, atomic adds)
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_dom, stride_doh, stride_dok,
    stride_dkm, stride_dkh, stride_dkk,
    stride_dvm, stride_dvh, stride_dvk,
    TOTAL_Q, TOTAL_K,
    cu_seqlens_q, cu_seqlens_k,
    seq_ids_q,
    H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BANDWIDTH: tl.constexpr,
):
    start_m_abs = tl.program_id(0) * BLOCK_M
    h = tl.program_id(1)
    offs_m_abs = start_m_abs + tl.arange(0, BLOCK_M)
    valid_row = offs_m_abs < TOTAL_Q
    seq0 = tl.load(seq_ids_q + start_m_abs, mask=valid_row, other=0)
    seq0 = tl.max(seq0, axis=0)
    q_lo = tl.load(cu_seqlens_q + seq0)
    q_hi = tl.load(cu_seqlens_q + seq0 + 1)
    k_lo = tl.load(cu_seqlens_k + seq0)
    k_hi = tl.load(cu_seqlens_k + seq0 + 1)
    in_seq = (offs_m_abs >= q_lo) & (offs_m_abs < q_hi) & valid_row

    # local offsets within sequence
    offs_m_local = (offs_m_abs - q_lo).to(tl.int32)
    offs_n_local = tl.arange(0, BLOCK_N)

    # pointers
    q_ptrs = Q + (offs_m_abs[:, None] * stride_qm + h * stride_qh + tl.arange(0, HEAD_DIM)[None, :] * stride_qk)
    do_ptrs = DO + (offs_m_abs[:, None] * stride_dom + h * stride_doh + tl.arange(0, HEAD_DIM)[None, :] * stride_dok)
    m_rows = tl.load(M + (h * TOTAL_Q + offs_m_abs), mask=in_seq, other=float("-inf"))
    Di = tl.load(D + (h * TOTAL_Q + offs_m_abs), mask=in_seq, other=0.0)

    # accumulators for dv/dk (local)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    lo = 0
    hi = (k_hi - k_lo)
    for start_n_local in range(lo, hi, BLOCK_N):
        offs_n_abs = (k_lo + start_n_local) + offs_n_local
        kv_in_seq = offs_n_abs < k_hi
        # load tiles
        qT = tl.load(q_ptrs, mask=in_seq[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=in_seq[:, None], other=0.0).to(tl.float32)
        k = tl.load(K + (offs_n_abs[None, :] * stride_km + h * stride_kh + tl.arange(0, HEAD_DIM)[:, None] * stride_kk), mask=kv_in_seq[None, :], other=0.0)
        v = tl.load(V + (offs_n_abs[:, None] * stride_vm + h * stride_vh + tl.arange(0, HEAD_DIM)[None, :] * stride_vk), mask=kv_in_seq[:, None], other=0.0)
        # qk^T (BLOCK_N, BLOCK_M)
        qkT = tl.dot(k, tl.trans(qT))
        # Compute pT
        pT = tl.math.exp2(qkT - m_rows[None, :])
        # Causal masking
        mask = kv_in_seq[:, None] & in_seq[None, :]
        if BANDWIDTH == 0:
            causal_ok = (offs_n_local[:, None] + start_n_local) <= offs_m_local[None, :]
            mask = mask & causal_ok
        else:
            causal_ok = (offs_n_local[:, None] + start_n_local) <= offs_m_local[None, :]
            within_bw = offs_m_local[None, :] < (offs_n_local[:, None] + start_n_local + BANDWIDTH)
            mask = mask & causal_ok & within_bw
        pT = tl.where(mask, pT, 0.0)
        # dV (cast pT to do.dtype)
        dv += tl.dot(pT.to(do.dtype), do)
        # dK (cast dsT to qT.dtype)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = (pT * (dpT - Di[None, :])).to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))
    # atomically add to global DV/DK
    dv_ptrs = DV + (offs_n_local[:, None] * stride_dvm + h * stride_dvh + tl.arange(0, HEAD_DIM)[None, :] * stride_dvk) + k_lo * stride_dvm
    dk_ptrs = DK + (offs_n_local[:, None] * stride_dkm + h * stride_dkh + tl.arange(0, HEAD_DIM)[None, :] * stride_dkk) + k_lo * stride_dkm
    # write with atomics across tiles
    for i in range(BLOCK_N):
        tl.atomic_add(dv_ptrs + i * stride_dvm, dv[i, :])
        tl.atomic_add(dk_ptrs + i * stride_dkm, dk[i, :])


@triton.jit
def _attn_bwd_packed_dq(
    Q, K, V, DO, M, D,
    DQ,
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_dom, stride_doh, stride_dok,
    stride_dqm, stride_dqh, stride_dqk,
    TOTAL_Q, TOTAL_K,
    cu_seqlens_q, cu_seqlens_k,
    seq_ids_q,
    H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BANDWIDTH: tl.constexpr,
):
    start_m_abs = tl.program_id(0) * BLOCK_M
    h = tl.program_id(1)
    offs_m_abs = start_m_abs + tl.arange(0, BLOCK_M)
    valid_row = offs_m_abs < TOTAL_Q
    seq0 = tl.load(seq_ids_q + start_m_abs, mask=valid_row, other=0)
    seq0 = tl.max(seq0, axis=0)
    q_lo = tl.load(cu_seqlens_q + seq0)
    q_hi = tl.load(cu_seqlens_q + seq0 + 1)
    k_lo = tl.load(cu_seqlens_k + seq0)
    k_hi = tl.load(cu_seqlens_k + seq0 + 1)
    in_seq = (offs_m_abs >= q_lo) & (offs_m_abs < q_hi) & valid_row

    offs_m_local = (offs_m_abs - q_lo).to(tl.int32)
    offs_n_local = tl.arange(0, BLOCK_N)

    # pointers
    q_ptrs = Q + (offs_m_abs[:, None] * stride_qm + h * stride_qh + tl.arange(0, HEAD_DIM)[None, :] * stride_qk)
    do_ptrs = DO + (offs_m_abs[:, None] * stride_dom + h * stride_doh + tl.arange(0, HEAD_DIM)[None, :] * stride_dok)
    dq_ptrs = DQ + (offs_m_abs[:, None] * stride_dqm + h * stride_dqh + tl.arange(0, HEAD_DIM)[None, :] * stride_dqk)

    m_rows = tl.load(M + (h * TOTAL_Q + offs_m_abs), mask=in_seq, other=float("-inf"))
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    lo = 0
    hi = (k_hi - k_lo)
    for start_n_local in range(lo, hi, BLOCK_N):
        offs_n_abs = (k_lo + start_n_local) + offs_n_local
        kv_in_seq = offs_n_abs < k_hi
        kT = tl.load(K + (offs_n_abs[None, :] * stride_km + h * stride_kh + tl.arange(0, HEAD_DIM)[:, None] * stride_kk), mask=kv_in_seq[None, :], other=0.0)
        vT = tl.load(V + (offs_n_abs[None, :] * stride_vm + h * stride_vh + tl.arange(0, HEAD_DIM)[:, None] * stride_vk), mask=kv_in_seq[None, :], other=0.0)
        q = tl.load(q_ptrs, mask=in_seq[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=in_seq[:, None], other=0.0)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m_rows[:, None])
        mask = in_seq[:, None] & kv_in_seq[None, :]
        if BANDWIDTH == 0:
            causal_ok = offs_m_local[:, None] >= (start_n_local + offs_n_local[None, :])
            mask = mask & causal_ok
        else:
            causal_ok = offs_m_local[:, None] >= (start_n_local + offs_n_local[None, :])
            within_bw = offs_m_local[:, None] < (start_n_local + offs_n_local[None, :] + BANDWIDTH)
            mask = mask & causal_ok & within_bw
        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, tl.trans(vT)).to(tl.float32)
        Di = tl.load(D + (h * TOTAL_Q + offs_m_abs), mask=in_seq, other=0.0)
        # Cast ds to q.dtype to match tl.dot(ds, kT)
        ds = (p * (dp - Di[:, None])).to(q.dtype)
        dq += tl.dot(ds, kT)
    dq *= 0.6931471824645996  # ln(2)
    tl.store(dq_ptrs, dq, mask=in_seq[:, None])


class _attention_packed(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, kv,
                cu_seqlens, cu_seqlens_k,
                softmax_scale,
                causal,
                bandwidth,
                sinks):
        # split kv
        k = kv[:, 0]
        v = kv[:, 1]
        total, H, D = q.shape
        o = torch.empty_like(q)
        M = torch.empty((H * total,), device=q.device, dtype=torch.float32)
        # strides for (TOTAL, H, D)
        stride_qm, stride_qh, stride_qk = q.stride(0), q.stride(1), q.stride(2)
        stride_km, stride_kh, stride_kk = k.stride(0), k.stride(1), k.stride(2)
        stride_vm, stride_vh, stride_vk = v.stride(0), v.stride(1), v.stride(2)
        stride_om, stride_oh, stride_ok = o.stride(0), o.stride(1), o.stride(2)
        # seq ids
        B = cu_seqlens.numel() - 1
        seq_ids_q = torch.empty(total, dtype=torch.int32, device=q.device)
        seq_ids_k = torch.empty(k.shape[0], dtype=torch.int32, device=q.device)
        for b in range(B):
            q_lo = int(cu_seqlens[b].item()); q_hi = int(cu_seqlens[b + 1].item())
            k_lo = int(cu_seqlens_k[b].item()); k_hi = int(cu_seqlens_k[b + 1].item())
            seq_ids_q[q_lo:q_hi] = b
            seq_ids_k[k_lo:k_hi] = b
        BLOCK_M = 128
        grid = (triton.cdiv(total, BLOCK_M), H)
        _attn_fwd_packed[grid](
            q, k, v, sinks,
            softmax_scale, M, o,
            stride_qm, stride_qh, stride_qk,
            stride_km, stride_kh, stride_kk,
            stride_vm, stride_vh, stride_vk,
            stride_om, stride_oh, stride_ok,
            total, H,
            cu_seqlens, cu_seqlens_k,
            seq_ids_q, seq_ids_k,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=64,
            CAUSAL=1 if causal else 0,
            BANDWIDTH=bandwidth,
        )
        ctx.save_for_backward(q, k, v, o, M, cu_seqlens, cu_seqlens_k, seq_ids_q, sinks)
        ctx.softmax_scale = softmax_scale
        ctx.H = H
        ctx.D = D
        ctx.bandwidth = bandwidth
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, cu_seqlens, cu_seqlens_k, seq_ids_q, sinks = ctx.saved_tensors
        total, H, D = q.shape
        # allocate grads
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        # preprocess: delta and dsinks
        Dvec = torch.empty(H * total, device=q.device, dtype=torch.float32)
        if sinks is not None:
            dsinks = torch.zeros_like(sinks)
        else:
            dsinks = None
        BLOCK_M = 128
        grid_pre = (triton.cdiv(total, BLOCK_M), H)
        _attn_bwd_packed_preprocess[grid_pre](
            # O and DO are laid out per-head contiguous: reshape by view
            o.contiguous().view(H, total, D),
            do.contiguous().view(H, total, D),
            sinks, dsinks,
            M, Dvec,
            total, H,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
        )
        # launch dkdv
        NUM_WARPS, NUM_STAGES = 4, 5
        grid = (triton.cdiv(total, 32), H)
        _attn_bwd_packed_dkdv[grid](
            q, k, v, do, M, Dvec,
            dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            total, k.shape[0],
            cu_seqlens, cu_seqlens_k,
            seq_ids_q,
            H,
            HEAD_DIM=D,
            BLOCK_M=32, BLOCK_N=64,
            BANDWIDTH=ctx.bandwidth,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
        # launch dq
        _attn_bwd_packed_dq[grid](
            q, k, v, do, M, Dvec,
            dq,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            total, k.shape[0],
            cu_seqlens, cu_seqlens_k,
            seq_ids_q,
            H,
            HEAD_DIM=D,
            BLOCK_M=32, BLOCK_N=64,
            BANDWIDTH=ctx.bandwidth,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
        return dq, torch.stack([dk, dv], dim=1), None, None, None, None, None, dsinks


# Replace the direct alias with a wrapper that supports variable-length inputs.
# Compatible with FlashAttention2 API used in model.py

def attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
    cu_seqlens_k: torch.Tensor = None,
    max_seqlen_k: int = None,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: tuple = (-1, -1),
    sinks: torch.Tensor = None,
):
    """Variable-length wrapper for Triton attention with optional band/window and sinks.

    Accepts either padded tensors (B, T, H, D) and (B, T, 2, H, D) or packed tensors
    (total, H, D) and (total, 2, H, D) with cu_seqlens.
    """
    del dropout_p  # not used
    # Split K/V from kv
    if kv.dim() == 4 and kv.shape[1] == 2:  # packed: (total, 2, H, D)
        k_flat = kv[:, 0]
        v_flat = kv[:, 1]
    elif kv.dim() == 5 and kv.shape[2] == 2:  # padded: (B, T, 2, H, D)
        k_flat = kv[:, :, 0]
        v_flat = kv[:, :, 1]
    else:
        raise ValueError("kv must be shaped as (..., 2, H, D)")

    # Infer scale
    if softmax_scale is None:
        softmax_scale = (1.0 / (q.shape[-1] ** 0.5))

    # Map window_size to bandwidth
    bandwidth = 0
    if window_size is not None and isinstance(window_size, tuple) and window_size[0] is not None and window_size[0] > 0:
        bandwidth = int(window_size[0]) + 1

    # Packed variable-length path
    if cu_seqlens is not None:
        assert q.dim() == 3 and k_flat.dim() == 3 and v_flat.dim() == 3
        total, H, D = q.shape
        # choose cu_seqlens_k if provided (cross-attn), else reuse cu_seqlens
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens

        # Use fused packed kernel for inference (no grads)
        if not q.requires_grad:
            # Build per-token seq id maps (GPU)
            B = cu_seqlens.numel() - 1
            seq_ids_q = torch.empty(total, dtype=torch.int32, device=q.device)
            seq_ids_k = torch.empty(k_flat.shape[0], dtype=torch.int32, device=q.device)
            for b in range(B):
                q_lo = int(cu_seqlens[b].item()); q_hi = int(cu_seqlens[b + 1].item())
                k_lo = int(cu_seqlens_k[b].item()); k_hi = int(cu_seqlens_k[b + 1].item())
                seq_ids_q[q_lo:q_hi] = b
                seq_ids_k[k_lo:k_hi] = b

            o = torch.empty_like(q)
            M = torch.empty((H * total,), device=q.device, dtype=torch.float32)
            # strides for (TOTAL, H, D)
            stride_qm, stride_qh, stride_qk = q.stride(0), q.stride(1), q.stride(2)
            stride_km, stride_kh, stride_kk = k_flat.stride(0), k_flat.stride(1), k_flat.stride(2)
            stride_vm, stride_vh, stride_vk = v_flat.stride(0), v_flat.stride(1), v_flat.stride(2)
            stride_om, stride_oh, stride_ok = o.stride(0), o.stride(1), o.stride(2)

            BLOCK_M = 128
            grid = (triton.cdiv(total, BLOCK_M), H)
            _attn_fwd_packed[grid](
                q, k_flat, v_flat, sinks,
                softmax_scale, M, o,
                stride_qm, stride_qh, stride_qk,
                stride_km, stride_kh, stride_kk,
                stride_vm, stride_vh, stride_vk,
                stride_om, stride_oh, stride_ok,
                total, H,
                cu_seqlens, cu_seqlens_k,
                seq_ids_q, seq_ids_k,
                HEAD_DIM=D,
                BLOCK_M=BLOCK_M,
                BLOCK_N=64,
                CAUSAL=1 if causal else 0,
                BANDWIDTH=bandwidth,
            )
            return o

        # Training path with fused packed autograd
        return _attention_packed.apply(q, torch.stack([k_flat, v_flat], dim=1), cu_seqlens, cu_seqlens_k, softmax_scale, causal, bandwidth, sinks)

    # Padded path: expect q (B, T, H, D), kv (B, T, 2, H, D)
    assert q.dim() == 4 and k_flat.dim() == 4 and v_flat.dim() == 4
    # reshape to (B, H, T, D)
    q_4d = q.permute(0, 2, 1, 3).contiguous()
    k_4d = k_flat.permute(0, 2, 1, 3).contiguous()
    v_4d = v_flat.permute(0, 2, 1, 3).contiguous()
    o_4d = _attention.apply(q_4d, k_4d, v_4d, sinks, causal, softmax_scale, bandwidth)
    # back to (B, T, H, D)
    return o_4d.permute(0, 2, 1, 3).contiguous()
