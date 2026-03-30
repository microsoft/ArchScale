# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2025, Tri Dao.
# As of 2025-04-23, we require triton >= 3.0

import torch
import triton
import triton.language as tl

from functools import partial
from typing import Optional, Tuple, Union

from torch import Tensor

from einops import rearrange, repeat


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch_fa(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


def _apply_rotary_emb_qkv(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
    seqlen_offsets: Union[int, Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    apply_rotary_fn = partial(
        apply_rotary,
        interleaved=interleaved,
        inplace=inplace,
        conjugate=conjugate,
        seqlen_offsets=seqlen_offsets
    )
    if cos_k is None and sin_k is None and qkv.is_contiguous():
        # Call 1 kernel instead of 2 kernels
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
        # dimensions, we get the same tensor
        if qkv.dim() == 5:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            qk = apply_rotary_fn(qk, cos, sin)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
            qk = qkv[:, :, :num_heads_q + num_heads_k]
            qk = apply_rotary_fn(qk, cos, sin)
        if not inplace:
            if qkv.dim() == 5:
                qkv = torch.cat([rearrange(qk, "b s (t h) d -> b s t h d", t=2), qkv[:, :, 2:]], dim=2)
            else:
                qkv = torch.cat([qk, qkv[:, :, num_heads_q + num_heads_k :]], dim=2)
    else:
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        if qkv.dim() == 5:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            q, k = qkv[:, :, 0], qkv[:, :, 1]
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
            q, k = qkv[:, :, :num_heads_q], qkv[:, :, num_heads_q : num_heads_q + num_heads_k]
        q = apply_rotary_fn(q, cos, sin)
        k = apply_rotary_fn(k, cos_k, sin_k)
        if not inplace:
            if qkv.dim() == 5:
                qkv = torch.stack([q, k, qkv[:, :, 2]], dim=2)
            else:
                qkv = torch.cat([q, k, qkv[:, :, num_heads_q + num_heads_k:]], dim=2)
    return qkv


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Optional[int] = None,
    ):
        # apply_rotary_emb_qkv_inplace(
        qkv = _apply_rotary_emb_qkv(
            qkv, cos, sin, cos_k, sin_k, interleaved=interleaved, inplace=True,
            seqlen_offsets=seqlen_offsets, num_heads_q=num_heads_q,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        dqkv = _apply_rotary_emb_qkv(
            dqkv, cos, sin, cos_k, sin_k, interleaved=ctx.interleaved, inplace=True,
            seqlen_offsets=seqlen_offsets, num_heads_q=ctx.num_heads_q, conjugate=True,
        )
        return dqkv, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(
        qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, num_heads_q
    )


class ApplyRotaryEmbKV_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, kv, cos, sin, interleaved=False, seqlen_offsets: Union[int, torch.Tensor] = 0):
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k = kv[:, :, 0]
        apply_rotary(
            k, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        return dkv, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            # We want fp32 here as well since inv_freq will be multiplied with t, and the output
            # will be large. Having it in bf16 will lose a lot of precision and cause the
            # cos & sin output to change significantly.
            # We want to recompute self.inv_freq if it was not loaded in fp32
            if self.inv_freq.dtype != torch.float32:
                inv_freq = self._compute_inv_freq(device=device)
            else:
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to bf16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            if kv is none, else it's just q of shape (batch, seqlen, nheads, headdim).
            If qkv has shape (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if kv is None:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached,
                self._sin_cached,
                self._cos_k_cached if self.scale is not None else None,
                self._sin_k_cached if self.scale is not None else None,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                num_heads_q=num_heads_q,
            )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            kv = apply_rotary_emb_kv_(
                kv,
                self._cos_cached if self.scale is None else self._cos_k_cached,
                self._sin_cached if self.scale is None else self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
            return q, kv

@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    # We want ROTARY_DIM to be constexpr, otherwise the triton compiler doesn't know that
    # the mask is constant every 8 elements, and it will generate LDG.16 instead of LDG.128
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch
        OUT = OUT + pid_batch * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk_half[None, None, :] * stride_x_headdim)
        OUT = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk_half[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0,).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk[None, None, :] * stride_x_headdim)
        OUT = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    grid = lambda META: (triton.cdiv(nheads, META["BLOCK_H"]), triton.cdiv(seqlen, META["BLOCK_M"]), batch)  # noqa
    BLOCK_M = 8 if rotary_dim <= 128 else 4

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(rotary_kernel)[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            seqlen_ro,
            output.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=BLOCK_M,
            BLOCK_H=2,
        )
    return output

apply_rotary_emb_func = torch.compiler.disable(apply_rotary_emb_func)

@torch.compile
def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)