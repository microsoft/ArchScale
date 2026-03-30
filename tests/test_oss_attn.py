import os
import pytest
import torch

from lit_gpt.oss_attn import attention as sink_attention

try:
    from flash_attn.bert_padding import unpad_input, pad_input
    _fa_ok = True
except Exception:
    _fa_ok = False


def _make_mask(batch, seqlen, device):
    # random variable lengths in [1, seqlen]
    lengths = torch.randint(1, seqlen + 1, (batch,), device=device)
    mask = torch.arange(seqlen, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask, lengths.max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _fa_ok, reason="flash-attn not available")
@pytest.mark.parametrize("window", [(-1, -1), (7, 0)])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_oss_attn_forward_equivalence(window, use_sinks):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    B, T, H, D = 3, 13, 4, 64
    scale = 1.0 / (D ** 0.5)

    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    kv = torch.stack([k, v], dim=2)  # (B, T, 2, H, D)

    sinks = torch.randn(H, device=device, dtype=torch.float32) if use_sinks else None

    mask, max_seqlen = _make_mask(B, T, device)

    # Padded path
    out_pad = sink_attention(q, kv, softmax_scale=scale, causal=True, window_size=window, sinks=sinks)

    # Packed path
    q_packed, indices_q, cu_q, max_q, _ = unpad_input(q, mask)
    kv_packed, _, cu_k, max_k, _ = unpad_input(kv, mask)
    out_packed = sink_attention(
        q_packed, kv_packed,
        cu_seqlens=cu_q, max_seqlen=max_q,
        cu_seqlens_k=cu_k, max_seqlen_k=max_k,
        softmax_scale=scale, causal=True, window_size=window, sinks=sinks,
    )
    out_padded_from_packed = pad_input(out_packed, indices_q, B, max_q)

    # Compare only on valid tokens
    valid = mask[:, : out_pad.shape[1]].unsqueeze(-1).unsqueeze(-1)
    diff = (out_pad[:, : max_q][valid] - out_padded_from_packed[valid]).float().abs().max().item()
    assert diff < 2e-2, f"forward mismatch max abs diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _fa_ok, reason="flash-attn not available")
@pytest.mark.parametrize("window", [(-1, -1), (7, 0)])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_oss_attn_backward_equivalence(window, use_sinks):
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.float16

    B, T, H, D = 2, 9, 2, 64
    scale = 1.0 / (D ** 0.5)

    # Inputs (padded) with grad
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    kv = torch.stack([k, v], dim=2)
    sinks = torch.randn(H, device=device, dtype=torch.float32) if use_sinks else None
    mask, _ = _make_mask(B, T, device)

    # Forward/backward padded
    out_pad = sink_attention(q, kv, softmax_scale=scale, causal=True, window_size=window, sinks=sinks)
    loss_pad = out_pad.float().pow(2).mean()
    loss_pad.backward()
    dq_pad, dk_pad, dv_pad = q.grad.detach(), k.grad.detach(), v.grad.detach()

    # Prepare packed inputs (clone for clean graph)
    q2 = q.detach().clone().requires_grad_(True)
    kv2 = kv.detach().clone().requires_grad_(True)
    q_packed, indices_q, cu_q, max_q, _ = unpad_input(q2, mask)
    kv_packed, kv_indices, cu_k, max_k, _ = unpad_input(kv2, mask)

    out_packed = sink_attention(
        q_packed, kv_packed,
        cu_seqlens=cu_q, max_seqlen=max_q,
        cu_seqlens_k=cu_k, max_seqlen_k=max_k,
        softmax_scale=scale, causal=True, window_size=window, sinks=sinks,
    )
    loss_packed = out_packed.float().pow(2).mean()
    loss_packed.backward()

    # Compare dq: unpad padded dq and compare to dq from packed
    dq_pad_unp, _, _, _, _ = unpad_input(dq_pad, mask)
    dq_diff = (dq_pad_unp.float() - q2.grad.float()).abs().max().item()
    assert dq_diff < 5e-2, f"dq mismatch max abs diff={dq_diff}"

    # Compare dk/dv via stacked kv grads
    kv_grad_pad = torch.stack([dk_pad, dv_pad], dim=2)
    kv_grad_pad_unp, _, _, _, _ = unpad_input(kv_grad_pad, mask)
    kv_grad_diff = (kv_grad_pad_unp.float() - kv2.grad.float()).abs().max().item()
    assert kv_grad_diff < 7e-2, f"dk/dv mismatch max abs diff={kv_grad_diff}" 