import torch
from typing import Any, List, Optional, Tuple

def build_segment_cu_seqlen(
    sorted_sequence_ids: torch.Tensor, tokens_per_expert: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct cu_seqlen for blocks that should not attend across sequence boundaries."""

    total_tokens = sorted_sequence_ids.numel()
    device = sorted_sequence_ids.device
    if total_tokens == 0:
        empty = torch.zeros(1, device=device, dtype=torch.int32)
        return empty, torch.empty(0, device=device, dtype=torch.int32)

    change = torch.zeros(total_tokens, device=device, dtype=torch.bool)
    change[0] = True

    if tokens_per_expert.numel() > 0:
        expert_offsets = tokens_per_expert.cumsum(0)[:-1]
        if expert_offsets.numel() > 0:
            change[expert_offsets.long()] = True

    if total_tokens > 1:
        seq_change_idx = torch.nonzero(
            sorted_sequence_ids[1:] != sorted_sequence_ids[:-1],
            as_tuple=False,
        ).flatten() + 1
        if seq_change_idx.numel() > 0:
            change[seq_change_idx] = True

    segment_starts = torch.nonzero(change, as_tuple=False).flatten().to(torch.int32)
    boundaries = torch.cat(
        (
            segment_starts,
            torch.tensor([total_tokens], device=device, dtype=torch.int32),
        )
    )

    segment_lengths = boundaries[1:] - boundaries[:-1]
    segment_lengths = segment_lengths.to(torch.int32)

    cu = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(segment_lengths.to(torch.int64), dim=0).to(torch.int32),
        )
    )
    return cu, segment_lengths

def test_build_segment_cu_seqlen_respects_sequence_and_expert_boundaries():
    sorted_seq_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32)
    tokens_per_expert = torch.tensor([3, 2], dtype=torch.int32)

    cu, m_split = build_segment_cu_seqlen(sorted_seq_ids, tokens_per_expert)

    assert cu.dtype == torch.int32
    assert m_split.dtype == torch.int32
    assert cu.tolist() == [0, 2, 3, 5]
    assert m_split.tolist() == [2, 1, 2]


def test_build_segment_cu_seqlen_handles_empty_inputs():
    sorted_seq_ids = torch.empty(0, dtype=torch.int32)
    tokens_per_expert = torch.empty(0, dtype=torch.int32)

    cu, m_split = build_segment_cu_seqlen(sorted_seq_ids, tokens_per_expert)

    assert cu.dtype == torch.int32
    assert cu.tolist() == [0]
    assert m_split.dtype == torch.int32
    assert m_split.numel() == 0
