"""Reciprocal-Rank Fusion math."""

from __future__ import annotations

from collections.abc import Sequence

from app.repositories.chunks import ChunkMatch
from app.services.retrieval import fuse_rrf


def _match(chunk_id: int, similarity: float = 0.9) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"content {chunk_id}",
        source_url=None,
        department_allowed="Engineering",
        clearance_level=2,
        similarity=similarity,
    )


def test_fusion_deduplicates_and_boosts_overlap() -> None:
    matches: Sequence[ChunkMatch] = [_match(i) for i in range(1, 4)]
    vector, fts = list(matches), [matches[1], matches[2]]  # ids: [1,2,3], [2,3]

    fused = fuse_rrf(vector, fts)

    # id 2 (and 3) gain a second list hit; id 2 leads, 3 trails id 1's
    # single strong hit behind its double hit but keeps rank over 1.
    assert fused[0] is matches[1]
    assert len(fused) == 3
    assert [match.chunk_id for match in fused] == [2, 3, 1]


def test_fusion_ties_break_deterministically_by_chunk_id() -> None:
    fused = fuse_rrf([_match(7), _match(2)], [_match(5)])

    # ids 7 and 5 tie at rank-1 points; both beat id 2's rank-2 point.
    assert [match.chunk_id for match in fused] == [5, 7, 2]


def test_fusion_handles_empty_rankings() -> None:
    assert fuse_rrf() == []
    assert fuse_rrf([], []) == []
