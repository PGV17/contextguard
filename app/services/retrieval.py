"""Hybrid retrieval: ANN similarity + Postgres full-text, fused by RRF.

Both legs enforce identical row-level-security predicates inside SQL, so the
fusion never mixes in rows the caller is not allowed to see. Reciprocal-Rank
Fusion is parameter-free, stable and cheap - ideal for a portfolio-grade,
database-only stack.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from app.core.config import Settings
from app.core.security import CurrentUser
from app.repositories.chunks import ChunkMatch, ChunkRepository
from app.services.authorization import ROLE_ADMIN
from app.services.reranking import RerankService

logger = logging.getLogger(__name__)


def fuse_rrf(
    *rankings: Sequence[ChunkMatch],
    k: int = 60,
) -> list[ChunkMatch]:
    """Reciprocal-Rank Fusion over any number of ranked result lists.

    score(d) = sum over lists of 1 / (k + rank(d)), ranks starting at 1.
    Ties break deterministically by chunk id so evaluations are reproducible.
    """
    scores: dict[int, float] = {}
    first_seen: dict[int, ChunkMatch] = {}
    for ranking in rankings:
        for position, match in enumerate(ranking):
            scores[match.chunk_id] = (
                scores.get(match.chunk_id, 0.0) + 1.0 / (k + position + 1)
            )
            first_seen.setdefault(match.chunk_id, match)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [first_seen[chunk_id] for chunk_id, _ in ordered]


class RetrievalService:
    """Step 2+3 of the pipeline: retrieve, fuse, re-rank.

    The caller embeds the question; this service decides *how* candidates are
    gathered (vector, vector+full-text) and in what order they are returned
    (optionally re-ranked by a cross-encoder).
    """

    def __init__(
        self,
        *,
        chunks: ChunkRepository,
        reranker: RerankService | None,
        settings: Settings,
    ) -> None:
        self._chunks = chunks
        self._reranker = reranker
        self._settings = settings

    async def retrieve(
        self,
        *,
        question: str,
        query_embedding: Sequence[float],
        user: CurrentUser,
        limit: int,
    ) -> list[ChunkMatch]:
        """Gather, fuse and rank permitted candidates; return at most ``limit``.

        Admins bypass the department predicate (but never the clearance cap)
        so they can audit and answer over the whole corpus.
        """
        include_all_departments = user.role == ROLE_ADMIN
        pool = max(limit, self._settings.retrieval_pool_size)

        vector_matches = await self._chunks.similarity_search(
            query_embedding=query_embedding,
            department=user.department,
            clearance_level=user.clearance,
            public_department=self._settings.public_department,
            limit=pool,
            min_similarity=self._settings.retrieval_min_similarity,
            hnsw_ef_search=self._settings.hnsw_ef_search,
            include_all_departments=include_all_departments,
        )

        results = vector_matches
        if self._settings.hybrid_enabled:
            text_matches = await self._chunks.full_text_search(
                query=question,
                department=user.department,
                clearance_level=user.clearance,
                public_department=self._settings.public_department,
                limit=pool,
                include_all_departments=include_all_departments,
            )
            results = fuse_rrf(
                vector_matches,
                text_matches,
                k=self._settings.hybrid_rrf_k,
            )

        if self._reranker is not None and results:
            results = await asyncio.to_thread(
                self._reranker.rerank, question=question, matches=results
            )

        return results[:limit]
