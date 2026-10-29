"""RAG orchestration: embed -> retrieve -> threshold -> synthesize -> audit."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import ValidationFailedError
from app.core.security import CurrentUser
from app.repositories.audit import AuditRepository
from app.repositories.chunks import ChunkMatch, ChunkRepository
from app.services.embeddings import EmbeddingService
from app.services.llm import INSUFFICIENT_CONTEXT_MESSAGE, ContextBlock, LLMService

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[Doc\s+(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RagResult:
    answer: str
    cited: list[ContextBlock]
    retrieved_chunks: int
    model: str


def resolve_citations(answer: str, context: Sequence[ContextBlock]) -> list[ContextBlock]:
    """Map [Doc N] markers in the answer to the chunks that were provided."""
    by_index = {block.index: block for block in context}
    cited: list[ContextBlock] = []
    seen: set[int] = set()
    for match in CITATION_PATTERN.finditer(answer):
        index = int(match.group(1))
        block = by_index.get(index)
        if block is not None and index not in seen:
            cited.append(block)
            seen.add(index)
    return cited


class RagService:
    """The permission-safe retrieval-augmented generation pipeline."""

    def __init__(
        self,
        *,
        chunks: ChunkRepository,
        audits: AuditRepository,
        embeddings: EmbeddingService,
        llm: LLMService,
        settings: Settings,
    ) -> None:
        self._chunks = chunks
        self._audits = audits
        self._embeddings = embeddings
        self._llm = llm
        self._settings = settings

    async def answer(
        self,
        *,
        question: str,
        user: CurrentUser,
        top_k: int | None = None,
    ) -> RagResult:
        question = question.strip()
        if not question:
            raise ValidationFailedError("The query must not be empty.")

        limit = self._resolve_limit(top_k)

        # 1) Embed the question locally or via the configured provider.
        query_embedding = await self._embeddings.embed_query(question)

        # 2) Retrieve only rows this principal may read; the RLS predicate and
        #    the similarity threshold both execute inside PostgreSQL.
        matches = await self._chunks.similarity_search(
            query_embedding=query_embedding,
            department=user.department,
            clearance_level=user.clearance,
            public_department=self._settings.public_department,
            limit=limit,
            min_similarity=self._settings.retrieval_min_similarity,
            hnsw_ef_search=self._settings.hnsw_ef_search,
        )

        context = self._build_context(matches)
        logger.info(
            "retrieval completed",
            extra={
                "username": user.username,
                "department": user.department,
                "clearance": user.clearance,
                "limit": limit,
                "min_similarity": self._settings.retrieval_min_similarity,
                "matches": len(matches),
                "context_chunks": len(context),
            },
        )

        # 3) Nothing cleared the threshold: answer safely without an LLM call.
        if not context:
            await self._audits.record(
                username=user.username,
                department=user.department,
                clearance_level=user.clearance,
                query=question,
                retrieved_chunk_ids=[],
                results_found=False,
                response=INSUFFICIENT_CONTEXT_MESSAGE,
            )
            return RagResult(
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                cited=[],
                retrieved_chunks=0,
                model=self._llm.model_name,
            )

        # 4) Synthesize. The audit row is written even when generation fails,
        #    so the retrieval attempt is never lost.
        try:
            answer_text = await self._llm.synthesize(question=question, context=context)
        except Exception:
            logger.exception("llm synthesis failed", extra={"username": user.username})
            await self._audits.record(
                username=user.username,
                department=user.department,
                clearance_level=user.clearance,
                query=question,
                retrieved_chunk_ids=[block.chunk_id for block in context],
                results_found=True,
                response=None,
            )
            raise

        cited = resolve_citations(answer_text, context)

        # 5) Audit the full interaction: query, permitted chunk ids, response.
        await self._audits.record(
            username=user.username,
            department=user.department,
            clearance_level=user.clearance,
            query=question,
            retrieved_chunk_ids=[block.chunk_id for block in context],
            results_found=True,
            response=answer_text,
        )

        return RagResult(
            answer=answer_text,
            cited=cited,
            retrieved_chunks=len(context),
            model=self._llm.model_name,
        )

    def _resolve_limit(self, top_k: int | None) -> int:
        requested = top_k if top_k is not None else self._settings.retrieval_top_k
        return max(1, min(requested, self._settings.retrieval_max_top_k))

    def _build_context(self, matches: Sequence[ChunkMatch]) -> list[ContextBlock]:
        """Number permitted chunks 1..N, respecting the context character budget."""
        blocks: list[ContextBlock] = []
        used_characters = 0
        for index, match in enumerate(matches, start=1):
            remaining = self._settings.max_context_characters - used_characters
            if blocks and len(match.content) > remaining:
                break
            blocks.append(
                ContextBlock(
                    index=index,
                    chunk_id=match.chunk_id,
                    document_id=match.document_id,
                    content=match.content,
                    source_url=match.source_url,
                    department_allowed=match.department_allowed,
                    clearance_level=match.clearance_level,
                    similarity=match.similarity,
                )
            )
            used_characters += len(match.content)
        return blocks
