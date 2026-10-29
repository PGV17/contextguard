"""RAG orchestration: embed -> retrieve -> fuse/re-rank -> synthesize -> audit."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import ValidationFailedError
from app.core.metrics import INSUFFICIENT_CONTEXT, RETRIEVED_CHUNKS
from app.core.security import CurrentUser
from app.repositories.audit import AuditRepository
from app.repositories.chunks import ChunkMatch
from app.services.embeddings import EmbeddingService
from app.services.llm import INSUFFICIENT_CONTEXT_MESSAGE, ContextBlock, LLMService
from app.services.retrieval import RetrievalService

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
        retrieval: RetrievalService,
        audits: AuditRepository,
        embeddings: EmbeddingService,
        llm: LLMService,
        settings: Settings,
    ) -> None:
        self._retrieval = retrieval
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

        # 1) Embed the question (the only step that talks to a provider).
        query_embedding = await self._embeddings.embed_query(question)

        # 2) Retrieve only rows this principal may read; RLS, the similarity
        #    threshold, hybrid fusion and re-ranking all happen here.
        matches = await self._retrieval.retrieve(
            question=question,
            query_embedding=query_embedding,
            user=user,
            limit=limit,
        )

        context = self._build_context(matches)
        logger.info(
            "retrieval completed",
            extra={
                "username": user.username,
                "role": user.role,
                "department": user.department,
                "clearance": user.clearance,
                "limit": limit,
                "min_similarity": self._settings.retrieval_min_similarity,
                "matches": len(matches),
                "context_chunks": len(context),
            },
        )
        RETRIEVED_CHUNKS.labels(department=user.department).inc(len(context))

        # 3) Nothing cleared the threshold: answer safely without an LLM call.
        if not context:
            INSUFFICIENT_CONTEXT.labels(department=user.department).inc()
            await self._record(
                user=user, query=question, chunk_ids=[], results_found=False,
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
            await self._record(
                user=user, query=question,
                chunk_ids=[block.chunk_id for block in context],
                results_found=True, response=None,
            )
            raise

        cited = resolve_citations(answer_text, context)

        # 5) Audit the full interaction: query, permitted chunk ids, response.
        await self._record(
            user=user, query=question,
            chunk_ids=[block.chunk_id for block in context],
            results_found=True, response=answer_text,
        )

        return RagResult(
            answer=answer_text,
            cited=cited,
            retrieved_chunks=len(context),
            model=self._llm.model_name,
        )

    async def _record(
        self,
        *,
        user: CurrentUser,
        query: str,
        chunk_ids: list[int],
        results_found: bool,
        response: str | None,
    ) -> None:
        await self._audits.record(
            username=user.username,
            department=user.department,
            clearance_level=user.clearance,
            role=user.role,
            query=query,
            retrieved_chunk_ids=chunk_ids,
            results_found=results_found,
            response=response,
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
