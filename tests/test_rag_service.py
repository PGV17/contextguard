"""RAG orchestration tests using in-memory stubs (no database, no network)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import UpstreamServiceError, ValidationFailedError
from app.core.security import CurrentUser
from app.repositories.audit import AuditRepository
from app.repositories.chunks import ChunkMatch
from app.services.llm import INSUFFICIENT_CONTEXT_MESSAGE, ContextBlock
from app.services.rag import RagService, resolve_citations
from app.services.retrieval import RetrievalService


class StubEmbeddings:
    model_name = "stub-embeddings"
    dimensions = 384

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.01] * self.dimensions

    async def embed_documents(self, texts):
        return [[0.01] * self.dimensions for _ in texts]


class StubLLM:
    model_name = "stub-llm"

    def __init__(
        self,
        answer: str = "The production database password is admin123 [Doc 1].",
        error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._error = error
        self.calls: list[tuple[str, list[ContextBlock]]] = []

    async def synthesize(self, *, question: str, context) -> str:
        self.calls.append((question, list(context)))
        if self._error is not None:
            raise self._error
        return self._answer


def make_match(**overrides) -> ChunkMatch:
    values = {
        "chunk_id": 1,
        "document_id": "eng-db-runbook",
        "content": "The production database password is 'admin123'.",
        "source_url": "https://intranet.example.com/engineering/db-runbook",
        "department_allowed": "Engineering",
        "clearance_level": 2,
        "similarity": 0.82,
    }
    values.update(overrides)
    return ChunkMatch(**values)


def build_service(
    settings: Settings,
    *,
    matches: list[ChunkMatch],
    llm: StubLLM | None = None,
):
    retrieval = AsyncMock(spec=RetrievalService)
    retrieval.retrieve.return_value = matches
    audits = AsyncMock(spec=AuditRepository)
    service = RagService(
        retrieval=retrieval,
        audits=audits,
        embeddings=StubEmbeddings(),
        llm=llm or StubLLM(),
        settings=settings,
    )
    return service, retrieval, audits


async def test_no_matches_returns_safe_answer_without_llm(
    settings: Settings, engineer: CurrentUser
) -> None:
    llm = StubLLM()
    service, _, audits = build_service(settings, matches=[], llm=llm)

    result = await service.answer(question="What is the password?", user=engineer)

    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.cited == []
    assert result.retrieved_chunks == 0
    assert llm.calls == []

    audit = audits.record.await_args.kwargs
    assert audit["retrieved_chunk_ids"] == []
    assert audit["results_found"] is False
    assert audit["response"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert audit["role"] == "editor"


async def test_retrieval_receives_principal_and_limit(
    settings: Settings, engineer: CurrentUser
) -> None:
    service, retrieval, _ = build_service(settings, matches=[make_match()])

    await service.answer(question="What is the password?", user=engineer, top_k=3)

    query = retrieval.retrieve.await_args.kwargs
    assert query["question"] == "What is the password?"
    assert query["user"] is engineer
    assert query["limit"] == 3


async def test_top_k_is_clamped_to_the_server_maximum(
    settings: Settings, engineer: CurrentUser
) -> None:
    service, retrieval, _ = build_service(settings, matches=[make_match()])

    await service.answer(question="query", user=engineer, top_k=50)

    assert retrieval.retrieve.await_args.kwargs["limit"] == settings.retrieval_max_top_k


async def test_audit_records_query_chunk_ids_and_response(
    settings: Settings, engineer: CurrentUser
) -> None:
    answer = "The production database password is admin123 [Doc 1]."
    service, _, audits = build_service(
        settings,
        matches=[make_match(chunk_id=11), make_match(chunk_id=12, document_id="other")],
        llm=StubLLM(answer=answer),
    )

    await service.answer(question="What is the password?", user=engineer)

    audit = audits.record.await_args.kwargs
    assert audit["username"] == "engineer"
    assert audit["department"] == "Engineering"
    assert audit["clearance_level"] == 2
    assert audit["query"] == "What is the password?"
    assert audit["retrieved_chunk_ids"] == [11, 12]
    assert audit["results_found"] is True
    assert audit["response"] == answer


async def test_llm_failure_is_audited_then_reraised(
    settings: Settings, engineer: CurrentUser
) -> None:
    service, _, audits = build_service(
        settings,
        matches=[make_match()],
        llm=StubLLM(error=UpstreamServiceError("provider down")),
    )

    with pytest.raises(UpstreamServiceError):
        await service.answer(question="What is the password?", user=engineer)

    audit = audits.record.await_args.kwargs
    assert audit["results_found"] is True
    assert audit["response"] is None


async def test_empty_query_is_rejected(
    settings: Settings, engineer: CurrentUser
) -> None:
    service, _, _ = build_service(settings, matches=[])

    with pytest.raises(ValidationFailedError):
        await service.answer(question="   ", user=engineer)


def test_citations_are_resolved_in_order_and_deduplicated() -> None:
    context = [
        ContextBlock(1, 10, "doc-1", "first", None, "Engineering", 2, 0.9),
        ContextBlock(2, 11, "doc-2", "second", None, "Engineering", 2, 0.8),
    ]

    cited = resolve_citations("A [Doc 2]. B [Doc 1][Doc 2] [Doc 99].", context)

    assert [block.index for block in cited] == [2, 1]


def test_unknown_citation_markers_are_ignored() -> None:
    context = [ContextBlock(1, 10, "doc-1", "first", None, "Engineering", 2, 0.9)]

    assert resolve_citations("Fabricated [Doc 42].", context) == []
