"""RAG chat completion endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUserDep, RagServiceDep
from app.schemas.chat import ChatRequest, ChatResponse, Citation

router = APIRouter()


@router.post(
    "/completions",
    response_model=ChatResponse,
    summary="Answer a question from permission-filtered documents",
    description=(
        "Runs the secure RAG pipeline: embed the query, retrieve only the "
        "chunks the caller may read, apply the similarity threshold, synthesize "
        "a cited answer, and audit the interaction."
    ),
)
async def create_completion(
    payload: ChatRequest,
    user: CurrentUserDep,
    service: RagServiceDep,
) -> ChatResponse:
    result = await service.answer(
        question=payload.query,
        user=user,
        top_k=payload.top_k,
    )
    return ChatResponse(
        answer=result.answer,
        citations=[
            Citation(
                marker=block.marker,
                chunk_id=block.chunk_id,
                document_id=block.document_id,
                source_url=block.source_url,
                similarity=round(block.similarity, 4),
            )
            for block in result.cited
        ],
        retrieved_chunks=result.retrieved_chunks,
        model=result.model,
    )
