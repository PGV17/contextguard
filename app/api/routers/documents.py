"""Document ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, IngestionServiceDep
from app.schemas.documents import IngestRequest, IngestResponse

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest, chunk and embed a document",
    description=(
        "Chunks the supplied text, generates embeddings and stores the chunks "
        "with permission metadata. Callers may only ingest documents for their "
        "own department and at or below their own clearance level."
    ),
)
async def ingest_document(
    payload: IngestRequest,
    user: CurrentUserDep,
    service: IngestionServiceDep,
) -> IngestResponse:
    result = await service.ingest(payload=payload, user=user)
    return IngestResponse(
        document_id=result.document_id,
        chunks_created=len(result.chunk_ids),
        chunk_ids=result.chunk_ids,
        embedding_model=result.embedding_model,
    )
