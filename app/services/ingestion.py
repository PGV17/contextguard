"""Document ingestion: authorization, chunking, embedding and persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import (
    UpstreamServiceError,
    ValidationFailedError,
)
from app.core.security import CurrentUser
from app.db.models import DocumentChunk
from app.repositories.chunks import ChunkRepository
from app.schemas.documents import IngestRequest
from app.services.authorization import authorize_ingestion
from app.services.chunking import split_text
from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    chunk_ids: list[int]
    embedding_model: str


class IngestionService:
    """Chunk, embed and store a document under the caller's permissions."""

    def __init__(
        self,
        *,
        chunks: ChunkRepository,
        embeddings: EmbeddingService,
        settings: Settings,
    ) -> None:
        self._chunks = chunks
        self._embeddings = embeddings
        self._settings = settings

    async def ingest(
        self, *, payload: IngestRequest, user: CurrentUser
    ) -> IngestionResult:
        authorize_ingestion(
            user=user,
            department=payload.department_allowed,
            clearance_level=payload.clearance_level,
            settings=self._settings,
        )

        text = payload.content.strip()
        if not text:
            raise ValidationFailedError("Document content must not be empty.")
        if len(text) > self._settings.max_document_characters:
            raise ValidationFailedError(
                f"Document exceeds the maximum of "
                f"{self._settings.max_document_characters} characters."
            )

        try:
            pieces = split_text(
                text,
                chunk_size=self._settings.chunk_size,
                overlap=self._settings.chunk_overlap,
            )
        except ValueError as exc:
            raise ValidationFailedError(str(exc)) from exc

        if not pieces:
            raise ValidationFailedError("Document did not produce any chunks.")
        if len(pieces) > self._settings.max_chunks_per_document:
            raise ValidationFailedError(
                f"Document produces {len(pieces)} chunks, exceeding the limit of "
                f"{self._settings.max_chunks_per_document}. Reduce its size."
            )

        vectors = await self._embeddings.embed_documents(pieces)
        if len(vectors) != len(pieces):
            raise UpstreamServiceError(
                "The embedding provider returned an unexpected number of vectors."
            )

        document_id = payload.document_id or uuid4().hex
        rows = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                content=piece,
                embedding=vector,
                department_allowed=payload.department_allowed,
                clearance_level=payload.clearance_level,
                source_url=payload.source_url,
            )
            for index, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
        ]

        chunk_ids = await self._chunks.add_many(rows)
        logger.info(
            "document ingested",
            extra={
                "document_id": document_id,
                "chunks": len(chunk_ids),
                "department": payload.department_allowed,
                "clearance": payload.clearance_level,
                "username": user.username,
            },
        )

        return IngestionResult(
            document_id=document_id,
            chunk_ids=chunk_ids,
            embedding_model=self._embeddings.model_name,
        )
