"""Seed the corpus with demo documents.

This is a trusted, system-level operation: it bypasses the API's ingestion
authorization so public and cross-department documents can be created.
Idempotent - re-running replaces the demo documents by id.

Usage:
    python -m scripts.seed
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import delete

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import DocumentChunk
from app.db.session import create_engine, create_session_factory
from app.services.chunking import split_text
from app.services.embeddings import create_embedding_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SeedDocument:
    document_id: str
    content: str
    department_allowed: str
    clearance_level: int
    source_url: str | None


SEED_DOCUMENTS: tuple[SeedDocument, ...] = (
    SeedDocument(
        document_id="eng-db-runbook",
        content="The production database password is 'admin123'.",
        department_allowed="Engineering",
        clearance_level=2,
        source_url="https://intranet.example.com/engineering/db-runbook",
    ),
    SeedDocument(
        document_id="hr-exec-compensation",
        content="The CEO's annual bonus is $500,000.",
        department_allowed="HR",
        clearance_level=2,
        source_url="https://intranet.example.com/hr/executive-compensation",
    ),
    SeedDocument(
        document_id="company-holiday-notice",
        content="The office is closed for the Friday public holiday.",
        department_allowed="All",
        clearance_level=1,
        source_url="https://intranet.example.com/notices/holiday",
    ),
    SeedDocument(
        document_id="marketing-q3-campaign",
        content=(
            "The Q3 marketing campaign launches on October 30 with the tagline "
            "'Build Bold'."
        ),
        department_allowed="Marketing",
        clearance_level=1,
        source_url="https://intranet.example.com/marketing/q3-campaign",
    ),
    SeedDocument(
        document_id="hr-leave-policy",
        content="Employees receive 25 days of paid annual leave per calendar year.",
        department_allowed="HR",
        clearance_level=1,
        source_url="https://intranet.example.com/hr/leave-policy",
    ),
)


async def seed() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    embedding_service = create_embedding_service(settings)
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            await session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id.in_(
                        [document.document_id for document in SEED_DOCUMENTS]
                    )
                )
            )

            total_chunks = 0
            for document in SEED_DOCUMENTS:
                pieces = split_text(
                    document.content,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )
                vectors = await embedding_service.embed_documents(pieces)
                for index, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
                    session.add(
                        DocumentChunk(
                            document_id=document.document_id,
                            chunk_index=index,
                            content=piece,
                            embedding=vector,
                            department_allowed=document.department_allowed,
                            clearance_level=document.clearance_level,
                            source_url=document.source_url,
                        )
                    )
                total_chunks += len(pieces)
                logger.info(
                    "seeded document",
                    extra={
                        "document_id": document.document_id,
                        "department": document.department_allowed,
                        "clearance": document.clearance_level,
                        "chunks": len(pieces),
                    },
                )

            await session.commit()
            logger.info(
                "seed complete",
                extra={
                    "documents": len(SEED_DOCUMENTS),
                    "chunks": total_chunks,
                    "embedding_model": embedding_service.model_name,
                },
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
