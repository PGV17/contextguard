"""Seed demo users and documents.

This is a trusted, system-level operation: the seed script runs outside the
API's authorization rules so public and cross-department documents can exist
(admins can also create these via ``POST /documents/ingest``). Idempotent -
re-running replaces the demo rows by id.

Usage:
    python -m scripts.seed
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.passwords import hash_password
from app.db.models import DocumentChunk, User
from app.db.session import create_engine, create_session_factory
from app.services.chunking import split_text
from app.services.embeddings import create_embedding_service

logger = logging.getLogger(__name__)

DEMO_PASSWORD = "demo-password"

# (username, department, clearance, role) - demo credentials only.
DEMO_USERS: tuple[tuple[str, str, int, str], ...] = (
    ("admin", "All", 5, "admin"),
    ("engineer", "Engineering", 2, "editor"),
    ("hr_lead", "HR", 2, "editor"),
    ("intern", "Marketing", 1, "viewer"),
)


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


async def seed_users(session) -> int:
    """Replace demo users by username (idempotent)."""
    usernames = [username for username, *_ in DEMO_USERS]
    await session.execute(
        delete(User).where(User.username.in_(usernames))
    )
    for username, department, clearance, role in DEMO_USERS:
        session.add(
            User(
                username=username,
                password_hash=hash_password(DEMO_PASSWORD),
                department=department,
                clearance=clearance,
                role=role,
            )
        )
        logger.info(
            "seeded user",
            extra={"username": username, "role": role, "department": department},
        )
    return len(DEMO_USERS)


async def seed_documents(session, settings, embedding_service) -> int:
    """Replace demo documents by document_id (idempotent)."""
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
    return total_chunks


async def seed() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    embedding_service = create_embedding_service(settings)
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            user_count = await seed_users(session)
            chunk_count = await seed_documents(session, settings, embedding_service)

            # Sanity check: the directory really is queryable afterwards.
            await session.flush()  # autoflush is off; make INSERTs visible first
            usernames = await session.execute(select(User.username))
            assert usernames.scalars().all(), "user seed must be readable"

            await session.commit()
            logger.info(
                "seed complete",
                extra={
                    "users": user_count,
                    "documents": len(SEED_DOCUMENTS),
                    "chunks": chunk_count,
                    "embedding_model": embedding_service.model_name,
                },
            )
            print(
                f"Seeded {user_count} users and {len(SEED_DOCUMENTS)} documents "
                f"({chunk_count} chunks). Demo password: {DEMO_PASSWORD}"
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
