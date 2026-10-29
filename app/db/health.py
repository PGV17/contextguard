"""Startup checks that fail fast with actionable messages."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_COLUMN_TYPE_SQL = text(
    """
    SELECT format_type(a.atttypid, a.atttypmod)
    FROM pg_attribute AS a
    WHERE a.attrelid = 'document_chunks'::regclass
      AND a.attname = 'embedding'
      AND NOT a.attisdropped
    """
)


async def verify_database_schema(engine: AsyncEngine, expected_dimensions: int) -> None:
    """Verify connectivity, migrations, pgvector and the configured vector size."""
    try:
        async with engine.connect() as connection:
            extension_version = await connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            column_type = await connection.scalar(_COLUMN_TYPE_SQL)
            has_hnsw_index = await connection.scalar(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM pg_indexes "
                    "  WHERE indexname = 'ix_document_chunks_embedding_hnsw'"
                    ")"
                )
            )
    except OperationalError as exc:
        raise ConfigurationError(
            "Could not connect to PostgreSQL. Is the database running? "
            "Start it with: docker compose up -d"
        ) from exc
    except ProgrammingError as exc:
        raise ConfigurationError(
            "The ContextGuard schema is missing. Apply migrations with: alembic upgrade head"
        ) from exc

    if not extension_version:
        raise ConfigurationError(
            "The pgvector extension is not enabled. Apply migrations with: alembic upgrade head"
        )

    if column_type is None:
        raise ConfigurationError(
            "Table 'document_chunks' has no embedding column. "
            "Apply migrations with: alembic upgrade head"
        )

    expected_type = f"vector({expected_dimensions})"
    if column_type != expected_type:
        raise ConfigurationError(
            f"Embedding dimension mismatch: the database column is {column_type}, "
            f"but configuration expects {expected_type}. Re-index the corpus with a "
            "migration or set CG_EMBEDDING_DIMENSIONS to match the stored vectors."
        )

    if not has_hnsw_index:
        logger.warning(
            "HNSW index is missing; similarity search will fall back to exact scan",
            extra={"index": "ix_document_chunks_embedding_hnsw"},
        )

    logger.info(
        "database schema verified",
        extra={
            "pgvector_version": str(extension_version),
            "embedding_type": column_type,
            "hnsw_index": bool(has_hnsw_index),
        },
    )
