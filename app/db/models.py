"""SQLAlchemy 2.0 models.

``document_chunks`` stores embedded chunks plus the authorization metadata that
drives row-level filtering. ``audit_logs`` is the append-only access trail.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Identity,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.db.base import Base

_VECTOR_DIMENSIONS = get_settings().vector_dimensions


class DocumentChunk(Base):
    """A permission-tagged, embedded slice of a source document."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(_VECTOR_DIMENSIONS), nullable=False
    )
    department_allowed: Mapped[str] = mapped_column(String(64), nullable=False)
    clearance_level: Mapped[int] = mapped_column(Integer, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # Composite index backing the row-level-security predicate.
        Index("ix_document_chunks_rls", "department_allowed", "clearance_level"),
        # Approximate nearest-neighbour index for cosine similarity search.
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<DocumentChunk id={self.id} document_id={self.document_id!r} "
            f"chunk_index={self.chunk_index} dept={self.department_allowed!r} "
            f"clearance={self.clearance_level}>"
        )


class AuditLog(Base):
    """Append-only record of every RAG query and its outcome."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(64), nullable=False)
    clearance_level: Mapped[int] = mapped_column(Integer, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[int]] = mapped_column(
        ARRAY(BigInteger), nullable=False, server_default="{}"
    )
    results_found: Mapped[bool] = mapped_column(Boolean, nullable=False)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_audit_logs_created_at", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<AuditLog id={self.id} username={self.username!r} "
            f"results_found={self.results_found}>"
        )
