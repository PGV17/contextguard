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
    Computed,
    DateTime,
    Identity,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
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
    # Stored full-text column backing the hybrid (BM25-style) retrieval leg.
    content_tsv: Mapped[object] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )
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
        # GIN index for the full-text leg of hybrid retrieval.
        Index(
            "ix_document_chunks_content_tsv",
            "content_tsv",
            postgresql_using="gin",
        ),
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
    role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[int]] = mapped_column(
        ARRAY(BigInteger), nullable=False, server_default="{}"
    )
    results_found: Mapped[bool] = mapped_column(Boolean, nullable=False)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Tamper-evident chain: sha256(prev_entry_hash | row payload).
    entry_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_audit_logs_created_at", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<AuditLog id={self.id} username={self.username!r} "
            f"results_found={self.results_found}>"
        )


class User(Base):
    """Directory-backed principal (replaces the V1 mock in-memory users)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    department: Mapped[str] = mapped_column(String(64), nullable=False)
    clearance: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_users_department_role", "department", "role"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<User id={self.id} username={self.username!r} "
            f"role={self.role!r} department={self.department!r}>"
        )
