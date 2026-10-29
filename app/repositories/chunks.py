"""Document-chunk persistence and the permission-filtered vector search.

Security note: the department/clearance predicate and the similarity threshold
are part of the SQL statement itself. There is no application-side post-filter,
so a restricted row is never fetched into the process.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Select, or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """A permitted chunk returned by a similarity search."""

    chunk_id: int
    document_id: str
    content: str
    source_url: str | None
    department_allowed: str
    clearance_level: int
    similarity: float


class ChunkRepository:
    """All SQL touching ``document_chunks`` lives here."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, chunks: Sequence[DocumentChunk]) -> list[int]:
        """Persist chunks and return their generated identifiers."""
        self._session.add_all(list(chunks))
        await self._session.flush()
        return [chunk.id for chunk in chunks]

    async def similarity_search(
        self,
        *,
        query_embedding: Sequence[float],
        department: str,
        clearance_level: int,
        public_department: str,
        limit: int,
        min_similarity: float,
        hnsw_ef_search: int,
    ) -> list[ChunkMatch]:
        """Return the top-k permitted chunks ordered by cosine distance."""
        await self._configure_ann(hnsw_ef_search)

        statement = build_search_statement(
            query_embedding=query_embedding,
            department=department,
            clearance_level=clearance_level,
            public_department=public_department,
            limit=limit,
            min_similarity=min_similarity,
        )
        result = await self._session.execute(statement)
        return [
            ChunkMatch(
                chunk_id=row.id,
                document_id=row.document_id,
                content=row.content,
                source_url=row.source_url,
                department_allowed=row.department_allowed,
                clearance_level=row.clearance_level,
                similarity=float(row.similarity),
            )
            for row in result.all()
        ]

    async def _configure_ann(self, ef_search: int) -> None:
        """Tune the ANN session for filtered queries (pgvector >= 0.8).

        Filtered HNSW scans can under-return when the predicate is selective.
        ``iterative_scan`` keeps scanning until enough permitted rows are found.
        Both settings are transaction-local and rolled back harmlessly on older
        pgvector versions.
        """
        try:
            async with self._session.begin_nested():
                await self._session.execute(
                    text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
                )
                await self._session.execute(
                    text("SET LOCAL hnsw.iterative_scan = relaxed_order")
                )
        except SQLAlchemyError:
            logger.debug("pgvector session tuning is unavailable", exc_info=True)


def build_search_statement(
    *,
    query_embedding: Sequence[float],
    department: str,
    clearance_level: int,
    public_department: str,
    limit: int,
    min_similarity: float,
) -> Select[tuple[object, ...]]:
    """Build the RLS-enforcing similarity query.

    The WHERE clause is the row-level security boundary:

        (department_allowed = :department OR department_allowed = :public)
        AND clearance_level <= :clearance
        AND cosine_distance < 1 - :min_similarity

    The similarity threshold is expressed as a distance predicate so PostgreSQL
    can apply it during the HNSW index scan instead of loading irrelevant rows
    first.
    """
    distance = DocumentChunk.embedding.cosine_distance(list(query_embedding))
    return (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.source_url,
            DocumentChunk.department_allowed,
            DocumentChunk.clearance_level,
            (1.0 - distance).label("similarity"),
        )
        .where(
            or_(
                DocumentChunk.department_allowed == department,
                DocumentChunk.department_allowed == public_department,
            ),
            DocumentChunk.clearance_level <= clearance_level,
            distance < (1.0 - min_similarity),
        )
        .order_by(distance)
        .limit(limit)
    )
