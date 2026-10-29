"""Document-chunk persistence and permission-filtered retrieval SQL.

Security notes:

* Every statement (vector *and* full-text) embeds the same row-level security
  predicate - restricted rows are never fetched into the process.
* The vector similarity threshold is part of the SQL so PostgreSQL applies it
  during the HNSW index scan instead of loading irrelevant rows first.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import (
    ColumnElement,
    Select,
    and_,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """A permitted chunk returned by a similarity or full-text search."""

    chunk_id: int
    document_id: str
    content: str
    source_url: str | None
    department_allowed: str
    clearance_level: int
    similarity: float
    text_rank: float | None = None


def rls_predicate(
    *,
    department: str,
    clearance_level: int,
    public_department: str,
    include_all_departments: bool = False,
) -> ColumnElement[bool]:
    """The row-level security boundary shared by every retrieval statement."""
    if include_all_departments:
        return DocumentChunk.clearance_level <= clearance_level
    return and_(
        or_(
            DocumentChunk.department_allowed == department,
            DocumentChunk.department_allowed == public_department,
        ),
        DocumentChunk.clearance_level <= clearance_level,
    )


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
        include_all_departments: bool = False,
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
            include_all_departments=include_all_departments,
        )
        result = await self._session.execute(statement)
        return [_match_from_row(row, text_rank=None) for row in result.all()]

    async def full_text_search(
        self,
        *,
        query: str,
        department: str,
        clearance_level: int,
        public_department: str,
        limit: int,
        include_all_departments: bool = False,
    ) -> list[ChunkMatch]:
        """Postgres full-text leg (RLS-enforced) ordered by ``ts_rank_cd``.

        Catches lexical matches that ANN misses: exact identifiers, product
        codes, names. The same security predicate applies - no exceptions.
        """
        statement = build_full_text_statement(
            query=query,
            department=department,
            clearance_level=clearance_level,
            public_department=public_department,
            limit=limit,
            include_all_departments=include_all_departments,
        )
        result = await self._session.execute(statement)
        rows = result.all()
        return [_match_from_row(row, text_rank=0.0) for row in rows]

    async def _configure_ann(self, ef_search: int) -> None:
        """Tune the ANN session for filtered queries (pgvector >= 0.8).

        Filtered HNSW scans can under-return when the predicate is selective.
        ``iterative_scan`` keeps scanning until enough permitted rows are found.
        Both settings are transaction-local and rolled back harmlessly on older
        pgvector versions.
        """
        # Full-text-only requests run inside the caller's transaction too.
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


def _match_from_row(row, *, text_rank: float | None) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=row.id,
        document_id=row.document_id,
        content=row.content,
        source_url=row.source_url,
        department_allowed=row.department_allowed,
        clearance_level=row.clearance_level,
        similarity=float(row.similarity),
        text_rank=text_rank,
    )


def _chunk_columns():
    return (
        DocumentChunk.id,
        DocumentChunk.document_id,
        DocumentChunk.content,
        DocumentChunk.source_url,
        DocumentChunk.department_allowed,
        DocumentChunk.clearance_level,
    )


def build_search_statement(
    *,
    query_embedding: Sequence[float],
    department: str,
    clearance_level: int,
    public_department: str,
    limit: int,
    min_similarity: float,
    include_all_departments: bool = False,
) -> Select:
    """Build the RLS-enforcing vector similarity query.

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
            *_chunk_columns(),
            (1.0 - distance).label("similarity"),
        )
        .where(
            rls_predicate(
                department=department,
                clearance_level=clearance_level,
                public_department=public_department,
                include_all_departments=include_all_departments,
            ),
            distance < (1.0 - min_similarity),
        )
        .order_by(distance)
        .limit(limit)
    )


def build_full_text_statement(
    *,
    query: str,
    department: str,
    clearance_level: int,
    public_department: str,
    limit: int,
    include_all_departments: bool = False,
) -> Select:
    """Build the RLS-enforcing full-text query over the stored tsvector.

    Each query term is stemmed by ``websearch_to_tsquery`` and the terms are
    OR-combined: AND semantics make the leg miss whenever one token is absent
    ("what is the admin123 password *used* for?"). ``ts_rank_cd`` restores
    precision afterwards - chunks matching more of the question outrank chunks
    matching a single common word.
    """
    terms = _fts_terms(query)
    ts_query = _combine_terms(terms)
    rank = func.ts_rank_cd(DocumentChunk.content_tsv, ts_query)
    return (
        select(
            *_chunk_columns(),
            (1.0 - rank).label("similarity"),
            rank.label("text_rank"),
        )
        .where(
            rls_predicate(
                department=department,
                clearance_level=clearance_level,
                public_department=public_department,
                include_all_departments=include_all_departments,
            ),
            DocumentChunk.content_tsv.op("@@")(ts_query),
        )
        .order_by(rank.desc())
        .limit(limit)
    )


def _fts_terms(query: str, *, max_terms: int = 12) -> list[str]:
    """Extract the query's word tokens (bounded for operator cardinality)."""
    terms = re.findall(r"\w+", query)
    return terms[:max_terms]


def _combine_terms(terms: list[str]) -> ColumnElement[bool]:
    """``w1 .. wN`` -> OR-combined tsquery (Postgres `||` ors tsquery values)."""
    expressions = [
        func.websearch_to_tsquery("english", term) for term in terms
    ] or [func.websearch_to_tsquery("english", "")]
    combined: ColumnElement[bool] = expressions[0]
    for expression in expressions[1:]:
        combined = combined.op("||")(expression)
    return combined
