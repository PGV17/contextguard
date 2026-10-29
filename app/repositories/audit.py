"""Append-only audit trail persistence."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog


class AuditRepository:
    """Writes one audit row per RAG request, including the permission context."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        username: str,
        department: str,
        clearance_level: int,
        query: str,
        retrieved_chunk_ids: Sequence[int],
        results_found: bool,
        response: str | None,
    ) -> int:
        entry = AuditLog(
            username=username,
            department=department,
            clearance_level=clearance_level,
            query=query,
            retrieved_chunk_ids=list(retrieved_chunk_ids),
            results_found=results_found,
            response=response,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry.id
