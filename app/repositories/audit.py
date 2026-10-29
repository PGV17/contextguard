"""Append-only, hash-chained audit trail.

Each row stores ``sha256(prev_entry_hash | row_payload)`` so silent edits or
deletions anywhere in the chain are detectable by ``scripts/verify_audit.py``.
An advisory transaction lock serializes writes in the rare case of concurrent
RAG requests so two rows cannot claim the same predecessor.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit_chain import GENESIS_HASH, compute_entry_hash
from app.db.models import AuditLog

_CHAIN_ADVISORY_LOCK = 314159265


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
        role: str,
        query: str,
        retrieved_chunk_ids: Sequence[int],
        results_found: bool,
        response: str | None,
    ) -> int:
        entry_hash = await self._next_entry_hash(
            username=username,
            department=department,
            clearance_level=clearance_level,
            role=role,
            query=query,
            retrieved_chunk_ids=list(retrieved_chunk_ids),
            results_found=results_found,
            response=response,
        )
        entry = AuditLog(
            username=username,
            department=department,
            clearance_level=clearance_level,
            role=role,
            query=query,
            retrieved_chunk_ids=list(retrieved_chunk_ids),
            results_found=results_found,
            response=response,
            entry_hash=entry_hash,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry.id

    async def _next_entry_hash(self, **fields: object) -> str:
        """Serialize chain writes, then hash onto the newest row's digest."""
        await self._session.execute(
            text(f"SELECT pg_advisory_xact_lock({_CHAIN_ADVISORY_LOCK})")
        )
        previous = await self._session.execute(
            select(AuditLog.entry_hash)
            .where(AuditLog.entry_hash.is_not(None))
            .order_by(AuditLog.id.desc())
            .limit(1)
        )
        prev_hash = previous.scalar_one_or_none() or GENESIS_HASH
        return compute_entry_hash(prev_hash, **fields)  # type: ignore[arg-type]
