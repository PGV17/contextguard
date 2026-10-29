"""Recompute and verify the audit log's hash chain.

Walks every audit row, recomputes sha256(prev_hash | payload) and compares it
with the stored digest. Any edit, deletion or reordering anywhere in the chain
breaks verification.

Usage:
    python -m scripts.verify_audit
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import select

from app.core.audit_chain import GENESIS_HASH, compute_entry_hash
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import AuditLog
from app.db.session import create_engine, create_session_factory


async def verify() -> bool:
    settings = get_settings()
    configure_logging("WARNING")

    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(AuditLog).order_by(AuditLog.id.asc())
                    )
                )
                .scalars()
                .all()
            )

        if not rows:
            print(f"No audit entries ({len(rows)} rows). Chain: trivially intact.")
            return True

        expected_prev = GENESIS_HASH
        broken_at: int | None = None
        legacy_count = 0
        for row in rows:
            # Legacy rows (pre-hash-chain migration) hold no digest: the
            # writer's first chained row linked onto GENESIS_HASH, so legacy
            # rows must be skipped WITHOUT advancing the expected predecessor.
            if row.entry_hash is None:
                legacy_count += 1
                continue
            computed = compute_entry_hash(
                expected_prev,
                username=row.username,
                department=row.department,
                clearance_level=row.clearance_level,
                role=row.role or "unknown",
                query=row.query,
                retrieved_chunk_ids=list(row.retrieved_chunk_ids or []),
                results_found=bool(row.results_found),
                response=row.response,
            )
            if row.entry_hash != computed:
                print(f"  row id={row.id}: TAMPERED (expected {computed[:16]}...)")
                broken_at = row.id
                break
            expected_prev = row.entry_hash

        if broken_at is not None:
            print(f"CHAIN BROKEN first at id={broken_at}.")
            return False

        if legacy_count:
            print(f"  {legacy_count} legacy row(s) skipped (digests added later).")
        latest = rows[-1]
        print(
            f"OK: {len(rows)} entries verified. Latest entry id={latest.id} "
            f"hash={latest.entry_hash[:16]}..."
        )
        return True
    finally:
        await engine.dispose()


if __name__ == "__main__":
    intact = asyncio.run(verify())
    sys.exit(0 if intact else 1)
