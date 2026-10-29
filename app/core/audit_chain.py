"""Digest generation for the append-only audit hash chain (pure, testable)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS_HASH = "0" * 64


def canonical_payload(fields: dict[str, Any]) -> str:
    """Stable JSON encoding: sorted keys, no whitespace, no unicode escapes."""
    return json.dumps(
        fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def compute_entry_hash(
    prev_hash: str,
    *,
    username: str,
    department: str,
    clearance_level: int,
    role: str,
    query: str,
    retrieved_chunk_ids: list[int],
    results_found: bool,
    response: str | None,
) -> str:
    """Hash the previous entry's digest together with this row's payload."""
    payload = canonical_payload(
        {
            "username": username,
            "department": department,
            "clearance_level": clearance_level,
            "role": role,
            "query": query,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "results_found": results_found,
            "response": response,
        }
    )
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()
