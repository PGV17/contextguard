"""Audit hash-chain primitives (pure functions, no database required)."""

from __future__ import annotations

from app.core.audit_chain import GENESIS_HASH, canonical_payload, compute_entry_hash


def _fields(**overrides) -> dict:
    values = {
        "username": "engineer",
        "department": "Engineering",
        "clearance_level": 2,
        "role": "editor",
        "query": "What is the production database password?",
        "retrieved_chunk_ids": [1],
        "results_found": True,
        "response": "It is admin123 [Doc 1].",
    }
    values.update(overrides)
    return values


def _hash(prev_hash: str, fields: dict) -> str:
    return compute_entry_hash(prev_hash, **fields)  # type: ignore[arg-type]


def test_hash_is_deterministic() -> None:
    first = _hash(GENESIS_HASH, _fields())
    second = _hash(GENESIS_HASH, _fields())

    assert first == second
    assert len(first) == 64
    assert first != GENESIS_HASH


def test_chaining_changes_the_digest() -> None:
    root = _hash(GENESIS_HASH, _fields())
    child = _hash(root, _fields())

    assert child != root
    assert _hash(GENESIS_HASH, _fields()) == root  # chain is stable per parent


def test_any_payload_edit_breaks_the_link() -> None:
    digest = _hash(GENESIS_HASH, _fields())

    for mutations in (
        {"response": "Tampered [Doc 1]."},
        {"retrieved_chunk_ids": [2]},
        {"username": "intern"},
        {"results_found": False},
    ):
        assert _hash(GENESIS_HASH, _fields(**mutations)) != digest


def test_predecessor_is_included() -> None:
    a = _hash("f" * 64, _fields())
    b = _hash(GENESIS_HASH, _fields())

    assert a != b


def test_payload_is_canonical() -> None:
    assert canonical_payload({"b": 1, "a": 2}) == '{"a":2,"b":1}'
