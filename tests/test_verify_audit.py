"""Unit tests for ``scripts.verify_audit`` helpers that need no database."""

from scripts.verify_audit import _format_hash


def test_format_hash_none_renders_legacy():
    """A legacy row (no digest) must render as ``legacy``, not crash."""
    assert _format_hash(None) == "legacy"


def test_format_hash_empty_string_renders_legacy():
    """An empty digest is indistinguishable from missing for display."""
    assert _format_hash("") == "legacy"


def test_format_hash_truncates_to_sixteen_chars():
    """Only the first 16 hex characters are shown in the summary line."""
    digest = "a" * 64
    assert _format_hash(digest) == "a" * 16


def test_format_hash_shorter_digest_is_returned_whole():
    """Digests shorter than 16 characters are returned intact."""
    assert _format_hash("abcdef") == "abcdef"
