"""Deterministic, boundary-aware text chunking.

The splitter targets paragraph boundaries first, then sentence/word boundaries,
never exceeding ``chunk_size`` characters and overlapping consecutive chunks so
that context spanning a boundary is not lost.
"""

from __future__ import annotations

_BOUNDARIES: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", " ")


def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` chars.

    Raises:
        ValueError: if the chunking parameters are inconsistent.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in the range [0, chunk_size)")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _find_boundary(normalized, start=start, end=end, chunk_size=chunk_size)

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _find_boundary(text: str, *, start: int, end: int, chunk_size: int) -> int:
    """Move ``end`` back to a natural boundary in the second half of the window."""
    window_start = start + chunk_size // 2
    for boundary in _BOUNDARIES:
        index = text.rfind(boundary, window_start, end)
        if index != -1:
            return index + len(boundary)
    return end
