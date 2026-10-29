"""Re-ranking providers.

A cross-encoder scores every (question, chunk) pair jointly, which is far more
accurate than the bi-encoder similarity used for ANN search - at the price of
running over a small candidate pool instead of the whole corpus. Deliberately
optional so deployments stay lean without the ``local`` extra.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from app.core.config import Settings
from app.repositories.chunks import ChunkMatch

logger = logging.getLogger(__name__)


class RerankService(Protocol):
    """Contract shared by re-ranking backends."""

    model_name: str

    def rerank(
        self, *, question: str, matches: Sequence[ChunkMatch]
    ) -> list[ChunkMatch]:
        ...


class CrossEncoderReranker:
    """``sentence-transformers`` cross-encoder, loaded lazily on first use."""

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.reranker_model
        self._model = None

    def rerank(
        self, *, question: str, matches: Sequence[ChunkMatch]
    ) -> list[ChunkMatch]:
        model = self._ensure_model()
        pairs = [(question, match.content) for match in matches]
        scores = model.predict(pairs)  # type: ignore[union-attr]

        scored = sorted(
            zip(matches, (float(score) for score in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [match for match, _ in scored]

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading cross-encoder re-ranker", extra={"model": self.model_name})
            self._model = CrossEncoder(self.model_name)
        return self._model


def create_reranker(settings: Settings) -> RerankService | None:
    """Build the re-ranker selected by configuration (or ``None`` off)."""
    if settings.reranker_enabled:
        return CrossEncoderReranker(settings)
    return None
