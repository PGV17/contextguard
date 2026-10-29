"""Embedding providers.

Two interchangeable implementations share the ``EmbeddingService`` protocol:

* ``openai`` - ``text-embedding-3-small`` via the OpenAI async SDK (production).
* ``local``  - ``sentence-transformers/all-MiniLM-L6-v2`` on CPU (zero-cost,
  offline, used for development and CI).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, UpstreamServiceError

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    """Contract shared by all embedding backends."""

    model_name: str
    dimensions: int

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingService:
    """OpenAI embeddings with batching and bounded retries."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ConfigurationError(
                "CG_OPENAI_API_KEY is not configured. Set it, or switch "
                "CG_EMBEDDING_PROVIDER=local and install the 'local' extra."
            )
        self.model_name = settings.embedding_model
        self.dimensions = settings.vector_dimensions
        self._batch_size = settings.embedding_batch_size
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,  # tenacity owns the retry policy
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8.0),
        reraise=True,
    )
    async def _request(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = await self._request(texts)
        except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
            raise UpstreamServiceError(
                "The embedding provider is unavailable after retries."
            ) from exc
        except OpenAIError as exc:
            logger.error("embedding request rejected", exc_info=True)
            raise UpstreamServiceError(
                "The embedding provider rejected the request."
            ) from exc

        self._validate(vectors)
        return vectors

    def _validate(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ConfigurationError(
                    f"Embedding model '{self.model_name}' returned {len(vector)} "
                    f"dimensions but CG_EMBEDDING_DIMENSIONS resolves to "
                    f"{self.dimensions}. Fix the configuration and re-index."
                )


class LocalEmbeddingService:
    """Sentence-Transformers embeddings, executed off the event loop."""

    def __init__(self, settings: Settings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigurationError(
                "Local embeddings require the 'local' extra: "
                "pip install -e '.[local]'"
            ) from exc

        logger.info(
            "loading local embedding model",
            extra={"model": settings.local_embedding_model},
        )
        self._model = SentenceTransformer(settings.local_embedding_model)
        self.model_name = settings.local_embedding_model
        get_dimensions = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or self._model.get_sentence_embedding_dimension
        raw_dimensions = get_dimensions()
        if raw_dimensions is None:
            raise ConfigurationError(
                f"Could not determine the embedding dimension of '{self.model_name}'."
            )
        self.dimensions = int(raw_dimensions)

        if self.dimensions != settings.vector_dimensions:
            raise ConfigurationError(
                f"'{self.model_name}' produces {self.dimensions}-dimensional vectors, "
                f"but the configured dimension is {settings.vector_dimensions}. "
                f"Set CG_EMBEDDING_DIMENSIONS={self.dimensions} and apply migrations."
            )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        matrix = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array.tolist()


def create_embedding_service(settings: Settings) -> EmbeddingService:
    """Build the embedding backend selected by configuration."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingService(settings)
    return LocalEmbeddingService(settings)
