"""Application configuration.

All settings are loaded from environment variables prefixed with ``CG_`` and,
when present, from a local ``.env`` file. ``Settings`` is the single source of
truth for runtime configuration and is injected through FastAPI dependencies.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SECRET_KEY = "contextguard-development-secret-key-change-me"

EmbeddingProviderName = Literal["openai", "local"]
LLMProviderName = Literal["openai", "extractive"]


class Settings(BaseSettings):
    """Runtime configuration for the ContextGuard service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CG_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application --------------------------------------------------------
    app_name: str = "ContextGuard"
    app_version: str = "2.0.0"
    environment: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # --- Security -----------------------------------------------------------
    secret_key: str = DEFAULT_SECRET_KEY
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=60, ge=1, le=1440)

    # --- Database -----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://postgres:password@127.0.0.1:5432/rag_db"
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=10, ge=0, le=50)
    db_echo: bool = False
    hnsw_ef_search: int = Field(default=100, ge=1, le=1000)

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = Field(default=None, ge=1, le=8192)
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, ge=1, le=512)

    # --- LLM ----------------------------------------------------------------
    llm_provider: LLMProviderName = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=800, ge=64, le=4096)
    openai_api_key: str | None = None
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    # --- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_max_top_k: int = Field(default=20, ge=1, le=50)
    retrieval_min_similarity: float = Field(default=0.65, ge=0.0, le=1.0)
    public_department: str = Field(default="All", min_length=1, max_length=64)
    max_context_characters: int = Field(default=8000, ge=500)

    # --- Ingestion ----------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    max_chunks_per_document: int = Field(default=500, ge=1)
    max_document_characters: int = Field(default=2_000_000, ge=1000)
    ingest_min_clearance: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def _validate_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.retrieval_top_k > self.retrieval_max_top_k:
            raise ValueError("retrieval_top_k must not exceed retrieval_max_top_k")
        return self

    @property
    def vector_dimensions(self) -> int:
        """Resolve the pgvector dimension from explicit config or the provider default."""
        if self.embedding_dimensions is not None:
            return self.embedding_dimensions
        return 1536 if self.embedding_provider == "openai" else 384

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
