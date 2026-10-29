"""Shared pytest fixtures (no database, no network)."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.security import CurrentUser


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings isolated from the developer's .env file."""
    return Settings(
        environment="test",
        secret_key="test-secret-key-0123456789abcdefghijklmnopqrstuvwxyz",
        embedding_provider="local",
        embedding_dimensions=384,
        llm_provider="extractive",
        openai_api_key=None,
        hybrid_enabled=True,
        reranker_enabled=False,
    )


@pytest.fixture
def engineer() -> CurrentUser:
    return CurrentUser(
        username="engineer", department="Engineering", clearance=2, role="editor"
    )


@pytest.fixture
def admin() -> CurrentUser:
    return CurrentUser(username="admin", department="All", clearance=5, role="admin")
