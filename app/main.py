"""ContextGuard V2 application factory.

Startup owns the expensive singletons (engine, session factory, embedding
model, LLM client); shutdown releases them. Routers stay thin: validation,
authentication and delegation only.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.routers import auth, chat, documents, health, metrics
from app.core.config import DEFAULT_SECRET_KEY, Settings, get_settings
from app.core.exceptions import ConfigurationError, register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.db.health import verify_database_schema
from app.db.session import create_engine, create_session_factory
from app.services.embeddings import create_embedding_service
from app.services.llm import create_llm_service
from app.services.reranking import create_reranker

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired FastAPI application (used by uvicorn and tests)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if settings.is_production and settings.secret_key == DEFAULT_SECRET_KEY:
            raise ConfigurationError(
                "CG_SECRET_KEY must be overridden with a strong secret in production."
            )

        engine = create_engine(settings)
        try:
            await verify_database_schema(engine, settings.vector_dimensions)
            application.state.engine = engine
            application.state.session_factory = create_session_factory(engine)
            application.state.embedding_service = create_embedding_service(settings)
            application.state.llm_service = create_llm_service(settings)
            application.state.reranker = create_reranker(settings)

            logger.info(
                "ContextGuard started",
                extra={
                    "environment": settings.environment,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_dimensions": settings.vector_dimensions,
                    "llm_provider": settings.llm_provider,
                "hybrid_enabled": settings.hybrid_enabled,
                "reranker_enabled": settings.reranker_enabled,
                },
            )
            yield
        finally:
            await engine.dispose()
            logger.info("ContextGuard stopped")

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Enterprise-grade, permission-filtered RAG engine. Retrieval is "
            "restricted by department and clearance inside the SQL query; "
            "answers carry inline citations and every request is audited."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(RequestContextMiddleware)
    register_exception_handlers(application)

    application.include_router(health.router, prefix="/health", tags=["operations"])
    application.include_router(metrics.router, prefix="/metrics", tags=["operations"])
    application.include_router(auth.router, prefix="/auth", tags=["authentication"])
    application.include_router(documents.router, prefix="/documents", tags=["documents"])
    application.include_router(chat.router, prefix="/chat", tags=["chat"])

    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    return application


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
