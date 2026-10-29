"""FastAPI dependency wiring.

Dependencies are the only place where the API layer touches infrastructure:
routers receive fully-constructed services and the authenticated principal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.security import CurrentUser, decode_access_token
from app.repositories.audit import AuditRepository
from app.repositories.chunks import ChunkRepository
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.llm import LLMService
from app.services.rag import RagService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Request-scoped session with commit-on-success / rollback-on-error."""
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CurrentUser:
    """Authenticate the bearer token and expose its signed permission claims."""
    if not token:
        raise AuthenticationError("A bearer token is required.")
    return decode_access_token(token, settings)


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def get_rag_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RagService:
    return RagService(
        chunks=ChunkRepository(session),
        audits=AuditRepository(session),
        embeddings=request.app.state.embedding_service,
        llm=request.app.state.llm_service,
        settings=settings,
    )


def get_ingestion_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionService:
    return IngestionService(
        chunks=ChunkRepository(session),
        embeddings=request.app.state.embedding_service,
        settings=settings,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
