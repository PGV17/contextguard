"""Operational probes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import SessionDep, SettingsDep

router = APIRouter()


@router.get("", summary="Liveness probe")
async def liveness(settings: SettingsDep) -> dict[str, str]:
    return {"status": "ok", "version": settings.app_version}


@router.get("/ready", summary="Readiness probe")
async def readiness(session: SessionDep) -> dict[str, str]:
    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is unavailable.",
        ) from exc
    return {"status": "ready"}
