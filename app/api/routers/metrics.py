"""Prometheus telemetry endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get(
    "",
    summary="Prometheus metrics",
    description=(
        "Expose request rates/latencies, retrieval hit counts and "
        "insufficient-context counters. Scrape without authentication and "
        "protect at the infrastructure layer."
    ),
)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
