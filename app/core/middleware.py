"""HTTP middleware: request correlation, structured access logs, timing."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import reset_request_id, set_request_id
from app.core.metrics import HTTP_DURATION, HTTP_REQUESTS

logger = logging.getLogger("contextguard.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id to the context and emit one structured access log line."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = set_request_id(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            duration_seconds = time.perf_counter() - started
            route = request.scope.get("route")
            path = route.path if route is not None else request.url.path
            response.headers["X-Request-ID"] = request_id
            HTTP_REQUESTS.labels(
                method=request.method, path=path, status=response.status_code
            ).inc()
            HTTP_DURATION.labels(method=request.method, path=path).observe(
                duration_seconds
            )
            logger.info(
                "request completed",
                extra={
                    "http_method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_seconds * 1000, 2),
                },
            )
            return response
        finally:
            reset_request_id(token)
