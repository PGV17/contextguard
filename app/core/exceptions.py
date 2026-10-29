"""Domain exceptions and the uniform JSON error envelope.

Repositories and services raise domain errors; only this module knows how they
map to HTTP status codes. Unexpected exceptions are logged with a stack trace
and never leak internal details to the client.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_request_id

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for expected, domain-level failures."""

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, details: Sequence[Any] | None = None) -> None:
        self.message = message or self.default_message
        self.details = list(details) if details else None
        super().__init__(self.message)


class AuthenticationError(AppError):
    status_code = 401
    code = "authentication_failed"
    default_message = "Authentication failed."


class PermissionDeniedError(AppError):
    status_code = 403
    code = "permission_denied"
    default_message = "You do not have permission to perform this action."


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    default_message = "The requested resource was not found."


class ValidationFailedError(AppError):
    status_code = 422
    code = "validation_failed"
    default_message = "The request could not be processed."


class ConfigurationError(AppError):
    status_code = 500
    code = "configuration_error"
    default_message = "The service is misconfigured."


class UpstreamServiceError(AppError):
    status_code = 502
    code = "upstream_service_error"
    default_message = "An upstream dependency failed."


def _error_body(code: str, message: str, details: Sequence[Any] | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": list(details) if details else None,
            "request_id": get_request_id(),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers that render every failure through one envelope."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        log = logger.warning if exc.status_code < 500 else logger.error
        log(
            "%s: %s",
            exc.code,
            exc.message,
            extra={"path": request.url.path, "status_code": exc.status_code},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, exc.details),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        logger.info(
            "request validation failed",
            extra={"path": request.url.path, "error_count": len(details)},
        )
        return JSONResponse(
            status_code=422,
            content=_error_body("validation_failed", "Request validation failed.", details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error."
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body("http_error", message),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "An unexpected error occurred."),
        )
