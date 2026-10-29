"""Structured logging and request correlation.

Every log line is emitted as JSON and carries the current ``request_id`` so a
single user request can be traced across the API, service and repository layers.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar, Token
from typing import Any

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_LOG_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "request_id"}


def get_request_id() -> str:
    """Return the correlation id bound to the current request."""
    return _request_id.get()


def set_request_id(value: str) -> Token[str]:
    return _request_id.set(value)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """Minimal structured JSON formatter (no external dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                + f".{int(record.msecs):03d}Z"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once at process start."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Our access middleware emits richer logs; silence uvicorn's duplicate lines.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Third-party HTTP clients are noisy at INFO and do not add request context.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
