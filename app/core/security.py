"""JWT creation/verification and the authenticated user context.

The service is stateless: authorization data travels inside the signed token and
is never read from client-controlled request fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from app.core.config import Settings
from app.core.exceptions import AuthenticationError


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """Authenticated principal derived exclusively from a verified JWT."""

    username: str
    department: str
    clearance: int
    role: str


ROLES: tuple[str, ...] = ("viewer", "editor", "admin")


def create_access_token(
    *,
    username: str,
    department: str,
    clearance: int,
    role: str,
    settings: Settings,
) -> str:
    """Issue a signed access token carrying the principal's permissions."""
    now = datetime.now(UTC)
    payload = {
        "sub": username,
        "department": department,
        "clearance": clearance,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> CurrentUser:
    """Verify a token's signature/expiry and return the authenticated principal."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "department", "clearance", "role", "exp"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("The access token has expired.") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError("The access token is invalid.") from exc

    if payload.get("type") != "access":
        raise AuthenticationError("The token type is not supported.")

    department = payload.get("department")
    clearance = payload.get("clearance")
    role = payload.get("role")
    if not isinstance(department, str) or not department:
        raise AuthenticationError("The token is missing a valid department claim.")
    if isinstance(clearance, bool) or not isinstance(clearance, int) or clearance < 1:
        raise AuthenticationError("The token is missing a valid clearance claim.")
    if role not in ROLES:
        raise AuthenticationError("The token is missing a valid role claim.")

    return CurrentUser(
        username=str(payload["sub"]),
        department=department,
        clearance=clearance,
        role=role,
    )
