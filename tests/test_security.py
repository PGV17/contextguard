"""JWT signing/verification, claim validation and bcrypt password hashing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest

from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.core.passwords import hash_password, verify_password
from app.core.security import CurrentUser, create_access_token, decode_access_token


def _encode(settings: Settings, payload: dict[str, Any], *, secret: str | None = None) -> str:
    return jwt.encode(
        payload, secret or settings.secret_key, algorithm=settings.jwt_algorithm
    )


def _valid_payload(settings: Settings) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "sub": "engineer",
        "department": "Engineering",
        "clearance": 2,
        "role": "editor",
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }


def test_create_and_decode_round_trip(settings: Settings) -> None:
    token = create_access_token(
        username="engineer",
        department="Engineering",
        clearance=2,
        role="editor",
        settings=settings,
    )

    user = decode_access_token(token, settings)

    assert user == CurrentUser(
        username="engineer", department="Engineering", clearance=2, role="editor"
    )


def test_expired_token_is_rejected(settings: Settings) -> None:
    now = datetime.now(UTC)
    payload = _valid_payload(settings)
    payload["iat"] = now - timedelta(hours=2)
    payload["exp"] = now - timedelta(hours=1)

    with pytest.raises(AuthenticationError):
        decode_access_token(_encode(settings, payload), settings)


def test_invalid_signature_is_rejected(settings: Settings) -> None:
    token = _encode(
        settings, _valid_payload(settings), secret="a-different-secret-key-0123456789"
    )

    with pytest.raises(AuthenticationError):
        decode_access_token(token, settings)


def test_missing_role_is_rejected(settings: Settings) -> None:
    payload = _valid_payload(settings)
    payload.pop("role")

    with pytest.raises(AuthenticationError):
        decode_access_token(_encode(settings, payload), settings)


def test_unknown_role_is_rejected(settings: Settings) -> None:
    payload = _valid_payload(settings)
    payload["role"] = "superuser"

    with pytest.raises(AuthenticationError):
        decode_access_token(_encode(settings, payload), settings)


def test_non_access_token_type_is_rejected(settings: Settings) -> None:
    payload = _valid_payload(settings)
    payload["type"] = "refresh"

    with pytest.raises(AuthenticationError):
        decode_access_token(_encode(settings, payload), settings)


def test_password_hash_round_trip() -> None:
    digest = hash_password("correct horse battery staple")

    assert digest != "correct horse battery staple"
    assert verify_password("correct horse battery staple", digest)
    assert not verify_password("wrong password", digest)


def test_malformed_digest_fails_closed() -> None:
    assert not verify_password("anything", "not-a-bcrypt-digest")
