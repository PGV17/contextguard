"""Authentication endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import ValidationError

from app.api.deps import SettingsDep
from app.core.exceptions import AuthenticationError, ValidationFailedError
from app.core.security import create_access_token
from app.core.users import get_mock_user
from app.schemas.auth import TokenRequest, TokenResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue a JWT access token",
    description=(
        "Exchanges a mock username for a signed access token carrying the "
        "principal's department and clearance level. Accepts JSON or an OAuth2 "
        "password form (for Swagger UI)."
    ),
)
async def issue_token(request: Request, settings: SettingsDep) -> TokenResponse:
    payload = await _parse_token_request(request)

    user = get_mock_user(payload.username.strip())
    if user is None:
        logger.info("token request rejected", extra={"username": payload.username})
        raise AuthenticationError("Unknown username.")

    token = create_access_token(
        username=user.username,
        department=user.department,
        clearance=user.clearance,
        settings=settings,
    )
    logger.info(
        "access token issued",
        extra={
            "username": user.username,
            "department": user.department,
            "clearance": user.clearance,
        },
    )
    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
        username=user.username,
        department=user.department,
        clearance=user.clearance,
    )


async def _parse_token_request(request: Request) -> TokenRequest:
    """Accept both JSON and OAuth2 password-form bodies on one endpoint."""
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()

    if content_type == "application/json":
        try:
            body = await request.json()
        except ValueError as exc:
            raise ValidationFailedError("Request body must be valid JSON.") from exc
        return _validate_token_request(body)

    if content_type in {"application/x-www-form-urlencoded", "multipart/form-data"}:
        form = await request.form()
        return _validate_token_request(
            {
                "username": form.get("username"),
                "password": form.get("password") or None,
            }
        )

    raise ValidationFailedError(
        "Content-Type must be application/json or application/x-www-form-urlencoded."
    )


def _validate_token_request(body: object) -> TokenRequest:
    try:
        return TokenRequest.model_validate(body)
    except ValidationError as exc:
        details = [
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        raise ValidationFailedError(
            "Authentication request failed validation.", details=details
        ) from exc
