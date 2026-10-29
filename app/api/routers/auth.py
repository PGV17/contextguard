"""Authentication and user administration endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status
from pydantic import ValidationError

from app.api.deps import AdminDep, SettingsDep, UserRepositoryDep
from app.core.exceptions import AuthenticationError, ValidationFailedError
from app.core.passwords import hash_password, verify_password
from app.core.security import create_access_token
from app.schemas.auth import (
    RegisterRequest,
    TokenRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue a JWT access token",
    description=(
        "Authenticates against the users table (bcrypt) and issues a signed "
        "token carrying department, clearance and role. Accepts JSON or an "
        "OAuth2 password form (for Swagger UI)."
    ),
)
async def issue_token(
    request: Request,
    settings: SettingsDep,
    users: UserRepositoryDep,
) -> TokenResponse:
    payload = await _parse_token_request(request)

    user = await users.get_by_username(payload.username.strip())
    if user is None or not user.is_active or not verify_password(
        payload.password, user.password_hash
    ):
        logger.info(
            "token request rejected",
            extra={"username": payload.username},
        )
        # Same message for unknown users and bad passwords - no user probing.
        raise AuthenticationError("Invalid username or password.")

    token = create_access_token(
        username=user.username,
        department=user.department,
        clearance=user.clearance,
        role=user.role,
        settings=settings,
    )
    logger.info(
        "access token issued",
        extra={
            "username": user.username,
            "role": user.role,
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
        role=user.role,
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user (admin only)",
    description=(
        "Creates a directory user with hashed credentials and a role of "
        "viewer, editor or admin. Admins manage departments/clearances via "
        "this endpoint; viewers and editors cannot create users."
    ),
)
async def register_user(
    payload: RegisterRequest,
    admin: AdminDep,
    users: UserRepositoryDep,
) -> UserResponse:
    logger.info(
        "user registration requested",
        extra={"by": admin.username, "new": payload.username, "role": payload.role},
    )
    existing = await users.get_by_username(payload.username)
    if existing is not None:
        raise ValidationFailedError("That username is already taken.")

    created = await users.create(
        username=payload.username,
        password_hash=hash_password(payload.password),
        department=payload.department,
        clearance=payload.clearance,
        role=payload.role,
    )
    logger.info(
        "user created",
        extra={"username": created.username, "role": created.role},
    )
    return UserResponse(
        username=created.username,
        role=created.role,
        department=created.department,
        clearance=created.clearance,
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
                "password": form.get("password"),
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
