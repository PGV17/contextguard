"""Authentication request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    """Credentials for the mock identity provider.

    ``password`` is accepted (and ignored) so OAuth2 password-form clients such
    as Swagger UI's authorize dialog work against the same endpoint.
    """

    username: str = Field(min_length=1, max_length=64, examples=["engineer"])
    password: str | None = Field(default=None, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")
    username: str
    department: str
    clearance: int
