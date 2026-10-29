"""Authentication and user-management request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UserRole = Literal["viewer", "editor", "admin"]


class TokenRequest(BaseModel):
    """Credentials authenticated against the ``users`` table (bcrypt)."""

    username: str = Field(min_length=1, max_length=64, examples=["engineer"])
    password: str = Field(min_length=1, max_length=128, examples=["demo-password"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")
    username: str
    department: str
    clearance: int
    role: str


class RegisterRequest(BaseModel):
    """User creation - reserved for admins."""

    username: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9._-]+$",
        examples=["qa_analyst"],
    )
    password: str = Field(min_length=8, max_length=128)
    department: str = Field(min_length=1, max_length=64, examples=["Marketing"])
    clearance: int = Field(ge=1, le=5, examples=[1])
    role: UserRole = "viewer"


class UserResponse(BaseModel):
    username: str
    role: str
    department: str
    clearance: int
