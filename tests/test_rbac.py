"""RBAC policy tests for ingestion and user administration."""

from __future__ import annotations

import pytest

from app.core.exceptions import PermissionDeniedError
from app.core.security import CurrentUser
from app.services.authorization import (
    ROLE_ADMIN,
    ROLE_EDITOR,
    ROLE_VIEWER,
    authorize_ingestion,
    ensure_admin,
)


def _user(role: str, clearance: int, department: str = "Engineering") -> CurrentUser:
    return CurrentUser(username="u", department=department, clearance=clearance, role=role)


def test_editor_may_ingest_own_department_within_clearance(settings) -> None:
    authorize_ingestion(
        user=_user(ROLE_EDITOR, 2),
        department="Engineering",
        clearance_level=2,
        settings=settings,
    )


def test_editor_cannot_ingest_above_own_clearance(settings) -> None:
    with pytest.raises(PermissionDeniedError):
        authorize_ingestion(
            user=_user(ROLE_EDITOR, 2),
            department="Engineering",
            clearance_level=3,
            settings=settings,
        )


def test_editor_cannot_ingest_cross_department(settings) -> None:
    with pytest.raises(PermissionDeniedError):
        authorize_ingestion(
            user=_user(ROLE_EDITOR, 2),
            department="HR",
            clearance_level=2,
            settings=settings,
        )


def test_viewers_may_not_ingest(settings) -> None:
    with pytest.raises(PermissionDeniedError):
        authorize_ingestion(
            user=_user(ROLE_VIEWER, 1, department="Marketing"),
            department="Marketing",
            clearance_level=1,
            settings=settings,
        )


def test_admin_may_ingest_any_department_and_clearance(settings) -> None:
    """Admins own the public/cross-department path (formerly seed-only)."""
    authorize_ingestion(
        user=_user(ROLE_ADMIN, 5, department="All"),
        department="HR",
        clearance_level=5,
        settings=settings,
    )


def test_editor_below_min_clearance_is_denied(settings) -> None:
    with pytest.raises(PermissionDeniedError):
        authorize_ingestion(
            user=_user(ROLE_EDITOR, 1, department="Marketing"),
            department="Marketing",
            clearance_level=1,
            settings=settings,
        )


def test_only_admins_manage_users(engineer, admin) -> None:
    ensure_admin(admin)

    with pytest.raises(PermissionDeniedError):
        ensure_admin(engineer)
