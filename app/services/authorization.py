"""Central authorization rules (RBAC + clearance policy).

Roles: ``viewer`` reads; ``editor`` additionally ingests within their own
department and clearance; ``admin`` manages users and may ingest any document,
including public/cross-department ones that the seed script previously created.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import PermissionDeniedError
from app.core.security import ROLES, CurrentUser

ROLE_VIEWER = "viewer"
ROLE_EDITOR = "editor"
ROLE_ADMIN = "admin"

assert set((ROLE_VIEWER, ROLE_EDITOR, ROLE_ADMIN)) == set(ROLES)


def ensure_role(user: CurrentUser, allowed: frozenset[str], message: str) -> None:
    """Raise unless the principal's role is in ``allowed``."""
    if user.role not in allowed:
        raise PermissionDeniedError(message)


def ensure_admin(user: CurrentUser) -> None:
    """Gate for user administration (``/auth/register``)."""
    ensure_role(user, frozenset({ROLE_ADMIN}), "Only admins can manage users.")


def authorize_ingestion(
    *,
    user: CurrentUser,
    department: str,
    clearance_level: int,
    settings: Settings,
) -> None:
    """Least-privilege ingestion policy.

    - Admins may write to any department and any clearance (including the
      public ``'All'`` department that was formerly seed-script-only).
    - Editors may only ingest documents for their own department, at or below
      their own clearance, and above the configured minimum clearance.
    - Viewers may not ingest at all.
    """
    if user.role == ROLE_ADMIN:
        return
    ensure_role(
        user,
        frozenset({ROLE_EDITOR}),
        "Your role does not allow document ingestion.",
    )
    if user.clearance < settings.ingest_min_clearance:
        raise PermissionDeniedError(
            "Your clearance level does not allow document ingestion."
        )
    if department != user.department:
        raise PermissionDeniedError(
            "You may only ingest documents for your own department."
        )
    if clearance_level > user.clearance:
        raise PermissionDeniedError(
            "You may not create documents above your own clearance level."
        )
