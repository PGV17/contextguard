"""Mock identity directory.

V2 keeps the stateless, mock-token flow from V1 so the security model can be
demonstrated end to end. Replace this module with a real identity provider
(database-backed users, SSO, OIDC) in production.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class MockUser:
    username: str
    department: str
    clearance: int
    display_name: str


MOCK_USERS: Mapping[str, MockUser] = MappingProxyType(
    {
        "intern": MockUser(
            username="intern",
            department="Marketing",
            clearance=1,
            display_name="Mia - Marketing Intern",
        ),
        "hr_lead": MockUser(
            username="hr_lead",
            department="HR",
            clearance=2,
            display_name="Hannah - HR Lead",
        ),
        "engineer": MockUser(
            username="engineer",
            department="Engineering",
            clearance=2,
            display_name="Evan - Platform Engineer",
        ),
    }
)


def get_mock_user(username: str) -> MockUser | None:
    return MOCK_USERS.get(username)
