"""User directory persistence for authentication and RBAC."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User


class UserRepository:
    """All SQL touching ``users`` lives here."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_username(self, username: str) -> User | None:
        result = await self._session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        username: str,
        password_hash: str,
        department: str,
        clearance: int,
        role: str,
    ) -> User:
        user = User(
            username=username,
            password_hash=password_hash,
            department=department,
            clearance=clearance,
            role=role,
        )
        self._session.add(user)
        await self._session.flush()
        return user
