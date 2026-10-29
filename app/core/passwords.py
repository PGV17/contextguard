"""Password hashing with bcrypt (evaluation cost is deliberately modest for demos)."""

from __future__ import annotations

import bcrypt

_ROUNDS = 10


def hash_password(password: str) -> str:
    """Hash a plaintext password and return the bcrypt digest."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time verification; malformed digests simply fail."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False
