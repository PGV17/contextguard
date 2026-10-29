"""Fail fast on schema drift: connectivity, migrations, pgvector and dims.

Used by CI and by humans after switching embedding configuration.

Usage:
    python -m scripts.check_schema
"""

from __future__ import annotations

import asyncio
import sys

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.health import verify_database_schema
from app.db.session import create_engine


async def check() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_engine(settings)
    try:
        await verify_database_schema(engine, settings.vector_dimensions)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(check())
    except Exception as exc:  # ConfigurationError carries an actionable message
        print(f"SCHEMA CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Schema check passed.")
