"""Configuration helpers for the matching service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from the environment."""

    database_uri: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    pool_pre_ping: bool
    max_retries: int
    retry_backoff: float

    @property
    def engine_options(self) -> dict[str, object]:
        """Base keyword arguments used to build the SQLAlchemy engine."""

        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "future": True,
        }


def _to_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "t", "yes", "y"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read settings from environment variables with sensible defaults."""

    return Settings(
        database_uri=os.getenv(
            "DATABASE_URI",
            "postgresql+psycopg://meetinity:meetinity@localhost:5432/meetinity",
        ),
        pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE", "1800")),
        pool_pre_ping=_to_bool(os.getenv("DATABASE_POOL_PRE_PING"), default=True),
        max_retries=int(os.getenv("DATABASE_MAX_RETRIES", "3")),
        retry_backoff=float(os.getenv("DATABASE_RETRY_BACKOFF", "0.5")),
    )


def refresh_settings() -> Settings:
    """Clear the cached settings so changes to the environment are picked up."""

    get_settings.cache_clear()
    return get_settings()

