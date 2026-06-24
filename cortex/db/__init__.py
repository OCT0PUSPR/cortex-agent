"""Persistence layer for cortex-agent (SQLAlchemy 2.0 async)."""

from __future__ import annotations

from .engine import (
    create_all,
    dispose_engine,
    drop_all,
    get_engine,
    get_session,
    get_sessionmaker,
    session_scope,
)
from .models import ApiKey, Base, MemoryEntry, Run, RunEvent, Session, User, metadata
from .repository import MemoryRepository, RunRepository, UserRepository

__all__ = [
    "Base",
    "metadata",
    "User",
    "ApiKey",
    "Session",
    "Run",
    "RunEvent",
    "MemoryEntry",
    "get_engine",
    "get_sessionmaker",
    "get_session",
    "session_scope",
    "create_all",
    "drop_all",
    "dispose_engine",
    "UserRepository",
    "RunRepository",
    "MemoryRepository",
]
