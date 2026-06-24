"""SQLAlchemy 2.0 ORM models — the durable state for cortex-agent.

Tables:

* ``users``      — accounts (email + password hash).
* ``api_keys``   — hashed API keys, per-user, with a per-key tool allowlist.
* ``sessions``   — a conversation/session grouping multiple runs.
* ``runs``       — a single agent run (goal, status, budgets, final answer).
* ``run_events`` — the full audit trail: every plan/thought/tool_call/observation/answer.
* ``memories``   — long-term memory entries (keyword + optional vector recall).

The models are backend-agnostic (SQLite default, Postgres via ``DATABASE_URL``).
``JSON`` columns use SQLAlchemy's portable ``JSON`` type. Embeddings are stored
as JSON text by default; a pgvector column can be layered on when available.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    api_keys: Mapped[List["ApiKey"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[List["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128), default="default")
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Per-key tool allowlist (JSON list); null/empty => all tools allowed.
    allowed_tools: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=30)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="api_keys")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="Untitled session")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[Optional["User"]] = relationship(back_populates="sessions")
    runs: Mapped[List["Run"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    goal: Mapped[str] = mapped_column(Text)
    backend: Mapped[str] = mapped_column(String(32), default="mock")
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # queued | running | completed | failed | cancelled | budget_exhausted
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    plan: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    steps_used: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped["Session"] = relationship(back_populates="runs")
    events: Mapped[List["RunEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunEvent.seq"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "goal": self.goal,
            "backend": self.backend,
            "model": self.model,
            "status": self.status,
            "answer": self.answer,
            "plan": self.plan,
            "steps_used": self.steps_used,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RunEvent(Base):
    """One structured event in a run's trajectory — the full audit trail."""

    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)  # monotonic order within a run
    type: Mapped[str] = mapped_column(String(32))  # plan|thought|tool_call|observation|answer|error
    step: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["Run"] = relationship(back_populates="events")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type,
            "step": self.step,
            "content": self.content,
            "data": self.data or {},
            "timestamp": self.created_at.timestamp() if self.created_at else None,
        }


class MemoryEntry(Base):
    """A long-term memory item (durable across runs)."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="note")
    embedding: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


# Convenience export used by Alembic env + create_all.
metadata = Base.metadata

__all__ = [
    "Base",
    "metadata",
    "User",
    "ApiKey",
    "Session",
    "Run",
    "RunEvent",
    "MemoryEntry",
    "func",  # re-exported for repository ordering helpers
]
