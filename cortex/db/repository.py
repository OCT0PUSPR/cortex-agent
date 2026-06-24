"""Repository layer: async data access for users, keys, runs, events, memory.

All methods take an :class:`AsyncSession` so the caller controls the
transaction boundary. This keeps persistence logic out of the API handlers and
the agent runtime, and makes it independently testable.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..security import hash_api_key, hash_password, verify_password
from .models import ApiKey, MemoryEntry, Run, RunEvent, Session, User

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Users & API keys
# --------------------------------------------------------------------------- #


class UserRepository:
    """Data access for users and API keys."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_user(self, email: str, password: str, is_admin: bool = False) -> User:
        user = User(email=email, password_hash=hash_password(password), is_admin=is_admin)
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_email(self, email: str) -> Optional[User]:
        res = await self.session.execute(select(User).where(User.email == email))
        return res.scalar_one_or_none()

    async def get_by_id(self, user_id: str) -> Optional[User]:
        return await self.session.get(User, user_id)

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        user = await self.get_by_email(email)
        if user and user.is_active and verify_password(password, user.password_hash):
            return user
        return None

    async def create_api_key(
        self,
        user_id: str,
        raw_key: str,
        name: str = "default",
        allowed_tools: Optional[List[str]] = None,
        rate_limit_per_minute: int = 30,
    ) -> ApiKey:
        api_key = ApiKey(
            user_id=user_id,
            name=name,
            key_hash=hash_api_key(raw_key),
            allowed_tools=allowed_tools,
            rate_limit_per_minute=rate_limit_per_minute,
        )
        self.session.add(api_key)
        await self.session.flush()
        return api_key

    async def get_api_key(self, raw_key: str) -> Optional[ApiKey]:
        res = await self.session.execute(
            select(ApiKey).where(ApiKey.key_hash == hash_api_key(raw_key), ApiKey.is_active.is_(True))
        )
        return res.scalar_one_or_none()

    async def touch_api_key(self, api_key: ApiKey) -> None:
        api_key.last_used_at = datetime.now(timezone.utc)
        await self.session.flush()


# --------------------------------------------------------------------------- #
# Sessions, runs & events
# --------------------------------------------------------------------------- #


class RunRepository:
    """Data access for sessions, runs, and the run-event audit trail."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, title: str = "Untitled session", user_id: Optional[str] = None) -> Session:
        sess = Session(title=title, user_id=user_id)
        self.session.add(sess)
        await self.session.flush()
        return sess

    async def get_session(self, session_id: str) -> Optional[Session]:
        return await self.session.get(Session, session_id)

    async def list_sessions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Session]:
        stmt = select(Session).order_by(Session.created_at.desc()).limit(limit)
        if user_id is not None:
            stmt = stmt.where(Session.user_id == user_id)
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def create_run(
        self,
        session_id: str,
        goal: str,
        backend: str = "mock",
        model: Optional[str] = None,
        status: str = "queued",
    ) -> Run:
        run = Run(session_id=session_id, goal=goal, backend=backend, model=model, status=status)
        self.session.add(run)
        await self.session.flush()
        return run

    async def get_run(self, run_id: str) -> Optional[Run]:
        return await self.session.get(Run, run_id)

    async def list_runs(self, session_id: str, limit: int = 50) -> List[Run]:
        res = await self.session.execute(
            select(Run).where(Run.session_id == session_id).order_by(Run.created_at.desc()).limit(limit)
        )
        return list(res.scalars().all())

    async def mark_running(self, run: Run) -> None:
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def finish_run(
        self,
        run: Run,
        status: str,
        answer: Optional[str] = None,
        plan: Optional[List[str]] = None,
        steps_used: int = 0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        run.status = status
        run.answer = answer
        run.plan = plan
        run.steps_used = steps_used
        run.tokens_used = tokens_used
        run.cost_usd = cost_usd
        run.error = error
        run.finished_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def next_seq(self, run_id: str) -> int:
        res = await self.session.execute(
            select(func.coalesce(func.max(RunEvent.seq), -1)).where(RunEvent.run_id == run_id)
        )
        return int(res.scalar_one()) + 1

    async def add_event(
        self,
        run_id: str,
        seq: int,
        type_: str,
        content: str = "",
        step: int = 0,
        data: Optional[dict] = None,
    ) -> RunEvent:
        event = RunEvent(run_id=run_id, seq=seq, type=type_, content=content, step=step, data=data)
        self.session.add(event)
        await self.session.flush()
        return event

    async def get_events(self, run_id: str, after_seq: int = -1) -> List[RunEvent]:
        res = await self.session.execute(
            select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.seq > after_seq).order_by(RunEvent.seq)
        )
        return list(res.scalars().all())


# --------------------------------------------------------------------------- #
# Long-term memory
# --------------------------------------------------------------------------- #


class MemoryRepository:
    """Async long-term memory with keyword recall (and optional vectors)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        content: str,
        kind: str = "note",
        user_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(content=content, kind=kind, user_id=user_id, embedding=embedding)
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def recall(self, query: str, top_k: int = 5, user_id: Optional[str] = None) -> List[MemoryEntry]:
        terms = set(_tokenize(query))
        if not terms:
            return []
        stmt = select(MemoryEntry)
        if user_id is not None:
            stmt = stmt.where(MemoryEntry.user_id == user_id)
        res = await self.session.execute(stmt)
        rows = list(res.scalars().all())

        scored = []
        for row in rows:
            doc_terms = _tokenize(row.content)
            if not doc_terms:
                continue
            overlap = sum(1 for t in doc_terms if t in terms)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(doc_terms))
            scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [row for _, row in scored[:top_k]]

    async def count(self) -> int:
        res = await self.session.execute(select(func.count()).select_from(MemoryEntry))
        return int(res.scalar_one())
