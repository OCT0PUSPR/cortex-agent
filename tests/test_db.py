"""Tests for the SQLAlchemy async persistence layer."""

from __future__ import annotations

import pytest_asyncio

from cortex.db.engine import create_all, dispose_engine, get_engine, session_scope
from cortex.db.repository import MemoryRepository, RunRepository, UserRepository
from cortex.security import generate_api_key


@pytest_asyncio.fixture
async def db():
    """An in-memory SQLite database, created fresh for each test."""
    url = "sqlite+aiosqlite:///:memory:"
    get_engine(url)
    await create_all(url)
    yield
    await dispose_engine()


async def test_user_and_api_key(db):
    async with session_scope() as s:
        ur = UserRepository(s)
        user = await ur.create_user("a@b.com", "password123", is_admin=True)
        assert user.id

        # authentication
        assert await ur.authenticate("a@b.com", "password123") is not None
        assert await ur.authenticate("a@b.com", "wrong") is None

        # api key
        raw = generate_api_key()
        await ur.create_api_key(user.id, raw, allowed_tools=["calculator"])
        got = await ur.get_api_key(raw)
        assert got is not None and got.allowed_tools == ["calculator"]


async def test_run_and_event_audit_trail(db):
    async with session_scope() as s:
        rr = RunRepository(s)
        sess = await rr.create_session("test session")
        run = await rr.create_run(sess.id, "Calculate 21*2", backend="mock")
        await rr.mark_running(run)

        seq = await rr.next_seq(run.id)
        await rr.add_event(run.id, seq, "plan", "the plan", step=0, data={"steps": ["a"]})
        await rr.add_event(run.id, seq + 1, "tool_call", "calculator", step=1)
        await rr.finish_run(run, "completed", answer="42", steps_used=2, tokens_used=10, cost_usd=0.001)

        events = await rr.get_events(run.id)
        assert len(events) == 2
        assert events[0].type == "plan" and events[1].type == "tool_call"

        fetched = await rr.get_run(run.id)
        assert fetched.status == "completed" and fetched.answer == "42"


async def test_event_replay_after_seq(db):
    async with session_scope() as s:
        rr = RunRepository(s)
        sess = await rr.create_session("s")
        run = await rr.create_run(sess.id, "g")
        for i in range(5):
            await rr.add_event(run.id, i, "thought", f"t{i}", step=i)
        # replay only events after seq 2
        later = await rr.get_events(run.id, after_seq=2)
        assert [e.seq for e in later] == [3, 4]


async def test_memory_recall(db):
    async with session_scope() as s:
        mr = MemoryRepository(s)
        await mr.add("The capital of France is Paris.", user_id="u1")
        await mr.add("Bananas are yellow fruit.", user_id="u1")
        hits = await mr.recall("where is Paris located", user_id="u1")
        assert hits and "Paris" in hits[0].content
        assert await mr.count() == 2


async def test_memory_scoped_by_user(db):
    async with session_scope() as s:
        mr = MemoryRepository(s)
        await mr.add("user1 secret note", user_id="u1")
        await mr.add("user2 secret note", user_id="u2")
        hits = await mr.recall("secret note", user_id="u1")
        assert all("user1" in h.content for h in hits)
