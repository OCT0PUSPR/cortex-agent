"""Tests for the RunService orchestration seam (persistence + streaming)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from cortex.agent.loop import EventType
from cortex.config import Settings
from cortex.db.engine import create_all, dispose_engine, get_engine, session_scope
from cortex.db.repository import RunRepository
from cortex.service import RunService, new_request_id


@pytest_asyncio.fixture
async def db():
    url = "sqlite+aiosqlite:///:memory:"
    get_engine(url)
    await create_all(url)
    yield
    await dispose_engine()


def _settings(tmp_path):
    # Hermetic settings: mock backend, no network, sandboxed workspace.
    return Settings(
        backend="mock",
        workspace=str(tmp_path / "ws"),
        enable_network_tools=False,
        max_steps=6,
    )


async def test_create_and_stream_run(db, tmp_path):
    svc = RunService(_settings(tmp_path))
    run_id = await svc.create_run("Calculate 21 * 2")
    assert run_id

    events = [e async for e in svc.stream_run(run_id)]
    types = [e.type for e in events]
    assert EventType.PLAN in types
    assert EventType.ANSWER in types
    answer = [e for e in events if e.type is EventType.ANSWER][0]
    assert "42" in answer.content


async def test_events_are_persisted(db, tmp_path):
    svc = RunService(_settings(tmp_path))
    run_id = await svc.create_run("Calculate 5 + 5")
    _ = [e async for e in svc.stream_run(run_id)]

    # The full trajectory must be durably recorded for audit/replay.
    async with session_scope() as s:
        stored = await RunRepository(s).get_events(run_id)
        run = await RunRepository(s).get_run(run_id)
    assert len(stored) > 0
    assert run.status == "completed"
    assert run.answer and "10" in run.answer


async def test_replay_events(db, tmp_path):
    svc = RunService(_settings(tmp_path))
    run_id = await svc.create_run("Calculate 3 * 3")
    _ = [e async for e in svc.stream_run(run_id)]

    replayed = await svc.replay_events(run_id)
    assert replayed
    assert replayed[0].type is EventType.PLAN
    # after_seq filters
    tail = await svc.replay_events(run_id, after_seq=0)
    assert len(tail) == len(replayed) - 1


async def test_run_to_completion_returns_row(db, tmp_path):
    svc = RunService(_settings(tmp_path))
    run_id = await svc.create_run("Calculate 9 * 9")
    row = await svc.run_to_completion(run_id)
    assert row["id"] == run_id
    assert row["status"] == "completed"


async def test_cancel_unknown_run_returns_false(db, tmp_path):
    svc = RunService(_settings(tmp_path))
    assert svc.cancel("does-not-exist") is False


async def test_stream_unknown_run_raises(db, tmp_path):
    svc = RunService(_settings(tmp_path))
    with pytest.raises(KeyError):
        _ = [e async for e in svc.stream_run("missing")]


def test_new_request_id_unique():
    assert new_request_id() != new_request_id()
    assert len(new_request_id()) == 16
