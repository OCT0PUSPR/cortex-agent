"""Tests for the arq worker tasks (Redis-free: drives the service directly)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from cortex.config import Settings
from cortex.db.engine import create_all, dispose_engine, get_engine
from cortex.service import RunService
from cortex.worker.tasks import execute_run, shutdown, startup


@pytest_asyncio.fixture
async def db():
    url = "sqlite+aiosqlite:///:memory:"
    get_engine(url)
    await create_all(url)
    yield
    await dispose_engine()


async def test_execute_run_drives_service_to_completion(db, tmp_path):
    settings = Settings(
        backend="mock",
        workspace=str(tmp_path / "ws"),
        enable_network_tools=False,
        max_steps=6,
    )
    svc = RunService(settings)
    run_id = await svc.create_run("Calculate 21 * 2")

    # The worker task receives the service via its arq ctx dict.
    ctx = {"run_service": svc}
    result = await execute_run(ctx, run_id)

    assert result["id"] == run_id
    assert result["status"] == "completed"
    assert "42" in (result.get("answer") or "")


async def test_worker_startup_and_shutdown_lifecycle(monkeypatch, tmp_path):
    # Point the worker at an in-memory DB so startup bootstraps a schema.
    monkeypatch.setenv("CORTEX_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    # get_settings is cached; clear it so the env override is picked up.
    from cortex.config import get_settings

    get_settings.cache_clear()
    ctx: dict = {}
    await startup(ctx)
    assert isinstance(ctx.get("run_service"), RunService)
    await shutdown(ctx)
    get_settings.cache_clear()


def test_build_worker_settings_requires_arq():
    pytest.importorskip("arq")
    from cortex.worker.tasks import build_worker_settings

    ws = build_worker_settings()
    assert hasattr(ws, "functions")
    assert execute_run in ws.functions
