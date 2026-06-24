"""Integration tests for the FastAPI app via TestClient (no network, MockLLM)."""

from __future__ import annotations

import json
import logging

import pytest

logging.disable(logging.CRITICAL)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient backed by an isolated temp SQLite DB + workspace."""
    monkeypatch.setenv("CORTEX_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/api.db")
    monkeypatch.setenv("CORTEX_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CORTEX_ENABLE_NETWORK_TOOLS", "false")
    monkeypatch.setenv("CORTEX_BACKEND", "mock")
    monkeypatch.setenv("CORTEX_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("CORTEX_LOG_LEVEL", "CRITICAL")

    from fastapi.testclient import TestClient

    from cortex.api import deps
    from cortex.config import get_settings

    get_settings.cache_clear()
    deps.reset_run_service()
    deps.get_rate_limiter().reset()

    from cortex.api.server import create_app

    app = create_app(get_settings())
    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def test_health_ready_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["ready"] is True
    m = client.get("/metrics")
    assert m.status_code == 200 and b"cortex_" in m.content


def test_security_headers_present(client):
    h = client.get("/health").headers
    assert h.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" in h
    assert "X-Request-ID" in h


def test_tools_endpoint(client):
    data = client.get("/tools").json()
    names = [t["name"] for t in data["tools"]]
    assert "calculator" in names
    assert any(t["dangerous"] for t in data["tools"])


def test_register_login_and_bad_credentials(client):
    reg = client.post("/v1/auth/register", json={"email": "a@b.com", "password": "password123"})
    assert reg.status_code == 201
    assert "access_token" in reg.json()

    ok = client.post("/v1/auth/login", json={"email": "a@b.com", "password": "password123"})
    assert ok.status_code == 200

    bad = client.post("/v1/auth/login", json={"email": "a@b.com", "password": "nope12345"})
    assert bad.status_code == 401


def test_goal_validation_rejects_too_long(client):
    resp = client.post("/v1/runs", json={"goal": "x" * 5000})
    assert resp.status_code == 422


def test_api_key_issuance_requires_jwt(client):
    # Anonymous cannot mint keys.
    anon = client.post("/v1/auth/api-keys", json={"name": "k"})
    assert anon.status_code in (401, 403)

    reg = client.post("/v1/auth/register", json={"email": "k@b.com", "password": "password123"})
    token = reg.json()["access_token"]
    ak = client.post(
        "/v1/auth/api-keys",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "k1", "allowed_tools": ["calculator"]},
    )
    assert ak.status_code == 201
    assert ak.json()["api_key"].startswith("ck_")


def test_full_sse_run_and_persistence(client):
    run_id = None
    events = []
    with client.stream("POST", "/v1/runs", json={"goal": "Calculate 21 * 2"}) as resp:
        assert resp.status_code == 200
        ev = "message"
        for line in resp.iter_lines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                events.append((ev, data))
                if ev == "run_created":
                    run_id = data["run_id"]

    kinds = [e for e, _ in events]
    assert "run_created" in kinds
    assert "plan" in kinds and "tool_call" in kinds and "answer" in kinds and "done" in kinds
    answer = [d for e, d in events if e == "answer"][0]
    assert "42" in answer["content"]

    # persisted + replayable
    replay = client.get(f"/v1/runs/{run_id}/events").json()
    assert len(replay["events"]) >= 4

    # run state
    state = client.get(f"/v1/runs/{run_id}").json()
    assert state["status"] == "completed" and "42" in state["answer"]


def test_sessions_history(client):
    with client.stream("POST", "/v1/runs", json={"goal": "Calculate 7 * 6"}) as resp:
        for _ in resp.iter_lines():
            pass
    sessions = client.get("/v1/sessions").json()
    assert len(sessions) >= 1
    runs = client.get(f"/v1/sessions/{sessions[0]['id']}/runs").json()
    assert len(runs) >= 1


def test_rate_limit_returns_429(client, monkeypatch):
    # Drive the limiter directly to a low limit by patching the principal limit.
    from cortex.api.deps import get_rate_limiter

    limiter = get_rate_limiter()
    limiter.reset()
    # 3 allowed, 4th blocked
    assert all(limiter.check("principal-x", 3) for _ in range(3))
    assert not limiter.check("principal-x", 3)
