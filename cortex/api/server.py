"""FastAPI server exposing the agent over HTTP + SSE, with a web UI.

Endpoints:

* ``GET /``        — serves the dark chat-style web UI.
* ``GET /health``  — liveness/health probe.
* ``GET /tools``   — list registered tools.
* ``POST /run``    — run a goal; streams ``AgentEvent``s as Server-Sent Events.

The SSE stream emits one ``event:`` per :class:`~cortex.agent.loop.AgentEvent`,
so the browser can render the plan, thoughts, tool calls, observations, and the
final answer live as they happen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..agent.loop import Agent
from ..config import load_settings
from ..llm import get_backend
from ..memory import Memory
from ..tools import build_default_registry

try:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    from sse_starlette.sse import EventSourceResponse
except ImportError as exc:  # pragma: no cover - server deps optional
    raise RuntimeError(
        "FastAPI server requires fastapi, sse-starlette, and uvicorn. "
        "Install them with `pip install -r requirements.txt`."
    ) from exc


_WEB_DIR = Path(__file__).parent / "web"


class RunRequest(BaseModel):
    """Request body for ``POST /run``."""

    goal: str
    backend: str = "mock"
    model: str | None = None
    max_steps: int = 8


def _make_agent(backend: str, model: str | None, max_steps: int) -> Agent:
    """Construct an agent for a single request."""
    cfg = load_settings(backend=backend, model=model, max_steps=max_steps)
    llm = get_backend(cfg.backend, model=cfg.model)
    registry = build_default_registry(
        workspace=cfg.workspace,
        enable_network=cfg.enable_network_tools,
    )
    memory = Memory.create(db_path=cfg.memory_db, use_vectors=cfg.use_vectors)
    return Agent(
        backend=llm,
        registry=registry,
        memory=memory,
        max_steps=cfg.max_steps,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    app = FastAPI(title="cortex-agent", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "cortex-agent", "version": "0.1.0"}

    @app.get("/tools")
    def tools() -> Dict[str, Any]:
        cfg = load_settings()
        registry = build_default_registry(
            workspace=cfg.workspace,
            enable_network=cfg.enable_network_tools,
        )
        return {
            "tools": [
                {"name": t.name, "description": t.description, "schema": t.to_schema()}
                for t in registry.all()
            ]
        }

    @app.post("/run")
    async def run(req: RunRequest) -> "EventSourceResponse":
        agent = _make_agent(req.backend, req.model, req.max_steps)

        def event_stream():
            try:
                for event in agent.stream(req.goal):
                    yield {
                        "event": event.type.value,
                        "data": json.dumps(event.to_dict()),
                    }
            except Exception as exc:  # noqa: BLE001 - report stream failures
                yield {
                    "event": "error",
                    "data": json.dumps({"type": "error", "content": str(exc)}),
                }
            yield {"event": "done", "data": json.dumps({"type": "done"})}

        return EventSourceResponse(event_stream())

    @app.get("/")
    def index():
        index_file = _WEB_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"detail": "Web UI not found."}, status_code=404)

    @app.get("/app.js")
    def app_js():
        return FileResponse(_WEB_DIR / "app.js", media_type="application/javascript")

    @app.get("/style.css")
    def style_css():
        return FileResponse(_WEB_DIR / "style.css", media_type="text/css")

    return app


# Module-level app for `uvicorn cortex.api.server:app`.
app = create_app()


def main() -> None:  # pragma: no cover - manual server launch
    """Launch the server with uvicorn using configured host/port."""
    import uvicorn

    cfg = load_settings()
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":  # pragma: no cover
    main()
