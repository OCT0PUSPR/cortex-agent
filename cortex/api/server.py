"""Production FastAPI app: auth, rate limiting, SSE runs, metrics, and the UI.

Endpoints (high level):

* ``GET /``                    — dark chat-style web UI (login + sessions + live run).
* ``GET /health``              — liveness probe.
* ``GET /ready``               — readiness probe (checks the database).
* ``GET /metrics``             — Prometheus exposition.
* ``GET /tools``               — registered tools + schemas.
* ``POST /v1/auth/*``          — register / login / api-keys.
* ``POST /v1/runs``            — create a run; stream events as SSE.
* ``GET  /v1/runs/{id}``       — run state; ``/events`` for replay; ``/cancel``.
* ``GET  /v1/sessions``        — sessions + per-session run history.

The app wires structured logging, security headers, a request-id, a CORS
allowlist, structured global error handlers, Prometheus metrics, and OpenAPI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Settings, get_settings
from ..observability import (
    METRICS,
    PROM_CONTENT_TYPE,
    configure_logging,
    get_logger,
    init_tracing,
)
from ..tools import build_default_registry

try:
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response
    from starlette.exceptions import HTTPException as StarletteHTTPException
except ImportError as exc:  # pragma: no cover - server deps optional
    raise RuntimeError(
        "FastAPI server requires fastapi, sse-starlette, and uvicorn. "
        "Install them with `pip install -r requirements.txt`."
    ) from exc

from ..db.engine import create_all, dispose_engine, get_engine, session_scope
from .deps import get_principal  # noqa: F401  (imported for OpenAPI discovery)
from .middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from .routes_auth import router as auth_router
from .routes_runs import router as runs_router

_WEB_DIR = Path(__file__).parent / "web"
_log = get_logger("cortex.api")


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """App lifespan: configure logging, bootstrap the DB, dispose on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    init_tracing(settings.enable_tracing)
    get_engine(settings.database_url)
    # Bootstrap schema for SQLite/dev; production should run Alembic migrations.
    if settings.database_url.startswith("sqlite"):
        await create_all(settings.database_url)
    _log.info("startup", backend=settings.backend, database=settings.database_url.split("://")[0])
    try:
        yield
    finally:
        await dispose_engine()
        _log.info("shutdown")


def create_app(settings: Optional[Settings] = None) -> "FastAPI":
    """Create and configure the FastAPI application."""
    settings = settings or get_settings()
    app = FastAPI(
        title="cortex-agent",
        version="0.2.0",
        description="An autonomous agentic AI framework — production API.",
        lifespan=lifespan,
    )

    # -- middleware (order matters: outermost first) -------------------- #
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )

    # -- structured global error handlers ------------------------------- #
    @app.exception_handler(StarletteHTTPException)
    async def http_exc_handler(request: Request, exc: StarletteHTTPException):
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "request_id": rid},
            headers=getattr(exc, "headers", None) or {},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                # exc.errors() can embed non-serializable objects (e.g. the
                # original ValueError in `ctx`); reduce to JSON-safe fields.
                "detail": [
                    {
                        "loc": list(e.get("loc", [])),
                        "msg": str(e.get("msg", "")),
                        "type": str(e.get("type", "")),
                    }
                    for e in exc.errors()
                ],
                "request_id": rid,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        METRICS.observe_error("api")
        _log.exception("unhandled_error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "request_id": rid},
        )

    # -- routers -------------------------------------------------------- #
    app.include_router(auth_router)
    app.include_router(runs_router)

    # -- health / readiness / metrics ----------------------------------- #
    @app.get("/health", tags=["ops"])
    def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "cortex-agent", "version": "0.2.0"}

    @app.get("/ready", tags=["ops"])
    async def ready():
        db_ok = True
        try:
            from sqlalchemy import text

            async with session_scope() as s:
                await s.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001
            db_ok = False
        status_code = 200 if db_ok else 503
        return JSONResponse(
            status_code=status_code,
            content={"ready": db_ok, "database": db_ok, "checks": {"database": db_ok}},
        )

    @app.get("/metrics", tags=["ops"])
    def metrics() -> "Response":
        if not settings.enable_metrics:
            return Response(status_code=404)
        return Response(content=METRICS.render(), media_type=PROM_CONTENT_TYPE)

    @app.get("/tools", tags=["ops"])
    def tools() -> Dict[str, Any]:
        registry = build_default_registry(workspace=settings.workspace, enable_network=settings.enable_network_tools)
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "dangerous": t.dangerous,
                    "schema": t.to_schema(),
                }
                for t in registry.all()
            ]
        }

    # -- web UI --------------------------------------------------------- #
    @app.get("/", include_in_schema=False)
    def index():
        index_file = _WEB_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"detail": "Web UI not found."}, status_code=404)

    @app.get("/app.js", include_in_schema=False)
    def app_js():
        return FileResponse(_WEB_DIR / "app.js", media_type="application/javascript")

    @app.get("/style.css", include_in_schema=False)
    def style_css():
        return FileResponse(_WEB_DIR / "style.css", media_type="text/css")

    return app


# Module-level app for `uvicorn cortex.api.server:app`.
app = create_app()


def main() -> None:  # pragma: no cover - manual server launch
    """Launch the server with uvicorn using configured host/port."""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":  # pragma: no cover
    main()
