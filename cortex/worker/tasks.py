"""arq worker tasks: execute queued agent runs out of the request path.

When ``CORTEX_USE_QUEUE=true``, the API enqueues a run and the worker picks it
up here, driving it to completion via the :class:`~cortex.service.RunService`.
The run's events stream into the database as usual, so clients tail progress via
``GET /v1/runs/{id}/events``.

``arq`` is a guarded import so the rest of the framework runs without it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import get_settings
from ..db.engine import create_all, dispose_engine, get_engine
from ..observability import configure_logging, get_logger
from ..service import RunService

_log = get_logger("cortex.worker")


async def execute_run(
    ctx: Dict[str, Any],
    run_id: str,
    allowed_tools: Optional[list] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """arq task: run a queued agent run to completion."""
    service: RunService = ctx["run_service"]
    _log.info("worker_run_start", run_id=run_id)
    result = await service.run_to_completion(run_id, allowed_tools=allowed_tools, user_id=user_id)
    _log.info("worker_run_done", run_id=run_id, status=result.get("status"))
    return result


async def startup(ctx: Dict[str, Any]) -> None:
    """Worker startup: configure logging, bootstrap the DB, build the service."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    get_engine(settings.database_url)
    if settings.database_url.startswith("sqlite"):
        await create_all(settings.database_url)
    ctx["run_service"] = RunService(settings)
    _log.info("worker_startup")


async def shutdown(ctx: Dict[str, Any]) -> None:
    """Worker shutdown: dispose the DB engine."""
    await dispose_engine()
    _log.info("worker_shutdown")


def build_worker_settings():
    """Build the arq ``WorkerSettings`` (lazy — arq is optional)."""
    from arq.connections import RedisSettings

    settings = get_settings()

    class WorkerSettings:
        functions = [execute_run]
        on_startup = startup
        on_shutdown = shutdown
        redis_settings = RedisSettings.from_dsn(settings.redis_url)
        max_jobs = settings.max_concurrent_runs
        job_timeout = settings.run_timeout_seconds + 30

    return WorkerSettings


async def enqueue_run(run_id: str, allowed_tools=None, user_id=None) -> Optional[str]:
    """Enqueue a run on the arq queue; returns the job id (or None if no arq)."""
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
    except ImportError:  # pragma: no cover
        return None
    settings = get_settings()
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    job = await pool.enqueue_job("execute_run", run_id, allowed_tools=allowed_tools, user_id=user_id)
    await pool.close()
    return job.job_id if job else None
