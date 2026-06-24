"""Run routes: create/stream runs (SSE), list sessions + history, cancel."""

from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from ..db.engine import get_session
from ..db.repository import RunRepository
from ..service import RunService
from .deps import Principal, get_run_service, rate_limited
from .schemas import RunRequest, RunResponse, SessionResponse

router = APIRouter(prefix="/v1", tags=["runs"])


@router.post("/runs", status_code=status.HTTP_200_OK)
async def create_and_stream_run(
    body: RunRequest,
    principal: Principal = Depends(rate_limited),
    service: RunService = Depends(get_run_service),
) -> EventSourceResponse:
    """Create a run and stream its events as Server-Sent Events.

    The run is persisted (audit trail); reconnect/replay is available via
    ``GET /v1/runs/{id}/events``.
    """
    run_id = await service.create_run(
        goal=body.goal,
        session_id=body.session_id,
        user_id=principal.user_id,
        backend=body.backend,
        model=body.model,
    )

    async def event_stream():
        # Emit the run id first so clients can subscribe / cancel.
        yield {"event": "run_created", "data": json.dumps({"run_id": run_id})}
        try:
            async for event in service.stream_run(
                run_id, allowed_tools=principal.allowed_tools, user_id=principal.user_id
            ):
                yield {"event": event.type.value, "data": json.dumps(event.to_dict())}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"type": "error", "content": str(exc)})}
        yield {"event": "done", "data": json.dumps({"type": "done", "run_id": run_id})}

    return EventSourceResponse(event_stream())


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    principal: Principal = Depends(rate_limited),
    session: AsyncSession = Depends(get_session),
) -> RunResponse:
    """Fetch a run's current state."""
    run = await RunRepository(session).get_run(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return RunResponse(**run.to_dict())


@router.get("/runs/{run_id}/events")
async def get_run_events(
    run_id: str,
    after_seq: int = -1,
    principal: Principal = Depends(rate_limited),
    service: RunService = Depends(get_run_service),
):
    """Return persisted events for a run (replay / reconnect)."""
    events = await service.replay_events(run_id, after_seq=after_seq)
    return {"run_id": run_id, "events": [e.to_dict() for e in events]}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    principal: Principal = Depends(rate_limited),
    service: RunService = Depends(get_run_service),
):
    """Request cancellation of an in-flight run."""
    cancelled = service.cancel(run_id)
    return {"run_id": run_id, "cancellation_requested": cancelled}


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    principal: Principal = Depends(rate_limited),
    session: AsyncSession = Depends(get_session),
) -> List[SessionResponse]:
    """List the caller's sessions (or all when anonymous/dev)."""
    rows = await RunRepository(session).list_sessions(user_id=principal.user_id)
    return [
        SessionResponse(id=s.id, title=s.title, created_at=s.created_at.isoformat() if s.created_at else None)
        for s in rows
    ]


@router.get("/sessions/{session_id}/runs", response_model=List[RunResponse])
async def list_session_runs(
    session_id: str,
    principal: Principal = Depends(rate_limited),
    session: AsyncSession = Depends(get_session),
) -> List[RunResponse]:
    """List runs within a session (history)."""
    rows = await RunRepository(session).list_runs(session_id)
    return [RunResponse(**r.to_dict()) for r in rows]
