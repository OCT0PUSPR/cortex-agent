"""Run orchestration service — durable, resumable agent runs.

The :class:`RunService` is the seam between the async agent runtime and the
persistence layer. It:

* builds a resilient backend + policy + tool registry from settings,
* persists every :class:`AgentEvent` to ``run_events`` as it streams (audit
  trail + resumability),
* updates the ``runs`` row's status/answer/usage,
* enforces a per-process concurrency limit (backpressure),
* supports cancellation and resuming the event stream of an in-flight or
  finished run.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Dict, List, Optional

from .agent.loop import AgentEvent, EventType
from .agent.runtime import AsyncAgent, RunContext
from .config import Settings, get_settings
from .db.engine import session_scope
from .db.repository import MemoryRepository, RunRepository
from .llm import build_resilient_from_settings
from .observability import get_logger
from .policy import Policy
from .tools import build_default_registry

_log = get_logger("cortex.service")


class RunService:
    """Orchestrates agent runs with persistence, concurrency, and cancellation."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        # The semaphore is created lazily inside the running event loop — on
        # Python 3.9 constructing it eagerly binds to whatever loop is current
        # at construction time (or none, in a worker thread), which breaks.
        self._semaphore: Optional[asyncio.Semaphore] = None
        # run_id -> cancel Event for in-flight runs.
        self._cancels: Dict[str, asyncio.Event] = {}

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return the concurrency semaphore, creating it in the active loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_runs)
        return self._semaphore

    # ------------------------------------------------------------------ #
    def cancel(self, run_id: str) -> bool:
        """Signal cancellation for an in-flight run. Returns True if found."""
        event = self._cancels.get(run_id)
        if event is not None:
            event.set()
            return True
        return False

    def _build_agent(self, allowed_tools: Optional[List[str]], user_id: Optional[str]):
        """Construct an :class:`AsyncAgent` wired with memory + tools + backend."""
        registry = build_default_registry(
            workspace=self.settings.workspace,
            enable_network=self.settings.enable_network_tools,
            python_cpu_seconds=self.settings.python_cpu_seconds,
            python_memory_mb=self.settings.python_memory_mb,
            python_wall_seconds=self.settings.python_wall_seconds,
            http_max_bytes=self.settings.http_max_bytes,
        )
        backend = build_resilient_from_settings(self.settings)

        async def recall(query: str, top_k: int) -> List[str]:
            async with session_scope() as s:
                rows = await MemoryRepository(s).recall(query, top_k=top_k, user_id=user_id)
                return [r.content for r in rows]

        async def write(content: str, kind: str) -> None:
            async with session_scope() as s:
                await MemoryRepository(s).add(content, kind=kind, user_id=user_id)

        agent = AsyncAgent(
            backend=backend,
            registry=registry,
            memory_recall=recall,
            memory_write=write,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            tool_timeout=float(self.settings.tool_timeout_seconds),
            run_timeout=float(self.settings.run_timeout_seconds),
        )
        return agent, registry

    # ------------------------------------------------------------------ #
    async def create_run(
        self,
        goal: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        backend: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Create a session (if needed) and a queued run; return the run id."""
        async with session_scope() as s:
            repo = RunRepository(s)
            if session_id is None:
                sess = await repo.create_session(title=goal[:60], user_id=user_id)
                session_id = sess.id
            run = await repo.create_run(
                session_id=session_id,
                goal=goal,
                backend=backend or self.settings.backend,
                model=model or self.settings.model,
                status="queued",
            )
            return run.id

    async def stream_run(
        self,
        run_id: str,
        allowed_tools: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        approval_cb=None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a queued run, persisting events as they stream out.

        Honors the concurrency semaphore (backpressure) and registers a cancel
        event so the run can be cancelled mid-flight.
        """
        # Load the run row.
        async with session_scope() as s:
            run = await RunRepository(s).get_run(run_id)
            if run is None:
                raise KeyError(f"run {run_id} not found")
            goal = run.goal

        policy = Policy.from_settings(self.settings, allowed_tools=allowed_tools)
        cancel_event = asyncio.Event()
        self._cancels[run_id] = cancel_event

        async with self._get_semaphore():  # backpressure / concurrency cap
            agent, _ = self._build_agent(allowed_tools, user_id)
            ctx = RunContext(
                run_id=run_id,
                goal=goal,
                policy=policy,
                cancel_event=cancel_event,
                approval_cb=approval_cb,
                user_id=user_id,
            )
            async with session_scope() as s:
                run = await RunRepository(s).get_run(run_id)
                assert run is not None  # loaded above; refetched in this session
                await RunRepository(s).mark_running(run)

            seq = 0
            final_status = "completed"
            final_answer = ""
            plan_steps: List[str] = []
            tokens_used = 0
            cost_usd = 0.0
            try:
                async for event in agent.stream(ctx):
                    # Persist the event (audit trail / resumability).
                    async with session_scope() as s:
                        await RunRepository(s).add_event(
                            run_id, seq, event.type.value, event.content, event.step, event.data
                        )
                    seq += 1
                    if event.type is EventType.PLAN:
                        plan_steps = event.data.get("steps", [])
                    elif event.type is EventType.ANSWER:
                        final_answer = event.content
                        final_status = event.data.get("status", "completed")
                        tokens_used = int(event.data.get("tokens_used", 0))
                        cost_usd = float(event.data.get("cost_usd", 0.0))
                    yield event
            except Exception as exc:  # noqa: BLE001 - record run failure
                final_status = "failed"
                final_answer = f"Run failed: {exc}"
                _log.error("run_failed", run_id=run_id, error=str(exc))
            finally:
                async with session_scope() as s:
                    run = await RunRepository(s).get_run(run_id)
                    assert run is not None  # the run row exists for the whole stream
                    await RunRepository(s).finish_run(
                        run,
                        status=final_status,
                        answer=final_answer,
                        plan=plan_steps,
                        steps_used=ctx.policy.budget.steps,
                        tokens_used=tokens_used or ctx.policy.budget.total_tokens,
                        cost_usd=cost_usd or ctx.policy.budget.cost_usd,
                    )
                self._cancels.pop(run_id, None)

    async def run_to_completion(
        self,
        run_id: str,
        allowed_tools: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """Drive a run to completion (used by the worker) and return its row."""
        async for _ in self.stream_run(run_id, allowed_tools, user_id):
            pass
        async with session_scope() as s:
            run = await RunRepository(s).get_run(run_id)
            return run.to_dict() if run else {"id": run_id, "status": "unknown"}

    async def replay_events(self, run_id: str, after_seq: int = -1) -> List[AgentEvent]:
        """Return persisted events for a run (resume/replay)."""
        async with session_scope() as s:
            rows = await RunRepository(s).get_events(run_id, after_seq=after_seq)
            return [
                AgentEvent(
                    type=EventType(r.type),
                    content=r.content,
                    step=r.step,
                    data=r.data or {},
                )
                for r in rows
            ]


def new_request_id() -> str:
    """Generate a short request id for log correlation."""
    return uuid.uuid4().hex[:16]
