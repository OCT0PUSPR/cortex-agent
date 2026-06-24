"""Observability: structured logging, Prometheus metrics, and OTel spans.

All three are guarded so the framework runs even when the optional packages are
absent. ``structlog`` is configured for JSON output keyed by ``run_id`` and
``request_id``; Prometheus exposes agent/tool/LLM metrics; OpenTelemetry spans
wrap agent steps and tool calls when enabled and installed.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any, Dict, Iterator, Optional

# --------------------------------------------------------------------------- #
# Structured logging (structlog with a stdlib fallback)
# --------------------------------------------------------------------------- #

try:
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:  # pragma: no cover
    structlog = None  # type: ignore[assignment]
    _HAS_STRUCTLOG = False

_configured = False


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure process-wide structured logging (idempotent)."""
    global _configured
    if _configured:
        return
    _configured = True

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    if not _HAS_STRUCTLOG:  # pragma: no cover - exercised only without structlog
        return

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "cortex") -> Any:
    """Return a bound logger (structlog if available, else stdlib)."""
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)


@contextlib.contextmanager
def bind_context(**kwargs: Any) -> Iterator[None]:
    """Bind context vars (run_id, request_id, ...) for the duration of a block."""
    if _HAS_STRUCTLOG:
        tokens = structlog.contextvars.bind_contextvars(**kwargs)
        try:
            yield
        finally:
            structlog.contextvars.reset_contextvars(**tokens)
    else:  # pragma: no cover
        yield


# --------------------------------------------------------------------------- #
# Prometheus metrics (guarded)
# --------------------------------------------------------------------------- #

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    _HAS_PROM = True
except ImportError:  # pragma: no cover
    _HAS_PROM = False
    CONTENT_TYPE_LATEST = "text/plain"


class _Metrics:
    """Prometheus metric collectors (no-ops when prometheus_client is absent)."""

    def __init__(self) -> None:
        self.registry: Optional["CollectorRegistry"] = None
        if _HAS_PROM:
            self.registry = CollectorRegistry()
            self.runs_total = Counter("cortex_runs_total", "Total agent runs", ["status"], registry=self.registry)
            self.agent_steps = Histogram(
                "cortex_agent_steps",
                "ReAct steps per run",
                registry=self.registry,
                buckets=(1, 2, 3, 5, 8, 13, 21),
            )
            self.tool_calls = Counter(
                "cortex_tool_calls_total", "Tool calls", ["tool", "status"], registry=self.registry
            )
            self.tool_latency = Histogram(
                "cortex_tool_latency_seconds", "Tool latency", ["tool"], registry=self.registry
            )
            self.llm_tokens = Counter(
                "cortex_llm_tokens_total",
                "LLM tokens",
                ["provider", "direction"],
                registry=self.registry,
            )
            self.llm_cost_usd = Counter(
                "cortex_llm_cost_usd_total", "LLM cost (USD)", ["provider"], registry=self.registry
            )
            self.run_duration = Histogram(
                "cortex_run_duration_seconds",
                "Run duration",
                registry=self.registry,
                buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
            )
            self.errors = Counter("cortex_errors_total", "Errors", ["component"], registry=self.registry)
        else:  # pragma: no cover
            self.registry = None

    def observe_tool(self, tool: str, latency: float, ok: bool) -> None:
        if not _HAS_PROM:
            return
        self.tool_calls.labels(tool=tool, status="ok" if ok else "error").inc()
        self.tool_latency.labels(tool=tool).observe(latency)

    def observe_tokens(self, provider: str, in_tokens: int, out_tokens: int) -> None:
        if not _HAS_PROM:
            return
        if in_tokens:
            self.llm_tokens.labels(provider=provider, direction="input").inc(in_tokens)
        if out_tokens:
            self.llm_tokens.labels(provider=provider, direction="output").inc(out_tokens)

    def observe_cost(self, provider: str, usd: float) -> None:
        if _HAS_PROM and usd:
            self.llm_cost_usd.labels(provider=provider).inc(usd)

    def observe_run(self, status: str, steps: int, duration: float) -> None:
        if not _HAS_PROM:
            return
        self.runs_total.labels(status=status).inc()
        self.agent_steps.observe(steps)
        self.run_duration.observe(duration)

    def observe_error(self, component: str) -> None:
        if _HAS_PROM:
            self.errors.labels(component=component).inc()

    def render(self) -> bytes:
        """Render metrics in Prometheus text exposition format."""
        if _HAS_PROM and self.registry is not None:
            return generate_latest(self.registry)
        return b"# prometheus_client not installed\n"


METRICS = _Metrics()
HAS_PROM = _HAS_PROM
PROM_CONTENT_TYPE = CONTENT_TYPE_LATEST


# --------------------------------------------------------------------------- #
# OpenTelemetry spans (guarded; no-op when disabled or absent)
# --------------------------------------------------------------------------- #

_TRACER = None


def init_tracing(enabled: bool, service_name: str = "cortex-agent") -> None:
    """Initialize an OTel tracer if enabled and the package is installed."""
    global _TRACER
    if not enabled:
        return
    try:  # pragma: no cover - optional + heavy
        from opentelemetry import trace

        _TRACER = trace.get_tracer(service_name)
    except ImportError:
        _TRACER = None


@contextlib.contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """Open an OTel span; a no-op when tracing is disabled/unavailable."""
    if _TRACER is None:
        yield
        return
    with _TRACER.start_as_current_span(name) as sp:  # pragma: no cover
        for key, value in (attributes or {}).items():
            sp.set_attribute(key, value)
        yield
