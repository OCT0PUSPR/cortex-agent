"""Reliability wrapper: retries, timeouts, circuit breaker, and failover.

:class:`ResilientBackend` wraps a primary backend plus an ordered failover
chain. Each backend call is:

* bounded by a per-call timeout (run in a worker thread),
* retried with exponential backoff (``tenacity`` when present, else a small
  built-in loop),
* guarded by a per-backend circuit breaker that trips after repeated failures
  and recovers after a cooldown.

When the primary backend exhausts retries (or its breaker is open), the wrapper
fails over to the next backend in the chain (e.g. anthropic → hf → mock), so a
run degrades gracefully instead of failing outright.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional

from ..observability import METRICS, get_logger
from .base import LLMBackend, LLMResponse, Message

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    _HAS_TENACITY = True
except ImportError:  # pragma: no cover
    _HAS_TENACITY = False

_log = get_logger("cortex.llm")


class BackendUnavailable(RuntimeError):
    """Raised when every backend in the chain has failed."""


class CircuitBreaker:
    """A simple thread-safe circuit breaker.

    After ``failure_threshold`` consecutive failures the breaker opens and
    short-circuits calls for ``reset_timeout`` seconds, then allows a trial call
    (half-open). A success closes it again.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._failures < self.failure_threshold:
                return False
            # Open — but allow a half-open trial after the cooldown.
            return (time.monotonic() - self._opened_at) < self.reset_timeout

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()


class ResilientBackend:
    """Wrap a primary backend + failover chain with retries and a breaker."""

    def __init__(
        self,
        backends: List[LLMBackend],
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        breaker_threshold: int = 5,
        breaker_reset: float = 30.0,
    ) -> None:
        if not backends:
            raise ValueError("ResilientBackend requires at least one backend.")
        self.backends = backends
        self.name = backends[0].name
        self.model = getattr(backends[0], "model", "")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._breakers: Dict[int, CircuitBreaker] = {
            id(b): CircuitBreaker(breaker_threshold, breaker_reset) for b in backends
        }
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")

    # -- timeout-bounded single call ----------------------------------- #
    def _call_with_timeout(self, backend: LLMBackend, **kwargs: Any) -> LLMResponse:
        future = self._executor.submit(backend.complete, **kwargs)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(f"{backend.name} call exceeded {self.timeout_seconds}s") from exc

    # -- retry wrapper ------------------------------------------------- #
    def _attempt(self, backend: LLMBackend, **kwargs: Any) -> LLMResponse:
        if _HAS_TENACITY:

            @retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.3, min=0.3, max=4.0),
                retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
                reraise=True,
            )
            def _run() -> LLMResponse:
                return self._call_with_timeout(backend, **kwargs)

            return _run()

        # Fallback retry loop. # pragma: no cover - tenacity present in CI
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_with_timeout(backend, **kwargs)
            except (TimeoutError, ConnectionError, RuntimeError) as exc:
                last_exc = exc
                time.sleep(min(0.3 * (2**attempt), 4.0))
        raise last_exc if last_exc else RuntimeError("retry loop failed")

    # -- public API ---------------------------------------------------- #
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        kwargs = dict(
            messages=messages,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        errors: List[str] = []
        for backend in self.backends:
            breaker = self._breakers[id(backend)]
            if breaker.is_open:
                errors.append(f"{backend.name}: circuit open")
                continue
            try:
                resp = self._attempt(backend, **kwargs)
                breaker.record_success()
                return resp
            except Exception as exc:  # noqa: BLE001 - try the next backend
                breaker.record_failure()
                METRICS.observe_error("llm")
                _log.warning("backend_failed", backend=backend.name, error=str(exc))
                errors.append(f"{backend.name}: {exc}")
                continue
        raise BackendUnavailable("All backends failed: " + "; ".join(errors))

    def shutdown(self) -> None:
        """Release the thread pool (graceful shutdown)."""
        self._executor.shutdown(wait=False)


def build_resilient_backend(
    primary: LLMBackend,
    fallbacks: Optional[List[LLMBackend]] = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 3,
) -> ResilientBackend:
    """Construct a :class:`ResilientBackend` from a primary + fallbacks."""
    chain: List[LLMBackend] = [primary] + list(fallbacks or [])
    return ResilientBackend(chain, timeout_seconds=timeout_seconds, max_retries=max_retries)
