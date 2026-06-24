"""Tests for the resilient backend: retries, timeouts, circuit breaker, failover."""

from __future__ import annotations

import time

import pytest

from cortex.llm import (
    MockLLM,
    build_resilient_backend,
    build_resilient_from_settings,
    get_backend,
)
from cortex.llm.base import LLMResponse, Message
from cortex.llm.resilient import (
    BackendUnavailable,
    CircuitBreaker,
    ResilientBackend,
)


class _FlakyBackend:
    """A backend that fails a fixed number of times, then succeeds."""

    def __init__(self, name="flaky", fail_times=0, always_fail=False, delay=0.0):
        self.name = name
        self.model = name
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.delay = delay
        self.calls = 0

    def complete(self, messages, tools=None, system=None, max_tokens=2048, temperature=0.7):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.always_fail or self.calls <= self.fail_times:
            raise RuntimeError(f"{self.name} boom #{self.calls}")
        return LLMResponse(text=f"ok from {self.name}", model=self.name)


def _msgs():
    return [Message(role="user", content="hi")]


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout=10.0)
    assert not cb.is_open
    cb.record_failure()
    assert not cb.is_open
    cb.record_failure()
    assert cb.is_open
    cb.record_success()
    assert not cb.is_open


def test_circuit_breaker_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
    cb.record_failure()
    assert cb.is_open
    time.sleep(0.08)
    assert not cb.is_open  # cooldown elapsed -> half-open (closed for a trial)


def test_resilient_requires_a_backend():
    with pytest.raises(ValueError):
        ResilientBackend([])


def test_resilient_retries_then_succeeds():
    flaky = _FlakyBackend(fail_times=2)
    rb = ResilientBackend([flaky], timeout_seconds=5, max_retries=5)
    resp = rb.complete(_msgs())
    assert "ok from flaky" in resp.text
    assert flaky.calls == 3  # 2 failures + 1 success


def test_resilient_fails_over_to_next_backend():
    primary = _FlakyBackend(name="primary", always_fail=True)
    fallback = MockLLM()
    rb = build_resilient_backend(primary, [fallback], timeout_seconds=5, max_retries=2)
    resp = rb.complete(_msgs())
    # MockLLM answers (it will call a tool or answer) — we just need a response.
    assert isinstance(resp, LLMResponse)
    assert primary.calls >= 1


def test_resilient_all_fail_raises():
    a = _FlakyBackend(name="a", always_fail=True)
    b = _FlakyBackend(name="b", always_fail=True)
    rb = ResilientBackend([a, b], timeout_seconds=5, max_retries=1)
    with pytest.raises(BackendUnavailable):
        rb.complete(_msgs())


def test_resilient_timeout_triggers_failover():
    slow = _FlakyBackend(name="slow", delay=0.5)
    fast = MockLLM()
    rb = ResilientBackend([slow, fast], timeout_seconds=0.1, max_retries=1)
    resp = rb.complete(_msgs())
    assert isinstance(resp, LLMResponse)


def test_resilient_name_and_model_from_primary():
    primary = MockLLM(model="mock-x")
    rb = build_resilient_backend(primary, [])
    assert rb.name == "mock"
    assert rb.model == "mock-x"
    rb.shutdown()


def test_build_resilient_from_settings():
    class S:
        backend = "mock"
        model = None
        fallback_chain = ["mock"]
        llm_timeout_seconds = 30
        llm_max_retries = 2

    rb = build_resilient_from_settings(S())
    resp = rb.complete(_msgs())
    assert isinstance(resp, LLMResponse)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("nonsense")
