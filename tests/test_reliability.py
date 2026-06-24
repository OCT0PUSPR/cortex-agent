"""Tests for the reliability layer: circuit breaker, retries, failover, cost."""

from __future__ import annotations

import time

import pytest

from cortex.llm import MockLLM, build_resilient_backend
from cortex.llm.base import Message
from cortex.llm.cost import compute_cost, usage_from
from cortex.llm.resilient import BackendUnavailable, CircuitBreaker


def _calc_tools():
    return [
        {
            "name": "calculator",
            "description": "math",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]


class _Broken:
    name = "broken"
    model = "broken-1"

    def complete(self, **kwargs):
        raise RuntimeError("boom")


def test_cost_accounting():
    # Sonnet: $3/M in + $15/M out -> 1M each = $18
    assert abs(compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) - 18.0) < 0.01
    u = usage_from("claude-haiku-4-5", 1000, 2000)
    assert u.total_tokens == 3000 and u.cost_usd > 0


def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
    assert not cb.is_open
    cb.record_failure()
    assert not cb.is_open
    cb.record_failure()
    assert cb.is_open
    time.sleep(0.15)
    assert not cb.is_open  # cooldown elapsed -> half-open


def test_failover_to_mock():
    rb = build_resilient_backend(_Broken(), [MockLLM()], timeout_seconds=5, max_retries=2)
    resp = rb.complete(messages=[Message(role="user", content="Calculate 2+2")], tools=_calc_tools())
    assert resp.model == "mock-1"  # served by the fallback
    rb.shutdown()


def test_all_backends_failed_raises():
    rb = build_resilient_backend(_Broken(), [_Broken()], max_retries=1)
    with pytest.raises(BackendUnavailable):
        rb.complete(messages=[Message(role="user", content="x")])
    rb.shutdown()


def test_mock_reports_usage_and_model():
    resp = MockLLM().complete(messages=[Message(role="user", content="Calculate 21*2")], tools=None)
    assert resp.model == "mock-1"
    assert resp.usage and resp.usage["output_tokens"] > 0
