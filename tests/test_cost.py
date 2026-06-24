"""Tests for token-cost accounting."""

from __future__ import annotations

from cortex.llm.cost import (
    Usage,
    compute_cost,
    estimate_tokens,
    price_for,
    usage_from,
)


def test_price_for_known_and_unknown():
    assert price_for("claude-opus-4-8") == (5.0, 25.0)
    assert price_for("mock-1") == (0.0, 0.0)
    # Unknown models fall back to a conservative default (non-zero).
    assert price_for("some-unknown-model") == (3.0, 15.0)


def test_compute_cost_math():
    # 1M input @ $3 + 1M output @ $15 = $18.
    assert abs(compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) - 18.0) < 1e-9
    assert compute_cost("mock-1", 1000, 1000) == 0.0


def test_usage_from_builds_cost():
    u = usage_from("claude-haiku-4-5", 1_000_000, 1_000_000)
    # haiku = (1.0, 5.0) per 1M -> $6 total.
    assert abs(u.cost_usd - 6.0) < 1e-9
    assert u.total_tokens == 2_000_000


def test_usage_from_handles_none():
    u = usage_from("mock-1", None, None)
    assert u.input_tokens == 0 and u.output_tokens == 0 and u.cost_usd == 0.0


def test_usage_add_accumulates():
    a = Usage(input_tokens=10, output_tokens=20, cost_usd=0.1)
    b = Usage(input_tokens=5, output_tokens=5, cost_usd=0.05)
    a.add(b)
    assert a.input_tokens == 15 and a.output_tokens == 25
    assert abs(a.cost_usd - 0.15) < 1e-9
    assert a.total_tokens == 40


def test_estimate_tokens():
    assert estimate_tokens("") == 1  # floor of 1
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens(None) == 1  # type: ignore[arg-type]
