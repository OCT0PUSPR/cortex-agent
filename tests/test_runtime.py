"""Tests for the async agent runtime: budget, policy, approval, cancellation."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.agent.loop import EventType
from cortex.agent.runtime import AsyncAgent, RunContext
from cortex.llm import MockLLM
from cortex.policy import Budget, Policy
from cortex.tools import build_default_registry


@pytest.fixture
def registry(tmp_path: Path):
    return build_default_registry(workspace=tmp_path / "ws", enable_network=False)


async def _collect(agent: AsyncAgent, ctx: RunContext):
    return [e async for e in agent.stream(ctx)]


async def test_full_async_run(registry):
    agent = AsyncAgent(MockLLM(), registry, max_tokens=512)
    ctx = RunContext("r1", "Calculate 21 * 2", Policy(budget=Budget(max_steps=6)))
    events = await _collect(agent, ctx)
    types = [e.type for e in events]
    assert EventType.PLAN in types and EventType.TOOL_CALL in types and EventType.ANSWER in types
    answer = [e for e in events if e.type is EventType.ANSWER][0]
    assert "42" in answer.content


async def test_observation_is_sanitized(registry):
    agent = AsyncAgent(MockLLM(), registry)
    ctx = RunContext("r2", "Calculate 5 + 5", Policy(budget=Budget(max_steps=4)))
    events = await _collect(agent, ctx)
    obs = [e for e in events if e.type is EventType.OBSERVATION]
    # Untrusted-content wrapper must be present.
    assert any("untrusted" in e.content for e in obs)


async def test_budget_stops_loop(registry):
    agent = AsyncAgent(MockLLM(), registry)
    ctx = RunContext("r3", "Calculate 2 + 2", Policy(budget=Budget(max_steps=1)))
    events = await _collect(agent, ctx)
    answer = [e for e in events if e.type is EventType.ANSWER][0]
    assert answer.data["status"] == "budget_exhausted"


async def test_policy_denies_disallowed_tool(registry):
    agent = AsyncAgent(MockLLM(), registry)
    ctx = RunContext(
        "r4",
        "Calculate 5 * 5",
        Policy(allowed_tools={"current_time"}, budget=Budget(max_steps=4)),
    )
    events = await _collect(agent, ctx)
    denied = [e for e in events if e.type is EventType.OBSERVATION and e.data.get("is_error")]
    assert any("not permitted" in e.content for e in denied)


async def test_approval_gate_denies(registry):
    async def deny(name, args):
        return False

    agent = AsyncAgent(MockLLM(), registry)
    ctx = RunContext(
        "r5",
        "Write a file out.txt",
        Policy(require_approval=True, budget=Budget(max_steps=4)),
        approval_cb=deny,
    )
    events = await _collect(agent, ctx)
    assert any("denied by approval" in e.content for e in events if e.type is EventType.OBSERVATION)


async def test_cancellation(registry):
    agent = AsyncAgent(MockLLM(), registry)
    ctx = RunContext("r6", "Calculate 9 * 9", Policy(budget=Budget(max_steps=8)))
    ctx.cancel_event.set()
    events = await _collect(agent, ctx)
    answer = [e for e in events if e.type is EventType.ANSWER][0]
    assert answer.data["status"] == "cancelled"


async def test_cost_and_tokens_tracked(registry):
    agent = AsyncAgent(MockLLM(), registry)
    ctx = RunContext("r7", "Calculate 3 * 3", Policy(budget=Budget(max_steps=4)))
    events = await _collect(agent, ctx)
    answer = [e for e in events if e.type is EventType.ANSWER][0]
    assert "tokens_used" in answer.data
    assert answer.data["cost_usd"] == 0.0  # mock backend is free
