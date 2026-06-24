"""End-to-end agent tests driven entirely by the offline MockLLM.

These assert that the full ReAct loop runs, calls at least one real tool,
produces observations, and synthesizes a final answer — with no network and no
API key.
"""

from __future__ import annotations

from cortex import build_agent
from cortex.agent.loop import Agent, EventType
from cortex.llm import MockLLM


def _make_agent(registry, memory) -> Agent:
    return Agent(backend=MockLLM(), registry=registry, memory=memory, max_steps=6)


def test_full_run_calls_tool_and_answers(registry, memory):
    agent = _make_agent(registry, memory)
    result = agent.run("Calculate 21 * 2 for me")

    types = [e.type for e in result.events]
    assert EventType.PLAN in types
    assert EventType.TOOL_CALL in types
    assert EventType.OBSERVATION in types
    assert EventType.ANSWER in types

    # A calculator tool call should have occurred with the right result.
    tool_calls = [e for e in result.events if e.type is EventType.TOOL_CALL]
    assert any(e.data.get("tool") == "calculator" for e in tool_calls)

    observations = [e for e in result.events if e.type is EventType.OBSERVATION]
    assert any("42" in e.content for e in observations)

    assert result.answer
    assert "42" in result.answer


def test_run_produces_plan_steps(registry, memory):
    agent = _make_agent(registry, memory)
    result = agent.run("Search for cortex-agent then summarize it")
    assert result.plan.steps, "expected a non-empty plan"


def test_stream_yields_events_in_order(registry, memory):
    agent = _make_agent(registry, memory)
    events = list(agent.stream("What time is it right now?"))
    assert events[0].type is EventType.PLAN
    assert events[-1].type is EventType.ANSWER
    # current_time tool should be invoked for a time question.
    assert any(e.type is EventType.TOOL_CALL and e.data.get("tool") == "current_time" for e in events)


def test_agent_writes_to_memory(registry, memory):
    agent = _make_agent(registry, memory)
    agent.run("Calculate 5 + 5")
    # The agent should persist observations/conclusions to long-term memory.
    assert memory.long_term.count() > 0


def test_run_respects_step_budget(registry, memory):
    agent = Agent(backend=MockLLM(), registry=registry, memory=memory, max_steps=1)
    result = agent.run("Calculate 2 + 2")
    assert result.steps_used <= 2  # 1 react step + final answer step


def test_result_serializes_to_dict(registry, memory):
    agent = _make_agent(registry, memory)
    result = agent.run("Calculate 10 * 10")
    payload = result.to_dict()
    assert "answer" in payload
    assert "events" in payload and isinstance(payload["events"], list)
    assert payload["events"][0]["type"] == "plan"


def test_build_agent_helper_runs(tmp_path, monkeypatch):
    # Point the workspace/db at the temp dir so the helper is hermetic.
    monkeypatch.setenv("CORTEX_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CORTEX_MEMORY_DB", str(tmp_path / "mem.sqlite"))
    monkeypatch.setenv("CORTEX_ENABLE_NETWORK_TOOLS", "false")
    agent = build_agent(backend="mock")
    result = agent.run("Calculate 3 * 3")
    assert "9" in result.answer


def test_no_relevant_tool_still_answers(registry, memory):
    # Even a vague goal yields a default calculation and an answer.
    agent = _make_agent(registry, memory)
    result = agent.run("Tell me something")
    assert result.answer
    assert any(e.type is EventType.ANSWER for e in result.events)
