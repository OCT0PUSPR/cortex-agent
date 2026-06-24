"""Offline tests for backend translation logic (no network, no API key)."""

from __future__ import annotations

from cortex.llm.anthropic_backend import _to_anthropic_messages
from cortex.llm.base import Message, ToolCall
from cortex.llm.hf_backend import _build_tool_prompt, _flatten_messages, _parse_tool_call


def test_anthropic_message_translation():
    msgs = [
        Message(role="user", content="Calculate 21 * 2"),
        Message(
            role="assistant",
            content="thinking",
            tool_calls=[ToolCall(id="t1", name="calculator", arguments={"expression": "21 * 2"})],
        ),
        Message(role="tool", tool_results=[{"tool_use_id": "t1", "content": "42", "is_error": False}]),
        Message(role="tool", tool_results=[{"tool_use_id": "t2", "content": "err", "is_error": True}]),
    ]
    out = _to_anthropic_messages(msgs)
    assert out[0] == {"role": "user", "content": "Calculate 21 * 2"}
    assert out[1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" and b["name"] == "calculator" for b in out[1]["content"])
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[3]["content"][0].get("is_error") is True


def test_hf_tool_parse_fenced():
    tc = _parse_tool_call('```json\n{"tool": "calculator", "arguments": {"expression": "2+2"}}\n```')
    assert tc and tc.name == "calculator" and tc.arguments["expression"] == "2+2"


def test_hf_tool_parse_bare():
    tc = _parse_tool_call('{"tool": "current_time", "arguments": {}}')
    assert tc and tc.name == "current_time"


def test_hf_tool_parse_plain_text_returns_none():
    assert _parse_tool_call("The answer is 4.") is None


def test_hf_build_tool_prompt():
    tools = [
        {
            "name": "calculator",
            "description": "math",
            "input_schema": {"properties": {"expression": {"type": "string"}}},
        }
    ]
    prompt = _build_tool_prompt(tools)
    assert "calculator" in prompt and "json" in prompt


def test_hf_flatten_messages():
    msgs = [
        Message(role="user", content="hi"),
        Message(role="tool", tool_results=[{"content": "result"}]),
    ]
    chat = _flatten_messages(msgs, system="be nice")
    assert chat[0]["role"] == "system"
    assert any("Tool result" in c["content"] for c in chat)


def test_get_backend_factory_choices():
    from cortex.llm import get_backend

    assert get_backend("mock").name == "mock"
    # anthropic/hf construct without connecting (lazy clients)
    assert get_backend("anthropic").name == "anthropic"
    assert get_backend("hf").name == "hf"
    try:
        get_backend("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
