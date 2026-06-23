"""Tests for the LLM backend layer and the MockLLM trajectory."""

from __future__ import annotations

from cortex.llm import LLMBackend, Message, MockLLM, ToolCall, get_backend
from cortex.llm.base import LLMResponse


def _tools():
    return [
        {
            "name": "calculator",
            "description": "do math",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]


def test_mock_conforms_to_protocol():
    assert isinstance(MockLLM(), LLMBackend)


def test_get_backend_returns_mock():
    backend = get_backend("mock")
    assert backend.name == "mock"


def test_get_backend_unknown_raises():
    try:
        get_backend("does-not-exist")
    except ValueError as exc:
        assert "Unknown backend" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_mock_first_turn_calls_tool():
    mock = MockLLM()
    resp = mock.complete(
        messages=[Message(role="user", content="Calculate 21 * 2")],
        tools=_tools(),
    )
    assert isinstance(resp, LLMResponse)
    assert resp.wants_tools
    call = resp.tool_calls[0]
    assert call.name == "calculator"
    assert "expression" in call.arguments


def test_mock_final_turn_answers_after_observation():
    mock = MockLLM()
    tools = _tools()
    messages = [
        Message(role="user", content="Calculate 21 * 2"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="t1", name="calculator", arguments={"expression": "21 * 2"})],
        ),
        Message(
            role="tool",
            tool_results=[{"tool_use_id": "t1", "content": "21 * 2 = 42", "is_error": False}],
        ),
    ]
    resp = mock.complete(messages=messages, tools=tools)
    assert not resp.wants_tools
    assert "42" in resp.text


def test_response_wants_tools_property():
    resp = LLMResponse(tool_calls=[ToolCall(id="x", name="calc")])
    assert resp.wants_tools
    assert not LLMResponse(text="hi").wants_tools
