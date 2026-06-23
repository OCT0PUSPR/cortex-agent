"""Tests for the tool registry and built-in tools."""

from __future__ import annotations

from cortex.tools import Tool, ToolRegistry, ToolResult
from cortex.tools.builtin import calculator, current_time, run_python


def test_registry_register_and_lookup():
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="echo", func=lambda text: text))
    assert reg.has("echo")
    assert "echo" in reg
    assert reg.names() == ["echo"]
    assert len(reg) == 1


def test_registry_schema_shape(registry):
    schemas = registry.to_schemas()
    assert isinstance(schemas, list) and schemas
    one = schemas[0]
    assert "name" in one and "description" in one
    assert one["input_schema"]["type"] == "object"


def test_registry_execute_unknown_tool():
    reg = ToolRegistry()
    result = reg.execute("nope", {})
    assert result.is_error
    assert "No such tool" in result.output


def test_calculator_basic():
    result = calculator("21 * 2")
    assert not result.is_error
    assert result.data == 42
    assert "42" in result.output


def test_calculator_handles_x_operator():
    assert calculator("6 x 7").data == 42


def test_calculator_rejects_code_injection():
    result = calculator("__import__('os').system('echo hi')")
    assert result.is_error


def test_calculator_division_by_zero():
    result = calculator("1 / 0")
    assert result.is_error


def test_current_time_returns_stamp():
    result = current_time()
    assert not result.is_error
    assert "UTC" in result.output


def test_read_write_file_roundtrip(registry):
    write = registry.execute("write_file", {"path": "note.txt", "content": "hello cortex"})
    assert not write.is_error

    read = registry.execute("read_file", {"path": "note.txt"})
    assert not read.is_error
    assert read.output == "hello cortex"


def test_file_tool_blocks_path_traversal(registry):
    result = registry.execute("write_file", {"path": "../escape.txt", "content": "x"})
    assert result.is_error
    assert "escapes the workspace" in result.output


def test_read_missing_file(registry):
    result = registry.execute("read_file", {"path": "does-not-exist.txt"})
    assert result.is_error


def test_run_python_captures_stdout():
    result = run_python("print(7 * 6)")
    assert not result.is_error
    assert "42" in result.output


def test_run_python_reports_errors():
    result = run_python("raise ValueError('boom')")
    assert result.is_error
    assert "boom" in result.output


def test_run_python_times_out():
    result = run_python("while True: pass", timeout=1)
    assert result.is_error
    assert "timed out" in result.output.lower()


def test_web_search_local_fixture(registry):
    result = registry.execute("web_search", {"query": "what is react"})
    assert not result.is_error
    assert "ReAct" in result.output or "react" in result.output.lower()


def test_tool_coerces_plain_return():
    tool = Tool(name="upper", description="", func=lambda text: text.upper())
    result = tool.run(text="hi")
    assert isinstance(result, ToolResult)
    assert result.output == "HI"


def test_network_tools_excluded_when_disabled(registry):
    assert not registry.has("http_get")
