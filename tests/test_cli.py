"""Smoke tests for the CLI (MockLLM backend, no network)."""

from __future__ import annotations

import pytest

from cortex.cli import build_parser, main


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CORTEX_MEMORY_DB", str(tmp_path / "mem.sqlite"))
    monkeypatch.setenv("CORTEX_ENABLE_NETWORK_TOOLS", "false")


def test_parser_backend_choices():
    parser = build_parser()
    args = parser.parse_args(["--backend", "tinybrain", "run", "x"])
    assert args.backend == "tinybrain"


def test_cli_run_command(capsys):
    rc = main(["--backend", "mock", "run", "Calculate 21 * 2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "42" in out


def test_cli_tools_command(capsys):
    rc = main(["tools"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "calculator" in out
