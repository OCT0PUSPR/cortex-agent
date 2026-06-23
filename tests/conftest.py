"""Shared pytest fixtures for cortex-agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.llm import MockLLM
from cortex.memory import Memory
from cortex.tools import build_default_registry


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An isolated temp workspace for file/python tools."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def registry(workspace: Path):
    """A tool registry sandboxed to the temp workspace (no network tools)."""
    return build_default_registry(workspace=workspace, enable_network=False)


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    """A memory backed by an in-workspace SQLite file (keyword recall)."""
    return Memory.create(db_path=str(tmp_path / "mem.sqlite"), use_vectors=False)


@pytest.fixture
def mock_backend() -> MockLLM:
    """The deterministic offline backend."""
    return MockLLM()
