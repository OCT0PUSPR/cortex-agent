"""cortex-agent: an autonomous agentic AI framework.

Planning + tool use + memory + a ReAct/plan-execute loop, with interchangeable
LLM backends (Anthropic Claude, HuggingFace, and an offline MockLLM).

Quick start::

    from cortex import build_agent

    agent = build_agent(backend="mock")
    result = agent.run("Calculate 21 * 2 and tell me the current time")
    print(result.answer)
"""

from __future__ import annotations

from typing import Optional

from .agent import Agent, AgentEvent, AgentResult, EventType, Plan, Planner
from .config import Settings, load_settings
from .llm import LLMBackend, LLMResponse, Message, MockLLM, ToolCall, get_backend
from .memory import Memory
from .tools import Tool, ToolRegistry, ToolResult, build_default_registry

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentResult",
    "EventType",
    "Plan",
    "Planner",
    "Settings",
    "load_settings",
    "LLMBackend",
    "LLMResponse",
    "Message",
    "MockLLM",
    "ToolCall",
    "get_backend",
    "Memory",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "build_agent",
]


def build_agent(
    backend: str = "mock",
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
    **backend_kwargs,
) -> Agent:
    """Build a fully-wired :class:`~cortex.agent.loop.Agent`.

    Args:
        backend: Backend name (``"mock"``, ``"anthropic"``, ``"hf"``).
        model: Optional model id override.
        settings: Optional pre-built :class:`Settings`; one is loaded if omitted.
        **backend_kwargs: Extra backend constructor options.

    Returns:
        A ready-to-run :class:`Agent` with built-in tools and memory.
    """
    cfg = settings or load_settings(backend=backend, model=model)
    llm = get_backend(cfg.backend, model=cfg.model, **backend_kwargs)
    registry = build_default_registry(
        workspace=cfg.workspace,
        enable_network=cfg.enable_network_tools,
    )
    memory = Memory.create(
        db_path=cfg.memory_db,
        use_vectors=cfg.use_vectors,
    )
    return Agent(
        backend=llm,
        registry=registry,
        memory=memory,
        max_steps=cfg.max_steps,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
