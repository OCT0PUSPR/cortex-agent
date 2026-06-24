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

from .agent import (
    Agent,
    AgentEvent,
    AgentResult,
    AsyncAgent,
    EventType,
    Plan,
    Planner,
    RunContext,
)
from .config import Settings, get_settings, load_settings
from .llm import (
    LLMBackend,
    LLMResponse,
    Message,
    MockLLM,
    ToolCall,
    build_resilient_from_settings,
    get_backend,
)
from .memory import Memory
from .policy import Budget, Policy
from .tools import Tool, ToolRegistry, ToolResult, build_default_registry

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentResult",
    "AsyncAgent",
    "RunContext",
    "EventType",
    "Plan",
    "Planner",
    "Settings",
    "get_settings",
    "load_settings",
    "LLMBackend",
    "LLMResponse",
    "Message",
    "MockLLM",
    "ToolCall",
    "get_backend",
    "build_resilient_from_settings",
    "Memory",
    "Policy",
    "Budget",
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
        python_cpu_seconds=cfg.python_cpu_seconds,
        python_memory_mb=cfg.python_memory_mb,
        python_wall_seconds=cfg.python_wall_seconds,
        http_max_bytes=cfg.http_max_bytes,
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
