"""LLM backends for cortex-agent.

Exposes the :class:`LLMBackend` protocol, the normalized data types, and a
:func:`get_backend` factory that constructs a backend by name.
"""

from __future__ import annotations

from typing import Optional

from .base import LLMBackend, LLMResponse, Message, ToolCall
from .mock_backend import MockLLM

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "Message",
    "ToolCall",
    "MockLLM",
    "get_backend",
]


def get_backend(
    name: str,
    model: Optional[str] = None,
    **kwargs,
) -> LLMBackend:
    """Construct a backend by name.

    Args:
        name: One of ``"mock"``, ``"anthropic"``, or ``"hf"``.
        model: Optional model id override.
        **kwargs: Backend-specific options (api_key, mode, ...).

    Returns:
        A constructed backend implementing :class:`LLMBackend`.
    """
    name = name.lower()
    if name == "mock":
        return MockLLM(model=model or "mock-1")
    if name == "anthropic":
        from .anthropic_backend import DEFAULT_MODEL, AnthropicBackend

        return AnthropicBackend(model=model or DEFAULT_MODEL, **kwargs)
    if name == "hf":
        from .hf_backend import DEFAULT_MODEL, HFBackend

        return HFBackend(model=model or DEFAULT_MODEL, **kwargs)
    raise ValueError(f"Unknown backend: {name!r}. Choose mock, anthropic, or hf.")
