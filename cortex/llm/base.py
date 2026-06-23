"""LLM backend protocol and normalized data types.

Every backend (Anthropic, HuggingFace, Mock) implements the :class:`LLMBackend`
protocol so the agent loop can treat them interchangeably. Tool calls are
normalized into :class:`ToolCall` objects regardless of the wire format the
provider uses, so the ReAct loop never has to special-case a backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single normalized tool invocation requested by the model.

    Attributes:
        id: Provider-assigned identifier, echoed back when returning results.
        name: The tool's registered name.
        arguments: Parsed keyword arguments for the tool.
    """

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A normalized response from any backend.

    Attributes:
        text: Free-form assistant text (the model's "thought" or final answer).
        tool_calls: Zero or more tool calls the model wants executed.
        stop_reason: Why generation stopped ("end_turn", "tool_use", ...).
        raw: The provider's raw response object, for debugging.
        usage: Optional token-usage dict (input/output) when available.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None
    usage: Optional[Dict[str, int]] = None

    @property
    def wants_tools(self) -> bool:
        """True when the model requested at least one tool call."""
        return bool(self.tool_calls)


@dataclass
class Message:
    """A conversation message in a provider-agnostic shape.

    ``role`` is one of ``"user"``, ``"assistant"``, or ``"tool"``. For tool
    results, ``tool_results`` carries ``(tool_use_id, content, is_error)``
    tuples; for assistant turns that called tools, ``tool_calls`` is populated.
    """

    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class LLMBackend(Protocol):
    """Interface every backend must implement.

    Implementations translate the normalized ``messages`` / ``tools`` inputs
    into their provider-specific request, call the model, and return a
    normalized :class:`LLMResponse`.
    """

    name: str

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a single completion.

        Args:
            messages: Conversation history (provider-agnostic).
            tools: JSON-schema tool definitions (Anthropic tool format).
            system: Optional system prompt.
            max_tokens: Output token cap.
            temperature: Sampling temperature (ignored by some backends).

        Returns:
            A normalized :class:`LLMResponse`.
        """
        ...
