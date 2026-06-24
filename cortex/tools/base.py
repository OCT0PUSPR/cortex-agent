"""Tool abstraction and registry.

A :class:`Tool` bundles a name, a human-readable description, a JSON-schema for
its parameters (used both for validation hints and to build the Anthropic
``tools`` payload), and a ``run`` callable. The :class:`ToolRegistry` holds a
collection of tools and renders them into the Anthropic tool-definition format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    """The outcome of running a tool.

    Attributes:
        output: String rendering of the tool's result (what the model sees).
        is_error: True when the tool failed; surfaced to the model as an error.
        data: Optional structured payload for programmatic consumers / the UI.
    """

    output: str
    is_error: bool = False
    data: Any = None


@dataclass
class Tool:
    """A single callable capability the agent can invoke.

    Attributes:
        name: Unique tool name (snake_case recommended).
        description: When/why to use the tool — the model relies on this.
        parameters: JSON-schema ``properties`` dict for the tool's arguments.
        required: Names of required parameters.
        func: Callable invoked with validated keyword arguments. May return a
            ``ToolResult``, a string, or any value (coerced to ``ToolResult``).
        dangerous: When True, the tool mutates state or reaches the network and
            may be gated behind human approval by a :class:`~cortex.policy.Policy`.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    func: Optional[Callable[..., Any]] = None
    dangerous: bool = False

    def to_schema(self) -> Dict[str, Any]:
        """Render this tool in Anthropic's ``tools`` definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, coercing the return value into a ``ToolResult``."""
        if self.func is None:
            return ToolResult(f"Tool {self.name!r} has no implementation.", is_error=True)
        try:
            result = self.func(**kwargs)
        except TypeError as exc:
            return ToolResult(f"Invalid arguments for {self.name}: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
            return ToolResult(f"Error running {self.name}: {exc}", is_error=True)

        if isinstance(result, ToolResult):
            return result
        return ToolResult(output=str(result), data=result)


class ToolRegistry:
    """An ordered collection of tools keyed by name."""

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add (or replace) a tool in the registry."""
        if not tool.name:
            raise ValueError("Tools must have a non-empty name.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if absent)."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool registered under ``name``, or ``None``."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """True if a tool named ``name`` is registered."""
        return name in self._tools

    def names(self) -> List[str]:
        """Return all registered tool names in insertion order."""
        return list(self._tools.keys())

    def all(self) -> List[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_schemas(self) -> List[Dict[str, Any]]:
        """Render every tool into the Anthropic tool-definition format."""
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Run a tool by name with the supplied arguments."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                f"No such tool: {name!r}. Available: {', '.join(self.names())}",
                is_error=True,
            )
        return tool.run(**(arguments or {}))

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
