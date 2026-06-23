"""The core Agent: a ReAct (think -> act -> observe) loop.

The :class:`Agent` ties together an LLM backend, a tool registry, and memory:

1. A :class:`~cortex.agent.planner.Planner` decomposes the goal into steps.
2. The agent loops: it calls the model with the conversation + tool schemas,
   the model emits a thought and optionally a tool call, the agent executes the
   tool, appends the observation, and repeats — up to ``max_steps``.
3. When the model stops requesting tools (or the budget is exhausted), the
   agent synthesizes a final answer.

Every meaningful event is emitted as a structured :class:`AgentEvent`, so a CLI
or web UI can stream the agent's reasoning live. Use :meth:`Agent.stream` to
consume events as they happen, or :meth:`Agent.run` for a blocking call that
returns the final :class:`AgentResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from ..llm.base import LLMBackend, Message
from ..memory.store import Memory
from ..tools.base import ToolRegistry
from .planner import Plan, Planner

DEFAULT_SYSTEM = (
    "You are Cortex, an autonomous agent. You solve the user's goal by reasoning "
    "step by step and using the available tools. Think briefly about what to do, "
    "then call a tool. After observing tool results, continue until you can give "
    "a complete final answer. When you have enough information, stop calling "
    "tools and respond with your final answer in plain text."
)


class EventType(str, Enum):
    """Kinds of structured events the agent emits."""

    PLAN = "plan"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    ANSWER = "answer"
    ERROR = "error"


@dataclass
class AgentEvent:
    """A single structured event in the agent's trajectory."""

    type: EventType
    content: str = ""
    step: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event (e.g. for SSE / JSON streaming)."""
        return {
            "type": self.type.value,
            "content": self.content,
            "step": self.step,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentResult:
    """The terminal outcome of an agent run."""

    answer: str
    plan: Plan
    events: List[AgentEvent] = field(default_factory=list)
    steps_used: int = 0
    completed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "plan": self.plan.steps,
            "steps_used": self.steps_used,
            "completed": self.completed,
            "events": [e.to_dict() for e in self.events],
        }


class Agent:
    """An autonomous ReAct agent with planning, tools, and memory."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        memory: Optional[Memory] = None,
        system_prompt: str = DEFAULT_SYSTEM,
        max_steps: int = 8,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.memory = memory
        self.planner = Planner(backend)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    def stream(self, goal: str) -> Iterator[AgentEvent]:
        """Run the agent, yielding :class:`AgentEvent`s as they occur."""
        # --- Plan --------------------------------------------------------
        plan = self.planner.plan(goal)
        yield AgentEvent(
            type=EventType.PLAN,
            content=plan.render(),
            step=0,
            data={"steps": plan.steps},
        )

        # --- Build the initial context, enriched with recalled memory ----
        system = self._build_system(goal)
        messages: List[Message] = [Message(role="user", content=goal)]

        if self.memory is not None:
            self.memory.observe("user", goal)

        tool_schemas = self.registry.to_schemas()
        final_answer = ""
        completed = False
        step = 0

        # --- ReAct loop --------------------------------------------------
        for step in range(1, self.max_steps + 1):
            try:
                resp = self.backend.complete(
                    messages=messages,
                    tools=tool_schemas,
                    system=system,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as exc:  # noqa: BLE001 - surface backend failures
                yield AgentEvent(
                    type=EventType.ERROR,
                    content=f"Backend error: {exc}",
                    step=step,
                )
                final_answer = f"The agent failed due to a backend error: {exc}"
                break

            if resp.text:
                yield AgentEvent(type=EventType.THOUGHT, content=resp.text, step=step)

            # Model wants to stop and answer.
            if not resp.wants_tools:
                final_answer = resp.text or "(no answer produced)"
                completed = True
                break

            # Record the assistant turn (text + tool calls) so the next request
            # includes the tool_use blocks the provider expects.
            messages.append(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
            )

            # Execute each requested tool and gather results.
            tool_results: List[Dict[str, Any]] = []
            for call in resp.tool_calls:
                yield AgentEvent(
                    type=EventType.TOOL_CALL,
                    content=f"{call.name}({_render_args(call.arguments)})",
                    step=step,
                    data={"tool": call.name, "arguments": call.arguments, "id": call.id},
                )
                result = self.registry.execute(call.name, call.arguments)
                tool_results.append(
                    {
                        "tool_use_id": call.id,
                        "content": result.output,
                        "is_error": result.is_error,
                    }
                )
                yield AgentEvent(
                    type=EventType.OBSERVATION,
                    content=result.output,
                    step=step,
                    data={"tool": call.name, "is_error": result.is_error},
                )
                if self.memory is not None and not result.is_error:
                    self.memory.remember(
                        f"Used {call.name}{_render_args(call.arguments)} -> {result.output}",
                        kind="observation",
                    )

            messages.append(Message(role="tool", tool_results=tool_results))
        else:
            # Loop exhausted without a natural stop: ask for a final synthesis.
            final_answer = self._force_final_answer(messages, system)
            completed = False

        if self.memory is not None and final_answer:
            self.memory.observe("assistant", final_answer)
            self.memory.remember(f"Goal: {goal} | Answer: {final_answer}", kind="conclusion")

        yield AgentEvent(
            type=EventType.ANSWER,
            content=final_answer,
            step=step,
            data={"completed": completed},
        )

    # ------------------------------------------------------------------ #
    def run(self, goal: str) -> AgentResult:
        """Run the agent to completion and return the :class:`AgentResult`."""
        events: List[AgentEvent] = []
        answer = ""
        plan = Plan(goal=goal, steps=[])
        completed = True
        steps_used = 0

        for event in self.stream(goal):
            events.append(event)
            if event.type is EventType.PLAN:
                plan = Plan(goal=goal, steps=event.data.get("steps", []))
            elif event.type is EventType.ANSWER:
                answer = event.content
                completed = bool(event.data.get("completed", True))
            steps_used = max(steps_used, event.step)

        return AgentResult(
            answer=answer,
            plan=plan,
            events=events,
            steps_used=steps_used,
            completed=completed,
        )

    # ------------------------------------------------------------------ #
    def _build_system(self, goal: str) -> str:
        """Augment the system prompt with the tool list and recalled memory."""
        parts = [self.system_prompt]
        names = self.registry.names()
        if names:
            parts.append("Available tools: " + ", ".join(names) + ".")
        if self.memory is not None:
            recalled = self.memory.recall(goal, top_k=3)
            if recalled:
                notes = "\n".join(f"- {r.content}" for r in recalled)
                parts.append("Relevant memories from past runs:\n" + notes)
        return "\n\n".join(parts)

    def _force_final_answer(self, messages: List[Message], system: str) -> str:
        """Ask the model for a final answer after the step budget is used up."""
        prompt = list(messages) + [
            Message(
                role="user",
                content=(
                    "You have reached the step budget. Using everything observed so "
                    "far, give your best final answer now in plain text."
                ),
            )
        ]
        try:
            resp = self.backend.complete(
                messages=prompt,
                tools=None,
                system=system,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return resp.text or "(no answer produced within the step budget)"
        except Exception as exc:  # noqa: BLE001
            return f"Could not synthesize a final answer: {exc}"


def _render_args(arguments: Dict[str, Any]) -> str:
    """Render tool arguments compactly for display."""
    if not arguments:
        return "()"
    inner = ", ".join(f"{k}={_truncate(v)!r}" for k, v in arguments.items())
    return f"({inner})"


def _truncate(value: Any, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
