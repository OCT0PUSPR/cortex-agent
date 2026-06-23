"""Goal decomposition: turn a high-level goal into an ordered task list.

The planner asks the LLM backend to decompose a goal into a short numbered
list of concrete steps. It is deliberately tolerant: if the model returns prose
instead of a clean list, we fall back to a heuristic split so the agent always
has *some* plan to work from. With the MockLLM the plan is derived purely from
the goal text, keeping it deterministic and offline-friendly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..llm.base import LLMBackend, Message

_PLAN_SYSTEM = (
    "You are a planning module. Given a user's goal, decompose it into a short, "
    "ordered list of concrete steps (3-6 steps). Respond with ONLY a numbered "
    "list, one step per line, no preamble."
)

_NUMBERED_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*)$")


@dataclass
class Plan:
    """An ordered list of steps toward a goal."""

    goal: str
    steps: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.steps

    def render(self) -> str:
        """Render the plan as a numbered string."""
        if not self.steps:
            return "(no explicit plan)"
        return "\n".join(f"{i}. {step}" for i, step in enumerate(self.steps, 1))


def _parse_steps(text: str) -> List[str]:
    """Extract numbered/bulleted steps from model output."""
    steps: List[str] = []
    for line in text.splitlines():
        match = _NUMBERED_RE.match(line)
        if match:
            step = match.group(1).strip()
            if step:
                steps.append(step)
    return steps


def _heuristic_plan(goal: str) -> List[str]:
    """A deterministic fallback plan derived from the goal text."""
    goal = goal.strip()
    # Split on conjunctions / sentence boundaries for a rough multi-step plan.
    parts = re.split(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\.)\s*", goal)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return [f"Handle: {p}" for p in parts[:5]]
    return [
        f"Understand the goal: {goal}",
        "Select and call the most relevant tool",
        "Inspect the observation",
        "Synthesize a final answer",
    ]


class Planner:
    """Decomposes goals into task lists using an LLM backend."""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def plan(self, goal: str, max_tokens: int = 512) -> Plan:
        """Produce a :class:`Plan` for ``goal``.

        For the MockLLM (and on any failure), this falls back to the heuristic
        planner so the agent always has steps to follow.
        """
        # The MockLLM is not a real planner, so skip the round-trip and use the
        # deterministic heuristic directly for reproducible demos/tests.
        if getattr(self.backend, "name", "") == "mock":
            return Plan(goal=goal, steps=_heuristic_plan(goal))

        try:
            resp = self.backend.complete(
                messages=[Message(role="user", content=goal)],
                system=_PLAN_SYSTEM,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            steps = _parse_steps(resp.text)
        except Exception:
            steps = []

        if not steps:
            steps = _heuristic_plan(goal)
        return Plan(goal=goal, steps=steps)
