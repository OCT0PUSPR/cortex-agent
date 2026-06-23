"""Deterministic MockLLM backend.

Drives a believable multi-step ReAct trajectory with no API key and no network,
so the full agent loop, the CLI demo, the web UI, and the test suite all run
end to end offline.

The mock inspects the conversation it is given and decides what to do next:

1. On the first turn it emits a short "thought" and calls a tool that is
   relevant to the goal (calculator, current_time, read_file, ...).
2. After it sees a tool result, it either calls a second relevant tool or
   synthesizes a final answer that quotes the observed result.

This is not a real model — it is a scripted planner whose only job is to make
the agent loop exercise real tool execution and memory in a reproducible way.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from .base import LLMResponse, Message, ToolCall


def _new_id() -> str:
    return f"mock_{uuid.uuid4().hex[:12]}"


def _last_user_goal(messages: List[Message]) -> str:
    for msg in messages:
        if msg.role == "user" and msg.content:
            return msg.content
    return ""


def _tool_names(tools: Optional[List[Dict[str, Any]]]) -> set:
    return {t["name"] for t in tools} if tools else set()


def _collect_observations(messages: List[Message]) -> List[str]:
    obs: List[str] = []
    for msg in messages:
        if msg.role == "tool":
            for r in msg.tool_results:
                obs.append(str(r.get("content", "")))
    return obs


def _already_called(messages: List[Message], name: str) -> bool:
    for msg in messages:
        for call in msg.tool_calls:
            if call.name == name:
                return True
    return False


_ARITHMETIC_RE = re.compile(r"[-+]?\d[\d\s.]*[-+*/x][\d\s.+\-*/x()]*\d")


class MockLLM:
    """A scripted, deterministic backend for offline demos and tests."""

    def __init__(self, model: str = "mock-1") -> None:
        self.name = "mock"
        self.model = model

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        goal = _last_user_goal(messages)
        goal_lower = goal.lower()
        available = _tool_names(tools)
        observations = _collect_observations(messages)

        # --- Step 1: choose a first tool based on the goal ----------------
        if not observations:
            call = self._first_tool(goal, goal_lower, available, messages)
            if call is not None:
                return LLMResponse(
                    text=f"I'll start by using the `{call.name}` tool to make progress on: {goal}",
                    tool_calls=[call],
                    stop_reason="tool_use",
                )
            # No relevant tool -> answer directly.
            return LLMResponse(
                text=self._final_answer(goal, observations),
                stop_reason="end_turn",
            )

        # --- Step 2: optionally chain a second, different tool ------------
        second = self._second_tool(goal, goal_lower, available, messages)
        if second is not None:
            return LLMResponse(
                text=f"Good. Now I'll use `{second.name}` to finish the task.",
                tool_calls=[second],
                stop_reason="tool_use",
            )

        # --- Step 3: synthesize the final answer --------------------------
        return LLMResponse(
            text=self._final_answer(goal, observations),
            stop_reason="end_turn",
        )

    # ------------------------------------------------------------------ #
    def _first_tool(
        self,
        goal: str,
        goal_lower: str,
        available: set,
        messages: List[Message],
    ) -> Optional[ToolCall]:
        match = _ARITHMETIC_RE.search(goal)
        if "calculator" in available and (
            match or any(w in goal_lower for w in ("calculate", "compute", "sum", "multiply", "plus", "times"))
        ):
            expr = match.group(0).strip() if match else "2 + 2"
            expr = expr.replace("x", "*")
            return ToolCall(_new_id(), "calculator", {"expression": expr})

        if "current_time" in available and any(
            w in goal_lower for w in ("time", "date", "today", "now", "clock")
        ):
            return ToolCall(_new_id(), "current_time", {})

        if "read_file" in available and "read" in goal_lower:
            fname = self._guess_filename(goal) or "notes.txt"
            return ToolCall(_new_id(), "read_file", {"path": fname})

        if "write_file" in available and "write" in goal_lower:
            fname = self._guess_filename(goal) or "output.txt"
            return ToolCall(
                _new_id(),
                "write_file",
                {"path": fname, "content": f"Generated for goal: {goal}"},
            )

        if "web_search" in available and any(
            w in goal_lower for w in ("search", "find", "look up", "what is", "who is")
        ):
            return ToolCall(_new_id(), "web_search", {"query": goal})

        if "run_python" in available and any(
            w in goal_lower for w in ("python", "script", "code", "program")
        ):
            return ToolCall(
                _new_id(),
                "run_python",
                {"code": "print(sum(range(1, 11)))"},
            )

        if "http_get" in available and ("http://" in goal or "https://" in goal):
            url_match = re.search(r"https?://\S+", goal)
            if url_match:
                return ToolCall(_new_id(), "http_get", {"url": url_match.group(0)})

        # Default: if a calculator exists, do a trivial calculation so the
        # demo always shows a real tool execution.
        if "calculator" in available:
            return ToolCall(_new_id(), "calculator", {"expression": "21 * 2"})
        if "current_time" in available:
            return ToolCall(_new_id(), "current_time", {})
        return None

    def _second_tool(
        self,
        goal: str,
        goal_lower: str,
        available: set,
        messages: List[Message],
    ) -> Optional[ToolCall]:
        # Chain time onto a calculation request that mentions logging/recording.
        if (
            "current_time" in available
            and not _already_called(messages, "current_time")
            and any(w in goal_lower for w in ("log", "record", "timestamp", "when"))
        ):
            return ToolCall(_new_id(), "current_time", {})
        # Chain a write after a read when asked to copy/transform.
        if (
            "write_file" in available
            and not _already_called(messages, "write_file")
            and _already_called(messages, "read_file")
            and any(w in goal_lower for w in ("copy", "save", "write", "store"))
        ):
            return ToolCall(
                _new_id(),
                "write_file",
                {"path": "result.txt", "content": "Processed content"},
            )
        return None

    def _final_answer(self, goal: str, observations: List[str]) -> str:
        if observations:
            joined = "; ".join(o.strip() for o in observations if o.strip())
            return (
                f"Done. For your request \"{goal}\", I used my tools and found: "
                f"{joined}. That completes the task."
            )
        return (
            f"For your request \"{goal}\": I considered the available tools and "
            f"can answer directly. (No tool call was necessary.)"
        )

    @staticmethod
    def _guess_filename(goal: str) -> Optional[str]:
        m = re.search(r"[\w./-]+\.\w{1,5}", goal)
        return m.group(0) if m else None
