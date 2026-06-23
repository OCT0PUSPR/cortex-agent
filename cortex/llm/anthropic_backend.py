"""Anthropic (Claude) backend with native tool use.

Uses the official ``anthropic`` SDK. Tool use is implemented with Anthropic's
native ``tools`` parameter plus ``tool_use`` / ``tool_result`` content blocks,
which we translate to and from the normalized :mod:`cortex.llm.base` types.

The ``anthropic`` package is imported lazily so the rest of the framework (and
the MockLLM-driven tests) runs without it installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import LLMResponse, Message, ToolCall

# Current Anthropic model IDs (see project README). claude-sonnet-4-6 is the
# balanced default; claude-opus-4-8 is the most capable; claude-haiku-4-5 is
# the fast/cheap option.
DEFAULT_MODEL = "claude-sonnet-4-6"


def _to_anthropic_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """Translate normalized messages into Anthropic content-block format."""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            # Tool results are sent as a user turn of tool_result blocks.
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": r["tool_use_id"],
                    "content": r["content"],
                    **({"is_error": True} if r.get("is_error") else {}),
                }
                for r in msg.tool_results
            ]
            out.append({"role": "user", "content": content})
        elif msg.role == "assistant" and msg.tool_calls:
            blocks: List[Dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": msg.role, "content": msg.content})
    return out


class AnthropicBackend:
    """Claude backend using the Anthropic Messages API with tool use."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self.name = "anthropic"
        self.model = model
        self._api_key = api_key
        self._client = None  # lazily constructed

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - exercised only w/o SDK
            raise RuntimeError(
                "The 'anthropic' package is required for the Anthropic backend. "
                "Install it with `pip install anthropic`, or use --backend mock."
            ) from exc
        # The SDK reads ANTHROPIC_API_KEY from the environment automatically;
        # pass an explicit key only when one was supplied.
        self._client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
        return self._client

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        client = self._ensure_client()

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = client.messages.create(**kwargs)

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input or {}),
                    )
                )

        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }

        return LLMResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=getattr(resp, "stop_reason", "end_turn") or "end_turn",
            raw=resp,
            usage=usage,
        )
