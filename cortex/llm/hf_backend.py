"""HuggingFace backend.

Supports two modes, selectable via ``mode``:

* ``"api"`` (default): call the HF Inference API over HTTPS with ``httpx``.
* ``"local"``: load a model with ``transformers`` and run it in-process.

Tool use is emulated with a JSON protocol: the system prompt instructs the
model to emit a fenced ``json`` block of the form
``{"tool": "name", "arguments": {...}}`` when it wants to call a tool. We parse
that out and surface it as a normalized :class:`ToolCall`. This keeps open
models (which lack Anthropic-native tool use) interoperable with the agent loop.

Heavy/optional imports (``httpx``, ``transformers``, ``torch``) are guarded so
the rest of the framework imports cleanly without them.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from .base import LLMResponse, Message, ToolCall

HF_API_URL = "https://api-inference.huggingface.co/models/"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_TOOL_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _build_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Render a tool-use instruction block for models without native tools."""
    lines = [
        "You can call tools. To call a tool, respond with ONLY a fenced json "
        "block of the form:",
        '```json',
        '{"tool": "<tool_name>", "arguments": {<args>}}',
        '```',
        "Available tools:",
    ]
    for tool in tools:
        schema = json.dumps(tool.get("input_schema", {}).get("properties", {}))
        lines.append(f"- {tool['name']}: {tool.get('description', '')} params={schema}")
    lines.append(
        "If you have the final answer and need no tool, respond in plain text."
    )
    return "\n".join(lines)


def _parse_tool_call(text: str) -> Optional[ToolCall]:
    """Extract a tool call from a model response, if present."""
    match = _TOOL_BLOCK_RE.search(text)
    candidate = match.group(1) if match else (text.strip() if text.strip().startswith("{") else None)
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if isinstance(data, dict) and "tool" in data:
        return ToolCall(
            id=f"hf_{uuid.uuid4().hex[:12]}",
            name=str(data["tool"]),
            arguments=dict(data.get("arguments", {})),
        )
    return None


def _flatten_messages(messages: List[Message], system: Optional[str]) -> List[Dict[str, str]]:
    """Collapse normalized messages into simple role/content chat turns."""
    chat: List[Dict[str, str]] = []
    if system:
        chat.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "tool":
            joined = "\n".join(str(r.get("content", "")) for r in msg.tool_results)
            chat.append({"role": "user", "content": f"Tool result:\n{joined}"})
        elif msg.role == "assistant" and msg.tool_calls:
            calls = "; ".join(f"{c.name}({c.arguments})" for c in msg.tool_calls)
            text = (msg.content + "\n" if msg.content else "") + f"[called: {calls}]"
            chat.append({"role": "assistant", "content": text})
        else:
            chat.append({"role": msg.role, "content": msg.content})
    return chat


class HFBackend:
    """HuggingFace Inference API / local-transformers backend."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        mode: str = "api",
        api_token: Optional[str] = None,
    ) -> None:
        self.name = "hf"
        self.model = model
        self.mode = mode
        self._api_token = api_token or os.environ.get("HF_TOKEN")
        self._pipeline = None  # for local mode

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        full_system = system or ""
        if tools:
            full_system = (full_system + "\n\n" + _build_tool_prompt(tools)).strip()

        chat = _flatten_messages(messages, full_system)

        if self.mode == "local":
            text = self._complete_local(chat, max_tokens, temperature)
        else:
            text = self._complete_api(chat, max_tokens, temperature)

        tool_call = _parse_tool_call(text) if tools else None
        if tool_call is not None:
            return LLMResponse(
                text="",
                tool_calls=[tool_call],
                stop_reason="tool_use",
                raw=text,
            )
        return LLMResponse(text=text.strip(), stop_reason="end_turn", raw=text)

    def _complete_api(
        self, chat: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is required for the HF API backend.") from exc

        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        # Render a simple prompt from the chat turns; many HF text-generation
        # endpoints accept a single "inputs" string.
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in chat)
        prompt += "\nassistant:"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "return_full_text": False,
            },
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(HF_API_URL + self.model, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("generated_text", "")
        if isinstance(data, dict):
            return data.get("generated_text", "")
        return str(data)

    def _complete_local(
        self, chat: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:  # pragma: no cover - requires heavy deps not installed in CI
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "transformers (and torch) are required for HF local mode."
                ) from exc
            self._pipeline = pipeline("text-generation", model=self.model)

        # Most chat models expose a chat template via the tokenizer.
        try:
            out = self._pipeline(
                chat,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
            )
            generated = out[0]["generated_text"]
            if isinstance(generated, list):  # chat-template returns message list
                return generated[-1]["content"]
            return str(generated)
        except Exception:  # fall back to plain prompt
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in chat)
            out = self._pipeline(prompt, max_new_tokens=max_tokens)
            return out[0]["generated_text"]
