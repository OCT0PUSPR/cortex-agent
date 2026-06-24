"""TinyBrainBackend — serve the from-scratch model via the LLMBackend protocol.

This adapts a locally-trained TinyBrain checkpoint to the cortex
:class:`~cortex.llm.base.LLMBackend` interface (``complete`` + a streaming
variant), so the agent loop, CLI, and API can drive it like any other backend.

Positioning: TinyBrain is a *tiny, zero-dependency local demo brain*, not a
production model. It is far too small to reliably emit structured tool calls, so
the normalized tool-call interface is **best-effort**: the backend parses an
optional ``TOOL: name {json}`` convention from the generated text, but in
practice will usually return plain text. Use Anthropic/HF for real tool use and
TinyBrain to demonstrate a fully self-contained, from-scratch model in the loop.

``torch`` and a trained checkpoint are required at construction time.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from ..llm.base import LLMResponse, Message, ToolCall

_TOOL_RE = re.compile(r"TOOL:\s*(\w+)\s*(\{.*?\})", re.DOTALL)


def _flatten(messages: List[Message], system: Optional[str]) -> str:
    """Render the conversation into a single prompt string for the LM."""
    parts: List[str] = []
    if system:
        parts.append(system.strip())
    for msg in messages:
        if msg.role == "tool":
            joined = "\n".join(str(r.get("content", "")) for r in msg.tool_results)
            parts.append(f"Observation: {joined}")
        elif msg.role == "assistant" and msg.tool_calls:
            calls = "; ".join(f"{c.name}({c.arguments})" for c in msg.tool_calls)
            parts.append(f"Assistant: {msg.content} [{calls}]")
        else:
            parts.append(f"{msg.role.capitalize()}: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


class TinyBrainBackend:
    """LLMBackend backed by a locally-trained TinyBrain model."""

    def __init__(
        self,
        checkpoint_path: str = ".cortex/tinybrain",
        max_new_tokens: int = 200,
        top_k: int = 40,
        device: Optional[str] = None,
    ) -> None:
        self.name = "tinybrain"
        self.checkpoint_path = checkpoint_path
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self._device_pref = device
        self._loaded = None  # (model, tokenizer, device) tuple, lazily loaded

    def _ensure_loaded(self):
        if self._loaded is None:
            from .device import select_device
            from .generate import load_model

            device = select_device(self._device_pref or "auto")
            self._loaded = load_model(self.checkpoint_path, device)
            self.model = self._loaded[0]
        return self._loaded

    # -- generation ----------------------------------------------------- #
    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import torch

        model, tokenizer, device = self._ensure_loaded()
        ids = tokenizer.encode(prompt) or [tokenizer.bos_id]
        # Keep within the model's context window.
        block = model.config.block_size
        if len(ids) > block:
            ids = ids[-block:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=min(max_tokens, self.max_new_tokens),
            temperature=max(temperature, 1e-4),
            top_k=self.top_k,
        )
        full = tokenizer.decode(out[0].tolist())
        # Return only the newly generated continuation.
        return full[len(tokenizer.decode(ids)) :] if full.startswith(tokenizer.decode(ids)) else full

    def _parse_tool(self, text: str) -> Optional[ToolCall]:
        """Best-effort tool-call parse from the ``TOOL: name {json}`` convention."""
        m = _TOOL_RE.search(text)
        if not m:
            return None
        try:
            args = json.loads(m.group(2))
        except (ValueError, TypeError):
            args = {}
        return ToolCall(id=f"tb_{uuid.uuid4().hex[:10]}", name=m.group(1), arguments=args)

    # -- LLMBackend API ------------------------------------------------- #
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        prompt = _flatten(messages, system)
        text = self._generate(prompt, max_tokens, temperature).strip()

        tool_call = self._parse_tool(text) if tools else None
        in_tokens = max(1, len(prompt) // 4)
        out_tokens = max(1, len(text) // 4)
        usage = {"input_tokens": in_tokens, "output_tokens": out_tokens}

        if tool_call is not None:
            return LLMResponse(
                text="",
                tool_calls=[tool_call],
                stop_reason="tool_use",
                model="tinybrain",
                usage=usage,
                raw=text,
            )
        # Trim at the next role marker so the model doesn't ramble into a fake turn.
        for marker in ("\nUser:", "\nObservation:", "\nSystem:"):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        return LLMResponse(
            text=text or "(tinybrain produced no output)",
            stop_reason="end_turn",
            model="tinybrain",
            usage=usage,
            raw=text,
        )

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        """Token-by-token streaming generator (yields text chunks).

        A simple wrapper that decodes one token at a time so a UI can render
        the local model's output as it is produced.
        """
        import torch

        model, tokenizer, device = self._ensure_loaded()
        prompt = _flatten(messages, system)
        ids = tokenizer.encode(prompt) or [tokenizer.bos_id]
        block = model.config.block_size
        if len(ids) > block:
            ids = ids[-block:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        prev_text = ""
        for _ in range(min(max_tokens, self.max_new_tokens)):
            idx = model.generate(idx, max_new_tokens=1, temperature=max(temperature, 1e-4), top_k=self.top_k)
            text = tokenizer.decode(idx[0].tolist())
            chunk = text[len(prev_text) :]
            prev_text = text
            if chunk:
                yield chunk
