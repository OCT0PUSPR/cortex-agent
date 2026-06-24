"""Token-cost accounting for LLM providers.

A small per-model price table (USD per 1M tokens, input/output) lets the agent
loop convert token usage into a dollar cost that feeds the run budget and the
Prometheus cost metric. Unknown models fall back to a conservative default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# USD per 1,000,000 tokens: (input, output). Sourced from current public pricing.
_PRICE_TABLE: Dict[str, Tuple[float, float]] = {
    # Anthropic Claude
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # HuggingFace open models (self-host ~ free; nominal token price for accounting)
    "Qwen/Qwen2.5-7B-Instruct": (0.0, 0.0),
    "meta-llama/Llama-3.1-8B-Instruct": (0.0, 0.0),
    # Mock backend is free.
    "mock-1": (0.0, 0.0),
}

# Conservative default for unknown models (assume a mid-tier price).
_DEFAULT_PRICE = (3.0, 15.0)


@dataclass
class Usage:
    """Token usage and its computed dollar cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd


def price_for(model: str) -> Tuple[float, float]:
    """Return the (input, output) USD-per-1M price for a model."""
    return _PRICE_TABLE.get(model, _DEFAULT_PRICE)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute the USD cost of a call given token counts."""
    in_price, out_price = price_for(model)
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def usage_from(model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Usage:
    """Build a :class:`Usage` (with cost) from token counts."""
    it = int(input_tokens or 0)
    ot = int(output_tokens or 0)
    return Usage(input_tokens=it, output_tokens=ot, cost_usd=compute_cost(model, it, ot))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for backends that omit usage."""
    return max(1, len(text or "") // 4)
