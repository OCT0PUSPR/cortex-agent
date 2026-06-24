"""LLM backends for cortex-agent.

Exposes the :class:`LLMBackend` protocol, the normalized data types, and a
:func:`get_backend` factory that constructs a backend by name.
"""

from __future__ import annotations

from typing import List, Optional

from .base import LLMBackend, LLMResponse, Message, ToolCall
from .cost import Usage, compute_cost, usage_from
from .mock_backend import MockLLM
from .resilient import BackendUnavailable, ResilientBackend, build_resilient_backend

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "Message",
    "ToolCall",
    "MockLLM",
    "Usage",
    "compute_cost",
    "usage_from",
    "ResilientBackend",
    "BackendUnavailable",
    "build_resilient_backend",
    "get_backend",
    "build_resilient_from_settings",
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
    if name == "tinybrain":
        from ..tinybrain.backend import TinyBrainBackend

        # `model` doubles as the checkpoint path for the local model.
        ckpt = model or kwargs.pop("checkpoint_path", ".cortex/tinybrain")
        return TinyBrainBackend(checkpoint_path=ckpt, **kwargs)
    raise ValueError(f"Unknown backend: {name!r}. Choose mock, anthropic, hf, or tinybrain.")


def build_resilient_from_settings(settings: object) -> ResilientBackend:
    """Build a resilient backend chain (primary + failover) from settings.

    The primary is ``settings.backend``; failover backends come from
    ``settings.fallback_chain`` (deduplicated, primary excluded). Construction
    of a fallback backend never raises here — unavailable backends are skipped.
    """
    primary_name = getattr(settings, "backend", "mock")
    model = getattr(settings, "model", None)
    primary = get_backend(primary_name, model=model)

    fallbacks: List[LLMBackend] = []
    seen = {primary_name}
    for name in getattr(settings, "fallback_chain", ["mock"]):
        if name in seen:
            continue
        seen.add(name)
        try:
            fallbacks.append(get_backend(name))
        except Exception:  # nosec B112  # pragma: no cover - skip un-constructable fallbacks
            # Intentional: a fallback backend that can't be built (missing SDK /
            # checkpoint) is simply omitted from the chain, never fatal.
            continue
    if "mock" not in seen:  # always keep a last-resort offline fallback
        fallbacks.append(MockLLM())

    return build_resilient_backend(
        primary,
        fallbacks,
        timeout_seconds=float(getattr(settings, "llm_timeout_seconds", 60)),
        max_retries=int(getattr(settings, "llm_max_retries", 3)),
    )
