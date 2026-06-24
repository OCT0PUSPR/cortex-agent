"""TinyBrain — a from-scratch decoder-only Transformer LM.

This package implements a small GPT-style language model (RoPE, RMSNorm,
multi-head causal attention, SwiGLU MLP, weight tying) along with a BPE
tokenizer, a real training pipeline, evaluation, generation, and a
:class:`~cortex.tinybrain.backend.TinyBrainBackend` that plugs into the cortex
``LLMBackend`` protocol.

``torch`` is required for everything here, so it is imported lazily: importing
this package never raises if torch is missing. Use :func:`torch_available` to
check, and import the submodules (``model``, ``train``, ...) only when torch is
present. This keeps the MockLLM CI path importable without torch.
"""

from __future__ import annotations


def torch_available() -> bool:
    """Return True if PyTorch is importable in this environment."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = ["torch_available"]


def __getattr__(name: str):
    """Lazily expose torch-dependent symbols only when torch is installed."""
    if name in {"TinyBrain", "TinyBrainConfig"}:
        from .model import TinyBrain, TinyBrainConfig

        return {"TinyBrain": TinyBrain, "TinyBrainConfig": TinyBrainConfig}[name]
    if name in {"TrainConfig", "train"}:
        from .train import TrainConfig, train

        return {"TrainConfig": TrainConfig, "train": train}[name]
    if name == "TinyBrainBackend":
        from .backend import TinyBrainBackend

        return TinyBrainBackend
    if name == "evaluate":
        from .eval import evaluate

        return evaluate
    if name == "generate_text":
        from .generate import generate_text

        return generate_text
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
