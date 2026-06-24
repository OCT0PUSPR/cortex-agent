"""Device auto-selection: MPS > CUDA > CPU."""

from __future__ import annotations


def select_device(prefer: str = "auto"):
    """Return the best available torch device.

    Order: explicit preference if available, else MPS (Apple Silicon), then
    CUDA, then CPU.
    """
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device) -> str:
    """Human-readable device label."""
    return str(device)
