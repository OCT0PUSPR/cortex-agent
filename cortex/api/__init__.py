"""HTTP API for cortex-agent (FastAPI + SSE)."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    """Lazily import and build the FastAPI app (keeps server deps optional)."""
    from .server import create_app as _create_app

    return _create_app()
