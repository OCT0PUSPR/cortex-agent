"""Configuration via pydantic-settings.

Settings are read from environment variables (prefixed ``CORTEX_``) and an
optional ``.env`` file. API keys live in their conventional env vars
(``ANTHROPIC_API_KEY``, ``HF_TOKEN``) and are read by the backends directly, so
they are never hardcoded here.

A small compatibility shim lets the module import even if ``pydantic-settings``
is not installed (e.g. a minimal environment), falling back to a plain
dataclass-style object backed by ``os.environ``.
"""

from __future__ import annotations

import os
from typing import Optional

try:  # Preferred: pydantic-settings
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Runtime configuration for cortex-agent."""

        model_config = SettingsConfigDict(
            env_prefix="CORTEX_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        backend: str = Field(
            default="mock",
            description="LLM backend: 'mock', 'anthropic', or 'hf'.",
        )
        model: Optional[str] = Field(
            default=None,
            description="Model id override; backend default when unset.",
        )
        max_steps: int = Field(default=8, description="Max ReAct steps per run.")
        max_tokens: int = Field(default=2048, description="Max output tokens per call.")
        temperature: float = Field(default=0.7, description="Sampling temperature.")
        workspace: str = Field(
            default=".cortex/workspace",
            description="Sandbox directory for file/python tools.",
        )
        memory_db: str = Field(
            default=".cortex/memory.sqlite",
            description="SQLite path for long-term memory.",
        )
        use_vectors: bool = Field(
            default=False,
            description="Enable sentence-transformers vector recall when available.",
        )
        enable_network_tools: bool = Field(
            default=True,
            description="Expose network tools (http_get) to the agent.",
        )
        host: str = Field(default="127.0.0.1", description="API server host.")
        port: int = Field(default=8000, description="API server port.")

except ImportError:  # pragma: no cover - fallback when pydantic-settings absent

    class Settings:  # type: ignore[no-redef]
        """Minimal env-backed settings fallback."""

        def __init__(self, **overrides) -> None:
            def env(key: str, default: str) -> str:
                return os.environ.get(f"CORTEX_{key.upper()}", default)

            self.backend = overrides.get("backend", env("backend", "mock"))
            self.model = overrides.get("model", os.environ.get("CORTEX_MODEL") or None)
            self.max_steps = int(overrides.get("max_steps", env("max_steps", "8")))
            self.max_tokens = int(overrides.get("max_tokens", env("max_tokens", "2048")))
            self.temperature = float(overrides.get("temperature", env("temperature", "0.7")))
            self.workspace = overrides.get("workspace", env("workspace", ".cortex/workspace"))
            self.memory_db = overrides.get("memory_db", env("memory_db", ".cortex/memory.sqlite"))
            self.use_vectors = bool(overrides.get("use_vectors", env("use_vectors", "") == "true"))
            self.enable_network_tools = bool(
                overrides.get("enable_network_tools", env("enable_network_tools", "true") != "false")
            )
            self.host = overrides.get("host", env("host", "127.0.0.1"))
            self.port = int(overrides.get("port", env("port", "8000")))


def load_settings(**overrides) -> "Settings":
    """Load settings from the environment, applying any keyword overrides."""
    settings = Settings()
    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    return settings
