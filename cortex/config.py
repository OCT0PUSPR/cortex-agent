"""Configuration via pydantic-settings.

Settings are read from environment variables (prefixed ``CORTEX_``) and an
optional ``.env`` file. API keys live in their conventional env vars
(``ANTHROPIC_API_KEY``, ``HF_TOKEN``) and are read by the backends directly, so
they are never hardcoded here.

A small compatibility shim lets the module import even if ``pydantic-settings``
is not installed (e.g. a minimal environment), falling back to a plain
object backed by ``os.environ``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

try:  # Preferred: pydantic-settings
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Runtime configuration for cortex-agent."""

        model_config = SettingsConfigDict(
            env_prefix="CORTEX_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # -- LLM / agent --------------------------------------------------
        backend: str = Field(default="mock", description="LLM backend: mock, anthropic, hf.")
        model: Optional[str] = Field(default=None, description="Model id override.")
        fallback_backends: str = Field(
            default="hf,mock",
            description="Comma-separated failover chain used when the primary backend fails.",
        )
        max_steps: int = Field(default=8, description="Max ReAct steps per run.")
        max_tokens: int = Field(default=2048, description="Max output tokens per call.")
        max_total_tokens: int = Field(default=100_000, description="Hard token budget per run (stops the loop).")
        max_cost_usd: float = Field(default=1.0, description="Hard USD cost budget per run (stops the loop).")
        temperature: float = Field(default=0.7, description="Sampling temperature.")
        run_timeout_seconds: int = Field(default=300, description="Wall-clock timeout for an entire run.")
        llm_timeout_seconds: int = Field(default=60, description="Per-LLM-call timeout.")
        llm_max_retries: int = Field(default=3, description="LLM retry attempts.")

        # -- tools / sandbox ----------------------------------------------
        workspace: str = Field(default=".cortex/workspace", description="File/python sandbox dir.")
        enable_network_tools: bool = Field(default=True, description="Expose http_get.")
        tool_timeout_seconds: int = Field(default=15, description="Per-tool timeout.")
        python_cpu_seconds: int = Field(default=5, description="run_python CPU rlimit (s).")
        python_memory_mb: int = Field(default=256, description="run_python memory rlimit (MB).")
        python_wall_seconds: int = Field(default=10, description="run_python wall-clock timeout.")
        require_approval: bool = Field(
            default=False,
            description="Require human approval for dangerous tools (write/python/http).",
        )
        http_max_bytes: int = Field(default=100_000, description="http_get response size cap.")

        # -- memory / state -----------------------------------------------
        memory_db: str = Field(default=".cortex/memory.sqlite", description="Legacy SQLite memory.")
        database_url: str = Field(
            default="sqlite+aiosqlite:///./.cortex/cortex.db",
            description="Async DB URL (sqlite default; Postgres via env).",
        )
        sync_database_url: Optional[str] = Field(
            default=None,
            description="Sync DB URL for Alembic/CLI; derived from database_url if unset.",
        )
        use_vectors: bool = Field(default=False, description="Enable vector recall when available.")

        # -- queue / scale ------------------------------------------------
        redis_url: str = Field(default="redis://localhost:6379", description="Redis for arq queue.")
        use_queue: bool = Field(default=False, description="Enqueue runs via arq instead of inline.")
        max_concurrent_runs: int = Field(default=16, description="Concurrency limit per process.")

        # -- security -----------------------------------------------------
        jwt_secret: str = Field(
            default="dev-insecure-change-me",
            description="HS256 signing secret for JWTs (override in production).",
        )
        jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm.")
        jwt_expire_minutes: int = Field(default=60, description="Access-token lifetime.")
        auth_required: bool = Field(default=False, description="Require auth on protected endpoints.")
        cors_origins: str = Field(
            default="http://localhost:8000,http://127.0.0.1:8000",
            description="Comma-separated CORS allowlist.",
        )
        rate_limit_per_minute: int = Field(default=30, description="Requests/min per principal.")
        max_goal_length: int = Field(default=4000, description="Max characters in a goal.")

        # -- server / observability ---------------------------------------
        host: str = Field(default="127.0.0.1", description="API server host.")
        port: int = Field(default=8000, description="API server port.")
        log_level: str = Field(default="INFO", description="Log level.")
        log_json: bool = Field(default=True, description="Emit JSON logs (vs console).")
        enable_metrics: bool = Field(default=True, description="Expose /metrics.")
        enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry spans.")

        @field_validator("backend")
        @classmethod
        def _valid_backend(cls, v: str) -> str:
            if v not in {"mock", "anthropic", "hf", "tinybrain"}:
                raise ValueError("backend must be one of: mock, anthropic, hf, tinybrain")
            return v

        # -- derived helpers ----------------------------------------------
        @property
        def cors_origin_list(self) -> List[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

        @property
        def fallback_chain(self) -> List[str]:
            return [b.strip() for b in self.fallback_backends.split(",") if b.strip()]

        def resolved_sync_url(self) -> str:
            """Return a synchronous DB URL for Alembic/migrations."""
            if self.sync_database_url:
                return self.sync_database_url
            url = self.database_url
            return (
                url.replace("+aiosqlite", "")
                .replace("+asyncpg", "+psycopg")
                .replace("postgresql+psycopg", "postgresql")
            )

except ImportError:  # pragma: no cover - fallback when pydantic-settings absent

    class Settings:  # type: ignore[no-redef]
        """Minimal env-backed settings fallback (no pydantic available)."""

        def __init__(self, **overrides: object) -> None:
            def env(key: str, default: str) -> str:
                return os.environ.get(f"CORTEX_{key.upper()}", default)

            def b(key: str, default: bool) -> bool:
                raw = env(key, "true" if default else "false").lower()
                return raw in {"1", "true", "yes", "on"}

            self.backend = str(overrides.get("backend", env("backend", "mock")))
            self.model = overrides.get("model", os.environ.get("CORTEX_MODEL") or None)
            self.fallback_backends = env("fallback_backends", "hf,mock")
            self.max_steps = int(str(overrides.get("max_steps", env("max_steps", "8"))))
            self.max_tokens = int(str(overrides.get("max_tokens", env("max_tokens", "2048"))))
            self.max_total_tokens = int(env("max_total_tokens", "100000"))
            self.max_cost_usd = float(env("max_cost_usd", "1.0"))
            self.temperature = float(str(overrides.get("temperature", env("temperature", "0.7"))))
            self.run_timeout_seconds = int(env("run_timeout_seconds", "300"))
            self.llm_timeout_seconds = int(env("llm_timeout_seconds", "60"))
            self.llm_max_retries = int(env("llm_max_retries", "3"))
            self.workspace = str(overrides.get("workspace", env("workspace", ".cortex/workspace")))
            self.enable_network_tools = bool(overrides.get("enable_network_tools", b("enable_network_tools", True)))
            self.tool_timeout_seconds = int(env("tool_timeout_seconds", "15"))
            self.python_cpu_seconds = int(env("python_cpu_seconds", "5"))
            self.python_memory_mb = int(env("python_memory_mb", "256"))
            self.python_wall_seconds = int(env("python_wall_seconds", "10"))
            self.require_approval = b("require_approval", False)
            self.http_max_bytes = int(env("http_max_bytes", "100000"))
            self.memory_db = str(overrides.get("memory_db", env("memory_db", ".cortex/memory.sqlite")))
            self.database_url = env("database_url", "sqlite+aiosqlite:///./.cortex/cortex.db")
            self.sync_database_url = os.environ.get("CORTEX_SYNC_DATABASE_URL") or None
            self.use_vectors = bool(overrides.get("use_vectors", b("use_vectors", False)))
            self.redis_url = env("redis_url", "redis://localhost:6379")
            self.use_queue = b("use_queue", False)
            self.max_concurrent_runs = int(env("max_concurrent_runs", "16"))
            self.jwt_secret = env("jwt_secret", "dev-insecure-change-me")
            self.jwt_algorithm = env("jwt_algorithm", "HS256")
            self.jwt_expire_minutes = int(env("jwt_expire_minutes", "60"))
            self.auth_required = b("auth_required", False)
            self.cors_origins = env("cors_origins", "http://localhost:8000,http://127.0.0.1:8000")
            self.rate_limit_per_minute = int(env("rate_limit_per_minute", "30"))
            self.max_goal_length = int(env("max_goal_length", "4000"))
            self.host = str(overrides.get("host", env("host", "127.0.0.1")))
            self.port = int(str(overrides.get("port", env("port", "8000"))))
            self.log_level = env("log_level", "INFO")
            self.log_json = b("log_json", True)
            self.enable_metrics = b("enable_metrics", True)
            self.enable_tracing = b("enable_tracing", False)

        @property
        def cors_origin_list(self) -> List[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

        @property
        def fallback_chain(self) -> List[str]:
            return [b.strip() for b in self.fallback_backends.split(",") if b.strip()]

        def resolved_sync_url(self) -> str:
            if self.sync_database_url:
                return self.sync_database_url
            url = self.database_url
            return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


def load_settings(**overrides: object) -> "Settings":
    """Load settings from the environment, applying any keyword overrides."""
    settings = Settings()
    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Return a cached process-wide settings instance."""
    return Settings()
