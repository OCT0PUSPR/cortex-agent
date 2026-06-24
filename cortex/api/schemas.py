"""Pydantic request/response schemas with strict validation.

Goal length, backend choice, and step counts are validated at the edge so
malformed or abusive requests are rejected before reaching the agent runtime.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from ..config import get_settings


class RunRequest(BaseModel):
    """Body for ``POST /v1/runs``."""

    goal: str = Field(..., min_length=1, description="The task for the agent.")
    session_id: Optional[str] = Field(default=None, description="Existing session to append to.")
    backend: Optional[str] = Field(default=None, description="mock | anthropic | hf")
    model: Optional[str] = Field(default=None, description="Model id override.")

    @field_validator("goal")
    @classmethod
    def _goal_len(cls, v: str) -> str:
        max_len = get_settings().max_goal_length
        if len(v) > max_len:
            raise ValueError(f"goal exceeds the maximum length of {max_len} characters")
        return v

    @field_validator("backend")
    @classmethod
    def _backend_choice(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"mock", "anthropic", "hf"}:
            raise ValueError("backend must be one of: mock, anthropic, hf")
        return v


class RunResponse(BaseModel):
    """A run summary."""

    id: str
    session_id: str
    goal: str
    status: str
    answer: Optional[str] = None
    plan: Optional[List[str]] = None
    steps_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ApiKeyRequest(BaseModel):
    name: str = Field(default="default", max_length=128)
    allowed_tools: Optional[List[str]] = Field(default=None, description="Per-key tool allowlist.")
    rate_limit_per_minute: int = Field(default=30, ge=1, le=10000)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    api_key: str = Field(..., description="Shown once — store it securely.")
    allowed_tools: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "cortex-agent"
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    database: bool
    checks: dict


class ToolInfo(BaseModel):
    name: str
    description: str
    dangerous: bool
    schema_: dict = Field(alias="schema")

    model_config = {"populate_by_name": True}


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
