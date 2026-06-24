"""API dependencies: authentication, rate limiting, and the run service.

Authentication accepts either a Bearer JWT (``Authorization: Bearer <jwt>``) or
an API key (``X-API-Key: ck_...``). When ``auth_required`` is False (the default
for local/dev and tests) requests are allowed through as an anonymous principal,
but a presented credential is still validated and attached.

The rate limiter is an in-process sliding-window counter keyed by principal
(user id or API key id, falling back to client IP). For multi-process
deployments swap in a Redis-backed limiter; the interface is identical.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from ..config import Settings, get_settings
from ..db.engine import get_session
from ..db.repository import UserRepository
from ..security import JWTError, decode_access_token
from ..service import RunService


@dataclass
class Principal:
    """The authenticated (or anonymous) caller of a request."""

    id: str
    kind: str  # "user" | "apikey" | "anonymous"
    is_admin: bool = False
    allowed_tools: Optional[List[str]] = None
    rate_limit: int = 30
    user_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Singleton run service
# --------------------------------------------------------------------------- #

_run_service: Optional[RunService] = None


def get_run_service() -> RunService:
    """Return the process-wide run service."""
    global _run_service
    if _run_service is None:
        _run_service = RunService(get_settings())
    return _run_service


def reset_run_service() -> None:
    """Reset the cached run service (used by tests)."""
    global _run_service
    _run_service = None


# --------------------------------------------------------------------------- #
# Rate limiting (in-process sliding window)
# --------------------------------------------------------------------------- #


@dataclass
class RateLimiter:
    """Sliding-window rate limiter keyed by principal."""

    window_seconds: float = 60.0
    _hits: Dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def check(self, key: str, limit: int) -> bool:
        """Record a hit; return False if ``limit`` is exceeded in the window."""
        now = time.monotonic()
        bucket = self._hits[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True

    def reset(self) -> None:
        self._hits.clear()


_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


# --------------------------------------------------------------------------- #
# Authentication dependency
# --------------------------------------------------------------------------- #


async def get_principal(
    request: Request,
    settings: Settings = Depends(get_settings),
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    session=Depends(get_session),
) -> Principal:
    """Resolve the request's principal from a JWT or API key.

    Order: API key (``X-API-Key``) → Bearer JWT. When ``auth_required`` is False
    and no credential is presented, an anonymous principal is returned.
    """
    # 1. API key.
    if x_api_key:
        repo = UserRepository(session)
        api_key = await repo.get_api_key(x_api_key)
        if api_key is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
        await repo.touch_api_key(api_key)
        return Principal(
            id=api_key.id,
            kind="apikey",
            allowed_tools=api_key.allowed_tools,
            rate_limit=api_key.rate_limit_per_minute,
            user_id=api_key.user_id,
        )

    # 2. Bearer JWT.
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            claims = decode_access_token(token, settings.jwt_secret, settings.jwt_algorithm)
        except JWTError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc
        return Principal(
            id=claims["sub"],
            kind="user",
            is_admin=bool(claims.get("is_admin", False)),
            rate_limit=settings.rate_limit_per_minute,
            user_id=claims["sub"],
        )

    # 3. No credential.
    if settings.auth_required:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return Principal(
        id=f"anon:{request.client.host if request.client else 'unknown'}",
        kind="anonymous",
        rate_limit=settings.rate_limit_per_minute,
    )


async def rate_limited(
    principal: Principal = Depends(get_principal),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> Principal:
    """Dependency that enforces the per-principal rate limit."""
    if not limiter.check(principal.id, principal.rate_limit):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({principal.rate_limit}/min). Try again shortly.",
            headers={"Retry-After": "60"},
        )
    return principal
