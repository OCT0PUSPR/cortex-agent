"""ASGI middleware: request-id propagation and security headers.

* :class:`RequestContextMiddleware` assigns a request id (honoring an inbound
  ``X-Request-ID``), binds it to the structured-logging context, logs the
  request/response, and echoes the id back in the response.
* :class:`SecurityHeadersMiddleware` adds a conservative set of security headers
  to every response.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..observability import METRICS, bind_context, get_logger
from ..service import new_request_id

_log = get_logger("cortex.api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, bind logging context, and log each request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or new_request_id()
        request.state.request_id = request_id
        start = time.monotonic()
        with bind_context(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                METRICS.observe_error("api")
                _log.exception("request_error", path=request.url.path, method=request.method)
                raise
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            _log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
            )
            response.headers["X-Request-ID"] = request_id
            return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("X-XSS-Protection", "1; mode=block")
        headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "frame-ancestors 'none'",
        )
        headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        # HSTS only meaningful over HTTPS; harmless on localhost.
        headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response
