"""Security primitives: JWT, API keys, password hashing, and secret redaction.

These helpers are dependency-light and degrade gracefully:

* JWT uses ``python-jose`` when present, else a small HMAC-SHA256 fallback so
  tokens still work in minimal environments and tests.
* Password hashing uses ``passlib[bcrypt]`` when present, else PBKDF2-HMAC from
  the standard library.
* API keys are generated with ``secrets`` and stored only as SHA-256 hashes.
* :func:`redact_secrets` scrubs common secret patterns from any text before it
  is logged or returned to a client.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #

API_KEY_PREFIX = "ck_"


def generate_api_key() -> str:
    """Generate a new opaque API key (shown to the user once)."""
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest used to store an API key at rest."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Constant-time comparison of a presented key against its stored hash."""
    return hmac.compare_digest(hash_api_key(api_key), stored_hash)


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #

# We use the `bcrypt` library directly (not via passlib) to avoid passlib's
# brittle bcrypt-version detection, and fall back to stdlib PBKDF2-HMAC-SHA256
# when bcrypt is unavailable. bcrypt caps inputs at 72 bytes, so we pre-hash the
# password with SHA-256 → base64 to support arbitrarily long passwords safely.
try:
    import bcrypt as _bcrypt

    _HAS_BCRYPT = True
except ImportError:  # pragma: no cover
    _HAS_BCRYPT = False

_PBKDF2_ROUNDS = 200_000


def _bcrypt_input(password: str) -> bytes:
    """Pre-hash to a fixed-length token so bcrypt's 72-byte cap never truncates."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    """Hash a password (bcrypt when available, else PBKDF2-HMAC-SHA256)."""
    if _HAS_BCRYPT:
        return _bcrypt.hashpw(_bcrypt_input(password), _bcrypt.gensalt()).decode("ascii")
    salt = secrets.token_bytes(16)  # pragma: no cover
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return "pbkdf2$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against its stored hash."""
    if stored.startswith("pbkdf2$"):  # pragma: no cover
        try:
            _, salt_b64, dk_b64 = stored.split("$", 2)
            salt = base64.b64decode(salt_b64)
            expected = base64.b64decode(dk_b64)
        except (ValueError, TypeError):
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
        return hmac.compare_digest(dk, expected)
    if _HAS_BCRYPT:
        try:
            return _bcrypt.checkpw(_bcrypt_input(password), stored.encode("ascii"))
        except (ValueError, TypeError):
            return False
    return False  # pragma: no cover


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #

try:
    from jose import JWTError, jwt

    _HAS_JOSE = True
except ImportError:  # pragma: no cover
    _HAS_JOSE = False

    class JWTError(Exception):  # type: ignore[no-redef]
        """Fallback JWT error type."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def create_access_token(
    subject: str,
    secret: str,
    algorithm: str = "HS256",
    expires_minutes: int = 60,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a signed JWT access token."""
    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_minutes * 60,
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)

    if _HAS_JOSE:
        return jwt.encode(payload, secret, algorithm=algorithm)

    # Minimal HS256 fallback. # pragma: no cover
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + _b64url(sig)


def decode_access_token(token: str, secret: str, algorithm: str = "HS256") -> Dict[str, Any]:
    """Decode and verify a JWT; raise :class:`JWTError` on failure."""
    if _HAS_JOSE:
        return jwt.decode(token, secret, algorithms=[algorithm])

    # Minimal HS256 fallback. # pragma: no cover
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise JWTError("Malformed token") from exc
    signing_input = header_b64 + "." + payload_b64
    expected = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
        raise JWTError("Bad signature")
    payload = json.loads(_b64url_decode(payload_b64))
    if payload.get("exp", 0) < int(time.time()):
        raise JWTError("Token expired")
    return payload


# --------------------------------------------------------------------------- #
# Secret redaction (defense against leaking creds in logs / tool output)
# --------------------------------------------------------------------------- #

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic keys
    re.compile(r"hf_[A-Za-z0-9]{8,}"),  # HuggingFace tokens
    re.compile(r"ck_[A-Za-z0-9_\-]{8,}"),  # cortex API keys
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),  # GitHub PAT
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),  # JWTs
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"),
]

_REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace anything that looks like a secret with ``[REDACTED]``."""
    if not text:
        return text
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(_REDACTED, out)
    return out


def redact_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact secret-looking values in a flat mapping (e.g. tool arguments)."""
    sensitive = {"password", "secret", "token", "api_key", "apikey", "authorization"}
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in sensitive:
            out[key] = _REDACTED
        elif isinstance(value, str):
            out[key] = redact_secrets(value)
        else:
            out[key] = value
    return out
