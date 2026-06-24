"""Tests for security primitives: API keys, passwords, JWT, redaction."""

from __future__ import annotations

import pytest

from cortex.security import (
    JWTError,
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    redact_mapping,
    redact_secrets,
    verify_api_key,
    verify_password,
)


def test_api_key_roundtrip():
    key = generate_api_key()
    assert key.startswith("ck_")
    h = hash_api_key(key)
    assert verify_api_key(key, h)
    assert not verify_api_key("ck_wrong", h)


def test_api_key_hash_is_not_reversible():
    key = generate_api_key()
    assert key not in hash_api_key(key)


def test_password_hash_and_verify():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong", h)


def test_password_long_input():
    pw = "x" * 300  # exceeds bcrypt's 72-byte limit; pre-hash must handle it
    h = hash_password(pw)
    assert verify_password(pw, h)


def test_jwt_roundtrip_and_claims():
    token = create_access_token("user-1", "secret", expires_minutes=5, extra_claims={"is_admin": True})
    claims = decode_access_token(token, "secret")
    assert claims["sub"] == "user-1" and claims["is_admin"] is True


def test_jwt_rejects_bad_signature():
    token = create_access_token("u", "secret")
    with pytest.raises(JWTError):
        decode_access_token(token, "other-secret")


def test_redact_secrets():
    text = "key sk-ant-abcdefgh12345678 and hf_xxxxxxxxyyyyyyyy and ck_aaaabbbbccccdddd"
    red = redact_secrets(text)
    assert "sk-ant" not in red
    assert "hf_xxxx" not in red
    assert "REDACTED" in red


def test_redact_mapping():
    out = redact_mapping({"password": "hunter2", "note": "token=ck_aaaabbbbcccc", "n": 5})
    assert out["password"] == "[REDACTED]"
    assert "REDACTED" in out["note"]
    assert out["n"] == 5
