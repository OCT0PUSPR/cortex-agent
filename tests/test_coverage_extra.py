"""Extra targeted tests to exercise fallback paths and edge cases."""

from __future__ import annotations

import importlib

from cortex.agent.planner import Planner, _heuristic_plan, _parse_steps
from cortex.llm import MockLLM
from cortex.tools.builtin import (
    build_default_registry,
    calculator,
    current_time,
    make_http_get,
    make_read_file,
    make_web_search,
)

# --- planner --------------------------------------------------------------- #


def test_heuristic_plan_multistep():
    steps = _heuristic_plan("Read the file, then write a summary, and email it")
    assert len(steps) >= 2


def test_heuristic_plan_single():
    steps = _heuristic_plan("Do something")
    assert len(steps) >= 1


def test_parse_steps_numbered():
    text = "1. First\n2) Second\n- Third\n* Fourth"
    steps = _parse_steps(text)
    assert steps == ["First", "Second", "Third", "Fourth"]


def test_planner_mock_uses_heuristic():
    plan = Planner(MockLLM()).plan("Calculate 2+2 and tell the time")
    assert plan.steps


# --- builtin tool edge cases ----------------------------------------------- #


def test_calculator_exponent_cap():
    assert calculator("2 ** 99999").is_error


def test_calculator_float_result():
    r = calculator("10 / 4")
    assert not r.is_error and r.data == 2.5


def test_current_time_local():
    r = current_time("local")
    assert not r.is_error and "local" in r.output


def test_read_file_not_a_file(tmp_path):
    (tmp_path / "adir").mkdir()
    read = make_read_file(tmp_path)
    assert read("adir").is_error


def test_web_search_no_results():
    search = make_web_search({"x": "y"})
    r = search("zzzznotfound")
    assert "No local results" in r.output


def test_http_get_rejects_bad_scheme():
    get = make_http_get()
    r = get("ftp://example.com")
    assert r.is_error


def test_registry_includes_network_when_enabled(tmp_path):
    reg = build_default_registry(workspace=tmp_path / "w", enable_network=True)
    assert reg.has("http_get")


# --- security PBKDF2 fallback (force jose/bcrypt-absent paths) -------------- #


def test_pbkdf2_password_fallback(monkeypatch):
    import cortex.security as sec

    importlib.reload(sec)
    # Force the PBKDF2 branch by pretending bcrypt is unavailable.
    monkeypatch.setattr(sec, "_HAS_BCRYPT", False)
    h = sec.hash_password("secret123")
    assert h.startswith("pbkdf2$")
    assert sec.verify_password("secret123", h)
    assert not sec.verify_password("wrong", h)
    importlib.reload(sec)


def test_jwt_fallback_without_jose(monkeypatch):
    import cortex.security as sec

    importlib.reload(sec)
    monkeypatch.setattr(sec, "_HAS_JOSE", False)
    token = sec.create_access_token("u", "secret", expires_minutes=5)
    claims = sec.decode_access_token(token, "secret")
    assert claims["sub"] == "u"
    try:
        sec.decode_access_token(token, "wrong")
        raise AssertionError("expected JWTError")
    except sec.JWTError:
        pass
    importlib.reload(sec)
