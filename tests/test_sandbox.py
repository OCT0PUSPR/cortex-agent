"""Security tests for the tool sandbox: path jail, run_python, SSRF."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cortex.tools.sandbox import (
    PathJailError,
    SSRFError,
    assert_safe_url,
    jail_path,
    run_python_sandboxed,
)

# --- path jail ------------------------------------------------------------- #


def test_jail_allows_normal_path(tmp_path: Path):
    target = jail_path(tmp_path, "sub/file.txt")
    assert str(target).startswith(str(tmp_path.resolve()))


def test_jail_blocks_parent_traversal(tmp_path: Path):
    with pytest.raises(PathJailError):
        jail_path(tmp_path, "../../etc/passwd")


def test_jail_blocks_absolute_path(tmp_path: Path):
    with pytest.raises(PathJailError):
        jail_path(tmp_path, "/etc/passwd")


def test_jail_blocks_symlink_escape(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    link = ws / "escape"
    os.symlink(str(outside), str(link))
    with pytest.raises(PathJailError):
        jail_path(ws, "escape/secret.txt")


def test_jail_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        jail_path(tmp_path, "nope.txt", must_exist=True)


# --- run_python sandbox ---------------------------------------------------- #


def test_run_python_basic_stdout():
    r = run_python_sandboxed("print(6 * 7)", wall_seconds=5)
    assert r.ok and r.stdout.strip() == "42"


def test_run_python_network_blocked():
    r = run_python_sandboxed("import socket; socket.create_connection(('1.1.1.1', 80))", wall_seconds=5)
    assert not r.ok
    assert "not permitted" in r.stderr or "disabled" in r.stderr or "ImportError" in r.stderr


def test_run_python_import_allowlist_blocks_os():
    r = run_python_sandboxed("import os; print(os.listdir('/'))", wall_seconds=5)
    assert not r.ok and "not permitted" in r.stderr


def test_run_python_allowed_module_works():
    r = run_python_sandboxed("import math; print(math.factorial(6))", wall_seconds=5)
    assert r.ok and r.stdout.strip() == "720"


def test_run_python_wall_timeout_kills_infinite_loop():
    r = run_python_sandboxed("while True:\n    pass", wall_seconds=2)
    assert not r.ok and "timed out" in (r.error or "").lower()


def test_run_python_subprocess_module_blocked():
    r = run_python_sandboxed("import subprocess; subprocess.run(['echo','hi'])", wall_seconds=5)
    assert not r.ok and "not permitted" in r.stderr


# --- SSRF guard ------------------------------------------------------------ #


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
    ],
)
def test_ssrf_blocks_internal(url: str):
    with pytest.raises(SSRFError):
        assert_safe_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/", "gopher://x/"])
def test_ssrf_blocks_bad_scheme(url: str):
    with pytest.raises(SSRFError):
        assert_safe_url(url)


def test_ssrf_blocks_no_host():
    with pytest.raises(SSRFError):
        assert_safe_url("http:///nopath")
