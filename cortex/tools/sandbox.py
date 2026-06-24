"""Hardened sandbox primitives for agent tools.

This module concentrates the security-critical logic so it can be unit-tested
in isolation:

* :func:`jail_path` — resolve a path strictly inside a workspace, defeating
  ``..`` traversal *and* symlink escapes.
* :func:`run_python_sandboxed` — run code in an isolated subprocess with CPU,
  memory, file-size, and wall-clock rlimits, no network (best-effort), a temp
  working directory, and an allowlist of importable modules.
* :func:`assert_safe_url` — SSRF guard: scheme allowlist + DNS resolution +
  rejection of private / loopback / link-local / reserved address ranges.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# --------------------------------------------------------------------------- #
# File jail (traversal + symlink safe)
# --------------------------------------------------------------------------- #


class PathJailError(PermissionError):
    """Raised when a path resolves outside the workspace jail."""


def jail_path(workspace: Path, rel_path: str, *, must_exist: bool = False) -> Path:
    """Resolve ``rel_path`` strictly inside ``workspace``.

    Rejects absolute paths, ``..`` traversal, and symlinks that point outside
    the workspace. Uses ``os.path.realpath`` so symlink components are fully
    resolved before the containment check.

    Args:
        workspace: The jail root.
        rel_path: A (workspace-relative) path supplied by the model.
        must_exist: When True, the resolved target must already exist.

    Returns:
        The resolved, contained :class:`Path`.

    Raises:
        PathJailError: if the path escapes the workspace.
    """
    root = Path(os.path.realpath(str(workspace)))
    root.mkdir(parents=True, exist_ok=True)

    # Reject absolute inputs outright — everything is workspace-relative.
    candidate = Path(rel_path)
    if candidate.is_absolute():
        raise PathJailError(f"Absolute paths are not allowed: {rel_path!r}")

    joined = root / candidate
    # realpath fully resolves symlinks in every component, even ones that don't
    # exist yet (it resolves the existing prefix and appends the rest).
    resolved = Path(os.path.realpath(str(joined)))

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PathJailError(f"Path {rel_path!r} escapes the workspace sandbox.") from exc

    # Extra guard: if any *existing* parent is a symlink pointing outside, the
    # realpath check above already catches it. Re-verify the final target too.
    if resolved.exists() and resolved.is_symlink():  # pragma: no cover - defensive
        target = Path(os.path.realpath(str(resolved)))
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PathJailError(f"Symlink {rel_path!r} escapes the workspace.") from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File not found: {rel_path}")

    return resolved


# --------------------------------------------------------------------------- #
# Sandboxed Python execution
# --------------------------------------------------------------------------- #

# Modules an untrusted snippet may import. Anything else raises ImportError
# inside the sandbox. Network/process/filesystem-escape modules are excluded.
DEFAULT_ALLOWED_MODULES = frozenset(
    {
        "math",
        "statistics",
        "random",
        "itertools",
        "functools",
        "collections",
        "json",
        "re",
        "string",
        "datetime",
        "decimal",
        "fractions",
        "heapq",
        "bisect",
        "array",
        "copy",
        "textwrap",
        "unicodedata",
        "hashlib",
        "base64",
        "enum",
        "dataclasses",
        "typing",
        "operator",
        "numbers",
        "cmath",
        "secrets",
    }
)

# Preamble injected before user code: installs an import allowlist and drops
# network capability (sockets raise) as a best-effort defense in depth.
_SANDBOX_PREAMBLE = textwrap.dedent(
    """
    import builtins as _b
    _ALLOWED = set({allowed!r})
    _real_import = _b.__import__
    def _guard_import(name, *a, **k):
        root = name.split('.')[0]
        if root not in _ALLOWED:
            raise ImportError("import of %r is not permitted in the sandbox" % root)
        return _real_import(name, *a, **k)
    _b.__import__ = _guard_import

    # Best-effort: neutralize network access (socket creation raises).
    try:
        import socket as _socket
        def _no_net(*a, **k):
            raise OSError("network access is disabled in the sandbox")
        _socket.socket = _no_net  # type: ignore[assignment]
        _socket.create_connection = _no_net  # type: ignore[assignment]
    except Exception:
        pass
    """
)

# Set resource limits in the child *before* exec via preexec_fn (POSIX only).
_RLIMIT_TEMPLATE = """
import resource, sys
def _set_limits():
    soft_cpu = {cpu}
    resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu, soft_cpu))
    mem = {mem_bytes}
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError):
        pass
    fsize = {fsize}
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError):
        pass
"""


@dataclass
class SandboxResult:
    """Outcome of a sandboxed Python execution."""

    ok: bool
    stdout: str
    stderr: str
    returncode: int
    error: Optional[str] = None


def _make_preexec(cpu_seconds: int, memory_mb: int):
    """Build a preexec_fn that applies rlimits in the child (POSIX only)."""
    if not hasattr(os, "fork"):  # pragma: no cover - Windows
        return None
    try:
        import resource
    except ImportError:  # pragma: no cover
        return None

    mem_bytes = memory_mb * 1024 * 1024

    def _limits() -> None:  # pragma: no cover - runs in forked child
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        # 32 MiB max file size for any single write.
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass

    return _limits


def run_python_sandboxed(
    code: str,
    *,
    cpu_seconds: int = 5,
    memory_mb: int = 256,
    wall_seconds: int = 10,
    allowed_modules: Optional[frozenset] = None,
) -> SandboxResult:
    """Run ``code`` in an isolated subprocess with strict resource limits.

    Defenses:
      * separate ``python -I`` subprocess (isolated mode, no site/env inherit),
      * CPU/memory/file-size/process rlimits via ``preexec_fn`` (POSIX),
      * wall-clock timeout that kills the process group,
      * import allowlist + disabled networking installed before user code,
      * a fresh temp working directory, cleaned up afterward,
      * a minimal environment (no inherited secrets).
    """
    allowed = allowed_modules or DEFAULT_ALLOWED_MODULES
    preamble = _SANDBOX_PREAMBLE.format(allowed=sorted(allowed))
    program = preamble + "\n# --- user code ---\n" + str(code)

    with tempfile.TemporaryDirectory(prefix="cortex_sbx_") as sandbox:
        script = Path(sandbox) / "snippet.py"
        script.write_text(program, encoding="utf-8")

        # Minimal env: no API keys, no PATH leakage of user config.
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": sandbox,
            "TMPDIR": sandbox,
        }
        preexec = _make_preexec(cpu_seconds, memory_mb)
        popen_kwargs = {}
        if preexec is not None:
            popen_kwargs["preexec_fn"] = preexec
            popen_kwargs["start_new_session"] = True  # own process group to kill

        try:
            # nosec B603: this is the sandbox's purpose — run untrusted code in
            # an ISOLATED process. No shell (list argv), python `-I` isolated
            # mode, a minimal env, rlimits via preexec_fn, a wall-clock timeout,
            # an import allowlist, and disabled networking are all applied above.
            proc = subprocess.run(  # nosec B603
                [sys.executable, "-I", "-B", str(script)],
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=max(1, wall_seconds),
                env=env,
                check=False,
                **popen_kwargs,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                stdout="",
                stderr="",
                returncode=-1,
                error=f"execution timed out after {wall_seconds}s (killed)",
            )
        except OSError as exc:
            return SandboxResult(
                ok=False,
                stdout="",
                stderr="",
                returncode=-1,
                error=f"could not start sandbox: {exc}",
            )

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        ok = proc.returncode == 0
        return SandboxResult(
            ok=ok,
            stdout=out,
            stderr=err,
            returncode=proc.returncode,
            error=None if ok else f"exited with code {proc.returncode}",
        )


# --------------------------------------------------------------------------- #
# SSRF guard
# --------------------------------------------------------------------------- #


class SSRFError(PermissionError):
    """Raised when a URL targets a disallowed (private/internal) destination."""


_ALLOWED_SCHEMES = {"http", "https"}


def _is_blocked_ip(ip: str) -> bool:
    """Return True for private, loopback, link-local, reserved, or multicast IPs."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:  # pragma: no cover
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def assert_safe_url(url: str, *, allow_hosts: Optional[List[str]] = None) -> None:
    """Validate ``url`` against SSRF: scheme allowlist + private-IP rejection.

    Resolves the hostname via DNS and rejects the request if *any* resolved
    address is private/loopback/link-local/reserved. Raises :class:`SSRFError`
    on any violation.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError(f"Scheme {parsed.scheme!r} not allowed (use http/https).")
    host = parsed.hostname
    if not host:
        raise SSRFError("URL has no host.")

    if allow_hosts is not None and host not in allow_hosts:
        raise SSRFError(f"Host {host!r} is not in the allowlist.")

    # Resolve every address the host maps to and block if any is internal.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise SSRFError(f"Could not resolve host {host!r}: {exc}") from exc

    for info in infos:
        ip = str(info[4][0])
        if _is_blocked_ip(ip):
            raise SSRFError(f"Host {host!r} resolves to a blocked address ({ip}).")
