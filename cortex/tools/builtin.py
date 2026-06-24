"""Built-in tools: real, working, and hardened.

All tools are genuinely functional and security-conscious:

* ``calculator`` — safe arithmetic via AST evaluation (no ``eval``).
* ``current_time`` — current UTC/local time.
* ``read_file`` / ``write_file`` — strictly jailed to a workspace directory;
  ``..`` traversal *and* symlink escapes are rejected (see :mod:`cortex.tools.sandbox`).
* ``run_python`` — executes code in an isolated subprocess with CPU/memory/
  file-size/wall-clock rlimits, no network, a temp working dir, and an import
  allowlist.
* ``http_get`` — HTTP GET via ``httpx`` with SSRF protection (scheme allowlist,
  private/loopback/link-local IP rejection, size + time caps).
* ``web_search`` — pluggable; falls back to a local docs fixture offline.

Use :func:`build_default_registry` to get a ``ToolRegistry`` populated with the
tools, sandboxed to a given workspace.
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .base import Tool, ToolRegistry, ToolResult
from .sandbox import (
    PathJailError,
    SSRFError,
    assert_safe_url,
    jail_path,
    run_python_sandboxed,
)

# --------------------------------------------------------------------------- #
# Calculator: safe arithmetic evaluation via AST
# --------------------------------------------------------------------------- #

_ALLOWED_BINOPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY: Dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Cap exponent magnitude so `2 ** 99999999` can't hang/OOM the parent process.
_MAX_POW_EXP = 1000


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST, rejecting anything unsafe."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric literals are allowed.")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXP:
            raise ValueError("Exponent too large.")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression. Only +-*/ %, ** and parentheses are allowed.")


def calculator(expression: str) -> ToolResult:
    """Safely evaluate a basic arithmetic expression."""
    expr = str(expression).strip().replace("x", "*").replace("X", "*")
    try:
        tree = ast.parse(expr, mode="eval")
        value = _safe_eval(tree)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as exc:
        return ToolResult(f"Could not evaluate {expression!r}: {exc}", is_error=True)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return ToolResult(output=f"{expression} = {value}", data=value)


# --------------------------------------------------------------------------- #
# Current time
# --------------------------------------------------------------------------- #


def current_time(timezone: str = "utc") -> ToolResult:
    """Return the current date and time (UTC by default)."""
    if str(timezone).lower() == "local":
        now = _dt.datetime.now()
        label = "local time"
    else:
        now = _dt.datetime.now(_dt.timezone.utc)
        label = "UTC"
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return ToolResult(output=f"Current {label}: {stamp}", data=stamp)


# --------------------------------------------------------------------------- #
# Jailed file tools
# --------------------------------------------------------------------------- #


def make_read_file(workspace: Path):
    """Build a workspace-jailed ``read_file`` implementation."""

    def read_file(path: str, max_bytes: int = 20000) -> ToolResult:
        try:
            target = jail_path(workspace, path, must_exist=True)
        except (PathJailError, PermissionError) as exc:
            return ToolResult(str(exc), is_error=True)
        except FileNotFoundError as exc:
            return ToolResult(str(exc), is_error=True)
        if not target.is_file():
            return ToolResult(f"Not a file: {path}", is_error=True)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")[: int(max_bytes)]
        except OSError as exc:
            return ToolResult(f"Could not read {path}: {exc}", is_error=True)
        return ToolResult(output=content, data=content)

    return read_file


def make_write_file(workspace: Path):
    """Build a workspace-jailed ``write_file`` implementation."""

    def write_file(path: str, content: str) -> ToolResult:
        try:
            target = jail_path(workspace, path)
        except (PathJailError, PermissionError) as exc:
            return ToolResult(str(exc), is_error=True)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"Could not write {path}: {exc}", is_error=True)
        root = Path(os.path.realpath(str(workspace)))
        rel = target.relative_to(root)
        return ToolResult(output=f"Wrote {len(str(content))} bytes to {rel}", data=str(rel))

    return write_file


# --------------------------------------------------------------------------- #
# Sandboxed Python execution
# --------------------------------------------------------------------------- #


def make_run_python(cpu_seconds: int = 5, memory_mb: int = 256, wall_seconds: int = 10):
    """Build a ``run_python`` bound to specific rlimits."""

    def run_python(code: str, timeout: Optional[int] = None) -> ToolResult:
        return _run_python_impl(
            code,
            timeout=timeout,
            cpu_seconds=cpu_seconds,
            memory_mb=memory_mb,
            wall_seconds=wall_seconds,
        )

    return run_python


def _run_python_impl(
    code: str,
    timeout: Optional[int] = None,
    cpu_seconds: int = 5,
    memory_mb: int = 256,
    wall_seconds: int = 10,
) -> ToolResult:
    """Run ``code`` in the hardened subprocess sandbox and render a ToolResult."""
    wall = int(timeout) if timeout else wall_seconds
    result = run_python_sandboxed(
        code,
        cpu_seconds=cpu_seconds,
        memory_mb=memory_mb,
        wall_seconds=wall,
    )
    if not result.ok:
        detail = result.error or "execution failed"
        body = f"{detail}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}".strip()
        return ToolResult(
            output=body,
            is_error=True,
            data={
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
    rendered = result.stdout if result.stdout else "(no stdout)"
    return ToolResult(
        output=rendered,
        data={"returncode": 0, "stdout": result.stdout, "stderr": result.stderr},
    )


def run_python(code: str, timeout: Optional[int] = None) -> ToolResult:
    """Module-level convenience wrapper around the sandboxed Python runner.

    Uses the default rlimits; suitable for tests and ad-hoc use. The agent
    registry uses :func:`make_run_python` to bind configured limits instead.
    """
    return _run_python_impl(code, timeout=timeout)


# --------------------------------------------------------------------------- #
# HTTP GET with SSRF protection
# --------------------------------------------------------------------------- #


def make_http_get(max_bytes: int = 100_000, timeout: float = 15.0):
    """Build an ``http_get`` with SSRF protection and size/time caps."""

    def http_get(url: str, max_bytes_override: Optional[int] = None) -> ToolResult:
        cap = int(max_bytes_override) if max_bytes_override else max_bytes
        try:
            import httpx
        except ImportError:
            return ToolResult("httpx is not installed; cannot perform http_get.", is_error=True)

        try:
            assert_safe_url(url)
        except SSRFError as exc:
            return ToolResult(f"Blocked by SSRF protection: {exc}", is_error=True)

        try:
            # follow_redirects=False so a redirect can't bounce us to an
            # internal address that bypassed the initial SSRF check.
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                resp = client.get(url)
                body = resp.text[:cap]
        except Exception as exc:  # noqa: BLE001 - report any network failure
            return ToolResult(f"Request failed: {exc}", is_error=True)

        if resp.status_code in (301, 302, 303, 307, 308):
            return ToolResult(
                f"Redirect ({resp.status_code}) not followed for SSRF safety: {resp.headers.get('location', '')}",
                data={"status": resp.status_code},
            )
        return ToolResult(
            output=f"HTTP {resp.status_code} from {url}:\n{body}",
            data={"status": resp.status_code, "body": body},
        )

    return http_get


# --------------------------------------------------------------------------- #
# Web search (pluggable; local-fixture fallback)
# --------------------------------------------------------------------------- #

_LOCAL_DOCS: Dict[str, str] = {
    "cortex-agent": (
        "cortex-agent is an autonomous agentic AI framework with planning, "
        "tool use, memory, and a ReAct/plan-execute loop."
    ),
    "react": (
        "ReAct interleaves Reasoning and Acting: the agent thinks, chooses a "
        "tool, observes the result, and repeats until it can answer."
    ),
    "claude": (
        "Claude is Anthropic's family of large language models, accessible via "
        "the Anthropic Messages API with native tool use."
    ),
    "python": (
        "Python is a high-level, general-purpose programming language widely "
        "used for AI, scripting, and web development."
    ),
}


def make_web_search(docs: Optional[Dict[str, str]] = None):
    """Build a ``web_search`` tool backed by a local docs fixture."""
    corpus = docs or _LOCAL_DOCS

    def web_search(query: str, top_k: int = 3) -> ToolResult:
        q = str(query).lower()
        scored: List[tuple] = []
        for key, text in corpus.items():
            score = sum(1 for word in q.split() if word in key.lower() or word in text.lower())
            if score:
                scored.append((score, key, text))
        scored.sort(reverse=True)
        hits = scored[: int(top_k)]
        if not hits:
            return ToolResult(
                output=f"No local results for {query!r}. (Configure a real search provider for live results.)",
                data=[],
            )
        lines = [f"- {key}: {text}" for _, key, text in hits]
        return ToolResult(
            output="Search results:\n" + "\n".join(lines),
            data=[{"title": k, "snippet": t} for _, k, t in hits],
        )

    return web_search


# --------------------------------------------------------------------------- #
# Registry factory
# --------------------------------------------------------------------------- #


def build_default_registry(
    workspace: Optional[Union[str, os.PathLike]] = None,
    enable_network: bool = True,
    python_cpu_seconds: int = 5,
    python_memory_mb: int = 256,
    python_wall_seconds: int = 10,
    http_max_bytes: int = 100_000,
) -> ToolRegistry:
    """Build a registry of the built-in tools, sandboxed to ``workspace``.

    Args:
        workspace: Directory file tools are confined to.
        enable_network: When False, the network tool (``http_get``) is omitted.
        python_cpu_seconds / python_memory_mb / python_wall_seconds: rlimits
            applied to ``run_python``.
        http_max_bytes: Response size cap for ``http_get``.
    """
    ws = Path(workspace) if workspace else Path(".cortex/workspace")
    ws.mkdir(parents=True, exist_ok=True)

    tools: List[Tool] = [
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression (+, -, *, /, %, **, parentheses).",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression, e.g. '21 * 2 + 3'.",
                }
            },
            required=["expression"],
            func=calculator,
        ),
        Tool(
            name="current_time",
            description="Get the current date and time. Pass timezone='local' for local time.",
            parameters={"timezone": {"type": "string", "description": "'utc' (default) or 'local'."}},
            required=[],
            func=current_time,
        ),
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from the sandboxed workspace.",
            parameters={
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "max_bytes": {"type": "integer", "description": "Maximum number of bytes to read."},
            },
            required=["path"],
            func=make_read_file(ws),
            dangerous=False,
        ),
        Tool(
            name="write_file",
            description="Write a UTF-8 text file into the sandboxed workspace.",
            parameters={
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            required=["path", "content"],
            func=make_write_file(ws),
            dangerous=True,
        ),
        Tool(
            name="run_python",
            description=(
                "Run a short Python 3 snippet in an isolated, resource-limited "
                "subprocess (no network) and return its stdout."
            ),
            parameters={
                "code": {"type": "string", "description": "Python source to execute."},
                "timeout": {"type": "integer", "description": "Wall-clock timeout in seconds."},
            },
            required=["code"],
            func=make_run_python(python_cpu_seconds, python_memory_mb, python_wall_seconds),
            dangerous=True,
        ),
        Tool(
            name="web_search",
            description="Search for information. Falls back to a local knowledge base when offline.",
            parameters={
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "Max number of results."},
            },
            required=["query"],
            func=make_web_search(),
        ),
    ]

    if enable_network:
        tools.append(
            Tool(
                name="http_get",
                description="Perform an HTTP GET request (SSRF-protected) and return the body.",
                parameters={
                    "url": {"type": "string", "description": "The http(s) URL to fetch."},
                    "max_bytes_override": {
                        "type": "integer",
                        "description": "Maximum number of body bytes to return.",
                    },
                },
                required=["url"],
                func=make_http_get(http_max_bytes),
                dangerous=True,
            )
        )

    return ToolRegistry(tools)
