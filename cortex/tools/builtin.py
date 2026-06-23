"""Built-in tools: real, working, and safe.

All tools are genuinely functional:

* ``calculator`` — safe arithmetic via AST evaluation (no ``eval``).
* ``current_time`` — current UTC/local time.
* ``read_file`` / ``write_file`` — sandboxed to a workspace directory; path
  traversal outside the workspace is rejected.
* ``run_python`` — executes code in a separate ``python`` subprocess inside a
  temporary sandbox directory, with a timeout.
* ``http_get`` — HTTP GET via ``httpx`` (guarded import).
* ``web_search`` — pluggable; falls back to a local docs fixture when no
  search key/provider is configured.

Use :func:`build_default_registry` to get a ``ToolRegistry`` populated with the
tools, sandboxed to a given workspace.
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolRegistry, ToolResult

# --------------------------------------------------------------------------- #
# Calculator: safe arithmetic evaluation via AST
# --------------------------------------------------------------------------- #

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST, rejecting anything unsafe."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric literals are allowed.")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression. Only +-*/ %, ** and parentheses are allowed.")


def calculator(expression: str) -> ToolResult:
    """Safely evaluate a basic arithmetic expression."""
    expr = str(expression).strip().replace("x", "*").replace("X", "*")
    try:
        tree = ast.parse(expr, mode="eval")
        value = _safe_eval(tree)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return ToolResult(f"Could not evaluate {expression!r}: {exc}", is_error=True)
    # Render integers without a trailing .0 for readability.
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
# Sandboxed file tools
# --------------------------------------------------------------------------- #


def _resolve_in_workspace(workspace: Path, path: str) -> Path:
    """Resolve ``path`` against the workspace, rejecting escapes."""
    workspace = workspace.resolve()
    candidate = (workspace / path).resolve()
    # Python 3.9-compatible containment check.
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise PermissionError(
            f"Path {path!r} escapes the workspace sandbox."
        ) from exc
    return candidate


def make_read_file(workspace: Path):
    """Build a workspace-sandboxed ``read_file`` implementation."""

    def read_file(path: str, max_bytes: int = 20000) -> ToolResult:
        try:
            target = _resolve_in_workspace(workspace, path)
        except PermissionError as exc:
            return ToolResult(str(exc), is_error=True)
        if not target.exists():
            return ToolResult(f"File not found: {path}", is_error=True)
        if not target.is_file():
            return ToolResult(f"Not a file: {path}", is_error=True)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")[: int(max_bytes)]
        except OSError as exc:
            return ToolResult(f"Could not read {path}: {exc}", is_error=True)
        return ToolResult(output=content, data=content)

    return read_file


def make_write_file(workspace: Path):
    """Build a workspace-sandboxed ``write_file`` implementation."""

    def write_file(path: str, content: str) -> ToolResult:
        try:
            target = _resolve_in_workspace(workspace, path)
        except PermissionError as exc:
            return ToolResult(str(exc), is_error=True)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"Could not write {path}: {exc}", is_error=True)
        rel = target.relative_to(workspace.resolve())
        return ToolResult(
            output=f"Wrote {len(str(content))} bytes to {rel}",
            data=str(rel),
        )

    return write_file


# --------------------------------------------------------------------------- #
# Sandboxed Python execution (subprocess + timeout)
# --------------------------------------------------------------------------- #


def run_python(code: str, timeout: int = 10) -> ToolResult:
    """Execute Python ``code`` in an isolated subprocess + temp sandbox.

    The code runs with the temp directory as its working directory and a
    wall-clock timeout. stdout/stderr are captured and returned.
    """
    with tempfile.TemporaryDirectory(prefix="cortex_py_") as sandbox:
        script = Path(sandbox) / "snippet.py"
        try:
            script.write_text(str(code), encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"Could not stage code: {exc}", is_error=True)

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Execution timed out after {timeout}s.", is_error=True
            )
        except OSError as exc:
            return ToolResult(f"Could not run Python: {exc}", is_error=True)

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return ToolResult(
                output=f"Exited with code {proc.returncode}.\nstdout:\n{out}\nstderr:\n{err}",
                is_error=True,
                data={"returncode": proc.returncode, "stdout": out, "stderr": err},
            )
        rendered = out if out else "(no stdout)"
        return ToolResult(
            output=rendered,
            data={"returncode": 0, "stdout": out, "stderr": err},
        )


# --------------------------------------------------------------------------- #
# HTTP GET
# --------------------------------------------------------------------------- #


def http_get(url: str, max_bytes: int = 5000) -> ToolResult:
    """Fetch a URL with an HTTP GET and return the (truncated) body."""
    try:
        import httpx
    except ImportError:
        return ToolResult(
            "httpx is not installed; cannot perform http_get.", is_error=True
        )
    if not str(url).startswith(("http://", "https://")):
        return ToolResult("URL must start with http:// or https://", is_error=True)
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            body = resp.text[: int(max_bytes)]
    except Exception as exc:  # noqa: BLE001 - report any network failure
        return ToolResult(f"Request failed: {exc}", is_error=True)
    return ToolResult(
        output=f"HTTP {resp.status_code} from {url}:\n{body}",
        data={"status": resp.status_code, "body": body},
    )


# --------------------------------------------------------------------------- #
# Web search (pluggable; local-fixture fallback)
# --------------------------------------------------------------------------- #

# A tiny local "knowledge base" used when no real search provider is wired up.
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
    """Build a ``web_search`` tool backed by a local docs fixture.

    In production you would swap the implementation for a real provider. When
    no provider is configured, this performs a keyword match against the local
    fixture so the agent loop still produces useful observations offline.
    """
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
                output=f"No local results for {query!r}. "
                "(Configure a real search provider for live results.)",
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
    workspace: Optional[os.PathLike] = None,
    enable_network: bool = True,
) -> ToolRegistry:
    """Build a registry of the built-in tools, sandboxed to ``workspace``.

    Args:
        workspace: Directory file tools are confined to. Defaults to
            ``./.cortex/workspace`` and is created if missing.
        enable_network: When False, the network tools (``http_get``) are
            omitted (useful for offline/CI contexts).
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
            parameters={
                "timezone": {
                    "type": "string",
                    "description": "'utc' (default) or 'local'.",
                }
            },
            required=[],
            func=current_time,
        ),
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from the sandboxed workspace.",
            parameters={
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum number of bytes to read.",
                },
            },
            required=["path"],
            func=make_read_file(ws),
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
        ),
        Tool(
            name="run_python",
            description="Run a short Python 3 snippet in an isolated subprocess and return its stdout.",
            parameters={
                "code": {"type": "string", "description": "Python source to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Wall-clock timeout in seconds.",
                },
            },
            required=["code"],
            func=run_python,
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
                description="Perform an HTTP GET request and return the response body.",
                parameters={
                    "url": {"type": "string", "description": "The http(s) URL to fetch."},
                    "max_bytes": {
                        "type": "integer",
                        "description": "Maximum number of body bytes to return.",
                    },
                },
                required=["url"],
                func=http_get,
            )
        )

    return ToolRegistry(tools)
