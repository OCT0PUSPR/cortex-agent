# Contributing to cortex-agent

Thanks for contributing! `cortex-agent` is a production-grade autonomous agent
framework. This guide gets you set up, explains the quality gates, and shows how
to extend the three most common extension points: **tools**, **LLM backends**,
and **database migrations**.

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) first for the system overview and
[`SECURITY.md`](./SECURITY.md) when touching the sandbox, auth, or tools.

---

## 1. Development setup

The framework core has **no ML dependencies** — `torch` is import-guarded, so you
can develop and run the full test suite offline with the MockLLM. Use the minimal
dev requirements unless you're working on TinyBrain.

```bash
# 1. Create and activate a virtualenv (Python 3.9+; CI runs 3.9, 3.11, 3.12).
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the minimal dev/CI deps (lint + type + full test suite, no torch).
pip install -r requirements-min.txt

# 3. Install the package in editable mode.
pip install -e .

# 4. (optional) Install pre-commit hooks.
pip install pre-commit && pre-commit install
```

Other requirement sets:

| File | Purpose |
|---|---|
| `requirements.txt` | Full runtime stack (FastAPI, SQLAlchemy, anthropic SDK, …). |
| `requirements-min.txt` | Minimal CI deps: run the **full** suite + lint/type/security offline. |
| `requirements-dev.txt` | Runtime + QA tooling (pulls in `requirements.txt`). |
| `requirements-train.txt` | TinyBrain training extras (`torch`, `numpy`, `tokenizers`). |

Quick smoke test (offline, deterministic):

```python
from cortex import build_agent
agent = build_agent(backend="mock")
print(agent.run("Calculate 21 * 2 and tell me the current time").answer)
```

---

## 2. Makefile targets

`make` (or `make help`) lists everything. The common ones:

| Target | What it does |
|---|---|
| `make install-dev` | `pip install -r requirements-min.txt` + `pip install -e .` |
| `make install-train` | Install TinyBrain training deps. |
| `make test` | `pytest` with coverage (`--cov=cortex --cov-report=term-missing`). |
| `make lint` | `ruff check` + `ruff format --check` (no changes). |
| `make format` | `ruff format` (auto-format). |
| `make typecheck` | `mypy cortex --ignore-missing-imports`. |
| `make security` | `bandit -r cortex -c pyproject.toml` + `pip-audit`. |
| `make migrate` | `alembic upgrade head`. |
| `make run` | Run the API with autoreload (`uvicorn cortex.api.server:app --reload`). |
| `make worker` | Run the arq queue worker. |
| `make train` | Train the from-scratch TinyBrain model. |
| `make docker-build` / `make docker-up` / `make docker-down` | Docker image / full stack / teardown. |
| `make all` | `lint` + `typecheck` + `test`. |
| `make clean` | Remove caches and local SQLite/state. |

---

## 3. Running tests, lint, typecheck & security locally

Run these before every PR (this mirrors CI):

```bash
make lint         # ruff check + ruff format --check
make typecheck    # mypy cortex
make security     # bandit + pip-audit
make test         # pytest with coverage
```

Notes:

- **Tests are offline and deterministic** — they use the MockLLM backend and need
  no `torch`, Redis, or API keys. The torch-gated TinyBrain tests skip cleanly
  when torch is absent.
- **Coverage gate:** CI runs `pytest --cov=cortex --cov-fail-under=80`. Keep the
  core ≥ 80%. New code should ship with tests in `tests/`.
- **Compile check:** CI runs `python -m py_compile` over all tracked `*.py` —
  make sure everything parses on 3.9.
- Run a single test file with `pytest tests/test_tools.py -q`.

---

## 4. Code style

- **Formatter / linter:** `ruff` (`ruff.toml` + `[tool.ruff]` in
  `pyproject.toml`). Line length **120**; rule sets `E, F, W, I`.
- **Python 3.9 compatibility is required.** Use `from __future__ import
  annotations` and **`typing.Optional` / `typing.List` / `typing.Dict`** — **not**
  `X | None`, `list[...]`, or `dict[...]` in annotations that are evaluated, and
  do not adopt pyupgrade (`UP`) rewrites. The lint config deliberately excludes
  `UP` rules to preserve 3.9 typing.
- **`torch` must stay import-guarded.** Never import `torch` (or other heavy ML
  deps) at framework module import time — guard it inside `cortex/tinybrain/` or
  behind a lazy import, so the no-ML CI path keeps working.
- **Type hints + docstrings** on public functions/classes (module docstrings
  explain intent). `mypy` runs with `python_version = "3.9"`.
- **Security mindset:** treat LLM/tool/web output as untrusted; redact secrets;
  never log credentials; keep dangerous capabilities behind the policy/sandbox.

---

## 5. Adding a new tool

Tools are `Tool` objects (`cortex/tools/base.py`) registered in a
`ToolRegistry`. A tool's `func` may return a `ToolResult`, a string, or any
value (coerced). **Mark state-mutating or network tools `dangerous=True`** so the
policy/approval gate can govern them.

1. Implement the function. Validate inputs and return a `ToolResult`; never raise
   into the loop (the registry catches, but explicit errors are clearer). If it
   touches the filesystem or network, **use the sandbox primitives** in
   `cortex/tools/sandbox.py` (`jail_path`, `assert_safe_url`,
   `run_python_sandboxed`).

   ```python
   from cortex.tools.base import ToolResult

   def reverse_text(text: str) -> ToolResult:
       """Reverse a string."""
       out = str(text)[::-1]
       return ToolResult(output=out, data=out)
   ```

2. Register it in `build_default_registry()` (`cortex/tools/builtin.py`):

   ```python
   Tool(
       name="reverse_text",
       description="Reverse the characters of a string.",
       parameters={"text": {"type": "string", "description": "Text to reverse."}},
       required=["text"],
       func=reverse_text,
       dangerous=False,   # set True for write/network/exec tools
   )
   ```

3. If the tool is dangerous, add its name to `DANGEROUS_TOOLS` in
   `cortex/policy.py` so the approval gate covers it.

4. Add tests in `tests/test_tools.py` (happy path **and** abuse cases — traversal,
   SSRF, oversized input, etc.).

`to_schema()` automatically renders the tool into Anthropic's tool-definition
format, so it works across all backends and shows up in `GET /tools`.

---

## 6. Adding a new LLM backend

Implement the `LLMBackend` protocol (`cortex/llm/base.py`): a `name` attribute
and a `complete(...)` that returns a normalized `LLMResponse`.

1. Create `cortex/llm/yourprovider_backend.py`:

   ```python
   from __future__ import annotations
   from typing import Any, Dict, List, Optional
   from .base import LLMResponse, Message, ToolCall

   DEFAULT_MODEL = "your-model-id"

   class YourBackend:
       name = "yourprovider"

       def __init__(self, model: str = DEFAULT_MODEL, **kwargs: Any) -> None:
           self.model = model
           # read keys from env (e.g. os.environ["YOURPROVIDER_API_KEY"]) —
           # never accept them via Settings or hardcode them.

       def complete(
           self,
           messages: List[Message],
           tools: Optional[List[Dict[str, Any]]] = None,
           system: Optional[str] = None,
           max_tokens: int = 2048,
           temperature: float = 0.7,
       ) -> LLMResponse:
           # 1. translate Message/tools into the provider's request
           # 2. call the provider
           # 3. normalize back into LLMResponse(text, tool_calls=[ToolCall(...)],
           #    usage={"input_tokens": ..., "output_tokens": ...}, model=self.model)
           ...
   ```

2. Wire it into `get_backend()` (`cortex/llm/__init__.py`) under a new name, and
   add it to the `_valid_backend` set + validator note in `cortex/config.py`.

3. If you want cost accounting, add the model's price to `_PRICE_TABLE` in
   `cortex/llm/cost.py`.

4. Keep heavy SDKs as **lazy imports** inside the backend module (mirroring the
   `anthropic`/`hf`/`tinybrain` backends) so the core stays importable without
   them. Add tests in `tests/test_backends.py` / `tests/test_llm.py`.

Your backend automatically benefits from `ResilientBackend` (retries, timeout,
circuit breaker, failover) when used through `build_resilient_from_settings`.

---

## 7. Adding a database migration

State lives in SQLAlchemy 2.0 async models (`cortex/db/models.py`) with Alembic
migrations in `migrations/`.

1. Edit the ORM models in `cortex/db/models.py`.

2. Autogenerate a revision (Alembic uses a **sync** URL derived from
   `DATABASE_URL` via `Settings.resolved_sync_url()`):

   ```bash
   alembic revision --autogenerate -m "add my_table"
   ```

3. **Review the generated script** in `migrations/versions/` — autogenerate is a
   draft, not gospel. Check column types, nullability, indexes, and the
   `down_revision` chain; SQLite has limited `ALTER` support, so some changes
   need a batch operation.

4. Apply and verify:

   ```bash
   alembic upgrade head        # or: make migrate
   ```

5. Keep `create_all` (the SQLite/dev bootstrap in `cortex/db/engine.py`) and the
   migrations consistent. Add/extend tests in `tests/test_db.py`. CI runs the
   migrations against a temp SQLite database, so they must apply cleanly from
   scratch.

---

## 8. Project layout

See [`ARCHITECTURE.md` §5](./ARCHITECTURE.md#5--module--package-layout) for the
full annotated tree. Quick orientation:

- `cortex/agent/` — the ReAct loop (`loop.py` sync, `runtime.py` async) + planner.
- `cortex/llm/` — backends + the `LLMBackend` protocol + resilience + cost.
- `cortex/tools/` — tool abstraction (`base.py`), built-ins (`builtin.py`),
  sandbox (`sandbox.py`).
- `cortex/db/` — models, async engine, repositories.
- `cortex/api/` — FastAPI app, deps (auth/rate limit), middleware, routes.
- `cortex/service.py`, `cortex/policy.py`, `cortex/security.py`,
  `cortex/observability.py`, `cortex/config.py` — orchestration + cross-cutting.
- `cortex/worker/` — optional arq+redis queue.
- `cortex/tinybrain/` — the from-scratch Transformer LM (torch-gated).
- `tests/` — offline, deterministic test suite.
- `deploy/`, `migrations/`, `Dockerfile`, `docker-compose.yml` — ops.

---

## 9. PR checklist

Before opening a pull request:

- [ ] `make lint` is clean (`ruff check` + `ruff format --check`).
- [ ] `make typecheck` is clean (`mypy cortex`).
- [ ] `make security` is clean (`bandit` + `pip-audit`); don't blindly add bandit
      skips — justify any in `pyproject.toml`.
- [ ] `make test` passes and **coverage is maintained** (CI gate: `--cov-fail-under=80`).
- [ ] New behavior has tests, including abuse/edge cases for security-relevant code.
- [ ] Code is **Python 3.9 compatible** (`typing.Optional`/`List`, no `X | None`);
      `torch` stays import-guarded.
- [ ] Docs updated if you changed architecture, security posture, config, or the
      public API (`ARCHITECTURE.md` / `SECURITY.md` / docstrings).
- [ ] No secrets, weights, datasets, or local DBs committed (pre-commit blocks
      files > 25 MB and detects private keys).

---

## 10. Commit & branch conventions

- **Branches:** short, descriptive, type-prefixed — e.g. `feat/web-search-tool`,
  `fix/ssrf-redirect`, `docs/architecture`, `chore/bump-deps`.
- **Commits:** imperative mood, concise subject (≤ ~72 chars), with a body
  explaining *why* when non-obvious. Conventional-commit prefixes (`feat:`,
  `fix:`, `docs:`, `test:`, `refactor:`, `chore:`) are encouraged.
- **Keep PRs focused** — one logical change per PR; separate refactors from
  behavior changes so review and `git bisect` stay tractable.
- Reference related issues in the PR description and note any security or
  migration impact explicitly.

Run `pre-commit run --all-files` before pushing to catch formatting and hygiene
issues early. Thanks again for contributing!
</content>
