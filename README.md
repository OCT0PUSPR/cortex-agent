<div align="center">

# ◆ cortex-agent

**A single, powerful autonomous agent — planning, tool use, memory, and a ReAct loop.**

[![CI](https://github.com/OCT0PUSPR/cortex-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/OCT0PUSPR/cortex-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2dd4bf.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-5b9bff.svg)](https://www.python.org/)
[![Backends](https://img.shields.io/badge/backends-anthropic%20%7C%20hf%20%7C%20mock-c084fc.svg)](#-llm-backends)
[![Runs offline](https://img.shields.io/badge/runs%20offline-MockLLM-34d399.svg)](#-the-mockllm-backend)

</div>

---

`cortex-agent` is a clean, runnable, open-source **autonomous agentic AI framework**. It gives
you one capable agent that **plans** a goal into steps, **reasons** about what to do, **calls
tools**, **observes** the results, and **remembers** across runs — all streamed as structured
events you can render in a terminal or the web UI.

It ships with three interchangeable LLM backends behind one protocol: **Anthropic Claude**
(native tool use), **HuggingFace** (Inference API or local `transformers`), and a deterministic
**MockLLM** so the entire agent loop — and the test suite — runs **end to end with no API key and
no network**.

```bash
# No key, no network — the full loop runs on the MockLLM:
cortex run "Calculate 21 * 2 and tell me the current time"
```

## ✨ Features

- **ReAct / plan-execute loop** — think → choose tool → observe → repeat, with a max-steps budget
  and a planner that decomposes goals into a task list.
- **Real, safe tools** — calculator (AST eval, no `eval`), sandboxed `read_file`/`write_file`,
  `run_python` (subprocess + timeout in a temp sandbox), `http_get`, `current_time`, and a
  pluggable `web_search` with an offline fixture fallback.
- **Pluggable LLM backends** — Anthropic (native `tools` + `tool_use`/`tool_result` blocks),
  HuggingFace (API or local), and an offline **MockLLM** that drives a believable multi-step
  trajectory.
- **Memory** — a short-term conversation buffer plus long-term SQLite storage with **keyword
  recall** and **optional vector recall** (`sentence-transformers`, guarded; falls back to keyword).
- **Structured event stream** — every thought, tool call, observation, and answer is an
  `AgentEvent`, so UIs can stream the agent's reasoning live.
- **Three surfaces** — a `rich` CLI, a FastAPI server with **SSE streaming**, and a clean dark
  **web UI** that shows the plan and a live step timeline.
- **Authoring-friendly** — add a tool in ~10 lines; configure with env vars / `.env`.
- **Batteries included** — Dockerfile, docker-compose, pytest suite, and CI that passes offline.

## 🧠 Architecture: the agent loop

```mermaid
flowchart TD
    G([Goal]) --> P[Planner<br/>decompose into steps]
    P --> CTX[Build context<br/>system + recalled memory + tool schemas]
    CTX --> LLM{{LLM Backend<br/>anthropic · hf · mock}}
    LLM -->|thought + tool_use| EXEC[Tool Registry<br/>execute tool]
    EXEC -->|observation| OBS[Append tool_result<br/>+ write to memory]
    OBS --> LLM
    LLM -->|no more tools / budget hit| ANS[Synthesize final answer]
    ANS --> OUT([AgentResult + event stream])

    subgraph Events [Structured AgentEvents -> CLI / SSE / Web UI]
        E1[plan] --- E2[thought] --- E3[tool_call] --- E4[observation] --- E5[answer]
    end

    P -.emit.-> E1
    LLM -.emit.-> E2
    EXEC -.emit.-> E3
    OBS -.emit.-> E4
    ANS -.emit.-> E5

    subgraph Memory [Memory]
        STM[Short-term buffer]
        LTM[(Long-term SQLite<br/>keyword + optional vectors)]
    end
    CTX -. recall .-> LTM
    OBS -. remember .-> LTM
```

## 🚀 Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/OCT0PUSPR/cortex-agent.git
cd cortex-agent

# 2. (Recommended) create a virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 3. Install
pip install -e .
#   ...or just the deps:  pip install -r requirements.txt

# 4. Run the agent — works immediately on the offline MockLLM
cortex run "Search for cortex-agent and summarize it"
```

> The **MockLLM** backend is the default, so the command above needs **no API key and no network**.
> To use a real model, set `--backend anthropic` (and `ANTHROPIC_API_KEY`) or `--backend hf`.

### Use a real model

```bash
cp .env.example .env          # then edit .env
export ANTHROPIC_API_KEY=sk-ant-...      # or put it in .env

cortex --backend anthropic --model claude-sonnet-4-6 run "Plan a 3-step launch checklist"
cortex --backend hf --model Qwen/Qwen2.5-7B-Instruct run "What is a ReAct agent?"
```

## 🛠 Usage

### CLI

```bash
cortex run "<goal>"           # run once, streaming reasoning to the terminal
cortex chat                   # interactive REPL with persistent memory
cortex tools                  # list registered tools

# Flags (global): --backend mock|anthropic|hf  --model <id>  --max-steps <n>
cortex --backend mock --max-steps 6 run "Write a file plan.txt then read it back"
```

### Web UI + API server

```bash
# Start the FastAPI server (serves the web UI at http://127.0.0.1:8000)
uvicorn cortex.api.server:app --reload
#   ...or: python -m cortex.api.server
```

Open **http://127.0.0.1:8000** and give the agent a goal — the plan, thoughts, tool calls,
observations, and final answer stream in live over Server-Sent Events, with a step timeline.

API endpoints:

| Method | Path      | Description                                              |
| ------ | --------- | ------------------------------------------------------- |
| `GET`  | `/`       | Dark chat-style web UI                                   |
| `GET`  | `/health` | Health probe                                            |
| `GET`  | `/tools`  | List registered tools + JSON schemas                    |
| `POST` | `/run`    | Run a goal; **streams `AgentEvent`s as SSE**            |

```bash
# Stream a run over SSE from the command line:
curl -N -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Calculate 21 * 2", "backend": "mock", "max_steps": 6}'
```

### Python API

```python
from cortex import build_agent

# One-liner: fully wired agent (tools + memory) on the offline backend.
agent = build_agent(backend="mock")            # or backend="anthropic"
result = agent.run("Calculate 21 * 2 and tell me the current time")

print(result.answer)
print("plan:", result.plan.steps)
print("steps used:", result.steps_used)

# Or stream the structured events as they happen:
for event in agent.stream("Search for cortex-agent"):
    print(event.type.value, "->", event.content)
```

Compose the pieces yourself for full control:

```python
from cortex.agent import Agent
from cortex.llm import get_backend
from cortex.memory import Memory
from cortex.tools import build_default_registry

agent = Agent(
    backend=get_backend("anthropic", model="claude-opus-4-8"),
    registry=build_default_registry(workspace="./.cortex/workspace"),
    memory=Memory.create(db_path="./.cortex/memory.sqlite"),
    max_steps=8,
)
print(agent.run("Summarize what tools you have.").answer)
```

## 🧩 Tool-authoring guide

A tool is a name, a description (the model reads this to decide *when* to use it), a JSON-schema
for its parameters, and a `run` callable. Adding one is ~10 lines:

```python
from cortex.tools import Tool, build_default_registry

def reverse_text(text: str) -> str:
    """Return the input string reversed."""
    return text[::-1]

registry = build_default_registry()
registry.register(
    Tool(
        name="reverse_text",
        description="Reverse a string. Use when the user asks to reverse text.",
        parameters={"text": {"type": "string", "description": "Text to reverse."}},
        required=["text"],
        func=reverse_text,
    )
)
```

Your `func` may return a plain value (coerced to a `ToolResult`) or an explicit
`cortex.tools.ToolResult(output=..., is_error=..., data=...)` for richer control. The registry
renders every tool into Anthropic's native tool format automatically — Claude calls them via
`tool_use` blocks; open models call them via a JSON protocol the HF backend parses. Pass the
registry to an `Agent` and the new tool is immediately available.

**Safety notes:** file tools are sandboxed to the workspace dir (path traversal is rejected);
`run_python` executes in an isolated subprocess with a wall-clock timeout; the calculator parses
an AST and never uses `eval`.

## ⚙️ Configuration

Configuration comes from environment variables (prefix `CORTEX_`) and an optional `.env`
(via `pydantic-settings`). Copy `.env.example` to `.env`. **API keys are read from their standard
env vars and are never hardcoded.**

| Variable                      | Default                  | Description                                    |
| ----------------------------- | ------------------------ | ---------------------------------------------- |
| `ANTHROPIC_API_KEY`           | —                        | Claude API key (read by the Anthropic backend) |
| `HF_TOKEN`                    | —                        | HuggingFace token (read by the HF backend)     |
| `CORTEX_BACKEND`              | `mock`                   | `mock`, `anthropic`, or `hf`                    |
| `CORTEX_MODEL`                | backend default          | Model id override                              |
| `CORTEX_MAX_STEPS`            | `8`                      | Max ReAct steps per run                        |
| `CORTEX_MAX_TOKENS`           | `2048`                   | Max output tokens per LLM call                 |
| `CORTEX_TEMPERATURE`          | `0.7`                    | Sampling temperature                           |
| `CORTEX_WORKSPACE`            | `.cortex/workspace`      | Sandbox dir for file/python tools              |
| `CORTEX_MEMORY_DB`            | `.cortex/memory.sqlite`  | SQLite path for long-term memory               |
| `CORTEX_USE_VECTORS`          | `false`                  | Enable vector recall (if deps installed)       |
| `CORTEX_ENABLE_NETWORK_TOOLS` | `true`                   | Expose `http_get` to the agent                 |
| `CORTEX_HOST` / `CORTEX_PORT` | `127.0.0.1` / `8000`     | API server bind address                        |

### Model IDs (Anthropic)

`claude-opus-4-8` (most capable) · `claude-sonnet-4-6` (balanced default) · `claude-haiku-4-5`
(fast/cheap). HuggingFace examples: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`.

## 🤖 LLM backends

All three implement the same `LLMBackend` protocol (`complete(messages, tools) -> LLMResponse`)
with a normalized tool-call representation, so they are fully interchangeable.

| Backend     | How tool use works                                        | Needs a key? |
| ----------- | --------------------------------------------------------- | ------------ |
| `anthropic` | Native Anthropic `tools` + `tool_use`/`tool_result` blocks | `ANTHROPIC_API_KEY` |
| `hf`        | JSON tool-call protocol via Inference API or local `transformers` | `HF_TOKEN` (API mode) |
| `mock`      | Scripted, deterministic trajectory                        | **No** |

### The MockLLM backend

The `MockLLM` inspects the conversation and drives a believable, reproducible multi-step run: it
emits a thought, picks a tool relevant to the goal (calculator, time, file, search…), and after
seeing the observation either chains a second tool or synthesizes a final answer that quotes the
result. This is what makes the demo, the web UI, and the **entire test suite** run offline.

## 🐳 Docker

```bash
# Build + run the API/web UI (offline MockLLM by default)
docker compose up --build
# -> open http://127.0.0.1:8000

# Run the CLI in the container
docker run --rm cortex-agent cortex run "Calculate 21 * 2"

# Use a real backend by passing the key through
docker run --rm -e ANTHROPIC_API_KEY=sk-ant-... cortex-agent \
  cortex --backend anthropic run "Plan my day"
```

## 🧪 Tests

The suite covers the tool registry, the calculator/file/python tools, memory recall, and a **full
agent run end to end** — all on the MockLLM, with **no network and no API key**.

```bash
pip install -r requirements-min.txt
pytest -q
```

CI runs lint (`ruff`) + `pytest` on Python 3.9/3.11/3.12 against the MockLLM only.

## 📁 Project tree

```
cortex-agent/
├── cortex/
│   ├── __init__.py            # build_agent() + public API
│   ├── config.py              # pydantic-settings (with stdlib fallback)
│   ├── cli.py                 # rich CLI: run / chat / tools
│   ├── llm/
│   │   ├── base.py            # LLMBackend protocol + normalized types
│   │   ├── anthropic_backend.py  # Claude, native tool use
│   │   ├── hf_backend.py      # HuggingFace API / local transformers
│   │   └── mock_backend.py    # deterministic offline backend
│   ├── tools/
│   │   ├── base.py            # Tool + ToolRegistry
│   │   └── builtin.py         # calculator, files, run_python, http_get, ...
│   ├── memory/
│   │   └── store.py           # short-term buffer + SQLite long-term recall
│   ├── agent/
│   │   ├── loop.py            # the Agent + ReAct loop + AgentEvents
│   │   └── planner.py         # goal -> task list
│   └── api/
│       ├── server.py          # FastAPI: /run (SSE), /health, /tools, /
│       └── web/               # index.html + app.js + style.css (dark UI)
├── tests/                     # pytest: tools, memory, llm, full agent run
├── .github/workflows/ci.yml   # lint + pytest (offline, MockLLM only)
├── requirements.txt           # full runtime deps
├── requirements-min.txt       # minimal deps for CI (no heavy ML)
├── pyproject.toml             # packaging + console script + ruff/pytest config
├── Dockerfile / docker-compose.yml
├── .env.example
└── LICENSE
```

## 🗺 Roadmap

- [ ] Streaming token-level output from the Anthropic backend into the event stream.
- [ ] Parallel tool execution within a single step.
- [ ] Pluggable real web-search providers (Tavily, SerpAPI, Brave).
- [ ] Reflection / self-critique step before finalizing an answer.
- [ ] Persisted, resumable sessions and a conversation history viewer in the web UI.
- [ ] A `cortex serve` CLI subcommand and richer per-tool permission policies.
- [ ] More backends (OpenAI-compatible local servers, Ollama) behind the same protocol.

## 📄 License

[MIT](LICENSE) © 2026 OCT0PUSPR
