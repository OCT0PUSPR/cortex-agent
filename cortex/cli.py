"""Rich-powered command-line interface for cortex-agent.

Commands:

* ``cortex run "<goal>"`` — run the agent once, streaming its reasoning.
* ``cortex chat`` — interactive REPL with persistent memory across turns.
* ``cortex tools`` — list the registered tools.

Backend is selected with ``--backend mock|anthropic|hf`` (default ``mock`` so it
works with no API key).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .agent.loop import Agent, AgentEvent, EventType
from .config import load_settings
from .llm import get_backend
from .memory import Memory
from .tools import build_default_registry

# rich is optional at import time; we degrade to plain printing if absent.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console: Optional["Console"] = Console()
except ImportError:  # pragma: no cover
    _console = None


_STYLES = {
    EventType.PLAN: ("📋 Plan", "cyan"),
    EventType.THOUGHT: ("💭 Thought", "yellow"),
    EventType.TOOL_CALL: ("🔧 Tool call", "magenta"),
    EventType.OBSERVATION: ("👁  Observation", "green"),
    EventType.ANSWER: ("✅ Answer", "bold green"),
    EventType.ERROR: ("❌ Error", "bold red"),
}


def _emit(event: AgentEvent) -> None:
    """Render a single agent event to the terminal."""
    label, color = _STYLES.get(event.type, (event.type.value, "white"))
    if _console is not None:
        if event.type in (EventType.ANSWER, EventType.PLAN):
            _console.print(Panel(event.content or "(empty)", title=label, border_style=color))
        else:
            prefix = f"[{color}]{label}[/{color}]"
            step = f"[dim](step {event.step})[/dim] " if event.step else ""
            _console.print(f"{step}{prefix}: {event.content}")
    else:
        step = f"(step {event.step}) " if event.step else ""
        print(f"{step}{label}: {event.content}")


def _build_agent(args: argparse.Namespace) -> Agent:
    cfg = load_settings(
        backend=args.backend,
        model=args.model,
        max_steps=args.max_steps,
    )
    backend = get_backend(cfg.backend, model=cfg.model)
    registry = build_default_registry(
        workspace=cfg.workspace,
        enable_network=cfg.enable_network_tools,
    )
    memory = Memory.create(db_path=cfg.memory_db, use_vectors=cfg.use_vectors)
    return Agent(
        backend=backend,
        registry=registry,
        memory=memory,
        max_steps=cfg.max_steps,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    if _console is not None:
        _console.rule(f"[bold]cortex-agent[/bold]  (backend: {agent.backend.name})")
    else:
        print(f"=== cortex-agent (backend: {agent.backend.name}) ===")
    for event in agent.stream(args.goal):
        _emit(event)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    banner = f"cortex-agent chat (backend: {agent.backend.name}). Type 'exit' to quit."
    if _console is not None:
        _console.print(Panel(banner, border_style="cyan"))
    else:
        print(banner)
    while True:
        try:
            goal = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if goal.lower() in {"exit", "quit", ":q"}:
            break
        if not goal:
            continue
        for event in agent.stream(goal):
            _emit(event)
    return 0


def _cmd_tools(args: argparse.Namespace) -> int:
    cfg = load_settings()
    registry = build_default_registry(
        workspace=cfg.workspace,
        enable_network=cfg.enable_network_tools,
    )
    if _console is not None:
        table = Table(title="Registered tools")
        table.add_column("name", style="magenta")
        table.add_column("description")
        for tool in registry.all():
            table.add_row(tool.name, tool.description)
        _console.print(table)
    else:
        for tool in registry.all():
            print(f"- {tool.name}: {tool.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="cortex-agent: an autonomous agentic AI framework.",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "anthropic", "hf", "tinybrain"],
        help="LLM backend (default: mock; works with no API key). 'tinybrain' serves the local from-scratch model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id override (for tinybrain: the checkpoint dir/path).",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum ReAct steps.")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the agent on a single goal.")
    run.add_argument("goal", help="The goal/task for the agent.")
    run.set_defaults(func=_cmd_run)

    chat = sub.add_parser("chat", help="Interactive chat with the agent.")
    chat.set_defaults(func=_cmd_chat)

    tools = sub.add_parser("tools", help="List the registered tools.")
    tools.set_defaults(func=_cmd_tools)

    return parser


def main(argv: Optional[list] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
