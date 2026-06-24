"""arq worker package for cortex-agent."""

from __future__ import annotations

__all__ = ["build_worker_settings", "enqueue_run", "execute_run"]


def __getattr__(name: str):  # lazy so arq stays optional
    if name in __all__:
        from . import tasks

        return getattr(tasks, name)
    raise AttributeError(name)


# The arq CLI imports `cortex.worker.WorkerSettings`; provide it lazily.
def __dir__():  # pragma: no cover
    return list(__all__) + ["WorkerSettings"]
