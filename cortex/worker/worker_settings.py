"""arq entry point: ``arq cortex.worker.worker_settings.WorkerSettings``.

Importing this module requires ``arq`` (and a reachable Redis at runtime).
"""

from __future__ import annotations

from .tasks import build_worker_settings

WorkerSettings = build_worker_settings()
