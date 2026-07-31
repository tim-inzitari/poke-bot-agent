"""Recyclable process supervisor (C++ core)."""

from __future__ import annotations

from ._proc_pool import (
    ProcPoolError,
    Supervisor,
    TaskResult,
    WorkerSpec,
    __version__,
)

__all__ = [
    "ProcPoolError",
    "Supervisor",
    "TaskResult",
    "WorkerSpec",
    "__version__",
]
