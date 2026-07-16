"""Pickle-safe entry points for remote WorkerPool job dispatch.

``scripts/run_remote_worker.py`` loads ``train_round_robin`` via importlib under the
synthetic module name ``train_round_robin_remote``. Multiprocessing (spawn) cannot
unpickle those callables in child workers — they raise
``ModuleNotFoundError: train_round_robin_remote``, die, leak leaf response slots,
and leave the farm stuck at ``jobs_completed=0`` while hello/health still pass.

These wrappers live in an importable package module so ``Pool.apply`` round-trips
cleanly under spawn.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_RR = None
_MODULE_NAME = "train_round_robin_remote"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_round_robin_module():
    """Load ``scripts/train_round_robin.py`` once per process (lazy)."""
    global _RR
    if _RR is not None:
        return _RR
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        _RR = existing
        return _RR
    path = _repo_root() / "scripts" / "train_round_robin.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    _RR = mod
    return _RR


def remote_play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Whole-game collection job (importable; safe to ``Pool.apply``)."""
    return load_round_robin_module()._worker_play(job)


def remote_promotion_job(job: dict[str, Any]) -> dict[str, Any]:
    """Promotion evaluation job (importable; safe to ``Pool.apply``)."""
    return load_round_robin_module()._worker_promotion(job)
