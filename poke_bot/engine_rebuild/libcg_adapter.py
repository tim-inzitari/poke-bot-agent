"""Adapters bridging today's process-isolated libcg toward MultiEnv.

``LibcgProcessAdapter`` is a *design stub*: it documents the intended shape of
wrapping N worker processes (each with one official battle) behind the same
:class:`~poke_bot.engine_rebuild.interfaces.MultiEnv` protocol so collect code
can switch to an in-process fork later without rewriting callers.

It does **not** import ``cg`` at module load (this cloud env has no libcg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .interfaces import Action, BatchObs, EnvObs, ResetSpec


@dataclass
class LibcgProcessAdapter:
    """Emulate MultiEnv with one OS process per slot (status quo).

    Real implementation would own a ``WorkerPool`` or persistent child procs and
    RPC ``battle_start`` / ``battle_select``. Left unimplemented here on purpose
    so the spike stays import-safe without competition binaries.
    """

    num_envs: int
    _closed: bool = False

    def reset(self, specs: Sequence[ResetSpec]) -> BatchObs:
        self._ensure()
        raise NotImplementedError(
            "LibcgProcessAdapter.reset requires libcg workers; "
            "use FakeMultiEnv in tests or implement against WorkerPool on the "
            "training box after setup_competition_data.sh"
        )

    def step_batch(self, actions: Sequence[Optional[Action]]) -> BatchObs:
        self._ensure()
        raise NotImplementedError(
            "LibcgProcessAdapter.step_batch requires libcg workers"
        )

    def close(self) -> None:
        self._closed = True

    def _ensure(self) -> None:
        if self._closed:
            raise RuntimeError("LibcgProcessAdapter is closed")


def empty_batch(num_envs: int) -> BatchObs:
    """Helper: terminal placeholders for all slots."""
    return BatchObs(
        envs=[
            EnvObs(env_id=i, obs={"current": {"result": -1}, "select": None}, done=True)
            for i in range(num_envs)
        ]
    )
