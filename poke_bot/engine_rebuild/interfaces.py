"""Protocols for a multi-instance CABT-compatible simulator.

Designed so a future C++ fork can implement the same surface via pybind11 /
ctypes without changing collect loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable


#: Selected option indices for one env (CABT ``battle_select`` / ``search_step``).
Action = list[int]


@dataclass(frozen=True)
class ResetSpec:
    """Decks + seed for one env reset.

    ``seed`` must fully determine stochastic outcomes in a forked engine
    (shuffle, coin). Official ``libcg`` seeding is opaque — parity suites should
    prefer ``manual_coin`` search paths or recorded golden traces when comparing
    to the binary.
    """

    deck0: list[int]
    deck1: list[int]
    seed: int = 0


@dataclass
class EnvObs:
    """Minimal observation handle used by the spike.

    Production code will keep using raw ``cg`` obs dicts; this struct exists so
    tests and adapters share one shape before the fork lands.
    """

    env_id: int
    obs: dict
    done: bool = False
    winner: Optional[int] = None  # 0 / 1 / 2(draw) / None


@dataclass
class BatchObs:
    """Result of ``step_batch`` / ``reset_batch``."""

    envs: list[EnvObs] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.envs)


@runtime_checkable
class MultiEnv(Protocol):
    """N independent battles in one address space (fork target API).

    Official ``cg.game`` only supports ``num_envs == 1`` per process. A rebuild
    should implement this protocol natively; adapters may emulate it with a
    process pool until then.
    """

    @property
    def num_envs(self) -> int:
        """Fixed capacity (arena size)."""

    def reset(self, specs: Sequence[ResetSpec]) -> BatchObs:
        """Reset the first ``len(specs)`` slots (must be ``<= num_envs``)."""

    def step_batch(self, actions: Sequence[Optional[Action]]) -> BatchObs:
        """Advance every env that has a non-``None`` action.

        ``actions[i] is None`` means “skip / waiting on policy”. Length must
        equal ``num_envs``. Finished envs should either stay terminal or
        auto-reset per implementation policy (document which).
        """

    def close(self) -> None:
        """Release native resources."""
