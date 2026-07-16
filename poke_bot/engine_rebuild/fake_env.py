"""Deterministic toy MultiEnv for interface tests (not CABT rules).

Each env is a tiny counter game: legal actions are ``[0]`` or ``[1]``; reaching
``target`` ends the episode. Used to validate batch-step / parity wiring without
``libcg``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .interfaces import Action, BatchObs, EnvObs, ResetSpec


class FakeMultiEnv:
    """In-process multi-env implementing the :class:`MultiEnv` protocol."""

    def __init__(self, num_envs: int, *, target: int = 3) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self._n = num_envs
        self._target = target
        self._counts = [0] * num_envs
        self._done = [True] * num_envs  # inactive until reset
        self._seeds = [0] * num_envs
        self._closed = False

    @property
    def num_envs(self) -> int:
        return self._n

    def reset(self, specs: Sequence[ResetSpec]) -> BatchObs:
        self._ensure_open()
        if len(specs) > self._n:
            raise ValueError(f"reset size {len(specs)} > num_envs {self._n}")
        out: list[EnvObs] = []
        for i, spec in enumerate(specs):
            if len(spec.deck0) != 60 or len(spec.deck1) != 60:
                raise ValueError("decks must be length 60 (CABT contract)")
            # Toy: seed only offsets the start count (keeps tests deterministic).
            self._seeds[i] = int(spec.seed)
            self._counts[i] = int(spec.seed) % self._target
            self._done[i] = False
            out.append(self._obs(i))
        for i in range(len(specs), self._n):
            self._done[i] = True
            self._counts[i] = 0
        return BatchObs(envs=out)

    def step_batch(self, actions: Sequence[Optional[Action]]) -> BatchObs:
        self._ensure_open()
        if len(actions) != self._n:
            raise ValueError(
                f"actions length {len(actions)} != num_envs {self._n}"
            )
        out: list[EnvObs] = []
        for i, action in enumerate(actions):
            if action is None or self._done[i]:
                out.append(self._obs(i))
                continue
            if not action:
                raise ValueError(f"env {i}: empty action")
            # Toy transition: any select increments; choice bit mixes into count.
            self._counts[i] += 1 + (action[0] & 1)
            if self._counts[i] >= self._target:
                self._done[i] = True
            out.append(self._obs(i))
        return BatchObs(envs=out)

    def close(self) -> None:
        self._closed = True

    def _obs(self, env_id: int) -> EnvObs:
        done = self._done[env_id]
        count = self._counts[env_id]
        winner = (count % 2) if done else None
        obs = {
            "current": {
                "result": (-1 if not done else (winner if winner is not None else 2)),
                "count": count,
                "target": self._target,
            },
            "select": None
            if done
            else {
                "minCount": 1,
                "maxCount": 1,
                "option": [{"id": 0}, {"id": 1}],
            },
            "logs": [],
        }
        return EnvObs(env_id=env_id, obs=obs, done=done, winner=winner)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("FakeMultiEnv is closed")
