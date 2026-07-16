"""MultiEnv adapter over official libcg using native ApiData* handles.

M0 finding (2026-07-16): the C ABI is already multi-handle —

* ``BattleStart`` → ``StartData.battlePtr`` (``ApiData*``)
* ``Select(ApiData*, …)`` / ``GetBattleData(ApiData*)`` / ``BattleFinish(ApiData*)``

The process-global singleton lives in **Python** ``cg.game.Battle.battle_ptr``,
not in the shipped binary. This adapter calls ``cg.sim.lib`` directly so many
battles share one process / one ``libcg.so``.

Card tables (``CardTable`` etc.) are process-global and initialized once by
``GameInitialize()`` — treated as shared read-mostly metadata.
"""

from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Optional, Sequence

from poke_bot.engine_rebuild.interfaces import (
    Action,
    BatchObs,
    EnvObs,
    ResetSpec,
)


def _load_sim_lib() -> Any:
    """Import competition ``cg.sim`` (triggers ``GameInitialize`` once)."""
    from cg import sim  # type: ignore

    return sim.lib


class LibcgMultiEnv:
    """N concurrent battles via official ``libcg`` handles (no C++ fork).

    When ``libcg_step_batch.so`` is present, ``step_batch`` uses the native
    C++ ``StepBatch`` export (one ctypes call for N Select+GetBattleData).
    Set ``prefer_native_step_batch=False`` or ``POKEBOT_STEP_BATCH=0`` to force
    the Python loop (stock path).
    """

    def __init__(
        self,
        num_envs: int,
        *,
        lib: Any | None = None,
        prefer_native_step_batch: bool | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self._num_envs = int(num_envs)
        self._lib = lib if lib is not None else _load_sim_lib()
        self._ptrs: list[Optional[int]] = [None] * self._num_envs
        self._obs: list[Optional[dict]] = [None] * self._num_envs
        self._done: list[bool] = [True] * self._num_envs
        if prefer_native_step_batch is None:
            flag = os.environ.get("POKEBOT_STEP_BATCH", "1").strip().lower()
            prefer_native_step_batch = flag not in ("0", "false", "no", "off")
        self._step_batch_lib: Any | None = None
        if prefer_native_step_batch:
            from poke_bot.engine_rebuild.libcg_step_batch import load_step_batch_lib

            self._step_batch_lib = load_step_batch_lib()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def uses_native_step_batch(self) -> bool:
        return self._step_batch_lib is not None

    def reset(self, specs: Sequence[ResetSpec]) -> BatchObs:
        if len(specs) > self._num_envs:
            raise ValueError(f"got {len(specs)} specs for capacity {self._num_envs}")
        out: list[EnvObs] = []
        for i, spec in enumerate(specs):
            self._finish(i)
            if len(spec.deck0) != 60 or len(spec.deck1) != 60:
                raise ValueError("each deck must contain 60 cards")
            # Official BattleStart ignores seed in the public C ABI; seed is kept
            # on ResetSpec for fork parity later.
            cards = (ctypes.c_int * 120)(*(list(spec.deck0) + list(spec.deck1)))
            start = self._lib.BattleStart(cards)
            ptr = int(start.battlePtr or 0)
            if not ptr:
                raise RuntimeError(
                    f"BattleStart failed env={i} errorPlayer={start.errorPlayer} "
                    f"errorType={start.errorType}"
                )
            self._ptrs[i] = ptr
            obs = self._get_obs(ptr)
            done = self._is_done(obs)
            self._obs[i] = obs
            self._done[i] = done
            out.append(EnvObs(env_id=i, obs=obs, done=done, winner=self._winner(obs)))
        # untouched slots stay as-is
        for i in range(len(specs), self._num_envs):
            if self._obs[i] is not None:
                out.append(
                    EnvObs(
                        env_id=i,
                        obs=self._obs[i] or {},
                        done=self._done[i],
                        winner=self._winner(self._obs[i]),
                    )
                )
        return BatchObs(envs=out)

    def step_batch(self, actions: Sequence[Optional[Action]]) -> BatchObs:
        if len(actions) != self._num_envs:
            raise ValueError(f"actions length {len(actions)} != num_envs {self._num_envs}")
        if self._step_batch_lib is not None:
            return self._step_batch_native(actions)
        return self._step_batch_python(actions)

    def _step_batch_python(self, actions: Sequence[Optional[Action]]) -> BatchObs:
        out: list[EnvObs] = []
        for i, action in enumerate(actions):
            ptr = self._ptrs[i]
            if ptr is None or action is None or self._done[i]:
                out.append(
                    EnvObs(
                        env_id=i,
                        obs=self._obs[i] or {},
                        done=self._done[i] if ptr is not None else True,
                        winner=self._winner(self._obs[i]),
                    )
                )
                continue
            sel = (ctypes.c_int * len(action))(*action)
            err = int(self._lib.Select(ptr, sel, len(action)))
            if err != 0:
                raise RuntimeError(f"Select failed env={i} err={err}")
            obs = self._get_obs(ptr)
            done = self._is_done(obs)
            self._obs[i] = obs
            self._done[i] = done
            out.append(EnvObs(env_id=i, obs=obs, done=done, winner=self._winner(obs)))
        return BatchObs(envs=out)

    def _step_batch_native(self, actions: Sequence[Optional[Action]]) -> BatchObs:
        from poke_bot.engine_rebuild.libcg_step_batch import step_batch_native

        step_actions: list[Optional[Action]] = []
        for i, action in enumerate(actions):
            if self._ptrs[i] is None or action is None or self._done[i]:
                step_actions.append(None)
            else:
                step_actions.append(action)
        errors, jsons, _sp = step_batch_native(
            self._step_batch_lib, self._ptrs, step_actions, fetch_obs_on_skip=False
        )
        out: list[EnvObs] = []
        for i, action in enumerate(actions):
            ptr = self._ptrs[i]
            if ptr is None or action is None or self._done[i]:
                out.append(
                    EnvObs(
                        env_id=i,
                        obs=self._obs[i] or {},
                        done=self._done[i] if ptr is not None else True,
                        winner=self._winner(self._obs[i]),
                    )
                )
                continue
            err = errors[i]
            if err != 0:
                raise RuntimeError(f"Select failed env={i} err={err}")
            raw = jsons[i]
            if raw is None:
                raise RuntimeError(f"StepBatch returned empty JSON env={i}")
            obs = json.loads(raw)
            done = self._is_done(obs)
            self._obs[i] = obs
            self._done[i] = done
            out.append(EnvObs(env_id=i, obs=obs, done=done, winner=self._winner(obs)))
        return BatchObs(envs=out)

    def close(self) -> None:
        for i in range(self._num_envs):
            self._finish(i)

    def _finish(self, i: int) -> None:
        ptr = self._ptrs[i]
        if ptr:
            self._lib.BattleFinish(ptr)
        self._ptrs[i] = None
        self._obs[i] = None
        self._done[i] = True

    def _get_obs(self, ptr: int) -> dict:
        sd = self._lib.GetBattleData(ptr)
        return json.loads(sd.json.decode())

    @staticmethod
    def _is_done(obs: Optional[dict]) -> bool:
        if not obs:
            return True
        current = obs.get("current") or {}
        # CABT: result == -1 while ongoing; 0/1/2 (etc.) when finished.
        result = current.get("result")
        if result is None:
            return False
        return int(result) != -1

    @staticmethod
    def _winner(obs: Optional[dict]) -> Optional[int]:
        if not obs:
            return None
        current = obs.get("current") or {}
        result = current.get("result")
        if result is None or int(result) == -1:
            return None
        return int(result)
