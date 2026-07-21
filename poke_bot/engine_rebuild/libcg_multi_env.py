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
from pathlib import Path
from typing import Any, Optional, Sequence

from poke_bot.engine_rebuild.interfaces import (
    Action,
    BatchObs,
    EnvObs,
    ResetSpec,
)


_CUSTOM_LIBS: dict[str, Any] = {}


def _load_sim_lib() -> Any:
    """Load official libcg or the explicit training-only hidden-state fork."""
    # Resolve the vendored competition runtime from the repository before
    # importing ``cg``.  Systemd/launchd/container jobs intentionally start
    # with clean environments and must not depend on an interactive shell's
    # PYTHONPATH or CG_LIB_PATH.
    from poke_bot import cg_env

    cg_env.ensure_cg_importable()
    from cg import sim  # type: ignore

    override = os.environ.get("POKEBOT_LIBCG_PATH", "").strip()
    if not override:
        return sim.lib
    resolved = str(Path(override).expanduser().resolve())
    if resolved in _CUSTOM_LIBS:
        return _CUSTOM_LIBS[resolved]
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"POKEBOT_LIBCG_PATH does not exist: {resolved}")
    lib = ctypes.cdll.LoadLibrary(resolved)
    lib.GameInitialize()
    lib.BattleStart.restype = sim.StartData
    lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.BattleFinish.argtypes = [ctypes.c_void_p]
    lib.GetBattleData.restype = sim.SerialData
    lib.GetBattleData.argtypes = [ctypes.c_void_p]
    lib.Select.restype = ctypes.c_int
    lib.Select.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    try:
        lib.GetHiddenSnapshot.restype = ctypes.c_int
        lib.GetHiddenSnapshot.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.HiddenSnapshotAbiVersion.restype = ctypes.c_int
        if int(lib.HiddenSnapshotAbiVersion()) != 1:
            raise RuntimeError("unsupported hidden-state engine ABI")
    except AttributeError as exc:
        raise RuntimeError(
            "POKEBOT_LIBCG_PATH must point to the training hidden-state fork"
        ) from exc
    _CUSTOM_LIBS[resolved] = lib
    return lib


class LibcgMultiEnv:
    """N concurrent battles via official ``libcg`` handles (no C++ fork)."""

    def __init__(self, num_envs: int, *, lib: Any | None = None) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self._num_envs = int(num_envs)
        self._lib = lib if lib is not None else _load_sim_lib()
        self._ptrs: list[Optional[int]] = [None] * self._num_envs
        self._obs: list[Optional[dict]] = [None] * self._num_envs
        self._done: list[bool] = [True] * self._num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, specs: Sequence[ResetSpec]) -> BatchObs:
        if len(specs) > self._num_envs:
            raise ValueError(f"got {len(specs)} specs for capacity {self._num_envs}")
        out: list[EnvObs] = []
        for i, spec in enumerate(specs):
            self._finish(i)
            if len(spec.deck0) != 60 or len(spec.deck1) != 60:
                raise ValueError("each deck must contain 60 cards")
            cards = (ctypes.c_int * 120)(*(list(spec.deck0) + list(spec.deck1)))
            # The official library has only BattleStart. The private fork adds
            # BattleStartSeeded so individual-vs-batch benchmarks can execute
            # exactly the same games instead of comparing different RNG draws.
            try:
                seeded_start = self._lib.BattleStartSeeded
            except AttributeError:
                seeded_start = None
            start = (
                seeded_start(cards, ctypes.c_uint32(int(spec.seed) & 0xFFFFFFFF))
                if seeded_start is not None
                else self._lib.BattleStart(cards)
            )
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

    def close(self) -> None:
        for i in range(self._num_envs):
            self._finish(i)

    def hidden_snapshot(self, env_id: int, player: int) -> dict[str, list[int]]:
        """Return exact private zones for an auxiliary target only.

        Official libcg intentionally lacks this symbol.  Calling this method is
        therefore an explicit training-fork operation and cannot silently run
        in submission/evaluation code.
        """
        i = int(env_id)
        if not 0 <= i < self._num_envs:
            raise IndexError(f"invalid env_id {env_id}")
        ptr = self._ptrs[i]
        if not ptr:
            raise RuntimeError(f"environment {i} has no active battle")
        try:
            read_hidden = self._lib.GetHiddenSnapshot
        except AttributeError as exc:
            raise RuntimeError("hidden snapshot requested from official libcg") from exc
        buffer = (ctypes.c_int * 64)()
        used = int(read_hidden(ptr, int(player), buffer, len(buffer)))
        if used < 3 or used > len(buffer):
            raise RuntimeError(f"invalid hidden snapshot size {used}")
        hand_n, deck_n, prize_n = (int(buffer[j]) for j in range(3))
        if min(hand_n, deck_n, prize_n) < 0 or 3 + hand_n + deck_n + prize_n != used:
            raise RuntimeError(
                "hidden snapshot zone counts do not match returned payload"
            )
        offset = 3
        hand = [int(buffer[j]) for j in range(offset, offset + hand_n)]
        offset += hand_n
        deck = [int(buffer[j]) for j in range(offset, offset + deck_n)]
        offset += deck_n
        prize = [int(buffer[j]) for j in range(offset, offset + prize_n)]
        return {"hand": hand, "deck": deck, "prize": prize}

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
