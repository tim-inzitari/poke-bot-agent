"""ctypes adapter for the private, competition-licensed batched libcg fork.

The fork preserves the official ABI and adds four opt-in symbols:

``BatchAbiVersion``
    Capability/version probe.
``BattleStartBatchSeeded``
    Reset independent native handles with deterministic per-game seeds.
``GetBattleDataBatch``
    Serialize several handles with one Python/native boundary crossing.
``StepBatch``
    Apply flattened actions and serialize all resulting observations in one
    boundary crossing.

No competition engine source or binary is contained in this module.  Callers
must point it at a locally built, appropriately licensed library.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from poke_bot.engine_rebuild.interfaces import Action, BatchObs, EnvObs, ResetSpec


class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


class StepData(ctypes.Structure):
    _fields_ = [("serial", SerialData), ("error", ctypes.c_int)]


_INITIALIZED_PATHS: set[str] = set()


def _configure_library(lib: Any) -> None:
    """Declare the preserved official ABI and the fork-only batch ABI."""
    lib.GameInitialize.restype = None
    lib.GameInitialize.argtypes = []

    lib.BattleStart.restype = StartData
    lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

    lib.BattleFinish.restype = None
    lib.BattleFinish.argtypes = [ctypes.c_void_p]

    lib.GetBattleData.restype = SerialData
    lib.GetBattleData.argtypes = [ctypes.c_void_p]

    lib.Select.restype = ctypes.c_int
    lib.Select.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

    lib.BatchAbiVersion.restype = ctypes.c_int
    lib.BatchAbiVersion.argtypes = []

    lib.BattleStartSeeded.restype = StartData
    lib.BattleStartSeeded.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint32]

    lib.BattleStartBatchSeeded.restype = ctypes.c_int
    lib.BattleStartBatchSeeded.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.POINTER(StartData),
    ]

    lib.GetBattleDataBatch.restype = ctypes.c_int
    lib.GetBattleDataBatch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.POINTER(SerialData),
    ]

    lib.StepBatch.restype = ctypes.c_int
    lib.StepBatch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(StepData),
    ]


def load_batch_library(
    path: str | os.PathLike[str], *, initialize: bool = True
) -> Any:
    """Load and validate a batch-fork library.

    ``GameInitialize`` is not idempotent in the upstream engine, so it is
    called at most once per resolved library path in a Python process.
    """
    resolved = str(Path(path).expanduser().resolve())
    lib = ctypes.CDLL(resolved)
    try:
        _configure_library(lib)
        version = int(lib.BatchAbiVersion())
    except AttributeError as exc:
        raise RuntimeError(f"libcg at {resolved} has no batched ABI") from exc
    if version != 1:
        raise RuntimeError(f"unsupported libcg batch ABI {version}; expected 1")
    if initialize and resolved not in _INITIALIZED_PATHS:
        lib.GameInitialize()
        _INITIALIZED_PATHS.add(resolved)
    return lib


class BatchedLibcgMultiEnv:
    """Many deterministic CABT games behind one native batch call per ply."""

    def __init__(
        self,
        num_envs: int,
        *,
        lib_path: str | os.PathLike[str] | None = None,
        lib: Any | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if lib is None:
            path = lib_path or os.environ.get("POKEBOT_BATCH_LIBCG")
            if not path:
                raise ValueError("lib_path or POKEBOT_BATCH_LIBCG is required")
            lib = load_batch_library(path)
        else:
            _configure_library(lib)
            if int(lib.BatchAbiVersion()) != 1:
                raise RuntimeError("unsupported libcg batch ABI")

        self._num_envs = int(num_envs)
        self._lib = lib
        self._ptrs: list[Optional[int]] = [None] * self._num_envs
        self._obs: list[Optional[dict]] = [None] * self._num_envs
        self._done: list[bool] = [True] * self._num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, specs: Sequence[ResetSpec]) -> BatchObs:
        count = len(specs)
        if count > self._num_envs:
            raise ValueError(f"got {count} specs for capacity {self._num_envs}")
        if count == 0:
            return self._snapshot()

        for i, spec in enumerate(specs):
            self._finish(i)
            if len(spec.deck0) != 60 or len(spec.deck1) != 60:
                raise ValueError(f"env {i}: each deck must contain 60 cards")

        flat_cards = [
            card
            for spec in specs
            for card in (list(spec.deck0) + list(spec.deck1))
        ]
        cards = (ctypes.c_int * len(flat_cards))(*flat_cards)
        seeds = (ctypes.c_uint32 * count)(
            *(int(spec.seed) & 0xFFFFFFFF for spec in specs)
        )
        starts = (StartData * count)()
        rc = int(self._lib.BattleStartBatchSeeded(cards, seeds, count, starts))

        errors: list[str] = []
        for i, start in enumerate(starts):
            ptr = int(start.battlePtr or 0)
            if ptr:
                self._ptrs[i] = ptr
            else:
                errors.append(
                    f"env={i} player={start.errorPlayer} type={start.errorType}"
                )
        if rc == 30 or errors:
            for i in range(count):
                self._finish(i)
            detail = ", ".join(errors) if errors else "malformed batch arguments"
            raise RuntimeError(f"BattleStartBatchSeeded failed: {detail}")

        ptr_array = (ctypes.c_void_p * count)(
            *(self._ptrs[i] or 0 for i in range(count))
        )
        serials = (SerialData * count)()
        failures = int(self._lib.GetBattleDataBatch(ptr_array, count, serials))
        if failures:
            for i in range(count):
                self._finish(i)
            raise RuntimeError(f"GetBattleDataBatch failed for {failures}/{count} envs")
        for i, serial in enumerate(serials):
            self._set_serial(i, serial)
        return self._snapshot()

    def step_batch(self, actions: Sequence[Optional[Action]]) -> BatchObs:
        if len(actions) != self._num_envs:
            raise ValueError(
                f"actions length {len(actions)} != num_envs {self._num_envs}"
            )

        active: list[tuple[int, Action]] = []
        for i, action in enumerate(actions):
            if action is None or self._ptrs[i] is None or self._done[i]:
                continue
            if not isinstance(action, list) or not all(
                isinstance(value, int) for value in action
            ):
                raise ValueError(f"env {i}: action must be list[int] or None")
            active.append((i, action))
        if not active:
            return self._snapshot()

        ptr_array = (ctypes.c_void_p * len(active))(
            *(self._ptrs[i] or 0 for i, _ in active)
        )
        offsets_list = [0]
        flat_actions: list[int] = []
        for _, action in active:
            flat_actions.extend(action)
            offsets_list.append(len(flat_actions))
        offsets = (ctypes.c_int * len(offsets_list))(*offsets_list)
        if flat_actions:
            selections: Any = (ctypes.c_int * len(flat_actions))(*flat_actions)
        else:
            selections = ctypes.POINTER(ctypes.c_int)()
        results = (StepData * len(active))()

        rc = int(
            self._lib.StepBatch(
                ptr_array, selections, offsets, len(active), results
            )
        )
        if rc:
            raise RuntimeError(f"StepBatch rejected arguments: err={rc}")
        for lane, ((slot, _), result) in enumerate(zip(active, results)):
            if result.error:
                raise RuntimeError(
                    f"StepBatch failed lane={lane} env={slot} err={result.error}"
                )
            self._set_serial(slot, result.serial)
        return self._snapshot()

    def close(self) -> None:
        for i in range(self._num_envs):
            self._finish(i)

    def __enter__(self) -> "BatchedLibcgMultiEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _finish(self, i: int) -> None:
        ptr = self._ptrs[i]
        if ptr:
            self._lib.BattleFinish(ptr)
        self._ptrs[i] = None
        self._obs[i] = None
        self._done[i] = True

    def _set_serial(self, slot: int, serial: SerialData) -> None:
        if not serial.json:
            raise RuntimeError(f"env {slot}: native engine returned empty JSON")
        obs = json.loads(serial.json.decode("utf-8"))
        if serial.data and serial.count > 0:
            obs["search_begin_input"] = ctypes.string_at(
                serial.data, serial.count
            ).decode("ascii")
        self._obs[slot] = obs
        self._done[slot] = self._is_done(obs)

    def _snapshot(self) -> BatchObs:
        return BatchObs(
            envs=[
                EnvObs(
                    env_id=i,
                    obs=self._obs[i] or {},
                    done=self._done[i] if self._ptrs[i] is not None else True,
                    winner=self._winner(self._obs[i]),
                )
                for i in range(self._num_envs)
            ]
        )

    @staticmethod
    def _is_done(obs: Optional[dict]) -> bool:
        if not obs:
            return True
        result = (obs.get("current") or {}).get("result")
        return result is not None and int(result) != -1

    @staticmethod
    def _winner(obs: Optional[dict]) -> Optional[int]:
        if not obs:
            return None
        result = (obs.get("current") or {}).get("result")
        if result is None or int(result) == -1:
            return None
        return int(result)
