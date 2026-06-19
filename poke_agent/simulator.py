from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class SimulatorState:
    lib_path: str | None
    available: bool
    error: str | None
    battle_start: Callable[..., Any] | None
    battle_select: Callable[..., Any] | None
    battle_finish: Callable[..., Any] | None
    to_observation_class: Callable[..., Any] | None


def find_cg_lib(root: Path) -> str | None:
    candidates: list[str] = []
    if os.environ.get("CG_LIB_PATH"):
        candidates.append(os.environ["CG_LIB_PATH"])
    candidates.extend(glob.glob("/kaggle/input/**/cg-lib", recursive=True))
    candidates.extend(glob.glob(str(root / "kaggle/input/**/cg-lib"), recursive=True))
    return candidates[0] if candidates else None


def load_simulator(root: Path) -> SimulatorState:
    lib_path = find_cg_lib(root)
    if not lib_path:
        return SimulatorState(lib_path=None, available=False, error=None, battle_start=None, battle_select=None, battle_finish=None, to_observation_class=None)

    sys.path.append(lib_path)
    try:
        from cg.game import battle_finish, battle_select, battle_start
        from cg.api import to_observation_class

        return SimulatorState(
            lib_path=lib_path,
            available=True,
            error=None,
            battle_start=battle_start,
            battle_select=battle_select,
            battle_finish=battle_finish,
            to_observation_class=to_observation_class,
        )
    except Exception as exc:
        return SimulatorState(
            lib_path=lib_path,
            available=False,
            error=repr(exc),
            battle_start=None,
            battle_select=None,
            battle_finish=None,
            to_observation_class=None,
        )


def print_simulator_status(state: SimulatorState) -> None:
    print("cg_lib_path", state.lib_path)
    print("cg_available", state.available)
    if state.error:
        print("cg_error", state.error)
