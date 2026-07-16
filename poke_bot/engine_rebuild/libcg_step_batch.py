"""Loader for additive ``libcg_step_batch.so`` (C++ StepBatch shim).

Prefers a forked / shim library when present next to stock ``libcg.so`` or via
``LIBCG_STEP_BATCH_SO``. Falls back to ``None`` so callers use the Python
Select+GetBattleData loop.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Optional, Sequence


_STEP_BATCH_LIB: Any | None = None
_STEP_BATCH_TRIED = False


def _candidate_paths() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("LIBCG_STEP_BATCH_SO")
    if env:
        out.append(Path(env).expanduser())
    # Next to this package's native/build
    here = Path(__file__).resolve().parent
    out.append(here / "native" / "build" / "libcg_step_batch.so")
    out.append(here / "native" / "libcg_step_batch.so")
    # Next to stock cg runtime
    try:
        from poke_bot.paths import cg_runtime_dir

        cg = cg_runtime_dir() / "cg"
        out.append(cg / "libcg_step_batch.so")
    except Exception:
        pass
    # Common competition layout
    root = here.parents[1]
    out.append(
        root
        / "kaggle"
        / "input"
        / "pokemon-tcg-ai-battle"
        / "sample_submission"
        / "sample_submission"
        / "cg"
        / "libcg_step_batch.so"
    )
    return out


def load_step_batch_lib() -> Any | None:
    """Return ctypes CDLL for the shim, or None if unavailable."""
    global _STEP_BATCH_LIB, _STEP_BATCH_TRIED
    if _STEP_BATCH_TRIED:
        return _STEP_BATCH_LIB
    _STEP_BATCH_TRIED = True

    # Ensure stock path is discoverable for the shim's dlopen("libcg.so").
    stock = os.environ.get("LIBCG_SO")
    if not stock:
        for parent in _candidate_paths():
            sibling = parent.parent / "libcg.so"
            if sibling.is_file():
                os.environ.setdefault("LIBCG_SO", str(sibling.resolve()))
                break
        if "LIBCG_SO" not in os.environ:
            try:
                from poke_bot.paths import cg_runtime_dir

                sib = cg_runtime_dir() / "cg" / "libcg.so"
                if sib.is_file():
                    os.environ["LIBCG_SO"] = str(sib.resolve())
            except Exception:
                pass

    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path.resolve()))
        except OSError:
            continue
        _bind(lib)
        if int(lib.StepBatchReady()) != 1:
            err = lib.StepBatchLastError()
            msg = err.decode() if isinstance(err, (bytes, bytearray)) else str(err)
            # Keep trying other candidates
            continue
        _STEP_BATCH_LIB = lib
        return lib
    _STEP_BATCH_LIB = None
    return None


def _bind(lib: Any) -> None:
    lib.StepBatchReady.restype = ctypes.c_int
    lib.StepBatchReady.argtypes = []
    lib.StepBatchLastError.restype = ctypes.c_char_p
    lib.StepBatchLastError.argtypes = []
    lib.StepBatch.restype = ctypes.c_int
    lib.StepBatch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,  # fetch_obs_on_skip
        ctypes.c_int,  # copy_json
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.StepBatchFreeJsons.restype = None
    lib.StepBatchFreeJsons.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
    ]


def step_batch_native(
    lib: Any,
    handles: Sequence[Optional[int]],
    actions: Sequence[Optional[Sequence[int]]],
    *,
    fetch_obs_on_skip: bool = False,
    copy_json: bool = False,
) -> tuple[list[int], list[Optional[str]], list[int]]:
    """Run C ``StepBatch``. Returns (errors, json_strings, select_players).

    Default ``copy_json=False`` matches stock ``GetBattleData`` lifetime: decode
    immediately before the next Select/Get/StepBatch on that handle.
    """
    n = len(handles)
    if len(actions) != n:
        raise ValueError("handles/actions length mismatch")

    handle_arr = (ctypes.c_void_p * n)(
        *[ctypes.c_void_p(h or 0) for h in handles]
    )
    flat: list[int] = []
    offsets = (ctypes.c_int * n)()
    lens = (ctypes.c_int * n)()
    for i, action in enumerate(actions):
        offsets[i] = len(flat)
        if action is None or handles[i] is None:
            lens[i] = 0
        else:
            lens[i] = len(action)
            flat.extend(int(x) for x in action)
    flat_arr = (ctypes.c_int * len(flat))(*flat) if flat else (ctypes.c_int * 0)()
    errors = (ctypes.c_int * n)()
    jsons = (ctypes.c_char_p * n)()
    select_players = (ctypes.c_int * n)()

    rc = int(
        lib.StepBatch(
            handle_arr,
            n,
            flat_arr,
            offsets,
            lens,
            1 if fetch_obs_on_skip else 0,
            1 if copy_json else 0,
            errors,
            jsons,
            select_players,
        )
    )
    if rc != 0:
        err = lib.StepBatchLastError()
        msg = err.decode() if isinstance(err, (bytes, bytearray)) else str(err)
        raise RuntimeError(f"StepBatch failed rc={rc}: {msg}")

    out_errors = [int(errors[i]) for i in range(n)]
    out_json: list[Optional[str]] = []
    out_sp = [int(select_players[i]) for i in range(n)]
    try:
        for i in range(n):
            raw = jsons[i]
            if not raw:
                out_json.append(None)
            else:
                out_json.append(raw.decode())
    finally:
        if copy_json:
            lib.StepBatchFreeJsons(jsons, n)
    return out_errors, out_json, out_sp


def has_step_batch() -> bool:
    return load_step_batch_lib() is not None
