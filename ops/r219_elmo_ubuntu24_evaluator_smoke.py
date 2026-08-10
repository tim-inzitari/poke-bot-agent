#!/usr/bin/env python3
"""One-process, non-game r219 Ubuntu 24 evaluator compatibility smoke.

It deliberately performs no action selection or evaluation game.  The native
seeded start/finalize is only the required ABI smoke, while ``main._ensure_runtime``
loads the frozen r195 model and verifies its submitted adapter activation path on
CPU.  A caller captures this program's single JSON object as the immutable
receipt material.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


PACKAGE = Path("/opt/pokebot/r195-package")
ENGINE = Path("/opt/pokebot/libcg_hidden_pristine_batch_b77afbd3.so")
EXPECTED = {
    "model.pt": "261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a",
    "matchup_tree.json": "e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049",
    "engine": "b77afbd363fe80de968c7cf20a0bbf5eb616fefcacbeab7eeeda94213fad9ea6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck() -> list[int]:
    cards = [
        int(line.strip().split(",")[0])
        for line in (PACKAGE / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(cards) != 60:
        raise RuntimeError(f"r195 deck count changed: {len(cards)}")
    return cards


class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


def max_glibcxx() -> str:
    library = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    versions = {
        match.decode("ascii")
        for match in re.findall(rb"GLIBCXX_[0-9.]+", library.read_bytes())
    }
    if not versions:
        raise RuntimeError("libstdc++ exposes no GLIBCXX version markers")
    return max(versions, key=lambda value: tuple(map(int, value.split("_")[1].split("."))))


def main() -> None:
    actual = {
        "model.pt": sha256(PACKAGE / "model.pt"),
        "matchup_tree.json": sha256(PACKAGE / "matchup_tree.json"),
        "engine": sha256(ENGINE),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"frozen r195/b77 identity mismatch: {actual}")

    # The smoke must not see or reserve a production GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["POKEBOT_LIBCG_PATH"] = str(ENGINE)
    os.environ["POKEBOT_SUBMISSION_SEARCH_DISABLE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(PACKAGE)

    engine = ctypes.CDLL(str(ENGINE))
    engine.GameInitialize.argtypes = []
    engine.GameInitialize.restype = None
    engine.BattleStartSeeded.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint32]
    engine.BattleStartSeeded.restype = StartData
    engine.BattleFinish.argtypes = [ctypes.c_void_p]
    engine.BattleFinish.restype = None
    engine.GameInitialize()
    cards = deck() * 2
    started = engine.BattleStartSeeded((ctypes.c_int * len(cards))(*cards), 219)
    if not started.battlePtr:
        raise RuntimeError(
            f"BattleStartSeeded failed: player={started.errorPlayer} type={started.errorType}"
        )
    engine.BattleFinish(started.battlePtr)

    import torch
    import main as submitted_main

    loaded_deck, model, policy = submitted_main._ensure_runtime()
    model_device = str(next(model.parameters()).device)
    if loaded_deck != deck():
        raise RuntimeError("submitted r195 deck changed during runtime load")
    if os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER") != "0":
        raise RuntimeError("NO-RTP submitted profile was not enforced")
    if getattr(policy, "matchup_adapter_runtime", None) is not True:
        raise RuntimeError("frozen r195 matchup adapter did not activate")

    libc = subprocess.check_output(
        ["getconf", "GNU_LIBC_VERSION"], text=True
    ).strip()
    receipt = {
        "schema": "poke_bot.r219_elmo_ubuntu24_evaluator_smoke/v1",
        "purpose": "one_process_import_and_seeded_abi_smoke_no_game_actions",
        "frozen_identity": actual,
        "seeded_engine": {
            "battle_start_seeded": True,
            "seed": 219,
            "native_battle_finalized_without_actions": True,
        },
        "r195_runtime": {
            "main_imported": str(Path(submitted_main.__file__).resolve()),
            "model_device": model_device,
            "matchup_adapter_runtime": True,
            "recursive_turn_planner": os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER"),
        },
        "platform": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "glibc": libc,
            "glibcxx_max": max_glibcxx(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
