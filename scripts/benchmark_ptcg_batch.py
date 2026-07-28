#!/usr/bin/env python3
"""Measure completed-game throughput for official and batched libcg paths."""

from __future__ import annotations

import argparse
import ctypes
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Optional

from poke_bot.engine_rebuild.interfaces import ResetSpec
from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv


class _StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class _SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


def _load_individual_library(path: Path) -> Any:
    """Load an optimized fork that preserves only the official handle ABI."""
    lib = ctypes.CDLL(str(path.expanduser().resolve()))
    lib.GameInitialize.restype = None
    lib.GameInitialize.argtypes = []
    lib.BattleStart.restype = _StartData
    lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.BattleFinish.restype = None
    lib.BattleFinish.argtypes = [ctypes.c_void_p]
    lib.GetBattleData.restype = _SerialData
    lib.GetBattleData.argtypes = [ctypes.c_void_p]
    lib.Select.restype = ctypes.c_int
    lib.Select.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.GameInitialize()
    return lib


def _deck(path: Path) -> list[int]:
    cards = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(cards)}")
    return cards


def _action(obs: dict) -> Optional[list[int]]:
    select = (obs or {}).get("select") or {}
    options = select.get("option") or []
    if not options:
        return None
    minimum = int(select.get("minCount") or 0)
    maximum = int(select.get("maxCount") or 1)
    count = max(minimum, min(maximum, len(options)))
    return list(range(count)) if count > 0 else []


def _run_games(
    env: Any,
    deck: list[int],
    *,
    games: int,
    num_envs: int,
    seed_start: int,
    max_steps: int,
) -> tuple[int, int]:
    completed = 0
    decisions = 0
    while completed < games:
        wave = min(num_envs, games - completed)
        specs = [
            ResetSpec(deck, deck, seed=seed_start + completed + i)
            for i in range(wave)
        ]
        batch = env.reset(specs)
        for _ in range(max_steps):
            if all(item.done for item in batch.envs[:wave]):
                break
            actions: list[Optional[list[int]]] = [None] * num_envs
            for i, item in enumerate(batch.envs[:wave]):
                if item.done:
                    continue
                action = _action(item.obs)
                if action is None:
                    raise RuntimeError(f"env {i} is live but exposes no action")
                actions[i] = action
                decisions += 1
            batch = env.step_batch(actions)
        else:
            unfinished = [i for i, item in enumerate(batch.envs[:wave]) if not item.done]
            raise RuntimeError(
                f"games exceeded {max_steps} decisions; unfinished={unfinished}"
            )
        completed += wave
    return completed, decisions


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _make_env(args: argparse.Namespace) -> Any:
    if args.backend == "official":
        if not args.cg_parent:
            raise ValueError("--cg-parent is required for the official backend")
        parent = str(args.cg_parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from cg import sim  # type: ignore

        return LibcgMultiEnv(args.num_envs, lib=sim.lib)
    if not args.lib:
        raise ValueError("--lib is required for fork backends")
    if args.backend == "fork-individual":
        return LibcgMultiEnv(args.num_envs, lib=_load_individual_library(args.lib))
    # Keep the proven official/fork-individual benchmarks deployable on workers
    # that intentionally do not carry the experimental private batch ABI.
    from poke_bot.engine_rebuild.libcg_batch import BatchedLibcgMultiEnv

    return BatchedLibcgMultiEnv(args.num_envs, lib_path=args.lib)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("official", "fork-individual", "batch"),
        required=True,
    )
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--cg-parent", type=Path)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--warmup-games", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()
    if args.num_envs < 1 or args.games < 1 or args.warmup_games < 0:
        parser.error("num-envs/games must be positive and warmup-games nonnegative")

    cards = _deck(args.deck)
    env = _make_env(args)
    try:
        if args.warmup_games:
            _run_games(
                env,
                cards,
                games=args.warmup_games,
                num_envs=args.num_envs,
                seed_start=0x10000000,
                max_steps=args.max_steps,
            )
        start = time.perf_counter()
        completed, decisions = _run_games(
            env,
            cards,
            games=args.games,
            num_envs=args.num_envs,
            seed_start=0x20000000,
            max_steps=args.max_steps,
        )
        elapsed = time.perf_counter() - start
    finally:
        env.close()

    report = {
        "backend": args.backend,
        "num_envs": args.num_envs,
        "games": completed,
        "decisions": decisions,
        "elapsed_s": elapsed,
        "games_per_s": completed / elapsed,
        "decisions_per_s": decisions / elapsed,
        "decisions_per_game": decisions / completed,
        "peak_rss_mib": _peak_rss_mib(),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
