#!/usr/bin/env python3
"""Generate CABT rollout data in a Linux/Kaggle environment.

The CABT simulator depends on `cg/libcg.so`, which is a Linux shared library.
Run this script on Kaggle by default, or in the optional Linux container.
It writes simple JSONL rows that the notebook can consume for local Torch/MPS
experiments later.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Any


SAMPLE_DECK = [
    721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
    1092, 1121, 1121, 1145, 1145, 1163, 1163,
    1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227,
    1262, 1262,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3,
]


def read_deck_file(path: Path) -> list[int]:
    values = []
    for raw_token in path.read_text().replace(",", "\n").splitlines():
        token = raw_token.strip()
        if token:
            values.append(int(token))
    if len(values) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs")
    return values


def read_default_deck() -> tuple[str, list[int]]:
    candidates = [
        Path("deck.csv"),
        Path("submission/deck.csv"),
        Path("/kaggle/working/deck.csv"),
        Path("/kaggle_simulations/agent/deck.csv"),
    ]
    for path in candidates:
        if path.exists():
            return (path.stem, read_deck_file(path))
    return ("sample", SAMPLE_DECK)


def read_deck_pool(path: str | None) -> list[tuple[str, list[int]]]:
    if not path:
        return [read_default_deck()]

    root = Path(path)
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".deck"}
    )
    if not files:
        raise FileNotFoundError(f"no deck files found in {root}")
    return [(p.stem, read_deck_file(p)) for p in files]


def add_cg_lib_to_path() -> str:
    candidates: list[str] = []
    if os.environ.get("CG_LIB_PATH"):
        candidates.append(os.environ["CG_LIB_PATH"])
    candidates.extend(glob.glob("/kaggle/input/**/cg-lib", recursive=True))
    candidates.extend(glob.glob(os.path.join(os.getcwd(), "kaggle/input/**/cg-lib"), recursive=True))
    if not candidates:
        raise FileNotFoundError(
            "Could not find cg-lib. On Kaggle, attach kiyotah/cg-lib. "
            "Locally, run scripts/download-kaggle-inputs.sh or set CG_LIB_PATH."
        )
    sys.path.append(candidates[0])
    return candidates[0]


def random_agent(obs_dict: dict[str, Any]) -> list[int]:
    from cg.api import to_observation_class

    obs = to_observation_class(obs_dict)
    options = list(range(len(obs.select.option)))
    return random.sample(options, min(obs.select.maxCount, len(options)))


def features_from_observation(obs: dict[str, Any]) -> list[float]:
    current = obs.get("current") or {}
    players = current.get("players") or [{}, {}]
    p0 = players[0] if len(players) > 0 else {}
    p1 = players[1] if len(players) > 1 else {}
    select = obs.get("select") or {}
    return [
        float(current.get("turn", 0)),
        float(current.get("yourIndex", 0)),
        float(p0.get("deckCount", 0)),
        float(p0.get("handCount", 0)),
        float(len(p0.get("bench", []))),
        float(p1.get("deckCount", 0)),
        float(p1.get("handCount", 0)),
        float(len(p1.get("bench", []))),
        float(len(select.get("option", []))),
        float(select.get("maxCount", 0)),
    ]


def choose_matchup(
    episode: int,
    deck0_pool: list[tuple[str, list[int]]],
    deck1_pool: list[tuple[str, list[int]]],
    mode: str,
) -> tuple[str, list[int], str, list[int]]:
    if mode == "round-robin":
        i = episode % len(deck0_pool)
        j = (episode // len(deck0_pool)) % len(deck1_pool)
        name0, deck0 = deck0_pool[i]
        name1, deck1 = deck1_pool[j]
    else:
        name0, deck0 = random.choice(deck0_pool)
        name1, deck1 = random.choice(deck1_pool)
    return name0, deck0, name1, deck1


def play_episode(
    episode: int,
    max_steps: int,
    deck0_name: str,
    deck0: list[int],
    deck1_name: str,
    deck1: list[int],
) -> list[dict[str, Any]]:
    from cg.game import battle_finish, battle_select, battle_start

    rows: list[dict[str, Any]] = []
    obs, start_data = battle_start(deck0, deck1)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"deck error type={start_data.errorType} player={start_data.errorPlayer}")

    try:
        step = 0
        while obs["current"]["result"] < 0 and step < max_steps:
            rows.append({
                "episode": episode,
                "step": step,
                "features": features_from_observation(obs),
                "player": int(obs["current"]["yourIndex"]),
                "deck0": deck0_name,
                "deck1": deck1_name,
            })
            obs = battle_select(random_agent(obs))
            step += 1

        result = int(obs["current"]["result"])
        for row in rows:
            row["result"] = result
            if result == 2:
                row["value"] = 0.0
            else:
                row["value"] = 1.0 if row["player"] == result else -1.0
        return rows
    finally:
        battle_finish()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def run_episode_range(
    args: tuple[
        int,
        int,
        int,
        list[tuple[str, list[int]]],
        list[tuple[str, list[int]]],
        str,
        int | None,
    ]
) -> tuple[int, list[dict[str, Any]]]:
    start, stop, max_steps, deck0_pool, deck1_pool, matchup_mode, seed = args
    if seed is not None:
        random.seed(seed + start)
    add_cg_lib_to_path()
    rows: list[dict[str, Any]] = []
    for episode in range(start, stop):
        deck0_name, deck0, deck1_name, deck1 = choose_matchup(episode, deck0_pool, deck1_pool, matchup_mode)
        rows.extend(play_episode(episode, max_steps, deck0_name, deck0, deck1_name, deck1))
    return start, rows


def episode_chunks(episodes: int, workers: int) -> list[tuple[int, int]]:
    chunk_size = max(1, math.ceil(episodes / workers))
    return [(start, min(episodes, start + chunk_size)) for start in range(0, episodes, chunk_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deck-dir", default=None)
    parser.add_argument("--deck0-dir", default=None)
    parser.add_argument("--deck1-dir", default=None)
    parser.add_argument("--matchups", choices=["sample", "round-robin"], default="sample")
    parser.add_argument("--out", default="data/cabt_rollouts.jsonl")
    args = parser.parse_args()

    cg_path = add_cg_lib_to_path()
    deck0_pool = read_deck_pool(args.deck0_dir or args.deck_dir)
    deck1_pool = read_deck_pool(args.deck1_dir or args.deck_dir)
    workers = max(1, min(args.workers, args.episodes))
    if workers == 1:
        rows: list[dict[str, Any]] = []
        if args.seed is not None:
            random.seed(args.seed)
        for episode in range(args.episodes):
            deck0_name, deck0, deck1_name, deck1 = choose_matchup(
                episode, deck0_pool, deck1_pool, args.matchups
            )
            rows.extend(play_episode(episode, args.max_steps, deck0_name, deck0, deck1_name, deck1))
    else:
        tasks = [
            (start, stop, args.max_steps, deck0_pool, deck1_pool, args.matchups, args.seed)
            for start, stop in episode_chunks(args.episodes, workers)
        ]
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            results = pool.map(run_episode_range, tasks)
        rows = []
        for _, chunk_rows in sorted(results, key=lambda item: item[0]):
            rows.extend(chunk_rows)
    write_jsonl(Path(args.out), rows)
    print(f"cg-lib={cg_path}")
    print(f"deck0_pool={len(deck0_pool)}")
    print(f"deck1_pool={len(deck1_pool)}")
    print(f"matchups={args.matchups}")
    print(f"episodes={args.episodes}")
    print(f"workers={workers}")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
