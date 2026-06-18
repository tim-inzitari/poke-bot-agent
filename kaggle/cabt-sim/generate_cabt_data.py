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


def read_deck_csv() -> list[int]:
    candidates = [
        Path("deck.csv"),
        Path("submission/deck.csv"),
        Path("/kaggle/working/deck.csv"),
        Path("/kaggle_simulations/agent/deck.csv"),
    ]
    for path in candidates:
        if path.exists():
            values = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
            if len(values) != 60:
                raise ValueError(f"{path} must contain exactly 60 card IDs")
            return values
    return SAMPLE_DECK


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


def play_episode(episode: int, max_steps: int, deck: list[int]) -> list[dict[str, Any]]:
    from cg.game import battle_finish, battle_select, battle_start

    rows: list[dict[str, Any]] = []
    obs, start_data = battle_start(deck, deck)
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
            })
            obs = battle_select(random_agent(obs))
            step += 1

        result = int(obs["current"]["result"])
        for row in rows:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out", default="data/cabt_rollouts.jsonl")
    args = parser.parse_args()

    cg_path = add_cg_lib_to_path()
    deck = read_deck_csv()
    rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        rows.extend(play_episode(episode, args.max_steps, deck))
    write_jsonl(Path(args.out), rows)
    print(f"cg-lib={cg_path}")
    print(f"deck_cards={len(deck)}")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
