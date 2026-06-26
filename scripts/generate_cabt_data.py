#!/usr/bin/env python3
"""Generate CABT rollout data in a Linux/Kaggle environment.

The CABT simulator depends on `cg/libcg.so`, which is a Linux shared library.
Run this script on Kaggle by default, or in the optional Linux container.
It writes JSONL transition rows that the notebook can consume for local
Torch/MPS experiments later.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.archetypes import weighted_deck_pool, slug_from_deck_name, load_archetype_registry

SAMPLE_DECK = [
    119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121,
    305, 305, 66, 66, 112, 112, 235, 1071, 140, 1227,
    1227, 1227, 1227, 1182, 1182, 1182, 1198, 1198, 1240, 1213,
    1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1121, 1121,
    1121, 1121, 1120, 1120, 1120, 1120, 1097, 1097, 1080, 1260,
    1260, 2, 2, 2, 5, 5, 5, 7, 7,
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
        Path("decks/submission.csv"),
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


from poke_agent.features import features_from_observation
from poke_agent.game_tracker import GameEventTracker
from poke_agent.rewards import assign_episode_values, is_complete_episode


def random_agent(obs_dict: dict[str, Any]) -> list[int]:
    from cg.api import to_observation_class

    obs = to_observation_class(obs_dict)
    options = list(range(len(obs.select.option)))
    return random.sample(options, min(obs.select.maxCount, len(options)))


def json_snapshot(value: Any) -> Any:
    """Deep-copy simulator output while proving it can be written as JSON."""
    return json.loads(json.dumps(value, separators=(",", ":")))


def choose_matchup(
    episode: int,
    deck0_pool: list[tuple[str, list[int]]],
    deck1_pool: list[tuple[str, list[int]]],
    mode: str,
    *,
    weighted0: list[tuple[str, list[int], float]] | None = None,
    weighted1: list[tuple[str, list[int], float]] | None = None,
) -> tuple[str, list[int], str, list[int]]:
    if mode == "round-robin":
        i = episode % len(deck0_pool)
        j = (episode // len(deck0_pool)) % len(deck1_pool)
        name0, deck0 = deck0_pool[i]
        name1, deck1 = deck1_pool[j]
    elif weighted0 and weighted1:
        name0, deck0, _ = random.choices(
            weighted0, weights=[item[2] for item in weighted0], k=1
        )[0]
        name1, deck1, _ = random.choices(
            weighted1, weights=[item[2] for item in weighted1], k=1
        )[0]
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
    *,
    rewards: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    from cg.game import battle_finish, battle_select, battle_start

    rows: list[dict[str, Any]] = []
    tracker = GameEventTracker()
    reward_cfg = rewards or {}
    registry = load_archetype_registry(ROOT)
    deck0_slug = slug_from_deck_name(deck0_name, registry)
    deck1_slug = slug_from_deck_name(deck1_name, registry)
    obs, start_data = battle_start(deck0, deck1)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"deck error type={start_data.errorType} player={start_data.errorPlayer}")

    try:
        step = 0
        truncated = False
        while obs["current"]["result"] < 0 and step < max_steps:
            select = obs.get("select") or {}
            options = select.get("option") or []
            action = random_agent(obs)
            next_obs = battle_select(action)
            terminal = int((next_obs.get("current") or {}).get("result", -1)) >= 0
            step_features, _ = features_from_observation(obs, tracker)
            next_tracker = copy.deepcopy(tracker)
            step_next_features, _ = features_from_observation(next_obs, next_tracker)
            step_features = [float(v) for v in step_features]
            step_next_features = [float(v) for v in step_next_features]
            rows.append({
                "episode": episode,
                "step": step,
                "features": step_features,
                "next_features": step_next_features,
                "observation": json_snapshot(obs),
                "action": json_snapshot(action),
                "next_observation": json_snapshot(next_obs),
                "legal_action_count": len(options),
                "select_min_count": int(select.get("minCount", 0)),
                "select_max_count": int(select.get("maxCount", 0)),
                "terminal": terminal,
                "reward": 0.0,
                "player": int(obs["current"]["yourIndex"]),
                "deck0": deck0_slug,
                "deck1": deck1_slug,
                "deck0_cards": list(deck0),
                "deck1_cards": list(deck1),
                "source": "multideck-cabt",
                "source_episode_id": str(episode),
            })
            obs = next_obs
            step += 1

        if obs["current"]["result"] < 0:
            truncated = True
        result = int(obs["current"]["result"])
        if truncated and result < 0:
            return []
        if not is_complete_episode(result, terminal_obs=obs, truncated=truncated):
            return []
        assign_episode_values(
            rows,
            result,
            terminal_obs=obs,
            value_win=float(reward_cfg.get("value_win", 1.0)),
            value_not_win=float(reward_cfg.get("value_not_win", -1.0)),
            value_timeout=float(reward_cfg.get("value_timeout", -2.0)),
            value_per_own_prize_taken=float(reward_cfg.get("value_per_own_prize_taken", 1.0 / 6)),
            value_per_opp_prize_taken=float(reward_cfg.get("value_per_opp_prize_taken", -1.0 / 6)),
        )
        for row in rows:
            row["complete"] = True
            row["truncated"] = False
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
        list[tuple[str, list[int], float]] | None,
        list[tuple[str, list[int], float]] | None,
        dict[str, float],
    ],
) -> tuple[int, list[dict[str, Any]]]:
    start, stop, max_steps, deck0_pool, deck1_pool, matchup_mode, seed, weighted0, weighted1, reward_cfg = args
    if seed is not None:
        random.seed(seed + start)
    add_cg_lib_to_path()
    rows: list[dict[str, Any]] = []
    target = stop - start
    complete = 0
    attempt = start
    while complete < target and attempt < start + target * 5:
        deck0_name, deck0, deck1_name, deck1 = choose_matchup(
            attempt,
            deck0_pool,
            deck1_pool,
            matchup_mode,
            weighted0=weighted0,
            weighted1=weighted1,
        )
        episode_rows = play_episode(
            start + complete,
            max_steps,
            deck0_name,
            deck0,
            deck1_name,
            deck1,
            rewards=reward_cfg,
        )
        if episode_rows:
            rows.extend(episode_rows)
            complete += 1
        attempt += 1
    return start, rows


def episode_chunks(episodes: int, workers: int) -> list[tuple[int, int]]:
    chunk_size = max(1, math.ceil(episodes / workers))
    return [(start, min(episodes, start + chunk_size)) for start in range(0, episodes, chunk_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="number of CABT games (default: DATASET_GAMES env or 10)",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel CABT workers (0 = auto from CPU count)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deck-dir", default=None)
    parser.add_argument("--deck0-dir", default=None)
    parser.add_argument("--deck1-dir", default=None)
    parser.add_argument("--matchups", choices=["sample", "round-robin", "weighted"], default="weighted")
    parser.add_argument("--shares", default="decks/archetype-shares.txt")
    parser.add_argument("--out", default="data/multideck_rollouts.jsonl")
    args = parser.parse_args()
    if args.episodes is None:
        if os.environ.get("DATASET_GAMES"):
            parsed = int(os.environ["DATASET_GAMES"])
            args.episodes = 0 if parsed <= 0 else parsed
        else:
            args.episodes = 10

    cg_path = add_cg_lib_to_path()
    deck0_pool = read_deck_pool(args.deck0_dir or args.deck_dir or "decks/archetype-samples")
    deck1_pool = read_deck_pool(args.deck1_dir or args.deck_dir or "decks/archetype-samples")
    weighted0 = weighted1 = None
    if args.matchups == "weighted":
        shares = ROOT / args.shares
        weighted0 = weighted_deck_pool(ROOT, shares_path=shares)
        weighted1 = weighted_deck_pool(ROOT, shares_path=shares)
    reward_cfg = {
        "value_win": float(os.environ.get("VALUE_WIN", "1.0")),
        "value_not_win": float(os.environ.get("VALUE_NOT_WIN", "-1.0")),
        "value_timeout": float(os.environ.get("VALUE_TIMEOUT", "-2.0")),
    }
    from poke_agent.data_pipeline import default_cabt_generation_workers

    worker_count = args.workers if args.workers > 0 else default_cabt_generation_workers(episodes=args.episodes)
    workers = max(1, min(worker_count, args.episodes))
    complete_games = 0
    if workers == 1:
        rows: list[dict[str, Any]] = []
        if args.seed is not None:
            random.seed(args.seed)
        attempt = 0
        while complete_games < args.episodes and attempt < args.episodes * 5:
            deck0_name, deck0, deck1_name, deck1 = choose_matchup(
                attempt, deck0_pool, deck1_pool, args.matchups,
                weighted0=weighted0, weighted1=weighted1,
            )
            episode_rows = play_episode(
                complete_games, args.max_steps, deck0_name, deck0, deck1_name, deck1,
                rewards=reward_cfg,
            )
            if episode_rows:
                rows.extend(episode_rows)
                complete_games += 1
            attempt += 1
    else:
        # Cap chunk size so there are many more tasks than workers — imap_unordered then
        # yields finished chunks steadily, giving a live progress bar instead of one silent
        # block until all games are done.
        chunk_cap = max(1, min(25, math.ceil(args.episodes / workers)))
        chunks = [
            (start, min(args.episodes, start + chunk_cap))
            for start in range(0, args.episodes, chunk_cap)
        ]
        tasks = [
            (
                start,
                stop,
                args.max_steps,
                deck0_pool,
                deck1_pool,
                args.matchups,
                args.seed,
                weighted0,
                weighted1,
                reward_cfg,
            )
            for start, stop in chunks
        ]
        from tqdm.auto import tqdm

        collected: list[tuple[int, list[dict[str, Any]]]] = []
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            with tqdm(total=args.episodes, desc="cabt games", unit="game") as progress:
                for start, chunk_rows in pool.imap_unordered(run_episode_range, tasks):
                    collected.append((start, chunk_rows))
                    done = len({int(row["episode"]) for row in chunk_rows})
                    progress.update(done)
        rows = []
        for _, chunk_rows in sorted(collected, key=lambda item: item[0]):
            rows.extend(chunk_rows)
    write_jsonl(Path(args.out), rows)
    print(f"cg-lib={cg_path}")
    print(f"deck0_pool={len(deck0_pool)}")
    print(f"deck1_pool={len(deck1_pool)}")
    print(f"matchups={args.matchups}")
    print(f"games={args.episodes}")
    print(f"workers={workers}")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
