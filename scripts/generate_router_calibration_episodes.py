#!/usr/bin/env python3
"""Generate causal public-prefix calibration episodes for sparse archetypes.

The opponent label comes only from the pinned representative deck selected
before simulation.  Stored observations contain the acting seat index and the
two players' public active/bench/discard zones; hand, deck, prizes, agent IDs,
and future observations are never written.  A later independent train/val
split and the unchanged router audit decide whether any route is usable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import zipfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TARGETS = (
    "dragapult-blaziken",
    "dragapult-dusknoir",
    "dudunsparce",
    "gardevoir",
    "ns-zoroark",
)


def _public_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    current = obs.get("current")
    if not isinstance(current, Mapping):
        return {"current": None}
    players = current.get("players")
    if not isinstance(players, (list, tuple)) or len(players) != 2:
        return {"current": None}
    public_players = []
    for player in players:
        if not isinstance(player, Mapping):
            public_players.append({})
            continue
        public_players.append(
            {
                zone: player.get(zone) if isinstance(player.get(zone), list) else []
                for zone in ("active", "bench", "discard")
            }
        )
    return {
        "current": {
            "yourIndex": int(current.get("yourIndex", 0)),
            "players": public_players,
        }
    }


def _setup_steps(decks: tuple[list[int], list[int]]) -> list[list[dict[str, Any]]]:
    """Match the official replay setup shape consumed by ``extract_setup_decks``."""

    return [
        [
            {
                "action": [],
                "observation": {"current": None},
                "visualize": [{"action": [decks[0], decks[1]]}],
            },
            {"action": [], "observation": {"current": None}},
        ],
        [
            {"action": list(decks[0]), "observation": {"current": None}},
            {"action": list(decks[1]), "observation": {"current": None}},
        ],
    ]


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    from poke_bot.agent import play_game
    from poke_bot.cg_env import random_legal_select

    target = str(job["target"])
    target_deck = [int(value) for value in job["target_deck"]]
    opponent_deck = [int(value) for value in job["opponent_deck"]]
    target_seat = int(job["target_seat"])
    seed = int(job["seed"])
    if len(target_deck) != 60 or len(opponent_deck) != 60:
        raise RuntimeError("router calibration decks must contain 60 cards")
    decks = (
        (target_deck, opponent_deck)
        if target_seat == 0
        else (opponent_deck, target_deck)
    )
    facing_seat = 1 - target_seat
    captured: list[dict[str, Any]] = []
    rng0 = random.Random(seed * 2 + 1)
    rng1 = random.Random(seed * 2 + 2)

    def agent(seat: int, rng: random.Random):
        def act(obs: dict[str, Any]) -> list[int]:
            if seat == facing_seat:
                captured.append(_public_observation(obs))
            return random_legal_select(obs, rng)

        return act

    outcome = play_game(
        agent(0, rng0),
        agent(1, rng1),
        decks[0],
        decks[1],
        max_steps=4000,
    )
    steps = _setup_steps(decks)
    for observation in captured:
        rows = [{"observation": {"current": None}}, {"observation": {"current": None}}]
        rows[facing_seat] = {"observation": observation}
        steps.append(rows)
    episode_id = f"router-{target}-{seed:09d}"
    return {
        "id": episode_id,
        "info": {"EpisodeId": episode_id},
        "configuration": {"seed": seed},
        "rewards": [
            1 if int(outcome["winner"]) == 0 else -1,
            1 if int(outcome["winner"]) == 1 else -1,
        ],
        "statuses": ["DONE", "DONE"],
        "steps": steps,
        "router_calibration": {
            "schema": "poke_bot.router_calibration_episode/v1",
            "target_archetype": target,
            "target_seat": target_seat,
            "causal_public_zones_only": True,
            "hidden_fields_written": False,
            "game_completed": not bool(outcome.get("incomplete")),
            "failed_seat": outcome.get("failed_seat"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument(
        "--target-deck",
        action="append",
        default=[],
        metavar="ARCHETYPE=CSV",
        help="Pinned 60-card deck override for targets absent from representatives.",
    )
    parser.add_argument("--opponent", default="starmie")
    parser.add_argument("--games-per-target", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=735_100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if int(args.games_per_target) <= 0 or int(args.workers) <= 0:
        raise ValueError("games and workers must be positive")
    payload = json.loads(args.representatives.read_text(encoding="utf-8"))
    decks = dict(payload.get("decks") or {})
    for raw in args.target_deck:
        archetype, separator, path = str(raw).partition("=")
        if not separator or not archetype or not path:
            raise ValueError("--target-deck requires ARCHETYPE=CSV")
        card_ids = [
            int(value)
            for value in Path(path).read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if len(card_ids) != 60:
            raise RuntimeError(
                f"router target deck {archetype} has {len(card_ids)} cards"
            )
        decks[archetype] = {"card_ids": card_ids}
    targets = tuple(dict.fromkeys(args.target or DEFAULT_TARGETS))
    if args.opponent not in decks or any(target not in decks for target in targets):
        raise RuntimeError("requested router calibration representative is missing")
    jobs = []
    for target_index, target in enumerate(targets):
        for game_index in range(int(args.games_per_target)):
            jobs.append(
                {
                    "target": target,
                    "target_deck": list(decks[target]["card_ids"]),
                    "opponent_deck": list(decks[args.opponent]["card_ids"]),
                    "target_seat": game_index % 2,
                    "seed": int(args.seed)
                    + target_index * int(args.games_per_target)
                    + game_index,
                }
            )
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(fd)
    temporary = Path(raw)
    try:
        completed = 0
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=3,
        ) as archive:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(args.workers)
            ) as pool:
                futures = [pool.submit(_worker, job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    episode = future.result()
                    name = str(episode["id"]) + ".json"
                    archive.writestr(
                        name,
                        json.dumps(
                            episode,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    )
                    completed += 1
                    if completed % 100 == 0 or completed == len(jobs):
                        print(
                            f"router calibration: {completed}/{len(jobs)} games",
                            flush=True,
                        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "schema": "poke_bot.router_calibration_corpus/v1",
                "output": str(output),
                "targets": list(targets),
                "games_per_target": int(args.games_per_target),
                "games_total": len(jobs),
                "causal_public_zones_only": True,
                "hidden_fields_written": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
