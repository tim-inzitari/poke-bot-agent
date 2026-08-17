#!/usr/bin/env python3
"""Compare decoded Kaggle replays for Marnie policy checkpoints.

This analysis is descriptive.  It uses Kaggle's omniscient ``visualize``
timeline for board milestones, public episode metadata for outcome/seat, and
the final engine ``Result`` log for the terminal reason.  It does not infer
counterfactual action quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from analyze_kaggle_alakazam_replays import (
    _card_name,
    _episode_id,
    _load,
    _own_turn_rank,
    _rate,
    _timeline,
    analyze_replay as analyze_base_replay,
)


CORE_CHAIN = (
    "Marnie's Impidimp",
    "Marnie's Morgrem",
    "Marnie's Grimmsnarl ex",
)
SUPPORT_POKEMON = ("Marnie's Morpeko", "Snorunt", "Froslass", "Munkidori")
ATTACK_NAMES = {
    (646, 934): "Filch",
    (646, 935): "Corkscrew Punch",
    (647, 936): "Corkscrew Punch",
    (648, 937): "Shadow Bullet",
    (649, 938): "Spiky Wheel",
}
TERMINAL_NAMES = {1: "prize", 2: "deckout", 3: "board_wipe"}


def _wilson(wins: int, games: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if games <= 0:
        return None, None
    p = wins / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _average(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _median(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return median(values) if values else None


def _board_names(player: dict[str, Any], names: dict[int, str]) -> list[str]:
    return [
        _card_name(card, names)
        for card in list(player.get("active") or []) + list(player.get("bench") or [])
    ]


def analyze_replay(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    row = analyze_base_replay(path, meta)
    for key in (
        "first_seen_global_turn",
        "first_seen_own_turn",
        "alakazam_by_own_turn_2",
        "alakazam_by_own_turn_3",
        "max_alakazam",
        "max_psychic_energy_in_play",
        "own_turn_snapshots",
    ):
        row.pop(key, None)

    replay = _load(path)
    frames = _timeline(replay)
    own_index = int(row["own_index"])
    initial = frames[0]["current"]
    names: dict[int, str] = {}
    for player in initial["players"]:
        for card in player.get("deck", []):
            if "id" in card and "name" in card:
                names[int(card["id"])] = str(card["name"])

    own_turn_rank = _own_turn_rank(frames, own_index)
    tracked = CORE_CHAIN + SUPPORT_POKEMON
    first_seen_global = {name: None for name in tracked}
    first_seen_own = {name: None for name in tracked}
    evolution_events: list[dict[str, Any]] = []
    attack_events: list[dict[str, Any]] = []
    result_log: dict[str, Any] | None = None

    for frame in frames:
        current = frame.get("current") or {}
        turn = int(current.get("turn", 0) or 0)
        players = current.get("players") or []
        if own_index < len(players):
            board = _board_names(players[own_index], names)
            for card_name in tracked:
                if card_name in board and first_seen_global[card_name] is None:
                    first_seen_global[card_name] = turn
                    first_seen_own[card_name] = own_turn_rank.get(turn)

        for event in frame.get("logs") or []:
            if event.get("type") == "Result":
                result_log = dict(event)
                continue
            if int(event.get("playerIndex", -1)) != own_index:
                continue
            if event.get("type") == "Evolve":
                evolved_id = int(event.get("cardId", -1))
                target_id = int(event.get("cardIdTarget", -1))
                evolved_name = names.get(evolved_id, f"card_{evolved_id}")
                if evolved_name in CORE_CHAIN:
                    evolution_events.append(
                        {
                            "global_turn": turn,
                            "own_turn": own_turn_rank.get(turn),
                            "card_id": evolved_id,
                            "card": evolved_name,
                            "from_card_id": target_id,
                            "from_card": names.get(target_id, f"card_{target_id}"),
                            "direct_basic_to_stage2": bool(
                                evolved_name == "Marnie's Grimmsnarl ex"
                                and names.get(target_id) == "Marnie's Impidimp"
                            ),
                        }
                    )
            elif event.get("type") in ("Attack", 15):
                card_id = int(event.get("cardId", -1))
                attack_id = int(event.get("attackId", -1))
                attack_events.append(
                    {
                        "global_turn": turn,
                        "own_turn": own_turn_rank.get(turn),
                        "card_id": card_id,
                        "attacker": names.get(card_id, f"card_{card_id}"),
                        "attack_id": attack_id,
                        "attack": ATTACK_NAMES.get(
                            (card_id, attack_id), f"attack_{attack_id}"
                        ),
                    }
                )

    reason = int(result_log.get("reason", -1)) if result_log else -1
    winner = int(result_log.get("result", -1)) if result_log else -1
    grimmsnarl_evolutions = [
        event for event in evolution_events if event["card"] == "Marnie's Grimmsnarl ex"
    ]
    shadow_bullets = sum(event["attack"] == "Shadow Bullet" for event in attack_events)

    row.update(
        {
            "terminal_type": TERMINAL_NAMES.get(reason, "unknown"),
            "terminal_reason_raw": reason,
            "terminal_winner": winner,
            "first_seen_global_turn": first_seen_global,
            "first_seen_own_turn": first_seen_own,
            "impidimp_by_own_turn_1": first_seen_own["Marnie's Impidimp"] is not None
            and int(first_seen_own["Marnie's Impidimp"]) <= 1,
            "morgrem_by_own_turn_2": first_seen_own["Marnie's Morgrem"] is not None
            and int(first_seen_own["Marnie's Morgrem"]) <= 2,
            "grimmsnarl_by_own_turn_2": first_seen_own["Marnie's Grimmsnarl ex"] is not None
            and int(first_seen_own["Marnie's Grimmsnarl ex"]) <= 2,
            "grimmsnarl_by_own_turn_3": first_seen_own["Marnie's Grimmsnarl ex"] is not None
            and int(first_seen_own["Marnie's Grimmsnarl ex"]) <= 3,
            "first_grimmsnarl_own_turn": first_seen_own["Marnie's Grimmsnarl ex"],
            "evolution_events": evolution_events,
            "direct_basic_to_grimmsnarl": bool(
                grimmsnarl_evolutions
                and grimmsnarl_evolutions[0]["direct_basic_to_stage2"]
            ),
            "attack_events": attack_events,
            "shadow_bullet_count": shadow_bullets,
        }
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(bool(row["win"]) for row in rows)
    losses = [row for row in rows if not row["win"] and not row["draw"]]
    first = [row for row in rows if row["went_first"]]
    second = [row for row in rows if not row["went_first"]]
    low, high = _wilson(wins, games)
    return {
        "games": games,
        "wins": wins,
        "draws": sum(bool(row["draw"]) for row in rows),
        "win_rate": wins / games if games else None,
        "win_rate_wilson_95": [low, high],
        "first_seat": {"games": len(first), "wins": sum(row["win"] for row in first), "win_rate": _rate(first)},
        "second_seat": {"games": len(second), "wins": sum(row["win"] for row in second), "win_rate": _rate(second)},
        "terminal_counts": dict(sorted(Counter(row["terminal_type"] for row in rows).items())),
        "loss_terminal_counts": dict(sorted(Counter(row["terminal_type"] for row in losses).items())),
        "impidimp_by_own_turn_1_rate": _rate(rows, "impidimp_by_own_turn_1"),
        "morgrem_by_own_turn_2_rate": _rate(rows, "morgrem_by_own_turn_2"),
        "grimmsnarl_by_own_turn_2_rate": _rate(rows, "grimmsnarl_by_own_turn_2"),
        "grimmsnarl_by_own_turn_3_rate": _rate(rows, "grimmsnarl_by_own_turn_3"),
        "median_first_grimmsnarl_own_turn": _median(rows, "first_grimmsnarl_own_turn"),
        "direct_basic_to_grimmsnarl_rate": _rate(rows, "direct_basic_to_grimmsnarl"),
        "mean_attack_count": _average(rows, "own_attack_count"),
        "mean_shadow_bullet_count": _average(rows, "shadow_bullet_count"),
        "mean_own_prizes_taken": _average(rows, "own_prizes_taken"),
        "mean_opponent_prizes_taken": _average(rows, "opponent_prizes_taken"),
        "median_elapsed_s": _median(rows, "elapsed_s"),
        "median_own_turns": _median(rows, "own_turns"),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    iterations = sorted({int(row["iteration"]) for row in rows})
    archetypes = sorted({str(row["opponent_archetype"]) for row in rows})
    by_iteration = {
        str(iteration): summarize([row for row in rows if row["iteration"] == iteration])
        for iteration in iterations
    }
    by_iteration_archetype: dict[str, Any] = {}
    for iteration in iterations:
        for archetype in archetypes:
            group = [
                row
                for row in rows
                if row["iteration"] == iteration and row["opponent_archetype"] == archetype
            ]
            if group:
                by_iteration_archetype[f"iter_{iteration:05d}:{archetype}"] = summarize(group)

    matched: dict[str, Any] = {}
    if len(iterations) == 2:
        left, right = iterations
        for archetype in archetypes:
            left_rows = [
                row for row in rows if row["iteration"] == left and row["opponent_archetype"] == archetype
            ]
            right_rows = [
                row for row in rows if row["iteration"] == right and row["opponent_archetype"] == archetype
            ]
            if not left_rows or not right_rows:
                continue
            left_summary = summarize(left_rows)
            right_summary = summarize(right_rows)
            matched[archetype] = {
                f"iteration_{left}": left_summary,
                f"iteration_{right}": right_summary,
                "win_rate_delta": right_summary["win_rate"] - left_summary["win_rate"],
                "sample_warning": bool(len(left_rows) < 10 or len(right_rows) < 10),
            }
    return {
        "by_iteration": by_iteration,
        "by_iteration_archetype": by_iteration_archetype,
        "matched_archetype_comparison": matched,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id", "iteration", "submission_id", "own_index", "went_first",
        "win", "draw", "opponent_team", "opponent_submission_id", "opponent_archetype",
        "terminal_type", "terminal_reason_raw", "elapsed_s", "max_global_turn", "own_turns",
        "own_attack_count", "shadow_bullet_count", "impidimp_by_own_turn_1",
        "morgrem_by_own_turn_2", "grimmsnarl_by_own_turn_2", "grimmsnarl_by_own_turn_3",
        "first_grimmsnarl_own_turn", "direct_basic_to_grimmsnarl", "own_prizes_taken",
        "opponent_prizes_taken", "final_own_deck_count", "final_opponent_deck_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    source = _load(args.replay_root / "episodes.json")
    metadata = {int(row["episode_id"]): row for row in source["episodes"]}
    replay_paths = sorted(args.replay_root.glob("iter_*/*-replay.json"))
    rows = [analyze_replay(path, metadata[_episode_id(path)]) for path in replay_paths]
    if len(rows) != len(metadata):
        raise RuntimeError(
            f"replay/metadata mismatch: replays={len(rows)} metadata={len(metadata)}"
        )

    payload = {
        "schema": "poke_bot.kaggle_marnie_replay_analysis/v1",
        "source": {
            "replay_root": str(args.replay_root),
            "episodes_json": str(args.replay_root / "episodes.json"),
            "download_receipt": str(args.replay_root / "DOWNLOAD_RECEIPT.json"),
            "submissions": source.get("submissions"),
        },
        "definitions": {
            "went_first": "decoded replay firstPlayer equals own agent index",
            "terminal_type": "final engine Result reason: 1=prize, 2=deckout, 3=board wipe",
            "setup_timing": "first omniscient visualize frame where the card is active or benched",
            "win_rate_wilson_95": "uncorrected two-sided 95% Wilson interval",
        },
        "limitations": [
            "Public opponent mix is uncontrolled and differs between submissions.",
            "Visualize timelines are omniscient and support description, not causal action attribution.",
            "Small archetype/seat cohorts are flagged and should not drive policy changes alone.",
        ],
        "summary": aggregate(rows),
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, rows)
    print(json.dumps({"episodes": len(rows), "output": str(args.output), "csv": str(args.csv)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
