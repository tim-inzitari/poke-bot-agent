#!/usr/bin/env python3
"""Aggregate day-level Slowking replay receipts without mixing deck lineages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BEHAVIOR_FIELDS = (
    "initial_active",
    "initial_bench",
    "main_action_types",
    "played_cards",
    "attached_energy",
    "attachment_targets",
    "evolved_cards",
    "ability_sources",
    "academy_targets",
    "seek_sources",
    "seek_copied_attack_ids",
    "attack_users",
    "top_turn_bigrams",
    "top_turn_trigrams",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{value}"


def wilson(wins: int, games: int, z: float = 1.959963984540054) -> list[float] | None:
    if games == 0:
        return None
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denominator
    return [center - half, center + half]


def outcome_summary(games: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(game["result"] == "win" for game in games)
    losses = sum(game["result"] == "loss" for game in games)
    draws = len(games) - wins - losses
    turn_order: dict[str, dict[str, Any]] = {}
    for order in ("first", "second", "unknown"):
        rows = [game for game in games if game["turn_order"] == order]
        order_wins = sum(game["result"] == "win" for game in rows)
        turn_order[order] = {
            "games": len(rows),
            "wins": order_wins,
            "win_rate": order_wins / len(rows) if rows else None,
            "wilson_95": wilson(order_wins, len(rows)),
        }
    return {
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(games) if games else None,
        "wilson_95": wilson(wins, len(games)),
        "turn_order": turn_order,
    }


def merge_counter_rows(receipts: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str | None]] = Counter()
    for receipt in receipts:
        for row in receipt["behavior"].get(field, []):
            counts[(str(row["key"]), row.get("card_name"))] += int(row["count"])
    total = sum(counts.values())
    rows = [
        {
            "key": key,
            "card_name": name,
            "count": count,
            "fraction": count / total if total else None,
        }
        for (key, name), count in counts.items()
    ]
    rows.sort(key=lambda row: (-row["count"], row["key"], row["card_name"] or ""))
    return rows


def merge_effect_selected(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        for effect_key, effect in receipt["behavior"].get("effect_selected_cards", {}).items():
            target = merged.setdefault(
                effect_key,
                {"effect_card_name": effect.get("effect_card_name"), "counts": Counter()},
            )
            for row in effect.get("targets", []):
                target["counts"][(str(row["key"]), row.get("card_name"))] += int(row["count"])
    output = {}
    for effect_key, effect in merged.items():
        total = sum(effect["counts"].values())
        targets = [
            {
                "key": key,
                "card_name": name,
                "count": count,
                "fraction": count / total if total else None,
            }
            for (key, name), count in effect["counts"].items()
        ]
        targets.sort(key=lambda row: (-row["count"], row["key"], row["card_name"] or ""))
        output[effect_key] = {
            "effect_card_name": effect["effect_card_name"],
            "targets": targets,
        }
    return output


def merge_behavior(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    output = {field: merge_counter_rows(receipts, field) for field in BEHAVIOR_FIELDS}
    output["effect_selected_cards"] = merge_effect_selected(receipts)
    return output


def main() -> None:
    args = parse_args()
    inputs = []
    for path in args.daily:
        receipt = json.loads(path.read_text())
        inputs.append((receipt["source"]["date"], path, receipt))
    inputs.sort(key=lambda row: row[0])

    all_games: list[dict[str, Any]] = []
    lineage_games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lineage_receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    team_games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    deck_rows: dict[str, dict[str, Any]] = {}
    daily_rows = []

    for date, path, receipt in inputs:
        games = [{**game, "date": date} for game in receipt["games"]]
        all_games.extend(games)
        for game in games:
            lineage_games[game["deck_fingerprint"]].append(game)
            team_games[game["team_name"]].append(game)
        for deck in receipt["identity"]["decks"]:
            lineage_receipts[deck["fingerprint"]].append(receipt)
            prior = deck_rows.setdefault(
                deck["fingerprint"],
                {
                    "fingerprint": deck["fingerprint"],
                    "cards": deck["cards"],
                    "dates": [],
                    "teams": set(),
                },
            )
            if prior["cards"] != deck["cards"]:
                raise ValueError(f"card-list conflict for {deck['fingerprint']}")
            prior["dates"].append(date)
            prior["teams"].update(receipt["identity"]["unique_team_names"])
        daily_rows.append(
            {
                "date": date,
                "receipt": str(path),
                "receipt_sha256": digest(path),
                "manifest_episode_count": receipt["source"].get("manifest_episode_count"),
                "archive_json_members": receipt["source"].get("archive_json_members"),
                **outcome_summary(games),
                "teams": receipt["identity"]["unique_team_names"],
                "fingerprints": [deck["fingerprint"] for deck in receipt["identity"]["decks"]],
            }
        )

    lineages = []
    for fingerprint, games in lineage_games.items():
        deck = deck_rows[fingerprint]
        lineages.append(
            {
                "fingerprint": fingerprint,
                "dates": sorted(set(deck["dates"])),
                "teams": sorted(deck["teams"]),
                "cards": deck["cards"],
                "behavior": merge_behavior(lineage_receipts[fingerprint]),
                **outcome_summary(games),
            }
        )
    lineages.sort(key=lambda row: (row["dates"][0], row["fingerprint"]))

    teams = []
    for team_name, games in team_games.items():
        teams.append(
            {
                "team_name": team_name,
                "dates": sorted({game["date"] for game in games}),
                "fingerprints": sorted({game["deck_fingerprint"] for game in games}),
                **outcome_summary(games),
            }
        )
    teams.sort(key=lambda row: (-row["games"], row["team_name"]))

    nonempty_receipts = [receipt for _, _, receipt in inputs if receipt["outcomes"]["games"]]
    output = {
        "schema": "poke_bot.slowking_multi_day_replay_distillation/v1",
        "status": "research_only_no_training_or_runtime_authority",
        "source": {
            "start_date": inputs[0][0],
            "end_date": inputs[-1][0],
            "daily_receipts": len(inputs),
            "days_with_slowking": sum(bool(row["games"]) for row in daily_rows),
        },
        "overall": outcome_summary(all_games),
        "by_day": daily_rows,
        "by_team": teams,
        "by_deck_lineage": lineages,
        "behavior_all_lineages": merge_behavior(nonempty_receipts),
        "games": all_games,
        "limitations": [
            "The 768-game total combines two independently named teams and three exact deck fingerprints.",
            "Behavior frequencies are availability-conditioned and must not be treated as unconditional action probabilities.",
            "Daily outcome differences are observational and confounded by opponent mix and list evolution.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.out), "games": len(all_games), "teams": len(teams), "lineages": len(lineages)}, indent=2))


if __name__ == "__main__":
    main()
