#!/usr/bin/env python3
"""Summarize decoded CABT/Kaggle replays for submitted Alakazam checkpoints.

The Kaggle replay stores an omniscient, name-decoded ``visualize`` timeline in
the first agent's initial step.  This script intentionally uses that timeline
for board milestones and public episode metadata for outcome/seat labels.  It
does not claim counterfactual action quality or per-head attribution: those
require an instrumented model replay pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ARCHETYPE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("marnie_grimmsnarl", ("Marnie's Grimmsnarl ex",)),
    ("alakazam_mirror", ("Alakazam",)),
    ("cynthia_garchomp", ("Cynthia's Garchomp ex",)),
    ("mega_lucario", ("Mega Lucario ex",)),
    ("archaludon", ("Archaludon ex",)),
    ("crustle", ("Crustle",)),
    ("dragapult", ("Dragapult ex",)),
    ("mega_lopunny", ("Mega Lopunny ex",)),
    ("mega_abomasnow", ("Mega Abomasnow ex",)),
    ("gouging_fire", ("Gouging Fire ex",)),
    ("teal_ogerpon", ("Teal Mask Ogerpon ex",)),
    ("mega_kangaskhan", ("Mega Kangaskhan ex",)),
)


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _episode_id(path: Path) -> int:
    match = re.search(r"episode-(\d+)-replay\.json$", path.name)
    if match is None:
        raise ValueError(f"unrecognized replay path: {path}")
    return int(match.group(1))


def _card_name(card: dict[str, Any], names: dict[int, str]) -> str:
    name = card.get("name")
    if isinstance(name, str):
        return name
    return names.get(int(card.get("id", -1)), f"card_{card.get('id', 'unknown')}")


def _deck_names(initial_current: dict[str, Any], index: int) -> Counter[str]:
    return Counter(
        card.get("name", f"card_{card.get('id', 'unknown')}")
        for card in initial_current["players"][index].get("deck", [])
    )


def _classify(deck: Counter[str]) -> str:
    names = set(deck)
    for label, required in ARCHETYPE_SIGNATURES:
        if all(name in names for name in required):
            return label
    pokemon = [
        name
        for name in names
        if not any(
            token in name
            for token in (
                "Energy", "Ball", "Orders", "Poffin", "Stretcher", "Patch",
                "Candy", "Pad", "Catcher", "Academy", "Hammer", "Switch",
            )
        )
    ]
    return "other:" + (sorted(pokemon)[0] if pokemon else "unknown")


def _timeline(replay: dict[str, Any]) -> list[dict[str, Any]]:
    for agent in replay.get("steps", [[]])[0]:
        visualize = agent.get("visualize")
        if isinstance(visualize, list) and visualize:
            return visualize
    return []


def _safe_prize_count(player: dict[str, Any]) -> int | None:
    prize = player.get("prize")
    return len(prize) if isinstance(prize, list) else None


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _agent_log_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    raw = _load(path)
    durations: list[float] = []
    stderr: list[str] = []
    for outer in raw if isinstance(raw, list) else []:
        for row in outer if isinstance(outer, list) else []:
            if isinstance(row.get("duration"), (int, float)):
                durations.append(float(row["duration"]))
            if row.get("stderr"):
                stderr.append(str(row["stderr"]))
    return {
        "available": True,
        "calls": len(durations),
        "first_s": durations[0] if durations else None,
        "subsequent_p95_s": percentile(durations[1:], 0.95) if len(durations) > 1 else None,
        "max_s": max(durations) if durations else None,
        "total_s": sum(durations),
        "stderr": stderr,
    }


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    location = (len(ordered) - 1) * q
    low = math.floor(location)
    high = math.ceil(location)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def _own_turn_rank(frames: list[dict[str, Any]], own_index: int) -> dict[int, int]:
    turns = sorted(
        {
            int(frame["current"]["turn"])
            for frame in frames
            if isinstance(frame.get("current"), dict)
            and frame["current"].get("yourIndex") == own_index
            and isinstance(frame["current"].get("turn"), int)
            and int(frame["current"]["turn"]) > 0
        }
    )
    return {turn: rank + 1 for rank, turn in enumerate(turns)}


def analyze_replay(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    replay = _load(path)
    frames = _timeline(replay)
    if not frames:
        raise ValueError(f"missing decoded visualize timeline: {path}")
    own_index = int(meta["own_agent"]["index"])
    opponent_index = 1 - own_index
    initial = frames[0]["current"]
    own_deck = _deck_names(initial, own_index)
    opponent_deck = _deck_names(initial, opponent_index)
    names: dict[int, str] = {}
    for player in initial["players"]:
        for card in player.get("deck", []):
            if "id" in card and "name" in card:
                names[int(card["id"])] = str(card["name"])

    turn_rank = _own_turn_rank(frames, own_index)
    first_player: int | None = None
    first_seen: dict[str, int | None] = {
        "Abra": None,
        "Kadabra": None,
        "Alakazam": None,
    }
    first_seen_own_turn: dict[str, int | None] = dict(first_seen)
    own_prompts: Counter[str] = Counter()
    own_attacks: Counter[str] = Counter()
    own_attack_ids: Counter[str] = Counter()
    own_plays: Counter[str] = Counter()
    own_evolutions: Counter[str] = Counter()
    own_attachments: Counter[str] = Counter()
    own_attack_turns: list[int] = []
    max_bench = 0
    max_alakazam = 0
    max_psychic_energy_in_play = 0
    last_current = initial
    own_turn_end: dict[int, dict[str, Any]] = {}

    for frame in frames:
        current = frame.get("current")
        if not isinstance(current, dict):
            continue
        last_current = current
        if current.get("firstPlayer") in (0, 1):
            first_player = int(current["firstPlayer"])
        actor = current.get("yourIndex")
        turn = int(current.get("turn", 0) or 0)
        if actor == own_index:
            select = frame.get("select") or {}
            context = select.get("context")
            if context:
                own_prompts[str(context)] += 1
            if turn in turn_rank:
                own_turn_end[turn_rank[turn]] = current

        player = current["players"][own_index]
        in_play = list(player.get("active") or []) + list(player.get("bench") or [])
        in_play_names = [_card_name(card, names) for card in in_play]
        max_bench = max(max_bench, len(player.get("bench") or []))
        max_alakazam = max(max_alakazam, in_play_names.count("Alakazam"))
        psychic_energy = sum(
            1
            for card in in_play
            for energy in card.get("energyCards", [])
            if _card_name(energy, names) in {"Basic {P} Energy", "Telepath Psychic Energy"}
        )
        max_psychic_energy_in_play = max(max_psychic_energy_in_play, psychic_energy)
        for milestone in first_seen:
            if milestone in in_play_names and first_seen[milestone] is None:
                first_seen[milestone] = turn
                first_seen_own_turn[milestone] = turn_rank.get(turn)
        for log in frame.get("logs") or []:
            if log.get("type") in (15, "Attack") and log.get("playerIndex") == own_index:
                attacker = names.get(int(log.get("cardId", -1)), "unknown")
                own_attacks[attacker] += 1
                own_attack_ids[str(log.get("attackId", "unknown"))] += 1
                own_attack_turns.append(turn)
            elif log.get("type") == "Play" and log.get("playerIndex") == own_index:
                own_plays[names.get(int(log.get("cardId", -1)), "unknown")] += 1
            elif log.get("type") == "Evolve" and log.get("playerIndex") == own_index:
                own_evolutions[names.get(int(log.get("cardId", -1)), "unknown")] += 1
            elif log.get("type") == "Attach" and log.get("playerIndex") == own_index:
                own_attachments[names.get(int(log.get("cardId", -1)), "unknown")] += 1

    final_players = last_current["players"]
    final_own = final_players[own_index]
    final_opp = final_players[opponent_index]
    own_prizes_remaining = _safe_prize_count(final_own)
    opp_prizes_remaining = _safe_prize_count(final_opp)
    own_reward = float(meta["own_agent"]["reward"])
    opponent = meta["agents"][opponent_index]
    elapsed_s = (_parse_time(meta["end_time"]) - _parse_time(meta["create_time"])).total_seconds()

    own_turn_snapshots: dict[str, Any] = {}
    for rank, current in sorted(own_turn_end.items()):
        player = current["players"][own_index]
        cards = list(player.get("active") or []) + list(player.get("bench") or [])
        card_names = [_card_name(card, names) for card in cards]
        own_turn_snapshots[str(rank)] = {
            "global_turn": int(current.get("turn", 0)),
            "bench": len(player.get("bench") or []),
            "abra": card_names.count("Abra"),
            "kadabra": card_names.count("Kadabra"),
            "alakazam": card_names.count("Alakazam"),
            "deck_count": player.get("deckCount"),
            "hand_count": player.get("handCount"),
            "prizes_remaining": _safe_prize_count(player),
        }

    log_path = path.with_name(f"episode-{_episode_id(path)}-agent-{own_index}-logs.json")
    own_log = _agent_log_stats(log_path)
    final_own_board = list(final_own.get("active") or []) + list(final_own.get("bench") or [])
    return {
        "episode_id": _episode_id(path),
        "iteration": int(meta["iteration"]),
        "submission_id": int(meta["submission_id"]),
        "own_index": own_index,
        "first_player": first_player,
        "went_first": first_player == own_index,
        "reward": own_reward,
        "win": own_reward > 0,
        "draw": own_reward == 0,
        "opponent_team": opponent["team_name"],
        "opponent_submission_id": int(opponent["submission_id"]),
        "opponent_archetype": _classify(opponent_deck),
        "opponent_deck": dict(sorted(opponent_deck.items())),
        "own_deck": dict(sorted(own_deck.items())),
        "elapsed_s": elapsed_s,
        "frames": len(frames),
        "max_global_turn": max(int((f.get("current") or {}).get("turn", 0) or 0) for f in frames),
        "own_turns": len(turn_rank),
        "own_prompts": dict(own_prompts),
        "own_attacks": dict(own_attacks),
        "own_attack_ids": dict(own_attack_ids),
        "own_plays": dict(own_plays),
        "own_evolutions": dict(own_evolutions),
        "own_attachments": dict(own_attachments),
        "own_attack_count": sum(own_attacks.values()),
        "first_attack_global_turn": min(own_attack_turns) if own_attack_turns else None,
        "first_seen_global_turn": first_seen,
        "first_seen_own_turn": first_seen_own_turn,
        "alakazam_by_own_turn_2": first_seen_own_turn["Alakazam"] is not None
        and int(first_seen_own_turn["Alakazam"]) <= 2,
        "alakazam_by_own_turn_3": first_seen_own_turn["Alakazam"] is not None
        and int(first_seen_own_turn["Alakazam"]) <= 3,
        "max_bench": max_bench,
        "max_alakazam": max_alakazam,
        "max_psychic_energy_in_play": max_psychic_energy_in_play,
        "own_prizes_remaining": own_prizes_remaining,
        "opponent_prizes_remaining": opp_prizes_remaining,
        "own_prizes_taken": None if own_prizes_remaining is None else 6 - own_prizes_remaining,
        "opponent_prizes_taken": None if opp_prizes_remaining is None else 6 - opp_prizes_remaining,
        "final_own_deck_count": final_own.get("deckCount"),
        "final_opponent_deck_count": final_opp.get("deckCount"),
        "final_own_board_size": len(final_own_board),
        "probable_deckout_loss": bool(
            own_reward < 0 and final_own.get("deckCount") == 0 and len(final_own_board) > 0
        ),
        "probable_board_wipe_loss": bool(own_reward < 0 and len(final_own_board) == 0),
        "own_turn_snapshots": own_turn_snapshots,
        "own_agent_log": own_log,
    }


def _rate(rows: list[dict[str, Any]], field: str = "win") -> float | None:
    return mean(float(row[field]) for row in rows) if rows else None


def _avg(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        losses = [row for row in group if not row["win"] and not row["draw"]]
        card_names = sorted({name for row in group for name in row["own_plays"]})
        play_deltas = {}
        wins = [row for row in group if row["win"]]
        for name in card_names:
            win_mean = mean(row["own_plays"].get(name, 0) for row in wins) if wins else None
            loss_mean = mean(row["own_plays"].get(name, 0) for row in losses) if losses else None
            play_deltas[name] = {
                "mean_per_win": win_mean,
                "mean_per_loss": loss_mean,
                "loss_minus_win": None
                if win_mean is None or loss_mean is None
                else loss_mean - win_mean,
            }
        return {
            "games": len(group),
            "wins": sum(bool(row["win"]) for row in group),
            "draws": sum(bool(row["draw"]) for row in group),
            "win_rate": _rate(group),
            "went_first_rate": _rate(group, "went_first"),
            "win_rate_went_first": _rate([row for row in group if row["went_first"]]),
            "win_rate_went_second": _rate([row for row in group if not row["went_first"]]),
            "alakazam_by_own_turn_2_rate": _rate(group, "alakazam_by_own_turn_2"),
            "alakazam_by_own_turn_3_rate": _rate(group, "alakazam_by_own_turn_3"),
            "win_rate_with_alakazam_by_turn_2": _rate(
                [row for row in group if row["alakazam_by_own_turn_2"]]
            ),
            "win_rate_without_alakazam_by_turn_2": _rate(
                [row for row in group if not row["alakazam_by_own_turn_2"]]
            ),
            "mean_own_prizes_taken": _avg(group, "own_prizes_taken"),
            "mean_opponent_prizes_taken": _avg(group, "opponent_prizes_taken"),
            "mean_own_attack_count": _avg(group, "own_attack_count"),
            "median_elapsed_s": median(row["elapsed_s"] for row in group),
            "probable_deckout_losses": sum(row["probable_deckout_loss"] for row in losses),
            "probable_board_wipe_losses": sum(row["probable_board_wipe_loss"] for row in losses),
            "losses_with_agent_log": sum(row["own_agent_log"]["available"] for row in losses),
            "losses_with_stderr": sum(bool(row["own_agent_log"].get("stderr")) for row in losses),
            "loss_inference_p95_s": percentile(
                [
                    row["own_agent_log"]["max_s"]
                    for row in losses
                    if row["own_agent_log"].get("max_s") is not None
                ],
                0.95,
            ),
            "card_play_comparison": play_deltas,
        }

    by_iteration: dict[str, Any] = {}
    for iteration in sorted({row["iteration"] for row in rows}):
        group = [row for row in rows if row["iteration"] == iteration]
        by_iteration[str(iteration)] = summarize(group)

    by_archetype: dict[str, Any] = {}
    for archetype in sorted({row["opponent_archetype"] for row in rows}):
        group = [row for row in rows if row["opponent_archetype"] == archetype]
        by_archetype[archetype] = summarize(group)

    iteration_archetype: dict[str, Any] = {}
    for iteration in sorted({row["iteration"] for row in rows}):
        for archetype in sorted({row["opponent_archetype"] for row in rows}):
            group = [
                row
                for row in rows
                if row["iteration"] == iteration and row["opponent_archetype"] == archetype
            ]
            if group:
                iteration_archetype[f"iter_{iteration:05d}:{archetype}"] = summarize(group)

    archetype_setup_seat: dict[str, Any] = {}
    for archetype in sorted({row["opponent_archetype"] for row in rows}):
        archetype_rows = [row for row in rows if row["opponent_archetype"] == archetype]
        if len(archetype_rows) < 8:
            continue
        for went_first in (True, False):
            for setup in (True, False):
                group = [
                    row
                    for row in archetype_rows
                    if row["went_first"] is went_first
                    and row["alakazam_by_own_turn_2"] is setup
                ]
                if group:
                    archetype_setup_seat[
                        f"{archetype}:first={str(went_first).lower()}:setup2={str(setup).lower()}"
                    ] = {"games": len(group), "wins": sum(row["win"] for row in group), "win_rate": _rate(group)}

    return {
        "all": summarize(rows),
        "by_iteration": by_iteration,
        "by_archetype": by_archetype,
        "by_iteration_archetype": iteration_archetype,
        "by_archetype_setup_seat": archetype_setup_seat,
        "opponent_team_counts": dict(Counter(row["opponent_team"] for row in rows).most_common()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id", "iteration", "submission_id", "own_index", "went_first",
        "reward", "win", "opponent_team", "opponent_archetype", "elapsed_s",
        "max_global_turn", "own_turns", "own_attack_count", "alakazam_by_own_turn_2",
        "alakazam_by_own_turn_3", "max_bench", "max_alakazam",
        "max_psychic_energy_in_play", "own_prizes_taken", "opponent_prizes_taken",
        "final_own_deck_count", "final_opponent_deck_count", "final_own_board_size",
        "probable_deckout_loss", "probable_board_wipe_loss",
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
    rows: list[dict[str, Any]] = []
    for path in sorted(args.replay_root.glob("iter_*/*-replay.json")):
        episode_id = _episode_id(path)
        if episode_id not in metadata:
            raise KeyError(f"missing metadata for episode {episode_id}")
        rows.append(analyze_replay(path, metadata[episode_id]))

    payload = {
        "schema": "poke_bot.kaggle_alakazam_replay_analysis/v1",
        "source": {
            "replay_root": str(args.replay_root),
            "episodes_json": str(args.replay_root / "episodes.json"),
            "download_receipt": str(args.replay_root / "DOWNLOAD_RECEIPT.json"),
            "submissions": source.get("submissions"),
        },
        "definitions": {
            "win_rate": "episodes with own reward > 0 divided by episodes",
            "went_first": "decoded replay firstPlayer equals own agent index",
            "alakazam_by_own_turn_2": "Alakazam appears active or benched no later than the second distinct own turn",
            "probable_deckout_loss": "loss with final own deck count zero and a nonempty board",
            "probable_board_wipe_loss": "loss with no final own active or benched Pokemon",
        },
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
