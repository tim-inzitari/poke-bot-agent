#!/usr/bin/env python3
"""Capture clean accepted End-turn transitions from unmodified official libcg."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from poke_bot.engine_rebuild.interfaces import ResetSpec
from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv


def state_slice(obs: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    players = current.get("players") or [{}, {}]
    out: dict[str, Any] = {
        "turn": int(current.get("turn", -1)),
        "your_index": int(current.get("yourIndex", -1)),
        "first_player": int(current.get("firstPlayer", -1)),
        "result": int(current.get("result", -1)),
        "supporter_played": bool(current.get("supporterPlayed", False)),
        "stadium_played": bool(current.get("stadiumPlayed", False)),
        "energy_attached": bool(current.get("energyAttached", False)),
        "retreated": bool(current.get("retreated", False)),
        "select_type": int(select.get("type", -1)),
        "select_context": int(select.get("context", -1)),
        "select_min": int(select.get("minCount", 0)),
        "select_max": int(select.get("maxCount", 0)),
        "option_count": len(select.get("option") or []),
    }
    for player in range(2):
        value = players[player] or {}
        out[f"deck_count_{player}"] = int(value.get("deckCount", 0))
        out[f"hand_count_{player}"] = int(value.get("handCount", 0))
        out[f"active_count_{player}"] = len(value.get("active") or [])
        out[f"bench_count_{player}"] = len(value.get("bench") or [])
        for status in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
            out[f"{status}_{player}"] = bool(value.get(status, False))
    return out


def choose(obs: dict[str, Any], first_choice: int) -> tuple[list[int], bool]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    if int(select.get("type", -1)) == 9 and int(select.get("context", -1)) == 41:
        return [int(first_choice)], False
    if int(select.get("type", -1)) == 0 and int(select.get("context", -1)) == 0:
        for index, option in enumerate(options):
            if int(option.get("type", -1)) == 14:
                return [index], True
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", 0))
    count = max(minimum, min(maximum, len(options)))
    return list(range(count)), False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--official-lib", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    deck = [int(value) for value in args.deck.read_text().split()]
    if len(deck) != 60:
        raise ValueError("deck must contain exactly 60 cards")
    rows: list[dict[str, Any]] = []
    terminal_games = 0
    env = LibcgMultiEnv(1)
    try:
        for game in range(args.games):
            obs = env.reset([ResetSpec(deck, deck, seed=game)]).envs[0].obs
            for step in range(args.max_steps):
                before = state_slice(obs)
                if before["result"] != -1:
                    terminal_games += 1
                    break
                action, is_end = choose(obs, game % 2)
                next_obs = env.step_batch([action]).envs[0].obs
                if is_end:
                    rows.append({
                        "game": game,
                        "step": step,
                        "action": action,
                        "selected_option_type": 14,
                        "before": before,
                        "after": state_slice(next_obs),
                    })
                obs = next_obs
            else:
                raise RuntimeError(f"game {game} exceeded {args.max_steps} steps")
    finally:
        env.close()
    report = {
        "schema": "poke_bot.official_clean_end_turn_fixtures/v1",
        "status": "complete",
        "scope": "accepted clean End action through draw/deckout control-flow transition",
        "official_lib_sha256": hashlib.sha256(args.official_lib.read_bytes()).hexdigest(),
        "games": args.games,
        "terminal_games": terminal_games,
        "fixture_count": len(rows),
        "terminal_transition_count": sum(row["after"]["result"] != -1 for row in rows),
        "fixtures": rows,
        "completed_at": time.time(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.json_out)
    print(json.dumps({key: report[key] for key in (
        "status", "scope", "official_lib_sha256", "games", "terminal_games",
        "fixture_count", "terminal_transition_count",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
