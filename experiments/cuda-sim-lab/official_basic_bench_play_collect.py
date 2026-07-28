#!/usr/bin/env python3
"""Collect clean official main-phase Basic Pokemon hand-to-Bench Plays.

The controlled deck contains only vanilla Hippopotas and Basic Fighting
Energy. Setup deliberately places no optional Bench Pokemon, then the rollout
records each main-phase Play of Hippopotas. No Energy is ever attached. This
admits an exact independent legal-action constructor: Play for each Basic in
hand, Attach for every Energy x in-play Pokemon pair, followed by End.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from poke_bot import cg_env

cg_env.ensure_cg_importable()

from poke_bot.engine_rebuild.interfaces import ResetSpec
from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv


POKEMON_CARD_ID = 22
ENERGY_CARD_ID = 6
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_END = 14
AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5
CONTEXT_SETUP_BENCH = 2


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_deck(deck: list[int]) -> None:
    if Counter(deck) != Counter({POKEMON_CARD_ID: 4, ENERGY_CARD_ID: 56}):
        raise ValueError("expected controlled 4x Hippopotas + 56x Basic Fighting Energy deck")


def loaded_library_path(env: LibcgMultiEnv) -> Path:
    name = getattr(env._lib, "_name", None)
    if not name:
        raise RuntimeError("ctypes did not expose the loaded libcg path")
    return Path(str(name)).resolve()


def expected_options(current: dict[str, Any]) -> list[dict[str, int]]:
    actor = int(current["yourIndex"])
    player = current["players"][actor]
    hand = player["hand"]
    if hand is None:
        raise RuntimeError("selecting hand is hidden")
    active = player.get("active") or []
    bench = player.get("bench") or []
    if len(active) != 1:
        raise RuntimeError("clean Play slice requires exactly one Active Pokemon")
    in_play = [(AREA_ACTIVE, 0)] + [(AREA_BENCH, index) for index in range(len(bench))]
    options: list[dict[str, int]] = []
    for hand_index, card in enumerate(hand):
        card_id = int(card["id"])
        if card_id == POKEMON_CARD_ID:
            if len(bench) < int(player["benchMax"]):
                options.append({"type": OPTION_PLAY, "index": hand_index})
        elif card_id == ENERGY_CARD_ID:
            if not bool(current["energyAttached"]):
                for area, index in in_play:
                    options.append({
                        "type": OPTION_ATTACH,
                        "area": AREA_HAND,
                        "index": hand_index,
                        "inPlayArea": area,
                        "inPlayIndex": index,
                    })
        else:
            raise RuntimeError(f"card {card_id} escaped the controlled clean deck")
    options.append({"type": OPTION_END})
    return options


def expected_select(current: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": 0,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": expected_options(current),
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


def expected_after_current(
    before: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    expected = copy.deepcopy(before)
    actor = int(expected["yourIndex"])
    player = expected["players"][actor]
    source_index = int(selected["index"])
    moved = player["hand"].pop(source_index)
    if int(moved["id"]) != POKEMON_CARD_ID:
        raise RuntimeError(f"selected non-Hippopotas Play: {moved}")
    player["handCount"] = int(player["handCount"]) - 1
    player["bench"].append({
        "id": int(moved["id"]),
        "serial": int(moved["serial"]),
        "playerIndex": int(moved["playerIndex"]),
        "hp": 90,
        "maxHp": 90,
        "appearThisTurn": True,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    })
    expected["turnActionCount"] = int(expected["turnActionCount"]) + 1
    return expected


def terminal(obs: dict[str, Any]) -> bool:
    return int((obs.get("current") or {}).get("result", -1)) != -1


def generic_action(obs: dict[str, Any], game: int) -> list[int]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    select_type = int(select.get("type", -1))
    context = int(select.get("context", -1))
    if select_type == 9 and context == 41:
        return [game % 2]
    if context == CONTEXT_SETUP_BENCH:
        # minCount is zero. Preserve all remaining Basics for main-phase Plays.
        return []
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", 0))
    count = max(minimum, min(maximum, len(options)))
    return list(range(count))


def clean_play_candidates(obs: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    select = obs.get("select") or {}
    current = obs.get("current") or {}
    if int(select.get("type", -1)) != 0 or int(select.get("context", -1)) != 0:
        return []
    actor = int(current.get("yourIndex", -1))
    if actor not in (0, 1):
        return []
    player = current["players"][actor]
    if bool(current.get("energyAttached", False)):
        return []
    if len(player.get("active") or []) != 1 or len(player.get("bench") or []) >= int(player["benchMax"]):
        return []
    for pokemon in (player.get("active") or []) + (player.get("bench") or []):
        if int(pokemon.get("id", -1)) != POKEMON_CARD_ID:
            return []
        if pokemon.get("energies") or pokemon.get("energyCards") or pokemon.get("tools") or pokemon.get("preEvolution"):
            return []
    if canonical(select) != canonical(expected_select(current)):
        return []
    return [
        (index, option)
        for index, option in enumerate(select.get("option") or [])
        if int(option.get("type", -1)) == OPTION_PLAY
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--official-lib", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--fixtures", type=int, default=256)
    parser.add_argument("--max-games", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=256)
    args = parser.parse_args()
    deck = [int(value) for value in args.deck.read_text().split()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(deck)}")
    validate_deck(deck)
    oracle = args.official_lib.resolve()
    oracle_sha = hashlib.sha256(oracle.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    skipped_non_clean = 0
    games_started = 0
    env = LibcgMultiEnv(1)
    try:
        actual = loaded_library_path(env)
        actual_sha = hashlib.sha256(actual.read_bytes()).hexdigest()
        if actual_sha != oracle_sha:
            raise RuntimeError(
                f"loaded libcg {actual_sha} != named official oracle {oracle_sha}"
            )
        for game in range(args.max_games):
            if len(rows) >= args.fixtures:
                break
            games_started += 1
            obs = env.reset([ResetSpec(deck, deck, seed=game)]).envs[0].obs
            for step in range(args.max_steps):
                if terminal(obs):
                    break
                select = obs.get("select") or {}
                current = obs.get("current") or {}
                if int(select.get("type", -1)) == 0 and int(select.get("context", -1)) == 0:
                    candidates = clean_play_candidates(obs)
                    if candidates:
                        option_index, selected = candidates[len(rows) % len(candidates)]
                        before_current = copy.deepcopy(current)
                        before_select = copy.deepcopy(select)
                        next_obs = env.step_batch([[option_index]]).envs[0].obs
                        after_current = copy.deepcopy(next_obs["current"])
                        after_select = copy.deepcopy(next_obs.get("select") or {})
                        current_exact = canonical(after_current) == canonical(
                            expected_after_current(before_current, selected)
                        )
                        legal_exact = canonical(after_select) == canonical(expected_select(after_current))
                        terminal_exact = not terminal(next_obs)
                        if not (current_exact and legal_exact and terminal_exact):
                            skipped_non_clean += 1
                        else:
                            rows.append({
                                "fixture": len(rows),
                                "game": game,
                                "step": step,
                                "action": [option_index],
                                "official_error": 0,
                                "selected_option": copy.deepcopy(selected),
                                "before": {
                                    "current": before_current,
                                    "select": before_select,
                                    "terminal": False,
                                    "public_state_sha256": digest(before_current),
                                    "legal_sha256": digest(before_select),
                                },
                                "after": {
                                    "current": after_current,
                                    "select": after_select,
                                    "terminal": False,
                                    "public_state_sha256": digest(after_current),
                                    "legal_sha256": digest(after_select),
                                },
                                "oracle_checks": {
                                    "full_public_current_exact_to_reference": current_exact,
                                    "full_legal_select_exact_to_reference": legal_exact,
                                    "terminal_exact_to_reference": terminal_exact,
                                },
                            })
                        obs = next_obs
                        if len(rows) >= args.fixtures:
                            break
                        continue
                    end = [
                        index
                        for index, option in enumerate(select.get("option") or [])
                        if int(option.get("type", -1)) == OPTION_END
                    ]
                    if not end:
                        raise RuntimeError("clean main state has no End option")
                    action = [end[0]]
                else:
                    action = generic_action(obs, game)
                obs = env.step_batch([action]).envs[0].obs
            else:
                raise RuntimeError(f"game {game} exceeded {args.max_steps} steps")
    finally:
        env.close()
    if len(rows) != args.fixtures:
        raise RuntimeError(
            f"collected {len(rows)}/{args.fixtures} fixtures after {games_started} games"
        )
    report = {
        "schema": "poke_bot.official_basic_bench_play_fixtures/v1",
        "status": "complete",
        "scope": "accepted clean main-phase Basic Pokemon hand-to-Bench Play transition",
        "official_lib_path": str(oracle),
        "official_lib_sha256": oracle_sha,
        "oracle": "unmodified official libcg BattleStart/Select/GetBattleData",
        "seed_control": "official BattleStart RNG is opaque; fixtures are recorded oracle transitions",
        "deck_sha256": hashlib.sha256(args.deck.read_bytes()).hexdigest(),
        "fixture_count": len(rows),
        "games_started": games_started,
        "skipped_non_clean": skipped_non_clean,
        "coverage": {
            "step": "one accepted vanilla Basic Pokemon Play from Hand to next Bench slot",
            "legal": "exact complete next Main options rebuilt from hand x in-play state",
            "terminal": "exact ongoing result/terminal flag",
            "public_state": "entire official current object equals independent hand-to-Bench reference",
        },
        "excluded": [
            "Pokemon with Abilities, enter-play effects, or non-vanilla state",
            "full Bench, evolution, Tool, Special Energy, Trainer, Attack, Ability, and Retreat options",
            "hidden-state transitions, knockouts, prizes, status, and full seeded games",
        ],
        "full_seeded_game_parity": False,
        "full_card_effect_parity": False,
        "production_eligible": False,
        "fixtures": rows,
        "completed_at": time.time(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.json_out)
    print(json.dumps({key: report[key] for key in (
        "status", "scope", "official_lib_sha256", "fixture_count",
        "games_started", "skipped_non_clean", "production_eligible",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
