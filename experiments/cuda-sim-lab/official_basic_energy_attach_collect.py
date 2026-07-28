#!/usr/bin/env python3
"""Collect official clean basic-Energy attachment transitions.

The fixture deck deliberately contains only four vanilla Basic Pokemon and
fifty-six matching Basic Energy cards.  Before recording a transition, the
rollout empties all playable Basic Pokemon from the hand and requires the
active Pokemon to have no Energy.  The accepted action then attaches one Basic
Energy to a Benched Pokemon.  This makes the complete legal-state delta
independently implementable: every pre-step legal option is Attach or End and
the exact post-step legal list is one End option.

Ground truth is always the unmodified official competition ``libcg.so``.  The
binary actually loaded by ctypes is hashed and must match ``--official-lib``.
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

# Some deployed adapters predate their own import bootstrap.  Establish the
# official runtime path before importing the adapter so this collector remains
# self-contained on those hosts as well.
cg_env.ensure_cg_importable()

from poke_bot.engine_rebuild.interfaces import ResetSpec
from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv


POKEMON_CARD_ID = 22  # Hippopotas: vanilla Basic, no Ability.
ENERGY_CARD_ID = 6  # Basic Fighting Energy.
ENERGY_TYPE = 6
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_END = 14
AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def loaded_library_path(env: LibcgMultiEnv) -> Path:
    name = getattr(env._lib, "_name", None)
    if not name:
        raise RuntimeError("ctypes did not expose the loaded libcg path")
    return Path(str(name)).resolve()


def validate_deck(deck: list[int]) -> None:
    expected = Counter({POKEMON_CARD_ID: 4, ENERGY_CARD_ID: 56})
    if Counter(deck) != expected:
        raise ValueError(
            "clean attach deck must contain exactly 4x Hippopotas (22) and "
            "56x Basic Fighting Energy (6)"
        )


def option_signature(option: dict[str, Any]) -> list[int]:
    """Exact numeric legal-option encoding used by the CUDA gate."""
    return [
        int(option.get("type", -1)),
        int(option.get("area", -1)),
        int(option.get("index", -1)),
        int(option.get("playerIndex", -1)),
        int(option.get("inPlayArea", -1)),
        int(option.get("inPlayIndex", -1)),
        int(option.get("attackId", -1)),
    ]


def terminal(obs: dict[str, Any]) -> bool:
    return int((obs.get("current") or {}).get("result", -1)) != -1


def generic_valid_action(obs: dict[str, Any], game: int) -> list[int]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    # Alternate the official IsFirst choice to cover both first-player orders.
    if int(select.get("type", -1)) == 9 and int(select.get("context", -1)) == 41:
        return [game % 2]
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", 0))
    count = max(minimum, min(maximum, len(options)))
    return list(range(count))


def card_at_hand(current: dict[str, Any], player: int, index: int) -> dict[str, Any]:
    hand = (current.get("players") or [])[player].get("hand")
    if hand is None:
        raise RuntimeError("selecting player's hand unexpectedly hidden")
    return hand[index]


def expected_after_current(
    before: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    """Pure public-state reference for this deliberately clean slice."""
    expected = copy.deepcopy(before)
    actor = int(before["yourIndex"])
    players = expected["players"]
    source_index = int(selected["index"])
    hand = players[actor]["hand"]
    moved = hand.pop(source_index)
    if int(moved["id"]) != ENERGY_CARD_ID:
        raise RuntimeError(f"selected non-fixture Energy card: {moved}")
    players[actor]["handCount"] = int(players[actor]["handCount"]) - 1
    target_area = int(selected["inPlayArea"])
    target_index = int(selected["inPlayIndex"])
    if target_area == AREA_ACTIVE:
        target = players[actor]["active"][target_index]
    elif target_area == AREA_BENCH:
        target = players[actor]["bench"][target_index]
    else:
        raise RuntimeError(f"unexpected attach target area {target_area}")
    target["energies"].append(ENERGY_TYPE)
    target["energyCards"].append(moved)
    expected["energyAttached"] = True
    expected["turnActionCount"] = int(expected["turnActionCount"]) + 1
    return expected


def clean_candidate(
    obs: dict[str, Any], selector: int
) -> tuple[int, dict[str, Any]] | None:
    select = obs.get("select") or {}
    current = obs.get("current") or {}
    if int(select.get("type", -1)) != 0 or int(select.get("context", -1)) != 0:
        return None
    if bool(current.get("energyAttached", False)):
        return None
    options = select.get("option") or []
    if any(int(option.get("type", -1)) not in (OPTION_ATTACH, OPTION_END) for option in options):
        return None
    actor = int(current.get("yourIndex", -1))
    players = current.get("players") or []
    if actor not in (0, 1) or len(players) != 2:
        return None
    active = players[actor].get("active") or []
    if len(active) != 1 or active[0].get("energies") or active[0].get("energyCards"):
        return None
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != OPTION_ATTACH:
            continue
        if int(option.get("area", -1)) != AREA_HAND:
            continue
        if int(option.get("inPlayArea", -1)) != AREA_BENCH:
            continue
        source = card_at_hand(current, actor, int(option["index"]))
        if int(source.get("id", -1)) == ENERGY_CARD_ID:
            candidates.append((index, option))
    return candidates[selector % len(candidates)] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--official-lib", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--fixtures", type=int, default=128)
    parser.add_argument("--max-games", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=256)
    args = parser.parse_args()
    deck = [int(value) for value in args.deck.read_text().split()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(deck)}")
    validate_deck(deck)
    expected_lib = args.official_lib.resolve()
    expected_sha = hashlib.sha256(expected_lib.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    skipped_non_clean = 0
    games_started = 0
    env = LibcgMultiEnv(1)
    try:
        actual_lib = loaded_library_path(env)
        actual_sha = hashlib.sha256(actual_lib.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"loaded libcg digest {actual_sha} != oracle digest {expected_sha}; "
                f"loaded={actual_lib} oracle={expected_lib}"
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
                options = select.get("option") or []
                if int(select.get("type", -1)) == 0 and int(select.get("context", -1)) == 0:
                    # Empty all Basic Pokemon from hand first, keeping the clean
                    # attach transition free of independent Play actions.
                    plays = [i for i, option in enumerate(options) if int(option.get("type", -1)) == OPTION_PLAY]
                    if plays:
                        action = [plays[0]]
                    else:
                        candidate = clean_candidate(obs, len(rows))
                        if candidate is not None:
                            option_index, selected = candidate
                            before_current = copy.deepcopy(obs["current"])
                            before_select = copy.deepcopy(select)
                            next_obs = env.step_batch([[option_index]]).envs[0].obs
                            after_current = copy.deepcopy(next_obs["current"])
                            after_select = copy.deepcopy(next_obs.get("select") or {})
                            expected_current = expected_after_current(before_current, selected)
                            current_exact = canonical(after_current) == canonical(expected_current)
                            legal_exact = [option_signature(option) for option in (after_select.get("option") or [])] == [
                                [OPTION_END, -1, -1, -1, -1, -1, -1]
                            ]
                            control_exact = (
                                int(after_select.get("type", -1)) == 0
                                and int(after_select.get("context", -1)) == 0
                                and int(after_select.get("minCount", -1)) == 1
                                and int(after_select.get("maxCount", -1)) == 1
                            )
                            if not (current_exact and legal_exact and control_exact and not terminal(next_obs)):
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
                                        "full_legal_select_exact_to_reference": legal_exact and control_exact,
                                        "terminal_exact_to_reference": not terminal(next_obs),
                                    },
                                })
                            obs = next_obs
                            if len(rows) >= args.fixtures:
                                break
                            continue
                        end = [i for i, option in enumerate(options) if int(option.get("type", -1)) == OPTION_END]
                        if not end:
                            raise RuntimeError("clean-deck main state has no End option")
                        action = [end[0]]
                else:
                    action = generic_valid_action(obs, game)
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
        "schema": "poke_bot.official_basic_energy_attach_fixtures/v1",
        "status": "complete",
        "scope": "accepted clean Basic Energy hand-to-bench attachment transition",
        "official_lib_path": str(expected_lib),
        "official_lib_sha256": expected_sha,
        "oracle": "unmodified official libcg BattleStart/Select/GetBattleData",
        "seed_control": "official BattleStart RNG is opaque; fixtures are recorded oracle transitions",
        "deck_sha256": hashlib.sha256(args.deck.read_bytes()).hexdigest(),
        "deck": {
            "pokemon_card_id": POKEMON_CARD_ID,
            "pokemon_count": 4,
            "energy_card_id": ENERGY_CARD_ID,
            "energy_type": ENERGY_TYPE,
            "energy_count": 56,
        },
        "fixture_count": len(rows),
        "games_started": games_started,
        "skipped_non_clean": skipped_non_clean,
        "coverage": {
            "step": "one accepted Basic Energy Attach option from Hand to Bench",
            "legal": "exact post-step Main selection: min=max=1 and sole End option",
            "terminal": "exact ongoing result/terminal flag",
            "public_state": "entire official current object equals the independent clean-attach reference",
        },
        "excluded": [
            "attachment to Active Pokemon",
            "Special Energy and Pokemon Tool attachments",
            "Energy-attach triggers, abilities, and card effects",
            "states with Play/Evolve/Attack/Ability/Retreat legal options",
            "knockout, prize, status, retreat, attack, and full seeded-game transitions",
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
