#!/usr/bin/env python3
"""Reverse-engineer the unique Slowking policy in a daily ladder archive.

The analysis is deliberately research-only.  It identifies seats by the
presence of Slowking SCR 58, proves how many exact 60-card multisets occur,
and emits compact behavioral evidence without copying the large raw episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SLOWKING = 163
ACADEMY_AT_NIGHT = 1248
CIPHERMANIAC = 1188
WONDROUS_PATCH = 1146
SEEK_INSPIRATION_ATTACK = 213

OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

OPTION_NAMES = {
    OPT_YES: "yes",
    OPT_NO: "no",
    OPT_CARD: "card",
    OPT_ENERGY_CARD: "energy_card",
    OPT_ENERGY: "energy",
    OPT_PLAY: "play",
    OPT_ATTACH: "attach",
    OPT_EVOLVE: "evolve",
    OPT_ABILITY: "ability",
    OPT_DISCARD: "discard",
    OPT_RETREAT: "retreat",
    OPT_ATTACK: "attack",
    OPT_END: "end",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--card-data", type=Path, default=Path("cards/EN_Card_Data.csv"))
    parser.add_argument("--source-date", required=True)
    parser.add_argument(
        "--manifest-episode-count",
        type=int,
        default=None,
        help="Optional episode count advertised by the public index manifest.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def deck_fingerprint(deck: Iterable[int]) -> str:
    payload = json.dumps(sorted(int(value) for value in deck), separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def card_catalog(path: Path) -> tuple[dict[int, str], dict[int, list[str]]]:
    names: dict[int, str] = {}
    moves: dict[int, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            card_id = int(row["Card ID"])
            names[card_id] = str(row["Card Name"])
            move = str(row.get("Move Name") or "").strip()
            if move and move not in moves[card_id]:
                moves[card_id].append(move)
    return names, dict(moves)


def setup_decks(payload: dict[str, Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for seat, entry in enumerate(step[:2]):
            action = entry.get("action") if isinstance(entry, dict) else None
            if (
                decks[seat] is None
                and isinstance(action, list)
                and len(action) == 60
                and all(isinstance(value, int) for value in action)
            ):
                decks[seat] = list(action)
        if all(deck is not None for deck in decks):
            break
    return decks


def card_id(card: Any) -> int | None:
    if not isinstance(card, dict):
        return None
    value = card.get("id")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def resolve_card(observation: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or []
    option_type = int(option.get("type", -1))
    area = AREA_HAND if option_type == OPT_PLAY else option.get("area")
    index = option.get("index")
    if not isinstance(index, int) or index < 0:
        return None
    if area == AREA_DECK:
        zone = select.get("deck") or []
    elif area == AREA_STADIUM:
        zone = current.get("stadium") or []
    elif area == AREA_LOOKING:
        zone = current.get("looking") or []
    else:
        seat = option.get("playerIndex", current.get("yourIndex"))
        if seat not in (0, 1) or len(players) != 2:
            return None
        key = {
            AREA_HAND: "hand",
            AREA_DISCARD: "discard",
            AREA_ACTIVE: "active",
            AREA_BENCH: "bench",
            AREA_PRIZE: "prize",
        }.get(area)
        if key is None:
            return None
        zone = players[seat].get(key) or []
    return zone[index] if index < len(zone) and isinstance(zone[index], dict) else None


def resolve_target(observation: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    current = observation.get("current") or {}
    players = current.get("players") or []
    seat = current.get("yourIndex")
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if seat not in (0, 1) or len(players) != 2 or not isinstance(index, int):
        return None
    key = {AREA_ACTIVE: "active", AREA_BENCH: "bench"}.get(area)
    if key is None:
        return None
    zone = players[seat].get(key) or []
    return zone[index] if 0 <= index < len(zone) and isinstance(zone[index], dict) else None


def effect_id(select: dict[str, Any]) -> int | None:
    return card_id(select.get("effect")) or card_id(select.get("contextCard"))


def acting_frames(payload: dict[str, Any], seat: int) -> list[tuple[int, dict[str, Any]]]:
    frames: list[tuple[int, dict[str, Any]]] = []
    for env_step, step in enumerate(payload.get("steps") or []):
        if not isinstance(step, list) or seat >= len(step) or not isinstance(step[seat], dict):
            continue
        entry = step[seat]
        observation = entry.get("observation") or {}
        current = observation.get("current")
        select = observation.get("select")
        action = entry.get("action")
        if (
            not isinstance(current, dict)
            or current.get("yourIndex") != seat
            or not isinstance(select, dict)
            or not isinstance(action, list)
        ):
            continue
        options = select.get("option") or []
        if not action or not all(
            isinstance(index, int) and 0 <= index < len(options) for index in action
        ):
            continue
        frames.append((env_step, entry))

    # Kaggle can record several attempted responses for one unchanged prompt.
    # Only the final response for an identical turn/action-count/context can
    # advance the environment.  Counting all attempts badly inflates setup and
    # play frequencies (for example, more than one initial Active per game).
    deduplicated: dict[tuple[int, int, int], tuple[int, dict[str, Any]]] = {}
    for env_step, entry in frames:
        observation = entry.get("observation") or {}
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        key = (
            int(current.get("turn", -1)),
            int(current.get("turnActionCount", -1)),
            int(select.get("context", -1)),
        )
        deduplicated[key] = (env_step, entry)
    return sorted(deduplicated.values(), key=lambda row: row[0])


def copied_attack(payload: dict[str, Any], seat: int, after_step: int) -> tuple[int | None, int | None]:
    source: int | None = None
    attack: int | None = None
    for step in (payload.get("steps") or [])[after_step + 1 : after_step + 10]:
        if not isinstance(step, list) or seat >= len(step):
            continue
        observation = (step[seat].get("observation") or {}) if isinstance(step[seat], dict) else {}
        for log in observation.get("logs") or []:
            if not isinstance(log, dict) or log.get("playerIndex") != seat:
                continue
            if (
                source is None
                and log.get("fromArea") == AREA_DECK
                and log.get("toArea") == AREA_DISCARD
                and isinstance(log.get("cardId"), int)
            ):
                source = int(log["cardId"])
            if (
                log.get("type") == 15
                and isinstance(log.get("attackId"), int)
                and int(log["attackId"]) != SEEK_INSPIRATION_ATTACK
            ):
                attack = int(log["attackId"])
        if source is not None and attack is not None:
            break
    return source, attack


def action_confirmed(
    payload: dict[str, Any],
    seat: int,
    env_step: int,
    option: dict[str, Any],
    resolved_id: int | None,
) -> bool:
    """Require a subsequent state/log transition for a proposed action.

    Public episode rows can contain retries on an unchanged prompt.  A chosen
    option is evidence only when the next few serialized states show the
    corresponding card/action transition.
    """
    option_type = int(option.get("type", -1))
    attack_id = option.get("attackId")
    start_step = (payload.get("steps") or [])[env_step]
    start_entry = start_step[seat] if isinstance(start_step, list) and seat < len(start_step) else {}
    start_observation = start_entry.get("observation") or {}
    start_current = start_observation.get("current") or {}
    start_turn = start_current.get("turn")
    start_logs = start_observation.get("logs") or []
    for step in (payload.get("steps") or [])[env_step + 1 : env_step + 6]:
        if not isinstance(step, list) or seat >= len(step) or not isinstance(step[seat], dict):
            continue
        observation = step[seat].get("observation") or {}
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        logs = observation.get("logs") or []
        if option_type == OPT_END and current.get("turn") != start_turn:
            return True
        if option_type == OPT_ABILITY and effect_id(select) == resolved_id:
            return True
        if logs == start_logs:
            continue
        for log in logs:
            if not isinstance(log, dict) or log.get("playerIndex") != seat:
                continue
            if option_type == OPT_ATTACK and log.get("type") == 15 and log.get("attackId") == attack_id:
                return True
            if option_type == OPT_RETREAT and log.get("type") == 8:
                return True
            if resolved_id is None or log.get("cardId") != resolved_id:
                continue
            if option_type in {OPT_PLAY, OPT_ABILITY} and log.get("type") in {6, 10}:
                return True
            if option_type == OPT_ATTACH and log.get("type") == 11:
                return True
            if option_type == OPT_EVOLVE and log.get("type") == 12:
                return True
            if option_type == OPT_DISCARD and log.get("type") == 6:
                return True
            if option_type in {OPT_CARD, OPT_ENERGY_CARD, OPT_ENERGY} and log.get("type") in {6, 11, 12}:
                return True
    return False


def counter_rows(counter: Counter[Any], names: dict[int, str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows = []
    for key, count in counter.most_common():
        row: dict[str, Any] = {"key": key, "count": count, "fraction": count / total if total else 0.0}
        if isinstance(key, int):
            row["card_name"] = names.get(key, f"card_{key}")
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    names, moves = card_catalog(args.card_data)
    list_counts: Counter[str] = Counter()
    lists: dict[str, list[int]] = {}
    games: list[dict[str, Any]] = []
    initial_active: Counter[int] = Counter()
    initial_bench: Counter[int] = Counter()
    main_actions: Counter[str] = Counter()
    played_cards: Counter[int] = Counter()
    attached_energy: Counter[int] = Counter()
    attachment_targets: Counter[int] = Counter()
    evolved_cards: Counter[int] = Counter()
    ability_sources: Counter[int] = Counter()
    academy_targets: Counter[int] = Counter()
    search_targets: dict[int, Counter[int]] = defaultdict(Counter)
    seek_sources: Counter[int] = Counter()
    seek_copied_attacks: Counter[int] = Counter()
    attack_users: Counter[int] = Counter()
    turn_bigrams: Counter[str] = Counter()
    turn_trigrams: Counter[str] = Counter()
    opponent_results: dict[str, Counter[str]] = defaultdict(Counter)

    with zipfile.ZipFile(args.archive) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for member_index, member in enumerate(members, 1):
            payload = json.loads(archive.read(member))
            decks = setup_decks(payload)
            for seat, deck in enumerate(decks):
                if deck is None or SLOWKING not in deck:
                    continue
                fingerprint = deck_fingerprint(deck)
                list_counts[fingerprint] += 1
                lists[fingerprint] = deck
                team_names = list((payload.get("info") or {}).get("TeamNames") or ("", ""))
                while len(team_names) < 2:
                    team_names.append("")
                reward = int((payload.get("rewards") or [0, 0])[seat])
                frames = acting_frames(payload, seat)
                first_player = None
                for _env_step, entry in frames:
                    current = (entry.get("observation") or {}).get("current") or {}
                    if current.get("firstPlayer") in (0, 1):
                        first_player = int(current["firstPlayer"])
                        break
                game = {
                    "episode_id": str((payload.get("info") or {}).get("EpisodeId") or Path(member).stem),
                    "seat": seat,
                    "team_name": team_names[seat],
                    "opponent_team_name": team_names[1 - seat],
                    "deck_fingerprint": fingerprint,
                    "reward": reward,
                    "result": "win" if reward > 0 else "loss" if reward < 0 else "draw",
                    "turn_order": "first" if first_player == seat else "second" if first_player in (0, 1) else "unknown",
                    "decision_frames": len(frames),
                    "seek_sources": [],
                }
                opponent_results[team_names[1 - seat]][game["result"]] += 1
                turn_tokens: dict[int, list[str]] = defaultdict(list)
                for env_step, entry in frames:
                    observation = entry.get("observation") or {}
                    current = observation.get("current") or {}
                    select = observation.get("select") or {}
                    options = select.get("option") or []
                    chosen = [options[index] for index in entry["action"]]
                    context = select.get("context")
                    effect = effect_id(select)
                    turn = int(current.get("turn", -1))
                    players = current.get("players") or [{}, {}]
                    me = players[seat] if len(players) == 2 else {}
                    active = (me.get("active") or [None])[0]
                    active_id = card_id(active)

                    for option in chosen:
                        option_type = int(option.get("type", -1))
                        resolved = resolve_card(observation, option)
                        resolved_id = card_id(resolved)
                        confirmed = action_confirmed(
                            payload, seat, env_step, option, resolved_id
                        )
                        token = OPTION_NAMES.get(option_type, f"option_{option_type}")
                        if context == 1 and resolved_id is not None and confirmed:
                            initial_active[resolved_id] += 1
                        if context == 2 and resolved_id is not None and confirmed:
                            initial_bench[resolved_id] += 1
                        if context == 0 and confirmed:
                            main_actions[token] += 1
                            if option_type == OPT_PLAY and resolved_id is not None:
                                played_cards[resolved_id] += 1
                                token += ":" + names.get(resolved_id, str(resolved_id))
                            elif option_type == OPT_ATTACH:
                                if resolved_id is not None:
                                    attached_energy[resolved_id] += 1
                                    token += ":" + names.get(resolved_id, str(resolved_id))
                                target_id = card_id(resolve_target(observation, option))
                                if target_id is not None:
                                    attachment_targets[target_id] += 1
                                    token += "->" + names.get(target_id, str(target_id))
                            elif option_type == OPT_EVOLVE:
                                if resolved_id is not None:
                                    evolved_cards[resolved_id] += 1
                                    token += ":" + names.get(resolved_id, str(resolved_id))
                            elif option_type == OPT_ABILITY:
                                if resolved_id is not None:
                                    ability_sources[resolved_id] += 1
                                    token += ":" + names.get(resolved_id, str(resolved_id))
                            elif option_type == OPT_ATTACK:
                                if active_id is not None:
                                    attack_users[active_id] += 1
                                    token += ":" + names.get(active_id, str(active_id))
                                attack_id = option.get("attackId")
                                if active_id == SLOWKING and attack_id == SEEK_INSPIRATION_ATTACK:
                                    source, copied = copied_attack(payload, seat, env_step)
                                    if source is not None:
                                        seek_sources[source] += 1
                                        game["seek_sources"].append(source)
                                    if copied is not None:
                                        seek_copied_attacks[copied] += 1
                            turn_tokens[turn].append(token)
                        if resolved_id is not None and context != 0:
                            # Selection prompts are confirmed by a later move of
                            # the selected card.  The generic confirmation above
                            # is main-action oriented, so check the following logs
                            # for the exact selected identity here.
                            selected_confirmed = False
                            for future in (payload.get("steps") or [])[env_step + 1 : env_step + 5]:
                                if not isinstance(future, list) or seat >= len(future):
                                    continue
                                future_obs = (future[seat].get("observation") or {}) if isinstance(future[seat], dict) else {}
                                if any(
                                    isinstance(log, dict)
                                    and log.get("playerIndex") == seat
                                    and log.get("cardId") == resolved_id
                                    and log.get("type") in {6, 11, 12}
                                    for log in (future_obs.get("logs") or [])
                                ):
                                    selected_confirmed = True
                                    break
                            if context == 9 and effect == ACADEMY_AT_NIGHT and selected_confirmed:
                                academy_targets[resolved_id] += 1
                            if effect is not None and selected_confirmed:
                                search_targets[effect][resolved_id] += 1
                for tokens in turn_tokens.values():
                    for index in range(len(tokens) - 1):
                        turn_bigrams[" > ".join(tokens[index : index + 2])] += 1
                    for index in range(len(tokens) - 2):
                        turn_trigrams[" > ".join(tokens[index : index + 3])] += 1
                games.append(game)
            if member_index % 500 == 0:
                print(f"scanned={member_index} slowking_games={len(games)}", flush=True)

    wins = sum(game["result"] == "win" for game in games)
    losses = sum(game["result"] == "loss" for game in games)
    draws = len(games) - wins - losses
    turn_order: dict[str, dict[str, Any]] = {}
    for order in ("first", "second", "unknown"):
        subset = [game for game in games if game["turn_order"] == order]
        subset_wins = sum(game["result"] == "win" for game in subset)
        turn_order[order] = {
            "games": len(subset),
            "wins": subset_wins,
            "win_rate": subset_wins / len(subset) if subset else None,
        }
    deck_rows = []
    for fingerprint, count in list_counts.most_common():
        counts = Counter(lists[fingerprint])
        deck_rows.append(
            {
                "fingerprint": fingerprint,
                "games": count,
                "card_count": sum(counts.values()),
                "cards": [
                    {"card_id": value, "card_name": names.get(value, f"card_{value}"), "count": amount}
                    for value, amount in sorted(counts.items())
                ],
            }
        )
    opponent_rows = []
    for opponent, results in opponent_results.items():
        total = sum(results.values())
        opponent_rows.append(
            {
                "opponent_team_name": opponent,
                "games": total,
                "wins": results["win"],
                "losses": results["loss"],
                "draws": results["draw"],
                "win_rate": results["win"] / total if total else 0.0,
            }
        )
    opponent_rows.sort(key=lambda row: (-row["games"], row["opponent_team_name"]))

    output = {
        "schema": "poke_bot.slowking_top_replay_distillation/v1",
        "status": "research_only_no_training_or_runtime_authority",
        "source": {
            "date": args.source_date,
            "archive": str(args.archive.resolve()),
            "archive_sha256": digest_file(args.archive),
            "archive_json_members": len(members),
            "manifest_episode_count": args.manifest_episode_count,
            "manifest_minus_archive_members": (
                args.manifest_episode_count - len(members)
                if args.manifest_episode_count is not None
                else None
            ),
            "episodes_scanned": len(members),
        },
        "identity": {
            "unique_deck_lists": len(deck_rows),
            "unique_team_names": sorted({game["team_name"] for game in games}),
            "decks": deck_rows,
        },
        "outcomes": {
            "games": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / len(games) if games else None,
            "turn_order": turn_order,
            "by_opponent_team": opponent_rows,
        },
        "behavior": {
            "initial_active": counter_rows(initial_active, names),
            "initial_bench": counter_rows(initial_bench, names),
            "main_action_types": counter_rows(main_actions, names),
            "played_cards": counter_rows(played_cards, names),
            "attached_energy": counter_rows(attached_energy, names),
            "attachment_targets": counter_rows(attachment_targets, names),
            "evolved_cards": counter_rows(evolved_cards, names),
            "ability_sources": counter_rows(ability_sources, names),
            "academy_targets": counter_rows(academy_targets, names),
            "effect_selected_cards": {
                str(effect): {
                    "effect_card_name": names.get(effect, f"card_{effect}"),
                    "targets": counter_rows(counter, names),
                }
                for effect, counter in sorted(search_targets.items())
            },
            "seek_sources": counter_rows(seek_sources, names),
            "seek_copied_attack_ids": counter_rows(seek_copied_attacks, names),
            "attack_users": counter_rows(attack_users, names),
            "top_turn_bigrams": counter_rows(Counter(dict(turn_bigrams.most_common(50))), names),
            "top_turn_trigrams": counter_rows(Counter(dict(turn_trigrams.most_common(50))), names),
        },
        "card_moves": {str(card): move_names for card, move_names in sorted(moves.items()) if card in set(lists[next(iter(lists))])} if lists else {},
        "games": games,
        "limitations": [
            "Observational replay evidence cannot identify unchosen counterfactual actions.",
            "Opponent team names are not equivalent to normalized deck archetypes.",
            "Win rate is conditional on this public daily opponent mix, not a formal paired gate.",
            "Repeated deterministic actions may reflect simulator or policy implementation artifacts.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "games": len(games), "wins": wins, "unique_lists": len(deck_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
