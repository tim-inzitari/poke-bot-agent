"""Build decision-level Slowking distill corpora from episode archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from poke_bot.slowking_reverse_engineered_policy import (
    audit_decision,
    is_slowking_archetype,
)

from .authority import DECISION_SCHEMA, RESEARCH_ONLY, TRAINING_AUTHORITY
from .day_split import game_id


SLOWKING = 163


def _setup_decks(payload: dict[str, Any]) -> list[list[int] | None]:
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


def _acting_frames(payload: dict[str, Any], seat: int) -> list[tuple[int, dict[str, Any]]]:
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
    dedup: dict[tuple[int, int, int], tuple[int, dict[str, Any]]] = {}
    for env_step, entry in frames:
        observation = entry.get("observation") or {}
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        key = (
            int(current.get("turn", -1)),
            int(current.get("turnActionCount", -1)),
            int(select.get("context", -1)),
        )
        dedup[key] = (env_step, entry)
    return sorted(dedup.values(), key=lambda row: row[0])


def _enumerate_single_combos(select: dict[str, Any]) -> list[list[int]]:
    options = select.get("option") or []
    lo = select.get("minCount", 1)
    hi = select.get("maxCount", 1)
    if lo == 1 and hi == 1 and isinstance(options, list):
        return [[i] for i in range(len(options))]
    # Keep multi-select as the chosen action only (research corpus rows).
    return []


def iter_decisions_from_episode(
    payload: dict[str, Any],
    *,
    source_date: str,
    episode_id: str,
) -> Iterator[dict[str, Any]]:
    """Yield research decision tuples for Slowking seats (include losses)."""
    decks = _setup_decks(payload)
    team_names = list((payload.get("info") or {}).get("TeamNames") or ("", ""))
    while len(team_names) < 2:
        team_names.append("")
    rewards = list(payload.get("rewards") or [0, 0])
    while len(rewards) < 2:
        rewards.append(0)

    for seat, deck in enumerate(decks):
        if deck is None or SLOWKING not in deck or not is_slowking_archetype(deck):
            continue
        reward = int(rewards[seat])
        frames = _acting_frames(payload, seat)
        first_player = None
        for _env_step, entry in frames:
            current = (entry.get("observation") or {}).get("current") or {}
            if current.get("firstPlayer") in (0, 1):
                first_player = int(current["firstPlayer"])
                break
        game = {
            "episode_id": episode_id,
            "seat": seat,
            "source_date": source_date,
            "team_name": team_names[seat],
            "opponent_team_name": team_names[1 - seat],
            "deck": list(deck),
            "reward": reward,
            "result": "win" if reward > 0 else "loss" if reward < 0 else "draw",
            "turn_order": (
                "first"
                if first_player == seat
                else "second"
                if first_player in (0, 1)
                else "unknown"
            ),
        }
        gid = game_id(game)
        for env_step, entry in frames:
            observation = entry.get("observation") or {}
            select = observation.get("select") or {}
            action = list(entry["action"])
            options = select.get("option") or []
            if not options:
                continue
            legal = _enumerate_single_combos(select)
            if not legal:
                # Store the chosen combo alone when multi-select.
                legal = [action]
            try:
                selected_index = legal.index(action)
            except ValueError:
                legal = [action] + [c for c in legal if c != action]
                selected_index = 0
            heuristic = audit_decision(observation, legal, deck=deck)
            yield {
                "schema": DECISION_SCHEMA,
                "research_only": RESEARCH_ONLY,
                "training_authority": TRAINING_AUTHORITY,
                "game_id": gid,
                "source_date": source_date,
                "episode_id": episode_id,
                "seat": seat,
                "env_step": env_step,
                "team_name": game["team_name"],
                "deck": list(deck),
                "result": game["result"],
                "reward": reward,
                "value_target": float(reward),
                "turn_order": game["turn_order"],
                "observation": observation,
                "action": action,
                "legal_action_combos": legal,
                "selected_index": selected_index,
                "legal_action_count": len(legal),
                "heuristic": (
                    None
                    if heuristic is None
                    else {
                        "stage_class": heuristic.get("stage_class"),
                        "scores": heuristic.get("scores"),
                        "preferred_combo_index": heuristic.get("preferred_combo_index"),
                        "margin": heuristic.get("margin"),
                        "combo_rule_ids": heuristic.get("combo_rule_ids"),
                        "abstained": False,
                    }
                ),
                "heuristic_abstained": heuristic is None,
            }


def extract_archive_to_jsonl(
    archive: Path,
    *,
    source_date: str,
    out_jsonl: Path,
    max_episodes: int = 0,
) -> dict[str, int]:
    """Scan one daily ZIP and append Slowking decision rows to JSONL."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_episodes = 0
    n_games = 0
    n_decisions = 0
    seen_games: set[str] = set()
    with zipfile.ZipFile(archive) as zf, out_jsonl.open("a", encoding="utf-8") as out:
        members = sorted(name for name in zf.namelist() if name.endswith(".json"))
        for member in members:
            if max_episodes and n_episodes >= max_episodes:
                break
            payload = json.loads(zf.read(member))
            n_episodes += 1
            episode_id = str(
                (payload.get("info") or {}).get("EpisodeId") or Path(member).stem
            )
            for row in iter_decisions_from_episode(
                payload, source_date=source_date, episode_id=episode_id
            ):
                if row["game_id"] not in seen_games:
                    seen_games.add(row["game_id"])
                    n_games += 1
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                n_decisions += 1
    return {
        "episodes_scanned": n_episodes,
        "slowking_games": n_games,
        "decisions": n_decisions,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_split(
    rows: Iterable[dict[str, Any]],
    *,
    game_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("game_id")) in game_ids]
