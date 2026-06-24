from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from poke_agent.archetypes import ArchetypeRegistry, load_archetype_registry, slug_from_deck_name
from poke_agent.features import features_from_observation
from poke_agent.game_tracker import GameEventTracker
from poke_agent.rewards import assign_episode_values, is_complete_episode, is_timeout_observation


def _is_option_index_action(action: Any, option_count: int) -> bool:
    if not isinstance(action, list) or not action:
        return False
    if option_count <= 0:
        return False
    return all(isinstance(value, int) and 0 <= value < option_count for value in action)


def _extract_setup_deck(action: list[Any]) -> list[int] | None:
    if not action:
        return None
    if all(isinstance(value, int) and value > 20 for value in action):
        if len(action) >= 60:
            return [int(value) for value in action[:60]]
    return None


def _agent_names(payload: dict[str, Any]) -> tuple[str, str]:
    info = payload.get("info") or {}
    team_names = info.get("TeamNames") or []
    if len(team_names) >= 2:
        return str(team_names[0]), str(team_names[1])
    agents = info.get("Agents") or []
    if len(agents) >= 2:
        return str((agents[0] or {}).get("Name") or "agent0"), str((agents[1] or {}).get("Name") or "agent1")
    return "agent0", "agent1"


def _terminal_observation(payload: dict[str, Any]) -> dict[str, Any] | None:
    steps = payload.get("steps") or []
    if not steps:
        return None
    last = steps[-1]
    if not isinstance(last, list):
        return None
    for entry in last[:2]:
        observation = entry.get("observation")
        if observation:
            return observation
    return None


def _final_result(payload: dict[str, Any]) -> int:
    from poke_agent.rewards import resolve_game_result

    terminal_obs = _terminal_observation(payload)
    if terminal_obs is not None:
        raw = int((terminal_obs.get("current") or {}).get("result", -1))
        if raw >= 0:
            return resolve_game_result(raw, terminal_obs)
    rewards = payload.get("rewards") or []
    if len(rewards) >= 2:
        if rewards[0] > rewards[1]:
            return 0
        if rewards[1] > rewards[0]:
            return 1
    return 2


def is_replay_complete(payload: dict[str, Any]) -> bool:
    result = _final_result(payload)
    terminal_obs = _terminal_observation(payload)
    return is_complete_episode(result, terminal_obs=terminal_obs, truncated=False)


def replay_to_rollout_rows(
    payload: dict[str, Any],
    *,
    episode: int,
    registry: ArchetypeRegistry | None = None,
    deck0_name: str | None = None,
    deck1_name: str | None = None,
    source: str = "",
    source_episode_id: str = "",
    value_win: float = 1.0,
    value_not_win: float = -1.0,
    value_timeout: float = -2.0,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    if require_complete and not is_replay_complete(payload):
        return []

    registry = registry or load_archetype_registry(Path("."))
    agent0_name, agent1_name = _agent_names(payload)
    setup_decks: list[list[int] | None] = [None, None]

    steps = payload.get("steps") or []
    rows: list[dict[str, Any]] = []

    for step_index, step in enumerate(steps):
        if not isinstance(step, list):
            continue
        next_step = steps[step_index + 1] if step_index + 1 < len(steps) else None

        for agent_index, entry in enumerate(step[:2]):
            observation = entry.get("observation") or {}
            select = observation.get("select") or {}
            options = select.get("option") or []
            action = entry.get("action") or []

            setup = _extract_setup_deck(action if isinstance(action, list) else [])
            if setup is not None:
                setup_decks[agent_index] = setup
                continue

            if not _is_option_index_action(action, len(options)):
                continue

            next_observation = observation
            if next_step and isinstance(next_step, list) and agent_index < len(next_step):
                next_observation = next_step[agent_index].get("observation") or observation

            current = observation.get("current") or {}
            terminal = int((next_observation.get("current") or {}).get("result", -1)) >= 0

            tracker = GameEventTracker()
            step_features, _ = features_from_observation(observation, tracker)
            next_tracker = copy.deepcopy(tracker)
            next_features, _ = features_from_observation(next_observation, next_tracker)
            step_features = [float(v) for v in step_features]
            next_features = [float(v) for v in next_features]

            deck0_slug = deck0_name or slug_from_deck_name(agent0_name, registry)
            deck1_slug = deck1_name or slug_from_deck_name(agent1_name, registry)
            if setup_decks[0] is not None:
                deck0_slug, _ = registry.classify_deck(setup_decks[0])
            if setup_decks[1] is not None:
                deck1_slug, _ = registry.classify_deck(setup_decks[1])

            rows.append({
                "episode": episode,
                "step": len(rows),
                "features": step_features,
                "next_features": next_features,
                "observation": observation,
                "action": action,
                "next_observation": next_observation,
                "legal_action_count": len(options),
                "select_min_count": int(select.get("minCount", 0)),
                "select_max_count": int(select.get("maxCount", 0)),
                "terminal": terminal,
                "reward": float(entry.get("reward") or 0.0),
                "player": int(current.get("yourIndex", agent_index)),
                "deck0": deck0_slug,
                "deck1": deck1_slug,
                "deck0_cards": list(setup_decks[0]) if setup_decks[0] is not None else None,
                "deck1_cards": list(setup_decks[1]) if setup_decks[1] is not None else None,
                "source": source,
                "source_episode_id": source_episode_id,
                "truncated": False,
            })

    if not rows:
        return []

    result = _final_result(payload)
    terminal_obs = rows[-1]["next_observation"] if rows else _terminal_observation(payload)
    assign_episode_values(
        rows,
        result,
        value_win=value_win,
        value_not_win=value_not_win,
        value_timeout=value_timeout,
        terminal_obs=terminal_obs,
    )
    if is_timeout_observation(terminal_obs):
        for row in rows:
            row["complete"] = False
    else:
        for row in rows:
            row["complete"] = is_complete_episode(result, terminal_obs=terminal_obs, truncated=False)
    return rows


def load_replay_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_replay_file(
    replay_path: Path,
    *,
    episode: int,
    registry: ArchetypeRegistry | None = None,
    root: Path | None = None,
    source: str = "",
    rewards: dict[str, float] | None = None,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    root = root or Path(".")
    registry = registry or load_archetype_registry(root)
    payload = load_replay_payload(replay_path)
    episode_id = str((payload.get("info") or {}).get("EpisodeId") or replay_path.stem)
    reward_cfg = rewards or {}
    return replay_to_rollout_rows(
        payload,
        episode=episode,
        registry=registry,
        source=source or replay_path.parent.name,
        source_episode_id=episode_id,
        value_win=float(reward_cfg.get("value_win", 1.0)),
        value_not_win=float(reward_cfg.get("value_not_win", -1.0)),
        value_timeout=float(reward_cfg.get("value_timeout", -2.0)),
        require_complete=require_complete,
    )
