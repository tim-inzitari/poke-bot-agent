from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

from poke_agent.features import features_from_observation
from poke_agent.game_tracker import GameEventTracker
from poke_agent.rewards import assign_episode_values, is_complete_episode
from poke_agent.simulator import SimulatorState


def make_random_agent(to_observation_class: Callable[..., Any]) -> Callable[[dict], list[int]]:
    def random_agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        options = list(range(len(obs.select.option)))
        return random.sample(options, min(obs.select.maxCount, len(options)))

    return random_agent


def json_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, separators=(",", ":")))


def play_match(
    episode: int,
    deck0: list[int],
    deck1: list[int],
    simulator: SimulatorState,
    agent0: Callable[[dict], list[int]],
    agent1: Callable[[dict], list[int]],
    *,
    deck0_name: str = "deck0",
    deck1_name: str = "deck1",
    max_steps: int = 300,
    rewards: dict[str, float] | None = None,
) -> list[dict]:
    """Play one CABT game between two seat-specific agents."""
    if not simulator.available or simulator.battle_start is None or simulator.battle_select is None or simulator.battle_finish is None:
        raise RuntimeError("CABT simulator is not available")

    rows: list[dict] = []
    tracker = GameEventTracker()
    obs, start_data = simulator.battle_start(deck0, deck1)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"deck error type={start_data.errorType} player={start_data.errorPlayer}")
    try:
        reward_cfg = rewards or {}
        step = 0
        truncated = False
        while obs["current"]["result"] < 0 and step < max_steps:
            select = obs.get("select") or {}
            options = select.get("option") or []
            player_index = int(obs["current"]["yourIndex"])
            action = agent0(obs) if player_index == 0 else agent1(obs)
            next_obs = simulator.battle_select(action)
            terminal = int((next_obs.get("current") or {}).get("result", -1)) >= 0
            next_tracker = copy.deepcopy(tracker)
            rows.append({
                "episode": episode,
                "step": step,
                "features": features_from_observation(obs, tracker),
                "next_features": features_from_observation(next_obs, next_tracker),
                "observation": json_snapshot(obs),
                "action": json_snapshot(action),
                "next_observation": json_snapshot(next_obs),
                "legal_action_count": len(options),
                "select_min_count": int(select.get("minCount", 0)),
                "select_max_count": int(select.get("maxCount", 0)),
                "terminal": terminal,
                "reward": 0.0,
                "player": player_index,
                "deck0": deck0_name,
                "deck1": deck1_name,
                "deck0_cards": list(deck0),
                "deck1_cards": list(deck1),
            })
            obs = next_obs
            step += 1
        if obs["current"]["result"] < 0:
            truncated = True
        result = int(obs["current"]["result"])
        if truncated and result < 0:
            return []
        if not is_complete_episode(result, terminal_obs=obs, truncated=truncated):
            return []
        assign_episode_values(
            rows,
            result,
            terminal_obs=obs,
            value_win=float(reward_cfg.get("value_win", 1.0)),
            value_not_win=float(reward_cfg.get("value_not_win", -1.0)),
            value_timeout=float(reward_cfg.get("value_timeout", -2.0)),
            value_per_own_prize_taken=float(
                reward_cfg.get("value_per_own_prize_taken", 1.0 / 6)
            ),
            value_per_opp_prize_taken=float(
                reward_cfg.get("value_per_opp_prize_taken", -1.0 / 6)
            ),
        )
        for row in rows:
            row["complete"] = True
            row["truncated"] = False
        return rows
    finally:
        simulator.battle_finish()


def play_episode(
    episode: int,
    deck: list[int],
    simulator: SimulatorState,
    agent: Callable[[dict], list[int]],
    *,
    deck0_name: str = "deck0",
    deck1_name: str = "deck1",
    max_steps: int = 300,
    rewards: dict[str, float] | None = None,
) -> list[dict]:
    return play_match(
        episode,
        deck,
        deck,
        simulator,
        agent,
        agent,
        deck0_name=deck0_name,
        deck1_name=deck1_name,
        max_steps=max_steps,
        rewards=rewards,
    )


def generate_rollouts(
    simulator: SimulatorState,
    deck: list[int],
    episodes: int,
    output_path: Path,
    *,
    deck_name: str | None = None,
) -> int:
    if not simulator.available or simulator.to_observation_class is None:
        print("skipping CABT generation in this runtime")
        return 0

    if episodes <= 0:
        print("skipping CABT generation in this runtime")
        return 0

    label = deck_name or "deck"
    agent = make_random_agent(simulator.to_observation_class)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for episode in range(episodes):
        rows.extend(
            play_episode(
                episode,
                deck,
                simulator,
                agent,
                deck0_name=label,
                deck1_name=label,
            )
        )
    print(f"generated {len(rows):,} rows from {episodes:,} games -> {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)
