from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

from poke_agent.features import features_from_observation
from poke_agent.game_tracker import GameEventTracker
from poke_agent.rewards import assign_episode_values
from poke_agent.simulator import SimulatorState


def make_random_agent(to_observation_class: Callable[..., Any]) -> Callable[[dict], list[int]]:
    def random_agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        options = list(range(len(obs.select.option)))
        return random.sample(options, min(obs.select.maxCount, len(options)))

    return random_agent


def json_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, separators=(",", ":")))


def play_episode(
    episode: int,
    deck: list[int],
    simulator: SimulatorState,
    agent: Callable[[dict], list[int]],
    *,
    deck0_name: str = "deck0",
    deck1_name: str = "deck1",
    max_steps: int = 300,
) -> list[dict]:
    if not simulator.available or simulator.battle_start is None or simulator.battle_select is None or simulator.battle_finish is None:
        raise RuntimeError("CABT simulator is not available")

    rows: list[dict] = []
    tracker = GameEventTracker()
    obs, start_data = simulator.battle_start(deck, deck)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"deck error type={start_data.errorType} player={start_data.errorPlayer}")
    try:
        step = 0
        while obs["current"]["result"] < 0 and step < max_steps:
            select = obs.get("select") or {}
            options = select.get("option") or []
            action = agent(obs)
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
                "player": int(obs["current"]["yourIndex"]),
                "deck0": deck0_name,
                "deck1": deck1_name,
            })
            obs = next_obs
            step += 1
        result = int(obs["current"]["result"])
        assign_episode_values(rows, result, terminal_obs=obs)
        return rows
    finally:
        simulator.battle_finish()


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
