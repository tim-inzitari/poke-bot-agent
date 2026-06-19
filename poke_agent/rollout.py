from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from poke_agent.simulator import SimulatorState


from poke_agent.features import features_from_observation
from poke_agent.game_tracker import GameEventTracker


def make_random_agent(to_observation_class: Callable[..., Any]) -> Callable[[dict], list[int]]:
    def random_agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        options = list(range(len(obs.select.option)))
        return random.sample(options, min(obs.select.maxCount, len(options)))

    return random_agent


def play_episode(
    episode: int,
    deck: list[int],
    simulator: SimulatorState,
    agent: Callable[[dict], list[int]],
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
            rows.append({
                "episode": episode,
                "step": step,
                "features": features_from_observation(obs, tracker),
                "player": int(obs["current"]["yourIndex"]),
            })
            obs = simulator.battle_select(agent(obs))
            step += 1
        result = int(obs["current"]["result"])
        for row in rows:
            row["value"] = 0.0 if result == 2 else (1.0 if row["player"] == result else -1.0)
        return rows
    finally:
        simulator.battle_finish()


def generate_rollouts(
    simulator: SimulatorState,
    deck: list[int],
    episodes: int,
    output_path: Path,
) -> int:
    if not simulator.available or simulator.to_observation_class is None:
        print("skipping CABT generation in this runtime")
        return 0

    if episodes <= 0:
        print("skipping CABT generation in this runtime")
        return 0

    agent = make_random_agent(simulator.to_observation_class)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for episode in range(episodes):
        rows.extend(play_episode(episode, deck, simulator, agent))
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"generated {len(rows)} rows -> {output_path}")
    return len(rows)
