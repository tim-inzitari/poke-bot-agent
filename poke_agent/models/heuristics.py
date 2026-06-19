from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class CabtAgent(Protocol):
    def __call__(self, obs_dict: dict[str, Any]) -> list[int]:
        ...


@dataclass(frozen=True)
class HeuristicAgent:
    name: str
    strategy: str
    fn: Callable[[dict[str, Any]], list[int]]

    def __call__(self, obs_dict: dict[str, Any]) -> list[int]:
        return self.fn(obs_dict)


def _parse_observation(obs_dict: dict[str, Any], to_observation_class: Callable[..., Any]):
    return to_observation_class(obs_dict)


def make_random_agent(to_observation_class: Callable[..., Any]) -> HeuristicAgent:
    def act(obs_dict: dict[str, Any]) -> list[int]:
        obs = _parse_observation(obs_dict, to_observation_class)
        options = list(range(len(obs.select.option)))
        return random.sample(options, min(obs.select.maxCount, len(options)))

    return HeuristicAgent(name="random", strategy="random", fn=act)


def make_first_legal_agent(to_observation_class: Callable[..., Any]) -> HeuristicAgent:
    def act(obs_dict: dict[str, Any]) -> list[int]:
        obs = _parse_observation(obs_dict, to_observation_class)
        count = min(obs.select.maxCount, len(obs.select.option))
        return list(range(count))

    return HeuristicAgent(name="first_legal", strategy="first_legal", fn=act)


def make_max_option_agent(to_observation_class: Callable[..., Any]) -> HeuristicAgent:
    def act(obs_dict: dict[str, Any]) -> list[int]:
        obs = _parse_observation(obs_dict, to_observation_class)
        options = list(range(len(obs.select.option)))
        count = min(obs.select.maxCount, len(options))
        return options[:count]

    return HeuristicAgent(name="max_option", strategy="max_option", fn=act)


HEURISTIC_BUILDERS = {
    "random": make_random_agent,
    "first_legal": make_first_legal_agent,
    "max_option": make_max_option_agent,
}
