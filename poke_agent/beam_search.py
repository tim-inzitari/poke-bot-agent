from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from poke_agent.actions import legal_actions
from poke_agent.attack_plan import score_actions_with_attack_plan
from poke_agent.features import encode_observation_step, stable_hash_index
from poke_agent.game_tracker import GameEventTracker

if TYPE_CHECKING:
    from poke_agent.policy_agent import PolicyRuntime, PolicySession


@dataclass
class BeamSearchConfig:
    width: int = 8
    time_budget_ms: int = 10_000
    min_remaining_sec: int = 120
    sim_mode: bool = False

    @classmethod
    def from_training_config(cls, config: dict[str, Any]) -> BeamSearchConfig:
        beam = dict(config.get("beam_search", {}))
        return cls(
            width=int(beam.get("width", 8)),
            time_budget_ms=int(beam.get("time_budget_ms", 10_000)),
            min_remaining_sec=0 if beam.get("sim_mode") else int(beam.get("min_remaining_sec", 120)),
            sim_mode=bool(beam.get("sim_mode", False)),
        )


OPPONENT_FILLER_CARD = 1072
OPPONENT_ENERGY_CARD = 1


def remaining_overage_seconds(obs_dict: dict[str, Any]) -> float | None:
    remaining = obs_dict.get("remainingOverageTime")
    if remaining is None:
        return None
    return float(remaining)


def should_skip_beam_search(obs_dict: dict[str, Any], config: BeamSearchConfig) -> bool:
    if not obs_dict.get("search_begin_input"):
        return True
    if config.sim_mode:
        return False
    remaining = remaining_overage_seconds(obs_dict)
    if remaining is not None and remaining < config.min_remaining_sec:
        return True
    return False


def guess_hidden_cards(obs_dict: dict[str, Any], our_deck: list[int]) -> tuple[list[int], ...]:
    from cg.api import to_observation_class

    obs = to_observation_class(obs_dict)
    state = obs.current
    your_index = state.yourIndex
    opponent_index = 1 - your_index
    your_player = state.players[your_index]
    opponent_player = state.players[opponent_index]

    deck_pool = list(our_deck) if our_deck else [OPPONENT_FILLER_CARD]
    your_deck_guess = random.sample(deck_pool, min(your_player.deckCount, len(deck_pool)))
    if len(your_deck_guess) < your_player.deckCount:
        your_deck_guess.extend(random.choices(deck_pool, k=your_player.deckCount - len(your_deck_guess)))

    prize_count = len(your_player.prize)
    your_prize_guess = random.sample(deck_pool, min(prize_count, len(deck_pool)))
    if len(your_prize_guess) < prize_count:
        your_prize_guess.extend(random.choices(deck_pool, k=prize_count - len(your_prize_guess)))

    opponent_deck_guess = [OPPONENT_FILLER_CARD] * opponent_player.deckCount
    opponent_prize_guess = [OPPONENT_ENERGY_CARD] * len(opponent_player.prize)
    opponent_hand_guess = [OPPONENT_ENERGY_CARD] * opponent_player.handCount

    opponent_active: list[int] = []
    active = opponent_player.active
    if len(active) > 0 and active[0] is None:
        opponent_active = [OPPONENT_FILLER_CARD]

    return (
        your_deck_guess,
        your_prize_guess,
        opponent_deck_guess,
        opponent_prize_guess,
        opponent_hand_guess,
        opponent_active,
    )


def rank_actions_by_policy(
    runtime: PolicyRuntime,
    logits: np.ndarray,
    actions: list[list[int]],
    obs_dict: dict[str, Any] | None = None,
) -> list[tuple[float, list[int]]]:
    ranked: list[tuple[float, list[int]]] = []
    plan_scores = score_actions_with_attack_plan(obs_dict, actions) if obs_dict is not None else [0.0 for _ in actions]
    max_abs_plan = max((abs(score) for score in plan_scores), default=0.0)
    for action in actions:
        index = len(ranked)
        action_key = json.dumps(action, sort_keys=True, separators=(",", ":"))
        action_class = stable_hash_index(action_key, runtime._policy_dim)
        policy_score = float(logits[action_class])
        plan_score = (plan_scores[index] / max_abs_plan) if max_abs_plan > 0 else 0.0
        ranked.append((policy_score + 0.35 * plan_score, action))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _normalize_json(value: Any) -> Any:
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_json(item) for key, item in value.items()}
    return value


def observation_to_dict(observation: Any) -> dict[str, Any]:
    if isinstance(observation, dict):
        return observation
    return _normalize_json(asdict(observation))


def evaluate_search_leaf(
    runtime: PolicyRuntime,
    session: PolicySession,
    leaf_obs_dict: dict[str, Any],
    root_your_index: int,
    our_deck: list[int] | None = None,
) -> float:
    assert runtime._model is not None
    assert runtime._feature_mean is not None and runtime._feature_std is not None

    leaf_tracker = GameEventTracker()
    leaf_features = encode_observation_step(
        leaf_obs_dict,
        leaf_tracker,
        state_hash_dim=runtime._state_hash_dim,
        our_deck=our_deck,
    ).reshape(-1)
    leaf_norm = ((leaf_features - runtime._feature_mean) / runtime._feature_std).astype(np.float32)

    window = (session.history + [leaf_norm])[-runtime._window_size :]
    pad_count = runtime._window_size - len(window)
    if pad_count > 0:
        pad = np.zeros_like(leaf_norm)
        window = [pad] * pad_count + window

    x = torch.tensor(np.stack(window), dtype=torch.float32, device=runtime.device).unsqueeze(0)
    mask = torch.ones((1, runtime._window_size), dtype=torch.float32, device=runtime.device)
    with torch.no_grad():
        value = float(runtime._model(x, mask)["value"].squeeze().cpu().item())

    current = leaf_obs_dict.get("current") or {}
    if int(current.get("yourIndex", root_your_index)) != root_your_index:
        value = -value
    if int(current.get("result", -1)) >= 0:
        if int(current.get("result")) == root_your_index:
            value = max(value, 1.0)
        else:
            value = min(value, -1.0)
    return value


def expand_action_in_search(
    runtime: PolicyRuntime,
    session: PolicySession,
    obs_dict: dict[str, Any],
    our_deck: list[int],
    action: list[int],
    root_your_index: int,
    deadline: float,
) -> float:
    from cg.api import search_begin, search_end, search_step, to_observation_class

    obs = to_observation_class(obs_dict)
    (
        your_deck_guess,
        your_prize_guess,
        opponent_deck_guess,
        opponent_prize_guess,
        opponent_hand_guess,
        opponent_active,
    ) = guess_hidden_cards(obs_dict, our_deck)

    search_state = search_begin(
        obs,
        your_deck=your_deck_guess,
        your_prize=your_prize_guess,
        opponent_deck=opponent_deck_guess,
        opponent_prize=opponent_prize_guess,
        opponent_hand=opponent_hand_guess,
        opponent_active=opponent_active,
    )
    current_state = search_state
    try:
        while time.perf_counter() < deadline:
            current_state = search_step(current_state.searchId, action)
            leaf_obs = observation_to_dict(current_state.observation)
            current = leaf_obs.get("current") or {}
            if int(current.get("result", -1)) >= 0:
                return evaluate_search_leaf(runtime, session, leaf_obs, root_your_index, our_deck)
            select = leaf_obs.get("select") or {}
            options = select.get("option") or []
            if not options:
                return evaluate_search_leaf(runtime, session, leaf_obs, root_your_index, our_deck)
            min_count = int(select.get("minCount", 1))
            max_count = int(select.get("maxCount", 1))
            fallback = legal_actions(len(options), min_count, max_count)
            action = fallback[0] if fallback else [0]
        return evaluate_search_leaf(
            runtime,
            session,
            observation_to_dict(current_state.observation),
            root_your_index,
            our_deck,
        )
    finally:
        search_end()


def run_beam_search(
    runtime: PolicyRuntime,
    obs_dict: dict[str, Any],
    session: PolicySession,
    our_deck: list[int],
    actions: list[list[int]],
    root_your_index: int,
    config: BeamSearchConfig | None = None,
) -> list[int]:
    config = config or BeamSearchConfig()
    deadline = time.perf_counter() + (config.time_budget_ms / 1000.0)
    logits = runtime._model_logits(session)
    ranked = rank_actions_by_policy(runtime, logits, actions, obs_dict)[: max(1, config.width)]

    best_action = ranked[0][1]
    best_value = float("-inf")
    for policy_score, action in ranked:
        if time.perf_counter() >= deadline:
            break
        try:
            leaf_value = expand_action_in_search(
                runtime,
                session,
                obs_dict,
                our_deck,
                action,
                root_your_index,
                deadline,
            )
        except Exception:
            leaf_value = policy_score
        if leaf_value > best_value:
            best_value = leaf_value
            best_action = action
    return best_action
