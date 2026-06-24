from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cg.api import search_begin, search_end, search_step, to_observation_class
from features import encode_observation_step, stable_hash_index
from game_tracker import GameEventTracker
from policy_runtime import legal_actions

if TYPE_CHECKING:
    from policy_runtime import TrainedPolicyAgent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


BEAM_WIDTH = _env_int("BEAM_WIDTH", 8)
BEAM_TIME_BUDGET_MS = _env_int("BEAM_TIME_BUDGET_MS", 10_000)
BEAM_MIN_REMAINING_SEC = _env_int("BEAM_MIN_REMAINING_SEC", 120)

OPPONENT_FILLER_CARD = 1072
OPPONENT_ENERGY_CARD = 1


@dataclass
class BeamSearchConfig:
    width: int = BEAM_WIDTH
    time_budget_ms: int = BEAM_TIME_BUDGET_MS
    min_remaining_sec: int = BEAM_MIN_REMAINING_SEC


def remaining_overage_seconds(obs_dict: dict[str, Any]) -> float | None:
    remaining = obs_dict.get("remainingOverageTime")
    if remaining is None:
        return None
    return float(remaining)


def should_skip_beam_search(obs_dict: dict[str, Any], config: BeamSearchConfig) -> bool:
    if not obs_dict.get("search_begin_input"):
        return True
    remaining = remaining_overage_seconds(obs_dict)
    if remaining is not None and remaining < config.min_remaining_sec:
        return True
    return False


def guess_hidden_cards(obs_dict: dict[str, Any], our_deck: list[int]) -> tuple[list[int], ...]:
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
    agent: TrainedPolicyAgent,
    logits: np.ndarray,
    actions: list[list[int]],
) -> list[tuple[float, list[int]]]:
    ranked: list[tuple[float, list[int]]] = []
    for action in actions:
        action_key = json.dumps(action, sort_keys=True, separators=(",", ":"))
        action_class = stable_hash_index(action_key, agent._policy_dim)
        ranked.append((float(logits[action_class]), action))
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


def _winner_from_prizes(leaf_obs_dict: dict[str, Any]) -> int | None:
    players = (leaf_obs_dict.get("current") or {}).get("players") or [{}, {}]

    def prize_count(player_index: int) -> int:
        if player_index >= len(players):
            return 6
        player = players[player_index] or {}
        prize = player.get("prize")
        if isinstance(prize, list):
            return len(prize)
        raw = player.get("prizeCount")
        return int(raw) if raw is not None else 6

    p0 = prize_count(0)
    p1 = prize_count(1)
    if p0 == 0 and p1 > 0:
        return 0
    if p1 == 0 and p0 > 0:
        return 1
    return None


def evaluate_search_leaf(
    agent: TrainedPolicyAgent,
    leaf_obs_dict: dict[str, Any],
    root_your_index: int,
    our_deck: list[int] | None = None,
) -> float:
    assert agent._model is not None
    assert agent._feature_mean is not None and agent._feature_std is not None

    leaf_tracker = GameEventTracker()
    leaf_features, leaf_cards = encode_observation_step(
        leaf_obs_dict,
        leaf_tracker,
        state_hash_dim=agent._state_hash_dim,
        our_deck=our_deck,
        card_vocab_size=agent._card_vocab_size,
    )
    leaf_norm = ((leaf_features.reshape(-1) - agent._feature_mean) / agent._feature_std).astype(np.float32)
    value = agent._model_value(leaf_features=leaf_norm, leaf_cards=leaf_cards)

    current = leaf_obs_dict.get("current") or {}
    if int(current.get("yourIndex", root_your_index)) != root_your_index:
        value = -value
    if int(current.get("result", -1)) >= 0:
        prize_winner = _winner_from_prizes(leaf_obs_dict)
        winner = prize_winner if prize_winner is not None else int(current.get("result"))
        if winner == root_your_index:
            value = max(value, 1.0)
        else:
            value = min(value, -1.0)
    return value


def expand_action_in_search(
    agent: TrainedPolicyAgent,
    obs_dict: dict[str, Any],
    our_deck: list[int],
    action: list[int],
    root_your_index: int,
    deadline: float,
) -> float:
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
                return evaluate_search_leaf(agent, leaf_obs, root_your_index, our_deck)
            select = leaf_obs.get("select") or {}
            options = select.get("option") or []
            if not options:
                return evaluate_search_leaf(agent, leaf_obs, root_your_index, our_deck)
            min_count = int(select.get("minCount", 1))
            max_count = int(select.get("maxCount", 1))
            fallback = legal_actions(len(options), min_count, max_count)
            action = fallback[0] if fallback else [0]
        return evaluate_search_leaf(
            agent,
            observation_to_dict(current_state.observation),
            root_your_index,
            our_deck,
        )
    finally:
        search_end()


def run_beam_search(
    agent: TrainedPolicyAgent,
    obs_dict: dict[str, Any],
    our_deck: list[int],
    actions: list[list[int]],
    root_your_index: int,
    config: BeamSearchConfig | None = None,
) -> list[int]:
    config = config or BeamSearchConfig()
    deadline = time.perf_counter() + (config.time_budget_ms / 1000.0)
    logits = agent._model_logits()
    ranked = rank_actions_by_policy(agent, logits, actions)[: max(1, config.width)]

    best_action = ranked[0][1]
    best_value = float("-inf")
    for policy_score, action in ranked:
        if time.perf_counter() >= deadline:
            break
        try:
            leaf_value = expand_action_in_search(
                agent,
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
