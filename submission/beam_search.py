from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from cg.api import search_begin, search_end, search_step, to_observation_class
from features import encode_observation_step, stable_hash_index
from game_tracker import GameEventTracker
from policy_runtime import legal_actions
from rewards import winner_from_prizes

if TYPE_CHECKING:
    from policy_runtime import TrainedPolicyAgent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


# Kaggle inference: 5s/move keeps the per-side overage bank (default 600s) from draining
# in long games — a 10s budget burned out late and forced unsearched "whimper" moves.
# Matches poke_agent/config.py BEAM_TIME_BUDGET_MS.
BEAM_WIDTH = _env_int("BEAM_WIDTH", 12)
BEAM_TIME_BUDGET_MS = _env_int("BEAM_TIME_BUDGET_MS", 5_000)
BEAM_MIN_REMAINING_SEC = _env_int("BEAM_MIN_REMAINING_SEC", 120)
BEAM_MAX_SEARCH_STEPS = _env_int("BEAM_MAX_SEARCH_STEPS", 128)
BEAM_ROLLOUT_POLICY_WIDTH = _env_int("BEAM_ROLLOUT_POLICY_WIDTH", 12)
SEARCH_DETERMINIZATIONS = _env_int("SEARCH_DETERMINIZATIONS", 2)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


# Archetype game-plan policy prior — defaults match training (config.py) so Kaggle
# inference biases root action ranking identically to collection-time search.
HEURISTIC_POLICY_BETA_LUCARIO = _env_float("HEURISTIC_POLICY_BETA_LUCARIO", 0.35)
HEURISTIC_POLICY_BETA_DRAGAPULT = _env_float("HEURISTIC_POLICY_BETA_DRAGAPULT", 0.15)

OPPONENT_FILLER_CARD = 1072
OPPONENT_ENERGY_CARD = 1


@dataclass
class BeamSearchConfig:
    width: int = BEAM_WIDTH
    time_budget_ms: int = BEAM_TIME_BUDGET_MS
    min_remaining_sec: int = BEAM_MIN_REMAINING_SEC
    sim_mode: bool = False
    max_search_steps: int = BEAM_MAX_SEARCH_STEPS
    rollout_policy_width: int = BEAM_ROLLOUT_POLICY_WIDTH
    num_determinizations: int = SEARCH_DETERMINIZATIONS
    heuristic_policy_beta_lucario: float = HEURISTIC_POLICY_BETA_LUCARIO
    heuristic_policy_beta_dragapult: float = HEURISTIC_POLICY_BETA_DRAGAPULT


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


# Per-move budget never exceeds this fraction of the remaining overage bank, so long
# games taper search smoothly instead of hitting a hard floor and "whimpering".
_OVERAGE_BUDGET_FRACTION = 0.06
_MIN_ADAPTIVE_BUDGET_MS = 1_000


def effective_time_budget_ms(obs_dict: dict[str, Any], config: BeamSearchConfig) -> int:
    """Scale the per-move search budget down as the overage bank drains."""
    if config.sim_mode:
        return int(config.time_budget_ms)
    remaining = remaining_overage_seconds(obs_dict)
    if remaining is None:
        return int(config.time_budget_ms)
    # Reserve the safety floor; spend only a fraction of what is left above it.
    spendable_ms = max(0.0, (remaining - config.min_remaining_sec)) * 1000.0
    adaptive = spendable_ms * _OVERAGE_BUDGET_FRACTION
    capped = min(float(config.time_budget_ms), adaptive)
    return int(max(_MIN_ADAPTIVE_BUDGET_MS, capped))


def _hidden_card_seed(obs_dict: dict[str, Any]) -> int:
    current = obs_dict.get("current") or {}
    players = current.get("players") or []
    parts = [
        int(current.get("yourIndex", 0)),
        int(current.get("turn", 0)),
        int(current.get("phase", 0)),
    ]
    for player in players[:2]:
        if not isinstance(player, dict):
            continue
        parts.extend(
            [
                int(player.get("deckCount", 0)),
                int(player.get("handCount", 0)),
                len(player.get("prize") or []),
            ]
        )
    return stable_hash_index("|".join(str(part) for part in parts), 1_000_003)


def _deterministic_fill(pool: list[int], count: int, seed: int) -> list[int]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    if count <= len(pool):
        return rng.sample(pool, count)
    picked = list(pool)
    picked.extend(rng.choices(pool, k=count - len(pool)))
    return picked


def guess_hidden_cards(
    obs_dict: dict[str, Any],
    our_deck: list[int],
    *,
    seed_offset: int = 0,
) -> tuple[list[int], ...]:
    obs = to_observation_class(obs_dict)
    state = obs.current
    your_index = state.yourIndex
    opponent_index = 1 - your_index
    your_player = state.players[your_index]
    opponent_player = state.players[opponent_index]

    deck_pool = list(our_deck) if our_deck else [OPPONENT_FILLER_CARD]
    seed = _hidden_card_seed(obs_dict) + int(seed_offset)
    your_deck_guess = _deterministic_fill(deck_pool, your_player.deckCount, seed)
    your_prize_guess = _deterministic_fill(deck_pool, len(your_player.prize), seed + 1)

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
    *,
    heuristic_scorer: Any | None = None,
    beta: float = 0.0,
) -> list[tuple[float, list[int]]]:
    ranked: list[tuple[float, list[int]]] = []
    for action in actions:
        action_key = json.dumps(action, sort_keys=True, separators=(",", ":"))
        action_class = stable_hash_index(action_key, agent._policy_dim)
        ranked.append((float(logits[action_class]), action))

    if heuristic_scorer is not None and beta > 0.0 and len(ranked) > 1:
        values = np.array([score for score, _ in ranked], dtype=np.float64)
        low = float(values.min())
        high = float(values.max())
        span = (high - low) if high > low else 1.0
        ranked.sort(
            key=lambda item: (item[0] - low) / span + beta * float(heuristic_scorer(item[1])),
            reverse=True,
        )
        return ranked

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


def _encode_leaf_step(
    agent: TrainedPolicyAgent,
    leaf_obs_dict: dict[str, Any],
    our_deck: list[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
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
    return leaf_norm, leaf_cards.astype(np.int64)


def evaluate_search_leaf(
    agent: TrainedPolicyAgent,
    leaf_obs_dict: dict[str, Any],
    root_your_index: int,
    our_deck: list[int] | None = None,
) -> float:
    leaf_norm, leaf_cards = _encode_leaf_step(agent, leaf_obs_dict, our_deck)
    value = agent._model_value(leaf_features=leaf_norm, leaf_cards=leaf_cards)

    current = leaf_obs_dict.get("current") or {}
    if int(current.get("yourIndex", root_your_index)) != root_your_index:
        value = -value
    if int(current.get("result", -1)) >= 0:
        prize_winner = winner_from_prizes(leaf_obs_dict)
        winner = prize_winner if prize_winner is not None else int(current.get("result"))
        if winner == root_your_index:
            value = max(value, 1.0)
        else:
            value = min(value, -1.0)
    return value


def choose_rollout_action(
    agent: TrainedPolicyAgent,
    leaf_obs_dict: dict[str, Any],
    actions: list[list[int]],
    our_deck: list[int] | None,
    *,
    policy_width: int,
) -> list[int]:
    if not actions:
        return [0]
    if len(actions) == 1:
        return actions[0]

    leaf_norm, leaf_cards = _encode_leaf_step(agent, leaf_obs_dict, our_deck)
    logits = agent._model_logits(leaf_features=leaf_norm, leaf_cards=leaf_cards)
    ranked = rank_actions_by_policy(agent, logits, actions)
    width = max(1, min(policy_width, len(ranked)))
    return ranked[0][1] if width == 1 else agent._choose_from_policy_logits(
        logits, [action for _, action in ranked[:width]]
    )


def expand_action_in_search(
    agent: TrainedPolicyAgent,
    obs_dict: dict[str, Any],
    our_deck: list[int],
    action: list[int],
    root_your_index: int,
    deadline: float,
    *,
    max_search_steps: int = 64,
    rollout_policy_width: int = 8,
    determinization_seed_offset: int = 0,
) -> float:
    obs = to_observation_class(obs_dict)
    (
        your_deck_guess,
        your_prize_guess,
        opponent_deck_guess,
        opponent_prize_guess,
        opponent_hand_guess,
        opponent_active,
    ) = guess_hidden_cards(obs_dict, our_deck, seed_offset=determinization_seed_offset)

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
    steps = 0
    try:
        while time.perf_counter() < deadline and steps < max_search_steps:
            steps += 1
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
            candidates = legal_actions(len(options), min_count, max_count)
            action = choose_rollout_action(
                agent,
                leaf_obs,
                candidates,
                our_deck,
                policy_width=rollout_policy_width,
            )
        return evaluate_search_leaf(
            agent,
            observation_to_dict(current_state.observation),
            root_your_index,
            our_deck,
        )
    finally:
        search_end()


def _expand_action_value(
    agent: TrainedPolicyAgent,
    obs_dict: dict[str, Any],
    our_deck: list[int],
    action: list[int],
    root_your_index: int,
    deadline: float,
    config: BeamSearchConfig,
    *,
    policy_score: float,
) -> float:
    num_worlds = max(1, int(config.num_determinizations))
    values: list[float] = []
    for world_index in range(num_worlds):
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
                max_search_steps=config.max_search_steps,
                rollout_policy_width=config.rollout_policy_width,
                determinization_seed_offset=world_index * 17,
            )
        except Exception:
            leaf_value = policy_score
        values.append(float(leaf_value))
    if not values:
        return float(policy_score)
    return float(sum(values) / len(values))


def run_beam_search(
    agent: TrainedPolicyAgent,
    obs_dict: dict[str, Any],
    our_deck: list[int],
    actions: list[list[int]],
    root_your_index: int,
    config: BeamSearchConfig | None = None,
) -> list[int]:
    config = config or BeamSearchConfig()
    budget_ms = effective_time_budget_ms(obs_dict, config)
    deadline = time.perf_counter() + (budget_ms / 1000.0)
    logits = agent._model_logits()

    heuristic_scorer = None
    heuristic_beta = 0.0
    if config.heuristic_policy_beta_lucario > 0.0 or config.heuristic_policy_beta_dragapult > 0.0:
        from archetype_heuristics import heuristic_for_deck

        heuristic = heuristic_for_deck(
            our_deck,
            lucario_beta=config.heuristic_policy_beta_lucario,
            dragapult_beta=config.heuristic_policy_beta_dragapult,
        )
        if heuristic.active:
            heuristic_beta = heuristic.beta
            heuristic_scorer = heuristic.make_action_scorer(obs_dict, root_your_index)

    ranked = rank_actions_by_policy(
        agent,
        logits,
        actions,
        heuristic_scorer=heuristic_scorer,
        beta=heuristic_beta,
    )[: max(1, config.width)]

    scored: list[tuple[float, list[int]]] = []
    for policy_score, action in ranked:
        if time.perf_counter() >= deadline:
            break
        leaf_value = _expand_action_value(
            agent,
            obs_dict,
            our_deck,
            action,
            root_your_index,
            deadline,
            config,
            policy_score=policy_score,
        )
        scored.append((leaf_value, action))

    if not scored:
        scored = list(ranked)

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]
