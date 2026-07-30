"""Competition submission agent: history-conditioned policy, fail-closed actions.

Hard constraints:
  - No ``__file__`` at import time (isolated tarball / Kaggle).
  - Deck from ``deck.csv`` next to ``main.py`` or ``/kaggle_simulations/agent/``.
  - Deterministically honor the packaged turn-order profile before importing
    cg or loading the model.
  - Info-set only (features.assert_info_set inside the policy runtime).
  - Fail-closed: illegal selects -> legal random fallback.
"""

from __future__ import annotations

import os
import json
import random
import sys
import time
from pathlib import Path


_PROCESS_STARTED = time.monotonic()
_AGENT_DIR_CANDIDATES = (
    Path.cwd(),
    Path("/kaggle_simulations/agent"),
)


def _agent_dir() -> Path:
    for directory in _AGENT_DIR_CANDIDATES:
        if (directory / "deck.csv").is_file():
            return directory
    return Path.cwd()


def _read_deck() -> list[int]:
    path = _agent_dir() / "deck.csv"
    deck: list[int] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        deck.append(int(line.split(",")[0]))
        if len(deck) >= 60:
            break
    if len(deck) != 60:
        raise ValueError(f"deck.csv must have 60 cards, got {len(deck)}")
    return deck


_DECK: list[int] | None = None
_MODEL = None
_CLOCK = None
_POLICY = None
_SEARCH_BUDGET = None
_SEARCH_CONFIG = None
_GAME_COUNT = 0
_RNG = random.Random(0)


def _turn_order_preference() -> str:
    """Read the immutable packaged preference without importing the runtime."""

    path = _agent_dir() / "turn_order_profile.json"
    if not path.is_file():
        return "first_if_allowed"
    payload = json.loads(path.read_text())
    preference = str(payload.get("turn_order_preference") or "")
    if preference not in {"first_if_allowed", "second_if_allowed"}:
        raise RuntimeError("invalid packaged turn-order preference")
    return preference


def _turn_order_choice(obs_dict: dict) -> list[int] | None:
    """Resolve IsFirst directly from the wire enum without runtime imports."""

    selection = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(selection, dict):
        return None
    context = selection.get("context")
    normalized_context = "".join(
        character for character in str(context).lower() if character.isalnum()
    )
    if context != 41 and normalized_context != "isfirst":
        return None
    options = list(selection.get("option") or [])
    desired_type = (
        "yes" if _turn_order_preference() == "first_if_allowed" else "no"
    )
    desired_integer = 1 if desired_type == "yes" else 2
    matches = [
        index
        for index, option in enumerate(options)
        if isinstance(option, dict)
        and (
            option.get("type") == desired_integer
            or str(option.get("type") or "").strip().lower() == desired_type
        )
    ]
    return matches if len(matches) == 1 else []


def _go_first_choice(obs_dict: dict) -> list[int] | None:
    """Backward-compatible alias for the packaged turn-order resolver."""

    return _turn_order_choice(obs_dict)


def _ensure_agent_path() -> None:
    agent_dir = str(_agent_dir())
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def _ensure_runtime():
    global _DECK, _MODEL, _CLOCK, _POLICY, _SEARCH_BUDGET, _SEARCH_CONFIG
    if _DECK is None:
        _DECK = _read_deck()
    if _MODEL is None:
        _ensure_agent_path()
        # Vendored ``cg/`` sits directly beside this entry point. The shared
        # runtime path resolver otherwise looks only for repository/Kaggle
        # development layouts that do not exist inside the submitted tarball.
        os.environ.setdefault("CG_LIB_PATH", str(_agent_dir()))
        import torch
        from poke_bot.agent import PolicyAgent
        from poke_bot.belief import EmpiricalDeckPosterior
        from poke_bot.checkpoint import (
            assert_trusted_policy_checkpoint,
            checkpoint_digest,
        )
        from poke_bot.submission_budget import SubmissionSearchBudget
        from poke_bot.train import load_model_from_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = _agent_dir() / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("model.pt is required")
        matchup_tree = _agent_dir() / "matchup_tree.json"
        if matchup_tree.is_file():
            # The shipped tree is itself runtime-gated and consumes only
            # cumulative public opponent cards. PolicyAgent validates the
            # artifact before enabling the frozen trained adapter bank.
            os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
            os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(matchup_tree)
        assert_trusted_policy_checkpoint(checkpoint)
        model = load_model_from_checkpoint(checkpoint, device=device)
        model.eval()
        _MODEL = model
        search_config_path = _agent_dir() / "search_config.json"
        belief_decks_path = _agent_dir() / "belief_decks.json"
        search_enabled = (
            os.environ.get("POKEBOT_SUBMISSION_SEARCH_DISABLE", "0") != "1"
            and search_config_path.is_file()
            and belief_decks_path.is_file()
        )
        if search_enabled:
            _SEARCH_CONFIG = json.loads(search_config_path.read_text())
            _SEARCH_BUDGET = SubmissionSearchBudget.from_config(
                _SEARCH_CONFIG,
                started_at=_PROCESS_STARTED,
            )
            if _SEARCH_CONFIG.get("enabled") is not True:
                # Canonical competition mode is the frozen policy-only path.
                # The digest-bound belief-MCTS implementation below remains
                # dormant for a separately validated future experiment.
                _POLICY = PolicyAgent(model=model, deck=_DECK, use_mcts=False)
                _CLOCK = None
            else:
                belief_payload = json.loads(belief_decks_path.read_text())
                deck_hypotheses = belief_payload.get("deck_lists") or ()
                if (
                    _SEARCH_CONFIG.get("algorithm")
                    != "public_history_root_sampled_belief_mcts"
                    or _SEARCH_CONFIG.get("leaf_evaluator")
                    != "trained_checkpoint_policy_value_head"
                    or _SEARCH_CONFIG.get("leaf_evaluator_checkpoint")
                    != "submission_model_pt"
                    or _SEARCH_CONFIG.get("require_trained_state_evaluator")
                    is not True
                    or _SEARCH_CONFIG.get("search_failure_behavior")
                    != "greedy_current_decision_then_retry"
                    or _SEARCH_CONFIG.get(
                        "game_wide_greedy_only_for_time_budget"
                    )
                    is not True
                    or _SEARCH_CONFIG.get("fallback")
                    != "frozen_model_greedy_policy"
                    or _SEARCH_CONFIG.get("oracle_inputs_allowed") is not False
                    or belief_payload.get("schema")
                    != "poke_bot.submission_belief_decks/v1"
                    or belief_payload.get("anonymous") is not True
                    or belief_payload.get("contains_opponent_identity") is not False
                    or int(belief_payload.get("deck_count") or 0)
                    != len(deck_hypotheses)
                    or len(deck_hypotheses) < 8
                    or any(
                        len(deck) != 60
                        or any(int(card) <= 0 for card in deck)
                        for deck in deck_hypotheses
                    )
                ):
                    raise RuntimeError("submission belief-deck prior changed")
                posterior = EmpiricalDeckPosterior(deck_hypotheses)
                model_digest = checkpoint_digest(checkpoint)
                _POLICY = PolicyAgent(
                    model=model,
                    deck=_DECK,
                    use_mcts=True,
                    belief_mcts=True,
                    belief_posterior=posterior,
                    checkpoint_digest=model_digest,
                    model_generation=0,
                    game_time_budget_s=float(
                        _SEARCH_CONFIG["total_search_budget_s"]
                    ),
                    game_watchdog_reserve_s=0.0,
                    expected_search_decisions=int(
                        _SEARCH_CONFIG["expected_search_decisions"]
                    ),
                    max_sims=int(_SEARCH_CONFIG["minimum_sims"]),
                    min_trusted_sims=int(_SEARCH_CONFIG["minimum_sims"]),
                    move_time_s=float(_SEARCH_CONFIG["maximum_move_s"]),
                )
                _CLOCK = _POLICY.clock
        else:
            _POLICY = PolicyAgent(model=model, deck=_DECK, use_mcts=False)
            _CLOCK = None
    return _DECK, _MODEL, _POLICY


def _fail_closed(obs_dict: dict, preferred: list[int]) -> list[int]:
    selection = obs_dict.get("select") if obs_dict else None
    if selection is None:
        return preferred
    option_count = len(selection.get("option") or [])
    if option_count <= 0:
        return []
    minimum = int(selection.get("minCount", 0) or 0)
    maximum = min(int(selection.get("maxCount", 0) or 0), option_count)
    minimum = max(0, min(minimum, maximum))
    clean: list[int] = []
    for raw in preferred:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= index < option_count and index not in clean:
            clean.append(index)
    if minimum <= len(clean) <= maximum and clean:
        return clean[:maximum]
    if maximum <= 0:
        return []
    count = _RNG.randint(minimum, maximum) if maximum >= minimum else maximum
    return _RNG.sample(range(option_count), count) if count > 0 else []


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point."""

    global _GAME_COUNT
    turn_order = _turn_order_choice(obs_dict)
    if turn_order is not None:
        return _fail_closed(obs_dict, turn_order)

    deck, _model, policy = _ensure_runtime()
    _ensure_agent_path()
    from cg.api import to_observation_class

    observation = to_observation_class(obs_dict)
    if observation.select is None:
        if policy is not None:
            policy.reset_game()
        if _SEARCH_BUDGET is not None:
            if _GAME_COUNT > 0:
                _SEARCH_BUDGET.reset()
            _GAME_COUNT += 1
        return list(deck)

    try:
        if _SEARCH_BUDGET is None:
            action = policy.trusted_search_or_greedy_select(
                obs_dict,
                search=False,
            )
        else:
            plan = _SEARCH_BUDGET.plan(obs_dict)
            policy.max_sims = plan.max_sims or policy.max_sims
            policy.move_time_s = plan.move_time_s or policy.move_time_s
            prior_result = policy.last_result
            started = time.monotonic()
            action = policy.trusted_search_or_greedy_select(
                obs_dict,
                search=plan.search,
            )
            elapsed = time.monotonic() - started
            if plan.search:
                result = (
                    policy.last_result
                    if policy.last_result is not prior_result
                    else None
                )
                _SEARCH_BUDGET.record_search(
                    elapsed_s=elapsed,
                    completed_sims=(
                        int(result.sims_run) if result is not None else 0
                    ),
                    succeeded=(
                        result is not None
                        and policy.last_search_fallback_reason is None
                    ),
                )
    except Exception:
        action = []
    return _fail_closed(obs_dict, action)
