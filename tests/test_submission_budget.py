from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.submission_budget import SubmissionSearchBudget
from scripts.build_submission_belief_posterior import build


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "submission" / "search_config.json").read_text(encoding="utf-8")
    )

def _enabled_config() -> dict:
    config = _config()
    config["enabled"] = True
    return config


def _decision(option_count: int = 3) -> dict:
    return {
        "current": {"player": 0},
        "select": {
            "option": [{"type": index} for index in range(option_count)],
            "minCount": 1,
            "maxCount": 1,
        },
    }


def test_budget_includes_startup_and_searches_within_internal_deadline() -> None:
    budget = SubmissionSearchBudget.from_config(_enabled_config(), started_at=0.0)
    plan = budget.plan(_decision(), now=14.0)
    assert plan.search is True
    assert plan.max_sims == 50
    assert 0.5 <= plan.move_time_s <= 4.0

    near_deadline = SubmissionSearchBudget.from_config(
        _enabled_config(), started_at=0.0
    )
    plan = near_deadline.plan(_decision(), now=539.0)
    assert plan.search is False
    assert plan.reason == "deadline_reserve"

    final_reserve = SubmissionSearchBudget.from_config(
        _enabled_config(), started_at=0.0
    )
    plan = final_reserve.plan(_decision(), now=580.0)
    assert plan.search is False
    assert plan.reason == "final_greedy_reserve"


def test_one_shallow_search_falls_back_for_one_decision_then_retries() -> None:
    budget = SubmissionSearchBudget.from_config(_enabled_config(), started_at=0.0)
    assert budget.plan(_decision(1), now=14.0).reason == "forced_or_trivial"
    budget.record_search(elapsed_s=1.0, completed_sims=49, succeeded=True)
    plan = budget.plan(_decision(), now=15.0)
    assert plan.search is True
    assert plan.reason == "trusted_belief_mcts"
    assert budget.consecutive_search_failures == 1
    assert budget.disabled_reason is None


def test_repeated_search_failures_never_force_game_wide_greedy() -> None:
    budget = SubmissionSearchBudget.from_config(_enabled_config(), started_at=0.0)
    for _ in range(10):
        budget.record_search(elapsed_s=1.0, completed_sims=49, succeeded=True)
    plan = budget.plan(_decision(), now=15.0)
    assert plan.search is True
    assert plan.reason == "trusted_belief_mcts"
    assert budget.consecutive_search_failures == 10
    assert budget.disabled_reason is None
    budget.reset(started_at=20.0)
    assert budget.plan(_decision(), now=21.0).search is True


def test_budget_adapts_simulations_without_exceeding_cap() -> None:
    budget = SubmissionSearchBudget.from_config(_enabled_config(), started_at=0.0)
    budget.record_search(elapsed_s=1.0, completed_sims=200, succeeded=True)
    plan = budget.plan(_decision(), now=15.0)
    assert plan.search is True
    assert plan.max_sims == 50


def test_budget_rejects_a_different_competition_hard_cap() -> None:
    config = _config()
    config["hard_cap_s"] = 601
    with pytest.raises(ValueError, match="unsafe"):
        SubmissionSearchBudget.from_config(config, started_at=0.0)


def test_budget_requires_explicit_twenty_second_greedy_tail() -> None:
    config = _config()
    config["final_greedy_reserve_s"] = 19.0
    with pytest.raises(ValueError, match="unsafe"):
        SubmissionSearchBudget.from_config(config, started_at=0.0)


def test_canonical_submission_is_policy_only_with_search_contract_retained() -> None:
    config = _config()
    assert config["leaf_evaluator"] == "trained_checkpoint_policy_value_head"
    assert config["leaf_evaluator_checkpoint"] == "submission_model_pt"
    assert config["require_trained_state_evaluator"] is True
    assert (
        config["search_failure_behavior"]
        == "greedy_current_decision_then_retry"
    )
    assert config["game_wide_greedy_only_for_time_budget"] is True
    budget = SubmissionSearchBudget.from_config(config, started_at=0.0)
    plan = budget.plan(_decision(), now=14.0)
    assert plan.search is False
    assert plan.reason == "disabled_by_config"


def test_public_prior_is_anonymous_and_deduplicated() -> None:
    payload = build(
        [
            ROOT / "data/training_mixes/top_ladder_representatives.v1.json",
            ROOT / "data/training_mixes/specialist_representatives.v1.json",
        ]
    )
    assert payload["anonymous"] is True
    assert payload["contains_opponent_identity"] is False
    assert payload["deck_count"] == len(payload["deck_lists"])
    assert payload["deck_count"] >= 8
    assert len({tuple(deck) for deck in payload["deck_lists"]}) == payload["deck_count"]
    assert all(len(deck) == 60 for deck in payload["deck_lists"])
    assert "names" not in payload
    assert "submission_ids" not in payload


def test_submission_smoke_accepts_frozen_v5_policy_diagnostics() -> None:
    source = (ROOT / "scripts" / "build_submission.sh").read_text(
        encoding="utf-8"
    )
    assert 'getattr(policy, "last_search_fallback_reason", None)' in source
    assert 'getattr(policy, "fail_closed_count", 0)' in source


def test_submission_build_stages_complete_adapter_loader_contract() -> None:
    source = (ROOT / "scripts" / "build_submission.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'cp "$ROOT/scripts/train_round_robin.py" '
        '"$STAGE/scripts/train_round_robin.py"'
    ) in source
