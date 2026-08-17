from __future__ import annotations

import copy

import pytest

from poke_bot.strategic_schedule import (
    EXPANDED_HEAD_IDS,
    expanded_head_epoch_plan,
    expanded_schedule_digest,
    validated_expanded_head_schedule,
)


def _contract() -> dict:
    weights = {
        "action_q": 0.10,
        "action_type": 0.05,
        "action_target": 0.025,
        "action_resource": 0.025,
        "action_utility": 0.05,
        "tactical_outcomes": 0.05,
        "opponent_response": 0.05,
        "resource_forecast": 0.025,
        "game_phase": 0.025,
        "outcome_distribution": 0.05,
        "remaining_turns": 0.025,
    }
    return {
        "schema": "poke_bot.expanded_strategic_heads/v1",
        "target_schema": "poke_bot.expanded_strategic_targets/v2",
        "checkpoint_contract_schema": "poke_bot.expanded_head_training/v1",
        "heads": {name: {"weight": weight} for name, weight in weights.items()},
        "bootstrap_stage_schedule": {
            "total_epochs": 25,
            "stages": [
                {
                    "epochs": [1, 5],
                    "enable": [
                        "action_q",
                        "action_type",
                        "action_target",
                        "action_resource",
                        "action_utility",
                    ],
                },
                {
                    "epochs": [6, 10],
                    "add": ["tactical_outcomes", "opponent_response"],
                },
                {
                    "epochs": [11, 15],
                    "add": ["resource_forecast", "game_phase"],
                },
                {
                    "epochs": [16, 20],
                    "add": ["outcome_distribution", "remaining_turns"],
                },
                {"epochs": [21, 25], "enable_all": True},
            ],
            "existing_enabled_heads_train_in_every_epoch": True,
            "exact_epoch_count_may_not_be_shortened": True,
        },
    }


def test_schedule_is_exact_cumulative_and_normalizes_tactical_name() -> None:
    canonical = validated_expanded_head_schedule(_contract())
    assert canonical["total_epochs"] == 25
    assert set(canonical["weights"]) == set(EXPANDED_HEAD_IDS)
    assert "tactical_outcome" in canonical["weights"]
    assert "tactical_outcomes" not in canonical["weights"]

    assert expanded_head_epoch_plan(_contract(), 1).enabled_heads == (
        "action_q",
        "action_type",
        "action_target",
        "action_resource",
        "action_utility",
    )
    assert "opponent_response" in expanded_head_epoch_plan(
        _contract(), 6
    ).enabled_heads
    assert set(expanded_head_epoch_plan(_contract(), 25).enabled_heads) == set(
        EXPANDED_HEAD_IDS
    )


def test_schedule_digest_is_syntax_independent_and_changes_with_weight() -> None:
    first = _contract()
    second = copy.deepcopy(first)
    second["heads"] = dict(reversed(list(second["heads"].items())))
    assert expanded_schedule_digest(first) == expanded_schedule_digest(second)
    second["heads"]["action_q"]["weight"] = 0.11
    assert expanded_schedule_digest(first) != expanded_schedule_digest(second)


def test_schedule_fails_closed_on_gap_or_wrong_epoch_count() -> None:
    raw = _contract()
    raw["bootstrap_stage_schedule"]["stages"][1]["epochs"] = [7, 10]
    with pytest.raises(ValueError, match="cover epochs"):
        validated_expanded_head_schedule(raw)
    with pytest.raises(ValueError, match="outside"):
        expanded_head_epoch_plan(_contract(), 26)
