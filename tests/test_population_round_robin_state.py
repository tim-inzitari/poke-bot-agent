from __future__ import annotations

import copy

import pytest

from scripts.population_round_robin_state import (
    eligible_own_opponents,
    initialize_state,
    record_completed_member_cycle,
)


def readiness() -> dict:
    members = []
    for index in range(14):
        specialist_id = f"specialist-{index:02d}"
        members.append(
            {
                "specialist_id": specialist_id,
                "checkpoint": f"/models/{specialist_id}/model.pt",
                "checkpoint_digest": "sha256:" + f"{index:064x}",
                "content_digest": "sha256:" + f"{index + 100:064x}",
                "opponent_id": f"specialist-{specialist_id}-frozen",
                "baseline_group": "specialists",
                "baseline_dir": f"{specialist_id}-frozen",
                "baseline_package": f"/baselines/specialists/{specialist_id}-frozen",
                "expert_manifest": f"/expert/{specialist_id}.json",
                "expert_manifest_digest": "sha256:"
                + f"{index + 200:064x}",
                "trainable_in_population": True,
                "external_agent": False,
            }
        )
    return {
        "schema": "poke_bot.population_round_robin_ready/v1",
        "status": "ready",
        "member_count": 14,
        "members": members,
        "training_opponent_scope": "own_models_only",
        "external_agents_training_eligible": False,
        "rl_epochs_per_cycle": 5,
        "expert_rehearsal_epochs_per_cycle": 5,
    }


def test_initial_population_has_exact_14_and_only_own_opponents() -> None:
    state = initialize_state(readiness())
    opponents = eligible_own_opponents(
        state,
        active_specialist_id=state["active_specialist_id"],
    )
    assert len(state["members"]) == 14
    assert len(opponents) == 14
    assert all(row["external_agent"] is False for row in opponents)


def test_member_rotation_requires_exact_five_plus_five() -> None:
    state = initialize_state(readiness())
    current = state["members"][0]["current"]
    boundary = {
        "schema": "poke_bot.population_member_cycle_boundary/v1",
        "specialist_id": "specialist-00",
        "rl_iterations_completed": 5,
        "expert_rehearsal_epochs_completed": 5,
        "external_agents_training_eligible": False,
        "parent": {"digest": current["checkpoint_digest"]},
        "rehearsed": {
            "path": "/population/specialist-00/cycle-0.pt",
            "digest": "sha256:" + "f" * 64,
        },
    }
    advanced = record_completed_member_cycle(
        state,
        boundary,
        {
            "opponent_id": "population-specialist-00-cycle-0000",
            "checkpoint_digest": "sha256:" + "f" * 64,
            "content_digest": "sha256:" + "e" * 64,
            "baseline_group": "population",
            "baseline_dir": "specialist-00-cycle-0000",
            "baseline_package": "/baselines/population/specialist-00-cycle-0000",
        },
    )
    assert advanced["active_specialist_id"] == "specialist-01"
    assert advanced["members"][0]["rl_epochs_completed"] == 5
    assert advanced["members"][0]["rehearsal_epochs_completed"] == 5
    opponents = eligible_own_opponents(
        advanced,
        active_specialist_id=advanced["active_specialist_id"],
    )
    assert len(opponents) == 15

    bad = copy.deepcopy(boundary)
    bad["rl_iterations_completed"] = 6
    with pytest.raises(RuntimeError, match="boundary is not exact"):
        record_completed_member_cycle(
            state,
            bad,
            {
                "opponent_id": "population-specialist-00-cycle-0000",
                "checkpoint_digest": "sha256:" + "f" * 64,
                "content_digest": "sha256:" + "e" * 64,
                "baseline_group": "population",
                "baseline_dir": "specialist-00-cycle-0000",
                "baseline_package": (
                    "/baselines/population/specialist-00-cycle-0000"
                ),
            },
        )


def test_refresh_current_preserves_original_as_selected_history() -> None:
    value = readiness()
    original = {
        key: value["members"][0][key]
        for key in (
            "checkpoint",
            "checkpoint_digest",
            "content_digest",
            "opponent_id",
            "baseline_group",
            "baseline_dir",
            "baseline_package",
        )
    }
    value["members"][0].update(
        {
            "checkpoint": "/models/specialist-00/refresh.pt",
            "checkpoint_digest": "sha256:" + "d" * 64,
            "content_digest": "sha256:" + "e" * 64,
            "opponent_id": "population-refresh-specialist-00",
            "baseline_group": "population-refresh",
            "baseline_dir": "specialist-00-refresh",
            "baseline_package": "/baselines/population-refresh/specialist-00-refresh",
            "current_role": "current_post_fleet_refresh",
            "selected_history": [original],
        }
    )
    state = initialize_state(value)
    first = state["members"][0]
    assert first["current"]["role"] == "current_post_fleet_refresh"
    assert first["current"]["checkpoint_digest"] == "sha256:" + "d" * 64
    assert first["selected_history"][0]["checkpoint_digest"] == original[
        "checkpoint_digest"
    ]
    assert len(
        eligible_own_opponents(
            state, active_specialist_id=state["active_specialist_id"]
        )
    ) == 15

def test_population_rejects_external_training_member() -> None:
    value = readiness()
    value["members"][0]["external_agent"] = True
    with pytest.raises(RuntimeError, match="own-model identity"):
        initialize_state(value)
