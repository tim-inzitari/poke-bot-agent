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
    for index in range(22):
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
        "member_count": 22,
        "members": members,
        "training_opponent_scope": "own_models_only",
        "external_agents_training_eligible": False,
        "rl_epochs_per_cycle": 5,
        "expert_rehearsal_epochs_per_cycle": 5,
    }


def test_initial_population_has_exact_22_and_only_own_opponents() -> None:
    state = initialize_state(readiness())
    opponents = eligible_own_opponents(
        state,
        active_specialist_id=state["active_specialist_id"],
    )
    assert len(state["members"]) == 22
    assert len(opponents) == 22
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
    assert len(opponents) == 23

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


def test_population_rejects_external_training_member() -> None:
    value = readiness()
    value["members"][0]["external_agent"] = True
    with pytest.raises(RuntimeError, match="own-model identity"):
        initialize_state(value)
