from __future__ import annotations

import torch
import pytest

from poke_bot.strategic_losses import (
    expanded_strategic_losses,
    guide_pairwise_route_ranking_loss,
    resident_expanded_strategic_losses,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


def _outputs(rows: int = 2, options: int = 3):
    option = {
        "action_q": torch.zeros(rows, options, requires_grad=True),
        "action_type": torch.zeros(rows, options, requires_grad=True),
        "action_target": torch.zeros(rows, options, requires_grad=True),
        "action_resource": torch.zeros(rows, options, requires_grad=True),
        "action_utility": torch.zeros(rows, options, 6, requires_grad=True),
    }
    state = {
        "tactical_outcome": torch.zeros(rows, 3, 6, requires_grad=True),
        "opponent_response": torch.zeros(rows, 7, requires_grad=True),
        "resource_forecast": torch.zeros(rows, 6, requires_grad=True),
        "game_phase": torch.zeros(rows, 5, requires_grad=True),
        "outcome_distribution": torch.zeros(rows, 3, requires_grad=True),
        "remaining_turns": torch.zeros(rows, 1, requires_grad=True),
    }
    return option, state


def _target() -> dict:
    return {
        "action_factors": [
            {"action_type": True, "target": True, "resource": False}
        ],
        "action_utility": {
            "values": [20.0, 2.0, 1.0, 0.0, 1.0, 1.0],
            "mask": [True, True, True, True, True, True],
        },
        "tactical_outcomes": {
            "values": [[0.0] * 6, [1.0] * 6, [2.0] * 6],
            "mask": [[True] * 6, [True] * 6, [False] * 6],
        },
        "opponent_response": {
            "values": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "mask": [True] * 7,
        },
        "resource_forecast": {
            "values": [6.0, 30.0, 3.0, 2.0, 1.0, 0.0],
            "mask": [True] * 6,
        },
        "game_phase": 2,
        "outcome_class": 2,
        "remaining_turns_log1p": 2.0,
    }


def test_expanded_losses_train_only_present_exact_targets() -> None:
    option, state = _outputs()
    weights = {name: 1.0 for name in EXPANDED_HEAD_IDS}
    total, metrics = expanded_strategic_losses(
        option_outputs=option,
        state_outputs=state,
        target_indices=torch.tensor([1, 0]),
        value_targets=torch.tensor([1.0, -1.0]),
        option_counts=[3, 2],
        stage_indices=[0, 0],
        decision_aux=[
            {"expanded_strategic": _target()},
            {},  # absence is masked, never converted into a zero target
        ],
        weights=weights,
    )
    assert torch.isfinite(total)
    assert total.item() > 0.0
    assert metrics.labeled["action_q"] == 1
    assert metrics.labeled["action_resource"] == 0
    assert metrics.labeled["outcome_distribution"] == 1
    assert metrics.masked["remaining_turns"] == 1
    total.backward()
    assert option["action_q"].grad is not None
    assert state["outcome_distribution"].grad is not None


def test_exact_multi_event_counts_project_to_binary_occurrence() -> None:
    option, state = _outputs(rows=1)
    target = _target()
    target["action_utility"]["values"][5] = 4.0
    target["opponent_response"]["values"][1] = 5.0
    target["opponent_response"]["values"][2] = 3.0

    total, metrics = expanded_strategic_losses(
        option_outputs=option,
        state_outputs=state,
        target_indices=torch.tensor([0]),
        value_targets=torch.tensor([1.0]),
        option_counts=[3],
        stage_indices=[0],
        decision_aux=[{"expanded_strategic": target}],
        weights={"action_utility": 1.0, "opponent_response": 1.0},
    )

    assert torch.isfinite(total)
    assert metrics.labeled["action_utility"] == 1
    assert metrics.labeled["opponent_response"] == 1


def test_fractional_event_count_fails_closed() -> None:
    option, state = _outputs(rows=1)
    target = _target()
    target["action_utility"]["values"][5] = 0.5

    with pytest.raises(
        ValueError,
        match="binary event-count target is not a nonnegative integer",
    ):
        expanded_strategic_losses(
            option_outputs=option,
            state_outputs=state,
            target_indices=torch.tensor([0]),
            value_targets=torch.tensor([1.0]),
            option_counts=[3],
            stage_indices=[0],
            decision_aux=[{"expanded_strategic": target}],
            weights={"action_utility": 1.0},
        )


def test_zero_weights_preserve_no_gradient_legacy_path() -> None:
    option, state = _outputs(rows=1)
    total, metrics = expanded_strategic_losses(
        option_outputs=option,
        state_outputs=state,
        target_indices=torch.tensor([0]),
        value_targets=torch.tensor([1.0]),
        option_counts=[3],
        stage_indices=[0],
        decision_aux=[{"expanded_strategic": _target()}],
        weights={},
    )
    assert total.item() == 0.0
    total.backward()
    assert option["action_q"].grad is not None
    assert torch.count_nonzero(option["action_q"].grad) == 0
    assert metrics.labeled == {name: 0 for name in EXPANDED_HEAD_IDS}


def test_state_targets_are_counted_once_across_factorized_stages() -> None:
    option, state = _outputs(rows=2)
    target = _target()
    total, metrics = expanded_strategic_losses(
        option_outputs=option,
        state_outputs=state,
        target_indices=torch.tensor([0, 1]),
        value_targets=torch.tensor([1.0, 1.0]),
        option_counts=[3, 3],
        stage_indices=[0, 1],
        decision_aux=[
            {"expanded_strategic": target},
            {"expanded_strategic": target},
        ],
        weights={"game_phase": 1.0},
    )
    assert torch.isfinite(total)
    assert metrics.total["game_phase"] == 1
    assert metrics.labeled["game_phase"] == 1


def _resident_targets(samples: int = 4, decisions: int = 4):
    targets = {
        "strategic_action_q_target": torch.zeros(samples),
        "strategic_action_q_mask": torch.zeros(samples, dtype=torch.uint8),
        "strategic_action_factor_mask": torch.zeros(
            samples, 3, dtype=torch.uint8
        ),
        "strategic_action_utility_target": torch.zeros(samples, 6),
        "strategic_action_utility_mask": torch.zeros(
            samples, 6, dtype=torch.uint8
        ),
        "strategic_tactical_outcome_target": torch.zeros(decisions, 3, 6),
        "strategic_tactical_outcome_mask": torch.zeros(
            decisions, 3, 6, dtype=torch.uint8
        ),
        "strategic_opponent_response_target": torch.zeros(decisions, 7),
        "strategic_opponent_response_mask": torch.zeros(
            decisions, 7, dtype=torch.uint8
        ),
        "strategic_resource_forecast_target": torch.zeros(decisions, 6),
        "strategic_resource_forecast_mask": torch.zeros(
            decisions, 6, dtype=torch.uint8
        ),
        "strategic_game_phase_target": torch.full(
            (decisions,), -1, dtype=torch.int16
        ),
        "strategic_game_phase_mask": torch.zeros(
            decisions, dtype=torch.uint8
        ),
        "strategic_outcome_class_target": torch.full(
            (decisions,), -1, dtype=torch.int16
        ),
        "strategic_outcome_class_mask": torch.zeros(
            decisions, dtype=torch.uint8
        ),
        "strategic_remaining_turns_target": torch.zeros(decisions),
        "strategic_remaining_turns_mask": torch.zeros(
            decisions, dtype=torch.uint8
        ),
    }
    return targets


def test_resident_losses_follow_sample_and_unique_decision_index_spaces() -> None:
    option, _ = _outputs(rows=3)
    _, state = _outputs(rows=2)
    targets = _resident_targets()

    # Caller order is deliberately not corpus order. Sample labels remain
    # sample-indexed while state labels use a separate unique decision index.
    sample_ids = torch.tensor([2, 0, 3])
    decision_ids = torch.tensor([3, 1])
    targets["strategic_action_q_target"][:] = torch.tensor(
        [-1.0, 0.0, 1.0, 1.0]
    )
    targets["strategic_action_q_mask"][:] = torch.tensor([1, 0, 1, 0])
    targets["strategic_action_factor_mask"][2] = torch.tensor([1, 1, 0])
    targets["strategic_action_factor_mask"][0] = torch.tensor([0, 0, 1])
    targets["strategic_action_utility_target"][3] = torch.tensor(
        [10.0, 2.0, 1.0, 0.0, 1.0, 1.0]
    )
    targets["strategic_action_utility_mask"][3] = 1

    targets["strategic_tactical_outcome_target"][3] = 1.0
    targets["strategic_tactical_outcome_mask"][3] = 1
    targets["strategic_opponent_response_target"][1, 0] = 1.0
    targets["strategic_opponent_response_mask"][1] = 1
    targets["strategic_resource_forecast_target"][3] = torch.tensor(
        [6.0, 30.0, 3.0, 2.0, 1.0, 0.0]
    )
    targets["strategic_resource_forecast_mask"][3] = 1
    targets["strategic_game_phase_target"][3] = 4
    targets["strategic_game_phase_mask"][3] = 1
    targets["strategic_outcome_class_target"][3] = 2
    targets["strategic_outcome_class_mask"][3] = 1
    targets["strategic_remaining_turns_target"][1] = 2.25
    targets["strategic_remaining_turns_mask"][1] = 1

    total, metrics = resident_expanded_strategic_losses(
        option_outputs=option,
        state_outputs=state,
        target_indices=torch.tensor([1, 0, 2]),
        option_counts=torch.tensor([3, 2, 3]),
        target_tensors=targets,
        sample_ids=sample_ids,
        decision_ids=decision_ids,
        weights={name: 1.0 for name in EXPANDED_HEAD_IDS},
    )

    assert torch.isfinite(total)
    assert total.item() > 0.0
    assert metrics.total["action_q"] == 3
    assert metrics.total["game_phase"] == 2
    assert metrics.labeled["action_q"] == 2
    assert metrics.labeled["action_type"] == 1
    assert metrics.labeled["action_target"] == 1
    assert metrics.labeled["action_resource"] == 1
    assert metrics.labeled["action_utility"] == 1
    assert metrics.labeled["tactical_outcome"] == 1
    assert metrics.labeled["opponent_response"] == 1
    assert metrics.labeled["resource_forecast"] == 1
    assert metrics.labeled["game_phase"] == 1
    assert metrics.labeled["outcome_distribution"] == 1
    assert metrics.labeled["remaining_turns"] == 1
    assert metrics.outcome_brier is not None
    assert metrics.outcome_ece is not None
    assert metrics.outcome_entropy is not None
    total.backward()
    assert option["action_q"].grad is not None
    assert state["game_phase"].grad is not None


def test_resident_zero_weight_path_does_not_require_v6_target_pack() -> None:
    option, _ = _outputs(rows=1)
    _, state = _outputs(rows=1)
    total, metrics = resident_expanded_strategic_losses(
        option_outputs=option,
        state_outputs=state,
        target_indices=torch.tensor([0]),
        option_counts=torch.tensor([3]),
        target_tensors={},
        sample_ids=torch.tensor([123]),
        decision_ids=torch.tensor([456]),
        weights={},
    )
    assert total.item() == 0.0
    total.backward()
    assert torch.count_nonzero(option["action_q"].grad) == 0
    assert metrics.labeled == {name: 0 for name in EXPANDED_HEAD_IDS}


def test_resident_state_decision_mapping_rejects_duplicates() -> None:
    option, _ = _outputs(rows=1)
    _, state = _outputs(rows=2)
    with pytest.raises(ValueError, match="decision_ids contain duplicates"):
        resident_expanded_strategic_losses(
            option_outputs=option,
            state_outputs=state,
            target_indices=torch.tensor([0]),
            option_counts=torch.tensor([3]),
            target_tensors=_resident_targets(),
            sample_ids=torch.tensor([0]),
            decision_ids=torch.tensor([1, 1]),
            weights={"game_phase": 1.0},
        )


def test_host_action_utility_only_trains_final_canonical_stage() -> None:
    option, state = _outputs(rows=2)
    target = _target()
    target["action_factors"] = [
        {"action_type": True, "target": False, "resource": False},
        {"action_type": False, "target": True, "resource": True},
    ]
    total, metrics = expanded_strategic_losses(
        option_outputs=option,
        state_outputs=state,
        target_indices=torch.tensor([0, 1]),
        value_targets=torch.tensor([1.0, 1.0]),
        option_counts=[3, 3],
        stage_indices=[0, 1],
        decision_aux=[
            {"expanded_strategic": target},
            {"expanded_strategic": target},
        ],
        weights={"action_utility": 1.0},
    )
    assert metrics.labeled["action_utility"] == 1
    total.backward()
    assert torch.count_nonzero(option["action_utility"].grad[0]) == 0
    assert torch.count_nonzero(option["action_utility"].grad[1]) > 0


def test_guide_pairwise_route_ranking_is_training_only_and_head_scoped() -> None:
    route_deltas = {
        "action_q": torch.zeros(3, 4, requires_grad=True),
        "action_resource": torch.zeros(3, 4, requires_grad=True),
        "setup_board_outcome": torch.zeros(3, 4, requires_grad=True),
        # Deliberately excluded: an unlabeled action_type route must not become
        # a hidden guide-imitation path.
        "action_type": torch.zeros(3, 4, requires_grad=True),
    }
    loss, metrics = guide_pairwise_route_ranking_loss(
        route_deltas=route_deltas,
        guide_target_indices=torch.tensor([2, -1, 1]),
        guide_confidences=torch.tensor([1.0, 0.0, 0.5]),
        option_counts=torch.tensor([4, 3, 2]),
    )
    assert loss.item() > 0.0
    assert metrics["eligible_rows"] == 2
    assert set(metrics["heads"]) == {
        "action_q",
        "action_resource",
        "setup_board_outcome",
    }
    loss.backward()
    assert torch.count_nonzero(route_deltas["action_q"].grad) > 0
    assert torch.count_nonzero(route_deltas["action_resource"].grad) > 0
    assert torch.count_nonzero(route_deltas["setup_board_outcome"].grad) > 0
    assert route_deltas["action_type"].grad is None


def test_guide_pairwise_route_ranking_masks_forced_single_option_rows() -> None:
    route = torch.zeros(3, 3, requires_grad=True)
    loss, metrics = guide_pairwise_route_ranking_loss(
        route_deltas={"action_q": route},
        guide_target_indices=torch.tensor([0, 1, 2]),
        guide_confidences=torch.tensor([1.0, 1.0, 1.0]),
        option_counts=torch.tensor([1, 2, 3]),
    )
    assert loss.item() > 0.0
    assert metrics["eligible_rows"] == 2
    loss.backward()
    assert torch.count_nonzero(route.grad[0]) == 0
    assert torch.count_nonzero(route.grad[1:]) > 0
