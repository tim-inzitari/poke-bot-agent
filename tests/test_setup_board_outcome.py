from __future__ import annotations

import inspect

import pytest
import torch

from poke_bot.setup_board_outcome import setup_board_outcome_loss


def _targets(rows: int) -> dict[str, torch.Tensor]:
    return {
        "resource_targets": torch.zeros(rows, 6),
        "resource_masks": torch.zeros(rows, 6, dtype=torch.bool),
        "outcome_targets": torch.zeros(rows, dtype=torch.long),
        "outcome_masks": torch.zeros(rows, dtype=torch.bool),
        "guide_confidences": torch.zeros(rows),
    }


def test_api_cannot_consume_guide_preference_index() -> None:
    parameters = inspect.signature(setup_board_outcome_loss).parameters
    assert "guide_target_index" not in parameters
    assert "guide_target_indices" not in parameters


def test_only_selected_setup_options_receive_observed_target_gradients() -> None:
    predictions = torch.zeros(2, 3, 9, requires_grad=True)
    targets = _targets(2)
    targets["resource_targets"][:, 0] = torch.tensor([2.0, 3.0])
    targets["resource_masks"][:, 0] = True
    targets["outcome_targets"][:] = torch.tensor([2, 0])
    targets["outcome_masks"][:] = True
    targets["guide_confidences"][:] = torch.tensor([0.5, 0.75])

    loss, metrics = setup_board_outcome_loss(
        predictions=predictions,
        selected_indices=torch.tensor([1, 2]),
        option_counts=torch.tensor([3, 3]),
        select_contexts=torch.tensor([1, 2]),
        selected_is_stop=torch.tensor([0, 1], dtype=torch.uint8),
        base_loss_weight=0.025,
        guide_loss_weight=0.05,
        **targets,
    )
    loss.backward()

    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad[0, 0]) == 0
    assert torch.count_nonzero(predictions.grad[0, 2]) == 0
    assert torch.count_nonzero(predictions.grad[1, 0]) == 0
    assert torch.count_nonzero(predictions.grad[1, 1]) == 0
    assert torch.count_nonzero(predictions.grad[0, 1]) > 0
    assert torch.count_nonzero(predictions.grad[1, 2]) > 0
    assert metrics.context_rows == {"setup_active": 1, "setup_bench": 1}
    assert metrics.stop_rows == 1
    assert metrics.non_stop_rows == 1


def test_guide_confidence_scales_observed_loss_without_imitation() -> None:
    predictions = torch.zeros(1, 2, 9)
    targets = _targets(1)
    targets["resource_targets"][0, 0] = 2.0
    targets["resource_masks"][0, 0] = True
    targets["outcome_targets"][0] = 2
    targets["outcome_masks"][0] = True

    def compute(confidence: float) -> float:
        targets["guide_confidences"][0] = confidence
        loss, _ = setup_board_outcome_loss(
            predictions=predictions,
            selected_indices=torch.tensor([0]),
            option_counts=torch.tensor([2]),
            select_contexts=torch.tensor([1]),
            base_loss_weight=0.0,
            guide_loss_weight=0.05,
            **targets,
        )
        return float(loss.item())

    half = compute(0.5)
    full = compute(1.0)
    assert half > 0.0
    assert full == pytest.approx(half * 2.0)


def test_setup_active_volume_does_not_dilute_setup_bench_context() -> None:
    def compute(active_rows: int) -> float:
        rows = active_rows + 1
        predictions = torch.zeros(rows, 1, 9)
        targets = _targets(rows)
        targets["resource_masks"][:, 0] = True
        targets["outcome_targets"][:] = 1
        targets["outcome_masks"][:] = True
        # Active rows are exact at zero; the one bench row has a fixed error.
        targets["resource_targets"][-1, 0] = 4.0
        loss, _ = setup_board_outcome_loss(
            predictions=predictions,
            selected_indices=torch.zeros(rows, dtype=torch.long),
            option_counts=torch.ones(rows, dtype=torch.long),
            select_contexts=torch.tensor([1] * active_rows + [2]),
            base_loss_weight=1.0,
            guide_loss_weight=0.0,
            **targets,
        )
        return float(loss.item())

    assert compute(1) == pytest.approx(compute(20))


def test_non_setup_rows_are_masked_even_with_present_targets() -> None:
    predictions = torch.randn(1, 2, 9, requires_grad=True)
    targets = _targets(1)
    targets["resource_targets"][0, 0] = 10.0
    targets["resource_masks"][0, 0] = True
    targets["outcome_targets"][0] = 2
    targets["outcome_masks"][0] = True
    targets["guide_confidences"][0] = 1.0
    loss, metrics = setup_board_outcome_loss(
        predictions=predictions,
        selected_indices=torch.tensor([0]),
        option_counts=torch.tensor([2]),
        select_contexts=torch.tensor([0]),
        base_loss_weight=1.0,
        guide_loss_weight=1.0,
        **targets,
    )
    loss.backward()
    assert float(loss.item()) == 0.0
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad) == 0
    assert metrics.eligible_rows == 0


def test_incomplete_setup_rows_are_masked_even_with_resource_targets() -> None:
    predictions = torch.randn(1, 2, 9, requires_grad=True)
    targets = _targets(1)
    targets["resource_targets"][0, 0] = 10.0
    targets["resource_masks"][0, 0] = True
    targets["guide_confidences"][0] = 1.0
    loss, metrics = setup_board_outcome_loss(
        predictions=predictions,
        selected_indices=torch.tensor([0]),
        option_counts=torch.tensor([2]),
        select_contexts=torch.tensor([1]),
        base_loss_weight=1.0,
        guide_loss_weight=1.0,
        **targets,
    )
    loss.backward()
    assert float(loss.item()) == 0.0
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad) == 0
    assert metrics.eligible_rows == 0


def test_invalid_binary_resource_target_fails_closed() -> None:
    targets = _targets(1)
    targets["resource_targets"][0, 4] = 2.0
    targets["resource_masks"][0, 4] = True
    with pytest.raises(ValueError, match="outside"):
        setup_board_outcome_loss(
            predictions=torch.zeros(1, 1, 9),
            selected_indices=torch.tensor([0]),
            option_counts=torch.tensor([1]),
            select_contexts=torch.tensor([2]),
            **targets,
        )
