from __future__ import annotations

import inspect

import pytest
import torch

from poke_bot.combo_state import COMBO_STATE_OUTPUT_WIDTH, VECTOR_WIDTH, combo_state_loss


def _inputs():
    predictions = torch.zeros(3, 2, COMBO_STATE_OUTPUT_WIDTH, requires_grad=True)
    return {
        "predictions": predictions,
        "selected_indices": torch.tensor([0, 1, 0]),
        "option_counts": torch.tensor([2, 2, 1]),
        "top_deck_targets": torch.tensor([1, 0, 4]),
        "top_deck_masks": torch.tensor([1, 0, 1], dtype=torch.bool),
        "seek_source_targets": torch.tensor([0, 6, 2]),
        "seek_source_masks": torch.tensor([0, 1, 0], dtype=torch.bool),
        "vector_targets": torch.zeros(3, VECTOR_WIDTH),
        "vector_masks": torch.ones(3, VECTOR_WIDTH, dtype=torch.bool),
        "guide_confidences": torch.tensor([1.0, 0.5, 0.0]),
        "base_loss_weight": 0.025,
        "guide_loss_weight": 0.05,
    }


def test_combo_state_loss_has_no_guide_preferred_action_input() -> None:
    assert "guide_preferred_indices" not in inspect.signature(
        combo_state_loss
    ).parameters


def test_combo_state_loss_trains_only_selected_observed_targets() -> None:
    kwargs = _inputs()
    loss, metrics = combo_state_loss(**kwargs)
    loss.backward()
    gradient = kwargs["predictions"].grad
    assert gradient is not None
    assert torch.isfinite(loss)
    assert metrics.eligible_rows == 3
    assert metrics.top_deck_labels == 2
    assert metrics.seek_source_labels == 1
    assert gradient[0, 0].abs().sum() > 0
    assert gradient[0, 1].abs().sum() == 0
    assert gradient[1, 1].abs().sum() > 0
    assert gradient[1, 0].abs().sum() == 0


def test_guide_confidence_scales_observed_loss_without_becoming_target() -> None:
    no_guide = _inputs()
    no_guide["guide_loss_weight"] = 0.0
    base, _ = combo_state_loss(**no_guide)
    guided = _inputs()
    with_guide, metrics = combo_state_loss(**guided)
    assert with_guide > base
    assert metrics.guide_observed_loss > 0.0
    assert metrics.guide_rows == 2


def test_all_missing_targets_produce_exact_zero_and_zero_gradients() -> None:
    kwargs = _inputs()
    kwargs["top_deck_masks"].zero_()
    kwargs["seek_source_masks"].zero_()
    kwargs["vector_masks"].zero_()
    loss, metrics = combo_state_loss(**kwargs)
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert metrics.eligible_rows == 0
    assert kwargs["predictions"].grad is not None
    assert kwargs["predictions"].grad.abs().sum().item() == pytest.approx(0.0)


def test_invalid_hidden_or_unmasked_values_fail_closed() -> None:
    kwargs = _inputs()
    kwargs["vector_targets"][0, 0] = 2.0
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        combo_state_loss(**kwargs)
