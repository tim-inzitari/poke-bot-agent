"""Focused contract tests for the trainer-only eight-output action critic."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from poke_bot.action_critic_sidecar import (  # noqa: E402
    ACTION_CRITIC_OUTPUT_NAMES,
    ActionCriticSidecar,
    ActionCriticSidecarConfig,
    ActionCriticSidecarError,
    action_critic_loss,
    build_action_critic_checkpoint,
    load_action_critic_checkpoint,
    restore_action_critic_checkpoint,
    save_action_critic_checkpoint,
    split_action_critic_predictions,
)


def _model() -> ActionCriticSidecar:
    torch.manual_seed(17)
    return ActionCriticSidecar(
        ActionCriticSidecarConfig(
            feature_dim=4,
            state_hidden_dim=12,
            action_hidden_dim=10,
            q_hidden_dim=14,
            max_action_stages=4,
        )
    )


def _batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    first_stage = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.0, 0.0, 0.0]],
            [[-0.2, 0.1, 0.3, 0.8], [0.9, -0.5, 0.2, 0.4], [0.6, 0.1, -0.7, 0.2]],
        ],
        dtype=torch.float32,
    )
    first_stage_mask = torch.tensor([[True, True, False], [True, True, True]])
    selected_stages = torch.tensor(
        [
            [[0.4, -0.5, 0.6, -0.7], [0.8, 0.9, -0.1, 0.2], [0.0, 0.0, 0.0, 0.0]],
            [[-0.9, 0.4, 0.2, 0.5], [0.3, -0.2, 0.1, 0.8], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    selected_mask = torch.tensor([[True, True, False], [True, True, False]])
    selected_option_indices = torch.tensor([[0, 1, 0], [2, 0, 0]], dtype=torch.int64)
    selected_legal_counts = torch.tensor([[2, 3, 0], [3, 2, 0]], dtype=torch.int64)
    selected_action_program_tokens = torch.tensor(
        [
            [[0, 0], [0, 2], [0, 0]],
            [[2, 1], [0, 0], [0, 0]],
        ],
        dtype=torch.int64,
    )
    selected_action_program_mask = torch.tensor(
        [
            [[True, False], [True, True], [False, False]],
            [[True, True], [True, False], [False, False]],
        ]
    )
    return (
        first_stage,
        first_stage_mask,
        selected_stages,
        selected_mask,
        selected_option_indices,
        selected_legal_counts,
        selected_action_program_tokens,
        selected_action_program_mask,
    )


def test_output_abi_and_v_path_use_only_full_first_stage_legal_menu() -> None:
    model = _model().eval()
    inputs = _batch()
    first_stage, first_stage_mask, selected_stages, selected_mask = inputs[:4]

    with torch.no_grad():
        original = model(*inputs)
        changed_selected = selected_stages.clone()
        changed_selected[:, :2] += 7.0
        changed = model(
            first_stage,
            first_stage_mask,
            changed_selected,
            selected_mask,
            *inputs[4:],
        )

    assert original.shape == (2, 8)
    assert model.output_names == ACTION_CRITIC_OUTPUT_NAMES
    assert torch.isfinite(original).all()
    assert torch.all(original[:, 2:] >= -1.0)
    assert torch.all(original[:, 2:] <= 1.0)
    # V heads cannot see selected-stage features.  This is exact, not merely a
    # tolerance, because their computation graph does not reference them.
    torch.testing.assert_close(original[:, (0, 2, 4, 6)], changed[:, (0, 2, 4, 6)], rtol=0, atol=0)
    assert not torch.allclose(original[:, (1, 3, 5, 7)], changed[:, (1, 3, 5, 7)])

    named = split_action_critic_predictions(original)
    assert named.as_dict().keys() == set(ACTION_CRITIC_OUTPUT_NAMES)
    assert named.v_win_probability.shape == (2,)
    assert torch.all(named.v_win_probability >= 0.0)
    assert torch.all(named.v_win_probability <= 1.0)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "first_stage_legal_features",
            torch.full((2, 3, 4), float("nan"), dtype=torch.float32),
            "finite",
        ),
        (
            "first_stage_legal_mask",
            torch.tensor([[True, False, True], [True, True, True]]),
            "right-padded",
        ),
        (
            "selected_stage_mask",
            torch.tensor([[True, False, True], [True, True, False]]),
            "right-padded",
        ),
    ],
)
def test_forward_rejects_nonfinite_and_ambiguous_masks(
    field: str, replacement: torch.Tensor, message: str
) -> None:
    model = _model()
    inputs = _batch()
    first_stage, first_stage_mask, selected_stages, selected_mask = inputs[:4]
    arguments = {
        "first_stage_legal_features": first_stage,
        "first_stage_legal_mask": first_stage_mask,
        "selected_stage_features": selected_stages,
        "selected_stage_mask": selected_mask,
        "selected_option_indices": inputs[4],
        "selected_legal_counts": inputs[5],
        "selected_action_program_tokens": inputs[6],
        "selected_action_program_mask": inputs[7],
    }
    arguments[field] = replacement
    with pytest.raises(ActionCriticSidecarError, match=message):
        model(**arguments)


def test_forward_rejects_wrong_feature_width_nonbool_mask_and_empty_action() -> None:
    model = _model()
    inputs = _batch()
    first_stage, first_stage_mask, selected_stages, selected_mask = inputs[:4]
    with pytest.raises(ActionCriticSidecarError, match="feature dimension"):
        model(first_stage[..., :3], first_stage_mask, selected_stages, selected_mask, *inputs[4:])
    with pytest.raises(ActionCriticSidecarError, match="must be bool"):
        model(
            first_stage,
            first_stage_mask.to(dtype=torch.int64),
            selected_stages,
            selected_mask,
            *inputs[4:],
        )
    with pytest.raises(ActionCriticSidecarError, match="at least one valid"):
        model(
            first_stage,
            first_stage_mask,
            selected_stages,
            torch.tensor([[False, False, False], [True, True, False]]),
            *inputs[4:],
        )


def test_q_action_structure_distinguishes_identical_selected_feature_vectors() -> None:
    """Legal positions/programs remain distinguishable even when 40-D rows alias."""

    model = _model().eval()
    first_stage = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]],
        dtype=torch.float32,
    ).expand(2, -1, -1).clone()
    first_stage_mask = torch.tensor([[True, True], [True, True]])
    selected_stage = torch.tensor(
        [[[0.5, -0.25, 0.75, -0.5]], [[0.5, -0.25, 0.75, -0.5]]],
        dtype=torch.float32,
    )
    selected_mask = torch.tensor([[True], [True]])
    # Both rows deliberately have identical state and selected 40-D vectors.
    # Only sealed current legal position/program identity differs.
    selected_option_indices = torch.tensor([[0], [1]], dtype=torch.int64)
    selected_legal_counts = torch.tensor([[2], [2]], dtype=torch.int64)
    selected_action_program_tokens = torch.tensor([[[0]], [[1]]], dtype=torch.int64)
    selected_action_program_mask = torch.tensor([[[True]], [[True]]])

    with torch.no_grad():
        values = model(
            first_stage,
            first_stage_mask,
            selected_stage,
            selected_mask,
            selected_option_indices,
            selected_legal_counts,
            selected_action_program_tokens,
            selected_action_program_mask,
        )
        structure = model._chosen_action_structure(
            selected_option_indices,
            selected_legal_counts,
            selected_action_program_tokens,
            selected_action_program_mask,
        )

    # The V path remains state-only, while Q receives a non-aliased action
    # representation and therefore can distinguish the two complete actions.
    torch.testing.assert_close(values[0, (0, 2, 4, 6)], values[1, (0, 2, 4, 6)], rtol=0, atol=0)
    assert not torch.equal(structure[0], structure[1])
    assert not torch.equal(values[0, (1, 3, 5, 7)], values[1, (1, 3, 5, 7)])


def test_win_bce_prize_masked_smooth_l1_and_backward() -> None:
    model = _model()
    predictions = model(*_batch())
    win_targets = torch.tensor([True, False])
    prize_targets = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.25, 0.5, -1.0]], dtype=torch.float32
    )
    prize_mask = torch.tensor([[True, False, True], [False, True, False]])
    result = action_critic_loss(
        predictions,
        win_targets=win_targets,
        prize_targets=prize_targets,
        prize_mask=prize_mask,
    )

    assert torch.isfinite(result.total)
    assert result.valid_prize_count_by_horizon == (1, 1, 1)
    result.total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    # False masks exclude labels completely.  The finite placeholder values are
    # checked for integrity but cannot become fabricated zero-reward targets.
    masked_only_changed = prize_targets.clone()
    masked_only_changed[0, 1] = 1.0
    masked_only_changed[1, 0] = 1.0
    masked_only_changed[1, 2] = 1.0
    again = action_critic_loss(
        predictions.detach(),
        win_targets=win_targets,
        prize_targets=masked_only_changed,
        prize_mask=prize_mask,
    )
    torch.testing.assert_close(result.total.detach(), again.total, rtol=0, atol=0)

    with pytest.raises(ActionCriticSidecarError, match="exactly binary"):
        action_critic_loss(
            predictions.detach(),
            win_targets=torch.tensor([1.0, 0.5], dtype=torch.float32),
            prize_targets=prize_targets,
            prize_mask=prize_mask,
        )
    with pytest.raises(ActionCriticSidecarError, match="clipped"):
        action_critic_loss(
            predictions.detach(),
            win_targets=win_targets,
            prize_targets=torch.full((2, 3), 1.1, dtype=torch.float32),
            prize_mask=prize_mask,
        )


def test_checkpoint_is_sidecar_only_and_restores_identical_predictions(tmp_path) -> None:
    model = _model().eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sample = _batch()
    with torch.no_grad():
        expected = model(*sample)

    payload = build_action_critic_checkpoint(
        model,
        optimizer=optimizer,
        training_state={"step": 11, "epoch": 2},
        metadata={"sealed_feature_pack_sha256": "sha256:fixture"},
    )
    assert set(payload) == {
        "schema",
        "config",
        "sidecar_state_dict",
        "optimizer_state_dict",
        "training_state",
        "metadata",
    }
    assert "model_state_dict" not in payload
    assert "policy_state_dict" not in payload

    path = tmp_path / "critic.pt"
    save_action_critic_checkpoint(
        path,
        model,
        optimizer=optimizer,
        training_state={"step": 11, "epoch": 2},
        metadata={"sealed_feature_pack_sha256": "sha256:fixture"},
    )
    restored = load_action_critic_checkpoint(path)
    restored.model.eval()
    with torch.no_grad():
        actual = restored.model(*sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.training_state == {"step": 11, "epoch": 2}

    destination = _model()
    destination_optimizer = torch.optim.AdamW(destination.parameters(), lr=1e-3)
    restore_action_critic_checkpoint(destination, path, optimizer=destination_optimizer)
    destination.eval()
    with torch.no_grad():
        torch.testing.assert_close(destination(*sample), expected, rtol=0, atol=0)

    with pytest.raises(ActionCriticSidecarError, match="policy state key"):
        build_action_critic_checkpoint(model, metadata={"model_state_dict": {}})

    foreign_parameter = torch.nn.Parameter(torch.ones(1))
    mixed_optimizer = torch.optim.SGD([*model.parameters(), foreign_parameter], lr=1e-3)
    with pytest.raises(ActionCriticSidecarError, match="non-sidecar"):
        build_action_critic_checkpoint(model, optimizer=mixed_optimizer)

    poisoned = copy.deepcopy(payload)
    poisoned["model_state_dict"] = {}
    with pytest.raises(ActionCriticSidecarError, match="policy state keys"):
        load_action_critic_checkpoint(poisoned)
