"""Focused safety tests for the isolated Prize-plan-v2 sidecar."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from poke_bot.prize_plan_v2_sidecar import (  # noqa: E402
    PRIZE_PLAN_V2_OUTPUT_NAMES,
    PrizePlanV2Sidecar,
    PrizePlanV2SidecarConfig,
    PrizePlanV2SidecarError,
    build_prize_plan_v2_checkpoint,
    load_prize_plan_v2_checkpoint,
    masked_prize_plan_v2_loss,
    restore_prize_plan_v2_checkpoint,
)


def _model() -> PrizePlanV2Sidecar:
    torch.manual_seed(23)
    return PrizePlanV2Sidecar(
        PrizePlanV2SidecarConfig(
            feature_dim=4,
            state_hidden_dim=12,
            action_hidden_dim=10,
            q_hidden_dim=14,
            max_action_stages=4,
            max_legal_options=8,
            max_action_program_tokens=4,
            max_action_token_value=7,
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
    return (
        torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [[-0.2, 0.1, 0.3, 0.8], [0.9, -0.5, 0.2, 0.4]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[True, True], [True, True]]),
        torch.tensor(
            [
                [[0.4, -0.5, 0.6, -0.7], [0.8, 0.9, -0.1, 0.2]],
                [[-0.9, 0.4, 0.2, 0.5], [0.3, -0.2, 0.1, 0.8]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[True, True], [True, True]]),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        torch.tensor([[2, 3], [3, 2]], dtype=torch.int64),
        torch.tensor(
            [
                [[0, 0], [1, 2]],
                [[2, 1], [0, 0]],
            ],
            dtype=torch.int64,
        ),
        torch.tensor(
            [
                [[True, False], [True, True]],
                [[True, True], [True, False]],
            ]
        ),
    )


def _materialize_adamw_state(model: PrizePlanV2Sidecar) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss = model(*_batch()).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _first_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for child in value.values():
            try:
                return _first_tensor(child)
            except LookupError:
                pass
    if isinstance(value, list):
        for child in value:
            try:
                return _first_tensor(child)
            except LookupError:
                pass
    raise LookupError("no tensor")


def test_v_menu_isolated_q_structure_and_masked_loss() -> None:
    model = _model().eval()
    inputs = _batch()
    with torch.no_grad():
        original = model(*inputs)
        changed = model(
            inputs[0],
            inputs[1],
            inputs[2] + 5.0,
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
        )
    assert original.shape == (2, 8)
    assert model.output_names == PRIZE_PLAN_V2_OUTPUT_NAMES
    assert torch.isfinite(original).all()
    assert torch.all(original >= -1.0)
    assert torch.all(original <= 1.0)
    torch.testing.assert_close(original[:, (0, 2, 4, 6)], changed[:, (0, 2, 4, 6)], rtol=0, atol=0)
    assert not torch.allclose(original[:, (1, 3, 5, 7)], changed[:, (1, 3, 5, 7)])

    result = masked_prize_plan_v2_loss(
        original,
        torch.tensor([[0.2, -0.1, 0.5, -0.8], [0.1, 0.3, -0.4, 0.7]], dtype=torch.float32),
        torch.tensor([[True, True, False, True], [True, False, True, True]]),
    )
    assert torch.isfinite(result.total)
    assert result.valid_by_horizon == {1: 2, 3: 1, 6: 1, 12: 2}


def test_checkpoint_is_sidecar_only_and_recursively_rejects_policy_state() -> None:
    model = _model().eval()
    optimizer = _materialize_adamw_state(model)
    payload = build_prize_plan_v2_checkpoint(
        model,
        optimizer=optimizer,
        training_state={"epoch": 1, "nested": {"receipt": "ok"}},
        metadata={"source": {"target_set_sha256": "sha256:fixture"}},
    )
    assert set(payload) == {
        "schema",
        "sidecar_schema",
        "config",
        "sidecar_state_dict",
        "optimizer_state_dict",
        "training_state",
        "metadata",
    }
    loaded = load_prize_plan_v2_checkpoint(payload)
    assert loaded.training_state == {"epoch": 1, "nested": {"receipt": "ok"}}

    with pytest.raises(PrizePlanV2SidecarError, match="policy state key"):
        build_prize_plan_v2_checkpoint(
            model,
            metadata={"ordinary": {"runtime_state_dict": {}}},
        )
    with pytest.raises(PrizePlanV2SidecarError, match="never tensors"):
        build_prize_plan_v2_checkpoint(model, metadata={"ordinary": torch.ones(1)})

    poisoned = copy.deepcopy(payload)
    poisoned["metadata"]["deep"] = {"policy_model_state_dict": {}}
    with pytest.raises(PrizePlanV2SidecarError, match="forbidden policy state key"):
        load_prize_plan_v2_checkpoint(poisoned)


def test_checkpoint_rejects_nonfinite_sidecar_and_optimizer_tensors() -> None:
    model = _model()
    payload = build_prize_plan_v2_checkpoint(model, optimizer=_materialize_adamw_state(model))

    poisoned_model = copy.deepcopy(payload)
    next(iter(poisoned_model["sidecar_state_dict"].values())).fill_(float("nan"))
    with pytest.raises(PrizePlanV2SidecarError, match="non-finite"):
        load_prize_plan_v2_checkpoint(poisoned_model)

    poisoned_optimizer = copy.deepcopy(payload)
    _first_tensor(poisoned_optimizer["optimizer_state_dict"]).fill_(float("nan"))
    with pytest.raises(PrizePlanV2SidecarError, match="non-finite"):
        load_prize_plan_v2_checkpoint(poisoned_optimizer)


def test_save_and_restore_reject_foreign_optimizer_and_restore_sidecar_only() -> None:
    model = _model().eval()
    optimizer = _materialize_adamw_state(model)
    payload = build_prize_plan_v2_checkpoint(model, optimizer=optimizer)
    foreign_parameter = torch.nn.Parameter(torch.ones(1))
    mixed = torch.optim.AdamW([*model.parameters(), foreign_parameter], lr=1.0e-3)
    with pytest.raises(PrizePlanV2SidecarError, match="non-sidecar"):
        build_prize_plan_v2_checkpoint(model, optimizer=mixed)

    destination = _model()
    destination_mixed = torch.optim.AdamW(
        [*destination.parameters(), foreign_parameter], lr=1.0e-3
    )
    with pytest.raises(PrizePlanV2SidecarError, match="non-sidecar"):
        restore_prize_plan_v2_checkpoint(destination, payload, optimizer=destination_mixed)

    clean_destination = _model()
    clean_optimizer = torch.optim.AdamW(clean_destination.parameters(), lr=1.0e-3)
    restored = restore_prize_plan_v2_checkpoint(
        clean_destination, payload, optimizer=clean_optimizer
    )
    assert restored.model is clean_destination
    for parameter in clean_destination.parameters():
        assert parameter.dtype == torch.float32

