from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from poke_bot import checkpoint, config, features
from poke_bot.model import (
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
    EXPANDED_HEAD_KEY_PREFIXES,
    EXPANDED_HEAD_NAMES,
    EXPANDED_HEAD_SCHEMA,
    EXPANDED_HEAD_SCHEMA_VERSION,
    PackedSparse,
    build_model,
)
from poke_bot.train import load_model_from_checkpoint


def _cfg(
    *,
    enabled: bool,
    fusion: bool = False,
    fusion_runtime: bool = False,
) -> config.ModelConfig:
    return config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        expanded_heads_enabled=enabled,
        decision_fusion_enabled=fusion,
        decision_fusion_runtime_enabled=fusion_runtime,
        decision_fusion_width=8,
        dropout=0.0,
    )


def _model(
    *,
    enabled: bool,
    fusion: bool = False,
    fusion_runtime: bool = False,
):
    return build_model(
        _cfg(
            enabled=enabled,
            fusion=fusion,
            fusion_runtime=fusion_runtime,
        ),
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )


def _options(n: int) -> features.SparseVector:
    value = features.SparseVector()
    for index in range(n):
        value.word_start()
        value.add(index + 1, 1.0)
    return value


def test_expanded_heads_are_strictly_opt_in() -> None:
    model = _model(enabled=False)
    assert model.expanded_heads_enabled is False
    assert model.expanded_head_schema_version == 0
    assert not any(
        key.startswith(EXPANDED_HEAD_KEY_PREFIXES)
        for key in model.state_dict()
    )
    inventory = model.expanded_head_inventory()
    assert inventory == {
        "schema": EXPANDED_HEAD_SCHEMA,
        "version": 0,
        "enabled": False,
        "runtime_enabled_heads": [],
        "modules": {},
    }
    with pytest.raises(RuntimeError, match="disabled"):
        model.expanded_state_logits(torch.zeros(1, model.d_model))
    with pytest.raises(RuntimeError, match="disabled"):
        model.expanded_option_logits(torch.zeros(1, 2, model.d_model))


def test_decode_hidden_api_preserves_default_logits_and_head_shapes() -> None:
    model = _model(enabled=True)
    model.eval()
    batch = 2
    spatial = torch.randn(batch, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(batch, model.d_model)
    options = [_options(3), _options(2)]
    counts = [3, 2]

    default = model.decode_options(
        options, spatial, state, n_options=counts
    )
    logits, hidden = model.decode_options(
        options,
        spatial,
        state,
        n_options=counts,
        return_hidden=True,
    )
    assert isinstance(default, torch.Tensor)
    torch.testing.assert_close(default, logits)
    assert tuple(hidden.shape) == (batch, 3, model.d_model)
    assert torch.isneginf(logits[1, 2])

    option = model.expanded_option_logits(hidden)
    assert set(option) == {
        "action_q",
        "action_type",
        "action_target",
        "action_resource",
        "action_utility",
    }
    for name in ("action_q", "action_type", "action_target", "action_resource"):
        assert tuple(option[name].shape) == (batch, 3)
    assert tuple(option["action_utility"].shape) == (batch, 3, 6)

    state_out = model.expanded_state_logits(state)
    assert tuple(state_out["tactical_outcome"].shape) == (batch, 3, 6)
    assert tuple(state_out["opponent_response"].shape) == (batch, 7)
    assert tuple(state_out["resource_forecast"].shape) == (batch, 6)
    assert tuple(state_out["game_phase"].shape) == (batch, 5)
    assert tuple(state_out["outcome_distribution"].shape) == (batch, 3)
    assert tuple(state_out["remaining_turns"].shape) == (batch, 1)


def test_packed_decode_hidden_api_preserves_default_tensor_contract() -> None:
    model = _model(enabled=True)
    model.eval()
    batch = 2
    max_options = 3
    spatial = torch.randn(batch, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(batch, model.d_model)
    packed = PackedSparse(
        index=torch.empty(0, dtype=torch.long),
        value=torch.empty(0),
        offset=torch.zeros(batch * max_options + 1, dtype=torch.long),
    )
    counts = torch.tensor([3, 1])
    default = model.decode_options_packed(
        packed,
        spatial,
        state,
        n_options=counts,
        batch_size=batch,
    )
    logits, hidden = model.decode_options_packed(
        packed,
        spatial,
        state,
        n_options=counts,
        batch_size=batch,
        return_hidden=True,
    )
    assert isinstance(default, torch.Tensor)
    torch.testing.assert_close(default, logits)
    assert tuple(hidden.shape) == (batch, max_options, model.d_model)
    assert torch.isneginf(logits[1, 1:]).all()


def test_expanded_head_initialization_is_deterministic_and_rng_isolated() -> None:
    torch.manual_seed(2468)
    base = _model(enabled=False)
    tail_without_heads = torch.rand(8)

    torch.manual_seed(2468)
    expanded = _model(enabled=True)
    tail_with_heads = torch.rand(8)
    torch.testing.assert_close(tail_without_heads, tail_with_heads, rtol=0, atol=0)

    base_state = base.state_dict()
    expanded_state = expanded.state_dict()
    for key, value in base_state.items():
        torch.testing.assert_close(value, expanded_state[key], rtol=0, atol=0)

    torch.manual_seed(1)
    first = _model(enabled=True)
    torch.manual_seed(9999)
    second = _model(enabled=True)
    for name in EXPANDED_HEAD_NAMES:
        first_module = getattr(first, name)
        second_module = getattr(second, name)
        assert first_module is not None and second_module is not None
        for key, value in first_module.state_dict().items():
            torch.testing.assert_close(
                value, second_module.state_dict()[key], rtol=0, atol=0
            )


def test_v5_state_load_preserves_every_existing_tensor() -> None:
    torch.manual_seed(77)
    v5 = _model(enabled=False)
    v5_state = {
        key: value.detach().clone() for key, value in v5.state_dict().items()
    }
    v6 = build_model(
        replace(v5.cfg, expanded_heads_enabled=True),
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )
    incompatible = v6.load_state_dict(v5_state, strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith(EXPANDED_HEAD_KEY_PREFIXES)
        for key in incompatible.missing_keys
    )
    for key, value in v5_state.items():
        torch.testing.assert_close(value, v6.state_dict()[key], rtol=0, atol=0)


def test_checkpoint_records_expanded_head_schema_and_tensor_inventory() -> None:
    model = _model(enabled=True)
    payload = checkpoint.build_checkpoint(model=model, model_config=model.cfg)
    metadata = payload["provenance"]["expanded_heads"]
    assert metadata["schema"] == EXPANDED_HEAD_SCHEMA
    assert metadata["version"] == EXPANDED_HEAD_SCHEMA_VERSION
    assert metadata["enabled"] is True
    assert metadata["runtime_enabled_heads"] == []
    assert set(metadata["modules"]) == set(EXPANDED_HEAD_NAMES)
    for name, module in metadata["modules"].items():
        assert module["input"] in {"option", "state"}
        assert module["outputs"] > 0
        assert module["parameters"] > 0
        assert set(module["tensors"]) == {"weight", "bias"}
        for tensor in module["tensors"].values():
            assert tensor["numel"] > 0
            assert tensor["dtype"] == "float32"
    assert payload["provenance"]["warm_started_expanded_heads"] == []


def test_decision_fusion_is_zero_safe_and_checkpoint_declared() -> None:
    model = _model(enabled=True, fusion=True, fusion_runtime=True)
    model.eval()
    spatial = torch.randn(2, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(2, model.d_model)
    options = [_options(3), _options(2)]
    model.decision_fusion_runtime_enabled = False
    flat = model.decode_options(options, spatial, state, n_options=[3, 2])
    model.decision_fusion_runtime_enabled = True
    fused_initial = model.decode_options(options, spatial, state, n_options=[3, 2])
    torch.testing.assert_close(flat, fused_initial, rtol=0, atol=0)

    payload = checkpoint.build_checkpoint(model=model, model_config=model.cfg)
    inventory = payload["provenance"]["decision_fusion"]
    assert inventory["schema"] == DECISION_FUSION_SCHEMA
    assert inventory["enabled"] is True
    assert inventory["runtime_enabled"] is True
    assert inventory["required_heads"] == list(DECISION_FUSION_REQUIRED_HEADS)
    assert inventory["parameters"] > 0


def test_staged_decision_fusion_trains_before_runtime_activation() -> None:
    model = _model(enabled=True, fusion=True, fusion_runtime=False)
    spatial = torch.randn(2, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(2, model.d_model)
    options = [_options(3), _options(3)]

    model.eval()
    flat = model.decode_options(options, spatial, state, n_options=[3, 3])
    assert model.decision_fusion is not None
    with torch.no_grad():
        model.decision_fusion.residual[-1].weight.fill_(0.05)
    bypassed = model.decode_options(options, spatial, state, n_options=[3, 3])
    torch.testing.assert_close(flat, bypassed, rtol=0, atol=0)

    model.train()
    training_logits = model.decode_options(
        options, spatial, state, n_options=[3, 3]
    )
    assert not torch.equal(flat, training_logits)
    torch.nn.functional.cross_entropy(
        training_logits, torch.tensor([0, 1], dtype=torch.long)
    ).backward()
    assert any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in model.decision_fusion.parameters()
    )


def test_decision_fusion_backpropagates_policy_loss_to_every_required_head() -> None:
    model = _model(enabled=True, fusion=True, fusion_runtime=True)
    assert model.decision_fusion is not None
    with torch.no_grad():
        model.decision_fusion.residual[-1].weight.fill_(0.05)
    model.train()
    spatial = torch.randn(2, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(2, model.d_model)
    logits = model.decode_options(
        [_options(3), _options(3)],
        spatial,
        state,
        n_options=[3, 3],
    )
    torch.nn.functional.cross_entropy(
        logits, torch.tensor([0, 1], dtype=torch.long)
    ).backward()

    required_modules = (
        model.value_head,
        model.aux_head,
        model.opp_hand_head,
        model.opp_remainder_head,
        model.lethal_threat_head,
        model.prize_race_head,
        *(getattr(model, name) for name in EXPANDED_HEAD_NAMES),
    )
    for module in required_modules:
        assert module is not None
        assert any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in module.parameters()
        )


def test_decision_fusion_migration_is_explicit_and_zero_safe(
    tmp_path,
) -> None:
    source = _model(enabled=True)
    payload = checkpoint.build_checkpoint(
        model=source,
        model_config=source.cfg,
        extra={
            "decision_fusion_migration": {
                "schema": "poke_bot.causal_decision_fusion_migration/v1",
                "target_schema": DECISION_FUSION_SCHEMA,
                "zero_safe_initialization": True,
                "runtime_enabled": False,
            }
        },
    )
    payload["model_config"]["decision_fusion_enabled"] = True
    payload["model_config"]["decision_fusion_runtime_enabled"] = False
    path = checkpoint.atomic_torch_save(payload, tmp_path / "fusion-migration.pt")

    migrated = load_model_from_checkpoint(path, device=torch.device("cpu"))

    assert migrated.decision_fusion_enabled is True
    assert migrated.decision_fusion_runtime_enabled is False
    assert migrated.warm_started_decision_fusion is True
    for key, value in source.state_dict().items():
        torch.testing.assert_close(
            value, migrated.state_dict()[key], rtol=0, atol=0
        )
