from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from poke_bot import checkpoint, config, features
from poke_bot.model import (
    COMBO_STATE_HEAD_NAME,
    COMBO_STATE_HEAD_OUTPUTS,
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
    DECISION_FUSION_V2_ROUTE_SCHEMA,
    DECISION_FUSION_V2_SCHEMA,
    DECISION_FUSION_V2_TOTAL_DELTA_CAP,
    DECISION_FUSION_V3_ROUTE_SCHEMA,
    DECISION_FUSION_V3_SCHEMA,
    EXPANDED_HEAD_KEY_PREFIXES,
    EXPANDED_HEAD_NAMES,
    EXPANDED_HEAD_SCHEMA,
    EXPANDED_HEAD_SCHEMA_VERSION,
    PackedSparse,
    SETUP_BOARD_OUTCOME_HEAD_NAME,
    SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
    build_model,
)
from poke_bot.strategic_losses import guide_pairwise_route_ranking_loss
from poke_bot.train import load_model_from_checkpoint


def _cfg(
    *,
    enabled: bool,
    fusion: bool = False,
    fusion_runtime: bool = False,
    setup_board_outcome: bool = False,
    combo_state: bool = False,
    combo_state_route_enabled: bool = True,
    dedicated_routes: bool = False,
    dedicated_routes_runtime: bool = False,
    typed_output_centered_routes: bool = False,
    action_type_reliability_cap: float = 1.0,
) -> config.ModelConfig:
    cfg = config.ModelConfig(
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
        setup_board_outcome_head_enabled=setup_board_outcome,
        combo_state_head_enabled=combo_state,
        combo_state_route_enabled=combo_state_route_enabled,
        decision_fusion_enabled=fusion,
        decision_fusion_runtime_enabled=fusion_runtime,
        decision_fusion_dedicated_routes_enabled=dedicated_routes,
        decision_fusion_dedicated_routes_runtime_enabled=(
            dedicated_routes_runtime
        ),
        decision_fusion_typed_output_centered_routes_enabled=(
            typed_output_centered_routes
        ),
        decision_fusion_action_type_reliability_cap=(
            action_type_reliability_cap
        ),
        decision_fusion_width=8,
        dropout=0.0,
    )
    return cfg


def _model(
    *,
    enabled: bool,
    fusion: bool = False,
    fusion_runtime: bool = False,
    setup_board_outcome: bool = False,
    combo_state: bool = False,
    combo_state_route_enabled: bool = True,
    dedicated_routes: bool = False,
    dedicated_routes_runtime: bool = False,
    typed_output_centered_routes: bool = False,
    action_type_reliability_cap: float = 1.0,
):
    return build_model(
        _cfg(
            enabled=enabled,
            fusion=fusion,
            fusion_runtime=fusion_runtime,
            setup_board_outcome=setup_board_outcome,
            combo_state=combo_state,
            combo_state_route_enabled=combo_state_route_enabled,
            dedicated_routes=dedicated_routes,
            dedicated_routes_runtime=dedicated_routes_runtime,
            typed_output_centered_routes=typed_output_centered_routes,
            action_type_reliability_cap=action_type_reliability_cap,
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


def _fusion_sources(model, *, batch: int, options: int):
    fusion = model.decision_fusion
    assert fusion is not None
    state_sources = {
        name: torch.randn(batch, projection.in_features)
        for name, projection in fusion.state_projections.items()
    }
    option_sources = {
        name: torch.randn(batch, options, width)
        for name, width in fusion._OPTION_DIMS.items()
    }
    if "setup_board_outcome" in fusion.required_heads:
        option_sources["setup_board_outcome"] = torch.randn(
            batch,
            options,
            SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
        )
    if "combo_state" in fusion.required_heads:
        option_sources["combo_state"] = torch.randn(
            batch,
            options,
            COMBO_STATE_HEAD_OUTPUTS,
        )
    return state_sources, option_sources


def test_slowking_combo_state_head_has_exact_scoped_capacity_and_route() -> None:
    model = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        combo_state=True,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
    )
    inventory = model.expanded_head_inventory()
    combo = inventory["modules"][COMBO_STATE_HEAD_NAME]
    assert combo["schema"] == "poke_bot.combo_state_head/v1"
    assert combo["outputs"] == 32
    assert combo["hidden_width"] == 192
    assert combo["parameters"] == 16 * 192 + 192 + 192 * 32 + 32
    fusion = model.decision_fusion_inventory()
    assert "combo_state" in fusion["required_heads"]
    route = model.decision_fusion.dedicated_routes["combo_state"]
    assert sum(parameter.numel() for parameter in route.parameters()) == (
        (16 + 32) * 16 + 16 + 16 + 1
    )
    option_hidden = torch.randn(2, 3, 16)
    assert model.combo_state_logits(option_hidden).shape == (2, 3, 32)


def test_combo_route_gate_keeps_tensors_but_excludes_policy_and_guide_gradients() -> None:
    """An H10-compatible combo module may be physically present but inactive."""

    model = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        combo_state=True,
        combo_state_route_enabled=False,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
        typed_output_centered_routes=True,
    )
    fusion = model.decision_fusion
    assert fusion is not None
    assert model.combo_state_head is not None
    assert model.combo_state_route_enabled is False
    # The architecture (including the V3 reliability scalar) stays loadable.
    assert "combo_state" in fusion.dedicated_routes
    assert "combo_state" in fusion.dedicated_route_log_reliability
    assert any(
        key.startswith("combo_state_head.") for key in model.state_dict()
    )
    assert any(
        key.startswith("decision_fusion.dedicated_routes.combo_state.")
        for key in model.state_dict()
    )
    assert (
        "decision_fusion.dedicated_route_log_reliability.combo_state"
        in model.state_dict()
    )

    assert "combo_state" in fusion.required_heads
    assert "combo_state" not in fusion.active_required_heads
    fusion_inventory = model.decision_fusion_inventory()["dedicated_routes"]
    assert "combo_state" in fusion_inventory["route_names"]
    assert "combo_state" not in fusion_inventory["active_route_names"]
    assert fusion_inventory["disabled_route_names"] == ["combo_state"]
    assert fusion_inventory["combo_state_route_enabled"] is False
    heads_inventory = model.expanded_head_inventory()
    assert COMBO_STATE_HEAD_NAME in heads_inventory["runtime_disabled_heads"]
    assert COMBO_STATE_HEAD_NAME not in heads_inventory["runtime_enabled_heads"]

    # Make active routes observably live before constructing either policy or
    # directional-guide autograd graphs.
    model.train()
    with torch.no_grad():
        for name, route in fusion.dedicated_routes.items():
            if name == "combo_state":
                continue
            route.network[-1].weight.fill_(0.05)
            route.network[-1].bias.zero_()
    option_hidden = torch.randn(2, 4, model.d_model)
    state_vec = torch.randn(2, model.d_model)
    state_sources, option_sources = model.decision_fusion_sources(
        option_hidden,
        state_vec,
    )
    assert "combo_state" not in option_sources
    route_deltas = fusion.dedicated_route_deltas(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )
    assert "combo_state" not in route_deltas
    guide_loss, guide_metrics = guide_pairwise_route_ranking_loss(
        route_deltas=route_deltas,
        guide_target_indices=torch.tensor([0, 1]),
        guide_confidences=torch.ones(2),
        option_counts=torch.tensor([4, 4]),
    )
    assert "combo_state" not in guide_metrics["heads"]

    # With combo's direct loss at zero, an ordinary policy plus directional
    # guide update must leave all combo tensors without a gradient.
    logits = model.decode_options(
        [_options(4), _options(4)],
        torch.randn(2, features.NUM_BOARD_TOKENS, model.d_model),
        torch.randn(2, model.d_model),
        n_options=[4, 4],
    )
    (
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        + guide_loss
    ).backward()
    combo_parameters = (
        *model.combo_state_head.parameters(),
        *fusion.dedicated_routes["combo_state"].parameters(),
        fusion.dedicated_route_log_reliability["combo_state"],
    )
    assert all(
        parameter.grad is None
        or not bool(torch.count_nonzero(parameter.grad).item())
        for parameter in combo_parameters
    )
    assert any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in fusion.dedicated_routes["action_q"].parameters()
    )


def test_expanded_heads_are_strictly_opt_in() -> None:
    model = _model(enabled=False)
    assert model.expanded_heads_enabled is False
    assert model.expanded_head_schema_version == 0
    assert not any(
        key.startswith(EXPANDED_HEAD_KEY_PREFIXES)
        for key in model.state_dict()
    )
    inventory = model.expanded_head_inventory()
    assert inventory["schema"] == EXPANDED_HEAD_SCHEMA
    assert inventory["version"] == 0
    assert inventory["enabled"] is False
    assert inventory["runtime_enabled_heads"] == []
    assert inventory["modules"] == {}
    assert inventory["fusion_roles"] == {}
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


def test_future_setup_head_requires_dedicated_fusion_v2_routes() -> None:
    with pytest.raises(ValueError, match="dedicated decision-fusion routes"):
        _model(
            enabled=True,
            fusion=True,
            setup_board_outcome=True,
        )
    with pytest.raises(ValueError, match="require decision_fusion_enabled"):
        _model(
            enabled=True,
            dedicated_routes=True,
        )
    with pytest.raises(ValueError, match="requires route tensors"):
        _model(
            enabled=True,
            fusion=True,
            dedicated_routes_runtime=True,
        )
    with pytest.raises(ValueError, match="requires decision fusion runtime"):
        _model(
            enabled=True,
            fusion=True,
            dedicated_routes=True,
            dedicated_routes_runtime=True,
        )


def test_fusion_v2_is_exactly_v1_safe_at_step_zero() -> None:
    torch.manual_seed(20260730)
    v1 = _model(enabled=True, fusion=True, fusion_runtime=True)
    torch.manual_seed(20260730)
    v2 = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
    )
    v1.eval()
    v2.eval()

    v1_state = v1.state_dict()
    v2_state = v2.state_dict()
    for key, value in v1_state.items():
        torch.testing.assert_close(value, v2_state[key], rtol=0, atol=0)
    route_keys = {
        key for key in v2_state if key.startswith("decision_fusion.dedicated_routes.")
    }
    setup_keys = {
        key for key in v2_state if key.startswith("setup_board_outcome_head.")
    }
    assert route_keys
    assert setup_keys
    assert set(v2_state) == set(v1_state) | route_keys | setup_keys

    spatial = torch.randn(3, features.NUM_BOARD_TOKENS, v1.d_model)
    state = torch.randn(3, v1.d_model)
    options = [_options(4), _options(3), _options(2)]
    counts = [4, 3, 2]
    v1_logits = v1.decode_options(options, spatial, state, n_options=counts)
    v2_logits = v2.decode_options(options, spatial, state, n_options=counts)
    torch.testing.assert_close(v1_logits, v2_logits, rtol=0, atol=0)
    assert torch.equal(v1_logits.argmax(dim=1), v2_logits.argmax(dim=1))

    inventory = v2.decision_fusion_inventory()
    assert inventory["schema"] == DECISION_FUSION_V2_SCHEMA
    assert inventory["required_heads"] == [
        *DECISION_FUSION_REQUIRED_HEADS,
        "setup_board_outcome",
    ]
    routes = inventory["dedicated_routes"]
    assert routes["schema"] == DECISION_FUSION_V2_ROUTE_SCHEMA
    assert routes["route_count"] == len(DECISION_FUSION_REQUIRED_HEADS) + 1
    assert routes["route_names"] == inventory["required_heads"]
    assert routes["aggregation"] == "fixed_mean"
    assert routes["total_delta_cap"] == DECISION_FUSION_V2_TOTAL_DELTA_CAP
    assert routes["zero_safe_final_projection"] is True
    assert routes["action_influence"] == "bounded_option_conditioned_route"
    assert routes["state_head_action_conditioning"] == (
        "typed_output_plus_board_state_cross_attended_legal_option"
    )
    assert routes["option_head_action_conditioning"] == (
        "typed_option_output_plus_board_state_cross_attended_legal_option"
    )
    assert inventory["guide_excluded"] is True
    payload = checkpoint.build_checkpoint(model=v2, model_config=v2.cfg)
    assert (
        payload["model_config"]["decision_fusion_dedicated_routes_enabled"]
        is True
    )
    assert (
        payload["model_config"][
            "decision_fusion_dedicated_routes_runtime_enabled"
        ]
        is True
    )
    assert payload["model_config"]["setup_board_outcome_head_enabled"] is True
    assert payload["provenance"]["decision_fusion"]["schema"] == (
        DECISION_FUSION_V2_SCHEMA
    )

    head = v2.expanded_head_inventory()["modules"][
        SETUP_BOARD_OUTCOME_HEAD_NAME
    ]
    assert head["computation_role"] == "independent_head"
    assert head["fusion_role"] == "fused_input"
    assert head["action_influence"] == "bounded_option_conditioned_route"
    assert head["direct_action_selection_authority"] is False


def test_fusion_v2_routes_are_distinct_option_conditioned_and_bounded() -> None:
    model = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
    )
    fusion = model.decision_fusion
    assert fusion is not None
    with torch.no_grad():
        for index, route in enumerate(fusion.dedicated_routes.values(), start=1):
            route.network[-1].weight.fill_(0.01 * index)
            route.network[-1].bias.zero_()

    batch, option_count = 3, 5
    option_hidden = torch.randn(batch, option_count, model.d_model)
    state_sources, option_sources = _fusion_sources(
        model,
        batch=batch,
        options=option_count,
    )
    route_deltas = fusion.dedicated_route_deltas(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )
    assert set(route_deltas) == set(fusion.required_heads)
    for name, delta in route_deltas.items():
        assert tuple(delta.shape) == (batch, option_count), name
        # Every state-level and option-level head owns a genuinely
        # option-conditioned route, not a row-constant diagnostic offset.
        centered = delta - delta.mean(dim=1, keepdim=True)
        assert bool(torch.count_nonzero(centered).item()), name

    total = fusion.dedicated_action_delta(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )
    assert bool(
        (
            total.abs()
            <= DECISION_FUSION_V2_TOTAL_DELTA_CAP + 1e-7
        ).all()
    )

    base_logits = torch.randn(batch, option_count)
    full = fusion(
        option_hidden,
        base_logits,
        state_sources=state_sources,
        option_sources=option_sources,
        dedicated_routes_active=True,
    )
    for name, route in fusion.dedicated_routes.items():
        saved_weight = route.network[-1].weight.detach().clone()
        saved_bias = route.network[-1].bias.detach().clone()
        with torch.no_grad():
            route.network[-1].weight.zero_()
            route.network[-1].bias.zero_()
        ablated = fusion(
            option_hidden,
            base_logits,
            state_sources=state_sources,
            option_sources=option_sources,
            dedicated_routes_active=True,
        )
        differential = full - ablated
        differential = differential - differential.mean(dim=1, keepdim=True)
        assert bool(torch.count_nonzero(differential).item()), name
        with torch.no_grad():
            route.network[-1].weight.copy_(saved_weight)
            route.network[-1].bias.copy_(saved_bias)


def test_fusion_v3_routes_require_typed_output_and_bound_reliability() -> None:
    model = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
        typed_output_centered_routes=True,
        action_type_reliability_cap=0.25,
    )
    fusion = model.decision_fusion
    assert fusion is not None
    assert fusion.inventory(runtime_enabled=True)["schema"] == DECISION_FUSION_V3_SCHEMA
    assert (
        fusion.inventory(runtime_enabled=True)["dedicated_routes"]["schema"]
        == DECISION_FUSION_V3_ROUTE_SCHEMA
    )
    with torch.no_grad():
        for route in fusion.dedicated_routes.values():
            route.network[-1].weight.fill_(0.1)
            route.network[-1].bias.fill_(0.2)

    batch, option_count = 2, 4
    option_hidden = torch.randn(batch, option_count, model.d_model)
    state_sources, option_sources = _fusion_sources(
        model, batch=batch, options=option_count
    )
    original_action_type = option_sources["action_type"]
    option_sources["action_type"] = torch.zeros_like(original_action_type)
    zero_delta = fusion.dedicated_route_deltas(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )["action_type"]
    torch.testing.assert_close(zero_delta, torch.zeros_like(zero_delta), rtol=0, atol=0)

    option_sources["action_type"] = original_action_type
    deltas = fusion.dedicated_route_deltas(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )
    assert bool(torch.count_nonzero(deltas["action_type"]).item())
    assert bool((deltas["action_type"].abs() <= 0.25 + 1e-7).all())
    total = fusion.dedicated_action_delta(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )
    assert bool((total.abs() <= DECISION_FUSION_V2_TOTAL_DELTA_CAP + 1e-7).all())


def test_fusion_v2_serving_route_is_separately_receipt_gated() -> None:
    model = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        dedicated_routes=True,
        dedicated_routes_runtime=False,
    )
    model.eval()
    fusion = model.decision_fusion
    assert fusion is not None
    with torch.no_grad():
        for route in fusion.dedicated_routes.values():
            route.network[-1].weight.fill_(0.1)
            route.network[-1].bias.zero_()

    spatial = torch.randn(2, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(2, model.d_model)
    options = [_options(4), _options(4)]
    before_receipt = model.decode_options(
        options,
        spatial,
        state,
        n_options=[4, 4],
    )
    model.decision_fusion_dedicated_routes_runtime_enabled = True
    after_receipt = model.decode_options(
        options,
        spatial,
        state,
        n_options=[4, 4],
    )
    assert not torch.equal(before_receipt, after_receipt)
    assert (
        model.decision_fusion_inventory()["dedicated_routes"][
            "runtime_enabled"
        ]
        is True
    )


def test_fusion_v2_policy_loss_reaches_every_head_and_dedicated_route() -> None:
    model = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
    )
    fusion = model.decision_fusion
    assert fusion is not None
    with torch.no_grad():
        for route in fusion.dedicated_routes.values():
            route.network[-1].weight.fill_(0.05)
            route.network[-1].bias.zero_()

    model.train()
    spatial = torch.randn(4, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(4, model.d_model)
    logits = model.decode_options(
        [_options(4), _options(4), _options(4), _options(4)],
        spatial,
        state,
        n_options=[4, 4, 4, 4],
    )
    torch.nn.functional.cross_entropy(
        logits,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    ).backward()

    source_modules = (
        model.value_head,
        model.aux_head,
        model.opp_hand_head,
        model.opp_remainder_head,
        model.lethal_threat_head,
        model.prize_race_head,
        *(getattr(model, name) for name in EXPANDED_HEAD_NAMES),
        model.setup_board_outcome_head,
    )
    for module in source_modules:
        assert module is not None
        assert any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in module.parameters()
        )
    for name, route in fusion.dedicated_routes.items():
        assert any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in route.parameters()
        ), name


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


def test_decision_fusion_v1_to_v2_additive_migration_is_exact(
    tmp_path,
) -> None:
    source = _model(enabled=True, fusion=True, fusion_runtime=True)
    target_cfg = _cfg(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        combo_state=True,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
    )
    inherited_fusion = sorted(
        key
        for key in source.state_dict()
        if key.startswith("decision_fusion.")
    )
    route_names = sorted(
        [
            *DECISION_FUSION_REQUIRED_HEADS,
            "setup_board_outcome",
            "combo_state",
        ]
    )
    payload = checkpoint.build_checkpoint(
        model=source,
        model_config=source.cfg,
        extra={
            "decision_fusion_migration": {
                "schema": "poke_bot.causal_decision_fusion_v2_migration/v1",
                "source_schema": DECISION_FUSION_SCHEMA,
                "target_schema": DECISION_FUSION_V2_SCHEMA,
                "zero_safe_initialization": True,
                "runtime_enabled": True,
                "activation_scope": "isolated_specialist_bootstrap",
                "serving_eligible": False,
                "all_inherited_tensors_preserved": True,
                "inherited_fusion_tensor_keys": inherited_fusion,
                "new_dedicated_route_names": route_names,
                "new_auxiliary_head_names": [
                    SETUP_BOARD_OUTCOME_HEAD_NAME,
                    COMBO_STATE_HEAD_NAME,
                ],
            }
        },
    )
    payload["model_config"] = target_cfg.__dict__.copy()
    path = checkpoint.atomic_torch_save(
        payload,
        tmp_path / "fusion-v1-to-v2.pt",
    )

    migrated = load_model_from_checkpoint(path, device=torch.device("cpu"))

    assert migrated.warm_started_decision_fusion is True
    assert set(migrated.decision_fusion.dedicated_routes) == set(route_names)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(
            value, migrated.state_dict()[key], rtol=0, atol=0
        )
    materialized = checkpoint.build_checkpoint(
        model=migrated,
        model_config=migrated.cfg,
        extra=payload["extra"],
    )
    materialized_path = checkpoint.atomic_torch_save(
        materialized,
        tmp_path / "fusion-v2-materialized.pt",
    )
    reloaded = load_model_from_checkpoint(
        materialized_path,
        device=torch.device("cpu"),
    )
    assert set(reloaded.decision_fusion.dedicated_routes) == set(route_names)

    partial = checkpoint.load_checkpoint(path, map_location="cpu")
    target = _model(
        enabled=True,
        fusion=True,
        fusion_runtime=True,
        setup_board_outcome=True,
        combo_state=True,
        dedicated_routes=True,
        dedicated_routes_runtime=True,
    )
    route_key = next(
        key
        for key in target.state_dict()
        if key.startswith("decision_fusion.dedicated_routes.")
    )
    partial["model_state_dict"][route_key] = target.state_dict()[route_key]
    partial_path = checkpoint.atomic_torch_save(
        partial,
        tmp_path / "fusion-v1-to-v2-partial.pt",
    )
    with pytest.raises(RuntimeError, match="partially missing"):
        load_model_from_checkpoint(partial_path, device=torch.device("cpu"))
