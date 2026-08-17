from __future__ import annotations

import torch

from poke_bot import config
from poke_bot.h10_migration import build_h10_child
from poke_bot.model import CausalDecisionFusion, build_model


def _ordinary_parent():
    cfg = config.ModelConfig(
        d_model=96,
        spatial_layers=4,
        temporal_layers=1,
        option_decoder_layers=4,
        n_heads=8,
        ff_dim=384,
        max_context=320,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dense_card2vec=True,
        expanded_heads_enabled=True,
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=True,
        dropout=0.0,
    )
    return build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=96,
        belief_card_vocab=32,
    )


def test_h10_expansion_is_exact_and_carries_nineteen_routes() -> None:
    parent = _ordinary_parent()
    child, evidence = build_h10_child(parent)
    assert evidence["step_zero_parity"]["passed"] is True
    assert set(evidence["step_zero_parity"]["max_abs_differences"].values()) == {0.0}
    assert (child.cfg.spatial_layers, child.cfg.temporal_layers) == (7, 3)
    assert child.cfg.option_decoder_layers == 7
    assert child.cfg.ff_dim == 2496
    assert child.cfg.h10_capacity_enabled is True
    assert isinstance(child.decision_fusion, CausalDecisionFusion)
    assert len(child.decision_fusion.required_heads) == 19
    assert tuple(child.decision_fusion.required_heads) == tuple(
        child.decision_fusion.dedicated_routes
    )


def test_h10_widening_preserves_parent_neurons_and_zeroes_new_outputs() -> None:
    parent = _ordinary_parent()
    child, _ = build_h10_child(parent)
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    incoming = "spatial_encoder.layers.0.linear1.weight"
    outgoing = "spatial_encoder.layers.0.linear2.weight"
    assert torch.equal(child_state[incoming][:384], parent_state[incoming])
    assert torch.equal(child_state[outgoing][:, :384], parent_state[outgoing])
    assert torch.count_nonzero(child_state[outgoing][:, 384:]).item() == 0
    for key in (
        "spatial_encoder.layers.4.self_attn.out_proj.weight",
        "spatial_encoder.layers.4.linear2.weight",
        "temporal_blocks.1.attn.out.weight",
        "temporal_blocks.1.ff.3.weight",
        "option_decoder.4.cross.out_proj.weight",
        "option_decoder.4.ff.3.weight",
    ):
        assert torch.count_nonzero(child_state[key]).item() == 0
    for key, tensor in child_state.items():
        if (
            key.startswith("h10_head_residuals.")
            or key.startswith("decision_fusion.dedicated_routes.")
        ) and ".network.2." in key:
            assert torch.count_nonzero(tensor).item() == 0


def test_h10_can_materialize_directional_fusion_v3_before_training() -> None:
    parent = _ordinary_parent()
    child, evidence = build_h10_child(parent, directional_fusion_v3=True)

    assert evidence["directional_fusion_v3"] is True
    assert evidence["step_zero_parity"]["passed"] is True
    assert child.cfg.decision_fusion_typed_output_centered_routes_enabled is True
    assert child.cfg.decision_fusion_action_type_reliability_cap == 0.25
    fusion = child.decision_fusion
    assert isinstance(fusion, CausalDecisionFusion)
    assert fusion.typed_output_centered_routes is True
    assert len(fusion.dedicated_route_log_reliability) == 19
    assert all(
        float(value.detach().item()) == 0.0
        for value in fusion.dedicated_route_log_reliability.values()
    )
    inventory = child.decision_fusion_inventory()
    assert inventory["schema"] == "poke_bot.causal_decision_fusion/v3"
    assert (
        inventory["dedicated_routes"]["schema"]
        == "typed_output_centered_per_head/v3"
    )
    assert inventory["dedicated_routes"]["reliability_bounds"] == [0.25, 4.0]
    assert inventory["dedicated_routes"]["action_type_reliability_cap"] == 0.25
