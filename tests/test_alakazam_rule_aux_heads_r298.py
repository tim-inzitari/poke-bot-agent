"""Isolated zero-off and gradient tests for the r298 derivative heads."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="r298 derivative head tests require optional torch",
)


def _target_vectors() -> dict:
    return {
        "lethal_threat": {"values": [1.0], "mask": [True]},
        "prize_race": {"values": [2.0] * 6, "mask": [True] * 6},
        "action_utility": {
            "values": [1.0] * 9,
            "mask": [True] * 9,
            "selected_option_indices": [0],
        },
        "game_phase": {"values": [1.0] * 6, "mask": [True] * 6},
        "terminal_conversion": {
            "values": [1.0] * 6,
            "mask": [True] * 6,
            "selected_option_indices": [0],
        },
        "turn_resources": {
            "values": [1.0] * 5,
            "mask": [True] * 5,
            "selected_option_indices": [0],
        },
        "attack_readiness": {
            "values": [1.0] * 4,
            "mask": [True] * 4,
            "selected_option_indices": [0],
        },
        "opponent_belief": {
            "hand_count_distribution": {"pairs": [[1, 2]], "mask": True},
            "remainder_count_distribution": {"pairs": [[2, 1]], "mask": True},
        },
    }


def test_r298_policy_route_is_exact_object_bypass_when_off() -> None:
    import torch

    from poke_bot.alakazam_rule_aux_heads_r298 import (
        R298RuleAuxHeadsConfig,
        R298RuleAuxiliaryHeads,
    )

    heads = R298RuleAuxiliaryHeads(
        R298RuleAuxHeadsConfig(d_model=4, route_width=5, belief_card_vocab=8)
    )
    base = torch.tensor([[1.0, -0.0]], dtype=torch.float32)
    # The disabled branch returns before it even validates / reads candidate
    # tensors, which is the strongest baseline-parity contract.
    bypass = heads.apply_to_policy(
        base,
        state_hidden=None,
        option_hidden=None,
        runtime_enabled=False,
    )
    assert bypass is base
    assert bypass.numpy().tobytes() == base.numpy().tobytes()

    gate_zero = heads.apply_to_policy(
        base,
        state_hidden=None,
        option_hidden=None,
        runtime_enabled=True,
        gate=0.0,
    )
    assert gate_zero is base
    assert gate_zero.numpy().tobytes() == base.numpy().tobytes()


def test_r298_armed_route_and_all_target_heads_have_gradients() -> None:
    import torch

    from poke_bot.alakazam_rule_aux_heads_r298 import (
        R298RuleAuxHeadsConfig,
        R298RuleAuxiliaryHeads,
        masked_rule_auxiliary_loss,
    )

    torch.manual_seed(7)
    heads = R298RuleAuxiliaryHeads(
        R298RuleAuxHeadsConfig(d_model=4, route_width=5, belief_card_vocab=8)
    )
    state = torch.randn(2, 4)
    options = torch.randn(2, 3, 4)
    base = torch.zeros(2, 3)
    prediction = heads.forward_heads(state, options)
    loss = masked_rule_auxiliary_loss(prediction, _target_vectors())

    # The final route is intentionally zero initialized; an armed call must
    # still create a gradient to it so isolated calibration can move it.
    armed = heads.apply_to_policy(
        base, state, options, runtime_enabled=True, gate=1.0
    )
    assert torch.equal(armed, base)  # zero-init residual initially
    # A linear probe has derivative one at the exact-zero residual, proving
    # the armed route can train from its safe initialization.
    loss = loss + armed.sum()
    loss.backward()

    for head in (
        heads.lethal_threat_head,
        heads.prize_race_head,
        heads.game_phase_head,
        heads.opponent_hand_belief_head,
        heads.opponent_remainder_belief_head,
        heads.action_utility_head,
        heads.terminal_conversion_head,
        heads.turn_resources_head,
        heads.attack_readiness_head,
    ):
        assert head.weight.grad is not None
        assert torch.count_nonzero(head.weight.grad) > 0
    assert heads.policy_route[-1].weight.grad is not None
    assert torch.count_nonzero(heads.policy_route[-1].weight.grad) > 0


def test_r298_config_remains_unwired_and_exact_zero() -> None:
    from poke_bot.alakazam_rule_aux_heads_r298 import load_r298_aux_heads_config

    config = load_r298_aux_heads_config()
    assert config["canonical_authority"]["goal_revision"] == 5
    assert config["canonical_authority"]["root_owner_revision"] == 303
    assert (
        config["canonical_authority"]["production_typed_source"]
        == "state/alakazam-new-list-direct-policy-r241.json"
    )
    assert config["runtime"]["runtime_wired"] is False
    assert config["runtime"]["enabled_default"] is False
    assert config["runtime"]["policy_gate"] == 0.0
    assert config["runtime"]["immediate_inzi_execution_authority"] is False
    assert config["training_boundary"]["candidate_training_enabled_now"] is False
