from __future__ import annotations

import torch

from poke_bot.model import ActionConditionedLatentLookahead


def test_latent_policy_aid_is_exact_zero_at_initialization() -> None:
    module = ActionConditionedLatentLookahead(16, width=32, policy_aid_cap=0.25)
    options = torch.randn(3, 7, 16)
    state = torch.randn(3, 16)
    outputs = module(options, state)
    assert outputs["predicted_next_state_latent"].shape == (3, 7, 16)
    assert outputs["continuation_value"].shape == (3, 7)
    assert torch.count_nonzero(outputs["policy_aid"]).item() == 0


def test_latent_module_is_action_conditioned_and_neural_only() -> None:
    module = ActionConditionedLatentLookahead(8, width=16)
    state = torch.randn(1, 8)
    option_a = torch.zeros(1, 2, 8)
    option_b = option_a.clone()
    option_b[:, 1] = 1.0
    out_a = module(option_a, state)["predicted_next_state_latent"]
    out_b = module(option_b, state)["predicted_next_state_latent"]
    torch.testing.assert_close(out_a[:, 0], out_b[:, 0], rtol=0, atol=0)
    assert not torch.equal(out_a[:, 1], out_b[:, 1])
    inventory = module.inventory(action_authority_enabled=False)
    assert inventory["single_forward_pass"] is True
    assert inventory["mcts_allowed"] is False
    assert inventory["beam_search_allowed"] is False
    assert inventory["competition_time_simulator_search_allowed"] is False
    assert inventory["action_authority_enabled"] is False
