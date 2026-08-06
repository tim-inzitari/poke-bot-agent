"""Swap-in wiring tests for Recursive Turn Planner ↔ PolicyAgent."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from poke_bot.agent import PolicyAgent
from poke_bot.recursive_turn_planner import (
    resolve_rtp_config_for_model,
    turn_key_from_obs,
)
from poke_bot.recursive_turn_planner.agent_bridge import RTPAgentBridge


def _mock_model(d_model: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        d_model=d_model,
        latent_lookahead=None,
        latent_lookahead_enabled=False,
        eval=lambda: None,
        parameters=lambda: iter((torch.zeros(1),)),
    )


@pytest.mark.unit
def test_turn_key_from_obs_dict() -> None:
    assert turn_key_from_obs({"current": {"yourIndex": 1, "turn": 7}}) == (1, 7)


@pytest.mark.unit
def test_resolve_rtp_config_binds_to_model_width() -> None:
    cfg = resolve_rtp_config_for_model(_mock_model(16))  # type: ignore[arg-type]
    assert cfg.d_model == 16
    assert cfg.dynamics_width == 32
    pure = resolve_rtp_config_for_model(None, profile_name="pure_rl")
    assert pure.d_model == 96
    global_cfg = resolve_rtp_config_for_model(_mock_model(256))  # type: ignore[arg-type]
    assert global_cfg.sizing_profile == "global_transformer"
    assert global_cfg.dynamics_width == 512


@pytest.mark.unit
def test_policy_agent_inits_rtp_bridge_by_default() -> None:
    model = _mock_model(16)
    # PolicyAgent.__post_init__ calls model.eval() and may touch parameters().
    agent = PolicyAgent(
        model=model,  # type: ignore[arg-type]
        deck=[1] * 60,
        use_mcts=False,
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent.use_recursive_turn_planner is True
    assert agent._rtp_bridge is not None
    assert isinstance(agent._rtp_bridge, RTPAgentBridge)
    assert agent._rtp_bridge.config.d_model == 16
    agent._rtp_bridge.active_turn_key = (0, 3)
    agent.reset_game()
    assert agent._rtp_bridge.active_turn_key == (-1, -1)
    assert agent._rtp_bridge.memory is None


@pytest.mark.unit
def test_policy_agent_can_disable_rtp() -> None:
    agent = PolicyAgent(
        model=_mock_model(),  # type: ignore[arg-type]
        deck=[1] * 60,
        use_mcts=False,
        use_recursive_turn_planner=False,
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent._rtp_bridge is None


@pytest.mark.unit
def test_policy_agent_without_model_skips_bridge() -> None:
    agent = PolicyAgent(
        model=None,
        deck=[1] * 60,
        use_mcts=False,
        matchup_adapter_shadow=False,
    )
    assert agent._rtp_bridge is None
