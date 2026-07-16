"""Phase 0: AWR pure-RL loss contract."""

from __future__ import annotations

import pytest
import torch

from poke_bot import config, features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.model import build_model
from poke_bot.train import TrainConfig, batch_losses


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    sv = features.SparseVector()
    for i in range(words):
        sv.word_start()
        sv.add((offset + i) % 32, 1.0)
    return sv


def _decision(index: int) -> DecisionSample:
    combos = [[0], [1]]
    return DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, index),
        options=_sparse(2, index + 3),
        action=[index % 2],
        action_combo_index=index % 2,
        action_combos=combos,
        env_step=index,
        action_token=_sparse(1, index + 7),
        policy_stages=[
            PolicyStage(
                options=_sparse(2, index + 3),
                action_combos=combos,
                target_index=index % 2,
            )
        ],
    )


def _small_model():
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
        dropout=0.0,
    )
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    model.eval()
    return model


def test_pure_rl_defaults_zero_aux_weights() -> None:
    cfg = TrainConfig.pure_rl_defaults()
    assert cfg.pure_rl is True
    assert cfg.aux_loss_weight == 0.0
    assert cfg.opp_hand_loss_weight == 0.0
    assert cfg.lethal_threat_loss_weight == 0.0
    assert cfg.prize_race_loss_weight == 0.0
    assert cfg.awr_normalize_advantages is True


def test_pure_rl_model_under_param_budget() -> None:
    from poke_bot.pure_rl.model_profile import (
        build_pure_rl_model,
        count_params,
        PURE_RL_PARAM_FAIL_MAX,
        PURE_RL_PARAM_TARGET_MAX,
    )

    model = build_pure_rl_model(device=torch.device("cpu"), validate=True)
    n = count_params(model)
    assert n <= PURE_RL_PARAM_FAIL_MAX
    assert n <= PURE_RL_PARAM_TARGET_MAX
    assert n <= 2_000_000  # Abhyuday-class prefer <2M
    assert n >= 1_000_000


def test_awr_runs_without_soft_targets() -> None:
    torch.manual_seed(0)
    model = _small_model()
    seq = GameSequence(
        episode_id="e",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0), _decision(1)],
    )
    loss, metrics = batch_losses(
        model,
        [seq],
        pure_rl=True,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics.n_decisions > 0
    assert metrics.awr_weight_mean > 0.0
    assert metrics.policy_selected_nll >= 0.0


def test_pure_rl_hard_fails_on_soft_behavior_targets() -> None:
    torch.manual_seed(0)
    model = _small_model()
    seq = GameSequence(
        episode_id="e",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=-1.0,
        decisions=[_decision(0)],
        factorized_policy_targets=[
            [
                {
                    "selected_index": 0,
                    "policy": [0.9, 0.1],
                    "action_combos": [[0], [1]],
                }
            ]
        ],
    )
    with pytest.raises(ValueError, match="PURE_RL"):
        batch_losses(
            model,
            [seq],
            pure_rl=True,
            aux_weight=0.0,
            opp_hand_weight=0.0,
            opp_remainder_weight=0.0,
        )
