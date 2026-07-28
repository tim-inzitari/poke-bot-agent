"""Phase 0: AWR pure-RL loss contract."""

from __future__ import annotations

import pytest
import torch

from poke_bot import checkpoint, config, features
from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, PolicyStage
from poke_bot.model import build_model
from poke_bot.train import TrainConfig, batch_losses, rl_train_step


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


def _small_stateless_model():
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="stateless",
        kv_cache=False,
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
    assert cfg.alakazam_guide_loss_weight == 0.0
    assert cfg.lethal_threat_loss_weight == 0.0
    assert cfg.prize_race_loss_weight == 0.0
    assert cfg.awr_normalize_advantages is True
    assert cfg.awr_freeze_baseline is True
    assert cfg.epochs == 2


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
    assert metrics.raw_advantage_mean_abs > 0.0
    assert metrics.mean_advantage == pytest.approx(metrics.raw_advantage_mean_abs)
    assert metrics.raw_advantage_std >= 0.0
    assert metrics.normalized_advantage_mean == pytest.approx(0.0, abs=1e-5)
    assert metrics.normalized_advantage_std == pytest.approx(1.0, abs=1e-4)
    assert 0.0 < metrics.awr_effective_sample_size <= metrics.n_decisions
    assert 0.0 < metrics.awr_effective_sample_fraction <= 1.0


def test_frozen_awr_baseline_survives_value_head_update() -> None:
    torch.manual_seed(0)
    model = _small_model()
    seq = GameSequence(
        episode_id="frozen",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0), _decision(1)],
    )
    cache: dict[tuple[int, int, int], float] = {}
    _, before = batch_losses(
        model,
        [seq],
        pure_rl=True,
        awr_capture_baseline=cache,
    )
    assert len(cache) == before.n_decisions

    with torch.no_grad():
        model.value_head[-1].bias.add_(0.75)
    _, frozen = batch_losses(
        model,
        [seq],
        pure_rl=True,
        awr_baseline_cache=cache,
    )
    _, online = batch_losses(model, [seq], pure_rl=True)

    assert frozen.raw_advantage_mean == pytest.approx(before.raw_advantage_mean)
    assert frozen.raw_advantage_mean_abs == pytest.approx(
        before.raw_advantage_mean_abs
    )
    assert online.raw_advantage_mean != pytest.approx(before.raw_advantage_mean)


def test_rl_train_step_restores_adam_and_global_counters(tmp_path) -> None:
    torch.manual_seed(0)
    model = _small_model()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-4,
        weight_decay=1e-4,
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()

    base = tmp_path / "base.pt"
    checkpoint.atomic_torch_save(
        checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            step=3,
            epoch=2,
            rl_iteration=4,
            model_config=model.cfg,
        ),
        base,
    )
    seq = GameSequence(
        episode_id="resume",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0), _decision(1)],
    )
    output = tmp_path / "candidate.pt"
    cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        lr=0.0,
        val_frac=0.0,
        early_stop_patience=0,
        games_per_batch=4,
        max_decisions_per_batch=32,
    )
    result = rl_train_step(
        BootstrapDataset(sequences=[seq]),
        base_ckpt=base,
        out_run_name="resume-test",
        archetype_id="core",
        epochs=1,
        device=torch.device("cpu"),
        cfg=cfg,
        output_path=output,
    )

    assert result["parent_step"] == 3
    assert result["step"] == 4
    assert result["optimizer_state_restored"] is True
    assert result["rl_iteration"] == 5
    assert result["awr_baseline_mode"] == "frozen_precomputed"
    assert result["policy_prev_agreement"] == pytest.approx(1.0)
    assert result["metrics"]["policy_prev_agreement"] == pytest.approx(1.0)

    saved = checkpoint.load_checkpoint(output)
    assert saved["step"] == 4
    assert saved["epoch"] == 3
    assert saved["rl_iteration"] == 5
    assert saved["extra"]["optimizer_state_restored"] is True
    assert saved["extra"]["policy_prev_agreement"] == pytest.approx(1.0)
    optimizer_steps = [
        int(state["step"].item())
        for state in saved["optimizer_state_dict"]["state"].values()
        if "step" in state
    ]
    assert max(optimizer_steps) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rl_train_step_exact_device_resident_integration(tmp_path) -> None:
    torch.manual_seed(4)
    model = _small_stateless_model()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-4,
    )
    base = tmp_path / "resident-base.pt"
    checkpoint.atomic_torch_save(
        checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            step=0,
            epoch=0,
            rl_iteration=0,
            model_config=model.cfg,
        ),
        base,
    )
    seq = GameSequence(
        episode_id="resident",
        seat=0,
        archetype="alakazam",
        opp_archetype="unknown",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0), _decision(1)],
    )
    output = tmp_path / "resident-candidate.pt"
    cfg = TrainConfig.pure_rl_defaults(
        amp=True,
        val_frac=0.0,
        early_stop_patience=0,
        games_per_batch=4,
        max_decisions_per_batch=32,
    )
    result = rl_train_step(
        BootstrapDataset(sequences=[seq]),
        base_ckpt=base,
        out_run_name="resident-integration",
        archetype_id="alakazam",
        epochs=1,
        device=torch.device("cuda:0"),
        cfg=cfg,
        output_path=output,
        device_resident=True,
        device_resident_min_free_gib=2.0,
    )

    assert result["device_resident_rl"] is True
    assert result["device_resident_bytes"] > 0
    assert result["device_resident_batch_size"] == 2
    assert result["device_resident_build_seconds"] > 0.0
    assert result["device_resident_samples"] == 2
    assert result["metrics"]["optimizer_samples_per_second"] > 0.0
    assert result["awr_baseline_mode"] == "frozen_device_resident"
    assert result["policy_prev_agreement_rows"] == 2


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
