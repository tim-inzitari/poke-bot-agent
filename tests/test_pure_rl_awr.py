"""Phase 0: AWR pure-RL loss contract."""

from __future__ import annotations

import math

import pytest
import torch

from poke_bot import checkpoint, config, features
from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, PolicyStage
from poke_bot.model import build_model
from poke_bot.train import (
    TrainConfig,
    _precompute_awr_baseline_cache,
    _precompute_awr_baseline_cache_reference,
    _precompute_awr_baseline_cache_value_only,
    _policy_argmax_predictions,
    batch_losses,
    prepare_host_sparse_batch,
    rl_train_step,
)


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


def test_offline_h3_metrics_match_live_awr_advantage_path() -> None:
    """The one-forward gate must reproduce the trainer's FP32 AWR math."""

    from scripts.evaluate_alakazam_prize_plan_v2_awr_gate import (
        _metric,
        _offline_h3_metric,
    )
    from poke_bot.train import evaluate

    torch.manual_seed(41)
    model = _small_model()
    sequences = [
        GameSequence(
            episode_id=f"offline-h3-{seat}",
            seat=seat,
            archetype="alakazam",
            opp_archetype="mirror",
            deck=[1] * 60,
            value=value,
            decisions=[_decision(seat * 2), _decision(seat * 2 + 1)],
        )
        for seat, value in ((0, 1.0), (1, -1.0))
    ]
    cfg = TrainConfig.pure_rl_defaults(
        games_per_batch=1,
        max_decisions_per_batch=8,
        capture_awr_weight_distribution=True,
    )
    baseline: dict[tuple[int, int, int], float] = {}
    evaluate(
        model,
        sequences,
        cfg=cfg,
        awr_capture_baseline=baseline,
        allow_masked_own_deck_ledger_rows=True,
    )
    additive = {
        key: (-0.025 if index % 2 else 0.0125)
        for index, key in enumerate(sorted(baseline))
    }
    live = _metric(
        evaluate(
            model,
            sequences,
            cfg=cfg,
            awr_advantage_cache=additive,
            allow_masked_own_deck_ledger_rows=True,
        )
    )
    offline = _offline_h3_metric(
        sequences,
        cfg=cfg,
        baseline=baseline,
        additive=additive,
    )
    assert offline.keys() == live.keys()
    for name in live:
        if name == "decisions":
            assert offline[name] == live[name]
        else:
            assert offline[name] == pytest.approx(
                live[name], rel=1e-6, abs=1e-5
            ), name


def test_policy_only_agreement_matches_argmax_and_skips_value_head(
    monkeypatch,
) -> None:
    torch.manual_seed(29)
    model = _small_model()
    seq = GameSequence(
        episode_id="agreement-fast-path",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0), _decision(1)],
    )
    reference: list[int] = []
    batch_losses(model, [seq], prediction_sink=reference)

    def fail_value_head(*_args, **_kwargs):
        raise AssertionError("policy-only agreement evaluated the value head")

    monkeypatch.setattr(model.value_head, "forward", fail_value_head)
    optimized: list[int] = []
    _loss, metrics = batch_losses(
        model,
        [seq],
        prediction_sink=optimized,
        prediction_only=True,
    )
    assert optimized == reference
    assert metrics.n_decisions == len(reference)


def test_policy_agreement_uses_its_proven_inference_decision_cap(
    monkeypatch,
) -> None:
    model = _small_model()
    seq = GameSequence(
        episode_id="agreement-cap",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0)],
    )
    cfg = TrainConfig.pure_rl_defaults(
        games_per_batch=7,
        max_decisions_per_batch=3072,
        agreement_max_decisions_per_batch=6144,
    )
    captured: dict[str, int] = {}

    def capture_batches(
        sequences, games_per_batch, max_decisions, shuffle, seed, epoch
    ):
        captured["games_per_batch"] = int(games_per_batch)
        captured["max_decisions"] = int(max_decisions)
        return []

    monkeypatch.setattr("poke_bot.train._iter_game_batches", capture_batches)
    assert _policy_argmax_predictions(model, [seq], cfg=cfg) == []
    assert captured == {"games_per_batch": 7, "max_decisions": 6144}


def test_policy_agreement_packs_independent_histories_with_exact_argmax_parity(
    monkeypatch,
) -> None:
    torch.manual_seed(31)
    model = _small_model()
    sequences = [
        GameSequence(
            episode_id="agreement-packed-a",
            seat=0,
            archetype="dragapult",
            opp_archetype="iono",
            deck=[1] * 60,
            value=1.0,
            decisions=[_decision(0), _decision(1), _decision(2)],
        ),
        GameSequence(
            episode_id="agreement-packed-b",
            seat=1,
            archetype="dragapult",
            opp_archetype="iono",
            deck=[1] * 60,
            value=-1.0,
            decisions=[_decision(3)],
        ),
    ]
    serial_cfg = TrainConfig.pure_rl_defaults(
        games_per_batch=8,
        max_decisions_per_batch=64,
        agreement_max_decisions_per_batch=64,
        agreement_pack_temporal_games=False,
    )
    packed_cfg = TrainConfig.pure_rl_defaults(
        games_per_batch=8,
        max_decisions_per_batch=64,
        agreement_max_decisions_per_batch=64,
        agreement_pack_temporal_games=True,
    )

    original_temporal_encode = model.temporal_encode
    calls = {"serial": 0, "packed": 0}
    mode = "serial"

    def count_temporal_calls(*args, **kwargs):
        calls[mode] += 1
        return original_temporal_encode(*args, **kwargs)

    monkeypatch.setattr(model, "temporal_encode", count_temporal_calls)
    serial = _policy_argmax_predictions(model, sequences, cfg=serial_cfg)
    mode = "packed"
    packed = _policy_argmax_predictions(model, sequences, cfg=packed_cfg)

    assert packed == serial
    assert calls == {"serial": 2, "packed": 1}


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


def test_optimizer_sparse_pack_preserves_loss_predictions_and_gradients() -> None:
    torch.manual_seed(97)
    reference_model = _small_model()
    candidate_model = _small_model()
    candidate_model.load_state_dict(reference_model.state_dict())
    reference_model.train()
    candidate_model.train()
    sequences = [
        GameSequence(
            episode_id=f"sparse-prefetch-{seat}",
            seat=seat,
            archetype="dragapult",
            opp_archetype="iono",
            deck=[1] * 60,
            value=(1.0 if seat == 0 else -1.0),
            decisions=[_decision(seat * 4), _decision(seat * 4 + 1)],
        )
        for seat in (0, 1)
    ]
    reference_predictions: list[int] = []
    reference_loss, reference_metrics = batch_losses(
        reference_model,
        sequences,
        pure_rl=True,
        prediction_sink=reference_predictions,
    )
    reference_loss.backward()
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference_model.named_parameters()
        if parameter.grad is not None
    }

    host = prepare_host_sparse_batch(
        sequences,
        board_words=reference_model.num_board_tokens,
        card_vocab=reference_model.belief_card_vocab,
        pin_memory=False,
    )
    prepared = host.to_device(torch.device("cpu"), non_blocking=False)
    candidate_predictions: list[int] = []
    candidate_loss, candidate_metrics = batch_losses(
        candidate_model,
        sequences,
        pure_rl=True,
        prediction_sink=candidate_predictions,
        prepared_sparse_batch=prepared,
    )
    candidate_loss.backward()

    assert candidate_predictions == reference_predictions
    assert candidate_metrics.__dict__ == reference_metrics.__dict__
    assert torch.equal(candidate_loss.detach(), reference_loss.detach())
    for name, parameter in candidate_model.named_parameters():
        if name in reference_gradients:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, reference_gradients[name]), name
        else:
            assert parameter.grad is None
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


def test_frozen_external_advantages_add_only_h3_to_raw_awr_credit() -> None:
    torch.manual_seed(31)
    model = _small_model()
    seq = GameSequence(
        episode_id="h3-provider",
        seat=0,
        archetype="alakazam",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0), _decision(1)],
    )
    _, legacy_metrics = batch_losses(
        model,
        [seq],
        pure_rl=True,
        awr_normalize_advantages=False,
    )
    # The frozen cache contributes only the H3 additive term; the policy's
    # actual in-batch z-V_existing baseline remains authoritative.
    supplied = {(id(seq), 0, 0): 0.25, (id(seq), 1, 0): -0.75}
    loss, metrics = batch_losses(
        model,
        [seq],
        pure_rl=True,
        awr_advantage_cache=supplied,
        awr_normalize_advantages=False,
    )
    assert torch.isfinite(loss)
    assert metrics.raw_advantage_mean == pytest.approx(
        legacy_metrics.raw_advantage_mean - 0.25
    )
    assert metrics.normalized_advantage_mean == pytest.approx(metrics.raw_advantage_mean)
    assert math.isfinite(metrics.normalized_advantage_std)
    assert metrics.normalized_advantage_std > 0.0


def test_frozen_external_advantages_fail_closed_on_missing_or_nonfinite_rows() -> None:
    model = _small_model()
    seq = GameSequence(
        episode_id="h3-provider-invalid",
        seat=0,
        archetype="alakazam",
        opp_archetype="iono",
        deck=[1] * 60,
        value=0.0,
        decisions=[_decision(0), _decision(1)],
    )
    with pytest.raises(KeyError, match="advantage cache is missing"):
        batch_losses(
            model,
            [seq],
            pure_rl=True,
            awr_advantage_cache={(id(seq), 0, 0): 0.0},
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        batch_losses(
            model,
            [seq],
            pure_rl=True,
            awr_advantage_cache={
                (id(seq), 0, 0): 0.0,
                (id(seq), 1, 0): float("nan"),
            },
        )


def test_value_only_packed_baseline_matches_reference_values_and_awr() -> None:
    torch.manual_seed(19)
    model = _small_model()
    sequences = [
        GameSequence(
            episode_id=f"packed-{length}",
            seat=length % 2,
            archetype="dragapult",
            opp_archetype="iono",
            deck=[1] * 60,
            value=(-1.0 if length == 2 else 1.0),
            decisions=[_decision(index + length * 5) for index in range(length)],
        )
        for length in (2, 3, 4, 5)
    ]
    cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        games_per_batch=3,
        max_decisions_per_batch=32,
        awr_baseline_prefetch_batches=1,
    )

    reference = _precompute_awr_baseline_cache_reference(
        model, sequences, cfg=cfg
    )
    optimized = _precompute_awr_baseline_cache_value_only(
        model, sequences, cfg=cfg
    )

    assert optimized.keys() == reference.keys()
    assert optimized == reference

    reference_loss, reference_metrics = batch_losses(
        model,
        sequences,
        pure_rl=True,
        awr_baseline_cache=reference,
        awr_beta=cfg.awr_beta,
        awr_weight_max=cfg.awr_weight_max,
        awr_normalize_advantages=cfg.awr_normalize_advantages,
    )
    optimized_loss, optimized_metrics = batch_losses(
        model,
        sequences,
        pure_rl=True,
        awr_baseline_cache=optimized,
        awr_beta=cfg.awr_beta,
        awr_weight_max=cfg.awr_weight_max,
        awr_normalize_advantages=cfg.awr_normalize_advantages,
    )
    assert torch.equal(optimized_loss, reference_loss)
    for field in (
        "awr_weight_mean",
        "awr_weight_sum",
        "awr_weight_sq_sum",
        "awr_weight_p50",
        "awr_weight_p95",
        "awr_weight_max_observed",
        "awr_weight_clip_frac",
        "awr_effective_sample_size",
        "awr_effective_sample_fraction",
    ):
        assert getattr(optimized_metrics, field) == getattr(
            reference_metrics, field
        )


def test_value_only_baseline_bypasses_policy_decoder(monkeypatch) -> None:
    torch.manual_seed(23)
    model = _small_model()
    sequence = GameSequence(
        episode_id="value-only",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0), _decision(1)],
    )
    cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        games_per_batch=2,
        max_decisions_per_batch=8,
    )

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("policy decoder ran during value-only baseline")

    monkeypatch.setattr(model, "decode_options", _forbidden)
    cache = _precompute_awr_baseline_cache_value_only(
        model, [sequence], cfg=cfg
    )
    assert len(cache) == 2


def test_packed_value_only_candidate_is_numerically_close_to_reference() -> None:
    torch.manual_seed(29)
    model = _small_model()
    sequences = [
        GameSequence(
            episode_id=f"packed-shadow-{index}",
            seat=index % 2,
            archetype="dragapult",
            opp_archetype="iono",
            deck=[1] * 60,
            value=(-1.0 if index == 0 else 1.0),
            decisions=[_decision(index * 9 + step) for step in range(length)],
        )
        for index, length in enumerate((3, 4))
    ]
    reference_cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        games_per_batch=2,
        max_decisions_per_batch=16,
    )
    packed_cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        games_per_batch=2,
        max_decisions_per_batch=16,
        awr_pack_temporal_baseline=True,
    )
    reference = _precompute_awr_baseline_cache_reference(
        model, sequences, cfg=reference_cfg
    )
    packed = _precompute_awr_baseline_cache_value_only(
        model, sequences, cfg=packed_cfg
    )
    assert packed.keys() == reference.keys()
    for key in reference:
        assert packed[key] == pytest.approx(reference[key], abs=1e-6, rel=1e-6)

    _, reference_metrics = batch_losses(
        model, sequences, pure_rl=True, awr_baseline_cache=reference
    )
    _, packed_metrics = batch_losses(
        model, sequences, pure_rl=True, awr_baseline_cache=packed
    )
    for field in (
        "awr_weight_mean",
        "awr_weight_sum",
        "awr_weight_sq_sum",
        "awr_weight_p50",
        "awr_weight_p95",
        "awr_weight_max_observed",
        "awr_effective_sample_size",
    ):
        assert getattr(packed_metrics, field) == pytest.approx(
            getattr(reference_metrics, field), abs=1e-5, rel=1e-5
        )


def test_value_only_baseline_prefetch_is_bounded() -> None:
    model = _small_model()
    sequence = GameSequence(
        episode_id="prefetch-bound",
        seat=0,
        archetype="dragapult",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0)],
    )
    cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        awr_baseline_prefetch_batches=2,
    )
    with pytest.raises(ValueError, match="bounded to zero or one"):
        _precompute_awr_baseline_cache_value_only(
            model, [sequence], cfg=cfg
        )


def test_value_only_activation_gate_requires_exact_cache_parity() -> None:
    torch.manual_seed(31)
    model = _small_model()
    sequences = [
        GameSequence(
            episode_id=f"exact-gate-{length}",
            seat=length % 2,
            archetype="dragapult",
            opp_archetype="iono",
            deck=[1] * 60,
            value=1.0,
            decisions=[_decision(length * 11 + step) for step in range(length)],
        )
        for length in (2, 3, 4)
    ]
    exact_cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        games_per_batch=3,
        max_decisions_per_batch=16,
        awr_baseline_exact_parity_check=True,
    )
    cache = _precompute_awr_baseline_cache(
        model, sequences, cfg=exact_cfg
    )
    assert len(cache) == sum(len(sequence.decisions) for sequence in sequences)

    packed_cfg = TrainConfig.pure_rl_defaults(
        amp=False,
        games_per_batch=3,
        max_decisions_per_batch=16,
        awr_pack_temporal_baseline=True,
        awr_baseline_exact_parity_check=True,
    )
    with pytest.raises(RuntimeError, match="exact parity failed"):
        _precompute_awr_baseline_cache(
            model, sequences, cfg=packed_cfg
        )


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
