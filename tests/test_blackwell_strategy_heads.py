"""Scope B Blackwell Hammer strategy heads — shapes, labels, masking, gating."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint, config, features
from poke_bot.blackwell_heads import (
    BLACKWELL_STRATEGY_HEAD_PREFIXES,
    attach_blackwell_strategy_labels,
    blackwell_strategy_heads_enabled,
    is_allowed_missing_blackwell_head_key,
    lethal_threat_label_from_trajectory,
    masked_bce_logit,
    masked_smooth_l1,
    prize_counts_from_obs,
    root_value_bias_from_lethal,
)
from poke_bot.features import SparseVector
from poke_bot.model import build_model
from poke_bot.train import (
    BELIEF_AUX_HEAD_KEY_PREFIXES,
    batch_losses,
    is_allowed_missing_belief_head_key,
    load_model_from_checkpoint,
)


def _tiny_cfg() -> config.ModelConfig:
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
        dropout=0.0,
    )


def _empty_board() -> SparseVector:
    sv = SparseVector()
    for _ in range(features.NUM_BOARD_TOKENS):
        sv.word_start()
    return sv


def _empty_options(n: int = 2) -> SparseVector:
    sv = SparseVector()
    for i in range(n):
        sv.word_start()
        sv.add((i * 7 + 3) % 32, 1.0)
    return sv


def _obs_with_prizes(own: int, opp: int, *, your_index: int = 0) -> dict:
    return {
        "current": {
            "yourIndex": your_index,
            "players": [
                {"prize": [None] * own, "hand": [], "bench": [], "active": []},
                {"prize": [None] * opp, "hand": None, "bench": [], "active": []},
            ],
        }
    }


def test_scope_b_prefixes_allowlisted_for_warmstart() -> None:
    for prefix in BLACKWELL_STRATEGY_HEAD_PREFIXES:
        assert prefix in BELIEF_AUX_HEAD_KEY_PREFIXES
        assert is_allowed_missing_blackwell_head_key(prefix + "weight")
        assert is_allowed_missing_belief_head_key(prefix + "bias")


def test_strategy_head_shapes_on_state_vec() -> None:
    model = build_model(
        _tiny_cfg(),
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )
    assert model.lethal_threat_head.out_features == 1
    assert model.prize_race_head.out_features == 2
    assert "lethal_threat_head" in model.aux_heads_present
    assert "prize_race_head" in model.aux_heads_present
    out = model.forward(_empty_board(), _empty_options(2), n_options=[2])
    assert tuple(out["lethal_threat_logits"].shape) == (1,)
    assert tuple(out["prize_race_pred"].shape) == (1, 2)


def test_label_attachment_prize_race_and_lethal() -> None:
    steps = [
        {"observation": _obs_with_prizes(6, 6), "aux_labels": {}},
        {"observation": _obs_with_prizes(6, 5), "aux_labels": {}},
        {"observation": _obs_with_prizes(5, 5), "aux_labels": {}},  # took prize
        {"observation": _obs_with_prizes(5, 4), "aux_labels": {}},
    ]
    attach_blackwell_strategy_labels(steps, horizon=2)
    assert steps[0]["aux_labels"]["prize_race"] == pytest.approx([1.0, 1.0])
    assert steps[0]["aux_labels"]["lethal_threat"] == 1.0  # within horizon
    assert steps[2]["aux_labels"]["prize_race"] == pytest.approx([5 / 6, 5 / 6])
    # From index 2 with horizon 2, no further own decrease → 0
    assert steps[2]["aux_labels"]["lethal_threat"] == 0.0
    assert lethal_threat_label_from_trajectory([6, 6, 5], 0, horizon=1) == 0.0
    assert lethal_threat_label_from_trajectory([6, 6, 5], 0, horizon=2) == 1.0
    assert prize_counts_from_obs(_obs_with_prizes(3, 4)) == (3, 4)


def test_masked_losses_when_labels_absent() -> None:
    logits = torch.randn(3, requires_grad=True)
    pred = torch.randn(3, 2, requires_grad=True)
    zero_l = masked_bce_logit(logits, None)
    zero_r = masked_smooth_l1(pred, None)
    assert float(zero_l.detach()) == 0.0
    assert float(zero_r.detach()) == 0.0
    (zero_l + zero_r + logits.sum() * 0 + pred.sum() * 0).backward()


def test_batch_losses_masks_scope_b_without_labels() -> None:
    from poke_bot.dataset import DecisionSample, GameSequence

    model = build_model(
        _tiny_cfg(),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=16,
    )
    model.train()
    decision = DecisionSample(
        board=_empty_board(),
        options=_empty_options(2),
        action=[0],
        action_combo_index=0,
        action_combos=[[0], [1]],
        env_step=0,
        action_token=None,
        aux_labels={},  # no Scope B labels
    )
    game = GameSequence(
        episode_id="scope-b-mask",
        seat=0,
        archetype="hammer-pult",
        opp_archetype="unknown",
        deck=[1, 2, 3],
        value=0.0,
        decisions=[decision],
    )
    total, metrics = batch_losses(
        model,
        [game],
        lethal_threat_weight=0.15,
        prize_race_weight=0.10,
    )
    assert torch.isfinite(total)
    assert metrics.lethal_threat_loss == 0.0
    assert metrics.prize_race_loss == 0.0
    total.backward()


def test_batch_losses_scope_b_with_labels() -> None:
    from poke_bot.dataset import DecisionSample, GameSequence

    model = build_model(
        _tiny_cfg(),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=16,
    )
    model.train()
    decision = DecisionSample(
        board=_empty_board(),
        options=_empty_options(2),
        action=[0],
        action_combo_index=0,
        action_combos=[[0], [1]],
        env_step=0,
        action_token=None,
        aux_labels={"lethal_threat": 1.0, "prize_race": [0.5, 0.75]},
    )
    game = GameSequence(
        episode_id="scope-b-labels",
        seat=0,
        archetype="hammer-pult",
        opp_archetype="unknown",
        deck=[1, 2, 3],
        value=1.0,
        decisions=[decision],
    )
    total, metrics = batch_losses(
        model,
        [game],
        lethal_threat_weight=0.15,
        prize_race_weight=0.10,
    )
    assert torch.isfinite(total)
    assert metrics.lethal_threat_loss > 0.0
    assert metrics.prize_race_loss > 0.0
    total.backward()


def test_gating_requires_blackwell_profile_or_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POKEBOT_BLACKWELL_STRATEGY_HEADS", raising=False)
    monkeypatch.setenv("POKEBOT_PRIMARY_ARCHETYPE", "hammer-pult")
    monkeypatch.setenv("POKEBOT_GPU_PROFILE", "3080ti")
    assert blackwell_strategy_heads_enabled() is False

    monkeypatch.setenv("POKEBOT_GPU_PROFILE", "blackwell")
    assert blackwell_strategy_heads_enabled() is True

    monkeypatch.setenv("POKEBOT_BLACKWELL_STRATEGY_HEADS", "0")
    assert blackwell_strategy_heads_enabled() is False

    monkeypatch.setenv("POKEBOT_BLACKWELL_STRATEGY_HEADS", "1")
    monkeypatch.setenv("POKEBOT_GPU_PROFILE", "3080ti")
    assert blackwell_strategy_heads_enabled() is True


def test_root_value_bias_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POKEBOT_BLACKWELL_STRATEGY_HEADS", "0")
    assert root_value_bias_from_lethal(torch.tensor(2.0)) == 0.0
    monkeypatch.setenv("POKEBOT_BLACKWELL_STRATEGY_HEADS", "1")
    bias = root_value_bias_from_lethal(torch.tensor(2.0), scale=0.05)
    assert bias > 0.0


def test_warmstart_allows_missing_scope_b_heads(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    legacy = build_model(
        cfg,
        aux_archetype_classes=4,
        encoder_vocab=37,
        decoder_vocab=41,
        belief_card_vocab=64,
    )
    payload = checkpoint.build_checkpoint(model=legacy, model_config=cfg)
    state = dict(payload["model_state_dict"])
    for key in list(state):
        if is_allowed_missing_belief_head_key(key):
            del state[key]
    payload["model_state_dict"] = state
    path = checkpoint.atomic_torch_save(payload, tmp_path / "legacy_no_scope_b.pt")
    loaded = load_model_from_checkpoint(path, device=torch.device("cpu"))
    warm = set(loaded.warm_started_belief_heads)
    assert "lethal_threat_head" in warm
    assert "prize_race_head" in warm
    out = loaded.forward(_empty_board(), _empty_options(2), n_options=[2])
    assert out["lethal_threat_logits"].numel() == 1
    assert tuple(out["prize_race_pred"].shape) == (1, 2)
