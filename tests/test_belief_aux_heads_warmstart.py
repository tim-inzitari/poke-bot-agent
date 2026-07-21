"""Assert belief aux heads can be late-added via warm-start (shapes + load).

Full particle reweight / live RR aux training stays gated on Phase A GO.
This file only proves the infrastructural contract from the aux-heads plan:
distinct named heads, allowlisted missing keys, uniform fallback, masked loss.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from poke_bot import checkpoint, config, features
from poke_bot.features import SparseVector
from poke_bot.model import (
    build_model,
    card_prior_logits_or_uniform,
)
from poke_bot.train import (
    BELIEF_AUX_HEAD_KEY_PREFIXES,
    batch_losses,
    is_allowed_missing_belief_head_key,
    load_model_from_checkpoint,
    masked_belief_card_bce,
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
    # One empty bag-word per board token slot (24).
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


def test_belief_head_key_allowlist_is_distinct_named() -> None:
    assert "opp_hand_head." in BELIEF_AUX_HEAD_KEY_PREFIXES
    assert "opp_remainder_head." in BELIEF_AUX_HEAD_KEY_PREFIXES
    assert "lethal_threat_head." in BELIEF_AUX_HEAD_KEY_PREFIXES
    assert "prize_race_head." in BELIEF_AUX_HEAD_KEY_PREFIXES
    assert not any(p.startswith("aux_head") for p in BELIEF_AUX_HEAD_KEY_PREFIXES)
    assert is_allowed_missing_belief_head_key("opp_hand_head.weight")
    assert is_allowed_missing_belief_head_key("opp_remainder_head.bias")
    assert is_allowed_missing_belief_head_key("lethal_threat_head.weight")
    assert is_allowed_missing_belief_head_key("prize_race_head.bias")
    assert not is_allowed_missing_belief_head_key("aux_head.3.weight")
    assert not is_allowed_missing_belief_head_key("board_bag.weight")


def test_model_constructs_named_belief_heads_with_card_vocab_shapes() -> None:
    cfg = _tiny_cfg()
    card_vocab = 1268
    model = build_model(
        cfg,
        aux_archetype_classes=5,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=card_vocab,
    )
    assert model.opp_hand_head.in_features == cfg.d_model
    assert model.opp_hand_head.out_features == card_vocab
    assert model.opp_remainder_head.out_features == card_vocab
    assert model.aux_head[-1].out_features == 5
    assert model.aux_heads_present == (
        "aux_head",
        "opp_hand_head",
        "opp_remainder_head",
        "lethal_threat_head",
        "prize_race_head",
    )

    board = _empty_board()
    options = _empty_options(3)
    out = model.forward(board, options, n_options=[3])
    assert out["opp_hand_logits"].shape[-1] == card_vocab
    assert out["opp_remainder_logits"].shape[-1] == card_vocab
    assert out["aux_logits"].shape[-1] == 5
    assert out["lethal_threat_logits"].shape[-1] == 1 or out["lethal_threat_logits"].ndim == 1
    assert out["prize_race_pred"].shape[-1] == 2


def test_warmstart_loads_old_ckpt_missing_only_new_belief_heads(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    # Champion-style ckpt without belief card heads (strip after save).
    # Missing heads → rebuild uses live card vocab (config/features).
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
    path = checkpoint.atomic_torch_save(payload, tmp_path / "legacy_no_belief_heads.pt")

    loaded = load_model_from_checkpoint(path, device=torch.device("cpu"))
    assert loaded.warm_started_belief_heads == (
        "lethal_threat_head",
        "opp_hand_head",
        "opp_remainder_head",
        "prize_race_head",
    )
    expect_vocab = features.card_vocab_size()
    assert loaded.opp_hand_head.out_features == expect_vocab
    assert loaded.opp_remainder_head.out_features == expect_vocab

    board = _empty_board()
    options = _empty_options(2)
    out = loaded.forward(board, options, n_options=[2])
    assert tuple(out["opp_hand_logits"].shape) == (1, expect_vocab)
    assert tuple(out["opp_remainder_logits"].shape) == (1, expect_vocab)
    assert out["lethal_threat_logits"].numel() == 1
    assert tuple(out["prize_race_pred"].shape) == (1, 2)

    # Trunk tensors match legacy; new heads are present but freshly init.
    legacy_state = legacy.state_dict()
    loaded_state = loaded.state_dict()
    for key, tensor in legacy_state.items():
        if is_allowed_missing_belief_head_key(key):
            continue
        assert torch.equal(tensor.cpu(), loaded_state[key].cpu()), key


def test_warmstart_rejects_missing_trunk_keys(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    model = build_model(
        cfg,
        aux_archetype_classes=3,
        encoder_vocab=32,
        decoder_vocab=32,
        belief_card_vocab=16,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    state = dict(payload["model_state_dict"])
    # Corrupt a trunk tensor (keep architecture shape keys present).
    state["value_head.0.weight"] = state["value_head.0.weight"][:1]
    for key in list(state):
        if is_allowed_missing_belief_head_key(key):
            del state[key]
    payload["model_state_dict"] = state
    path = checkpoint.atomic_torch_save(payload, tmp_path / "broken_trunk.pt")
    with pytest.raises(RuntimeError):
        load_model_from_checkpoint(path, device=torch.device("cpu"))


def test_warmstart_rejects_disallowed_missing_keys(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    model = build_model(
        cfg,
        aux_archetype_classes=3,
        encoder_vocab=32,
        decoder_vocab=32,
        belief_card_vocab=16,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    state = dict(payload["model_state_dict"])
    del state["value_head.0.weight"]
    for key in list(state):
        if is_allowed_missing_belief_head_key(key):
            del state[key]
    payload["model_state_dict"] = state
    path = checkpoint.atomic_torch_save(payload, tmp_path / "missing_value_head.pt")
    with pytest.raises(RuntimeError, match="non-belief-head"):
        load_model_from_checkpoint(path, device=torch.device("cpu"))


def test_uniform_fallback_and_masked_bce() -> None:
    vocab = 8
    uniform = card_prior_logits_or_uniform(None, vocab)
    assert tuple(uniform.shape) == (vocab,)
    assert torch.allclose(F.softmax(uniform, dim=-1), torch.full((vocab,), 1.0 / vocab))

    logits = torch.randn(2, vocab, requires_grad=True)
    zero = masked_belief_card_bce(logits, None)
    assert float(zero.detach()) == 0.0
    (zero + logits.sum() * 0).backward()  # graph stays usable

    labels = torch.zeros(2, vocab)
    labels[0, 1] = 1.0
    loss = masked_belief_card_bce(logits, labels)
    assert loss.ndim == 0
    assert float(loss.detach()) > 0.0


def test_batch_losses_accepts_masked_belief_heads_without_labels() -> None:
    """Training can attach belief head losses with absent labels (masked zero)."""
    from poke_bot.dataset import DecisionSample, GameSequence

    cfg = _tiny_cfg()
    model = build_model(
        cfg,
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )
    model.train()
    board = _empty_board()
    options = _empty_options(2)
    decision = DecisionSample(
        board=board,
        options=options,
        action=[0],
        action_combo_index=0,
        action_combos=[[0], [1]],
        env_step=0,
        action_token=None,
    )
    game = GameSequence(
        episode_id="warmstart-assert",
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
        value_weight=1.0,
        aux_weight=0.0,
        opp_hand_weight=0.2,
        opp_remainder_weight=0.15,
        opp_hand_multihot=None,
        opp_remainder_multihot=None,
    )
    assert torch.isfinite(total)
    assert metrics.opp_hand_loss == 0.0
    assert metrics.opp_remainder_loss == 0.0
    total.backward()


def test_real_legacy_champion_fails_closed_on_feature_schema() -> None:
    path = Path(
        "outputs/checkpoints/trusted_factorized_20260713T213354Z.evaluated.0aaa86457cf9ce3d.pt"
    )
    if not path.is_file():
        pytest.skip(f"missing champion ckpt {path}")
    # Belief-head warm-start remains covered by the synthetic schema-current
    # checkpoint above. This on-disk champion predates schema v5's composite
    # option binding and slot identity, so loading it would silently restore
    # the exact representation collision the new schema is meant to remove.
    with pytest.raises(
        RuntimeError, match="predates explicit slot/option-binding features"
    ):
        load_model_from_checkpoint(path, device=torch.device("cpu"))
