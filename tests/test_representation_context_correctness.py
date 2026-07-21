from __future__ import annotations

from enum import IntEnum
from types import SimpleNamespace

import pytest
import torch

from poke_bot import card2vec, cg_env, config, features
from poke_bot.model import (
    TemporalKVCache,
    build_model,
    pack_sparse_vectors,
)


class _AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class _OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class _SelectContext(IntEnum):
    MAIN = 0
    RECOVER_SPECIAL_CONDITION = 48


def _install_fake_vocab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, "_CARD_COUNT", 8)
    monkeypatch.setattr(features, "_ATTACK_COUNT", 8)
    monkeypatch.setattr(features, "_CARD_TABLE", {})
    # setitem avoids invoking cg_env.__getattr__ on hosts without the native
    # competition runtime.
    monkeypatch.setitem(cg_env.__dict__, "AreaType", _AreaType)
    monkeypatch.setitem(cg_env.__dict__, "OptionType", _OptionType)
    monkeypatch.setitem(cg_env.__dict__, "SelectContext", _SelectContext)


def _card(card_id: int = 3) -> SimpleNamespace:
    return SimpleNamespace(id=card_id, tools=[], energyCards=[])


def _option(option_type: _OptionType, **fields) -> SimpleNamespace:
    defaults = {
        "area": None,
        "index": None,
        "playerIndex": None,
        "inPlayArea": None,
        "inPlayIndex": None,
        "toolIndex": None,
        "energyIndex": None,
        "cardId": None,
    }
    defaults.update(fields)
    return SimpleNamespace(type=option_type, **defaults)


def _word_indices(sv: features.SparseVector, word: int) -> tuple[int, ...]:
    start = sv.offset[word]
    end = sv.offset[word + 1] if word + 1 < sv.num_words else len(sv.index)
    return tuple(sv.index[start:end])


def _word_signature(
    sv: features.SparseVector, word: int
) -> tuple[tuple[int, float], ...]:
    start = sv.offset[word]
    end = sv.offset[word + 1] if word + 1 < sv.num_words else len(sv.index)
    # EmbeddingBag sums duplicate indices, so coalesce them before comparing.
    coalesced: dict[int, float] = {}
    for index, value in zip(sv.index[start:end], sv.value[start:end]):
        coalesced[index] = coalesced.get(index, 0.0) + value
    return tuple(sorted(coalesced.items()))


def _small_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_context: int = 8,
    temporal_layers: int = 1,
    temporal_pos: str = "rope",
    dense_card2vec: bool = False,
    decoder_vocab: int = 64,
):
    _install_fake_vocab(monkeypatch)
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=temporal_layers,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=max_context,
        temporal_pos=temporal_pos,
        decision_context="history",
        kv_cache=True,
        card_embed_dim=8,
        dense_card2vec=dense_card2vec,
        dropout=0.0,
    )
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        belief_card_vocab=8,
        encoder_vocab=64,
        decoder_vocab=decoder_vocab,
    )
    model.eval()
    return model


def test_identical_cards_keep_owner_area_and_target_slot_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vocab(monkeypatch)
    same = [_card(), _card()]
    players = [
        SimpleNamespace(
            hand=same,
            discard=[_card()],
            active=[_card()],
            bench=[_card(), _card()],
            prize=[],
        ),
        SimpleNamespace(
            hand=[_card()],
            discard=[],
            active=[_card()],
            bench=[],
            prize=[],
        ),
    ]
    options = [
        _option(_OptionType.CARD, area=_AreaType.HAND, index=0, playerIndex=0),
        _option(_OptionType.CARD, area=_AreaType.HAND, index=1, playerIndex=0),
        _option(_OptionType.CARD, area=_AreaType.DISCARD, index=0, playerIndex=0),
        _option(_OptionType.CARD, area=_AreaType.HAND, index=0, playerIndex=1),
        _option(
            _OptionType.ATTACH,
            area=_AreaType.HAND,
            index=0,
            inPlayArea=_AreaType.BENCH,
            inPlayIndex=0,
        ),
        _option(
            _OptionType.ATTACH,
            area=_AreaType.HAND,
            index=0,
            inPlayArea=_AreaType.BENCH,
            inPlayIndex=1,
        ),
    ]
    obs = SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0,
            players=players,
            stadium=[],
            looking=[],
        ),
        select=SimpleNamespace(
            context=_SelectContext.MAIN,
            option=options,
            deck=[],
        ),
    )

    encoded = features.build_option_tokens(
        obs, [[i] for i in range(len(options))]
    )
    words = [_word_indices(encoded, i) for i in range(encoded.num_words)]

    # Same card id, but different source index, area, and owner stay distinct.
    assert len(set(words[:4])) == 4
    # Same source card and same target card id, but bench slot 0 != slot 1.
    assert words[4] != words[5]
    assert max(encoded.index) < features.decoder_vocab_size()

    # Adversarial multi-select case: separate owner/area/index marginals would
    # collide for these rank assignments, but composite tuple rows do not.
    multi = features.build_option_tokens(
        obs,
        [
            [0, 1, 3, 2],  # tuple weights (1, 2, 4, 3)
            [1, 0, 2, 3],  # tuple weights (2, 1, 3, 4)
        ],
    )
    assert _word_signature(multi, 0) != _word_signature(multi, 1)


def test_card2vec_vocab_estimate_tracks_binding_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vocab(monkeypatch)
    estimate = card2vec.estimate_bag_vocabs(
        features.card_vocab_size(), features.attack_vocab_size()
    )
    assert estimate.encoder_vocab == features.encoder_vocab_size()
    assert estimate.decoder_vocab == features.decoder_vocab_size()


def test_spatial_slot_embedding_distinguishes_identical_bench_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model(monkeypatch)
    board = features.SparseVector()
    for _ in range(features.NUM_BOARD_TOKENS):
        board.word_start()  # every content bag is identical and empty

    with torch.no_grad():
        model.spatial_slot_embedding.weight.zero_()
        model.spatial_slot_embedding.weight[:, 0] = torch.arange(
            features.NUM_BOARD_TOKENS, dtype=torch.float32
        )

    captured: list[torch.Tensor] = []

    def _capture(_module, args):
        captured.append(args[0].detach().clone())

    hook = model.spatial_encoder.register_forward_pre_hook(_capture)
    try:
        model.encode_board(board)
    finally:
        hook.remove()

    assert captured[0][0, 0, 0].item() == 0.0
    assert captured[0][0, 1, 0].item() == 1.0
    assert not torch.equal(captured[0][0, 0], captured[0][0, 1])


def test_dense_card_path_preserves_appended_binding_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vocab(monkeypatch)
    decoder_vocab = features.decoder_vocab_size()
    model = _small_model(
        monkeypatch,
        dense_card2vec=True,
        decoder_vocab=decoder_vocab,
    )
    binding = features.decoder_binding_offset()
    options = features.SparseVector()
    options.word_start()
    options.add(binding + 1, 1.0)  # one exact source binding tuple
    options.word_start()
    options.add(binding + 2, 1.0)  # a distinct source binding tuple
    packed = pack_sparse_vectors([options], torch.device("cpu"))

    with torch.no_grad():
        model.option_binding_embedding.weight.zero_()
        model.option_binding_projection.weight.zero_()
        model.option_binding_embedding.weight[1, 0] = 1.0
        model.option_binding_embedding.weight[2, 1] = 1.0
        model.option_binding_projection.weight[0, 0] = 1.0
        model.option_binding_projection.weight[1, 1] = 1.0
        embedded = model._embed_option_bag(
            packed.index, packed.offset, packed.value
        )

    assert not torch.equal(embedded[0], embedded[1])


def test_full_cache_keeps_absolute_rope_position_after_320(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model(monkeypatch, max_context=320)
    assert model.rope is not None
    head_dim = model.d_model // model.cfg.n_heads
    cached = torch.zeros(1, model.cfg.n_heads, 320, head_dim)
    cache = TemporalKVCache(
        layers=[(cached, cached.clone())],
        length=320,
        next_position=320,
    )
    observed_offsets: list[int] = []
    attended_key_lengths: list[int] = []
    original_forward = model.rope.forward

    def _capture_rope(seq_len: int, offset: int = 0):
        observed_offsets.append(offset)
        return original_forward(seq_len, offset=offset)

    monkeypatch.setattr(model.rope, "forward", _capture_rope)
    hook = model.temporal_blocks[0].attn.register_forward_hook(
        lambda _module, _args, output: attended_key_lengths.append(
            int(output[1].size(2))
        )
    )
    token = torch.zeros(1, 1, model.d_model)
    try:
        _, cache = model.temporal_encode(token, cache, append=True)
        assert cache is not None
        assert cache.length == 320
        assert cache.next_position == 321
        _, cache = model.temporal_encode(token, cache, append=True)
        assert cache is not None
        assert cache.length == 320
        assert cache.next_position == 322
    finally:
        hook.remove()
    assert observed_offsets == [320, 321]
    assert attended_key_lengths == [320, 320]


def test_training_context_cannot_silently_drop_unaligned_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model(monkeypatch, max_context=4)
    tokens = torch.zeros(1, 5, model.d_model)
    with pytest.raises(ValueError, match="truncate boards, actions, and targets"):
        model.temporal_encode(tokens, append=False, return_all=True)


@pytest.mark.parametrize("temporal_pos", ["rope", "learned"])
def test_rollover_recompute_matches_fresh_multilayer_training_window(
    monkeypatch: pytest.MonkeyPatch,
    temporal_pos: str,
) -> None:
    torch.manual_seed(7)
    model = _small_model(
        monkeypatch,
        max_context=4,
        temporal_layers=2,
        temporal_pos=temporal_pos,
    )
    tokens = torch.randn(1, 7, model.d_model)
    cache = None
    for step in range(tokens.size(1)):
        incremental, cache = model.temporal_encode(
            tokens[:, step : step + 1, :], cache, append=True
        )
        start = max(0, step + 1 - model.max_context)
        offline, _ = model.temporal_encode(
            tokens[:, start : step + 1, :],
            append=False,
            position_offset=start,
        )
        assert torch.allclose(incremental, offline, atol=1e-5)
    assert cache is not None
    assert cache.length == model.max_context
    assert cache.next_position == tokens.size(1)
    assert cache.input_tokens is not None
    assert cache.input_tokens.size(1) == model.max_context


def test_incompatible_feature_rows_and_legacy_checkpoints_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model(monkeypatch)
    bad_board = features.SparseVector()
    for slot in range(features.NUM_BOARD_TOKENS):
        bad_board.word_start()
        if slot == 0:
            bad_board.add(model.encoder_vocab, 1.0)
    with pytest.raises(ValueError, match="compatible checkpoint"):
        model.encode_board(bad_board)

    legacy_state = dict(model.state_dict())
    del legacy_state["_feature_schema_version"]
    fresh = _small_model(monkeypatch)
    with pytest.raises(RuntimeError, match="predates explicit slot/option-binding"):
        fresh.load_state_dict(legacy_state, strict=False)
