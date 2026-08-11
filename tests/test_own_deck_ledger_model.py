"""Dormant shared own-deck-ledger model integration guards."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from poke_bot import config, features
from poke_bot.features import SparseVector
from poke_bot.model import DECISION_FUSION_REQUIRED_HEADS, build_model
from poke_bot.own_deck_ledger import OPTION_FEATURE_DIM, OwnDeckLedger


@pytest.fixture(autouse=True)
def _stub_feature_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model-only coverage independent of an installed card-game runtime."""

    # ``build_model`` receives the main vocabularies explicitly below, but its
    # legacy attack/binding layout still consults these feature helpers.
    monkeypatch.setattr(features, "card_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "encoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 64)


def _cfg(**overrides: object) -> config.ModelConfig:
    values: dict[str, object] = {
        "d_model": 16,
        "spatial_layers": 1,
        "temporal_layers": 1,
        "option_decoder_layers": 1,
        "n_heads": 4,
        "ff_dim": 32,
        "max_context": 8,
        "temporal_pos": "rope",
        "decision_context": "history",
        "kv_cache": True,
        "dropout": 0.0,
    }
    values.update(overrides)
    return config.ModelConfig(**values)


def _model(cfg: config.ModelConfig):
    return build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )


def _board() -> SparseVector:
    result = SparseVector()
    for _ in range(features.NUM_BOARD_TOKENS):
        result.word_start()
    return result


def _options(n: int = 3) -> SparseVector:
    result = SparseVector()
    for index in range(n):
        result.word_start()
        result.add((index + 1) % 32, 1.0)
    return result


def _snapshot():
    ledger = OwnDeckLedger([1] * 60)
    observation = {
        "current": {
            "yourIndex": 0,
            "looking": [],
            "players": [
                {
                    "hand": [{"id": 1, "serial": 101}],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * 6,
                    "deckCount": 53,
                },
                {"hand": [], "active": [], "bench": [], "discard": [], "prize": []},
            ],
        },
        "select": {"deck": [], "option": []},
    }
    snapshot = ledger.observe(observation)
    assert snapshot.integrity_ok is True
    assert snapshot.fail_closed is False
    assert len(snapshot.scalar_vector) == 10
    return snapshot


def _two_card_snapshot():
    """A valid ledger whose two card rows have deliberately different facts."""

    ledger = OwnDeckLedger([1] * 30 + [2] * 30)
    observation = {
        "current": {
            "yourIndex": 0,
            "looking": [],
            "players": [
                {
                    "hand": [{"id": 1, "serial": 101}],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * 6,
                    "deckCount": 53,
                },
                {"hand": [], "active": [], "bench": [], "discard": [], "prize": []},
            ],
        },
        "select": {"deck": [], "option": []},
    }
    snapshot = ledger.observe(observation)
    assert snapshot.integrity_ok is True
    assert snapshot.fail_closed is False
    assert len(snapshot.card_availability) == 2
    return snapshot


def test_default_off_state_dict_strict_load_and_fusion_inventory_are_legacy() -> None:
    torch.manual_seed(17)
    parent = _model(_cfg())
    parent_state = parent.state_dict()
    assert not any("own_deck_ledger" in key for key in parent_state)
    assert not any("visible_tutor_completion" in key for key in parent_state)
    assert not any("terminal_conversion" in key for key in parent_state)

    torch.manual_seed(18)
    successor_off = _model(_cfg())
    result = successor_off.load_state_dict(parent_state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert tuple(successor_off.decision_fusion_inventory()["required_heads"]) == (
        DECISION_FUSION_REQUIRED_HEADS
    )
    assert successor_off.own_deck_ledger_inventory()["enabled"] is False


def test_ledger_snapshot_is_zero_safe_and_malformed_snapshot_is_neutral() -> None:
    snapshot = _snapshot()
    model = _model(
        _cfg(
            own_deck_ledger_enabled=True,
            own_deck_ledger_runtime_enabled=True,
        )
    )
    model.eval()
    board = _board()
    options = _options()

    baseline = model.forward(board, options, n_options=[3])
    zero_safe = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=snapshot,
    )
    for name in ("policy_logits", "value", "state_vec", "aux_logits"):
        torch.testing.assert_close(
            zero_safe[name], baseline[name], rtol=0.0, atol=0.0
        )

    assert model.own_deck_ledger_adapter is not None
    with torch.no_grad():
        model.own_deck_ledger_adapter.output.bias.fill_(0.25)
    enriched = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=snapshot,
    )
    assert not torch.equal(enriched["state_vec"], baseline["state_vec"])
    canonical_mapping = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=snapshot.to_dict(),
    )
    for name in ("policy_logits", "value", "state_vec", "aux_logits"):
        torch.testing.assert_close(
            canonical_mapping[name], enriched[name], rtol=0.0, atol=0.0
        )

    malformed = replace(snapshot, integrity_ok=False, fail_closed=True)
    neutral = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=malformed,
    )
    empty = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=[],
    )
    structurally_malformed = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots={"unexpected": "payload"},
    )
    tampered_fingerprint = snapshot.to_dict()
    tampered_fingerprint["fingerprint"] = "sha256:" + "0" * 64
    fingerprint_neutral = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=tampered_fingerprint,
    )
    tampered_starting_counts = snapshot.to_dict()
    tampered_starting_counts["starting_counts"] = [[1, 59]]
    starting_counts_neutral = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=tampered_starting_counts,
    )
    tampered_conservation = snapshot.to_dict()
    tampered_conservation["unknown_non_deck_slots"] = 999
    conservation_neutral = model.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=tampered_conservation,
    )
    for output in (
        neutral,
        empty,
        structurally_malformed,
        fingerprint_neutral,
        starting_counts_neutral,
        conservation_neutral,
    ):
        for name in ("policy_logits", "value", "state_vec", "aux_logits"):
            torch.testing.assert_close(
                output[name], baseline[name], rtol=0.0, atol=0.0
            )

    # A physically present adapter remains an exact serving no-op until its
    # independent runtime receipt is enabled.
    runtime_off = _model(_cfg(own_deck_ledger_enabled=True))
    runtime_off.eval()
    assert runtime_off.own_deck_ledger_adapter is not None
    with torch.no_grad():
        runtime_off.own_deck_ledger_adapter.output.bias.fill_(0.75)
    off_baseline = runtime_off.forward(board, options, n_options=[3])
    off_with_snapshot = runtime_off.forward(
        board,
        options,
        n_options=[3],
        ledger_snapshots=snapshot,
    )
    for name in ("policy_logits", "value", "state_vec", "aux_logits"):
        torch.testing.assert_close(
            off_with_snapshot[name], off_baseline[name], rtol=0.0, atol=0.0
        )
    # Shadow/validation loss code must opt in explicitly: model.eval() alone
    # is still a serving bypass, while the internal offline path exercises the
    # physical successor adapter without enabling runtime action authority.
    offline_residual = runtime_off.own_deck_ledger_residuals(
        snapshot,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        offline_training_path=True,
    )
    assert offline_residual is not None
    assert torch.count_nonzero(offline_residual) > 0

    assert runtime_off.own_deck_ledger_option_adapter is not None
    with torch.no_grad():
        runtime_off.own_deck_ledger_option_adapter.network[-1].bias.fill_(0.5)
    option_rows = [snapshot.features_for_card(1), (0.0,) * OPTION_FEATURE_DIM]
    assert runtime_off.own_deck_ledger_option_residuals(
        option_rows,
        batch_size=1,
        max_options=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) is None
    offline_option_residual = runtime_off.own_deck_ledger_option_residuals(
        option_rows,
        batch_size=1,
        max_options=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        offline_training_path=True,
    )
    assert offline_option_residual is not None
    assert torch.count_nonzero(offline_option_residual[0, 0]) > 0
    assert torch.count_nonzero(offline_option_residual[0, 1]) == 0
    ordinary_option_logits = runtime_off.decode_options(
        options,
        off_baseline["spatial_memory"],
        off_baseline["state_vec"],
        n_options=[3],
        ledger_option_features=[
            snapshot.features_for_card(1),
            (0.0,) * OPTION_FEATURE_DIM,
            (0.0,) * OPTION_FEATURE_DIM,
        ],
    )
    offline_option_logits = runtime_off.decode_options(
        options,
        off_baseline["spatial_memory"],
        off_baseline["state_vec"],
        n_options=[3],
        ledger_option_features=[
            snapshot.features_for_card(1),
            (0.0,) * OPTION_FEATURE_DIM,
            (0.0,) * OPTION_FEATURE_DIM,
        ],
        offline_training_path=True,
    )
    assert not torch.equal(ordinary_option_logits, offline_option_logits)


def test_ledger_option_features_are_explicit_and_stop_rows_stay_neutral() -> None:
    snapshot = _snapshot()
    model = _model(
        _cfg(
            own_deck_ledger_enabled=True,
            own_deck_ledger_runtime_enabled=True,
        )
    )
    model.eval()
    assert model.own_deck_ledger_option_adapter is not None
    with torch.no_grad():
        model.own_deck_ledger_option_adapter.network[-1].bias.fill_(0.5)

    rows = [snapshot.features_for_card(1), (0.0,) * OPTION_FEATURE_DIM, (0.0,) * OPTION_FEATURE_DIM]
    residuals = model.own_deck_ledger_option_residuals(
        rows,
        batch_size=1,
        max_options=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert residuals is not None
    assert torch.count_nonzero(residuals[0, 0]) > 0
    assert torch.count_nonzero(residuals[0, 1:]) == 0

    # Batched serving may mix a successor packet with a legacy packet.
    mixed = model.own_deck_ledger_option_residuals(
        [rows, None],
        batch_size=2,
        max_options=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mixed is not None
    assert torch.count_nonzero(mixed[0]) > 0
    assert torch.count_nonzero(mixed[1]) == 0


def test_ledger_availability_interaction_retains_card_identity_before_pooling() -> None:
    snapshot = _two_card_snapshot()
    first, second = snapshot.card_availability
    assert first.card_id != second.card_id
    assert (
        first.lower,
        first.upper,
        first.expected,
        first.probability_at_least_one,
        first.exact,
    ) != (
        second.lower,
        second.upper,
        second.expected,
        second.probability_at_least_one,
        second.exact,
    )
    swapped = replace(
        snapshot,
        card_availability=(
            replace(
                first,
                lower=second.lower,
                upper=second.upper,
                expected=second.expected,
                probability_at_least_one=second.probability_at_least_one,
                exact=second.exact,
            ),
            replace(
                second,
                lower=first.lower,
                upper=first.upper,
                expected=first.expected,
                probability_at_least_one=first.probability_at_least_one,
                exact=first.exact,
            ),
        ),
    )
    model = _model(
        _cfg(
            own_deck_ledger_enabled=True,
            own_deck_ledger_runtime_enabled=True,
            own_deck_ledger_width=16,
        )
    )
    model.eval()
    adapter = model.own_deck_ledger_adapter
    assert adapter is not None
    with torch.no_grad():
        adapter.output.weight.copy_(torch.eye(16))
        adapter.output.bias.zero_()
    residuals = model.own_deck_ledger_residuals(
        [snapshot, swapped],
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert residuals is not None
    assert not torch.equal(residuals[0], residuals[1])


def test_history_snapshots_and_typed_successor_heads_stay_outside_legacy_fusion() -> None:
    snapshot = _snapshot()
    cfg = _cfg(
        own_deck_ledger_enabled=True,
        own_deck_ledger_runtime_enabled=True,
        visible_tutor_completion_head_enabled=True,
        terminal_conversion_head_enabled=True,
    )
    model = _model(cfg)
    model.eval()
    result = model.forward_history_batch(
        [[_board(), _board()]],
        [_options()],
        n_options=[3],
        ledger_histories=[[snapshot, snapshot]],
    )
    assert tuple(result["visible_tutor_completion_logits"].shape) == (1, 3, 7)
    assert tuple(result["terminal_conversion_logits"].shape) == (1, 3, 6)
    inventory = model.own_deck_option_head_inventory()
    assert inventory["legacy_decision_fusion_denominator_changed"] is False
    assert set(inventory["modules"]) == {
        "visible_tutor_completion",
        "terminal_conversion",
    }
    assert tuple(model.decision_fusion_inventory()["required_heads"]) == (
        DECISION_FUSION_REQUIRED_HEADS
    )


def test_invalid_successor_runtime_gates_fail_closed() -> None:
    with pytest.raises(ValueError, match="own_deck_ledger_runtime_enabled"):
        _model(_cfg(own_deck_ledger_runtime_enabled=True))
    with pytest.raises(ValueError, match="heads require own_deck_ledger_enabled"):
        _model(_cfg(visible_tutor_completion_head_enabled=True))
    with pytest.raises(ValueError, match="heads require own_deck_ledger_enabled"):
        _model(_cfg(terminal_conversion_head_enabled=True))
    with pytest.raises(ValueError, match="visible tutor completion route"):
        _model(_cfg(visible_tutor_completion_route_enabled=True))
    with pytest.raises(ValueError, match="terminal conversion route"):
        _model(_cfg(terminal_conversion_route_runtime_enabled=True))
    with pytest.raises(ValueError, match="own_deck_ledger_runtime_enabled"):
        _model(
            _cfg(
                own_deck_ledger_enabled=True,
                visible_tutor_completion_head_enabled=True,
                visible_tutor_completion_route_enabled=True,
                visible_tutor_completion_route_runtime_enabled=True,
            )
        )
