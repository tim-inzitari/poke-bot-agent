"""Factual promotion telemetry for the staged own-deck successor."""

from __future__ import annotations

import copy

import pytest
import torch

from poke_bot import config, features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.features import SparseVector
from poke_bot.model import build_model
from poke_bot.own_deck_ledger import OPTION_FEATURE_DIM, OwnDeckLedger
from poke_bot.own_deck_promotion_metrics import (
    OwnDeckPromotionMetricsError,
    collect_own_deck_promotion_metrics,
    merge_own_deck_promotion_metrics,
)
from poke_bot.own_deck_supervision import (
    OWN_DECK_SUPERVISION_SCHEMA,
    OWN_DECK_SUPERVISION_VERSION,
)
from poke_bot.train import (
    TrainConfig,
    _maybe_own_deck_promotion_metrics_record,
    _merge_metrics,
    batch_losses,
    evaluate,
)


@pytest.fixture(autouse=True)
def _stub_feature_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real-model shadow collection check runtime-independent."""

    monkeypatch.setattr(features, "card_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "encoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 64)


def _row(
    *,
    terminal_class: int,
    prize_closeout: float,
    opponent_knockout: float,
    tutor_visible_selected: bool = False,
) -> dict:
    return {
        "schema": OWN_DECK_SUPERVISION_SCHEMA,
        "version": OWN_DECK_SUPERVISION_VERSION,
        "terminal_conversion": {
            "terminal_class": {"value": terminal_class, "mask": True},
            "prize_closeout": {"value": prize_closeout, "mask": True},
            "opponent_knockout": {
                "value": opponent_knockout,
                "mask": True,
            },
        },
        "visible_tutor_completion": {
            "selected_from_visible_deck": {
                "value": 1.0 if tutor_visible_selected else 0.0,
                "mask": tutor_visible_selected,
            },
            "selected_target_observed_after_action": {
                "value": 0.0,
                "mask": False,
            },
            "same_actor_followup": {"value": 0.0, "mask": False},
            "same_actor_terminal_class": {"value": 0, "mask": False},
        },
    }


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    # Rows 0 and 2 are actual observed tutor-menu selections.  The direct
    # policy gets row 0 right and row 2 wrong; no unselected option receives a
    # label or enters any denominator.
    policy_logits = torch.tensor(
        [
            [4.0, 1.0],
            [0.0, 2.0],
            [4.0, 1.0],
        ]
    )
    selected = torch.tensor([0, 1, 1])
    terminal_logits = torch.tensor(
        [
            [
                [0.0, 3.0, -2.0, -2.0, 3.0, -3.0],
                [0.0, -2.0, -2.0, -2.0, -3.0, -3.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, -2.0, -2.0, -2.0, -3.0, -3.0],
            ],
            [
                [0.0, -2.0, -2.0, -2.0, -3.0, -3.0],
                [0.0, 0.0, -2.0, -2.0, -3.0, -3.0],
            ],
        ]
    )
    rows = [
        _row(
            terminal_class=1,
            prize_closeout=1.0,
            opponent_knockout=1.0,
            tutor_visible_selected=True,
        ),
        _row(
            terminal_class=0,
            prize_closeout=0.0,
            opponent_knockout=0.0,
        ),
        _row(
            terminal_class=1,
            prize_closeout=1.0,
            opponent_knockout=1.0,
            tutor_visible_selected=True,
        ),
    ]
    return policy_logits, selected, terminal_logits, rows


def test_collects_selected_observed_label_metrics_only() -> None:
    policy_logits, selected, terminal_logits, rows = _inputs()

    metrics = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits,
        selected_indices=selected,
        terminal_logits=terminal_logits,
        supervision_rows=rows,
    )

    assert metrics["collection"] == {
        "observed_expert_labels_only": True,
        "gradient_contribution": False,
        "runtime_route_authority": False,
    }
    assert metrics["visible_tutor_observed_menu_expert_top1"] == {
        "numerator": 1,
        "denominator": 2,
        "agreement": 0.5,
    }
    recall = metrics["selected_option_factual_recall"]
    assert recall["own_win"] == {"numerator": 1, "denominator": 2, "recall": 0.5}
    assert recall["prize_closeout"] == {
        "numerator": 1,
        "denominator": 2,
        "recall": 0.5,
    }
    assert recall["opponent_knockout"] == {
        "numerator": 0,
        "denominator": 2,
        "recall": 0.0,
    }
    terminal = metrics["terminal_multiclass"]
    assert terminal["brier_denominator"] == 3
    assert sum(terminal["ece_bin_rows"]) == 3
    assert terminal["ece"] is not None
    assert metrics["terminal_scalar_brier"]["prize_closeout"]["brier_denominator"] == 3
    assert (
        metrics["terminal_scalar_brier"]["opponent_knockout"]["brier_denominator"] == 3
    )
    missed = metrics["missed_expert_closeout"]
    # The third factual closeout selected option 1, while direct policy top-1
    # chose option 0. This stays a policy-choice miss even if terminal-head
    # score thresholds change.
    assert missed["basis"] == "policy_top1_vs_observed_expert_selected_option"
    assert missed["numerator"] == 1
    assert missed["denominator"] == 2
    assert missed["miss_rate"] == 0.5
    assert missed["by_kind"]["own_win"] == {
        "numerator": 1,
        "denominator": 2,
        "miss_rate": 0.5,
    }
    assert missed["by_kind"]["opponent_knockout"] == {
        "numerator": 1,
        "denominator": 2,
        "miss_rate": 0.5,
    }

    higher_threshold = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits,
        selected_indices=selected,
        terminal_logits=terminal_logits,
        supervision_rows=rows,
        threshold=0.99,
    )
    assert higher_threshold["missed_expert_closeout"] == missed
    assert (
        higher_threshold["selected_option_factual_recall"]["own_win"]
        != (recall["own_win"])
    )


def test_merge_uses_sufficient_statistics_not_batch_averages() -> None:
    policy_logits, selected, terminal_logits, rows = _inputs()
    single = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits,
        selected_indices=selected,
        terminal_logits=terminal_logits,
        supervision_rows=rows,
    )
    first = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits[:1],
        selected_indices=selected[:1],
        terminal_logits=terminal_logits[:1],
        supervision_rows=rows[:1],
    )
    second = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits[1:],
        selected_indices=selected[1:],
        terminal_logits=terminal_logits[1:],
        supervision_rows=rows[1:],
    )

    merged = merge_own_deck_promotion_metrics([first, second])

    assert (
        merged["visible_tutor_observed_menu_expert_top1"]
        == single["visible_tutor_observed_menu_expert_top1"]
    )
    assert (
        merged["selected_option_factual_recall"]
        == single["selected_option_factual_recall"]
    )
    assert merged["missed_expert_closeout"] == single["missed_expert_closeout"]
    assert merged["terminal_multiclass"]["brier_denominator"] == 3
    assert merged["terminal_multiclass"]["brier_numerator"] == pytest.approx(
        single["terminal_multiclass"]["brier_numerator"]
    )
    assert merged["terminal_multiclass"]["ece"] == pytest.approx(
        single["terminal_multiclass"]["ece"]
    )
    assert merged["terminal_scalar_brier"]["prize_closeout"]["brier"] == pytest.approx(
        single["terminal_scalar_brier"]["prize_closeout"]["brier"]
    )


def test_stage_aligned_tutor_and_terminal_rows_are_collected_independently() -> None:
    policy_logits, selected, terminal_logits, rows = _inputs()

    # A factorized tutor card choice can occur before a later STOP/complete
    # stage. The separate arrays guarantee that tutor agreement is attributed
    # to the observed card stage while terminal facts remain on completed
    # action stages only.
    metrics = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits,
        selected_indices=selected,
        terminal_logits=terminal_logits,
        visible_tutor_supervision_rows=[rows[0], None, rows[2]],
        terminal_conversion_supervision_rows=[None, rows[1], rows[2]],
    )

    assert metrics["visible_tutor_observed_menu_expert_top1"] == {
        "numerator": 1,
        "denominator": 2,
        "agreement": 0.5,
    }
    assert metrics["terminal_multiclass"]["brier_denominator"] == 2
    assert metrics["selected_option_factual_recall"]["own_win"] == {
        "numerator": 0,
        "denominator": 1,
        "recall": 0.0,
    }


def test_empty_factual_denominators_are_explicit_and_not_zero_rates() -> None:
    policy_logits = torch.tensor([[1.0, 0.0]])
    selected = torch.tensor([0])
    terminal_logits = torch.zeros(1, 2, 6)
    masked = _row(terminal_class=0, prize_closeout=0.0, opponent_knockout=0.0)
    for family in ("terminal_conversion", "visible_tutor_completion"):
        for value in masked[family].values():
            if isinstance(value, dict) and "mask" in value:
                value["mask"] = False
    metrics = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits,
        selected_indices=selected,
        terminal_logits=terminal_logits,
        supervision_rows=[masked],
    )

    assert metrics["visible_tutor_observed_menu_expert_top1"]["denominator"] == 0
    assert metrics["visible_tutor_observed_menu_expert_top1"]["agreement"] is None
    assert metrics["selected_option_factual_recall"]["own_win"]["recall"] is None
    assert metrics["terminal_multiclass"]["brier"] is None
    assert metrics["terminal_multiclass"]["ece"] is None
    assert metrics["missed_expert_closeout"]["miss_rate"] is None


def test_collection_is_detached_and_rejects_contract_drift() -> None:
    policy_logits, selected, terminal_logits, rows = _inputs()
    policy_logits.requires_grad_()
    terminal_logits.requires_grad_()
    metrics = collect_own_deck_promotion_metrics(
        policy_logits=policy_logits,
        selected_indices=selected,
        terminal_logits=terminal_logits,
        supervision_rows=rows,
    )

    def _contains_tensor(value: object) -> bool:
        if torch.is_tensor(value):
            return True
        if isinstance(value, dict):
            return any(_contains_tensor(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return any(_contains_tensor(child) for child in value)
        return False

    assert _contains_tensor(metrics) is False
    assert policy_logits.grad is None
    assert terminal_logits.grad is None

    drifted = copy.deepcopy(metrics)
    drifted["closeout_threshold"] = 0.75
    with pytest.raises(OwnDeckPromotionMetricsError, match="threshold"):
        merge_own_deck_promotion_metrics([metrics, drifted])


def _sparse_board() -> SparseVector:
    result = SparseVector()
    for _ in range(features.NUM_BOARD_TOKENS):
        result.word_start()
    return result


def _sparse_options(count: int = 2) -> SparseVector:
    result = SparseVector()
    for index in range(count):
        result.word_start()
        result.add(index + 1, 1.0)
    return result


def _ledger_snapshot() -> object:
    ledger = OwnDeckLedger([1] * 60)
    snapshot = ledger.observe(
        {
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
                    {
                        "hand": [],
                        "active": [],
                        "bench": [],
                        "discard": [],
                        "prize": [],
                    },
                ],
            },
            "select": {"deck": [], "option": []},
        }
    )
    assert snapshot.integrity_ok is True
    assert snapshot.fail_closed is False
    return snapshot


def _shadow_successor_model(*, route_enabled: bool = False) -> torch.nn.Module:
    model_cfg = config.ModelConfig(
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
        own_deck_ledger_enabled=True,
        visible_tutor_completion_head_enabled=True,
        terminal_conversion_head_enabled=True,
        visible_tutor_completion_route_enabled=route_enabled,
        terminal_conversion_route_enabled=route_enabled,
    )
    return build_model(
        model_cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )


def _shadow_sequence() -> GameSequence:
    snapshot = _ledger_snapshot()
    labels = _row(
        terminal_class=1,
        prize_closeout=1.0,
        opponent_knockout=1.0,
        tutor_visible_selected=True,
    )
    options = _sparse_options()
    stage = PolicyStage(
        options=options,
        action_combos=[[0], [1]],
        target_index=0,
        ledger_option_features=(
            snapshot.features_for_card(1),
            (0.0,) * OPTION_FEATURE_DIM,
        ),
    )
    decision = DecisionSample(
        board=_sparse_board(),
        options=options,
        action=[0],
        action_combo_index=0,
        action_combos=stage.action_combos,
        env_step=1,
        policy_stages=[stage],
        ledger_snapshot=snapshot,
        own_deck_supervision=labels,
        own_deck_supervision_stage_indices={
            "visible_tutor_completion": (0,),
            "terminal_conversion": (0,),
        },
    )
    return GameSequence(
        episode_id="shadow-promotion-metrics",
        seat=0,
        archetype="alakazam",
        opp_archetype="other",
        deck=[1] * 60,
        value=0.0,
        decisions=[decision],
    )


def test_shadow_collection_is_evaluation_ready_with_zero_typed_loss_weights() -> None:
    """The explicit flag emits persisted facts without training either head."""

    torch.manual_seed(19)
    model = _shadow_successor_model()
    sequence = _shadow_sequence()
    common = {
        "value_weight": 0.0,
        "aux_weight": 0.0,
        "opp_hand_weight": 0.0,
        "opp_remainder_weight": 0.0,
        "alakazam_guide_weight": 0.0,
        "setup_board_outcome_loss_weight": 0.0,
        "combo_state_loss_weight": 0.0,
        "lethal_threat_weight": 0.0,
        "prize_race_weight": 0.0,
        "history_identity_weight": 0.0,
        "visible_tutor_completion_loss_weight": 0.0,
        "terminal_conversion_loss_weight": 0.0,
    }

    model.train()
    reference_loss, reference_metrics = batch_losses(model, [sequence], **common)
    assert reference_metrics.own_deck_promotion_metrics == {}

    model.zero_grad(set_to_none=True)
    shadow_loss, shadow_metrics = batch_losses(
        model,
        [sequence],
        collect_own_deck_promotion_metrics=True,
        **common,
    )
    torch.testing.assert_close(shadow_loss, reference_loss)
    assert (
        shadow_metrics.own_deck_promotion_metrics[
            "visible_tutor_observed_menu_expert_top1"
        ]["denominator"]
        == 1
    )
    assert (
        shadow_metrics.own_deck_promotion_metrics["terminal_multiclass"][
            "brier_denominator"
        ]
        == 1
    )

    shadow_loss.backward()
    assert model.visible_tutor_completion_head is not None
    assert model.terminal_conversion_head is not None
    assert all(
        parameter.grad is None
        for parameter in model.visible_tutor_completion_head.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.terminal_conversion_head.parameters()
    )
    assert model.visible_tutor_completion_route_enabled is False
    assert model.terminal_conversion_route_enabled is False

    evaluation_cfg = TrainConfig(
        amp=False,
        games_per_batch=1,
        max_decisions_per_batch=4,
        value_loss_weight=0.0,
        aux_loss_weight=0.0,
        opp_hand_loss_weight=0.0,
        opp_remainder_loss_weight=0.0,
        alakazam_guide_loss_weight=0.0,
        setup_board_outcome_loss_weight=0.0,
        combo_state_loss_weight=0.0,
        lethal_threat_loss_weight=0.0,
        prize_race_loss_weight=0.0,
        visible_tutor_completion_loss_weight=0.0,
        terminal_conversion_loss_weight=0.0,
        collect_own_deck_promotion_metrics=True,
    )
    evaluated = evaluate(model, [sequence], cfg=evaluation_cfg, desc="shadow")
    assert (
        evaluated.own_deck_promotion_metrics["selected_option_factual_recall"][
            "own_win"
        ]["denominator"]
        == 1
    )

    merged = _merge_metrics([shadow_metrics, evaluated])
    assert (
        merged.own_deck_promotion_metrics["terminal_multiclass"]["brier_denominator"]
        == 2
    )
    checkpoint_record = _maybe_own_deck_promotion_metrics_record(
        cfg=evaluation_cfg,
        train_metrics=shadow_metrics,
        validation_metrics=evaluated,
    )
    assert checkpoint_record["gradient_contribution"] is False
    assert checkpoint_record["runtime_route_authority"] is False
    assert checkpoint_record["train"]["terminal_multiclass"]["brier_denominator"] == 1


def test_shadow_flag_does_not_change_an_already_enabled_training_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metrics may observe a training route, but never activate it themselves."""

    torch.manual_seed(23)
    model = _shadow_successor_model(route_enabled=True)
    sequence = _shadow_sequence()
    assert model.visible_tutor_completion_route_enabled is True
    assert model.terminal_conversion_route_enabled is True
    assert model.visible_tutor_completion_route_runtime_enabled is False
    assert model.terminal_conversion_route_runtime_enabled is False
    assert model.visible_tutor_completion_route is not None
    assert model.terminal_conversion_route is not None
    # Make the otherwise zero-safe training routes visibly nonzero, so this
    # regression covers the configuration that can affect training policy.
    with torch.no_grad():
        model.visible_tutor_completion_route.network[-1].weight.fill_(0.025)
        model.terminal_conversion_route.network[-1].weight.fill_(0.025)

    observed_logits: list[torch.Tensor] = []
    return_hidden_flags: list[bool] = []
    original_decode = model.decode_options

    def _capture_decode(*args, **kwargs):
        return_hidden_flags.append(bool(kwargs.get("return_hidden", False)))
        decoded = original_decode(*args, **kwargs)
        logits = decoded[0] if isinstance(decoded, tuple) else decoded
        observed_logits.append(logits.detach().clone())
        return decoded

    monkeypatch.setattr(model, "decode_options", _capture_decode)
    common = {
        "value_weight": 0.0,
        "aux_weight": 0.0,
        "opp_hand_weight": 0.0,
        "opp_remainder_weight": 0.0,
        "alakazam_guide_weight": 0.0,
        "setup_board_outcome_loss_weight": 0.0,
        "combo_state_loss_weight": 0.0,
        "lethal_threat_weight": 0.0,
        "prize_race_weight": 0.0,
        "history_identity_weight": 0.0,
        "visible_tutor_completion_loss_weight": 0.0,
        "terminal_conversion_loss_weight": 0.0,
    }

    model.train()
    without_metrics, _ = batch_losses(model, [sequence], **common)
    with_metrics, metric_rows = batch_losses(
        model,
        [sequence],
        collect_own_deck_promotion_metrics=True,
        **common,
    )

    assert return_hidden_flags == [True, True]
    assert len(observed_logits) == 2
    torch.testing.assert_close(observed_logits[0], observed_logits[1])
    torch.testing.assert_close(without_metrics, with_metrics)
    assert metric_rows.own_deck_promotion_metrics
