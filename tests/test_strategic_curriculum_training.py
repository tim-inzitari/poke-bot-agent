from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from poke_bot import config, features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.model import build_model
from poke_bot.strategic_heads import (
    ACTION_UTILITY_NAMES,
    EXPANDED_STRATEGIC_KEY,
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    EXPANDED_STRATEGIC_SCHEMA_VERSION,
    OPPONENT_RESPONSE_NAMES,
    RESOURCE_FORECAST_NAMES,
    TACTICAL_HORIZONS,
    TACTICAL_OUTCOME_NAMES,
)
from poke_bot.strategic_losses import (
    GUIDE_OUTCOME_BACKED_HEAD_IDS,
    expanded_strategic_losses,
    guide_outcome_backed_loss_weights,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS
from poke_bot.train import (
    GUIDE_TRAINING_MODE_STRATEGIC,
    batch_losses,
    device_temporal_batch_losses,
)
from scripts.train_pure_rl import _parse_args


def _sparse(words: int, *, offset: int = 0) -> features.SparseVector:
    value = features.SparseVector()
    for index in range(words):
        value.word_start()
        value.add(offset + index, 1.0)
    return value


def _expanded_target() -> dict:
    tactical = [
        [float(horizon + column) for column in range(len(TACTICAL_OUTCOME_NAMES))]
        for horizon in range(len(TACTICAL_HORIZONS))
    ]
    for row in tactical:
        row[2] = 1.0
        row[3] = 0.0
    resources = [float(index + 1) for index in range(len(RESOURCE_FORECAST_NAMES))]
    resources[4] = 1.0
    resources[5] = 0.0
    utility = [float(index + 1) for index in range(len(ACTION_UTILITY_NAMES))]
    utility[-1] = 1.0
    return {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "version": EXPANDED_STRATEGIC_SCHEMA_VERSION,
        "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
        "action_factors": [
            {"action_type": True, "target": True, "resource": True}
        ],
        "action_utility": {
            "values": utility,
            "mask": [True] * len(ACTION_UTILITY_NAMES),
        },
        "tactical_outcomes": {
            "values": tactical,
            "mask": [
                [True] * len(TACTICAL_OUTCOME_NAMES)
                for _ in TACTICAL_HORIZONS
            ],
        },
        "opponent_response": {
            "values": [
                float(index % 2) for index in range(len(OPPONENT_RESPONSE_NAMES))
            ],
            "mask": [True] * len(OPPONENT_RESPONSE_NAMES),
        },
        "resource_forecast": {
            "values": resources,
            "mask": [True] * len(RESOURCE_FORECAST_NAMES),
        },
        "game_phase": 2,
        "outcome_class": 2,
        "remaining_turns_log1p": 1.25,
        "provenance": {
            "trajectory": "full_untruncated_same_seat",
            "transition_after": "explicit_public_post_action",
            "terminal_complete": True,
            "target_only": True,
        },
    }


def _sequence(guide_target_index: int) -> GameSequence:
    options = _sparse(2, offset=1)
    stage = PolicyStage(
        options=options,
        action_combos=[[0], [1]],
        target_index=0,
        guide_target_index=guide_target_index,
        guide_confidence=0.75,
        select_context=1,
        selected_is_stop=False,
    )
    decision = DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, offset=1),
        options=options,
        action=[0],
        action_combo_index=0,
        action_combos=stage.action_combos,
        env_step=0,
        policy_stages=[stage],
        aux_labels={EXPANDED_STRATEGIC_KEY: _expanded_target()},
    )
    return GameSequence(
        episode_id="strategic-guide",
        seat=0,
        archetype="archaludon-ex",
        opp_archetype="baseline",
        deck=[1] * 60,
        value=1.0,
        decisions=[decision],
    )


def _future_model():
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
        expanded_heads_enabled=True,
        setup_board_outcome_head_enabled=True,
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=False,
        decision_fusion_dedicated_routes_enabled=True,
        decision_fusion_dedicated_routes_runtime_enabled=False,
        decision_fusion_width=8,
    )
    model = build_model(
        cfg,
        aux_archetype_classes=4,
        encoder_vocab=128,
        decoder_vocab=128,
        belief_card_vocab=32,
    )
    model.train()
    return model


def test_curriculum_weights_cover_only_observed_outcome_heads() -> None:
    weights = guide_outcome_backed_loss_weights()
    assert set(weights) == set(EXPANDED_HEAD_IDS)
    assert {
        name for name, weight in weights.items() if weight > 0.0
    } == set(GUIDE_OUTCOME_BACKED_HEAD_IDS)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_curriculum_confidence_scales_observed_losses_not_factor_imitation() -> None:
    option_outputs = {
        "action_q": torch.zeros(1, 2, requires_grad=True),
        "action_type": torch.zeros(1, 2, requires_grad=True),
        "action_target": torch.zeros(1, 2, requires_grad=True),
        "action_resource": torch.zeros(1, 2, requires_grad=True),
        "action_utility": torch.zeros(1, 2, 6, requires_grad=True),
    }
    state_outputs = {
        "tactical_outcome": torch.zeros(1, 3, 6, requires_grad=True),
        "opponent_response": torch.zeros(1, 7, requires_grad=True),
        "resource_forecast": torch.zeros(1, 6, requires_grad=True),
        "game_phase": torch.zeros(1, 5, requires_grad=True),
        "outcome_distribution": torch.zeros(1, 3, requires_grad=True),
        "remaining_turns": torch.zeros(1, 1, requires_grad=True),
    }

    def compute(confidence: float) -> torch.Tensor:
        loss, _ = expanded_strategic_losses(
            option_outputs=option_outputs,
            state_outputs=state_outputs,
            target_indices=torch.tensor([0]),
            value_targets=torch.tensor([1.0]),
            option_counts=[2],
            stage_indices=[0],
            decision_aux=[{EXPANDED_STRATEGIC_KEY: _expanded_target()}],
            weights=guide_outcome_backed_loss_weights(),
            row_weights=torch.tensor([confidence]),
            state_row_weights=torch.tensor([confidence]),
        )
        return loss

    half = compute(0.5)
    full = compute(1.0)
    assert float(half.detach()) > 0.0
    assert float(full.detach()) == pytest.approx(float(half.detach()) * 2.0)
    full.backward()
    # Factor-classification heads are explicitly outside the guide curriculum.
    assert option_outputs["action_type"].grad is None
    assert option_outputs["action_target"].grad is None
    assert option_outputs["action_resource"].grad is None


def test_strategic_batch_loss_is_guide_target_permutation_invariant() -> None:
    torch.manual_seed(20260730)
    model = _future_model()

    def compute(guide_target_index: int):
        model.zero_grad(set_to_none=True)
        loss, metrics = batch_losses(
            model,
            [_sequence(guide_target_index)],
            aux_weight=0.0,
            opp_hand_weight=0.0,
            opp_remainder_weight=0.0,
            alakazam_guide_weight=0.05,
            current_deck_guide_training_mode=(
                GUIDE_TRAINING_MODE_STRATEGIC
            ),
            setup_board_outcome_loss_weight=0.025,
            expanded_head_weights={},
        )
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), metrics, gradients

    first_loss, first_metrics, first_gradients = compute(0)
    second_loss, second_metrics, second_gradients = compute(1)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    assert first_metrics.guide_strategic_curriculum_loss > 0.0
    assert first_metrics.setup_board_outcome_loss > 0.0
    assert second_metrics.alakazam_guide_loss == pytest.approx(
        first_metrics.alakazam_guide_loss
    )
    assert set(first_gradients) == set(second_gradients)
    for name in first_gradients:
        torch.testing.assert_close(
            first_gradients[name],
            second_gradients[name],
            rtol=0,
            atol=0,
        )


def test_resident_strategic_loss_is_guide_target_permutation_invariant() -> None:
    torch.manual_seed(20260730)
    model = _future_model()

    def compute(guide_target_index: int):
        corpus = DeviceResidentBootstrapCorpus.from_splits(
            [_sequence(guide_target_index)],
            [],
            device=torch.device("cpu"),
        )
        model.zero_grad(set_to_none=True)
        loss, metrics = device_temporal_batch_losses(
            model,
            corpus,
            torch.tensor([0], dtype=torch.long),
            alakazam_guide_weight=0.05,
            current_deck_guide_training_mode=(
                GUIDE_TRAINING_MODE_STRATEGIC
            ),
            setup_board_outcome_loss_weight=0.025,
            expanded_head_weights={},
        )
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), metrics, gradients

    first_loss, first_metrics, first_gradients = compute(0)
    second_loss, second_metrics, second_gradients = compute(1)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    assert first_metrics.guide_strategic_curriculum_loss > 0.0
    assert first_metrics.setup_board_outcome_loss > 0.0
    assert second_metrics.alakazam_guide_loss == pytest.approx(
        first_metrics.alakazam_guide_loss
    )
    assert set(first_gradients) == set(second_gradients)
    for name in first_gradients:
        torch.testing.assert_close(
            first_gradients[name],
            second_gradients[name],
            rtol=0,
            atol=0,
        )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_pure_rl_parser_accepts_receipt_bound_strategic_mode(
    tmp_path: Path,
) -> None:
    specialist = "archaludon-ex"
    role_map = tmp_path / "roles.json"
    role_map.write_text(
        json.dumps(
            {
                "schema": "poke_bot.future_specialist_strategic_head_roles/v1",
                "specialist_id": specialist,
            }
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "curriculum.json"
    spec.write_text(
        json.dumps(
            {
                "schema": "poke_bot.future_specialist_strategic_curriculum/v1",
                "specialist_id": specialist,
                "training_mode": GUIDE_TRAINING_MODE_STRATEGIC,
                "direct_policy_cross_entropy_allowed": False,
                "replace_observed_outcome_targets_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.future_specialist_strategic_"
                    "curriculum_validation/v1"
                ),
                "status": "validated",
                "specialist_id": specialist,
                "training_mode": GUIDE_TRAINING_MODE_STRATEGIC,
                "curriculum_spec_sha256": _sha256(spec),
                "head_role_map_sha256": _sha256(role_map),
                "checks": {
                    "guide_supervision_terminates_at_strategic_heads": True,
                    "direct_policy_cross_entropy_absent": True,
                    "observed_outcome_targets_not_replaced": True,
                    "all_training_paths_use_the_declared_mode": True,
                },
            }
        ),
        encoding="utf-8",
    )

    parsed = _parse_args(
        [
            "--run-name",
            "future-strategic",
            "--mode",
            "specialist",
            "--specialist-archetype",
            specialist,
            "--current-deck-guide-training-mode",
            GUIDE_TRAINING_MODE_STRATEGIC,
            "--current-deck-guide-curriculum-spec",
            str(spec),
            "--current-deck-guide-head-role-map",
            str(role_map),
            "--current-deck-guide-curriculum-validation-receipt",
            str(receipt),
        ]
    )
    assert parsed.current_deck_guide_training_mode == (
        GUIDE_TRAINING_MODE_STRATEGIC
    )
    assert parsed.setup_board_outcome_loss_weight == pytest.approx(0.025)
