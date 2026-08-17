from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from poke_bot.recursive_turn_planner.neural_leaf_reranker import FrozenPolicyIdentity
from poke_bot.recursive_turn_planner.r207_frozen_leaf_calibration import (
    R195_CHECKPOINT_BYTES,
    R195_CHECKPOINT_SHA256,
    R207_CALIBRATION_RECEIPT_SCHEMA,
    R207_FROZEN_PREDICTIONS_SCHEMA,
    R207_HELDOUT_EVIDENCE_SCHEMA,
    R207_TRAINING_SOURCE_MANIFEST_SCHEMA,
    FrozenLeafCalibrationError,
    FrozenLeafCalibrationPolicy,
    canonical_sha256,
    compile_r207_frozen_leaf_calibration_preflight,
    leaf_score_calibration_from_r207_receipt,
    verify_r207_frozen_leaf_calibration_receipt,
)


def _sha256(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _write_sealed_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        os.chmod(path, 0o644)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    os.chmod(path, 0o444)


def _file_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _policy(**overrides: Any) -> FrozenLeafCalibrationPolicy:
    payload: dict[str, Any] = {
        "min_terminal_leaves": 6,
        "min_distinct_source_ids": 3,
        "min_distinct_game_ids": 6,
        "min_per_outcome": 2,
        "min_per_source_id": 2,
        "max_outcome_brier": 0.01,
        "max_outcome_ece": 0.08,
        "max_value_rmse": 0.15,
        "max_value_mae": 0.15,
        "value_weight": 0.4,
        "outcome_weight": 0.6,
        "uncertainty_quantile": 0.95,
        "ece_bins": 3,
    }
    payload.update(overrides)
    return FrozenLeafCalibrationPolicy(**payload)


def _training_manifest() -> dict[str, Any]:
    training_source_ids = ["training-a", "training-b"]
    training_game_ids = ["training-game-a", "training-game-b"]
    return {
        "schema": R207_TRAINING_SOURCE_MANIFEST_SCHEMA,
        "status": "sealed",
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "checkpoint_bytes": R195_CHECKPOINT_BYTES,
        "source_role": "r195_training",
        "training_eligible": True,
        "attempt10_rtp_partial_rows_used": False,
        "r197_or_r198_rtp_partial_rows_used": False,
        "r195_rtp_sidecar_used": False,
        "r197_rtp_sidecar_used": False,
        "training_source_ids": training_source_ids,
        "training_source_ids_sha256": canonical_sha256(sorted(training_source_ids)),
        "training_game_ids": training_game_ids,
        "training_game_ids_sha256": canonical_sha256(sorted(training_game_ids)),
    }


def _source_exclusion(training_manifest_sha256: str) -> dict[str, Any]:
    heldout_source_ids = ["heldout-a", "heldout-b", "heldout-c"]
    heldout_game_ids = [
        "heldout-game-01",
        "heldout-game-02",
        "heldout-game-03",
        "heldout-game-04",
        "heldout-game-05",
        "heldout-game-06",
    ]
    return {
        "source_disjoint": True,
        "game_disjoint": True,
        "r195_training_source_manifest_sha256": training_manifest_sha256,
        "training_source_ids_sha256": canonical_sha256(["training-a", "training-b"]),
        "training_game_ids_sha256": canonical_sha256(
            ["training-game-a", "training-game-b"]
        ),
        "heldout_source_ids": heldout_source_ids,
        "heldout_game_ids": heldout_game_ids,
        "heldout_source_ids_sha256": canonical_sha256(sorted(heldout_source_ids)),
        "heldout_game_ids_sha256": canonical_sha256(sorted(heldout_game_ids)),
        "intersection_source_ids_sha256": canonical_sha256([]),
        "intersection_source_id_count": 0,
        "intersection_game_ids_sha256": canonical_sha256([]),
        "intersection_game_id_count": 0,
    }


def _evidence(training_manifest_sha256: str) -> dict[str, Any]:
    leaves = (
        ("leaf-01", "heldout-a", "heldout-game-01", "loss"),
        ("leaf-02", "heldout-a", "heldout-game-02", "win"),
        ("leaf-03", "heldout-b", "heldout-game-03", "draw"),
        ("leaf-04", "heldout-b", "heldout-game-04", "loss"),
        ("leaf-05", "heldout-c", "heldout-game-05", "win"),
        ("leaf-06", "heldout-c", "heldout-game-06", "draw"),
    )
    return {
        "schema": R207_HELDOUT_EVIDENCE_SCHEMA,
        "status": "sealed",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "evidence_role": "r207_source_excluded_terminal_leaf_calibration",
        "attempt10_rtp_partial_rows_used": False,
        "r197_or_r198_rtp_partial_rows_used": False,
        "r195_rtp_sidecar_used": False,
        "r197_rtp_sidecar_used": False,
        "outcome_perspective": "candidate_root",
        "terminal_outcomes_exact": True,
        "nonterminal_policy_visible_leaf_predictions_only": True,
        "source_exclusion": _source_exclusion(training_manifest_sha256),
        "terminal_leaf_count": len(leaves),
        "terminal_leaves": [
            {
                "leaf_id": leaf_id,
                "source_id": source_id,
                "game_id": game_id,
                "decision_ordinal": 1,
                "root_seat": 0,
                "horizon_atomic_actions": 2,
                "leaf_kind": "successor",
                "policy_visible": True,
                "is_terminal": False,
                "public_observation_sha256": _sha256(f"observation:{leaf_id}"),
                "legal_actions_sha256": _sha256(f"legal-actions:{leaf_id}"),
                "nonterminal_leaf_state_sha256": _sha256(f"leaf-state:{leaf_id}"),
                "terminal_outcome": outcome,
                "terminal_result_kind": "exact_simulator_terminal",
                "terminal_result_sha256": _sha256(f"terminal:{leaf_id}"),
            }
            for leaf_id, source_id, game_id, outcome in leaves
        ],
    }


def _prediction_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    probabilities = {
        "leaf-01": ({"loss": 0.96, "draw": 0.03, "win": 0.01}, -0.92),
        "leaf-02": ({"loss": 0.01, "draw": 0.03, "win": 0.96}, 0.91),
        "leaf-03": ({"loss": 0.03, "draw": 0.94, "win": 0.03}, 0.04),
        "leaf-04": ({"loss": 0.95, "draw": 0.03, "win": 0.02}, -0.90),
        "leaf-05": ({"loss": 0.02, "draw": 0.03, "win": 0.95}, 0.93),
        "leaf-06": ({"loss": 0.02, "draw": 0.95, "win": 0.03}, 0.02),
    }
    return [
        {
            "leaf_id": leaf["leaf_id"],
            "source_id": leaf["source_id"],
            "game_id": leaf["game_id"],
            "decision_ordinal": leaf["decision_ordinal"],
            "root_seat": leaf["root_seat"],
            "horizon_atomic_actions": leaf["horizon_atomic_actions"],
            "leaf_kind": leaf["leaf_kind"],
            "public_observation_sha256": leaf["public_observation_sha256"],
            "legal_actions_sha256": leaf["legal_actions_sha256"],
            "nonterminal_leaf_state_sha256": leaf["nonterminal_leaf_state_sha256"],
            "terminal_result_sha256": leaf["terminal_result_sha256"],
            "outcome_probabilities": probabilities[leaf["leaf_id"]][0],
            "value": probabilities[leaf["leaf_id"]][1],
        }
        for leaf in evidence["terminal_leaves"]
    ]


def _predictions(evidence: dict[str, Any], evidence_sha256: str) -> dict[str, Any]:
    identity = FrozenPolicyIdentity.r205_no_rtp()
    return {
        "schema": R207_FROZEN_PREDICTIONS_SCHEMA,
        "status": "frozen_inference_complete",
        "outcome_perspective": "candidate_root",
        "outcome_class_order": ["loss", "draw", "win"],
        "model_role": "r195_no_rtp_frozen_policy_and_leaf_model",
        "frozen_policy_identity_sha256": identity.identity_sha256,
        "r195_no_rtp_bundle_sha256": identity.bundle_sha256,
        "r195_no_rtp_deck_cards_sha256": identity.deck_cards_sha256,
        "r195_no_rtp_runtime_sha256": identity.nonplanner_runtime_sha256,
        "attempt10_rtp_partial_rows_used": False,
        "r197_or_r198_rtp_partial_rows_used": False,
        "r195_rtp_sidecar_used": False,
        "r197_rtp_sidecar_used": False,
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "checkpoint_bytes": R195_CHECKPOINT_BYTES,
        "heldout_evidence_sha256": evidence_sha256,
        "model_frozen": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "gradient_updates": 0,
        "training_eligible": False,
        "predictions": _prediction_rows(evidence),
    }


def _sealed_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    training_manifest_path = tmp_path / "r195-training-sources.json"
    training_manifest = _training_manifest()
    _write_sealed_json(training_manifest_path, training_manifest)
    evidence_path = tmp_path / "heldout-evidence.json"
    evidence = _evidence(_file_sha256(training_manifest_path))
    _write_sealed_json(evidence_path, evidence)
    predictions = _predictions(evidence, _file_sha256(evidence_path))
    predictions_path = tmp_path / "frozen-predictions.json"
    _write_sealed_json(predictions_path, predictions)
    return (
        training_manifest_path,
        evidence_path,
        predictions_path,
        training_manifest,
        evidence,
        predictions,
    )


def test_compiler_binds_exact_terminal_labels_and_emits_reranker_bounds(
    tmp_path: Path,
) -> None:
    training_manifest_path, evidence_path, predictions_path, _, _, _ = _sealed_inputs(tmp_path)

    receipt = compile_r207_frozen_leaf_calibration_preflight(
        evidence_path,
        predictions_path,
        training_manifest_path,
        policy=_policy(),
    )
    verified = verify_r207_frozen_leaf_calibration_receipt(
        receipt,
        heldout_evidence_path=evidence_path,
        frozen_predictions_path=predictions_path,
        r195_training_source_manifest_path=training_manifest_path,
    )

    assert verified == receipt
    assert receipt["schema"] == R207_CALIBRATION_RECEIPT_SCHEMA
    assert receipt["checkpoint"]["sha256"] == R195_CHECKPOINT_SHA256
    assert receipt["checkpoint"]["bytes"] == R195_CHECKPOINT_BYTES
    assert receipt["inputs"]["heldout_evidence"]["sha256"] == _file_sha256(evidence_path)
    assert receipt["inputs"]["frozen_predictions"]["sha256"] == _file_sha256(
        predictions_path
    )
    assert receipt["inputs"]["heldout_evidence"]["mode"] == "0444"
    assert receipt["source_exclusion"]["source_disjoint"] is True
    assert receipt["source_exclusion"]["intersection_source_id_count"] == 0
    assert receipt["support"]["terminal_leaf_count"] == 6
    assert receipt["support"]["per_outcome"] == {"draw": 2, "loss": 2, "win": 2}
    assert receipt["calibration"]["outcome_multiclass_brier"] < 0.01
    assert receipt["calibration"]["outcome_classwise_ece"] < 0.08
    assert receipt["reranker_bounds"]["simulator_terminal_results_must_remain_exact"] is True
    assert receipt["frozen_inference"]["terminal_exact_results_model_reranked"] is False
    assert receipt["authority"] == {
        "frozen_nonterminal_leaf_reranking_preflight_eligible": True,
        "training_authorized": False,
        "gradient_or_optimizer_update_authorized": False,
        "serving_authorized": False,
        "action_authority_enabled": False,
        "selector_change_authorized": False,
        "promotion_authorized": False,
    }


def test_compiler_rejects_wrong_checkpoint_or_terminal_identity(tmp_path: Path) -> None:
    training_manifest_path, evidence_path, predictions_path, _, evidence, predictions = (
        _sealed_inputs(tmp_path)
    )
    predictions["checkpoint_sha256"] = _sha256("wrong-checkpoint")
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="exact r195 checkpoint"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    predictions["checkpoint_sha256"] = R195_CHECKPOINT_SHA256
    predictions["model_role"] = "r195_rtp_sidecar_model"
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="exact r195 NO-RTP model role"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    predictions["model_role"] = "r195_no_rtp_frozen_policy_and_leaf_model"
    predictions["predictions"][0]["terminal_result_sha256"] = _sha256("wrong-terminal")
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="terminal result identity"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    assert evidence["terminal_leaves"][0]["terminal_result_kind"] == "exact_simulator_terminal"


def test_compiler_rejects_source_overlap_and_training_metadata(tmp_path: Path) -> None:
    training_manifest_path, evidence_path, predictions_path, _, evidence, predictions = (
        _sealed_inputs(tmp_path)
    )
    evidence["source_exclusion"]["heldout_source_ids"] = [
        "heldout-a",
        "heldout-b",
        "heldout-c",
        "training-a",
    ]
    evidence["source_exclusion"]["heldout_source_ids_sha256"] = canonical_sha256(
        sorted(evidence["source_exclusion"]["heldout_source_ids"])
    )
    _write_sealed_json(evidence_path, evidence)

    with pytest.raises(FrozenLeafCalibrationError, match="overlap"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    evidence = _evidence(_file_sha256(training_manifest_path))
    _write_sealed_json(evidence_path, evidence)
    predictions = _predictions(evidence, _file_sha256(evidence_path))
    predictions["optimizer_steps"] = True
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="exact integer"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )


def test_compiler_rejects_attempt10_rtp_rows_or_either_rtp_sidecar(
    tmp_path: Path,
) -> None:
    training_manifest_path, evidence_path, predictions_path, _, evidence, predictions = (
        _sealed_inputs(tmp_path)
    )
    evidence["attempt10_rtp_partial_rows_used"] = True
    _write_sealed_json(evidence_path, evidence)

    with pytest.raises(FrozenLeafCalibrationError, match="attempt10_rtp_partial_rows_used"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    evidence = _evidence(_file_sha256(training_manifest_path))
    _write_sealed_json(evidence_path, evidence)
    predictions = _predictions(evidence, _file_sha256(evidence_path))
    predictions["r195_rtp_sidecar_used"] = True
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="r195_rtp_sidecar_used"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    predictions["r195_rtp_sidecar_used"] = False
    predictions["r197_rtp_sidecar_used"] = True
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="r197_rtp_sidecar_used"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )


def test_compiler_fails_closed_for_bad_calibration_or_insufficient_support(
    tmp_path: Path,
) -> None:
    training_manifest_path, evidence_path, predictions_path, _, evidence, predictions = (
        _sealed_inputs(tmp_path)
    )
    predictions["predictions"][0]["outcome_probabilities"] = {
        "loss": 0.01,
        "draw": 0.03,
        "win": 0.96,
    }
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="outcome_multiclass_brier"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    evidence = _evidence(_file_sha256(training_manifest_path))
    _write_sealed_json(evidence_path, evidence)
    predictions = _predictions(evidence, _file_sha256(evidence_path))
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="terminal_leaf_support"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path,
            predictions_path,
            training_manifest_path,
            policy=_policy(min_terminal_leaves=7),
        )


def test_compiler_requires_unique_nonterminal_policy_visible_game_leaves(
    tmp_path: Path,
) -> None:
    training_manifest_path, evidence_path, predictions_path, _, evidence, predictions = (
        _sealed_inputs(tmp_path)
    )
    evidence["terminal_leaves"][0]["is_terminal"] = True
    _write_sealed_json(evidence_path, evidence)

    with pytest.raises(FrozenLeafCalibrationError, match="policy-visible nonterminal"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    evidence = _evidence(_file_sha256(training_manifest_path))
    evidence["terminal_leaves"][1]["game_id"] = evidence["terminal_leaves"][0]["game_id"]
    _write_sealed_json(evidence_path, evidence)

    with pytest.raises(FrozenLeafCalibrationError, match="exactly one calibration leaf"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    evidence = _evidence(_file_sha256(training_manifest_path))
    _write_sealed_json(evidence_path, evidence)
    predictions = _predictions(evidence, _file_sha256(evidence_path))
    predictions["predictions"][0]["public_observation_sha256"] = _sha256("wrong-observation")
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="public_observation_sha256"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    predictions = _predictions(evidence, _file_sha256(evidence_path))
    predictions["predictions"][0]["decision_ordinal"] = True
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="exact integer"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )


def test_compiler_rejects_nonimmutable_inputs_and_tampered_receipts(tmp_path: Path) -> None:
    training_manifest_path, evidence_path, predictions_path, _, _, _ = _sealed_inputs(tmp_path)
    os.chmod(evidence_path, 0o644)

    with pytest.raises(FrozenLeafCalibrationError, match="immutable mode 0444"):
        compile_r207_frozen_leaf_calibration_preflight(
            evidence_path, predictions_path, training_manifest_path, policy=_policy()
        )

    os.chmod(evidence_path, 0o444)
    receipt = compile_r207_frozen_leaf_calibration_preflight(
        evidence_path, predictions_path, training_manifest_path, policy=_policy()
    )
    tampered = copy.deepcopy(receipt)
    tampered["reranker_bounds"]["value_absolute_error_bound"] = 1.0

    with pytest.raises(FrozenLeafCalibrationError, match="receipt digest"):
        verify_r207_frozen_leaf_calibration_receipt(tampered)


def test_receipt_bridges_only_its_fixed_blend_to_leaf_score_calibration(
    tmp_path: Path,
) -> None:
    training_manifest_path, evidence_path, predictions_path, _, _, _ = _sealed_inputs(tmp_path)
    receipt = compile_r207_frozen_leaf_calibration_preflight(
        evidence_path, predictions_path, training_manifest_path, policy=_policy()
    )
    calibration = leaf_score_calibration_from_r207_receipt(
        receipt,
        heldout_evidence_path=evidence_path,
        frozen_predictions_path=predictions_path,
        r195_training_source_manifest_path=training_manifest_path,
    )

    assert calibration.receipt_sha256 == receipt["receipt_sha256"]
    assert calibration.frozen_policy_identity_sha256 == FrozenPolicyIdentity.r205_no_rtp().identity_sha256
    assert calibration.value_weight == pytest.approx(0.4)
    assert calibration.outcome_weight == pytest.approx(0.6)
    assert calibration.value_head_calibrated is True
    assert calibration.outcome_distribution_calibrated is True
    assert calibration.source_excluded is True
    assert calibration.serving_action_authority is False

    with pytest.raises(FrozenLeafCalibrationError, match="nonzero fixed value and outcome"):
        _policy(value_weight=1.0, outcome_weight=0.0)


def test_receipt_reverification_rejects_one_path_or_input_identity_drift(
    tmp_path: Path,
) -> None:
    training_manifest_path, evidence_path, predictions_path, _, evidence, predictions = (
        _sealed_inputs(tmp_path)
    )
    receipt = compile_r207_frozen_leaf_calibration_preflight(
        evidence_path, predictions_path, training_manifest_path, policy=_policy()
    )

    with pytest.raises(FrozenLeafCalibrationError, match="supplied together"):
        verify_r207_frozen_leaf_calibration_receipt(
            receipt,
            heldout_evidence_path=evidence_path,
        )

    predictions["predictions"][0]["value"] = -0.91
    _write_sealed_json(predictions_path, predictions)

    with pytest.raises(FrozenLeafCalibrationError, match="identity has drifted"):
        verify_r207_frozen_leaf_calibration_receipt(
            receipt,
            heldout_evidence_path=evidence_path,
            frozen_predictions_path=predictions_path,
            r195_training_source_manifest_path=training_manifest_path,
        )

    assert evidence["terminal_outcomes_exact"] is True
