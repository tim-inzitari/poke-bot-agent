from __future__ import annotations

import json

import pytest

from scripts.audit_public_matchup_tree_candidate import audit


def _candidate(tmp_path):
    targets = [f"deck-{index}" for index in range(22)]
    unknown_counts = [0.0] * (len(targets) + 1)
    unknown_counts[-1] = 1.0
    payload = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": False,
        "targets": targets,
        "input_contract": {
            "empty_public_state_exact_bypass": True,
        },
        "calibration_contract": {
            "probability_columns": "expanded_to_canonical_class_indexes",
            "canonical_class_count": 23,
        },
        "validation": {
            "classes": {
                target: {"weighted_support": 12_000}
                for target in targets
            }
        },
        "runtime_calibration": {
            "per_archetype": {
                target: {
                    "available": True,
                    "precision": 0.95,
                    "recall": 0.75,
                    "min_leaf_confidence": 0.8,
                }
                for target in targets
            }
        },
        "tree": {
            "class_names": [*targets, "unknown"],
            "children_left": [-1],
            "children_right": [-1],
            "feature_card_id": [-2],
            "threshold": [-2.0],
            "weighted_class_counts": [unknown_counts],
            "node_count": 1,
        },
    }
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_audit_preserves_thresholds_and_does_not_activate(tmp_path) -> None:
    path, payload = _candidate(tmp_path)
    payload["validation"]["classes"]["deck-3"]["weighted_support"] = 9_999
    path.write_text(json.dumps(payload))

    result = audit(
        path,
        minimum_precision=0.93,
        minimum_weighted_support=10_000,
        expected_targets=payload["targets"],
    )

    assert result["accepted_count"] == 21
    assert result["rejected_specialists"]["deck-3"] == ["support_below_floor"]
    assert result["runtime_enabled"] is False
    assert result["safe_to_activate_automatically"] is False
    assert result["empty_public_state"] == {
        "declared_exact_bypass": True,
        "verified_exact_bypass_after_calibration": True,
        "tree_leaf": 0,
        "raw_prediction": "unknown",
        "raw_confidence": 1.0,
        "calibrated_min_leaf_confidence": None,
        "calibrated_route": None,
    }


def test_audit_rejects_legacy_compressed_probability_contract(tmp_path) -> None:
    path, payload = _candidate(tmp_path)
    payload["calibration_contract"]["probability_columns"] = "compressed"
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="canonical calibration"):
        audit(path, minimum_precision=0.93, minimum_weighted_support=10_000)


def test_audit_requires_declared_empty_public_state_exact_bypass(
    tmp_path,
) -> None:
    path, payload = _candidate(tmp_path)
    payload["input_contract"].pop("empty_public_state_exact_bypass")
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="exact-bypass declaration"):
        audit(
            path,
            minimum_precision=0.93,
            minimum_weighted_support=10_000,
            expected_targets=payload["targets"],
        )


def test_audit_rejects_empty_public_state_route_after_calibration(
    tmp_path,
) -> None:
    path, payload = _candidate(tmp_path)
    routed_counts = [0.0] * 23
    routed_counts[0] = 9.0
    routed_counts[-1] = 1.0
    unknown_counts = [0.0] * 23
    unknown_counts[-1] = 1.0
    payload["tree"] = {
        "class_names": [*payload["targets"], "unknown"],
        "children_left": [1, -1, -1],
        "children_right": [2, -1, -1],
        "feature_card_id": [42, -2, -2],
        "threshold": [0.5, -2.0, -2.0],
        "weighted_class_counts": [
            unknown_counts,
            routed_counts,
            unknown_counts,
        ],
        "node_count": 3,
    }
    path.write_text(json.dumps(payload))

    with pytest.raises(
        RuntimeError,
        match="empty public state routes after calibration: deck-0",
    ):
        audit(
            path,
            minimum_precision=0.93,
            minimum_weighted_support=10_000,
            expected_targets=payload["targets"],
        )


def test_audit_accepts_raw_empty_prediction_below_calibrated_threshold(
    tmp_path,
) -> None:
    path, payload = _candidate(tmp_path)
    counts = [0.0] * 23
    counts[0] = 3.0
    counts[-1] = 1.0
    payload["tree"]["weighted_class_counts"] = [counts]
    path.write_text(json.dumps(payload))

    result = audit(
        path,
        minimum_precision=0.93,
        minimum_weighted_support=10_000,
        expected_targets=payload["targets"],
    )

    assert result["empty_public_state"]["raw_prediction"] == "deck-0"
    assert result["empty_public_state"]["raw_confidence"] == 0.75
    assert (
        result["empty_public_state"]["calibrated_min_leaf_confidence"]
        == 0.8
    )
    assert result["empty_public_state"]["calibrated_route"] is None


def test_audit_accepts_an_appended_v6_logical_route(tmp_path) -> None:
    path, payload = _candidate(tmp_path)
    payload["targets"] = payload["targets"][:18] + ["teal-mask-ogerpon-ex"]
    payload["calibration_contract"]["canonical_class_count"] = 20
    payload["validation"]["classes"] = {
        target: {"weighted_support": 12_000}
        for target in payload["targets"]
    }
    payload["runtime_calibration"]["per_archetype"] = {
        target: {
            "available": True,
            "precision": 0.95,
            "recall": 0.75,
            "min_leaf_confidence": 0.8,
        }
        for target in payload["targets"]
    }
    unknown_counts = [0.0] * 20
    unknown_counts[-1] = 1.0
    payload["tree"] = {
        "class_names": [*payload["targets"], "unknown"],
        "children_left": [-1],
        "children_right": [-1],
        "feature_card_id": [-2],
        "threshold": [-2.0],
        "weighted_class_counts": [unknown_counts],
        "node_count": 1,
    }
    path.write_text(json.dumps(payload))

    result = audit(
        path,
        minimum_precision=0.93,
        minimum_weighted_support=10_000,
        expected_targets=payload["targets"],
    )

    assert result["target_count"] == 19
    assert result["accepted_count"] == 19
    assert result["canonical_target_ids"][-1] == "teal-mask-ogerpon-ex"
