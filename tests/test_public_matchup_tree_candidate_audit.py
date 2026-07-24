from __future__ import annotations

import json

import pytest

from scripts.audit_public_matchup_tree_candidate import audit


def _candidate(tmp_path):
    targets = [f"deck-{index}" for index in range(22)]
    payload = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": False,
        "targets": targets,
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
    }
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_audit_preserves_thresholds_and_does_not_activate(tmp_path) -> None:
    path, payload = _candidate(tmp_path)
    payload["validation"]["classes"]["deck-3"]["weighted_support"] = 9_999
    path.write_text(json.dumps(payload))

    result = audit(path, minimum_precision=0.93, minimum_weighted_support=10_000)

    assert result["accepted_count"] == 21
    assert result["rejected_specialists"]["deck-3"] == ["support_below_floor"]
    assert result["runtime_enabled"] is False
    assert result["safe_to_activate_automatically"] is False


def test_audit_rejects_legacy_compressed_probability_contract(tmp_path) -> None:
    path, payload = _candidate(tmp_path)
    payload["calibration_contract"]["probability_columns"] = "compressed"
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="canonical calibration"):
        audit(path, minimum_precision=0.93, minimum_weighted_support=10_000)
