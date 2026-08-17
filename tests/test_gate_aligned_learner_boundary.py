from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import apply_gate_aligned_learner_at_boundary as boundary


def _receipt() -> dict:
    return {
        "boundary_next_iteration": 15,
        "changed_paths": [
            "learner.exact_gate_regression_margin",
            "learner.exact_gate_regression_patience",
            "collection.behavior_policy",
            "collection.research_control_phase.registry.path",
            "gates.active_contract.path",
            "source.source_tree_sha256",
        ],
        "current_contract": {
            "learner": {
                "exact_gate_regression_margin": 0.01,
                "exact_gate_regression_patience": 2,
            },
            "collection": {
                "behavior_policy": (
                    "gate_aligned_continuous_learner_with_exact_regression_rollback_v3"
                )
            },
        },
    }


def test_validate_receipt_accepts_only_gate_aligned_operational_delta(
    tmp_path: Path,
) -> None:
    path = tmp_path / "migration.json"
    path.write_text(json.dumps(_receipt()), encoding="utf-8")
    assert boundary.validate_receipt(path, target_next_iteration=15) == _receipt()


def test_validate_receipt_rejects_unrelated_design_change(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt["changed_paths"].append("learner.profile.hidden_dim")
    path = tmp_path / "migration.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected design paths"):
        boundary.validate_receipt(path, target_next_iteration=15)


def test_make_dropin_steady_revokes_authority_and_restores_stop_protection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reason = "gate-aligned-exact-regression-rollback-v18"
    path = tmp_path / "active.conf"
    path.write_text(
        "[Unit]\nRefuseManualStop=no\n[Service]\n"
        "ExecStart=/bin/tool --allow-clean-boundary-design-migration "
        f"--boundary-design-migration-reason {reason} --resume auto\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(boundary, "_unlock", lambda _path: None)
    monkeypatch.setattr(boundary, "_lock", lambda _path: None)

    boundary.make_dropin_steady(path, reason)

    text = path.read_text(encoding="utf-8")
    assert "RefuseManualStop=yes" in text
    assert "--allow-clean-boundary-design-migration" not in text
    assert "--boundary-design-migration-reason" not in text
    assert "--resume auto" in text
