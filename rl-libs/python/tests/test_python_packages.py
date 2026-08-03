from __future__ import annotations

import json
from pathlib import Path

from rl_eval import (
    AbortDecision,
    FieldReport,
    PromotionGateConfig,
    evaluate_aborts,
    evaluate_candidate_gate,
    next_iteration,
    wilson_lower,
)
from artifact_registry import ArtifactRegistry, retire_with_receipt


def test_wilson_and_field_report():
    assert 0.0 <= wilson_lower(5, 10) <= 1.0
    report = FieldReport(gate_threshold=0.0)
    for seat in (0, 1):
        report.merge_game("opp", our_seat=seat, winner=seat, is_mirror=False)
    assert "opp" in report.opponents_passing()


def test_promotion_and_aborts(tmp_path: Path):
    cfg = PromotionGateConfig(
        min_games=4, min_complete_pairs=2, threshold=0.0, bootstrap_resamples=200
    )
    rows = [
        {"valid": True, "candidate_seat": 0, "winner": 0},
        {"valid": True, "candidate_seat": 1, "winner": 1},
        {"valid": True, "candidate_seat": 0, "winner": 0},
        {"valid": True, "candidate_seat": 1, "winner": 1},
    ]
    gate = evaluate_candidate_gate(rows, cfg)
    assert gate["valid"] is True
    assert next_iteration({"last_completed_iteration": 3}) == 4
    d = evaluate_aborts(
        mean_advantages=[0.0, 0.0, 0.0],
        policy_prev_agreements=[0.99, 0.99, 0.99],
    )
    assert isinstance(d, AbortDecision)
    assert d.abort is True


def test_artifact_registry(tmp_path: Path):
    blob = tmp_path / "a.bin"
    blob.write_bytes(b"hello")
    reg = ArtifactRegistry(tmp_path / "reg.json")
    rec = reg.register("a", blob, kind="bin", meta={"k": 1})
    assert rec.digest.startswith("sha256:")
    assert reg.get("a") is not None
    receipt = tmp_path / "a.receipt.json"
    n = retire_with_receipt(blob, receipt, extra={"reason": "test"})
    assert n == 5
    assert not blob.exists()
    assert json.loads(receipt.read_text())["sha256"].startswith("sha256:")
