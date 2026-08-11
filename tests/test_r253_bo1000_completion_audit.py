from __future__ import annotations

import json

import pytest

from poke_bot.r253_bo1000_completion_audit import (
    R253CompletionAuditError,
    build_completion_audit,
    create_once,
    render_markdown,
    summarize_decision_quality,
)


def _decision(*, changed: bool, direct: float, selected: float, rank: int):
    return {
        "mode": "shared_tree_mcts",
        "action_changed": changed,
        "meaningful_choice_change": changed,
        "direct_action_probability": direct,
        "mcts_action_direct_probability": selected,
        "direct_probability_gap": direct - selected,
        "mcts_action_direct_rank": rank,
        "selected_action_value": -0.25 if changed else 0.5,
        "selected_action_visits": 12 if changed else 18,
        "root_visits": 20,
        "legal_action_count": 4,
        "distinct_root_actions_visited": 3,
        "rollout_count": 20,
    }


def test_decision_quality_reports_material_high_confidence_overrides():
    games = [{"mcts_decisions": [
        _decision(changed=True, direct=0.91, selected=0.03, rank=3),
        _decision(changed=False, direct=0.80, selected=0.80, rank=1),
    ]}]
    result = summarize_decision_quality(games)
    assert result["searched_decisions"] == 2
    assert result["meaningful_changed_decisions"] == 1
    assert result["changed"]["direct_confidence_threshold_count"]["gte_0_90"] == 1
    assert result["changed"]["direct_probability_gap"]["median"] == pytest.approx(0.88)
    assert result["changed"]["selected_visit_fraction"]["mean"] == 0.6


def test_decision_quality_fails_closed_on_missing_counterfactual_field():
    row = _decision(changed=True, direct=0.9, selected=0.1, rank=2)
    del row["selected_action_value"]
    with pytest.raises(R253CompletionAuditError, match="counterfactual"):
        summarize_decision_quality([{"mcts_decisions": [row]}])


def test_decision_quality_fails_closed_on_inconsistent_counterfactual_math():
    row = _decision(changed=True, direct=0.9, selected=0.1, rank=2)
    row["direct_probability_gap"] = 0.7
    with pytest.raises(R253CompletionAuditError, match="inconsistent"):
        summarize_decision_quality([{"mcts_decisions": [row]}])


def test_completion_audit_rejects_nonfinal_run(tmp_path):
    (tmp_path / "run-identity.json").write_text(json.dumps({
        "serial_rollout_revision": 253
    }))
    (tmp_path / "final-review.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(R253CompletionAuditError, match="not complete"):
        build_completion_audit(tmp_path)


def test_create_once_is_immutable_and_review_binds_audit(tmp_path):
    target = tmp_path / "receipt.json"
    create_once(target, b"first\n")
    assert target.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError):
        create_once(target, b"second\n")
    review = render_markdown({
        "game_receipts_rollup_sha256": "sha256:games",
        "validated_summary": {
            "games": 1000, "pairs": 500,
            "outcomes": {"mcts_win": 510, "direct_win": 490},
            "mcts_win_rate_excluding_draws": 0.51,
            "mcts_win_rate_wilson_95": [0.479, 0.541],
            "decisions": {"seen_total": 100, "searched_total": 50,
                          "meaningful_choice_change_total": 10,
                          "meaningful_change_rate_per_searched": 0.2},
            "search": {"mean_mcts_to_direct_decision_latency_ratio": 5.0,
                       "latency_seconds": {"mean": 1.0}},
            "throughput": {"aggregate_worker_games_per_second": 0.1},
        },
        "decision_quality": {"changed": {"direct_confidence_threshold_count": {
            "gte_0_80": 4, "gte_0_90": 2
        }}},
    }, audit_sha256="sha256:audit")
    assert "sha256:audit" in review
    assert "MCTS wins: 510" in review
