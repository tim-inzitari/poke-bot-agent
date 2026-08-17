from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.guide_weight_review import emit_review_request


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, completed_iteration: int = 5) -> dict[str, Path]:
    run = tmp_path / "run"
    (run / "commits").mkdir(parents=True)
    (run / "collection_receipts").mkdir()
    checkpoint = tmp_path / "seed.pt"
    checkpoint.write_bytes(b"seed")
    for iteration in range(1, 6):
        receipt = {
            "schema": "poke_bot.completed_collection/v1",
            "iteration": iteration,
            "checkpoint": str(checkpoint),
            "checkpoint_digest": _digest(checkpoint),
        }
        (run / "collection_receipts" / f"iter_{iteration:05d}.json").write_text(
            json.dumps(receipt)
        )
    commit = run / "commits" / f"iter_{completed_iteration:05d}.json"
    commit.write_text(
        json.dumps(
            {
                "mode": "specialist",
                "last_completed_iteration": completed_iteration,
                "next_iteration": completed_iteration + 1,
            }
        )
    )
    guide = tmp_path / "guide.yaml"
    guide.write_text("version: guide-v1\n")
    return {"run": run, "commit": commit, "guide": guide}


def test_emits_every_fifth_committed_guide_review(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = emit_review_request(
        run_dir=fixture["run"],
        specialist_id="archaludon-ex",
        completed_iteration=5,
        current_weight=0.35,
        iteration_commit=fixture["commit"],
        guide_contract=fixture["guide"],
        guide_version="guide-v1",
        prospective_policy_revision=44,
        learning_semantics_revision=46,
        consecutive_nonpositive_evaluations=0,
    )
    assert output is not None
    request = json.loads(output.read_text())
    assert request["owner_decision_revision"] == 43
    assert request["shadow_pair"]["guide_on_weight"] == 0.35
    assert request["shadow_pair"]["guide_off_weight"] == 0.0
    assert request["earliest_activation_boundary_next_iteration"] == 6
    assert request["application_boundary"] == (
        "first_available_future_five_iteration_hard_pause"
    )
    assert "activation_boundary_next_iteration" not in request
    assert len(request["review_window"]["collection_receipts"]) == 5
    assert request["evaluation"]["training_eligible"] is False
    assert request["evaluation"]["formal_gate"] is False
    assert request["weight_change_allowed_without_compiled_schedule"] is False
    assert emit_review_request(
        run_dir=fixture["run"],
        specialist_id="archaludon-ex",
        completed_iteration=5,
        current_weight=0.35,
        iteration_commit=fixture["commit"],
        guide_contract=fixture["guide"],
        guide_version="guide-v1",
        prospective_policy_revision=44,
        learning_semantics_revision=46,
        consecutive_nonpositive_evaluations=0,
    ) == output


def test_skips_nonreview_and_zero_weight_boundaries(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    assert emit_review_request(
        run_dir=fixture["run"],
        specialist_id="archaludon-ex",
        completed_iteration=4,
        current_weight=0.25,
        iteration_commit=fixture["commit"],
        guide_contract=fixture["guide"],
        guide_version="guide-v1",
        prospective_policy_revision=44,
        learning_semantics_revision=46,
        consecutive_nonpositive_evaluations=0,
    ) is None
    assert emit_review_request(
        run_dir=fixture["run"],
        specialist_id="archaludon-ex",
        completed_iteration=5,
        current_weight=0.0,
        iteration_commit=fixture["commit"],
        guide_contract=fixture["guide"],
        guide_version="guide-v1",
        prospective_policy_revision=44,
        learning_semantics_revision=46,
        consecutive_nonpositive_evaluations=0,
    ) is None


def test_rejects_uncommitted_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["commit"].write_text(
        json.dumps(
            {
                "mode": "specialist",
                "last_completed_iteration": 4,
                "next_iteration": 5,
            }
        )
    )
    with pytest.raises(RuntimeError, match="immutable specialist commit"):
        emit_review_request(
            run_dir=fixture["run"],
            specialist_id="archaludon-ex",
            completed_iteration=5,
            current_weight=0.25,
            iteration_commit=fixture["commit"],
            guide_contract=fixture["guide"],
            guide_version="guide-v1",
            prospective_policy_revision=44,
            learning_semantics_revision=46,
            consecutive_nonpositive_evaluations=0,
        )


def test_rejects_nonfuture_or_wrong_learning_policy(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="review identity"):
        emit_review_request(
            run_dir=fixture["run"],
            specialist_id="teal-mask-ogerpon-ex",
            completed_iteration=5,
            current_weight=0.25,
            iteration_commit=fixture["commit"],
            guide_contract=fixture["guide"],
            guide_version="guide-v1",
            prospective_policy_revision=0,
            learning_semantics_revision=0,
            consecutive_nonpositive_evaluations=0,
        )
