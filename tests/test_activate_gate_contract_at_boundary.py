from pathlib import Path

import pytest

from scripts.activate_gate_contract_at_boundary import (
    validate_boundary_commit,
    validate_staged_gate,
)


def test_staged_gate_activates_lc50_after_iteration_4() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = validate_staged_gate(
        root / "ops/alakazam_gate_program_v1.json",
        activate_after=4,
    )
    assert contract["fallback_transition"]["id"].endswith(
        "lc50-at-iter5-v1+frozen-specialists-r4"
    )


def test_exact_iteration_commit_is_required() -> None:
    row = {
        "iteration": 4,
        "completed": True,
        "candidate": {"path": "/candidate.pt", "digest": "sha256:candidate"},
        "learner_after": {"path": "/learner.pt", "digest": "sha256:learner"},
        "active_gate_result": {"iteration": 4, "gate_id": "lc55", "passed": False},
    }
    commit = {
        "last_completed_iteration": 4,
        "next_iteration": 5,
        "history": [row],
    }
    assert validate_boundary_commit(commit, expected_iteration=4) == row


def test_incomplete_or_wrong_next_iteration_commit_fails_closed() -> None:
    commit = {
        "last_completed_iteration": 4,
        "next_iteration": 6,
        "history": [],
    }
    with pytest.raises(RuntimeError, match="advance exactly one"):
        validate_boundary_commit(commit, expected_iteration=4)


def test_wrong_boundary_fails_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match="boundary-compatible"):
        validate_staged_gate(
            root / "ops/alakazam_gate_program_v1.json",
            activate_after=5,
        )
