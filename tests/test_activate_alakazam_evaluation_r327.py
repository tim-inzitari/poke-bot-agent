from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.alakazam_rule_derivative_evaluation_r327 import (
    build_revision25_formal_gate_contract,
)
from scripts.activate_alakazam_evaluation_r327 import (
    build_activation_receipt,
    write_json_create_only,
)


def _base_gate() -> dict[str, object]:
    return {
        "schema": "poke_bot.competition_gate_program/v1",
        "active_gate_id": "base-gate",
        "next_gate": {
            "id": "base-gate",
            "label": "Base gate",
            "exact_result_pointer": "/tmp/base-result.json",
            "evaluation": {
                "all_matchups_must_complete": True,
                "games_per_opponent": 250,
                "games_total": 500,
                "minimum_games_per_opponent": 250,
                "partial_results_gate_eligible": False,
                "seat0_games_per_opponent": 125,
                "seat1_games_per_opponent": 125,
                "sequential_early_stop": False,
            },
            "pass_criteria": {"audit_must_pass": True},
            "research_measurements": [],
            "roster": [
                {"opponent_id": "a", "weight": 1.0},
                {"opponent_id": "b", "weight": 2.0},
            ],
        },
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def test_build_receipt_uses_commit_next_iteration(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    formal = tmp_path / "formal.json"
    commit = tmp_path / "iter_00001.json"
    _write(base, _base_gate())
    _write(
        formal,
        build_revision25_formal_gate_contract(
            _base_gate(), exact_result_pointer="/tmp/r327-result.json"
        ),
    )
    _write(commit, {"next_iteration": 2})
    receipt = build_activation_receipt(
        base_contract=base,
        formal_contract=formal,
        boundary_commit=commit,
        run_name="alakazam_rule_derivative_g5_r12",
        elmo_endpoint="elmo:8765",
    )
    assert receipt["boundary_commit_iteration"] == 1
    assert receipt["first_formal_holdout_iteration"] == 2


def test_create_only_receipt_refuses_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    write_json_create_only(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    with pytest.raises(FileExistsError):
        write_json_create_only(path, {"ok": False})
