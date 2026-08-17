from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.alakazam_rule_derivative_evaluation_r327 import (
    CONTRACT_SHA256,
    GOAL_SHA256,
    MIGRATION_SCHEMA,
    Revision25EvaluationError,
    build_revision25_formal_gate_contract,
    sha256_file,
    validate_revision25_activation_receipt,
    validate_revision25_formal_gate_contract,
)


def _base_gate() -> dict[str, object]:
    roster = [
        {
            "opponent_id": f"opp-{index}",
            "content_digest": "sha256:" + f"{index + 1:064x}",
            "tier": "S" if index == 0 else "A",
            "weight": 2.0 if index == 0 else 1.0,
        }
        for index in range(18)
    ]
    return {
        "schema": "poke_bot.competition_gate_program/v1",
        "active_gate_id": "base-gate",
        "owner_decision_revision": 192,
        "next_gate": {
            "id": "base-gate",
            "label": "Base gate",
            "exact_result_pointer": "/tmp/base-result.json",
            "evaluation": {
                "all_matchups_must_complete": True,
                "checkpoint_digest_required": True,
                "confidence_level": 0.9,
                "confidence_method": "matched-seat cluster nonparametric bootstrap",
                "fixed_seed_manifest_required": True,
                "formal_eval_disjoint_from_training": True,
                "games_per_opponent": 250,
                "games_total": 4500,
                "minimum_games_per_opponent": 250,
                "mode": "greedy",
                "package_digest_deduplicated": True,
                "partial_results_gate_eligible": False,
                "seat0_games_per_opponent": 125,
                "seat1_games_per_opponent": 125,
                "sequential_early_stop": False,
            },
            "pass_criteria": {
                "accepted_official_holdout_non_regression": 0.6,
                "audit_must_pass": True,
                "individual_opponent_floor": 0.15,
                "skill_weighted_win_rate": 0.75,
            },
            "research_measurements": [
                {
                    "opponent_id": "research-only",
                    "games": 250,
                    "seat0_games": 125,
                    "seat1_games": 125,
                    "gate_weight": 0.0,
                    "included_in_gate_pass": False,
                }
            ],
            "roster": roster,
            "status": "queued",
        },
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def test_build_preserves_roster_weights_and_thresholds(tmp_path: Path) -> None:
    base = _base_gate()
    formal = build_revision25_formal_gate_contract(
        base,
        exact_result_pointer="/tmp/r327-result.json",
    )
    gate = formal["next_gate"]
    assert gate["roster"] == base["next_gate"]["roster"]
    assert gate["pass_criteria"] == base["next_gate"]["pass_criteria"]
    assert gate["research_measurements"] == base["next_gate"]["research_measurements"]
    assert gate["evaluation"]["games_total"] == 900
    assert gate["evaluation"]["games_per_opponent"] == 50
    assert gate["evaluation"]["seat0_games_per_opponent"] == 25
    assert gate["evaluation"]["seat1_games_per_opponent"] == 25
    path = tmp_path / "formal.json"
    _write(path, formal)
    assert json.loads(path.read_text()) == formal


def test_changed_roster_or_threshold_fails() -> None:
    base = _base_gate()
    formal = build_revision25_formal_gate_contract(
        base,
        exact_result_pointer="/tmp/r327-result.json",
    )
    formal["next_gate"]["roster"][0]["weight"] = 99.0
    with pytest.raises(Revision25EvaluationError, match="roster or weights"):
        validate_revision25_formal_gate_contract(base, formal)


def test_activation_receipt_reopens_exact_artifacts(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    formal_path = tmp_path / "formal.json"
    commit_path = tmp_path / "iter_00000.json"
    _write(base_path, _base_gate())
    _write(
        formal_path,
        build_revision25_formal_gate_contract(
            _base_gate(), exact_result_pointer="/tmp/r327-result.json"
        ),
    )
    _write(commit_path, {"schema": "test", "next_iteration": 1})
    receipt = {
        "schema": MIGRATION_SCHEMA,
        "status": "authorized_at_clean_boundary",
        "owner_goal_revision": 25,
        "root_owner_revision": 327,
        "run_name": "alakazam_rule_derivative_g5_r12",
        "first_formal_holdout_iteration": 1,
        "goal_sha256": GOAL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "base_gate_contract_sha256": sha256_file(base_path),
        "formal_gate_contract_sha256": sha256_file(formal_path),
        "boundary_commit_iteration": 0,
        "boundary_commit_sha256": sha256_file(commit_path),
        "formal_games_per_opponent": 50,
        "candidate_first_games_per_opponent": 25,
        "candidate_second_games_per_opponent": 25,
        "separate_research_control_games": 0,
        "training_collection_mix_changed": False,
        "elmo_remote_endpoint_preserved": "192.168.1.143:8765",
    }
    assert validate_revision25_activation_receipt(
        receipt,
        base_contract_path=base_path,
        formal_contract_path=formal_path,
        boundary_commit_path=commit_path,
    ) == receipt
    receipt["elmo_remote_endpoint_preserved"] = ""
    with pytest.raises(Revision25EvaluationError, match="invalid"):
        validate_revision25_activation_receipt(
            receipt,
            base_contract_path=base_path,
            formal_contract_path=formal_path,
            boundary_commit_path=commit_path,
        )


def test_zero_research_keeps_training_mix_and_is_boundary_scoped() -> None:
    pytest.importorskip("torch")
    from scripts.train_pure_rl import _safe_alakazam_r327_evaluation_migration

    roster = [{"opponent_id": "a", "weight": 1.0}]
    registry = {"path": "/fixed/registry.json", "digest": "sha256:" + "a" * 64}
    stored = {
        "games": {"heldout": 4500},
        "gates": {"active_contract": {"path": "/old", "digest": "sha256:old"}},
        "collection": {
            "group_games_per_iteration": {
                "self_play": 1024,
                "strong_public_practice": 4586,
                "diverse_public": 2586,
            },
            "strong_public_practice": {
                "roster": roster,
                "seed_contract": {
                    "formal_games": 4500,
                    "research_control_games": 1000,
                },
            },
            "research_control_phase": {
                "enabled": True,
                "games_per_iteration": 1000,
                "registry": registry,
            },
        },
    }
    current = json.loads(json.dumps(stored))
    current["games"]["heldout"] = 900
    current["gates"]["active_contract"] = {
        "path": "/new",
        "digest": "sha256:new",
    }
    current["collection"]["strong_public_practice"]["seed_contract"].update(
        {"formal_games": 900, "research_control_games": 0}
    )
    current["collection"]["research_control_phase"].update(
        {
            "enabled": False,
            "games_per_iteration": 0,
            "legacy_reclaimed_training_slots": 1000,
            "disabled_by_revision_25": True,
            "revision_25_activation_receipt": {
                "path": "/receipt",
                "digest": "sha256:receipt",
            },
        }
    )
    changed = [
        "games.heldout",
        "gates.active_contract.path",
        "gates.active_contract.digest",
        "collection.strong_public_practice.seed_contract.formal_games",
        "collection.strong_public_practice.seed_contract.research_control_games",
        "collection.research_control_phase.enabled",
        "collection.research_control_phase.games_per_iteration",
        "collection.research_control_phase.legacy_reclaimed_training_slots",
        "collection.research_control_phase.disabled_by_revision_25",
        "collection.research_control_phase.revision_25_activation_receipt",
    ]
    assert _safe_alakazam_r327_evaluation_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="owner_r327_future_holdout_50_no_research",
    )
    assert not _safe_alakazam_r327_evaluation_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="generic",
    )


def test_disable_flag_requires_receipt() -> None:
    pytest.importorskip("torch")
    from scripts.train_pure_rl import _parse_args

    with pytest.raises(SystemExit):
        _parse_args(["--disable-research-control-measurement"])
