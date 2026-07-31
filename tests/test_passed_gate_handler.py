from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model
from scripts.handle_passed_gate import (
    _canonical_digest,
    _decision_fusion_runtime_ready,
    freeze_exact_pass,
    materialize_pinned_specialist_deck,
    validate_exact_pass,
    validate_runtime_exact_gate,
)
from poke_bot.model import (
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_V2_OPTIONAL_HEADS,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _successor_fusion_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    terminal_schema: str = "poke_bot.causal_decision_fusion/v1",
    terminal_optional_heads: tuple[str, ...] = (),
    migrated_bootstrap: bool = False,
) -> tuple[Path, Path]:
    outputs = tmp_path / "outputs"
    run_dir = outputs / "pure_rl" / "successor-run"
    checkpoint_path = run_dir / "checkpoints" / "iter_00005.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"successor-fused-checkpoint")
    digest = "sha256:" + "5" * 64
    bootstrap_digest = "sha256:" + "6" * 64
    design_contract = {"fixture": "initial-successor-design-v1"}
    fingerprint = _canonical_digest(design_contract)
    learner = {"path": str(checkpoint_path.resolve()), "digest": digest}
    publish = {
        "checkpoint": str(checkpoint_path.resolve()),
        "digest": digest,
        "local_ok": True,
        "remote_ok": True,
    }
    commit = {
        "last_completed_iteration": 5,
        "next_iteration": 6,
        "design_fingerprint": fingerprint,
        "learner": learner,
        "history": [
            {
                "iteration": 5,
                "completed": True,
                "candidate": learner,
                "learner_after": learner,
                "next_collection_publish": publish,
            }
        ],
    }
    _write(run_dir / "loop_state.json", commit)
    _write(run_dir / "commits" / "iter_00005.json", commit)
    initial_digest = bootstrap_digest
    initial_path = run_dir / "checkpoints" / "bootstrap.pt"
    if migrated_bootstrap:
        initial_digest = "sha256:" + "7" * 64
        initial_path = (
            outputs
            / "pure_rl"
            / "_protected"
            / "models"
            / "test-specialist-fusion-v2"
            / "model.pt"
        )
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_path.write_bytes(b"migrated-bootstrap")
        frozen_manifest = {
            "schema": "poke_bot.frozen_model/v1",
            "immutable": True,
            "automatic_pruning_allowed": False,
            "checkpoint_digest": initial_digest,
            "model_path": str(initial_path.resolve()),
            "provenance": {
                "kind": "decision_fusion_v2_hot_start",
                "all_legacy_tensors_bit_identical": True,
                "source_checkpoint_digest": bootstrap_digest,
            },
        }
        monkeypatch.setattr(
            "scripts.handle_passed_gate.verify_frozen_model",
            lambda _family_dir: frozen_manifest,
        )
    _write(
        run_dir / "manifest.json",
        {
            "specialist_archetype": "test-specialist",
            "design_fingerprint": fingerprint,
            "design_contract": design_contract,
            "initial_learner_checkpoint": {
                "digest": initial_digest,
                "path": str(initial_path.resolve()),
            },
        },
    )
    required = list(DECISION_FUSION_REQUIRED_HEADS)
    terminal_required = [*required, *terminal_optional_heads]
    _write(
        outputs
        / "state"
        / "test-specialist-specialist-rl-activation-v1.json",
        {
            "schema": "poke_bot.specialist_rl_activation/v2",
            "status": "ready",
            "identity": {
                "next_specialist_bootstrap": {
                    "specialist_id": "test-specialist",
                    "checkpoint_digest": bootstrap_digest,
                    "decision_fusion": {
                        "schema": "poke_bot.causal_decision_fusion/v1",
                        "runtime_enabled": True,
                        "required_heads": required,
                    },
                },
                "runtime_registration": {
                    "specialist_id": "test-specialist",
                    "runtime_row": {
                        "initial_checkpoint_sha256": bootstrap_digest,
                        "decision_fusion": {
                            "schema": "poke_bot.causal_decision_fusion/v1",
                            "required": True,
                            "runtime_enabled": True,
                            "required_heads": required,
                        },
                    },
                },
            },
        },
    )
    payload = {
        "archetype_id": "test-specialist",
        "model_config": {
            "decision_fusion_enabled": True,
            "decision_fusion_runtime_enabled": True,
        },
        "provenance": {
            "decision_fusion": {
                "schema": terminal_schema,
                "runtime_enabled": True,
                "required_heads": terminal_required,
            }
        },
    }
    monkeypatch.setattr(
        "scripts.handle_passed_gate.checkpoint.checkpoint_digest",
        lambda _path: digest,
    )
    monkeypatch.setattr(
        "scripts.handle_passed_gate.checkpoint.load_checkpoint",
        lambda _path, map_location=None: payload,
    )
    return run_dir, run_dir / "commits" / "iter_00005.json"


def test_generated_successor_fusion_descendant_is_terminal_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _commit_path = _successor_fusion_fixture(tmp_path, monkeypatch)
    ready, reason = _decision_fusion_runtime_ready(run_dir)
    assert ready is True
    assert "verified successor fused descendant" in reason


def test_generated_successor_v2_fusion_descendant_accepts_optional_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _commit_path = _successor_fusion_fixture(
        tmp_path,
        monkeypatch,
        terminal_schema="poke_bot.causal_decision_fusion/v2",
        terminal_optional_heads=(DECISION_FUSION_V2_OPTIONAL_HEADS[0],),
    )
    ready, reason = _decision_fusion_runtime_ready(run_dir)
    assert ready is True
    assert "verified successor fused descendant" in reason


def test_generated_successor_v2_accepts_receipted_hot_start_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _commit_path = _successor_fusion_fixture(
        tmp_path,
        monkeypatch,
        terminal_schema="poke_bot.causal_decision_fusion/v2",
        terminal_optional_heads=(DECISION_FUSION_V2_OPTIONAL_HEADS[0],),
        migrated_bootstrap=True,
    )
    ready, reason = _decision_fusion_runtime_ready(run_dir)
    assert ready is True
    assert "verified successor fused descendant" in reason


def test_generated_successor_fusion_requires_fleet_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, commit_path = _successor_fusion_fixture(tmp_path, monkeypatch)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["history"][0]["next_collection_publish"]["remote_ok"] = False
    _write(commit_path, commit)
    ready, reason = _decision_fusion_runtime_ready(run_dir)
    assert ready is False
    assert "not published fleet-wide" in reason


def test_generated_successor_fusion_accepts_complete_terminal_gate_for_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, commit_path = _successor_fusion_fixture(tmp_path, monkeypatch)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    learner = commit["learner"]
    games = 3500
    commit["history"][0]["next_collection_publish"] = None
    commit["history"][0]["active_gate_result"] = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "iteration": 5,
        "games": games,
        "checkpoint": learner["path"],
        "checkpoint_digest": learner["digest"],
        "passed": False,
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "greedy": True,
            "both_seats": True,
            "valid_games": games,
            "rows": games,
            "requested_games": games,
            "checkpoint_digest": learner["digest"],
            "matchup_runtime": {
                "schema": "poke_bot.matchup_runtime_collection_audit/v1",
                "all_games_audited": True,
                "all_runtime_enabled": True,
                "contract_clean": True,
                "games": games,
                "audited_games": games,
                "runtime_enabled_games": games,
                "runtime_disabled_games": 0,
                "missing_games": 0,
                "malformed_games": 0,
                "transition_contract_violations": 0,
            },
        },
    }
    _write(commit_path, commit)
    _write(run_dir / "loop_state.json", commit)

    ready, reason = _decision_fusion_runtime_ready(
        run_dir,
        allow_terminal_gate_evidence=True,
    )

    assert ready is True
    assert "complete audited active gate" in reason


def test_generated_successor_fusion_rejects_incomplete_terminal_gate_for_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, commit_path = _successor_fusion_fixture(tmp_path, monkeypatch)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["history"][0]["next_collection_publish"] = None
    commit["history"][0]["active_gate_result"] = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "iteration": 5,
        "games": 3500,
        "checkpoint": commit["learner"]["path"],
        "checkpoint_digest": commit["learner"]["digest"],
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "greedy": True,
            "both_seats": True,
            "valid_games": 3500,
            "rows": 3500,
            "requested_games": 3500,
            "checkpoint_digest": commit["learner"]["digest"],
            "matchup_runtime": {
                "schema": "poke_bot.matchup_runtime_collection_audit/v1",
                "all_games_audited": True,
                "all_runtime_enabled": False,
                "contract_clean": True,
                "games": 3500,
                "audited_games": 3500,
                "runtime_enabled_games": 3499,
                "runtime_disabled_games": 1,
                "missing_games": 0,
                "malformed_games": 0,
                "transition_contract_violations": 0,
            },
        },
    }
    _write(commit_path, commit)
    _write(run_dir / "loop_state.json", commit)

    ready, reason = _decision_fusion_runtime_ready(
        run_dir,
        allow_terminal_gate_evidence=True,
    )

    assert ready is False
    assert "not published fleet-wide" in reason


def test_generated_successor_fusion_accepts_verified_design_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, commit_path = _successor_fusion_fixture(tmp_path, monkeypatch)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    previous_contract = manifest["design_contract"]
    previous_fingerprint = manifest["design_fingerprint"]
    current_contract = {**previous_contract, "source": {"revision": 2}}
    current_fingerprint = _canonical_digest(current_contract)
    migration_path = run_dir / "design_migrations" / "migration_0001.json"
    migration = {
        "schema": 1,
        "receipt": str(migration_path.resolve()),
        "reason": "receipt_backed_decision_fusion_runtime_v1",
        "boundary_next_iteration": 5,
        "last_completed_iteration": 4,
        "previous_fingerprint": previous_fingerprint,
        "current_fingerprint": current_fingerprint,
        "previous_contract": previous_contract,
        "current_contract": current_contract,
    }
    _write(migration_path, migration)
    history = [
        {
            "receipt": str(migration_path.resolve()),
            "fingerprint": current_fingerprint,
            "boundary_next_iteration": 5,
            "reason": "receipt_backed_decision_fusion_runtime_v1",
        }
    ]
    for path in (run_dir / "loop_state.json", commit_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["design_fingerprint"] = current_fingerprint
        payload["design_migration_history"] = history
        _write(path, payload)

    ready, reason = _decision_fusion_runtime_ready(run_dir)

    assert ready is True
    assert "verified successor fused descendant" in reason


def test_generated_successor_fusion_accepts_initial_resume_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, commit_path = _successor_fusion_fixture(tmp_path, monkeypatch)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    previous_contract = manifest["design_contract"]
    previous_fingerprint = manifest["design_fingerprint"]
    current_contract = {**previous_contract, "source": {"revision": 2}}
    current_fingerprint = _canonical_digest(current_contract)
    migration_path = run_dir / "design_migrations" / "migration_0001.json"
    reason = "receipt_backed_completed_collection_resume_v1"
    migration = {
        "schema": 1,
        "receipt": str(migration_path.resolve()),
        "reason": reason,
        "boundary_next_iteration": 0,
        "last_completed_iteration": -1,
        "previous_fingerprint": previous_fingerprint,
        "current_fingerprint": current_fingerprint,
        "previous_contract": previous_contract,
        "current_contract": current_contract,
    }
    _write(migration_path, migration)
    history = [
        {
            "receipt": str(migration_path.resolve()),
            "fingerprint": current_fingerprint,
            "boundary_next_iteration": 0,
            "reason": reason,
        }
    ]
    for path in (run_dir / "loop_state.json", commit_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["design_fingerprint"] = current_fingerprint
        payload["design_migration_history"] = history
        _write(path, payload)

    ready, reason_text = _decision_fusion_runtime_ready(run_dir)

    assert ready is True
    assert "verified successor fused descendant" in reason_text


def test_generated_successor_fusion_rejects_corrupt_design_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, commit_path = _successor_fusion_fixture(tmp_path, monkeypatch)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    migration_path = run_dir / "design_migrations" / "migration_0001.json"
    current_contract = {"fixture": "tampered"}
    current_fingerprint = _canonical_digest(current_contract)
    _write(
        migration_path,
        {
            "schema": 1,
            "receipt": str(migration_path.resolve()),
            "reason": "receipt_backed_decision_fusion_runtime_v1",
            "boundary_next_iteration": 5,
            "last_completed_iteration": 4,
            "previous_fingerprint": manifest["design_fingerprint"],
            "current_fingerprint": current_fingerprint,
            "previous_contract": {"not": "the manifest contract"},
            "current_contract": current_contract,
        },
    )
    history = [
        {
            "receipt": str(migration_path.resolve()),
            "fingerprint": current_fingerprint,
            "boundary_next_iteration": 5,
            "reason": "receipt_backed_decision_fusion_runtime_v1",
        }
    ]
    for path in (run_dir / "loop_state.json", commit_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["design_fingerprint"] = current_fingerprint
        payload["design_migration_history"] = history
        _write(path, payload)

    ready, reason = _decision_fusion_runtime_ready(run_dir)

    assert ready is False
    assert "successor design migration chain is corrupt" in reason


def _publish_exact_pointer(run_dir: Path, contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    iteration = int(
        json.loads((run_dir / "SPECIALIST_GATE_PASSED").read_text())["iteration"]
    )
    commit_path = (run_dir / "commits" / f"iter_{iteration:05d}.json").resolve()
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    row = next(
        item for item in commit["history"] if int(item["iteration"]) == iteration
    )
    pointer = {
        **row["active_gate_result"],
        "committed": True,
        "commit": str(commit_path),
        "commit_digest": _canonical_digest(commit),
        "created_at_utc": "2026-07-22T00:00:00+00:00",
    }
    _write(Path(contract["next_gate"]["exact_result_pointer"]), pointer)


def _rewrite_commit(
    run_dir: Path,
    contract_path: Path,
    mutate,
) -> None:
    commit_path = run_dir / "commits" / "iter_00007.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    mutate(commit)
    _write(commit_path, commit)
    _publish_exact_pointer(run_dir, contract_path)


def _exact_gate_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoints" / "iter_00007.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"exact gated model")
    digest = sha256(checkpoint)
    roster_ids = [f"baseline-{index}" for index in range(8)]
    contract_path = tmp_path / "gate.json"
    exact_result_pointer = tmp_path / "exact-result.json"
    _write(
        contract_path,
        {
            "schema": "poke_bot.competition_gate_program/v1",
            "active_gate_id": "exact-eight-v1",
            "next_gate": {
                "id": "exact-eight-v1",
                "exact_result_pointer": str(exact_result_pointer.resolve()),
                "evaluation": {
                    "games_total": 2000,
                    "games_per_opponent": 250,
                    "seat0_games_per_opponent": 125,
                    "seat1_games_per_opponent": 125,
                },
                "roster": [
                    {"opponent_id": opponent_id} for opponent_id in roster_ids
                ],
                "pass_criteria": {
                    "audit_must_pass": True,
                    "skill_weighted_win_rate": 0.5,
                    "skill_weighted_confidence_lower": 0.5,
                    "s_tier_mean_floor": 0.4,
                    "individual_opponent_floor": 0.25,
                },
            }
        },
    )
    allocation = {
        opponent_id: {"games": 250, "seat0": 125, "seat1": 125}
        for opponent_id in roster_ids
    }
    result = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "gate_id": "exact-eight-v1",
        "iteration": 7,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_digest": digest,
        "games": 2000,
        "skill_weighted_wr": 0.55,
        "confidence_lower": 0.51,
        "passed": True,
        "pipeline_gate_passed": True,
        "promotion_passed": True,
        "checks": {
            "audit": True,
            "skill_weighted_win_rate": True,
            "skill_weighted_confidence_lower": True,
            "s_tier_mean_floor": True,
            "individual_opponent_floor": True,
        },
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "greedy": True,
            "both_seats": True,
            "valid_games": 2000,
            "rows": 2000,
            "requested_games": 2000,
            "checkpoint_digest": digest,
            "per_opponent": allocation,
        },
        "matchups": [
            {"opponent_id": opponent_id, **counts}
            for opponent_id, counts in allocation.items()
        ],
    }
    marker = {
        "iteration": 7,
        "wr": 0.55,
        "confidence_lower": 0.51,
        "games": 2000,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_digest": digest,
    }
    _write(run_dir / "SPECIALIST_GATE_PASSED", marker)
    state = {
        "mode": "specialist",
        "last_completed_iteration": 7,
        "next_iteration": 8,
        "history": [
            {
                "iteration": 7,
                "completed": True,
                "candidate": {
                    "path": str(checkpoint.resolve()),
                    "digest": digest,
                },
                "stage_gate": {
                    "passed": True,
                    "win_rate": 0.55,
                    "confidence_lower": 0.51,
                    "games": 2000,
                },
                "active_gate_result": result,
            }
        ],
    }
    _write(run_dir / "loop_state.json", state)
    _write(run_dir / "commits" / "iter_00007.json", state)
    _publish_exact_pointer(run_dir, contract_path)
    return run_dir, contract_path, checkpoint


def _runtime_exact_gate_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    run_dir, contract_path, checkpoint = _exact_gate_fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    gate = contract["next_gate"]
    official_ids = [f"official-{index}" for index in range(4)]
    gate["research_measurements"] = [
        {"opponent_id": opponent_id, "games": 250}
        for opponent_id in official_ids
    ]
    gate["pass_criteria"]["accepted_official_holdout_non_regression"] = 0.5
    _write(contract_path, contract)

    commit_path = run_dir / "commits" / "iter_00007.json"
    digest = sha256(checkpoint)
    activation_path = tmp_path / "runtime-activation.json"
    _write(
        activation_path,
        {
            "schema": "poke_bot.causal_decision_fusion_runtime_boundary/v1",
            "boundary": {
                "last_completed_iteration": 7,
                "next_iteration": 8,
                "commit": str(commit_path.resolve()),
                "commit_digest": sha256(commit_path),
            },
            "runtime_learner": {
                "path": str(checkpoint.resolve()),
                "digest": digest,
            },
            "decision_fusion": {
                "runtime_enabled": True,
                "serving_eligible": True,
            },
        },
    )
    result = json.loads(
        (run_dir / "commits" / "iter_00007.json").read_text(encoding="utf-8")
    )["history"][-1]["active_gate_result"]
    result["research_checks"] = {
        "research_control_audit": True,
        "accepted_official_holdout_non_regression": True,
    }
    official_allocation = {
        opponent_id: {"games": 250, "seat0": 125, "seat1": 125}
        for opponent_id in official_ids
    }
    result["research_controls"] = {
        "games": 1000,
        "pooled_wr": 0.55,
        "gate_weight": 0.0,
        "included_in_skill_weighted_wr": False,
        "matchups": [
            {"opponent_id": opponent_id, **counts}
            for opponent_id, counts in official_allocation.items()
        ],
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "valid_games": 1000,
            "rows": 1000,
            "requested_games": 1000,
            "checkpoint_digest": digest,
            "per_opponent": official_allocation,
        },
    }
    receipt_path = tmp_path / "runtime-exact-gate.json"
    _write(
        receipt_path,
        {
            "schema": "poke_bot.causal_decision_fusion_exact_gate/v1",
            "complete": True,
            "training_eligible": False,
            "replay_eligible": False,
            "run_dir": str(run_dir.resolve()),
            "iteration": 7,
            "boundary": {
                "commit": str(commit_path.resolve()),
                "commit_digest": sha256(commit_path),
            },
            "activation_receipt": {
                "path": str(activation_path.resolve()),
                "digest": sha256(activation_path),
            },
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "digest": digest,
            },
            "contract": {
                "path": str(contract_path.resolve()),
                "digest": sha256(contract_path),
                "canonical_digest": _canonical_digest(contract),
                "gate_id": gate["id"],
            },
            "premium_gate_complete": True,
            "official_gate_complete": True,
            "premium_gate_passed": True,
            "official_gate_passed": True,
            "both_gates_passed": True,
            "completion_authority": "measured_both_gates_pass",
            "result": result,
            "result_digest": _canonical_digest(result),
        },
    )
    return run_dir, contract_path, checkpoint, receipt_path


def test_runtime_exact_gate_binds_both_gates_to_serving_child(
    tmp_path: Path,
) -> None:
    run_dir, contract, checkpoint, receipt = _runtime_exact_gate_fixture(tmp_path)
    plan = validate_runtime_exact_gate(
        run_dir,
        contract,
        receipt,
        accept_ceiling=True,
        ceiling_iteration=7,
    )
    assert plan["checkpoint"] == str(checkpoint.resolve())
    assert plan["checkpoint_digest"] == sha256(checkpoint)
    assert plan["completion_authority"] == "measured_both_gates_pass"
    assert plan["complete_holdouts"] == {
        "premium": {"games": 2000, "passed": True},
        "official": {"games": 1000, "passed": True},
    }


def test_runtime_exact_gate_rejects_flat_parent_substitution(
    tmp_path: Path,
) -> None:
    run_dir, contract, _checkpoint, receipt = _runtime_exact_gate_fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["checkpoint"]["digest"] = "sha256:" + "0" * 64
    _write(receipt, payload)
    with pytest.raises(RuntimeError, match="runtime exact gate validation failed"):
        validate_runtime_exact_gate(
            run_dir,
            contract,
            receipt,
            accept_ceiling=True,
            ceiling_iteration=7,
        )


def test_exact_gate_pass_validates_and_freezes(tmp_path: Path) -> None:
    run_dir, contract, checkpoint = _exact_gate_fixture(tmp_path)
    plan = validate_exact_pass(run_dir, contract)
    assert plan["games"] == 2000
    assert plan["games_per_opponent"] == 250
    assert plan["candidate_first_per_opponent"] == 125
    assert plan["candidate_second_per_opponent"] == 125
    assert plan["checkpoint_digest"] == sha256(checkpoint)

    frozen = freeze_exact_pass(
        plan,
        registry_root=tmp_path / "protected-models",
        family="alakazam-exact-gate-v1",
        display_name="Alakazam exact gate",
    )
    assert frozen["checkpoint_digest"] == plan["checkpoint_digest"]
    assert frozen["provenance"]["commit_digest"] == plan["commit_digest"]
    assert frozen["provenance"]["exact_result_pointer"] == plan[
        "exact_result_pointer"
    ]
    assert verify_frozen_model(Path(frozen["model_path"]).parent) == frozen


def test_exact_gate_pass_accepts_current_s_plus_floor_allowance_check(
    tmp_path: Path,
) -> None:
    run_dir, contract_path, _checkpoint = _exact_gate_fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["next_gate"]["pass_criteria"].update(
        {
            "s_plus_individual_floor": 0.30,
            "s_plus_below_floor_allowance": 2,
        }
    )
    _write(contract_path, contract)

    def add_current_check(commit: dict) -> None:
        result = commit["history"][-1]["active_gate_result"]
        result["checks"]["s_plus_matchup_floor_allowance"] = True
        result["s_plus_below_floor_allowance"] = 2
        result["s_plus_below_floor_count"] = 2

    _rewrite_commit(run_dir, contract_path, add_current_check)
    plan = validate_exact_pass(run_dir, contract_path)
    assert plan["validation"]["gate_criteria_set"] is True
    assert plan["validation"]["all_gate_criteria"] is True


def test_exact_gate_pass_supports_a_versioned_marker(tmp_path: Path) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)
    versioned = "SPECIALIST_GATE_PASSED.alakazam-lc55-v2.json"
    (run_dir / "SPECIALIST_GATE_PASSED").rename(run_dir / versioned)
    plan = validate_exact_pass(run_dir, contract, marker_name=versioned)
    assert plan["gate_id"] == "exact-eight-v1"
    with pytest.raises(RuntimeError, match="absent or invalid"):
        validate_exact_pass(run_dir, contract)


def test_exact_gate_pass_accepts_only_authorized_fallback_child(
    tmp_path: Path,
) -> None:
    run_dir, contract_path, _checkpoint = _exact_gate_fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["fallback_transition"] = {
        "id": "exact-eight-lc45-after-iter6-v1",
        "label": "Authorized exact fallback",
        "prior_gate_id": "exact-eight-v1",
        "activate_after_completed_iteration": 6,
        "only_if_prior_gate_unpassed": True,
        "prior_confidence_lower": 0.5,
        "skill_weighted_confidence_lower": 0.45,
    }
    _write(contract_path, contract)

    def use_fallback(commit: dict) -> None:
        row = commit["history"][-1]
        row["active_gate_result"]["gate_id"] = (
            "exact-eight-lc45-after-iter6-v1"
        )

    _rewrite_commit(run_dir, contract_path, use_fallback)
    plan = validate_exact_pass(run_dir, contract_path)
    assert plan["base_gate_id"] == "exact-eight-v1"
    assert plan["gate_id"] == "exact-eight-lc45-after-iter6-v1"
    assert plan["effective_contract_digest"] != _canonical_digest(contract)

    def use_unrelated_gate(commit: dict) -> None:
        commit["history"][-1]["active_gate_result"]["gate_id"] = "rogue-gate"

    _rewrite_commit(run_dir, contract_path, use_unrelated_gate)
    with pytest.raises(RuntimeError, match="gate_id"):
        validate_exact_pass(run_dir, contract_path)


def test_live_handler_waits_for_iteration_30_versioned_marker() -> None:
    root = Path(__file__).resolve().parents[1]
    unit = (
        root / "deploy/systemd/pokebot-passed-gate-handler.service"
    ).read_text(encoding="utf-8")
    runtime_path = (
        root / ".staging/zzzzzzzzzzzzzzzzzz-v31-matchup-runtime.conf"
    )
    if not runtime_path.is_file():
        pytest.skip("host-only V31 runtime staging artifact is unavailable")
    runtime = runtime_path.read_text(encoding="utf-8")

    marker = "SPECIALIST_GATE_PASSED.alakazam-lc55-v2"
    assert f"--marker-name {marker}" in unit
    assert "--minimum-completed-iteration 30" in unit
    assert f"--terminal-gate-marker-name {marker}" in runtime
    assert "--minimum-terminal-iteration 30" in runtime


def test_exact_gate_pass_rejects_unsafe_marker_name(tmp_path: Path) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="marker name"):
        validate_exact_pass(run_dir, contract, marker_name="../pass.json")


def test_mutable_loop_state_cannot_override_immutable_pass(tmp_path: Path) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)
    loop = json.loads((run_dir / "loop_state.json").read_text())
    loop["history"][0]["active_gate_result"]["passed"] = False
    loop["history"][0]["candidate"]["digest"] = "sha256:" + "0" * 64
    _write(run_dir / "loop_state.json", loop)

    plan = validate_exact_pass(run_dir, contract)
    assert plan["iteration"] == 7
    assert plan["validation"]["result_pointer_commit_digest"] is True


def test_gate_handler_refuses_tampered_immutable_commit(tmp_path: Path) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)
    commit_path = run_dir / "commits" / "iter_00007.json"
    commit = json.loads(commit_path.read_text())
    commit["unexpected_mutation"] = True
    _write(commit_path, commit)

    with pytest.raises(RuntimeError, match="result_pointer_commit_digest"):
        validate_exact_pass(run_dir, contract)


def test_gate_handler_requires_the_complete_exact_check_set(tmp_path: Path) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)

    def remove_required_check(state: dict) -> None:
        state["history"][0]["active_gate_result"]["checks"].pop(
            "individual_opponent_floor"
        )

    _rewrite_commit(run_dir, contract, remove_required_check)
    with pytest.raises(RuntimeError, match="gate_criteria_set"):
        validate_exact_pass(run_dir, contract)


def test_gate_handler_requires_canonical_exact_pointer_binding(tmp_path: Path) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)
    contract_payload = json.loads(contract.read_text())
    pointer_path = Path(contract_payload["next_gate"]["exact_result_pointer"])
    pointer = json.loads(pointer_path.read_text())
    pointer["commit_digest"] = "sha256:" + "f" * 64
    _write(pointer_path, pointer)

    with pytest.raises(RuntimeError, match="result_pointer_commit_digest"):
        validate_exact_pass(run_dir, contract)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("passed", False, "active_gate_passed"),
    ],
)
def test_gate_handler_refuses_nonpassing_result(
    tmp_path: Path, field: str, value: bool, match: str
) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)
    _rewrite_commit(
        run_dir,
        contract,
        lambda state: state["history"][0]["active_gate_result"].__setitem__(
            field, value
        ),
    )
    with pytest.raises(RuntimeError, match=match):
        validate_exact_pass(run_dir, contract)


def test_gate_handler_accepts_formal_pass_when_incumbent_h2h_fails(
    tmp_path: Path,
) -> None:
    run_dir, contract, checkpoint = _exact_gate_fixture(tmp_path)

    def record_diagnostic_h2h_failure(state: dict) -> None:
        row = state["history"][0]
        row["stage_gate"]["passed"] = False
        row["stage_gate"]["reason"] = "candidate_not_promoted"
        row["active_gate_result"]["pipeline_gate_passed"] = False
        row["active_gate_result"]["pipeline_gate_reason"] = (
            "candidate_not_promoted"
        )
        row["active_gate_result"]["promotion_passed"] = False

    _rewrite_commit(run_dir, contract, record_diagnostic_h2h_failure)
    plan = validate_exact_pass(run_dir, contract)
    assert plan["checkpoint_digest"] == sha256(checkpoint)
    assert plan["result"]["passed"] is True
    assert plan["result"]["promotion_passed"] is False


def test_gate_handler_refuses_partial_seat_allocation(tmp_path: Path) -> None:
    run_dir, contract, _checkpoint = _exact_gate_fixture(tmp_path)
    def corrupt(state: dict) -> None:
        state["history"][0]["active_gate_result"]["audit"]["per_opponent"][
            "baseline-0"
        ]["seat0"] = 124

    _rewrite_commit(run_dir, contract, corrupt)
    with pytest.raises(RuntimeError, match="audit_allocation:baseline-0"):
        validate_exact_pass(run_dir, contract)


def test_materialized_submission_deck_must_match_trained_specialist(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    cards = list(range(1, 61))
    _write(
        run_dir / "manifest.json",
        {
            "mode": "specialist",
            "specialist_archetype": "alakazam",
            "our_decks": ["alakazam"],
            "design_contract": {
                "measurement_deck_distribution": {
                    "entries": [
                        {
                            "name": "alakazam",
                            "cards_sha256": _canonical_digest(cards),
                        }
                    ]
                }
            },
        },
    )
    representatives = tmp_path / "representatives.json"
    _write(representatives, {"decks": {"alakazam": {"card_ids": cards}}})
    output = tmp_path / "submission" / "alakazam.csv"
    receipt = materialize_pinned_specialist_deck(
        run_dir=run_dir,
        representatives_path=representatives,
        archetype="alakazam",
        output_path=output,
    )
    assert receipt["cards"] == 60
    assert [int(line) for line in output.read_text().splitlines()] == cards

    broken = json.loads(representatives.read_text())
    broken["decks"]["alakazam"]["card_ids"][0] = 999
    _write(representatives, broken)
    with pytest.raises(RuntimeError, match="differs from the exact trained deck"):
        materialize_pinned_specialist_deck(
            run_dir=run_dir,
            representatives_path=representatives,
            archetype="alakazam",
            output_path=tmp_path / "other.csv",
        )
