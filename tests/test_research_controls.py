from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.research_controls import (
    load_research_control_registry,
    pin_research_control_registry_file,
    research_control_ids,
    retire_passed_gate,
    retire_passed_gate_file,
    validate_research_control_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "ops" / "research_control_registry_v1.json"
GATE = ROOT / "ops" / "alakazam_gate_program_v1.json"
DIGEST = "sha256:" + "a" * 64
CHECKPOINT = "sha256:" + "b" * 64


def _passing_result(contract: dict) -> tuple[dict, dict, str]:
    gate = contract["next_gate"]
    criteria = gate["pass_criteria"]
    per = int(gate["evaluation"]["games_per_opponent"])
    core = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "gate_id": gate["id"],
        "iteration": 20,
        "checkpoint_digest": CHECKPOINT,
        "games": int(gate["evaluation"]["games_total"]),
        "passed": True,
        "pipeline_gate_passed": True,
        "pipeline_gate_reason": "ok",
        "promotion_passed": True,
        "candidate_safety_passed": True,
        "skill_weighted_wr": 0.60,
        "confidence_lower": 0.55,
        "s_tier_mean": 0.60,
        "minimum_opponent_wr": 0.55,
        "s_plus_below_floor_count": 0,
        "s_plus_below_floor_allowance": int(
            criteria.get("s_plus_below_floor_allowance", 0)
        ),
        "checks": {
            "audit": True,
            "skill_weighted_win_rate": True,
            "skill_weighted_confidence_lower": True,
            "s_tier_mean_floor": True,
            "individual_opponent_floor": True,
            **(
                {"s_plus_matchup_floor_allowance": True}
                if "s_plus_individual_floor" in criteria
                else {}
            ),
            **(
                {"accepted_official_holdout_non_regression": True}
                if "accepted_official_holdout_non_regression" in criteria
                else {}
            ),
        },
        "audit": {
            "passed": True,
            "both_seats": True,
            "greedy": True,
            "exact_distribution": True,
            "exact_weights": True,
            "requested_games": int(gate["evaluation"]["games_total"]),
            "valid_games": int(gate["evaluation"]["games_total"]),
            "duplicate_job_ids": [],
            "missing_job_ids": [],
            "unexpected_job_ids": [],
        },
        "matchups": [
            {
                "opponent_id": row["opponent_id"],
                "games": per,
                "wr": 0.6,
                "seat0": per // 2,
                "seat1": per // 2,
            }
            for row in gate["roster"]
        ],
        **(
            {
                "official_control_gate": {
                    "audit_passed": True,
                    "checkpoint_digest_matches": True,
                    "games": 1000,
                    "minimum_win_rate": float(
                        criteria["accepted_official_holdout_non_regression"]
                    ),
                    "passed": True,
                    "replay_eligible": False,
                    "training_eligible": False,
                    "win_rate": 0.60,
                }
            }
            if "accepted_official_holdout_non_regression" in criteria
            else {}
        ),
    }
    commit = {
        "last_completed_iteration": 20,
        "next_iteration": 21,
        "history": [
            {
                "iteration": 20,
                "completed": True,
                "promoted": True,
                "candidate": {"digest": CHECKPOINT},
                "active_gate_result": core,
            }
        ],
    }
    commit_digest = "sha256:" + hashlib.sha256(
        json.dumps(commit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        **core,
        "committed": True,
        "commit": "/tmp/commit.json",
        "commit_digest": commit_digest,
        "created_at_utc": "2026-07-21T12:00:00Z",
    }
    return result, commit, commit_digest


def test_initial_registry_is_digest_bound_and_zero_weight() -> None:
    registry = load_research_control_registry(REGISTRY)
    assert research_control_ids(registry) == (
        "iono",
        "dragapult-ex",
        "mega-abomasnow-ex",
        "mega-lucario-ex",
    )
    assert all(row["gate_weight"] == 0.0 for row in registry["controls"])
    assert all(row["formal_eval"] is False for row in registry["controls"])
    assert all(row["training_eligible"] is False for row in registry["controls"])


def test_registry_rejects_unretired_or_malformed_controls() -> None:
    registry = load_research_control_registry(REGISTRY)
    unretired = json.loads(json.dumps(registry))
    unretired["controls"].append(
        {
            **unretired["controls"][0],
            "opponent_id": "unaudited-agent",
            "content_digest": "sha256:" + "f" * 64,
            "source_gate_id": "never-passed",
        }
    )
    with pytest.raises(ValueError, match="committed retirement proof"):
        validate_research_control_registry(unretired)

    missing_weight = json.loads(json.dumps(registry))
    missing_weight["controls"][0].pop("gate_weight")
    with pytest.raises(ValueError, match="gate_weight"):
        validate_research_control_registry(missing_weight)

    forged_seed = json.loads(json.dumps(registry))
    forged_seed["controls"][0]["content_digest"] = "sha256:" + "e" * 64
    with pytest.raises(ValueError, match="exact seed"):
        validate_research_control_registry(forged_seed)


def test_lineage_registry_snapshot_is_immutable_when_latest_advances(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "state" / "latest.json"
    latest.parent.mkdir()
    latest.write_text(REGISTRY.read_text(), encoding="utf-8")
    pinned = pin_research_control_registry_file(
        latest,
        snapshot_dir=tmp_path / "state" / "snapshots",
    )
    pinned_bytes = pinned.read_bytes()
    assert pinned != latest
    assert load_research_control_registry(pinned)["version"] == 1

    # A future lineage may advance the mutable latest pointer. The current
    # lineage's manifest path remains content-addressed and unchanged.
    advanced = json.loads(REGISTRY.read_text())
    advanced["updated_at_utc"] = "2026-07-22T00:00:00Z"
    latest.write_text(json.dumps(advanced), encoding="utf-8")
    assert pinned.read_bytes() == pinned_bytes
    assert load_research_control_registry(pinned)["version"] == 1
    assert pin_research_control_registry_file(
        pinned,
        snapshot_dir=tmp_path / "state" / "snapshots",
    ) == pinned


def test_resumed_trainer_reuses_manifest_pinned_registry_when_latest_advances(
    tmp_path: Path,
) -> None:
    from scripts import train_pure_rl

    latest = tmp_path / "state" / "latest.json"
    latest.parent.mkdir()
    latest.write_text(REGISTRY.read_text(), encoding="utf-8")
    snapshots = tmp_path / "state" / "snapshots"
    pinned = pin_research_control_registry_file(latest, snapshot_dir=snapshots)
    manifest = {
        "research_control_registry": train_pure_rl._path_content_identity(pinned)
    }

    advanced = json.loads(REGISTRY.read_text())
    advanced["updated_at_utc"] = "2026-07-23T00:00:00Z"
    latest.write_text(json.dumps(advanced), encoding="utf-8")

    assert (
        train_pure_rl._research_control_registry_for_lineage(
            latest,
            snapshot_dir=snapshots,
            immutable_manifest=manifest,
        )
        == pinned
    )


def test_committed_passed_gate_retires_exact_roster_once() -> None:
    registry = load_research_control_registry(REGISTRY)
    contract = json.loads(GATE.read_text())
    result, commit, commit_digest = _passing_result(contract)
    updated = retire_passed_gate(
        registry=registry,
        gate_contract=contract,
        exact_result=result,
        exact_result_digest=DIGEST,
        commit_record=commit,
        commit_digest=commit_digest,
    )
    retired_ids = tuple(row["opponent_id"] for row in contract["next_gate"]["roster"])
    assert updated["version"] == registry["version"] + 1
    assert research_control_ids(updated)[-len(retired_ids) :] == retired_ids
    assert updated["retirements"][0]["opponent_ids"] == list(retired_ids)
    assert all(
        row["gate_weight"] == 0.0
        and row["included_in_gate_pass"] is False
        and row["formal_eval"] is False
        and row["training_eligible"] is False
        for row in updated["controls"][-len(retired_ids) :]
    )
    assert retire_passed_gate(
        registry=updated,
        gate_contract=contract,
        exact_result=result,
        exact_result_digest=DIGEST,
        commit_record=commit,
        commit_digest=commit_digest,
    ) == updated


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("committed", False),
        ("pipeline_gate_passed", False),
        ("passed", False),
    ],
)
def test_retirement_rejects_incomplete_or_failed_gate(field: str, value: bool) -> None:
    contract = json.loads(GATE.read_text())
    result, commit, commit_digest = _passing_result(contract)
    result[field] = value
    with pytest.raises(ValueError, match="committed, pipeline-passed"):
        retire_passed_gate(
            registry=load_research_control_registry(REGISTRY),
            gate_contract=contract,
            exact_result=result,
            exact_result_digest=DIGEST,
            commit_record=commit,
            commit_digest=commit_digest,
        )


def test_retirement_rejects_partial_roster_and_package_alias() -> None:
    contract = json.loads(GATE.read_text())
    result, commit, commit_digest = _passing_result(contract)
    result["matchups"].pop()
    with pytest.raises(ValueError, match="exact contracted roster"):
        retire_passed_gate(
            registry=load_research_control_registry(REGISTRY),
            gate_contract=contract,
            exact_result=result,
            exact_result_digest=DIGEST,
            commit_record=commit,
            commit_digest=commit_digest,
        )


def test_retirement_accepts_and_verifies_official_holdout_check() -> None:
    registry = load_research_control_registry(REGISTRY)
    contract = json.loads(GATE.read_text())
    contract["next_gate"]["pass_criteria"][
        "accepted_official_holdout_non_regression"
    ] = 0.5
    result, commit, commit_digest = _passing_result(contract)
    updated = retire_passed_gate(
        registry=registry,
        gate_contract=contract,
        exact_result=result,
        exact_result_digest=DIGEST,
        commit_record=commit,
        commit_digest=commit_digest,
    )
    assert updated["version"] == registry["version"] + 1

    bad_result, bad_commit, bad_commit_digest = _passing_result(contract)
    bad_result["official_control_gate"]["checkpoint_digest_matches"] = False
    with pytest.raises(ValueError, match="contract thresholds"):
        retire_passed_gate(
            registry=registry,
            gate_contract=contract,
            exact_result=bad_result,
            exact_result_digest=DIGEST,
            commit_record=bad_commit,
            commit_digest=bad_commit_digest,
        )

    registry = load_research_control_registry(REGISTRY)
    alias_contract = json.loads(json.dumps(contract))
    alias_contract["next_gate"]["roster"][0]["content_digest"] = registry[
        "controls"
    ][0]["content_digest"]
    alias_result, alias_commit, alias_commit_digest = _passing_result(
        alias_contract
    )
    with pytest.raises(ValueError, match="overlaps an existing research control"):
        retire_passed_gate(
            registry=registry,
            gate_contract=alias_contract,
            exact_result=alias_result,
            exact_result_digest=DIGEST,
            commit_record=alias_commit,
            commit_digest=alias_commit_digest,
        )


def test_active_gate_and_research_rosters_cannot_overlap() -> None:
    registry = load_research_control_registry(REGISTRY)
    with pytest.raises(ValueError, match="also a research control"):
        validate_research_control_registry(
            registry,
            active_gate_ids=("iono",),
        )
    with pytest.raises(ValueError, match="package alias"):
        validate_research_control_registry(
            registry,
            active_gate_digests=(registry["controls"][0]["content_digest"],),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda result, _commit: result["checks"].update(audit=False),
            "audit/check evidence",
        ),
        (
            lambda result, _commit: result["audit"].update(
                unexpected_job_ids=[999]
            ),
            "audit/check evidence",
        ),
        (
            lambda result, _commit: result.update(confidence_lower=0.1),
            "contract thresholds",
        ),
        (
            lambda result, _commit: result["matchups"][0].update(seat0=124),
            "exact contracted roster",
        ),
        (
            lambda _result, commit: commit["history"][0]["candidate"].update(
                digest="sha256:" + "c" * 64
            ),
            "immutable iteration commit",
        ),
    ],
)
def test_retirement_fails_closed_on_inexact_pass_evidence(
    mutation, message: str
) -> None:
    contract = json.loads(GATE.read_text())
    result, commit, _commit_digest = _passing_result(contract)
    mutation(result, commit)
    commit_digest = "sha256:" + hashlib.sha256(
        json.dumps(commit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["commit_digest"] = commit_digest
    with pytest.raises(ValueError, match=message):
        retire_passed_gate(
            registry=load_research_control_registry(REGISTRY),
            gate_contract=contract,
            exact_result=result,
            exact_result_digest=DIGEST,
            commit_record=commit,
            commit_digest=commit_digest,
        )


def test_retirement_file_writes_separate_durable_registry(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(REGISTRY.read_text())
    contract = json.loads(GATE.read_text())
    result_path = tmp_path / "result.json"
    result, commit, _commit_digest = _passing_result(contract)
    commit_path = tmp_path / "commit.json"
    result["commit"] = str(commit_path.resolve())
    commit_path.write_text(json.dumps(commit, sort_keys=True))
    result_path.write_text(json.dumps(result, sort_keys=True))
    output = tmp_path / "state" / "latest.json"
    updated = retire_passed_gate_file(
        registry_path=source,
        gate_contract=contract,
        exact_result_path=result_path,
        commit_path=commit_path,
        output_path=output,
    )
    assert output.is_file()
    assert json.loads(source.read_text())["version"] == 1
    assert json.loads(output.read_text())["version"] == updated["version"] == 2


def test_retirement_file_adds_only_new_member_from_later_gate_revision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(REGISTRY.read_text())
    full_contract = json.loads(GATE.read_text())
    old_contract = json.loads(json.dumps(full_contract))
    old_contract["next_gate"]["id"] = "prior-eight-member-gate"
    old_contract["active_gate_id"] = "prior-eight-member-gate"
    old_contract["next_gate"]["roster"] = old_contract["next_gate"]["roster"][:-1]
    per = int(old_contract["next_gate"]["evaluation"]["games_per_opponent"])
    old_contract["next_gate"]["evaluation"]["games_total"] = (
        per * len(old_contract["next_gate"]["roster"])
    )

    output = tmp_path / "state" / "latest.json"
    old_result, old_commit, _ = _passing_result(old_contract)
    old_commit_path = tmp_path / "old-commit.json"
    old_result_path = tmp_path / "old-result.json"
    old_result["commit"] = str(old_commit_path.resolve())
    old_commit_path.write_text(json.dumps(old_commit, sort_keys=True))
    old_result_path.write_text(json.dumps(old_result, sort_keys=True))
    first = retire_passed_gate_file(
        registry_path=source,
        gate_contract=old_contract,
        exact_result_path=old_result_path,
        commit_path=old_commit_path,
        output_path=output,
    )
    assert first["version"] == 2

    full_result, full_commit, _ = _passing_result(full_contract)
    full_commit_path = tmp_path / "full-commit.json"
    full_result_path = tmp_path / "full-result.json"
    full_result["commit"] = str(full_commit_path.resolve())
    full_commit_path.write_text(json.dumps(full_commit, sort_keys=True))
    full_result_path.write_text(json.dumps(full_result, sort_keys=True))
    second = retire_passed_gate_file(
        registry_path=source,
        gate_contract=full_contract,
        exact_result_path=full_result_path,
        commit_path=full_commit_path,
        output_path=output,
    )

    added = full_contract["next_gate"]["roster"][-1]
    assert second["version"] == 3
    assert second["controls"][: len(first["controls"])] == first["controls"]
    assert second["controls"][-1]["opponent_id"] == added["opponent_id"]
    assert second["controls"][-1]["content_digest"] == added["content_digest"]
    assert second["retirements"][: len(first["retirements"])] == first["retirements"]
    assert second["retirements"][-1]["opponent_ids"] == [added["opponent_id"]]
    assert load_research_control_registry(output) == second
    assert (
        retire_passed_gate_file(
            registry_path=source,
            gate_contract=full_contract,
            exact_result_path=full_result_path,
            commit_path=full_commit_path,
            output_path=output,
        )
        == second
    )


def test_staged_trainer_preserves_gate_and_separates_research_phase() -> None:
    source = (ROOT / "scripts" / "train_pure_rl.py").read_text()
    assert "production full-loop launch requires --active-gate-contract" in source
    assert "research controls are diagnostic-only" in source
    assert "ACTIVE_GATE_PRACTICE_SEED_DISJOINT" in source
    assert "--research-control-games-per-iter" in source
    assert "collection_group_plan" in source
    assert '"strong_public_practice": {' in source
    assert '"research_control_phase": {' in source
    assert '"games_per_iteration": int(' in source
    assert "measure:research_controls" in source
    assert "_build_research_control_jobs(" in source
    assert "_research_control_measurement(" in source
    assert "_partition_research_control_jobs" not in source
    assert '"training_eligible": False' in source
    assert '"replay_eligible": False' in source
    assert "RESEARCH_CONTROL_GATE_RETIRED" in source
