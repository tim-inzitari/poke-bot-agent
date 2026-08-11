"""Focused provenance proof for the immutable r241 41-row baseline receipt."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from poke_bot import baselines_runtime
from poke_bot import r241_baseline_payload_snapshot as baseline_payload
from poke_bot import r241_canonical_baseline_roster as canonical_roster
from poke_bot.r241_direct_policy_runtime import (
    R241_H10_CONTENT_SHA256,
    R241_H10_MODEL_SHA256,
    R241_H10_OPPONENT_ID,
)


OWNER_SHA256 = "sha256:" + "a" * 64
HISTORICAL_MARNIE_ID = "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98"


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(baseline_payload.canonical_json(value))
    return path


def _row(opponent_id: str, group: str, directory: str) -> dict[str, object]:
    return {
        "id": opponent_id,
        "source": f"fixture/{opponent_id}",
        "content": {
            "kind": "directory",
            "path": f"/immutable/baselines/{group}/{directory}",
            "digest": _digest(f"content:{opponent_id}"),
        },
    }


def _reference(name: str) -> dict[str, object]:
    return {
        "kind": "file",
        "path": f"/immutable/contracts/{name}.json",
        "digest": _digest(f"contract:{name}"),
        "size": 123,
    }


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    collect = [_row(f"collect-{index:02d}", "roster", f"collect-{index:02d}") for index in range(19)]
    heldout = [
        _row(HISTORICAL_MARNIE_ID, "specialists", "historical-marnie"),
        *[_row(f"gate-{index:02d}", "specialists", f"gate-{index:02d}") for index in range(16)],
    ]
    research = [
        _row("iono", "official", "iono"),
        _row("dragapult-ex", "official", "dragapult-ex"),
        _row("mega-abomasnow-ex", "official", "mega-abomasnow-ex"),
        _row("mega-lucario-ex", "official", "mega-lucario-ex"),
    ]
    manifest = {
        "active_gate_contract": _reference("active-gate"),
        "design_contract": {
            "opponents": {
                "collect": collect,
                "heldout": heldout,
                "official_target_training": copy.deepcopy(heldout),
                "research_controls": research,
            },
            "gates": {"frozen_specialist_registry": _reference("frozen-specialists")},
            "collection": {
                "research_control_phase": {"registry": _reference("research-controls")}
            },
        },
    }
    r175_manifest = _write_json(tmp_path / "r175" / "manifest.json", manifest)
    r192_contract = _write_json(
        tmp_path / "r192" / "contract.json",
        {
            "schema": "poke_bot.alakazam_marnie_splusplus_opponent_r192/v1",
            "owner_decision_revision": 192,
            "opponent": {
                "opponent_id": R241_H10_OPPONENT_ID,
                "checkpoint_sha256": R241_H10_MODEL_SHA256,
                "content_digest": R241_H10_CONTENT_SHA256,
                "tier": "S++",
                "weight": 4.0,
                "floor_games_per_set": 1024,
                "distinct_additional_specialist_row": True,
                "duplicate_alias_row_allowed": False,
            },
            "historical_marnie": {
                "opponent_id": HISTORICAL_MARNIE_ID,
                "must_remain_distinct": True,
                "may_be_collapsed_or_substituted": False,
            },
        },
    )
    strong_ids = sorted([item["id"] for item in heldout] + [R241_H10_OPPONENT_ID])
    minimums = {opponent_id: 183 for opponent_id in strong_ids}
    minimums[R241_H10_OPPONENT_ID] = 1024
    per_opponent: dict[str, object] = {}
    extra_id = next(opponent_id for opponent_id in strong_ids if opponent_id != R241_H10_OPPONENT_ID)
    for opponent_id in strong_ids:
        if opponent_id == R241_H10_OPPONENT_ID:
            games, seat0, seat1 = 1042, 521, 521
        elif opponent_id == extra_id:
            games, seat0, seat1 = 616, 308, 308
        else:
            games, seat0, seat1 = 183, 92, 91
        per_opponent[opponent_id] = {
            "games": games,
            "minimum_games": minimums[opponent_id],
            "seat0": seat0,
            "seat1": seat1,
        }
    assert sum(int(row["games"]) for row in per_opponent.values()) == 4586
    plan = {
        "schema": "poke_bot.strong_public_practice_plan/v1",
        "iteration": 20,
        "training_eligible": True,
        "games": 4586,
        "group_games_per_iteration": {
            "self_play": 1024,
            "strong_public_practice": 4586,
            "diverse_public": 2586,
        },
        "minimum_games_by_opponent": minimums,
        "per_opponent": per_opponent,
        "research_controls": {
            "additive_measurement_games": 1000,
            "training_eligible": False,
            "replay_eligible": False,
            "stage": "measure:research_controls",
        },
    }
    iter20_plan = _write_json(tmp_path / "r175" / "collection_plans" / "iter_00020.json", plan)
    receipt = {
        "schema": "poke_bot.completed_collection/v1",
        "iteration": 20,
        "requested_games": 8196,
        "stats": {
            "n_baseline_jobs": 7172,
            "n_public_mix_jobs": 7172,
            "n_self_play_jobs": 1024,
            "n_research_control_jobs": 0,
            "self_play": 1024,
            "retained_public_mix_source_games": 7172,
            "retained_self_play_source_games": 1024,
            "retained_source_games": 8196,
            "ok": 8196,
            "strong_public_practice_plan": "/immutable/r175/collection_plans/iter_00020.json",
            "strong_public_practice_record_receipt": {
                "schema": "poke_bot.strong_public_practice_record_receipt/v1",
                "passed": True,
                "expected_results": 4586,
                "seen_results": 4586,
                "successful_results": 4586,
                "canonical_records_written": 4586,
                "failed_results": 0,
            },
            "training_seat_split": {
                "schema": "poke_bot.alakazam_refresh_seat_split_summary/v1",
                "policy": "exact_50_50_first_second_training",
                "passed": True,
                "assigned_source_games": {
                    "exact_50_50": True,
                    "seat0": 4098,
                    "seat1": 4098,
                    "total": 8196,
                },
                "retained_source_games": {
                    "exact_50_50": True,
                    "seat0": 4098,
                    "seat1": 4098,
                    "total": 8196,
                },
            },
        },
    }
    iter20_receipt = _write_json(
        tmp_path / "r175" / "collection_receipts" / "iter_00020.json", receipt
    )
    paths = {
        "r175_manifest": r175_manifest,
        "r192_h10_contract": r192_contract,
        "r175_iter20_plan": iter20_plan,
        "r175_iter20_receipt": iter20_receipt,
    }
    expected = {
        "r175_run_manifest": baseline_payload.sha256_file(r175_manifest),
        "r192_h10_contract": baseline_payload.sha256_file(r192_contract),
        "r175_iter20_strong_public_practice_plan": baseline_payload.sha256_file(iter20_plan),
        "r175_iter20_collection_receipt": baseline_payload.sha256_file(iter20_receipt),
    }
    return paths, expected


def test_builds_staging_compatible_exact_41_row_receipt(tmp_path: Path) -> None:
    paths, expected = _fixture(tmp_path)
    receipt = canonical_roster.build_receipt(
        **paths,
        owner_contract_sha256=OWNER_SHA256,
        expected_provenance_sha256s=expected,
    )

    assert receipt["schema"] == baseline_payload.CANONICAL_BASELINE_ROSTER_SCHEMA
    assert receipt["status"] == "passed"
    assert receipt["owner_contract_sha256"] == OWNER_SHA256
    assert receipt["counts"] == {
        "r175_diverse_public_rows": 19,
        "r175_historical_active_gate_rows": 17,
        "r192_h10_additional_rows": 1,
        "r175_iter20_strong_public_practice_rows": 18,
        "r175_research_control_rows": 4,
        "training_public_union_rows": 37,
        "canonical_baseline_union_rows": 41,
        "iter20_self_play_games": 1024,
        "iter20_diverse_public_games": 2586,
        "iter20_strong_public_practice_games": 4586,
        "iter20_public_mix_games": 7172,
        "iter20_total_games": 8196,
        "iter20_research_measurement_games": 1000,
        "iter20_research_training_eligible": False,
        "iter20_research_replay_eligible": False,
    }
    assert receipt["baseline_roster"] == sorted(
        receipt["baseline_roster"], key=lambda row: row["id"]
    )
    assert len(receipt["baseline_roster"]) == 41
    assert receipt["lane_membership"]["r175_diverse_public"]["count"] == 19
    assert receipt["lane_membership"]["r175_historical_active_gate"]["count"] == 17
    assert receipt["lane_membership"]["r192_h10_additional_strong_specialist"]["ids"] == [
        R241_H10_OPPONENT_ID
    ]
    assert receipt["lane_membership"]["r192_h10_additional_strong_specialist"][
        "not_in_historical_active_gate"
    ]
    assert receipt["lane_membership"]["r175_iter20_strong_public_practice"]["count"] == 18
    assert receipt["lane_membership"]["r175_research_controls"]["count"] == 4
    assert receipt["lane_membership"]["training_public_union"]["count"] == 37
    assert receipt["lane_membership"]["canonical_baseline_union"]["count"] == 41

    minimal = receipt["minimal_baseline_manifest"]
    minimal_bytes = baseline_payload.minimal_manifest_bytes(receipt["baseline_roster"])
    assert minimal["canonical_json_utf8"] == minimal_bytes.decode("utf-8")
    assert minimal["sha256"] == receipt["baseline_manifest_sha256"]
    assert minimal["sha256"] == baseline_payload.sha256_bytes(minimal_bytes)
    manifest_path = tmp_path / "derived" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(minimal_bytes)
    loaded = baselines_runtime.load_manifest(manifest_path)
    assert [spec.id for spec in loaded] == [row["id"] for row in receipt["baseline_roster"]]

    output = canonical_roster.write_receipt(tmp_path / "derived" / "canonical-roster.json", receipt)
    assert output.stat().st_mode & 0o222 == 0
    # Create-only is idempotent only for the same canonical bytes.
    assert canonical_roster.write_receipt(output, receipt) == output
    validated_path, validated = baseline_payload.validate_canonical_roster_receipt(
        output,
        expected_sha256=baseline_payload.sha256_file(output),
        owner_contract_sha256=OWNER_SHA256,
    )
    assert validated_path == output
    assert validated["baseline_roster_sha256"] == receipt["baseline_roster_sha256"]

    # The production CLI intentionally fixes these four values rather than
    # permitting an operator to substitute another lineage.
    assert canonical_roster.DEFAULT_PROVENANCE_SHA256S == {
        "r175_run_manifest": canonical_roster.R175_RUN_MANIFEST_SHA256,
        "r192_h10_contract": canonical_roster.R192_H10_CONTRACT_SHA256,
        "r175_iter20_strong_public_practice_plan": canonical_roster.R175_ITER20_PLAN_SHA256,
        "r175_iter20_collection_receipt": canonical_roster.R175_ITER20_RECEIPT_SHA256,
    }

    # Even if a caller supplies a matching test-only source SHA, a changed H10
    # floor is semantically rejected before it can become a canonical row.
    changed_r192 = json.loads(paths["r192_h10_contract"].read_text(encoding="utf-8"))
    changed_r192["opponent"]["floor_games_per_set"] = 1023
    _write_json(paths["r192_h10_contract"], changed_r192)
    changed_expected = dict(expected)
    changed_expected["r192_h10_contract"] = baseline_payload.sha256_file(
        paths["r192_h10_contract"]
    )
    with pytest.raises(canonical_roster.R241CanonicalBaselineRosterError, match="floor games"):
        canonical_roster.build_receipt(
            **paths,
            owner_contract_sha256=OWNER_SHA256,
            expected_provenance_sha256s=changed_expected,
        )
