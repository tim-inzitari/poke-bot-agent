"""Fail-closed tests for rev7 acceptance of immutable rev5/rev6 evidence."""

from __future__ import annotations

import copy
import json

import pytest

import poke_bot.alakazam_rule_derivative_predecessor_compat_rev7 as compat
from poke_bot.alakazam_collision_census_r298 import (
    MECHANICS_ATTACHMENT_SHA256,
    OWNER_GOAL_SHA256,
    R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
    R298_OWNER_REVISION,
    R298_RAW_CORPUS_RECEIPT_SCHEMA,
    RULE_DERIVATIVE_CONTRACT_SHA256,
    RULE_DERIVATIVE_GATEWAY_SHA256,
    canonical_sha256,
    revision_5_predecessor_classification,
)
from poke_bot.alakazam_rule_derivative_corpus_scope_rev6 import (
    REV6_CONTRACT_PATH,
    REV6_CONTRACT_SHA256,
    REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA,
    REV6_GOAL_REVISION,
    REV6_PREDECESSOR_CONTRACT_SHA256,
    REV6_PREDECESSOR_GATEWAY_SHA256,
)
from poke_bot.alakazam_rule_derivative_predecessor_compat_rev7 import (
    EXACT_WORKER_COUNT,
    REV7_CENSUS_PHYSICAL_INVENTORY_KEY,
    REV7_CONTRACT_SHA256,
    REV7_GOAL_GATEWAY_SHA256,
    Rev7PredecessorCompatibilityError,
    build_revision_6_scope_migration_under_revision_7,
    load_revision_7_contract,
    revision_7_parallel_execution_plan,
    validate_revision_6_scope_migration_under_revision_7,
    write_revision_6_scope_migration_under_revision_7_create_only,
)

SHA = "sha256:" + "1" * 64


def _migration() -> dict[str, object]:
    contract = load_revision_7_contract()
    scope = contract["revision_6_corpus_scope_and_predecessor_acceptance"]
    fixed = scope["migration_receipt"]["fixed_values"]
    row: dict[str, object] = {
        "schema": REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA,
        "goal_contract_path": REV6_CONTRACT_PATH,
        "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION,
        "predecessor_gateway_sha256": REV6_PREDECESSOR_GATEWAY_SHA256,
        "predecessor_contract_sha256": REV6_PREDECESSOR_CONTRACT_SHA256,
        "validated_30_day_raw_manifest_path": "/immutable/raw-manifest.json",
        "validated_30_day_raw_manifest_sha256": SHA,
        "raw_episode_count": 12,
        "raw_episode_scan_count": 12,
        "all_episode_collision_census_receipt_sha256": SHA,
        "feature_schema_sha256": SHA,
        "target_schema_sha256": SHA,
        "checklist_provenance_schema_sha256": SHA,
        "zero_bypass_receipt_sha256": SHA,
        "predecessor_schema_freeze_receipt_sha256": SHA,
        "predecessor_corpus_and_refeaturization_receipt_sha256s": [SHA],
        "eligibility_predicate": scope["row_eligibility"]["predicate"],
        "total_actor_visible_decisions_scanned": 20,
        "included_actor_visible_decisions": 7,
        "excluded_nonqualifying_acting_seat_decisions": 11,
        "excluded_missing_or_malformed_setup_deck_decisions": 2,
        "included_row_manifest_sha256": SHA,
        "excluded_row_reason_counts_sha256": SHA,
        "acting_deck_multiset_digest_algorithm": scope["list_variant_stratification"][
            "digest_algorithm"
        ],
        "list_variant_strata_manifest_sha256": SHA,
        "source_day_group_disjoint_split_manifest_sha256s": {
            "train": SHA,
            "validation": "sha256:" + "2" * 64,
            "evaluation": "sha256:" + "3" * 64,
        },
        "reused_artifact_sha256_inventory": {
            "raw_manifest": SHA,
            REV7_CENSUS_PHYSICAL_INVENTORY_KEY: SHA,
        },
        "filtered_or_rematerialized_artifact_sha256_inventory": {},
        "sealed_at_utc": "2026-08-12T20:00:00Z",
    }
    row.update(fixed)
    assert set(row) == set(scope["migration_receipt"]["required_fields"])
    return row


def test_live_revision_7_authority_and_reuse_boundary_are_exact() -> None:
    contract = load_revision_7_contract()
    assert contract["goal_revision"] == 7
    assert REV7_CONTRACT_SHA256.endswith("98e39d771569bdf778885848fc61732275c89c517e057a34bd3006144c23bdd1")
    assert REV7_GOAL_GATEWAY_SHA256.endswith("d275696bb8322c1741463d63b4506d9b04f1252c3c6cdbe7559c999b55b83da7")


def test_rev6_receipt_remains_rev6_and_validates_under_rev7() -> None:
    receipt = _migration()
    checked = validate_revision_6_scope_migration_under_revision_7(receipt)
    assert checked["goal_revision"] == 6
    assert checked["goal_contract_sha256"] == REV6_CONTRACT_SHA256


def test_rev7_current_builder_emits_the_same_closed_rev6_receipt() -> None:
    expected = _migration()
    built = build_revision_6_scope_migration_under_revision_7(
        validated_30_day_raw_manifest_path=str(
            expected["validated_30_day_raw_manifest_path"]
        ),
        validated_30_day_raw_manifest_sha256=str(
            expected["validated_30_day_raw_manifest_sha256"]
        ),
        raw_episode_count=int(expected["raw_episode_count"]),
        raw_episode_scan_count=int(expected["raw_episode_scan_count"]),
        all_episode_collision_census_receipt_sha256=str(
            expected["all_episode_collision_census_receipt_sha256"]
        ),
        feature_schema_sha256=str(expected["feature_schema_sha256"]),
        target_schema_sha256=str(expected["target_schema_sha256"]),
        checklist_provenance_schema_sha256=str(
            expected["checklist_provenance_schema_sha256"]
        ),
        zero_bypass_receipt_sha256=str(expected["zero_bypass_receipt_sha256"]),
        predecessor_schema_freeze_receipt_sha256=str(
            expected["predecessor_schema_freeze_receipt_sha256"]
        ),
        predecessor_corpus_and_refeaturization_receipt_sha256s=expected[
            "predecessor_corpus_and_refeaturization_receipt_sha256s"
        ],
        total_actor_visible_decisions_scanned=int(
            expected["total_actor_visible_decisions_scanned"]
        ),
        included_actor_visible_decisions=int(
            expected["included_actor_visible_decisions"]
        ),
        excluded_nonqualifying_acting_seat_decisions=int(
            expected["excluded_nonqualifying_acting_seat_decisions"]
        ),
        excluded_missing_or_malformed_setup_deck_decisions=int(
            expected["excluded_missing_or_malformed_setup_deck_decisions"]
        ),
        included_row_manifest_sha256=str(expected["included_row_manifest_sha256"]),
        excluded_row_reason_counts_sha256=str(
            expected["excluded_row_reason_counts_sha256"]
        ),
        list_variant_strata_manifest_sha256=str(
            expected["list_variant_strata_manifest_sha256"]
        ),
        source_day_group_disjoint_split_manifest_sha256s=expected[
            "source_day_group_disjoint_split_manifest_sha256s"
        ],
        reused_artifact_sha256_inventory=expected[
            "reused_artifact_sha256_inventory"
        ],
        filtered_or_rematerialized_artifact_sha256_inventory=expected[
            "filtered_or_rematerialized_artifact_sha256_inventory"
        ],
        sealed_at_utc=str(expected["sealed_at_utc"]),
    )
    assert built == expected


def test_rev7_current_writer_is_create_only_and_keeps_rev6_stamp(tmp_path) -> None:
    target = tmp_path / "scope-migration.json"
    published = write_revision_6_scope_migration_under_revision_7_create_only(
        target, _migration()
    )
    assert published["goal_revision"] == 6
    assert published["schema"] == REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA
    assert target.stat().st_size == published["size_bytes"]
    assert target.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(Rev7PredecessorCompatibilityError, match="already exists"):
        write_revision_6_scope_migration_under_revision_7_create_only(
            target, _migration()
        )


def test_rev6_receipt_cannot_be_blindly_retagged_to_rev7() -> None:
    receipt = _migration()
    receipt["goal_revision"] = 7
    receipt["goal_contract_sha256"] = REV7_CONTRACT_SHA256
    with pytest.raises(Rev7PredecessorCompatibilityError, match="retagged"):
        validate_revision_6_scope_migration_under_revision_7(receipt)


def test_rev6_receipt_has_closed_field_inventory_and_partitioned_counts() -> None:
    extra = _migration()
    extra["compatibility_receipt_sha256"] = SHA
    with pytest.raises(Rev7PredecessorCompatibilityError, match="inventory mismatch"):
        validate_revision_6_scope_migration_under_revision_7(extra)

    bad_counts = copy.deepcopy(_migration())
    bad_counts["included_actor_visible_decisions"] = 8
    with pytest.raises(Rev7PredecessorCompatibilityError, match="do not partition"):
        validate_revision_6_scope_migration_under_revision_7(bad_counts)


def _rev5_bundle(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, ...]:
    raw_manifest = {
        "total_raw_zip_json_members": 12,
        "total_validated_episodes": 12,
        "source_manifest_provenance": [{"source": "a"}],
        "source_receipt_day_coverage": [{"date": "2026-07-13"}],
        "episode_deduplication": {"unique": 12},
    }
    raw_receipt = {
        "schema": R298_RAW_CORPUS_RECEIPT_SCHEMA,
        "status": "passed",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": 5,
        "root_handoff_revision": 303,
        "owner_goal_sha256": OWNER_GOAL_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "mechanics_attachment_sha256": MECHANICS_ATTACHMENT_SHA256,
        "raw_expert_corpus_manifest_sha256": canonical_sha256(raw_manifest),
        "window_start_utc": "2026-07-13",
        "window_end_utc": "2026-08-11",
        "distinct_utc_day_count": 30,
        "completed_raw_zip_member_count": 12,
        "completed_validated_episode_count": 12,
        "recollection_authorized": False,
        "source_disjointness": {
            "archive_date_source_sha256_unique": True,
            "episode_identity_unique": True,
            "episode_id_content_unique": True,
            "source_window_blending_permitted": False,
            "training_eligible": False,
        },
        "source_manifest_provenance_sha256": canonical_sha256(
            raw_manifest["source_manifest_provenance"]
        ),
        "source_receipt_day_coverage_sha256": canonical_sha256(
            raw_manifest["source_receipt_day_coverage"]
        ),
        "episode_deduplication_sha256": canonical_sha256(
            raw_manifest["episode_deduplication"]
        ),
        "resource_observation": {"peak": 1},
        "run_identity_sha256": SHA,
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "production_activation": False,
            "inzi_mutation": False,
            "archive_mutation": False,
        },
    }
    schema_manifest = {
        "schema": R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
        "feature_schema_sha256": "sha256:" + "4" * 64,
        "target_schema_sha256": "sha256:" + "5" * 64,
        "checklist_provenance_schema_sha256": "sha256:" + "6" * 64,
    }
    zero_bypass = {"sealed": True}
    schema_manifest_sha = canonical_sha256(schema_manifest)
    zero_bypass_sha = canonical_sha256(zero_bypass)
    freeze = {
        "schema": "poke_bot.alakazam_rule_derivative_schema_freeze_receipt/v1",
        "goal_contract_path": REV6_CONTRACT_PATH,
        "goal_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "goal_revision": 5,
        "feature_schema_id": "feature/v1",
        "feature_schema_sha256": schema_manifest["feature_schema_sha256"],
        "target_schema_id": "target/v1",
        "target_schema_sha256": schema_manifest["target_schema_sha256"],
        "checklist_provenance_schema_id": "checklist/v1",
        "checklist_provenance_schema_sha256": schema_manifest[
            "checklist_provenance_schema_sha256"
        ],
        "public_catalog_manifest_sha256": SHA,
        "canonical_simulator_sha256": SHA,
        "new_branch_inventory": ["public_rule"],
        "census_supported_branch_inventory": [],
        "unsupported_zero_inert_branch_inventory": ["public_rule"],
        "q3_bench_only": True,
        "q5_q6_trace_only_zero": True,
        "public_information_contract_passed": True,
        "zero_bypass_receipt_sha256": zero_bypass_sha,
        "layer_off_bit_identical_baseline_logits": True,
        "frozen_at_utc": "2026-08-12T20:00:00Z",
    }
    preflight = {
        "schema": "poke_bot.alakazam_rule_derivative_r303_materialization_preflight/v1",
        "status": "ready_for_exact_30_day_elmo_refeaturization_diagnostic_targets_only",
        "goal_contract_path": REV6_CONTRACT_PATH,
        "goal_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "goal_revision": 5,
        "root_handoff_revision": 303,
        "raw_expert_corpus_manifest_sha256": canonical_sha256(raw_manifest),
        "raw_expert_corpus_receipt_sha256": canonical_sha256(raw_receipt),
        "frozen_schema_manifest_sha256": schema_manifest_sha,
        "zero_bypass_receipt_sha256": zero_bypass_sha,
        "schema_freeze_receipt_sha256": canonical_sha256(freeze),
        "utc_window": ["2026-07-13", "2026-08-11"],
        "utc_partition_count": 30,
        "required_record_surfaces": ["diagnostic"],
        "selected_action_target_mode": (
            "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator"
        ),
        "opponent_belief_target_mode": "unavailable_masked_no_loss",
        "source_day_group_seed_disjoint_split_manifest_required": True,
        "physical_shard_max_bytes": 96 * 1024 * 1024,
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }
    monkeypatch.setattr(compat, "validate_raw_corpus_manifest", lambda _value: None)
    monkeypatch.setattr(
        compat,
        "validate_frozen_schema_gate",
        lambda _manifest, _bypass: (schema_manifest_sha, zero_bypass_sha),
    )
    return raw_manifest, raw_receipt, schema_manifest, zero_bypass, freeze, preflight


def test_rev7_bundle_crosslinks_rev5_and_rev6_without_retagging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_manifest, raw_receipt, schema_manifest, zero_bypass, freeze, preflight = (
        _rev5_bundle(monkeypatch)
    )
    migration = _migration()
    migration.update(
        {
            "validated_30_day_raw_manifest_sha256": canonical_sha256(raw_manifest),
            "feature_schema_sha256": schema_manifest["feature_schema_sha256"],
            "target_schema_sha256": schema_manifest["target_schema_sha256"],
            "checklist_provenance_schema_sha256": schema_manifest[
                "checklist_provenance_schema_sha256"
            ],
            "zero_bypass_receipt_sha256": canonical_sha256(zero_bypass),
            "predecessor_schema_freeze_receipt_sha256": canonical_sha256(freeze),
        }
    )
    result = compat.validate_revision_7_predecessor_bundle(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass,
        schema_freeze_receipt=freeze,
        materialization_preflight=preflight,
        corpus_scope_migration_receipt=migration,
    )
    assert result["current_goal_revision"] == 7
    assert result["historical_receipts_retagged_or_rewritten"] is False
    assert result["recomputation_required_due_only_to_revision_7"] is False
    assert result["training_runtime_service_transfer_or_activation_authority"] is False


def test_running_census_compatibility_does_not_invent_a_later_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_manifest, raw_receipt, schema_manifest, zero_bypass, _freeze, _preflight = (
        _rev5_bundle(monkeypatch)
    )
    checked = compat.validate_revision_5_census_predecessors_under_revision_7(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass,
    )
    assert set(checked) == {
        "raw_expert_corpus_manifest_sha256",
        "raw_expert_corpus_receipt_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
    }


def test_post_census_bundle_distinguishes_file_sha_from_canonical_object_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    raw_manifest, raw_receipt, schema_manifest, zero_bypass, freeze, preflight = (
        _rev5_bundle(monkeypatch)
    )
    census = {"schema": "closed-r5-census", "status": "passed", "count": 12}
    census_path = tmp_path / "receipt.json"
    census_path.write_text(
        json.dumps(census, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    assert compat._file_sha256(census_path) != canonical_sha256(census)
    migration = _migration()
    migration.update(
        {
            "validated_30_day_raw_manifest_sha256": canonical_sha256(raw_manifest),
            "feature_schema_sha256": schema_manifest["feature_schema_sha256"],
            "target_schema_sha256": schema_manifest["target_schema_sha256"],
            "checklist_provenance_schema_sha256": schema_manifest[
                "checklist_provenance_schema_sha256"
            ],
            "zero_bypass_receipt_sha256": canonical_sha256(zero_bypass),
            "predecessor_schema_freeze_receipt_sha256": canonical_sha256(freeze),
            "all_episode_collision_census_receipt_sha256": canonical_sha256(census),
        }
    )
    migration["reused_artifact_sha256_inventory"][
        REV7_CENSUS_PHYSICAL_INVENTORY_KEY
    ] = compat._file_sha256(census_path)
    bridge = {
        "raw_expert_corpus_manifest_sha256": canonical_sha256(raw_manifest),
        "raw_expert_corpus_receipt_sha256": canonical_sha256(raw_receipt),
        "frozen_schema_manifest_sha256": canonical_sha256(schema_manifest),
        "zero_bypass_receipt_sha256": canonical_sha256(zero_bypass),
        "collision_census_receipt_sha256": canonical_sha256(census),
    }
    monkeypatch.setattr(
        compat, "validate_revision_5_census_validation_receipt", lambda value: dict(value)
    )
    result = compat.validate_revision_7_predecessor_bundle(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass,
        schema_freeze_receipt=freeze,
        materialization_preflight=preflight,
        corpus_scope_migration_receipt=migration,
        census_validation_receipt=bridge,
        collision_census_receipt=census,
        collision_census_receipt_path=census_path,
    )
    assert result["revision_5_census_validation_receipt_sha256"] == canonical_sha256(
        bridge
    )
    assert result["revision_5_collision_census_canonical_sha256"] == canonical_sha256(
        census
    )
    assert result["revision_5_collision_census_physical_file_sha256"] == compat._file_sha256(
        census_path
    )


def test_immediate_census_completion_bridge_needs_no_materialization_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    raw_manifest, raw_receipt, schema_manifest, zero_bypass, _freeze, _preflight = (
        _rev5_bundle(monkeypatch)
    )
    census = {"schema": "closed-r5-census", "status": "passed"}
    census_path = tmp_path / "receipt.json"
    census_path.write_text(
        json.dumps(census, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    bridge = {
        "raw_expert_corpus_manifest_sha256": canonical_sha256(raw_manifest),
        "raw_expert_corpus_receipt_sha256": canonical_sha256(raw_receipt),
        "frozen_schema_manifest_sha256": canonical_sha256(schema_manifest),
        "zero_bypass_receipt_sha256": canonical_sha256(zero_bypass),
        "collision_census_receipt_sha256": canonical_sha256(census),
    }
    monkeypatch.setattr(
        compat, "validate_revision_5_census_validation_receipt", lambda value: dict(value)
    )
    checked = compat.validate_revision_5_census_completion_under_revision_7(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass,
        census_validation_receipt=bridge,
        collision_census_receipt_path=census_path,
    )
    assert checked["materialization_preflight_claimed_or_required"] is False
    assert checked["revision_5_collision_census_canonical_sha256"] == canonical_sha256(
        census
    )
    assert checked["revision_5_collision_census_physical_file_sha256"] == compat._file_sha256(
        census_path
    )


def test_exact_24_process_plan_covers_all_30_days_once_without_claiming_evidence() -> None:
    plan = revision_7_parallel_execution_plan(workers=EXACT_WORKER_COUNT)
    days = [day for lane in plan["day_lanes"] for day in lane]
    assert len(days) == len(set(days)) == 30
    assert min(days) == "2026-07-13"
    assert max(days) == "2026-08-11"
    assert plan["experiment_ram_ceiling_bytes"] == 96 * 1024**3
    assert plan["aggregate_runtime_peak_evidence_required_for_completion_receipt"] is True
    assert plan["plan_itself_is_not_runtime_resource_evidence"] is True
    assert plan["training_runtime_service_transfer_or_activation_authority"] is False


@pytest.mark.parametrize("workers", [1, 23, 25, True])
def test_rev7_whole_window_plan_refuses_nonexact_worker_count(workers: object) -> None:
    with pytest.raises(Rev7PredecessorCompatibilityError, match="exactly 24"):
        revision_7_parallel_execution_plan(workers=workers)  # type: ignore[arg-type]
