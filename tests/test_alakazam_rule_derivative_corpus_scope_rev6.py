"""Strict tests for the revision-6 corpus-scope migration receipt."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import poke_bot.alakazam_rule_derivative_corpus_scope_rev6 as scope


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _build_receipt() -> dict[str, object]:
    return scope.build_corpus_scope_migration_receipt(
        validated_30_day_raw_manifest_path="/immutable/elmo/raw-manifest.json",
        validated_30_day_raw_manifest_sha256=_sha("1"),
        raw_episode_count=10,
        raw_episode_scan_count=10,
        all_episode_collision_census_receipt_sha256=_sha("2"),
        feature_schema_sha256=_sha("3"),
        target_schema_sha256=_sha("4"),
        checklist_provenance_schema_sha256=_sha("5"),
        zero_bypass_receipt_sha256=_sha("6"),
        predecessor_schema_freeze_receipt_sha256=_sha("7"),
        predecessor_corpus_and_refeaturization_receipt_sha256s=[_sha("8"), _sha("9")],
        total_actor_visible_decisions_scanned=100,
        included_actor_visible_decisions=40,
        excluded_nonqualifying_acting_seat_decisions=55,
        excluded_missing_or_malformed_setup_deck_decisions=5,
        included_row_manifest_sha256=_sha("a"),
        excluded_row_reason_counts_sha256=_sha("b"),
        list_variant_strata_manifest_sha256=_sha("c"),
        source_day_group_disjoint_split_manifest_sha256s={
            "train": _sha("d"),
            "validation": _sha("e"),
            "evaluation": _sha("f"),
        },
        reused_artifact_sha256_inventory={
            "raw_manifest": _sha("1"),
            "schema_freeze_receipt": _sha("7"),
        },
        filtered_or_rematerialized_artifact_sha256_inventory={},
        sealed_at_utc="2026-08-12T20:15:00Z",
    )


def test_rev6_spec_is_the_exact_canonical_46_field_inventory() -> None:
    spec = scope.corpus_scope_migration_receipt_spec()

    assert spec["schema"] == scope.REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA
    assert len(spec["required_fields"]) == 46
    assert len(spec["required_fields"]) == len(set(spec["required_fields"]))
    assert spec["fixed_values"]["eligibility_card_id"] == 743
    assert spec["fixed_values"]["exact_pilot_match_required"] is False
    assert spec["fixed_values"]["tech_inclusion_exception_exists"] is False
    assert spec["fixed_values"]["opponent_nonqualifying_decisions_included"] is False


def test_builder_emits_all_episode_audit_and_acting_seat_743_scope() -> None:
    receipt = _build_receipt()

    assert set(receipt) == set(scope.corpus_scope_migration_receipt_spec()["required_fields"])
    assert receipt["all_raw_episodes_scanned"] is True
    assert receipt["raw_episode_count"] == receipt["raw_episode_scan_count"] == 10
    assert receipt["acting_seat_only"] is True
    assert receipt["eligibility_card_id"] == 743
    assert receipt["exact_pilot_match_required"] is False
    assert receipt["tech_inclusion_exception_exists"] is False
    assert receipt["opponent_nonqualifying_decisions_included"] is False
    assert receipt["all_included_rows_have_acting_deck_multiset_sha256"] is True
    assert receipt["recomputation_required_due_only_to_revision_change"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("all_raw_episodes_scanned", False),
        ("no_episode_deck_seat_or_archetype_prefilter", False),
        ("eligibility_card_id", 742),
        ("acting_seat_only", False),
        ("exact_pilot_match_required", True),
        ("tech_inclusion_exception_exists", True),
        ("opponent_nonqualifying_decisions_included", True),
        ("all_included_rows_have_acting_deck_contains_743", False),
        ("all_included_rows_have_acting_deck_multiset_sha256", False),
        ("recomputation_required_due_only_to_revision_change", True),
    ],
)
def test_validator_rejects_scope_drift(field: str, value: object) -> None:
    receipt = _build_receipt()
    receipt[field] = value

    with pytest.raises(scope.Rev6CorpusScopeError):
        scope.validate_corpus_scope_migration_receipt(receipt)


def test_validator_rejects_incomplete_decision_partition() -> None:
    receipt = _build_receipt()
    receipt["included_actor_visible_decisions"] = 39

    with pytest.raises(scope.Rev6CorpusScopeError, match="do not partition"):
        scope.validate_corpus_scope_migration_receipt(receipt)


def test_validator_requires_named_source_day_group_splits() -> None:
    receipt = _build_receipt()
    receipt["source_day_group_disjoint_split_manifest_sha256s"] = {
        "train": _sha("d"),
        "validation": _sha("e"),
    }

    with pytest.raises(scope.Rev6CorpusScopeError, match="exactly train"):
        scope.validate_corpus_scope_migration_receipt(receipt)


def test_create_only_writer_publishes_canonical_bytes_once(tmp_path: Path) -> None:
    receipt = _build_receipt()
    target = tmp_path / "corpus-scope-migration-receipt.json"

    identity = scope.write_corpus_scope_migration_receipt_create_only(target, receipt)

    assert identity["path"] == str(target.resolve())
    assert identity["sha256"] == scope.sha256_file(target)
    assert identity["size_bytes"] == target.stat().st_size
    assert json.loads(target.read_text(encoding="utf-8")) == receipt
    with pytest.raises(scope.Rev6CorpusScopeError, match="already exists"):
        scope.write_corpus_scope_migration_receipt_create_only(target, receipt)


def test_validator_rejects_extra_field_and_stale_contract() -> None:
    receipt = _build_receipt()
    with_extra = copy.deepcopy(receipt)
    with_extra["uncontracted"] = True
    with pytest.raises(scope.Rev6CorpusScopeError, match="inventory mismatch"):
        scope.validate_corpus_scope_migration_receipt(with_extra)

    stale = copy.deepcopy(receipt)
    stale["goal_contract_sha256"] = _sha("0")
    with pytest.raises(scope.Rev6CorpusScopeError, match="contract identity"):
        scope.validate_corpus_scope_migration_receipt(stale)


def test_public_api_exports_the_error_and_producer() -> None:
    namespace: dict[str, object] = {}
    exec("from poke_bot.alakazam_rule_derivative_corpus_scope_rev6 import *", namespace)

    assert namespace["Rev6CorpusScopeError"] is scope.Rev6CorpusScopeError
    assert namespace["build_corpus_scope_migration_receipt"] is (
        scope.build_corpus_scope_migration_receipt
    )
    assert namespace["write_corpus_scope_migration_receipt_create_only"] is (
        scope.write_corpus_scope_migration_receipt_create_only
    )
