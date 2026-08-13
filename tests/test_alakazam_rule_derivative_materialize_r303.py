"""Focused guards for the isolated r303 diagnostic record path."""

from __future__ import annotations

import gzip
import json
import zipfile

import pytest

from poke_bot.alakazam_rule_derivative_materialize_r303 import (
    R303MaterializationError,
    R303_RECORD_SCHEMA,
    _ShardWriter,
    _acting_seat_eligibility,
    _diagnostic_selected_target,
    _sha256_file,
    _verify_archive,
    bounded_day_parallel_plan_r303,
    build_revision_6_scope_migration_from_inventory,
    conservative_r303_process_pool_resource_observation,
    probe_r303_whole_day_process_pool,
)
from poke_bot.alakazam_collision_census_r298 import canonical_sha256
from poke_bot.alakazam_rule_derivative_predecessor_compat_rev7 import (
    REV7_CENSUS_PHYSICAL_INVENTORY_KEY,
)


def test_r303_parallel_plan_covers_each_exact_day_once() -> None:
    plan = bounded_day_parallel_plan_r303(workers=24)
    days = [day for lane in plan["per_lane_assignment"].values() for day in lane]
    assert plan["all_days_exactly_once"] is True
    assert plan["worker_execution"] == "bounded_whole_day_process_pool"
    assert plan["maximum_whole_day_workers"] == 24
    assert len(days) == len(set(days)) == 30
    assert days[0] == "2026-07-13"
    assert "2026-08-11" in days


def test_r303_parallel_plan_refuses_unbounded_worker_count() -> None:
    with pytest.raises(R303MaterializationError, match="1 through 24"):
        bounded_day_parallel_plan_r303(workers=25)


def test_r303_actual_24_process_startup_and_pickle_probe() -> None:
    probe = probe_r303_whole_day_process_pool(
        workers=24,
        catalog=None,
        raw_manifest={"fixture": "raw"},
        schema_manifest={"fixture": "schema"},
    )

    assert probe["status"] == "passed_actual_whole_day_process_startup_and_pickle"
    assert probe["worker_count"] == 24
    assert probe["unique_child_process_count"] == 24
    assert len(probe["worker_process_ids"]) == len(set(probe["worker_process_ids"])) == 24


def test_diagnostic_targets_never_arm_a_training_mask_without_sealed_ledger() -> None:
    target, vectors = _diagnostic_selected_target({}, [], catalog=None)
    assert target["target_only"] is True
    assert target["policy_feature_eligible"] is False
    assert target["training_eligible"] is False
    assert vectors["target_training_eligible"] is False
    assert vectors["opponent_belief_target_mode"] == "unavailable_masked_no_loss"


def test_revision_6_filter_uses_only_same_acting_seat_literal_card_743_membership() -> None:
    decks = {0: (1, 2, 743), 1: (1, 2, 3)}
    eligible, reason, digest = _acting_seat_eligibility(decks, 0)
    assert eligible is True
    assert reason == "eligible_literal_same_acting_seat_setup_deck_contains_card_743"
    assert digest is not None

    eligible, reason, digest = _acting_seat_eligibility(decks, 1)
    assert eligible is False
    assert reason == "same_acting_seat_setup_deck_lacks_card_743"
    assert digest is not None


def test_revision_6_filter_accepts_a_single_743_tech_copy_without_pilot_match() -> None:
    # The literal membership predicate has no exact-list, digest, archetype,
    # or minimum-line requirement.  A one-copy tech list still qualifies.
    tech_variant = tuple([743] + [999] * 11)
    eligible, reason, digest = _acting_seat_eligibility({0: tech_variant}, 0)

    assert eligible is True
    assert reason == "eligible_literal_same_acting_seat_setup_deck_contains_card_743"
    assert digest is not None


def test_revision_6_filter_excludes_missing_setup_deck_without_guessing() -> None:
    eligible, reason, digest = _acting_seat_eligibility({}, 0)
    assert eligible is False
    assert reason == "missing_or_malformed_same_acting_seat_setup_deck"
    assert digest is None


def test_content_addressed_shard_is_gzip_valid_and_below_cap(tmp_path) -> None:
    day_root = tmp_path / "day=2026-07-13"
    day_root.mkdir()
    writer = _ShardWriter(day_root, day="2026-07-13")
    writer.write(
        {
            "schema": R303_RECORD_SCHEMA,
            "rule_head_target_vectors": {"target_training_eligible": False},
            "payload": "diagnostic-only",
        }
    )
    identities = writer.close()
    assert len(identities) == 1
    identity = identities[0]
    assert identity.relative_path.startswith("day=2026-07-13/shards/sha256-")
    assert identity.size_bytes <= 96 * 1024 * 1024
    with gzip.open(tmp_path / identity.relative_path, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["schema"] == R303_RECORD_SCHEMA


def test_revision_6_permits_a_fully_audited_day_with_zero_eligible_rows(tmp_path) -> None:
    day_root = tmp_path / "day=2026-07-13"
    day_root.mkdir()
    writer = _ShardWriter(day_root, day="2026-07-13")
    assert writer.close() == ()


def test_daily_archive_allows_only_value_free_manifest_csv_alongside_episode_json(
    tmp_path,
) -> None:
    archive_path = tmp_path / "episodes.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("episode-0001.json", '{"id":"episode-1"}')
        archive.writestr("manifest.csv", "episode_id\nepisode-1\n")
    archive = {
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
        "zip_json_member_count": 1,
    }

    _verify_archive(archive, archive_path)


def test_process_pool_resource_charge_uses_unique_process_peak_sum() -> None:
    observation = conservative_r303_process_pool_resource_observation(
        coordinator={"process_id": 100, "experiment_ram_peak_bytes": 10},
        workers=[
            {"process_id": 101, "experiment_ram_peak_bytes": 20},
            {"process_id": 101, "experiment_ram_peak_bytes": 25},
            {"process_id": 102, "experiment_ram_peak_bytes": 30},
        ],
        requested_worker_count=2,
        startup_probe=None,
    )

    assert observation["experiment_ram_peak_bytes"] == 65
    assert observation["worker_process_count"] == 2
    assert observation["worker_peak_rss_charge_bytes"] == 55
    assert observation["per_process_peak_rss_bytes"] == {
        "100": 10,
        "101": 25,
        "102": 30,
    }


def test_scope_migration_keeps_canonical_and_physical_census_identities_distinct(
    tmp_path,
) -> None:
    root = tmp_path / "scope"
    root.mkdir()
    variants = {"schema": "variant/v1", "rows": []}
    split = {"schema": "split/v1", "train": [], "validation": [], "evaluation": []}
    inventory = {
        "validated_30_day_raw_manifest_path": "/immutable/raw.json",
        "validated_30_day_raw_manifest_sha256": "sha256:" + "1" * 64,
        "raw_episode_count": 12,
        "raw_episode_scan_count": 12,
        "total_actor_visible_decisions_scanned": 20,
        "included_actor_visible_decisions": 7,
        "excluded_row_reason_counts": {
            "same_acting_seat_setup_deck_lacks_card_743": 11,
            "missing_or_malformed_same_acting_seat_setup_deck": 2,
        },
        "list_variant_strata_manifest_sha256": canonical_sha256(variants),
        "source_day_group_disjoint_split_manifest_sha256": canonical_sha256(split),
        "same_seat_card_743_filter_source_sha256": "sha256:" + "2" * 64,
        "same_seat_card_743_filter_test_sha256": "sha256:" + "3" * 64,
    }
    for name, value in (
        ("SCOPE_INVENTORY.json", inventory),
        ("LIST_VARIANT_STRATA_MANIFEST.json", variants),
        ("SPLIT_MANIFEST.json", split),
    ):
        (root / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    census = {"goal_revision": 5, "status": "passed", "decisions": 20}
    census_path = tmp_path / "census-receipt.json"
    census_path.write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")
    schema = {
        "feature_schema_sha256": "sha256:" + "4" * 64,
        "target_schema_sha256": "sha256:" + "5" * 64,
        "checklist_provenance_schema_sha256": "sha256:" + "6" * 64,
    }
    bypass = {"frozen_schema_manifest_sha256": canonical_sha256(schema)}
    freeze = {"zero_bypass_receipt_sha256": canonical_sha256(bypass)}

    receipt = build_revision_6_scope_migration_from_inventory(
        root,
        schema_manifest=schema,
        zero_bypass_receipt=bypass,
        schema_freeze_receipt=freeze,
        all_episode_collision_census_receipt_path=census_path,
        predecessor_corpus_and_refeaturization_receipt_sha256s=[
            "sha256:" + "7" * 64
        ],
        sealed_at_utc="2026-08-12T20:00:00Z",
    )

    assert receipt["all_episode_collision_census_receipt_sha256"] == canonical_sha256(
        census
    )
    assert receipt["reused_artifact_sha256_inventory"][
        REV7_CENSUS_PHYSICAL_INVENTORY_KEY
    ] == _sha256_file(census_path)
    assert canonical_sha256(census) != _sha256_file(census_path)
