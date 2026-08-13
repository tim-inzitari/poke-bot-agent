"""Revision-7 compatibility for immutable revision-5/6 evidence.

Revision 7 changes only the future checkpoint warm-start composition.  The
canonical contract explicitly keeps the revision-6 corpus, schema, census,
and migration artifacts reusable without recomputation.  This module is the
consumer-side compatibility seam for that rule:

* it authenticates the *current* revision-7 gateway and contract;
* it validates old revision-5 and revision-6 receipts under their original
  identities (old receipts are never retagged or rewritten); and
* it returns in-memory compatibility results with no training, runtime,
  service, transfer, or activation authority; and
* after the census exists, it can create the unchanged closed rev6 migration
  receipt under exclusive filesystem semantics.

There is deliberately no writer for a new rev7 compatibility receipt.  The
only writer below emits the already-canonical rev6 46-field receipt with its
original rev6 stamp.  Inventing another persistent receipt here would be a
competing authority rather than a compatibility check.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from .alakazam_collision_census_r298 import (
    MECHANICS_ATTACHMENT_SHA256,
    OWNER_GOAL_SHA256,
    R298_OWNER_REVISION,
    R298_RAW_CORPUS_RECEIPT_SCHEMA,
    RULE_DERIVATIVE_CONTRACT_SHA256,
    RULE_DERIVATIVE_GATEWAY_SHA256,
    canonical_sha256,
    validate_frozen_schema_gate,
    validate_raw_corpus_manifest,
    validate_revision_5_census_validation_receipt,
    validate_revision_5_predecessor_classification,
)
from .alakazam_rule_derivative_corpus_scope_rev6 import (
    REV6_CONTRACT_PATH,
    REV6_CONTRACT_SHA256,
    REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA,
    REV6_GOAL_REVISION,
    REV6_PREDECESSOR_CONTRACT_SHA256,
    REV6_PREDECESSOR_GATEWAY_SHA256,
)

REV7_GOAL_REVISION: Final = 7
REV7_ROOT_HANDOFF_REVISION: Final = 303
REV7_CONTRACT_PATH: Final = "goals/alakazam-elmo-rule-derivative/contract.json"
REV7_GOAL_PATH: Final = "goals/alakazam-elmo-rule-derivative/GOAL.md"
REV7_CONTRACT_SHA256: Final = (
    "sha256:98e39d771569bdf778885848fc61732275c89c517e057a34bd3006144c23bdd1"
)
REV7_GOAL_GATEWAY_SHA256: Final = (
    "sha256:d275696bb8322c1741463d63b4506d9b04f1252c3c6cdbe7559c999b55b83da7"
)
REV7_PREDECESSOR_VALIDATION_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_rev7_predecessor_validation/v1"
)
REV7_PARALLEL_PLAN_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_rev7_parallel_execution_plan/v1"
)
REV7_CENSUS_PHYSICAL_INVENTORY_KEY: Final = (
    "revision_5_all_episode_collision_census_receipt_file"
)
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_FILE: Final = PROJECT_ROOT / REV7_CONTRACT_PATH
GOAL_FILE: Final = PROJECT_ROOT / REV7_GOAL_PATH
EXACT_WINDOW_START_UTC: Final = "2026-07-13"
EXACT_WINDOW_END_UTC: Final = "2026-08-11"
EXACT_UTC_PARTITION_COUNT: Final = 30
EXACT_WORKER_COUNT: Final = 24
EXPERIMENT_RAM_LIMIT_BYTES: Final = 96 * 1024**3
_SPLITS: Final = ("train", "validation", "evaluation")
_RAW_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "status",
        "owner_revision",
        "goal_revision",
        "root_handoff_revision",
        "owner_goal_sha256",
        "rule_derivative_contract_sha256",
        "rule_derivative_gateway_sha256",
        "revision_5_predecessor_classification",
        "mechanics_attachment_sha256",
        "raw_expert_corpus_manifest_sha256",
        "window_start_utc",
        "window_end_utc",
        "distinct_utc_day_count",
        "completed_raw_zip_member_count",
        "completed_validated_episode_count",
        "recollection_authorized",
        "source_disjointness",
        "source_manifest_provenance_sha256",
        "source_receipt_day_coverage_sha256",
        "episode_deduplication_sha256",
        "resource_observation",
        "run_identity_sha256",
        "runtime_authority",
    }
)
_SCHEMA_FREEZE_FIELDS: Final = frozenset(
    {
        "schema",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "feature_schema_id",
        "feature_schema_sha256",
        "target_schema_id",
        "target_schema_sha256",
        "checklist_provenance_schema_id",
        "checklist_provenance_schema_sha256",
        "public_catalog_manifest_sha256",
        "canonical_simulator_sha256",
        "new_branch_inventory",
        "census_supported_branch_inventory",
        "unsupported_zero_inert_branch_inventory",
        "q3_bench_only",
        "q5_q6_trace_only_zero",
        "public_information_contract_passed",
        "zero_bypass_receipt_sha256",
        "layer_off_bit_identical_baseline_logits",
        "frozen_at_utc",
    }
)
_PREFLIGHT_FIELDS: Final = frozenset(
    {
        "schema",
        "status",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "root_handoff_revision",
        "raw_expert_corpus_manifest_sha256",
        "raw_expert_corpus_receipt_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "schema_freeze_receipt_sha256",
        "utc_window",
        "utc_partition_count",
        "required_record_surfaces",
        "selected_action_target_mode",
        "opponent_belief_target_mode",
        "source_day_group_seed_disjoint_split_manifest_required",
        "physical_shard_max_bytes",
        "runtime_authority",
    }
)


class Rev7PredecessorCompatibilityError(RuntimeError):
    """Current authority or immutable predecessor evidence is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Rev7PredecessorCompatibilityError("value is not canonical JSON") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Rev7PredecessorCompatibilityError(f"cannot read {path}") from exc
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Rev7PredecessorCompatibilityError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Rev7PredecessorCompatibilityError(f"{field} must be a list")
    return list(value)


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise Rev7PredecessorCompatibilityError(f"{field} must be a lowercase SHA-256 identity")
    suffix = value[7:]
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise Rev7PredecessorCompatibilityError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Rev7PredecessorCompatibilityError(f"{field} must be an integer >= {minimum}")
    return value


def _read_stable_json(path: Path, *, field: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise Rev7PredecessorCompatibilityError(f"{field} must be a regular non-symlink file")
    before = path.stat()
    digest = _file_sha256(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Rev7PredecessorCompatibilityError(f"cannot parse {field}") from exc
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or _file_sha256(path) != digest:
        raise Rev7PredecessorCompatibilityError(f"{field} changed while inspected")
    return dict(_mapping(value, field=field)), digest


def load_revision_7_contract() -> dict[str, Any]:
    """Authenticate the settled rev7 gateway/contract pair."""

    contract, digest = _read_stable_json(CONTRACT_FILE, field="revision-7 contract")
    if digest != REV7_CONTRACT_SHA256:
        raise Rev7PredecessorCompatibilityError("revision-7 contract identity is stale or foreign")
    if (
        contract.get("goal_id"),
        contract.get("goal_revision"),
        contract.get("root_handoff_revision"),
    ) != ("alakazam-elmo-rule-derivative", REV7_GOAL_REVISION, REV7_ROOT_HANDOFF_REVISION):
        raise Rev7PredecessorCompatibilityError("revision-7 goal identity drifted")
    if GOAL_FILE.is_symlink() or not GOAL_FILE.is_file():
        raise Rev7PredecessorCompatibilityError("revision-7 goal gateway is unavailable")
    before = GOAL_FILE.stat()
    gateway_digest = _file_sha256(GOAL_FILE)
    after = GOAL_FILE.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or _file_sha256(GOAL_FILE) != gateway_digest:
        raise Rev7PredecessorCompatibilityError("revision-7 goal gateway changed while inspected")
    if gateway_digest != REV7_GOAL_GATEWAY_SHA256:
        raise Rev7PredecessorCompatibilityError("revision-7 goal gateway identity drifted")
    authority = _mapping(contract.get("authority"), field="revision-7 authority")
    if authority.get(
        "immediate_runtime_service_preemption_training_submission_or_self_play_authority"
    ) is not False:
        raise Rev7PredecessorCompatibilityError("revision-7 unexpectedly grants immediate authority")
    revision_7 = _mapping(
        contract.get("revision_7_r195_compatible_weight_warmstart"),
        field="revision-7 delta",
    )
    if (
        revision_7.get("owner_goal_revision"),
        revision_7.get("predecessor_goal_revision"),
        revision_7.get(
            "revision_6_corpus_schema_census_and_migration_artifacts_reusable_without_recompute"
        ),
        revision_7.get("immediate_runtime_or_service_authority"),
    ) != (7, 6, True, False):
        raise Rev7PredecessorCompatibilityError("revision-7 predecessor-reuse boundary drifted")
    revision_6 = _mapping(
        contract.get("revision_6_corpus_scope_and_predecessor_acceptance"),
        field="embedded revision-6 scope",
    )
    unchanged = _mapping(
        revision_6.get("unchanged_revision_5_artifact_acceptance"),
        field="revision-5 predecessor acceptance",
    )
    if (
        unchanged.get("historical_receipt_bytes_may_be_mutated_or_reissued_in_place"),
        unchanged.get("raw_audit_or_schema_recompute_due_only_to_revision_6_identity"),
        unchanged.get("blind_hash_substitution_allowed"),
    ) != (False, False, False):
        raise Rev7PredecessorCompatibilityError("historical evidence migration boundary drifted")
    return contract


def _migration_spec(contract: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    scope = _mapping(
        contract.get("revision_6_corpus_scope_and_predecessor_acceptance"),
        field="embedded revision-6 scope",
    )
    migration = _mapping(scope.get("migration_receipt"), field="migration receipt contract")
    if migration.get("schema") != REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA:
        raise Rev7PredecessorCompatibilityError("embedded revision-6 migration schema drifted")
    fields = _rows(migration.get("required_fields"), field="migration required fields")
    if len(fields) != 46 or len(fields) != len(set(fields)) or any(
        not isinstance(field, str) or not field for field in fields
    ):
        raise Rev7PredecessorCompatibilityError("migration field inventory drifted")
    fixed = dict(_mapping(migration.get("fixed_values"), field="migration fixed values"))
    return fields, fixed


def validate_revision_6_scope_migration_under_revision_7(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an immutable rev6 46-field receipt while rev7 is current.

    The returned row retains its rev6 goal stamp.  A caller that changes it to
    rev7 is rejected instead of silently substituting a current hash.
    """

    contract = load_revision_7_contract()
    fields, fixed = _migration_spec(contract)
    row = dict(_mapping(receipt, field="revision-6 corpus-scope migration receipt"))
    expected = set(fields)
    if set(row) != expected:
        raise Rev7PredecessorCompatibilityError(
            "revision-6 migration field inventory mismatch; "
            f"missing={sorted(expected - set(row))!r} extra={sorted(set(row) - expected)!r}"
        )
    if (
        row.get("schema"),
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("goal_revision"),
        row.get("predecessor_gateway_sha256"),
        row.get("predecessor_contract_sha256"),
    ) != (
        REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA,
        REV6_CONTRACT_PATH,
        REV6_CONTRACT_SHA256,
        REV6_GOAL_REVISION,
        REV6_PREDECESSOR_GATEWAY_SHA256,
        REV6_PREDECESSOR_CONTRACT_SHA256,
    ):
        raise Rev7PredecessorCompatibilityError("revision-6 migration authority was retagged or drifted")
    for field, value in fixed.items():
        if row.get(field) != value:
            raise Rev7PredecessorCompatibilityError(f"revision-6 migration {field} drifted")
    path = row.get("validated_30_day_raw_manifest_path")
    if not isinstance(path, str) or not path or "\x00" in path:
        raise Rev7PredecessorCompatibilityError("validated raw-manifest path is invalid")
    sha_fields = (
        "validated_30_day_raw_manifest_sha256",
        "all_episode_collision_census_receipt_sha256",
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
        "zero_bypass_receipt_sha256",
        "predecessor_schema_freeze_receipt_sha256",
        "included_row_manifest_sha256",
        "excluded_row_reason_counts_sha256",
        "list_variant_strata_manifest_sha256",
    )
    for field in sha_fields:
        _sha(row.get(field), field=field)
    predecessors = [
        _sha(item, field="predecessor corpus/refeaturization identity")
        for item in _rows(
            row.get("predecessor_corpus_and_refeaturization_receipt_sha256s"),
            field="predecessor corpus/refeaturization identities",
        )
    ]
    if not predecessors or len(predecessors) != len(set(predecessors)):
        raise Rev7PredecessorCompatibilityError("predecessor identities must be nonempty and unique")
    splits = dict(
        _mapping(
            row.get("source_day_group_disjoint_split_manifest_sha256s"),
            field="source/day/group split identities",
        )
    )
    if set(splits) != set(_SPLITS):
        raise Rev7PredecessorCompatibilityError("split identities must be exactly train/validation/evaluation")
    for split in _SPLITS:
        _sha(splits[split], field=f"{split} split identity")
    inventories: list[dict[str, str]] = []
    for field, nonempty in (
        ("reused_artifact_sha256_inventory", True),
        ("filtered_or_rematerialized_artifact_sha256_inventory", False),
    ):
        inventory = dict(_mapping(row.get(field), field=field))
        if nonempty and not inventory:
            raise Rev7PredecessorCompatibilityError(f"{field} must be nonempty")
        for name, identity in inventory.items():
            if not isinstance(name, str) or not name or len(name) > 256:
                raise Rev7PredecessorCompatibilityError(f"{field} has an invalid logical name")
            _sha(identity, field=f"{field}.{name}")
        inventories.append(inventory)
    if set(inventories[0]) & set(inventories[1]):
        raise Rev7PredecessorCompatibilityError("an artifact cannot be reused and rematerialized")
    if REV7_CENSUS_PHYSICAL_INVENTORY_KEY not in inventories[0]:
        raise Rev7PredecessorCompatibilityError(
            "reused artifact inventory lacks the physical collision census receipt identity"
        )
    raw_count = _exact_int(row.get("raw_episode_count"), field="raw episode count", minimum=1)
    scan_count = _exact_int(row.get("raw_episode_scan_count"), field="raw episode scan count", minimum=1)
    if raw_count != scan_count:
        raise Rev7PredecessorCompatibilityError("raw audit did not scan every episode exactly once")
    total = _exact_int(row.get("total_actor_visible_decisions_scanned"), field="total decisions")
    included = _exact_int(row.get("included_actor_visible_decisions"), field="included decisions")
    excluded = _exact_int(
        row.get("excluded_nonqualifying_acting_seat_decisions"),
        field="excluded nonqualifying decisions",
    )
    malformed = _exact_int(
        row.get("excluded_missing_or_malformed_setup_deck_decisions"),
        field="excluded malformed decisions",
    )
    if total != included + excluded + malformed:
        raise Rev7PredecessorCompatibilityError("decision inclusion/exclusion counts do not partition scan")
    scope = _mapping(
        contract["revision_6_corpus_scope_and_predecessor_acceptance"],
        field="embedded revision-6 scope",
    )
    if row.get("eligibility_predicate") != _mapping(
        scope.get("row_eligibility"), field="row eligibility"
    ).get("predicate"):
        raise Rev7PredecessorCompatibilityError("acting-seat card-743 predicate drifted")
    if row.get("acting_deck_multiset_digest_algorithm") != _mapping(
        scope.get("list_variant_stratification"), field="list-variant stratification"
    ).get("digest_algorithm"):
        raise Rev7PredecessorCompatibilityError("list-variant digest algorithm drifted")
    timestamp = row.get("sealed_at_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise Rev7PredecessorCompatibilityError("sealed_at_utc must be explicit UTC")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise Rev7PredecessorCompatibilityError("sealed_at_utc is malformed") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Rev7PredecessorCompatibilityError("sealed_at_utc is not UTC")
    row["predecessor_corpus_and_refeaturization_receipt_sha256s"] = predecessors
    row["source_day_group_disjoint_split_manifest_sha256s"] = splits
    row["reused_artifact_sha256_inventory"] = inventories[0]
    row["filtered_or_rematerialized_artifact_sha256_inventory"] = inventories[1]
    return row


def build_revision_6_scope_migration_under_revision_7(
    *,
    validated_30_day_raw_manifest_path: str,
    validated_30_day_raw_manifest_sha256: str,
    raw_episode_count: int,
    raw_episode_scan_count: int,
    all_episode_collision_census_receipt_sha256: str,
    feature_schema_sha256: str,
    target_schema_sha256: str,
    checklist_provenance_schema_sha256: str,
    zero_bypass_receipt_sha256: str,
    predecessor_schema_freeze_receipt_sha256: str,
    predecessor_corpus_and_refeaturization_receipt_sha256s: Sequence[str],
    total_actor_visible_decisions_scanned: int,
    included_actor_visible_decisions: int,
    excluded_nonqualifying_acting_seat_decisions: int,
    excluded_missing_or_malformed_setup_deck_decisions: int,
    included_row_manifest_sha256: str,
    excluded_row_reason_counts_sha256: str,
    list_variant_strata_manifest_sha256: str,
    source_day_group_disjoint_split_manifest_sha256s: Mapping[str, str],
    reused_artifact_sha256_inventory: Mapping[str, str],
    filtered_or_rematerialized_artifact_sha256_inventory: Mapping[str, str],
    sealed_at_utc: str,
) -> dict[str, Any]:
    """Build the unchanged rev6 receipt after its census evidence exists.

    Current rev7 is the authority allowing reuse; the resulting receipt keeps
    the exact historical rev6 goal stamp and closed 46-field inventory.
    """

    contract = load_revision_7_contract()
    scope = _mapping(
        contract["revision_6_corpus_scope_and_predecessor_acceptance"],
        field="embedded revision-6 scope",
    )
    _fields, fixed = _migration_spec(contract)
    eligibility = _mapping(scope["row_eligibility"], field="row eligibility")
    strata = _mapping(
        scope["list_variant_stratification"], field="list-variant stratification"
    )
    receipt: dict[str, Any] = {
        "schema": REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA,
        "goal_contract_path": REV6_CONTRACT_PATH,
        "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION,
        "predecessor_gateway_sha256": REV6_PREDECESSOR_GATEWAY_SHA256,
        "predecessor_contract_sha256": REV6_PREDECESSOR_CONTRACT_SHA256,
        "validated_30_day_raw_manifest_path": validated_30_day_raw_manifest_path,
        "validated_30_day_raw_manifest_sha256": validated_30_day_raw_manifest_sha256,
        "window_start_utc": EXACT_WINDOW_START_UTC,
        "window_end_utc": EXACT_WINDOW_END_UTC,
        "utc_partition_count": EXACT_UTC_PARTITION_COUNT,
        "raw_episode_count": raw_episode_count,
        "raw_episode_scan_count": raw_episode_scan_count,
        "all_raw_episodes_scanned": True,
        "no_episode_deck_seat_or_archetype_prefilter": True,
        "all_episode_collision_census_receipt_sha256": (
            all_episode_collision_census_receipt_sha256
        ),
        "feature_schema_sha256": feature_schema_sha256,
        "target_schema_sha256": target_schema_sha256,
        "checklist_provenance_schema_sha256": checklist_provenance_schema_sha256,
        "zero_bypass_receipt_sha256": zero_bypass_receipt_sha256,
        "predecessor_schema_freeze_receipt_sha256": (
            predecessor_schema_freeze_receipt_sha256
        ),
        "predecessor_corpus_and_refeaturization_receipt_sha256s": list(
            predecessor_corpus_and_refeaturization_receipt_sha256s
        ),
        "eligibility_card_id": 743,
        "eligibility_predicate": eligibility["predicate"],
        "acting_seat_only": True,
        "exact_pilot_match_required": False,
        "tech_inclusion_exception_exists": False,
        "total_actor_visible_decisions_scanned": total_actor_visible_decisions_scanned,
        "included_actor_visible_decisions": included_actor_visible_decisions,
        "excluded_nonqualifying_acting_seat_decisions": (
            excluded_nonqualifying_acting_seat_decisions
        ),
        "excluded_missing_or_malformed_setup_deck_decisions": (
            excluded_missing_or_malformed_setup_deck_decisions
        ),
        "both_qualifying_seats_handled_independently": True,
        "opponent_nonqualifying_decisions_included": False,
        "included_row_manifest_sha256": included_row_manifest_sha256,
        "excluded_row_reason_counts_sha256": excluded_row_reason_counts_sha256,
        "acting_deck_multiset_digest_algorithm": strata["digest_algorithm"],
        "all_included_rows_have_acting_deck_contains_743": True,
        "all_included_rows_have_acting_deck_multiset_sha256": True,
        "no_included_row_requires_exact_pilot_digest": True,
        "list_variant_strata_manifest_sha256": list_variant_strata_manifest_sha256,
        "source_day_group_disjoint_split_manifest_sha256s": dict(
            source_day_group_disjoint_split_manifest_sha256s
        ),
        "reused_artifact_sha256_inventory": dict(reused_artifact_sha256_inventory),
        "filtered_or_rematerialized_artifact_sha256_inventory": dict(
            filtered_or_rematerialized_artifact_sha256_inventory
        ),
        "recomputation_required_due_only_to_revision_change": False,
        "migration_validation_passed": True,
        "sealed_at_utc": sealed_at_utc,
    }
    for field, value in fixed.items():
        if receipt.get(field) != value:
            raise Rev7PredecessorCompatibilityError(f"migration builder fixed {field} drifted")
    return validate_revision_6_scope_migration_under_revision_7(receipt)


def write_revision_6_scope_migration_under_revision_7_create_only(
    path: Path | str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one canonical rev6 receipt with exclusive, durable semantics."""

    row = validate_revision_6_scope_migration_under_revision_7(receipt)
    target = Path(path)
    if not target.name or target.name in {".", ".."}:
        raise Rev7PredecessorCompatibilityError("migration receipt target name is invalid")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise Rev7PredecessorCompatibilityError(
            "migration receipt parent must be an existing non-symlink directory"
        )
    if target.exists() or target.is_symlink():
        raise Rev7PredecessorCompatibilityError("create-only migration receipt already exists")
    payload = _canonical_bytes(row) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o444)
    except OSError as exc:
        raise Rev7PredecessorCompatibilityError(
            "cannot create migration receipt exclusively"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - operating-system guard
                raise Rev7PredecessorCompatibilityError("migration receipt write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    info = target.lstat()
    if target.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size != len(payload):
        raise Rev7PredecessorCompatibilityError("published migration receipt identity is invalid")
    if target.read_bytes() != payload:
        raise Rev7PredecessorCompatibilityError("published migration receipt bytes differ")
    return {
        "path": str(target.resolve()),
        "sha256": _file_sha256(target),
        "size_bytes": int(info.st_size),
        "schema": REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA,
        "goal_revision": REV6_GOAL_REVISION,
    }


def validate_revision_5_predecessors_under_revision_7(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
    materialization_preflight: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the sealed rev5 raw/freeze/parity graph without reopening r5."""

    load_revision_7_contract()
    try:
        validate_raw_corpus_manifest(raw_manifest)
        frozen_sha, bypass_sha = validate_frozen_schema_gate(
            schema_manifest, zero_bypass_receipt
        )
    except Exception as exc:
        raise Rev7PredecessorCompatibilityError("revision-5 raw/freeze predecessor failed") from exc
    raw = dict(_mapping(raw_receipt, field="revision-5 raw corpus receipt"))
    if set(raw) != _RAW_RECEIPT_FIELDS:
        raise Rev7PredecessorCompatibilityError("revision-5 raw receipt field inventory drifted")
    if (
        raw.get("schema"),
        raw.get("status"),
        raw.get("owner_revision"),
        raw.get("goal_revision"),
        raw.get("root_handoff_revision"),
        raw.get("owner_goal_sha256"),
        raw.get("rule_derivative_gateway_sha256"),
        raw.get("rule_derivative_contract_sha256"),
        raw.get("mechanics_attachment_sha256"),
        raw.get("window_start_utc"),
        raw.get("window_end_utc"),
        raw.get("distinct_utc_day_count"),
        raw.get("recollection_authorized"),
    ) != (
        R298_RAW_CORPUS_RECEIPT_SCHEMA,
        "passed",
        R298_OWNER_REVISION,
        5,
        303,
        OWNER_GOAL_SHA256,
        RULE_DERIVATIVE_GATEWAY_SHA256,
        RULE_DERIVATIVE_CONTRACT_SHA256,
        MECHANICS_ATTACHMENT_SHA256,
        EXACT_WINDOW_START_UTC,
        EXACT_WINDOW_END_UTC,
        EXACT_UTC_PARTITION_COUNT,
        False,
    ):
        raise Rev7PredecessorCompatibilityError("revision-5 raw receipt authority drifted")
    try:
        validate_revision_5_predecessor_classification(
            raw.get("revision_5_predecessor_classification")
        )
    except Exception as exc:
        raise Rev7PredecessorCompatibilityError("revision-5 raw predecessor classification failed") from exc
    raw_manifest_sha = canonical_sha256(raw_manifest)
    raw_receipt_sha = canonical_sha256(raw)
    if raw.get("raw_expert_corpus_manifest_sha256") != raw_manifest_sha:
        raise Rev7PredecessorCompatibilityError("revision-5 raw receipt binds another manifest")
    if (
        raw.get("completed_raw_zip_member_count"),
        raw.get("completed_validated_episode_count"),
    ) != (
        raw_manifest.get("total_raw_zip_json_members"),
        raw_manifest.get("total_validated_episodes"),
    ):
        raise Rev7PredecessorCompatibilityError("revision-5 raw receipt completion counts drifted")
    for field, key in (
        ("source_manifest_provenance_sha256", "source_manifest_provenance"),
        ("source_receipt_day_coverage_sha256", "source_receipt_day_coverage"),
        ("episode_deduplication_sha256", "episode_deduplication"),
    ):
        if raw.get(field) != canonical_sha256(raw_manifest[key]):
            raise Rev7PredecessorCompatibilityError(f"revision-5 raw receipt {field} drifted")
    if _mapping(raw.get("source_disjointness"), field="raw source disjointness") != {
        "archive_date_source_sha256_unique": True,
        "episode_identity_unique": True,
        "episode_id_content_unique": True,
        "source_window_blending_permitted": False,
        "training_eligible": False,
    }:
        raise Rev7PredecessorCompatibilityError("revision-5 raw source disjointness drifted")
    _mapping(raw.get("resource_observation"), field="raw resource observation")
    _sha(raw.get("run_identity_sha256"), field="raw run identity")
    if _mapping(raw.get("runtime_authority"), field="raw runtime authority") != {
        "elmo_only": True,
        "create_only": True,
        "production_activation": False,
        "inzi_mutation": False,
        "archive_mutation": False,
    }:
        raise Rev7PredecessorCompatibilityError("revision-5 raw runtime authority drifted")
    freeze = dict(_mapping(schema_freeze_receipt, field="revision-5 schema-freeze receipt"))
    if set(freeze) != _SCHEMA_FREEZE_FIELDS:
        raise Rev7PredecessorCompatibilityError("revision-5 schema-freeze field inventory drifted")
    if (
        freeze.get("schema"),
        freeze.get("goal_contract_path"),
        freeze.get("goal_contract_sha256"),
        freeze.get("goal_revision"),
        freeze.get("feature_schema_sha256"),
        freeze.get("target_schema_sha256"),
        freeze.get("checklist_provenance_schema_sha256"),
        freeze.get("zero_bypass_receipt_sha256"),
    ) != (
        "poke_bot.alakazam_rule_derivative_schema_freeze_receipt/v1",
        REV6_CONTRACT_PATH,
        RULE_DERIVATIVE_CONTRACT_SHA256,
        5,
        schema_manifest.get("feature_schema_sha256"),
        schema_manifest.get("target_schema_sha256"),
        schema_manifest.get("checklist_provenance_schema_sha256"),
        bypass_sha,
    ):
        raise Rev7PredecessorCompatibilityError("revision-5 schema-freeze cross-links drifted")
    if (
        freeze.get("q3_bench_only"),
        freeze.get("q5_q6_trace_only_zero"),
        freeze.get("public_information_contract_passed"),
        freeze.get("layer_off_bit_identical_baseline_logits"),
        freeze.get("census_supported_branch_inventory"),
    ) != (True, True, True, True, []):
        raise Rev7PredecessorCompatibilityError("revision-5 schema-freeze safety boundary drifted")
    preflight = dict(
        _mapping(materialization_preflight, field="revision-5 materialization preflight")
    )
    if set(preflight) != _PREFLIGHT_FIELDS:
        raise Rev7PredecessorCompatibilityError("revision-5 preflight field inventory drifted")
    expected = {
        "raw_expert_corpus_manifest_sha256": raw_manifest_sha,
        "raw_expert_corpus_receipt_sha256": raw_receipt_sha,
        "frozen_schema_manifest_sha256": frozen_sha,
        "zero_bypass_receipt_sha256": bypass_sha,
        "schema_freeze_receipt_sha256": canonical_sha256(freeze),
    }
    for field, identity in expected.items():
        if preflight.get(field) != identity:
            raise Rev7PredecessorCompatibilityError(f"revision-5 preflight {field} drifted")
    if (
        preflight.get("schema"),
        preflight.get("status"),
        preflight.get("goal_contract_path"),
        preflight.get("goal_contract_sha256"),
        preflight.get("goal_revision"),
        preflight.get("root_handoff_revision"),
        preflight.get("utc_window"),
        preflight.get("utc_partition_count"),
        preflight.get("selected_action_target_mode"),
        preflight.get("opponent_belief_target_mode"),
        preflight.get("source_day_group_seed_disjoint_split_manifest_required"),
        preflight.get("physical_shard_max_bytes"),
    ) != (
        "poke_bot.alakazam_rule_derivative_r303_materialization_preflight/v1",
        "ready_for_exact_30_day_elmo_refeaturization_diagnostic_targets_only",
        REV6_CONTRACT_PATH,
        RULE_DERIVATIVE_CONTRACT_SHA256,
        5,
        303,
        [EXACT_WINDOW_START_UTC, EXACT_WINDOW_END_UTC],
        EXACT_UTC_PARTITION_COUNT,
        "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
        "unavailable_masked_no_loss",
        True,
        96 * 1024 * 1024,
    ):
        raise Rev7PredecessorCompatibilityError("revision-5 materialization boundary drifted")
    if _mapping(preflight.get("runtime_authority"), field="preflight runtime authority") != {
        "elmo_only": True,
        "create_only": True,
        "training_allowed": False,
        "inzi_authority": False,
        "production_authority": False,
    }:
        raise Rev7PredecessorCompatibilityError("revision-5 preflight authority drifted")
    return expected


def validate_revision_5_census_predecessors_under_revision_7(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
) -> dict[str, str]:
    """Validate only evidence that existed before the running r5 census.

    A materialization preflight is intentionally not an input: the already
    running census did not consume one, and creating one after the fact would
    misrepresent its launch boundary.  Materialization continues to use the
    stricter six-artifact validator above.
    """

    load_revision_7_contract()
    try:
        validate_raw_corpus_manifest(raw_manifest)
        frozen_sha, bypass_sha = validate_frozen_schema_gate(
            schema_manifest, zero_bypass_receipt
        )
    except Exception as exc:
        raise Rev7PredecessorCompatibilityError(
            "revision-5 census raw/freeze predecessor failed"
        ) from exc
    raw = dict(_mapping(raw_receipt, field="revision-5 raw corpus receipt"))
    # Reuse the closed raw-receipt checks without fabricating later receipts.
    if set(raw) != _RAW_RECEIPT_FIELDS or (
        raw.get("schema"),
        raw.get("status"),
        raw.get("owner_revision"),
        raw.get("goal_revision"),
        raw.get("root_handoff_revision"),
        raw.get("owner_goal_sha256"),
        raw.get("rule_derivative_gateway_sha256"),
        raw.get("rule_derivative_contract_sha256"),
        raw.get("mechanics_attachment_sha256"),
        raw.get("window_start_utc"),
        raw.get("window_end_utc"),
        raw.get("distinct_utc_day_count"),
        raw.get("recollection_authorized"),
    ) != (
        R298_RAW_CORPUS_RECEIPT_SCHEMA,
        "passed",
        R298_OWNER_REVISION,
        5,
        303,
        OWNER_GOAL_SHA256,
        RULE_DERIVATIVE_GATEWAY_SHA256,
        RULE_DERIVATIVE_CONTRACT_SHA256,
        MECHANICS_ATTACHMENT_SHA256,
        EXACT_WINDOW_START_UTC,
        EXACT_WINDOW_END_UTC,
        EXACT_UTC_PARTITION_COUNT,
        False,
    ):
        raise Rev7PredecessorCompatibilityError("revision-5 census raw receipt drifted")
    try:
        validate_revision_5_predecessor_classification(
            raw.get("revision_5_predecessor_classification")
        )
    except Exception as exc:
        raise Rev7PredecessorCompatibilityError(
            "revision-5 census raw predecessor classification failed"
        ) from exc
    raw_manifest_sha = canonical_sha256(raw_manifest)
    if raw.get("raw_expert_corpus_manifest_sha256") != raw_manifest_sha:
        raise Rev7PredecessorCompatibilityError("revision-5 census raw manifest binding drifted")
    if (
        raw.get("completed_raw_zip_member_count"),
        raw.get("completed_validated_episode_count"),
    ) != (
        raw_manifest.get("total_raw_zip_json_members"),
        raw_manifest.get("total_validated_episodes"),
    ):
        raise Rev7PredecessorCompatibilityError("revision-5 census completion counts drifted")
    for field, key in (
        ("source_manifest_provenance_sha256", "source_manifest_provenance"),
        ("source_receipt_day_coverage_sha256", "source_receipt_day_coverage"),
        ("episode_deduplication_sha256", "episode_deduplication"),
    ):
        if raw.get(field) != canonical_sha256(raw_manifest[key]):
            raise Rev7PredecessorCompatibilityError(
                f"revision-5 census raw receipt {field} drifted"
            )
    if _mapping(raw.get("source_disjointness"), field="raw source disjointness") != {
        "archive_date_source_sha256_unique": True,
        "episode_identity_unique": True,
        "episode_id_content_unique": True,
        "source_window_blending_permitted": False,
        "training_eligible": False,
    }:
        raise Rev7PredecessorCompatibilityError("revision-5 census disjointness drifted")
    _mapping(raw.get("resource_observation"), field="raw resource observation")
    _sha(raw.get("run_identity_sha256"), field="raw run identity")
    if _mapping(raw.get("runtime_authority"), field="raw runtime authority") != {
        "elmo_only": True,
        "create_only": True,
        "production_activation": False,
        "inzi_mutation": False,
        "archive_mutation": False,
    }:
        raise Rev7PredecessorCompatibilityError("revision-5 census authority drifted")
    return {
        "raw_expert_corpus_manifest_sha256": raw_manifest_sha,
        "raw_expert_corpus_receipt_sha256": canonical_sha256(raw),
        "frozen_schema_manifest_sha256": frozen_sha,
        "zero_bypass_receipt_sha256": bypass_sha,
    }


def validate_revision_5_census_completion_under_revision_7(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    census_validation_receipt: Mapping[str, Any],
    collision_census_receipt_path: Path | str,
) -> dict[str, Any]:
    """Cross-link the completed historic census without a later preflight.

    This is the immediate post-census bridge.  It distinguishes the strict
    r5 bridge's canonical parsed-payload identity from the physical receipt
    file identity that the later rev6 migration retains in its reused-artifact
    inventory.  It creates no file and grants no next-phase authority.
    """

    expected = validate_revision_5_census_predecessors_under_revision_7(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
    )
    census_file, census_file_sha = _read_stable_json(
        Path(collision_census_receipt_path), field="collision census receipt"
    )
    try:
        bridge = validate_revision_5_census_validation_receipt(
            census_validation_receipt
        )
    except Exception as exc:
        raise Rev7PredecessorCompatibilityError(
            "revision-5 census validation bridge failed"
        ) from exc
    census_canonical_sha = canonical_sha256(census_file)
    if (
        bridge.get("raw_expert_corpus_manifest_sha256"),
        bridge.get("raw_expert_corpus_receipt_sha256"),
        bridge.get("frozen_schema_manifest_sha256"),
        bridge.get("zero_bypass_receipt_sha256"),
        bridge.get("collision_census_receipt_sha256"),
    ) != (
        expected["raw_expert_corpus_manifest_sha256"],
        expected["raw_expert_corpus_receipt_sha256"],
        expected["frozen_schema_manifest_sha256"],
        expected["zero_bypass_receipt_sha256"],
        census_canonical_sha,
    ):
        raise Rev7PredecessorCompatibilityError(
            "revision-5 completed census cross-links drifted"
        )
    return {
        "schema": REV7_PREDECESSOR_VALIDATION_SCHEMA,
        "status": "validated_in_memory_completed_revision_5_census",
        "current_goal_revision": REV7_GOAL_REVISION,
        "current_gateway_sha256": REV7_GOAL_GATEWAY_SHA256,
        "current_contract_sha256": REV7_CONTRACT_SHA256,
        **expected,
        "revision_5_census_validation_receipt_sha256": canonical_sha256(bridge),
        "revision_5_collision_census_canonical_sha256": census_canonical_sha,
        "revision_5_collision_census_physical_file_sha256": census_file_sha,
        "materialization_preflight_claimed_or_required": False,
        "historical_receipts_retagged_or_rewritten": False,
        "training_runtime_service_transfer_or_activation_authority": False,
    }


def validate_revision_7_predecessor_bundle(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
    materialization_preflight: Mapping[str, Any],
    corpus_scope_migration_receipt: Mapping[str, Any],
    census_validation_receipt: Mapping[str, Any] | None = None,
    collision_census_receipt: Mapping[str, Any] | None = None,
    collision_census_receipt_path: Path | str | None = None,
) -> dict[str, Any]:
    """Cross-link all supplied historical evidence under current rev7."""

    expected = validate_revision_5_predecessors_under_revision_7(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
        schema_freeze_receipt=schema_freeze_receipt,
        materialization_preflight=materialization_preflight,
    )
    migration = validate_revision_6_scope_migration_under_revision_7(
        corpus_scope_migration_receipt
    )
    if migration.get("validated_30_day_raw_manifest_sha256") != expected[
        "raw_expert_corpus_manifest_sha256"
    ]:
        raise Rev7PredecessorCompatibilityError("revision-6 migration binds another raw manifest")
    for migration_field, schema_field in (
        ("feature_schema_sha256", "feature_schema_sha256"),
        ("target_schema_sha256", "target_schema_sha256"),
        ("checklist_provenance_schema_sha256", "checklist_provenance_schema_sha256"),
    ):
        if migration.get(migration_field) != schema_manifest.get(schema_field):
            raise Rev7PredecessorCompatibilityError(f"revision-6 migration {migration_field} drifted")
    if migration.get("zero_bypass_receipt_sha256") != expected["zero_bypass_receipt_sha256"]:
        raise Rev7PredecessorCompatibilityError("revision-6 migration zero-bypass binding drifted")
    if migration.get("predecessor_schema_freeze_receipt_sha256") != expected[
        "schema_freeze_receipt_sha256"
    ]:
        raise Rev7PredecessorCompatibilityError("revision-6 migration schema-freeze binding drifted")

    census_validation_sha: str | None = None
    census_canonical_sha: str | None = None
    census_physical_sha: str | None = None
    if census_validation_receipt is not None:
        if collision_census_receipt_path is None:
            raise Rev7PredecessorCompatibilityError(
                "census validation requires the immutable collision receipt path"
            )
        census_file, census_file_sha = _read_stable_json(
            Path(collision_census_receipt_path), field="collision census receipt"
        )
        census_canonical_sha = canonical_sha256(census_file)
        census_physical_sha = census_file_sha
        if migration.get("all_episode_collision_census_receipt_sha256") != census_canonical_sha:
            raise Rev7PredecessorCompatibilityError(
                "revision-6 migration binds another canonical collision receipt payload"
            )
        if migration["reused_artifact_sha256_inventory"].get(
            REV7_CENSUS_PHYSICAL_INVENTORY_KEY
        ) != census_physical_sha:
            raise Rev7PredecessorCompatibilityError(
                "revision-6 migration binds another physical collision receipt file"
            )
        if collision_census_receipt is not None and canonical_sha256(
            collision_census_receipt
        ) != canonical_sha256(census_file):
            raise Rev7PredecessorCompatibilityError(
                "supplied collision receipt differs from immutable file"
            )
        try:
            census_validation = validate_revision_5_census_validation_receipt(
                census_validation_receipt
            )
        except Exception as exc:
            raise Rev7PredecessorCompatibilityError("revision-5 census bridge failed") from exc
        if (
            census_validation.get("raw_expert_corpus_manifest_sha256"),
            census_validation.get("raw_expert_corpus_receipt_sha256"),
            census_validation.get("frozen_schema_manifest_sha256"),
            census_validation.get("zero_bypass_receipt_sha256"),
            census_validation.get("collision_census_receipt_sha256"),
        ) != (
            expected["raw_expert_corpus_manifest_sha256"],
            expected["raw_expert_corpus_receipt_sha256"],
            expected["frozen_schema_manifest_sha256"],
            expected["zero_bypass_receipt_sha256"],
            census_canonical_sha,
        ):
            raise Rev7PredecessorCompatibilityError("revision-5 census bridge cross-links drifted")
        census_validation_sha = canonical_sha256(census_validation)

    return {
        "schema": REV7_PREDECESSOR_VALIDATION_SCHEMA,
        "status": "validated_in_memory_historical_receipts_unchanged",
        "current_goal_revision": REV7_GOAL_REVISION,
        "current_gateway_sha256": REV7_GOAL_GATEWAY_SHA256,
        "current_contract_sha256": REV7_CONTRACT_SHA256,
        "revision_5_raw_manifest_sha256": expected["raw_expert_corpus_manifest_sha256"],
        "revision_5_raw_receipt_sha256": expected["raw_expert_corpus_receipt_sha256"],
        "revision_5_schema_manifest_sha256": expected["frozen_schema_manifest_sha256"],
        "revision_5_zero_bypass_receipt_sha256": expected["zero_bypass_receipt_sha256"],
        "revision_5_schema_freeze_receipt_sha256": expected["schema_freeze_receipt_sha256"],
        "revision_6_scope_migration_receipt_sha256": canonical_sha256(migration),
        "revision_5_census_validation_receipt_sha256": census_validation_sha,
        "revision_5_collision_census_canonical_sha256": census_canonical_sha,
        "revision_5_collision_census_physical_file_sha256": census_physical_sha,
        "historical_receipts_retagged_or_rewritten": False,
        "recomputation_required_due_only_to_revision_7": False,
        "training_runtime_service_transfer_or_activation_authority": False,
    }


def revision_7_parallel_execution_plan(*, workers: int = EXACT_WORKER_COUNT) -> dict[str, Any]:
    """Return the bounded 24-process launch plan, not a resource receipt."""

    if type(workers) is not int or workers != EXACT_WORKER_COUNT:
        raise Rev7PredecessorCompatibilityError("the rev7 whole-window launch requires exactly 24 workers")
    days = []
    start = datetime.fromisoformat(EXACT_WINDOW_START_UTC)
    for index in range(EXACT_UTC_PARTITION_COUNT):
        day = (start + timedelta(days=index)).date().isoformat()
        days.append(day)
    lanes = [[] for _ in range(workers)]
    for index, day in enumerate(days):
        lanes[index % workers].append(day)
    return {
        "schema": REV7_PARALLEL_PLAN_SCHEMA,
        "worker_count": workers,
        "utc_partition_count": EXACT_UTC_PARTITION_COUNT,
        "window_start_utc": EXACT_WINDOW_START_UTC,
        "window_end_utc": EXACT_WINDOW_END_UTC,
        "day_lanes": lanes,
        "one_memory_heavy_phase_at_a_time": True,
        "experiment_ram_ceiling_bytes": EXPERIMENT_RAM_LIMIT_BYTES,
        "aggregate_runtime_peak_evidence_required_for_completion_receipt": True,
        "plan_itself_is_not_runtime_resource_evidence": True,
        "training_runtime_service_transfer_or_activation_authority": False,
    }


__all__ = [
    "EXACT_WORKER_COUNT",
    "EXPERIMENT_RAM_LIMIT_BYTES",
    "REV7_CENSUS_PHYSICAL_INVENTORY_KEY",
    "REV7_CONTRACT_PATH",
    "REV7_CONTRACT_SHA256",
    "REV7_GOAL_GATEWAY_SHA256",
    "REV7_GOAL_PATH",
    "REV7_GOAL_REVISION",
    "REV7_PARALLEL_PLAN_SCHEMA",
    "REV7_PREDECESSOR_VALIDATION_SCHEMA",
    "REV7_ROOT_HANDOFF_REVISION",
    "Rev7PredecessorCompatibilityError",
    "build_revision_6_scope_migration_under_revision_7",
    "load_revision_7_contract",
    "revision_7_parallel_execution_plan",
    "validate_revision_5_census_completion_under_revision_7",
    "validate_revision_5_census_predecessors_under_revision_7",
    "validate_revision_5_predecessors_under_revision_7",
    "validate_revision_6_scope_migration_under_revision_7",
    "validate_revision_7_predecessor_bundle",
    "write_revision_6_scope_migration_under_revision_7_create_only",
]
