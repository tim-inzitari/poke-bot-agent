"""Create-only revision-6 corpus-scope migration receipts.

This module is intentionally offline and receipt-only.  It does not scan or
filter a corpus, train a model, stage bytes, change a selector, call a service
manager, or grant activation authority.  A caller supplies the already-sealed
evidence identities and measured counts; this module builds the canonical
46-field receipt, validates it against the exact revision-6 goal contract, and
can publish the resulting JSON once with create-only filesystem semantics.

The receipt preserves the crucial scope split:

* the raw 2026-07-13..2026-08-11 audit/census scanned every episode; and
* materialized rows contain only a decision's acting seat when that same
  seat's setup deck literally contains card ID 743.

Revision-5 receipts remain immutable predecessor evidence.  They are named by
SHA-256 here; this module never rewrites or reseals them in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


REV6_GOAL_REVISION: Final = 6
REV6_ROOT_HANDOFF_REVISION: Final = 303
REV6_CONTRACT_PATH: Final = "goals/alakazam-elmo-rule-derivative/contract.json"
REV6_GOAL_PATH: Final = "goals/alakazam-elmo-rule-derivative/GOAL.md"
REV6_CONTRACT_SHA256: Final = (
    "sha256:49fc3d76d8bb36f2fcc63eb26d40ebc73308d43361236a8575d7fec0ac2c2960"
)
REV6_GOAL_GATEWAY_SHA256: Final = (
    "sha256:a869ade474d94ce8f77aa388c9623ed74b3482fd792d8a5dc3545202ca27315f"
)
REV6_PREDECESSOR_CONTRACT_SHA256: Final = (
    "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891"
)
REV6_PREDECESSOR_GATEWAY_SHA256: Final = (
    "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6"
)
REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_corpus_scope_migration_receipt/v1"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_FILE: Final = PROJECT_ROOT / REV6_CONTRACT_PATH
GOAL_FILE: Final = PROJECT_ROOT / REV6_GOAL_PATH
_SHA256_PREFIX: Final = "sha256:"
_SPLIT_NAMES: Final = ("train", "validation", "evaluation")


class Rev6CorpusScopeError(RuntimeError):
    """The proposed receipt is incomplete, stale, or semantically invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Rev6CorpusScopeError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return _SHA256_PREFIX + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Rev6CorpusScopeError(f"cannot read immutable file: {path}") from exc
    return _SHA256_PREFIX + digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Rev6CorpusScopeError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Rev6CorpusScopeError(f"{field} must be a list")
    return list(value)


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        raise Rev6CorpusScopeError(f"{field} must be a sha256 identity")
    raw = value.removeprefix(_SHA256_PREFIX)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise Rev6CorpusScopeError(f"{field} must use lowercase SHA-256 hex")
    return value


def _sha_sequence(value: Any, *, field: str, nonempty: bool = True) -> list[str]:
    rows = _rows(value, field=field)
    if nonempty and not rows:
        raise Rev6CorpusScopeError(f"{field} must be nonempty")
    result = [_sha(item, field=f"{field}[{index}]") for index, item in enumerate(rows)]
    if len(result) != len(set(result)):
        raise Rev6CorpusScopeError(f"{field} contains duplicate identities")
    return result


def _nonnegative_int(value: Any, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        kind = "positive" if positive else "nonnegative"
        raise Rev6CorpusScopeError(f"{field} must be an exact {kind} integer")
    return value


def _utc_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Rev6CorpusScopeError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Rev6CorpusScopeError(f"{field} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Rev6CorpusScopeError(f"{field} must use UTC")
    return value


def _sha_inventory(
    value: Any, *, field: str, nonempty: bool
) -> dict[str, str]:
    raw = _mapping(value, field=field)
    if nonempty and not raw:
        raise Rev6CorpusScopeError(f"{field} must be nonempty")
    result: dict[str, str] = {}
    for key, identity in raw.items():
        if not isinstance(key, str) or not key or len(key) > 256:
            raise Rev6CorpusScopeError(f"{field} has an invalid logical artifact name")
        result[key] = _sha(identity, field=f"{field}.{key}")
    return result


def _split_sha_inventory(value: Any) -> dict[str, str]:
    raw = _mapping(value, field="source/day/group split manifest identities")
    if set(raw) != set(_SPLIT_NAMES):
        raise Rev6CorpusScopeError(
            "source/day/group split manifest identities must have exactly "
            "train, validation, and evaluation"
        )
    return {
        split: _sha(raw[split], field=f"source/day/group split manifest {split}")
        for split in _SPLIT_NAMES
    }


def _read_regular_json(path: Path, *, field: str) -> tuple[Mapping[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise Rev6CorpusScopeError(f"{field} must be a regular non-symlink file")
    before = path.stat()
    digest = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Rev6CorpusScopeError(f"cannot parse {field}") from exc
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or sha256_file(path) != digest:
        raise Rev6CorpusScopeError(f"{field} changed while being inspected")
    return _mapping(payload, field=field), digest


def load_revision_6_contract() -> Mapping[str, Any]:
    """Load the exact rev6 gateway/contract pair; substitution fails closed."""

    contract, digest = _read_regular_json(CONTRACT_FILE, field="revision-6 contract")
    if digest != REV6_CONTRACT_SHA256:
        raise Rev6CorpusScopeError("revision-6 contract identity is stale or foreign")
    if (
        contract.get("goal_id"),
        contract.get("goal_revision"),
        contract.get("root_handoff_revision"),
    ) != ("alakazam-elmo-rule-derivative", REV6_GOAL_REVISION, REV6_ROOT_HANDOFF_REVISION):
        raise Rev6CorpusScopeError("revision-6 contract goal identity drifted")
    if GOAL_FILE.is_symlink() or not GOAL_FILE.is_file():
        raise Rev6CorpusScopeError("revision-6 goal gateway is unavailable")
    if sha256_file(GOAL_FILE) != REV6_GOAL_GATEWAY_SHA256:
        raise Rev6CorpusScopeError("revision-6 goal gateway identity drifted")
    authority = _mapping(contract.get("authority"), field="revision-6 authority")
    if authority.get("immediate_runtime_service_preemption_training_submission_or_self_play_authority") is not False:
        raise Rev6CorpusScopeError("revision-6 contract unexpectedly grants immediate authority")
    return contract


def corpus_scope_migration_receipt_spec() -> dict[str, Any]:
    """Return the canonical schema, exact field order, fixed values, and types."""

    contract = load_revision_6_contract()
    scope = _mapping(
        contract.get("revision_6_corpus_scope_and_predecessor_acceptance"),
        field="revision-6 corpus scope",
    )
    migration = _mapping(scope.get("migration_receipt"), field="migration receipt contract")
    schema = migration.get("schema")
    fields = _rows(migration.get("required_fields"), field="migration receipt fields")
    fixed = dict(_mapping(migration.get("fixed_values"), field="migration fixed values"))
    if schema != REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA:
        raise Rev6CorpusScopeError("migration receipt schema drifted")
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise Rev6CorpusScopeError("migration receipt fields are malformed")
    if len(fields) != len(set(fields)):
        raise Rev6CorpusScopeError("migration receipt repeats fields")
    return {
        "schema": schema,
        "required_fields": fields,
        "fixed_values": fixed,
        "structured_field_types": {
            "predecessor_corpus_and_refeaturization_receipt_sha256s": "nonempty_unique_sha256_array",
            "source_day_group_disjoint_split_manifest_sha256s": "exact_object_train_validation_evaluation_to_sha256",
            "reused_artifact_sha256_inventory": "nonempty_object_logical_name_to_sha256",
            "filtered_or_rematerialized_artifact_sha256_inventory": "object_logical_name_to_sha256_may_be_empty",
        },
    }


def validate_corpus_scope_migration_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate one complete 46-field rev6 migration receipt."""

    spec = corpus_scope_migration_receipt_spec()
    row = dict(_mapping(receipt, field="corpus-scope migration receipt"))
    expected = set(spec["required_fields"])
    actual = set(row)
    if actual != expected:
        raise Rev6CorpusScopeError(
            "corpus-scope migration receipt inventory mismatch; "
            f"missing={sorted(expected - actual)!r} extra={sorted(actual - expected)!r}"
        )
    if row.get("schema") != spec["schema"]:
        raise Rev6CorpusScopeError("corpus-scope migration receipt schema mismatch")
    if row.get("goal_contract_path") != REV6_CONTRACT_PATH:
        raise Rev6CorpusScopeError("corpus-scope migration receipt contract path mismatch")
    if row.get("goal_contract_sha256") != REV6_CONTRACT_SHA256:
        raise Rev6CorpusScopeError("corpus-scope migration receipt contract identity mismatch")
    if row.get("predecessor_gateway_sha256") != REV6_PREDECESSOR_GATEWAY_SHA256:
        raise Rev6CorpusScopeError("corpus-scope migration receipt predecessor gateway mismatch")
    if row.get("predecessor_contract_sha256") != REV6_PREDECESSOR_CONTRACT_SHA256:
        raise Rev6CorpusScopeError("corpus-scope migration receipt predecessor contract mismatch")
    for field, expected_value in spec["fixed_values"].items():
        if row.get(field) != expected_value:
            raise Rev6CorpusScopeError(f"{field} must be {expected_value!r}")

    path = row.get("validated_30_day_raw_manifest_path")
    if not isinstance(path, str) or not path or "\x00" in path:
        raise Rev6CorpusScopeError("validated raw manifest path must be nonempty text")
    for field in (
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
    ):
        _sha(row.get(field), field=field)
    _sha_sequence(
        row.get("predecessor_corpus_and_refeaturization_receipt_sha256s"),
        field="predecessor corpus and re-featurization receipt identities",
    )
    splits = _split_sha_inventory(row.get("source_day_group_disjoint_split_manifest_sha256s"))
    reused = _sha_inventory(
        row.get("reused_artifact_sha256_inventory"),
        field="reused artifact identity inventory",
        nonempty=True,
    )
    rebuilt = _sha_inventory(
        row.get("filtered_or_rematerialized_artifact_sha256_inventory"),
        field="filtered or rematerialized artifact identity inventory",
        nonempty=False,
    )
    if set(reused) & set(rebuilt):
        raise Rev6CorpusScopeError("an artifact cannot be both reused and rematerialized")

    raw_episode_count = _nonnegative_int(
        row.get("raw_episode_count"), field="raw episode count", positive=True
    )
    raw_episode_scan_count = _nonnegative_int(
        row.get("raw_episode_scan_count"), field="raw episode scan count", positive=True
    )
    if raw_episode_count != raw_episode_scan_count:
        raise Rev6CorpusScopeError("all raw episodes were not scanned exactly once")
    total = _nonnegative_int(
        row.get("total_actor_visible_decisions_scanned"),
        field="total actor-visible decisions scanned",
    )
    included = _nonnegative_int(
        row.get("included_actor_visible_decisions"),
        field="included actor-visible decisions",
    )
    excluded_nonqualifying = _nonnegative_int(
        row.get("excluded_nonqualifying_acting_seat_decisions"),
        field="excluded nonqualifying acting-seat decisions",
    )
    excluded_malformed = _nonnegative_int(
        row.get("excluded_missing_or_malformed_setup_deck_decisions"),
        field="excluded missing/malformed setup-deck decisions",
    )
    if total != included + excluded_nonqualifying + excluded_malformed:
        raise Rev6CorpusScopeError("acting-decision inclusion/exclusion counts do not partition the scan")

    scope = _mapping(
        load_revision_6_contract()["revision_6_corpus_scope_and_predecessor_acceptance"],
        field="revision-6 corpus scope",
    )
    expected_predicate = _mapping(scope["row_eligibility"], field="row eligibility")["predicate"]
    expected_algorithm = _mapping(
        scope["list_variant_stratification"], field="list-variant stratification"
    )["digest_algorithm"]
    if row.get("eligibility_predicate") != expected_predicate:
        raise Rev6CorpusScopeError("eligibility predicate differs from the canonical literal test")
    if row.get("acting_deck_multiset_digest_algorithm") != expected_algorithm:
        raise Rev6CorpusScopeError("acting-deck multiset digest algorithm drifted")
    _utc_timestamp(row.get("sealed_at_utc"), field="sealed_at_utc")

    # Normalize the structured fields to plain deterministic dictionaries/lists.
    row["predecessor_corpus_and_refeaturization_receipt_sha256s"] = _sha_sequence(
        row["predecessor_corpus_and_refeaturization_receipt_sha256s"],
        field="predecessor corpus and re-featurization receipt identities",
    )
    row["source_day_group_disjoint_split_manifest_sha256s"] = splits
    row["reused_artifact_sha256_inventory"] = reused
    row["filtered_or_rematerialized_artifact_sha256_inventory"] = rebuilt
    return row


def build_corpus_scope_migration_receipt(
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
    """Build and validate the canonical receipt from already measured evidence."""

    contract = load_revision_6_contract()
    scope = _mapping(
        contract["revision_6_corpus_scope_and_predecessor_acceptance"],
        field="revision-6 corpus scope",
    )
    migration = _mapping(scope["migration_receipt"], field="migration receipt contract")
    fixed = dict(_mapping(migration["fixed_values"], field="migration fixed values"))
    row_eligibility = _mapping(scope["row_eligibility"], field="row eligibility")
    strata = _mapping(scope["list_variant_stratification"], field="list-variant stratification")
    receipt: dict[str, Any] = {
        "schema": REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA,
        "goal_contract_path": REV6_CONTRACT_PATH,
        "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION,
        "predecessor_gateway_sha256": REV6_PREDECESSOR_GATEWAY_SHA256,
        "predecessor_contract_sha256": REV6_PREDECESSOR_CONTRACT_SHA256,
        "validated_30_day_raw_manifest_path": validated_30_day_raw_manifest_path,
        "validated_30_day_raw_manifest_sha256": validated_30_day_raw_manifest_sha256,
        "window_start_utc": "2026-07-13",
        "window_end_utc": "2026-08-11",
        "utc_partition_count": 30,
        "raw_episode_count": raw_episode_count,
        "raw_episode_scan_count": raw_episode_scan_count,
        "all_raw_episodes_scanned": True,
        "no_episode_deck_seat_or_archetype_prefilter": True,
        "all_episode_collision_census_receipt_sha256": all_episode_collision_census_receipt_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "target_schema_sha256": target_schema_sha256,
        "checklist_provenance_schema_sha256": checklist_provenance_schema_sha256,
        "zero_bypass_receipt_sha256": zero_bypass_receipt_sha256,
        "predecessor_schema_freeze_receipt_sha256": predecessor_schema_freeze_receipt_sha256,
        "predecessor_corpus_and_refeaturization_receipt_sha256s": list(
            predecessor_corpus_and_refeaturization_receipt_sha256s
        ),
        "eligibility_card_id": 743,
        "eligibility_predicate": row_eligibility["predicate"],
        "acting_seat_only": True,
        "exact_pilot_match_required": False,
        "tech_inclusion_exception_exists": False,
        "total_actor_visible_decisions_scanned": total_actor_visible_decisions_scanned,
        "included_actor_visible_decisions": included_actor_visible_decisions,
        "excluded_nonqualifying_acting_seat_decisions": excluded_nonqualifying_acting_seat_decisions,
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
            raise Rev6CorpusScopeError(f"builder fixed value drifted for {field}")
    return validate_corpus_scope_migration_receipt(receipt)


def write_corpus_scope_migration_receipt_create_only(
    path: Path | str, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and publish one immutable JSON receipt without overwriting."""

    row = validate_corpus_scope_migration_receipt(receipt)
    target = Path(path)
    parent = target.parent
    if not target.name or target.name in {".", ".."}:
        raise Rev6CorpusScopeError("receipt target name is invalid")
    if parent.is_symlink() or not parent.is_dir():
        raise Rev6CorpusScopeError("receipt parent must be an existing non-symlink directory")
    if target.exists() or target.is_symlink():
        raise Rev6CorpusScopeError("create-only receipt target already exists")
    payload = _canonical_json_bytes(row) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o444)
    except OSError as exc:
        raise Rev6CorpusScopeError("cannot create receipt target exclusively") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive operating-system guard
                raise Rev6CorpusScopeError("receipt write made no progress")
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
        raise Rev6CorpusScopeError("published receipt identity is invalid")
    if target.read_bytes() != payload:
        raise Rev6CorpusScopeError("published receipt bytes differ from canonical payload")
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "size_bytes": int(info.st_size),
    }


__all__ = [
    "REV6_CONTRACT_PATH",
    "REV6_CONTRACT_SHA256",
    "REV6_CORPUS_SCOPE_MIGRATION_RECEIPT_SCHEMA",
    "REV6_GOAL_GATEWAY_SHA256",
    "REV6_GOAL_REVISION",
    "REV6_PREDECESSOR_CONTRACT_SHA256",
    "REV6_PREDECESSOR_GATEWAY_SHA256",
    "REV6_ROOT_HANDOFF_REVISION",
    "Rev6CorpusScopeError",
    "build_corpus_scope_migration_receipt",
    "canonical_sha256",
    "corpus_scope_migration_receipt_spec",
    "load_revision_6_contract",
    "sha256_file",
    "validate_corpus_scope_migration_receipt",
    "write_corpus_scope_migration_receipt_create_only",
]
