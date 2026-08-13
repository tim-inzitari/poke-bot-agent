"""Strict, read-only revision-5 aggregate and consumer-migration validation.

This is an additive *receipt consumer*.  It cannot start, stop, pause, or
reload a service; it cannot load a checkpoint, stage a shard, create a Kaggle
entry, or start collection.  Its role is deliberately narrower:

* reject a revision-4 receipt that is merely relabelled with revision-5 hashes;
* re-open the exact revision-5 gateway and contract before interpreting a
  handoff, candidate, package, queue, or fleet receipt; and
* make a future handoff's pre-activation prerequisites inspectable without
  granting any activation authority.

The contract's typed receipts are the authority for their closed field
inventories.  This module adds the missing aggregate boundary: a portable
parser may be used for reporting, while the trusted verifier re-hashes every
referenced local receipt, migration test artifact, and source file.  Neither
mode is a service-control API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from . import alakazam_rule_derivative_handoff_r303 as handoff
from .alakazam_collision_census_r298 import (
    R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA,
    R298_REV5_PREDECESSOR_CLASSIFICATION_SCHEMA,
    validate_revision_5_census_validation_receipt,
    validate_revision_5_predecessor_classification,
)


R303_CONSUMER_MIGRATION_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_consumer_migration_receipt/v1"
)
R303_PREACTIVATION_AGGREGATE_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_pre_activation_aggregate/v1"
)
R303_CONSUMER_TEST_EVIDENCE_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_consumer_migration_test_evidence/v1"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
R303_GOAL_PATH: Final = handoff.R303_GOAL_PATH
R303_CONTRACT_PATH: Final = handoff.R303_CONTRACT_PATH
R303_GOAL_GATEWAY_SHA256: Final = handoff.R303_GOAL_GATEWAY_SHA256
R303_CONTRACT_SHA256: Final = handoff.R303_CONTRACT_SHA256
R303_GOAL_REVISION: Final = handoff.R303_GOAL_REVISION
R303_ROOT_OWNER_REVISION: Final = handoff.R303_ROOT_OWNER_REVISION
R303_R241_TYPED_SOURCE_PATH: Final = "state/alakazam-new-list-direct-policy-r241.json"
R303_R241_TYPED_SOURCE_SHA256: Final = (
    "sha256:8d83f5e9eafc8e554f33dcbfbda7e1b337b8f65dc2e49109be4acdd829850c1e"
)

# Every previously r298/r4 consumer which could influence the experiment has
# to be explicitly re-bound.  Merely changing an aggregate receipt's SHA does
# not make its source/configuration revision-5 aware.
MIGRATION_CONSUMER_SOURCES: Final[Mapping[str, str]] = {
    "public_rule_adapter": "poke_bot/alakazam_public_rule_adapter_r298.py",
    "simulator_rule_targets": "poke_bot/alakazam_simulator_rule_targets_r298.py",
    "checklist_provenance": "poke_bot/alakazam_checklist_provenance_r298.py",
    "collision_census": "poke_bot/alakazam_collision_census_r298.py",
    "training_materialization": "poke_bot/alakazam_rule_derivative_training_r298.py",
    "transfer_staging": "poke_bot/alakazam_rule_derivative_transfer_r298.py",
    "engine_verification": "scripts/run_alakazam_simulator_rules_r298_engine_verification.py",
    "bo250_consumer": "poke_bot/alakazam_turn_checklist_bo250_r289.py",
    "handoff_planner": "poke_bot/alakazam_rule_derivative_handoff_r303.py",
}

MIGRATION_TEST_ASSERTIONS: Final = (
    "unit_tests_rerun",
    "engine_tests_rerun",
    "information_set_tests_rerun",
    "layer_off_tests_rerun",
    "card2vec_tests_rerun",
    "contract_binding_tests_rerun",
)


class R303AggregateValidationError(RuntimeError):
    """A revision-5 aggregate lacks immutable or authority-safe proof."""


@dataclass(frozen=True)
class ImmutableFileIdentity:
    """Identity captured by a stable, non-symlink file read."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R303AggregateValidationError(f"{field} must be an object")
    return value


def _rows(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise R303AggregateValidationError(f"{field} must be a list")
    return list(value)


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise R303AggregateValidationError(f"{field} must be a lowercase SHA-256 identity")
    raw = value[7:]
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise R303AggregateValidationError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _positive_int(value: object, *, field: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise R303AggregateValidationError(
            f"{field} must be an exact {'nonnegative' if allow_zero else 'positive'} integer"
        )
    return value


def _require_bool(value: object, *, field: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise R303AggregateValidationError(f"{field} must be a boolean")
    if expected is not None and value is not expected:
        raise R303AggregateValidationError(f"{field} must be {expected}")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise R303AggregateValidationError(f"{field} must be a non-empty string")
    return value


def _assert_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise R303AggregateValidationError(
            f"{field} field inventory mismatch; missing={sorted(expected - set(value))!r} "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _relative_source_path(value: object, *, field: str) -> str:
    path = _require_string(value, field=field)
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != path:
        raise R303AggregateValidationError(f"{field} must be an exact project-relative path")
    return path


def _identity_from_mapping(value: object, *, field: str) -> ImmutableFileIdentity:
    row = _mapping(value, field=field)
    _assert_exact_keys(row, {"path", "sha256", "size_bytes"}, field=field)
    return ImmutableFileIdentity(
        path=_require_string(row.get("path"), field=f"{field}.path"),
        sha256=_sha(row.get("sha256"), field=f"{field}.sha256"),
        size_bytes=_positive_int(row.get("size_bytes"), field=f"{field}.size_bytes"),
    )


def _read_stable_file(
    path: Path | str,
    *,
    field: str,
    require_readonly: bool,
) -> ImmutableFileIdentity:
    """Hash one regular file twice around inspection, rejecting symlinks."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise R303AggregateValidationError(f"{field} must be a regular non-symlink file")
    try:
        before = target.stat()
    except OSError as exc:  # pragma: no cover - guarded by is_file.
        raise R303AggregateValidationError(f"{field} cannot be statted") from exc
    if require_readonly and before.st_mode & 0o222:
        raise R303AggregateValidationError(f"{field} must be immutable/read-only")
    digest = sha256_file(target)
    try:
        after = target.stat()
    except OSError as exc:  # pragma: no cover - race guard.
        raise R303AggregateValidationError(f"{field} disappeared while being read") from exc
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
    ) or sha256_file(target) != digest:
        raise R303AggregateValidationError(f"{field} changed while being inspected")
    return ImmutableFileIdentity(
        path=str(target.resolve()), sha256=digest, size_bytes=after.st_size
    )


def _ensure_within(path: Path, root: Path, *, field: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise R303AggregateValidationError(f"{field} trusted root must be a physical directory")
    resolved_root = root.resolve()
    if path.is_symlink():
        raise R303AggregateValidationError(f"{field} may not be a symlink")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise R303AggregateValidationError(f"{field} lies outside the trusted root") from exc
    cursor = resolved_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise R303AggregateValidationError(f"{field} contains a symlinked path component")
    return resolved


def _rooted_path(value: Path | str, root: Path, *, field: str) -> Path:
    """Resolve an absolute path or a root-relative evidence path safely."""

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return _ensure_within(candidate, root, field=field)


def _read_json_file(
    path: Path | str,
    *,
    field: str,
    trusted_root: Path | None,
    require_readonly: bool,
) -> tuple[Mapping[str, Any], ImmutableFileIdentity]:
    target = Path(path)
    if trusted_root is not None:
        target = _rooted_path(target, trusted_root, field=field)
    identity = _read_stable_file(target, field=field, require_readonly=require_readonly)
    try:
        raw = Path(identity.path).read_bytes()
        if (
            len(raw) != identity.size_bytes
            or "sha256:" + hashlib.sha256(raw).hexdigest() != identity.sha256
        ):
            raise R303AggregateValidationError(f"{field} bytes changed before JSON parse")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R303AggregateValidationError(f"{field} must contain a JSON object") from exc
    final_identity = _read_stable_file(
        identity.path, field=field, require_readonly=require_readonly
    )
    if final_identity != identity:
        raise R303AggregateValidationError(f"{field} changed while its JSON was inspected")
    return _mapping(payload, field=field), identity


def _validate_contract_binding(row: Mapping[str, Any], *, label: str) -> None:
    if (
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("goal_revision"),
    ) != (R303_CONTRACT_PATH, R303_CONTRACT_SHA256, R303_GOAL_REVISION):
        raise R303AggregateValidationError(f"{label} binds a stale or foreign revision-5 contract")


def _assert_contract_current() -> Mapping[str, Any]:
    """Delegate exact canonical re-hashing to the r303 handoff owner."""

    try:
        contract = handoff.load_r303_contract()
    except handoff.R303HandoffError as exc:
        raise R303AggregateValidationError("revision-5 canonical contract is not current") from exc
    r241 = PROJECT_ROOT / R303_R241_TYPED_SOURCE_PATH
    observed = _read_stable_file(r241, field="canonical r241 typed source", require_readonly=False)
    if observed.sha256 != R303_R241_TYPED_SOURCE_SHA256:
        raise R303AggregateValidationError("canonical r241 typed source identity drifted")
    return contract


def _migration_consumer(row: Mapping[str, Any], *, consumer_id: str) -> dict[str, Any]:
    expected = {
        "consumer_id",
        "source_path",
        "source_sha256",
        "source_size_bytes",
        "revision_5_authority_bound",
        "revision_4_receipt_is_historical_only",
        "zero_inert_and_layer_off_fail_closed",
        "runtime_wired",
        "test_evidence",
    }
    _assert_exact_keys(row, expected, field=f"migration consumer {consumer_id}")
    if row.get("consumer_id") != consumer_id:
        raise R303AggregateValidationError("migration consumer inventory has a mismatched identifier")
    source_path = _relative_source_path(
        row.get("source_path"), field=f"migration {consumer_id} source_path"
    )
    if source_path != MIGRATION_CONSUMER_SOURCES[consumer_id]:
        raise R303AggregateValidationError(f"migration {consumer_id} binds an unexpected source path")
    normalized = {
        "consumer_id": consumer_id,
        "source_path": source_path,
        "source_sha256": _sha(row.get("source_sha256"), field=f"migration {consumer_id} source SHA"),
        "source_size_bytes": _positive_int(
            row.get("source_size_bytes"), field=f"migration {consumer_id} source size"
        ),
        "revision_5_authority_bound": _require_bool(
            row.get("revision_5_authority_bound"),
            field=f"migration {consumer_id} revision-5 binding",
            expected=True,
        ),
        "revision_4_receipt_is_historical_only": _require_bool(
            row.get("revision_4_receipt_is_historical_only"),
            field=f"migration {consumer_id} revision-4 classification",
            expected=True,
        ),
        "zero_inert_and_layer_off_fail_closed": _require_bool(
            row.get("zero_inert_and_layer_off_fail_closed"),
            field=f"migration {consumer_id} zero/layer-off guard",
            expected=True,
        ),
        # A rebinding receipt proves no policy wiring by itself.  A later,
        # separately typed action-time receipt is required before any route
        # can claim candidate action authority.
        "runtime_wired": _require_bool(
            row.get("runtime_wired"), field=f"migration {consumer_id} runtime_wired", expected=False
        ),
        "test_evidence": _identity_from_mapping(
            row.get("test_evidence"), field=f"migration {consumer_id} test evidence"
        ).to_dict(),
    }
    return normalized


def validate_revision_5_consumer_migration_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a non-activating explicit rebind of every r298 consumer.

    This is intentionally a portable structural parser.  Use
    :func:`inspect_revision_5_consumer_migration_receipt` for a trusted
    evidence-root check that re-hashes the named files.
    """

    _assert_contract_current()
    row = _mapping(payload, field="revision-5 consumer migration receipt")
    expected = {
        "schema",
        "status",
        "goal_gateway_path",
        "goal_gateway_sha256",
        "goal_contract_path",
        "goal_contract_sha256",
        "r241_typed_source_path",
        "r241_typed_source_sha256",
        "goal_revision",
        "root_handoff_revision",
        "predecessor_evidence",
        "predecessor_receipt_files",
        "consumers",
        "test_assertions",
        "runtime_or_service_action_performed",
        "candidate_action_time_wrapper_sealed",
        "training_handoff_authorized",
        "kaggle_queue_or_fleet_authorized",
        "recorded_at_utc",
    }
    _assert_exact_keys(row, expected, field="revision-5 consumer migration receipt")
    if (
        row.get("schema"),
        row.get("status"),
        row.get("goal_gateway_path"),
        row.get("goal_gateway_sha256"),
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("r241_typed_source_path"),
        row.get("r241_typed_source_sha256"),
        row.get("goal_revision"),
        row.get("root_handoff_revision"),
    ) != (
        R303_CONSUMER_MIGRATION_SCHEMA,
        "passed_revision_5_rebind_nonactivating",
        R303_GOAL_PATH,
        R303_GOAL_GATEWAY_SHA256,
        R303_CONTRACT_PATH,
        R303_CONTRACT_SHA256,
        R303_R241_TYPED_SOURCE_PATH,
        R303_R241_TYPED_SOURCE_SHA256,
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
    ):
        raise R303AggregateValidationError("consumer migration receipt is not exact revision-5 evidence")
    try:
        predecessor = validate_revision_5_predecessor_classification(
            _mapping(row.get("predecessor_evidence"), field="predecessor evidence")
        )
    except Exception as exc:
        raise R303AggregateValidationError("revision-4 predecessor classification is invalid") from exc
    predecessor_files = _rows(
        row.get("predecessor_receipt_files"), field="predecessor receipt files"
    )
    classified = {
        (item["receipt_sha256"], item["schema"])
        for item in predecessor["consumed_predecessor_receipts"]
    }
    normalized_predecessor_files: list[dict[str, Any]] = []
    seen_predecessors: set[tuple[str, str]] = set()
    for index, item in enumerate(predecessor_files):
        evidence = _mapping(item, field=f"predecessor receipt file {index}")
        _assert_exact_keys(
            evidence, {"path", "sha256", "size_bytes", "schema"},
            field=f"predecessor receipt file {index}",
        )
        key = (_sha(evidence.get("sha256"), field=f"predecessor {index} SHA"), _require_string(
            evidence.get("schema"), field=f"predecessor {index} schema"
        ))
        if key in seen_predecessors:
            raise R303AggregateValidationError("predecessor receipt file inventory repeats an identity")
        seen_predecessors.add(key)
        _relative_source_path(evidence.get("path"), field=f"predecessor {index} path")
        normalized_predecessor_files.append(
            {
                "path": evidence["path"],
                "sha256": key[0],
                "size_bytes": _positive_int(evidence.get("size_bytes"), field=f"predecessor {index} size"),
                "schema": key[1],
            }
        )
    if seen_predecessors != classified:
        raise R303AggregateValidationError(
            "every classified revision-4 predecessor receipt must have one physical evidence file"
        )
    consumers = _rows(row.get("consumers"), field="migration consumers")
    if len(consumers) != len(MIGRATION_CONSUMER_SOURCES):
        raise R303AggregateValidationError("consumer migration inventory is incomplete")
    normalized_consumers: list[dict[str, Any]] = []
    seen_consumers: set[str] = set()
    for item in consumers:
        candidate = _mapping(item, field="migration consumer")
        consumer_id = candidate.get("consumer_id")
        if not isinstance(consumer_id, str) or consumer_id not in MIGRATION_CONSUMER_SOURCES:
            raise R303AggregateValidationError("migration receipt names an unknown consumer")
        if consumer_id in seen_consumers:
            raise R303AggregateValidationError("migration receipt repeats a consumer")
        seen_consumers.add(consumer_id)
        normalized_consumers.append(_migration_consumer(candidate, consumer_id=consumer_id))
    if seen_consumers != set(MIGRATION_CONSUMER_SOURCES):
        raise R303AggregateValidationError("migration receipt omits a required consumer")
    normalized_consumers.sort(key=lambda item: item["consumer_id"])
    assertions = _mapping(row.get("test_assertions"), field="migration test assertions")
    _assert_exact_keys(assertions, set(MIGRATION_TEST_ASSERTIONS), field="migration test assertions")
    for name in MIGRATION_TEST_ASSERTIONS:
        _require_bool(assertions.get(name), field=f"migration assertion {name}", expected=True)
    for name in (
        "runtime_or_service_action_performed",
        "candidate_action_time_wrapper_sealed",
        "training_handoff_authorized",
        "kaggle_queue_or_fleet_authorized",
    ):
        _require_bool(row.get(name), field=f"migration {name}", expected=False)
    recorded = _require_string(row.get("recorded_at_utc"), field="migration recorded_at_utc")
    return {
        "schema": R303_CONSUMER_MIGRATION_SCHEMA,
        "status": "passed_revision_5_rebind_nonactivating",
        "goal_gateway_path": R303_GOAL_PATH,
        "goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "r241_typed_source_path": R303_R241_TYPED_SOURCE_PATH,
        "r241_typed_source_sha256": R303_R241_TYPED_SOURCE_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "predecessor_evidence": predecessor,
        "predecessor_receipt_files": normalized_predecessor_files,
        "consumers": normalized_consumers,
        "test_assertions": {name: True for name in MIGRATION_TEST_ASSERTIONS},
        "runtime_or_service_action_performed": False,
        "candidate_action_time_wrapper_sealed": False,
        "training_handoff_authorized": False,
        "kaggle_queue_or_fleet_authorized": False,
        "recorded_at_utc": recorded,
    }


def _verify_project_source(
    *,
    source_path: str,
    claimed_sha256: str,
    claimed_size_bytes: int,
    consumer_id: str,
) -> ImmutableFileIdentity:
    """Re-hash an exact project source and prove it has an r5 binding seam.

    Source files are normally writable during development, so the immutable
    requirement applies to generated evidence rather than source checkout
    permissions.  We still reject aliases and ensure the receipt cannot point
    at an arbitrary same-named source outside this workspace.
    """

    expected = PROJECT_ROOT / MIGRATION_CONSUMER_SOURCES[consumer_id]
    actual = PROJECT_ROOT / source_path
    if actual != expected:
        raise R303AggregateValidationError(f"migration {consumer_id} source path drifted")
    target = _ensure_within(actual, PROJECT_ROOT, field=f"migration {consumer_id} source")
    identity = _read_stable_file(
        target, field=f"migration {consumer_id} source", require_readonly=False
    )
    if identity.sha256 != claimed_sha256 or identity.size_bytes != claimed_size_bytes:
        raise R303AggregateValidationError(
            f"migration {consumer_id} source bytes no longer match the receipt"
        )
    try:
        source_text = Path(identity.path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover - stable bytes above.
        raise R303AggregateValidationError(
            f"migration {consumer_id} source cannot be decoded"
        ) from exc
    has_literal_binding = (
        R303_GOAL_GATEWAY_SHA256 in source_text and R303_CONTRACT_SHA256 in source_text
    )
    has_loader_binding = (
        "R303_GOAL_GATEWAY_SHA256" in source_text
        and "R303_CONTRACT_SHA256" in source_text
    )
    if not has_literal_binding and not has_loader_binding:
        raise R303AggregateValidationError(
            f"migration {consumer_id} source has no inspectable revision-5 authority binding"
        )
    return identity


def _validate_consumer_test_evidence(
    payload: Mapping[str, Any],
    *,
    consumer: Mapping[str, Any],
) -> None:
    """Require a typed, source-bound immutable test result—not just a log."""

    row = _mapping(payload, field=f"migration {consumer['consumer_id']} test evidence")
    expected = {
        "schema",
        "status",
        "consumer_id",
        "goal_gateway_sha256",
        "goal_contract_sha256",
        "r241_typed_source_sha256",
        "goal_revision",
        "root_handoff_revision",
        "source_path",
        "source_sha256",
        "source_size_bytes",
        "test_assertions",
        "runtime_or_service_action_performed",
        "candidate_action_time_wrapper_sealed",
    }
    _assert_exact_keys(row, expected, field=f"migration {consumer['consumer_id']} test evidence")
    if (
        row.get("schema"),
        row.get("status"),
        row.get("consumer_id"),
        row.get("goal_gateway_sha256"),
        row.get("goal_contract_sha256"),
        row.get("r241_typed_source_sha256"),
        row.get("goal_revision"),
        row.get("root_handoff_revision"),
        row.get("source_path"),
        row.get("source_sha256"),
        row.get("source_size_bytes"),
    ) != (
        R303_CONSUMER_TEST_EVIDENCE_SCHEMA,
        "passed_revision_5_nonactivating_consumer_tests",
        consumer["consumer_id"],
        R303_GOAL_GATEWAY_SHA256,
        R303_CONTRACT_SHA256,
        R303_R241_TYPED_SOURCE_SHA256,
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        consumer["source_path"],
        consumer["source_sha256"],
        consumer["source_size_bytes"],
    ):
        raise R303AggregateValidationError(
            f"migration {consumer['consumer_id']} test evidence does not bind its exact source"
        )
    assertions = _mapping(row.get("test_assertions"), field="consumer test assertions")
    _assert_exact_keys(assertions, set(MIGRATION_TEST_ASSERTIONS), field="consumer test assertions")
    for name in MIGRATION_TEST_ASSERTIONS:
        _require_bool(assertions.get(name), field=f"consumer test assertion {name}", expected=True)
    _require_bool(
        row.get("runtime_or_service_action_performed"),
        field="consumer test evidence runtime/service action",
        expected=False,
    )
    _require_bool(
        row.get("candidate_action_time_wrapper_sealed"),
        field="consumer test evidence action-time wrapper",
        expected=False,
    )


def inspect_revision_5_consumer_migration_receipt(
    path: Path | str,
    *,
    trusted_evidence_root: Path | str,
) -> tuple[dict[str, Any], ImmutableFileIdentity]:
    """Strictly re-open a migration receipt and every referenced artifact.

    The evidence root must be a sealed/create-only experiment directory.  A
    portable JSON that merely has well-formed digest strings is insufficient:
    this check compares every claimed SHA/size to bytes read from disk now.
    It remains non-activating even when all evidence passes.
    """

    root = Path(trusted_evidence_root)
    payload, identity = _read_json_file(
        path,
        field="revision-5 consumer migration receipt",
        trusted_root=root,
        require_readonly=True,
    )
    normalized = validate_revision_5_consumer_migration_receipt(payload)
    for consumer in normalized["consumers"]:
        _verify_project_source(
            source_path=consumer["source_path"],
            claimed_sha256=consumer["source_sha256"],
            claimed_size_bytes=consumer["source_size_bytes"],
            consumer_id=consumer["consumer_id"],
        )
        evidence = _identity_from_mapping(
            consumer["test_evidence"], field=f"migration {consumer['consumer_id']} test evidence"
        )
        evidence_payload, observed = _read_json_file(
            evidence.path,
            field=f"migration {consumer['consumer_id']} test evidence",
            trusted_root=root,
            require_readonly=True,
        )
        if observed.sha256 != evidence.sha256 or observed.size_bytes != evidence.size_bytes:
            raise R303AggregateValidationError(
                f"migration {consumer['consumer_id']} test evidence bytes drifted"
            )
        _validate_consumer_test_evidence(evidence_payload, consumer=consumer)
    for predecessor in normalized["predecessor_receipt_files"]:
        predecessor_path = _rooted_path(
            predecessor["path"], root, field="revision-4 predecessor receipt"
        )
        observed = _read_stable_file(
            predecessor_path,
            field="revision-4 predecessor receipt",
            require_readonly=True,
        )
        if (
            observed.sha256 != predecessor["sha256"]
            or observed.size_bytes != predecessor["size_bytes"]
        ):
            raise R303AggregateValidationError("revision-4 predecessor receipt bytes drifted")
        try:
            raw = json.loads(Path(observed.path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R303AggregateValidationError("revision-4 predecessor receipt is not JSON") from exc
        if not isinstance(raw, Mapping) or raw.get("schema") != predecessor["schema"]:
            raise R303AggregateValidationError("revision-4 predecessor receipt schema drifted")
    return normalized, identity


def _handoff_identity_file(
    path: Path | str,
    *,
    trusted_receipt_root: Path,
    field: str,
) -> ImmutableFileIdentity:
    target = _rooted_path(path, trusted_receipt_root, field=field)
    return _read_stable_file(target, field=field, require_readonly=True)


def build_pre_activation_aggregate(
    *,
    migration_receipt_path: Path | str,
    migration_evidence_root: Path | str,
    census_validation_receipt_path: Path | str,
    handoff_receipt_paths: Mapping[str, Path | str],
    trusted_handoff_receipt_root: Path | str,
) -> dict[str, Any]:
    """Re-open pre-handoff receipts and summarize them without activation.

    This has no writer on purpose.  A caller that wants to preserve the result
    must use a separate create-only experiment receipt whose consumer records
    this payload's canonical digest.  That avoids accidentally treating a
    convenience aggregate as a contract activation receipt.
    """

    _assert_contract_current()
    migration, migration_identity = inspect_revision_5_consumer_migration_receipt(
        migration_receipt_path, trusted_evidence_root=migration_evidence_root
    )
    census_payload, census_identity = _read_json_file(
        census_validation_receipt_path,
        field="revision-5 census validation receipt",
        trusted_root=Path(migration_evidence_root),
        require_readonly=True,
    )
    try:
        census = validate_revision_5_census_validation_receipt(census_payload)
    except Exception as exc:
        raise R303AggregateValidationError("revision-5 census validation receipt is invalid") from exc
    handoff_root = Path(trusted_handoff_receipt_root)
    required_kinds = {
        "parent",
        "corpus",
        "schema_freeze",
        "frozen_tensors",
        "blackwell_preflight",
        "rollback_plan",
    }
    if set(handoff_receipt_paths) != required_kinds:
        raise R303AggregateValidationError(
            "pre-activation aggregate needs exactly the six contract handoff receipts"
        )
    # Check immutability before handing paths to the handoff owner, whose
    # parser owns the contract's exact top-level inventories and semantics.
    physical_identities = {
        kind: _handoff_identity_file(
            handoff_receipt_paths[kind],
            trusted_receipt_root=handoff_root,
            field=f"handoff {kind} receipt",
        )
        for kind in sorted(required_kinds)
    }
    try:
        receipts, inspected = handoff.inspect_handoff_bundle(
            handoff_receipt_paths, trusted_receipt_root=handoff_root
        )
    except handoff.R303HandoffError as exc:
        raise R303AggregateValidationError("revision-5 handoff receipt bundle is invalid") from exc
    for kind, expected in physical_identities.items():
        observed = inspected[kind]
        if (
            observed.path != expected.path
            or observed.sha256 != expected.sha256
            or observed.size_bytes != expected.size_bytes
        ):
            raise R303AggregateValidationError(
                f"handoff {kind} receipt changed between aggregate checks"
            )
    parent = receipts["parent"]
    corpus = receipts["corpus"]
    frozen = receipts["frozen_tensors"]
    preflight = receipts["blackwell_preflight"]
    if corpus["validated_deduplicated_raw_manifest_sha256"] != census[
        "raw_expert_corpus_manifest_sha256"
    ]:
        raise R303AggregateValidationError(
            "handoff corpus receipt does not bind the exact revision-5 census raw manifest"
        )
    if corpus["collision_census_receipt_sha256"] != census["collision_census_receipt_sha256"]:
        raise R303AggregateValidationError(
            "handoff corpus receipt does not bind the exact revision-5 census receipt"
        )
    aggregate = {
        "schema": R303_PREACTIVATION_AGGREGATE_SCHEMA,
        "status": "validated_pre_activation_only_no_runtime_or_handoff_authority",
        "goal_gateway_path": R303_GOAL_PATH,
        "goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "r241_typed_source_path": R303_R241_TYPED_SOURCE_PATH,
        "r241_typed_source_sha256": R303_R241_TYPED_SOURCE_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "migration_receipt": migration_identity.to_dict(),
        "migration_predecessor_evidence_schema": R298_REV5_PREDECESSOR_CLASSIFICATION_SCHEMA,
        "census_validation_receipt": census_identity.to_dict(),
        "census_validation_receipt_schema": R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA,
        "handoff_receipts": {
            kind: physical_identities[kind].to_dict() for kind in sorted(physical_identities)
        },
        "immutable_parent_checkpoint_sha256": parent["old_checkpoint_sha256"],
        "immutable_parent_optimizer_state_sha256": parent["old_optimizer_state_sha256"],
        "staged_corpus_parity_receipt_sha256": corpus["source_remote_parity_receipt_sha256"],
        "frozen_tensor_parent_checkpoint_sha256": frozen["parent_checkpoint_sha256"],
        "blackwell_preflight_host": preflight["host"],
        "blackwell_preflight_device": preflight["device"],
        "blackwell_preflight_no_runtime_or_service_change": preflight[
            "no_runtime_or_service_change_performed"
        ],
        "staged_shards_training_eligible_now": False,
        "old_r274_services_paused": False,
        "new_derivative_service_started": False,
        "shared_kaggle_queue_service_touched": False,
        "production_serving_selector_authority": False,
        "candidate_action_time_wrapper_sealed": False,
        "training_handoff_authorized": False,
        "kaggle_queue_or_fleet_authorized": False,
    }
    return validate_pre_activation_aggregate(aggregate)


def validate_pre_activation_aggregate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if a pre-activation summary tries to grant authority."""

    row = _mapping(payload, field="pre-activation aggregate")
    expected = {
        "schema",
        "status",
        "goal_gateway_path",
        "goal_gateway_sha256",
        "goal_contract_path",
        "goal_contract_sha256",
        "r241_typed_source_path",
        "r241_typed_source_sha256",
        "goal_revision",
        "root_handoff_revision",
        "migration_receipt",
        "migration_predecessor_evidence_schema",
        "census_validation_receipt",
        "census_validation_receipt_schema",
        "handoff_receipts",
        "immutable_parent_checkpoint_sha256",
        "immutable_parent_optimizer_state_sha256",
        "staged_corpus_parity_receipt_sha256",
        "frozen_tensor_parent_checkpoint_sha256",
        "blackwell_preflight_host",
        "blackwell_preflight_device",
        "blackwell_preflight_no_runtime_or_service_change",
        "staged_shards_training_eligible_now",
        "old_r274_services_paused",
        "new_derivative_service_started",
        "shared_kaggle_queue_service_touched",
        "production_serving_selector_authority",
        "candidate_action_time_wrapper_sealed",
        "training_handoff_authorized",
        "kaggle_queue_or_fleet_authorized",
    }
    _assert_exact_keys(row, expected, field="pre-activation aggregate")
    if (
        row.get("schema"),
        row.get("status"),
        row.get("goal_gateway_path"),
        row.get("goal_gateway_sha256"),
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("r241_typed_source_path"),
        row.get("r241_typed_source_sha256"),
        row.get("goal_revision"),
        row.get("root_handoff_revision"),
        row.get("migration_predecessor_evidence_schema"),
        row.get("census_validation_receipt_schema"),
        row.get("blackwell_preflight_host"),
        row.get("blackwell_preflight_device"),
    ) != (
        R303_PREACTIVATION_AGGREGATE_SCHEMA,
        "validated_pre_activation_only_no_runtime_or_handoff_authority",
        R303_GOAL_PATH,
        R303_GOAL_GATEWAY_SHA256,
        R303_CONTRACT_PATH,
        R303_CONTRACT_SHA256,
        R303_R241_TYPED_SOURCE_PATH,
        R303_R241_TYPED_SOURCE_SHA256,
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        R298_REV5_PREDECESSOR_CLASSIFICATION_SCHEMA,
        R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA,
        "inzi",
        "cuda:1",
    ):
        raise R303AggregateValidationError("pre-activation aggregate binding drifted")
    _identity_from_mapping(row.get("migration_receipt"), field="aggregate migration receipt")
    _identity_from_mapping(
        row.get("census_validation_receipt"), field="aggregate census validation receipt"
    )
    receipts = _mapping(row.get("handoff_receipts"), field="aggregate handoff receipts")
    expected_kinds = {
        "parent",
        "corpus",
        "schema_freeze",
        "frozen_tensors",
        "blackwell_preflight",
        "rollback_plan",
    }
    _assert_exact_keys(receipts, expected_kinds, field="aggregate handoff receipts")
    for kind in expected_kinds:
        _identity_from_mapping(receipts[kind], field=f"aggregate handoff {kind} receipt")
    for name in (
        "immutable_parent_checkpoint_sha256",
        "immutable_parent_optimizer_state_sha256",
        "staged_corpus_parity_receipt_sha256",
        "frozen_tensor_parent_checkpoint_sha256",
    ):
        _sha(row.get(name), field=f"aggregate {name}")
    _require_bool(
        row.get("blackwell_preflight_no_runtime_or_service_change"),
        field="aggregate Blackwell preflight boundary",
        expected=True,
    )
    for name in (
        "staged_shards_training_eligible_now",
        "old_r274_services_paused",
        "new_derivative_service_started",
        "shared_kaggle_queue_service_touched",
        "production_serving_selector_authority",
        "candidate_action_time_wrapper_sealed",
        "training_handoff_authorized",
        "kaggle_queue_or_fleet_authorized",
    ):
        _require_bool(row.get(name), field=f"aggregate {name}", expected=False)
    if row["immutable_parent_checkpoint_sha256"] != row["frozen_tensor_parent_checkpoint_sha256"]:
        raise R303AggregateValidationError(
            "pre-activation aggregate frozen tensor parent diverges from sealed r274 parent"
        )
    return dict(row)


def _receipt_spec(
    contract: Mapping[str, Any],
    *,
    section: str,
    kind: str,
) -> tuple[str, tuple[str, ...]]:
    """Return one closed receipt inventory owned by the final contract."""

    if section == "candidate":
        root = _mapping(contract.get("baselines_and_candidate"), field="candidate contract")
        schema = root.get("candidate_validation_receipt_schema")
        fields = root.get("candidate_validation_receipt_required_fields")
    elif section == "fleet":
        root = _mapping(contract.get("full_available_fleet_self_play"), field="fleet contract")
        receipts = _mapping(root.get("receipt_contract"), field="fleet receipt contract")
        spec = _mapping(receipts.get(kind), field=f"fleet receipt contract {kind}")
        schema = spec.get("schema")
        fields = spec.get("required_fields")
    else:  # pragma: no cover - private callers enumerate these literals.
        raise R303AggregateValidationError(f"unknown revision-5 receipt section: {section}")
    if not isinstance(schema, str) or not schema:
        raise R303AggregateValidationError(f"{section} receipt contract lacks a schema")
    sequence = _rows(fields, field=f"{section} receipt contract fields")
    if not sequence or any(not isinstance(field, str) or not field for field in sequence):
        raise R303AggregateValidationError(f"{section} receipt contract field inventory is malformed")
    if len(set(sequence)) != len(sequence):
        raise R303AggregateValidationError(f"{section} receipt contract repeats a field")
    return schema, tuple(sequence)


def _validate_exact_contract_receipt(
    receipt: Mapping[str, Any],
    *,
    schema: str,
    fields: Sequence[str],
    label: str,
) -> dict[str, Any]:
    row = _mapping(receipt, field=label)
    _assert_exact_keys(row, set(fields), field=label)
    if row.get("schema") != schema:
        raise R303AggregateValidationError(f"{label} schema mismatch")
    _validate_contract_binding(row, label=label)
    for key, value in row.items():
        if key.endswith("_sha256"):
            _sha(value, field=f"{label}.{key}")
    return dict(row)


def validate_candidate_validation_receipt_r303(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the final contract's candidate evidence without launching it.

    The contract intentionally parameterizes the candidate checkpoint path;
    therefore this parser validates the receipt's closed inventory and
    identities but does *not* claim a portable JSON has re-hashed checkpoint
    bytes.  A future activation consumer must still use a receipt-rooted
    checkpoint inspection boundary.
    """

    contract = _assert_contract_current()
    schema, fields = _receipt_spec(contract, section="candidate", kind="candidate")
    row = _validate_exact_contract_receipt(
        receipt, schema=schema, fields=fields, label="revision-5 candidate validation receipt"
    )
    baseline = _mapping(contract.get("baselines_and_candidate"), field="candidate baseline contract")
    candidate = _mapping(baseline.get("derivative_candidate"), field="derivative candidate contract")
    expected = {
        "root_owner_revision": R303_ROOT_OWNER_REVISION,
        "candidate_id": candidate["candidate_id"],
        "baseline_r274_candidate_id": baseline["r241_r274_same_architecture_baseline"][
            "candidate_id"
        ],
        "exact_new_list_canonical_multiset_sha256": baseline[
            "r241_r274_same_architecture_baseline"
        ]["exact_new_list_canonical_multiset_sha256"],
        "frozen_backbone_true": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
        "all_frozen_tensor_bit_identity": True,
        "public_information_metamorphic_tests_passed": True,
        "simulator_engine_tests_passed": True,
        "bootstrap_validation_passed": True,
        "production_serving_selector_authority": False,
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise R303AggregateValidationError(
                f"revision-5 candidate validation receipt {field} mismatch"
            )
    for field in (
        "candidate_checkpoint_size_bytes",
        "baseline_r274_checkpoint_size_bytes",
    ):
        _positive_int(row.get(field), field=f"candidate {field}")
    if not isinstance(row.get("candidate_checkpoint_model_schema"), str) or not row[
        "candidate_checkpoint_model_schema"
    ]:
        raise R303AggregateValidationError("candidate checkpoint model schema is missing")
    for field in (
        "candidate_checkpoint_parameter_inventory_sha256",
        "bootstrap_training_manifest_sha256",
        "bootstrap_optimizer_config_sha256",
        "training_handoff_activation_receipt_sha256",
        "parent_receipt_sha256",
        "corpus_receipt_sha256",
        "schema_freeze_receipt_sha256",
        "frozen_tensor_receipt_sha256",
        "blackwell_preflight_receipt_sha256",
    ):
        _sha(row.get(field), field=f"candidate {field}")
    if not isinstance(row.get("bootstrap_rows_steps_and_epochs"), Mapping):
        raise R303AggregateValidationError("candidate bootstrap row/step/epoch evidence is malformed")
    if not isinstance(row.get("bootstrap_loss_gradient_and_resource_peaks"), Mapping):
        raise R303AggregateValidationError("candidate bootstrap loss/resource evidence is malformed")
    _require_bool(
        row.get("trainable_parameters_changed_and_finite"),
        field="candidate trainable parameters changed and finite",
        expected=True,
    )
    return row


def validate_kaggle_queue_chain_r303(
    *,
    candidate_receipt: Mapping[str, Any],
    candidate_receipt_identity: ImmutableFileIdentity,
    package_receipt: Mapping[str, Any],
    package_receipt_identity: ImmutableFileIdentity,
    queue_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Check a receipt chain for one future queue entry without queueing it."""

    candidate = validate_candidate_validation_receipt_r303(candidate_receipt)
    try:
        package = handoff.validate_kaggle_receipt("package", package_receipt)
        queue = handoff.validate_kaggle_receipt("queue", queue_receipt)
    except handoff.R303HandoffError as exc:
        raise R303AggregateValidationError("Kaggle package or queue receipt is invalid") from exc
    if package["candidate_validation_receipt_sha256"] != candidate_receipt_identity.sha256:
        raise R303AggregateValidationError("Kaggle package does not bind exact candidate receipt bytes")
    if package["candidate_checkpoint_sha256"] != candidate["candidate_checkpoint_sha256"]:
        raise R303AggregateValidationError("Kaggle package candidate checkpoint diverges from validation")
    if package["parent_checkpoint_sha256"] != candidate["baseline_r274_checkpoint_sha256"]:
        raise R303AggregateValidationError("Kaggle package parent checkpoint diverges from validation")
    if queue["package_receipt_sha256"] != package_receipt_identity.sha256:
        raise R303AggregateValidationError("Kaggle queue does not bind exact package receipt bytes")
    if queue["package_sha256"] != package["package_sha256"]:
        raise R303AggregateValidationError("Kaggle queue package identity diverges from package receipt")
    if queue["candidate_checkpoint_sha256"] != candidate["candidate_checkpoint_sha256"]:
        raise R303AggregateValidationError("Kaggle queue candidate diverges from validation")
    return {
        "candidate_receipt_sha256": candidate_receipt_identity.sha256,
        "package_receipt_sha256": package_receipt_identity.sha256,
        "queue_entry_status": queue["queue_entry_status"],
        "queue_allows_future_self_play_only_after_separate_activation_receipt": True,
        "queue_or_upload_performed_by_validator": False,
    }


def validate_fleet_receipt_r303(kind: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Parse one exact final-contract fleet receipt without starting workers."""

    if kind not in {"fleet_inventory", "package_parity", "routing", "activation", "collection_shard"}:
        raise R303AggregateValidationError(f"unknown fleet receipt kind: {kind}")
    contract = _assert_contract_current()
    schema, fields = _receipt_spec(contract, section="fleet", kind=kind)
    row = _validate_exact_contract_receipt(
        receipt, schema=schema, fields=fields, label=f"revision-5 fleet {kind} receipt"
    )
    fleet = _mapping(contract.get("full_available_fleet_self_play"), field="fleet contract")
    known_hosts = set(_rows(fleet.get("known_candidate_hosts"), field="fleet known hosts"))

    def per_host(value: object, *, field: str, all_true: bool = False) -> Mapping[str, Any]:
        result = _mapping(value, field=field)
        if set(result) != known_hosts:
            raise R303AggregateValidationError(f"{field} does not account for every known host")
        if all_true:
            for host, status in result.items():
                _require_bool(status, field=f"{field}.{host}", expected=True)
        return result

    if kind == "fleet_inventory":
        if row.get("known_candidate_hosts") != fleet["known_candidate_hosts"]:
            raise R303AggregateValidationError("fleet inventory does not cover the exact known-host set")
        _require_bool(
            row.get("full_available_fleet_included"),
            field="fleet inventory full_available_fleet_included",
            expected=True,
        )
        for field in (
            "per_host_availability_eligibility_and_exclusion_evidence",
            "per_host_os_arch_cpu_ram_and_network_identity",
            "per_host_gpu_device_models_uuids_memory_and_runtime_versions",
            "per_host_simulator_capacity_and_pack_sizes",
            "per_host_resource_caps",
            "per_host_managed_worker_service_names",
        ):
            per_host(row.get(field), field=f"fleet inventory {field}")
    elif kind == "package_parity":
        _require_bool(
            row.get("all_available_hosts_checksum_exact"),
            field="fleet package parity checksum exactness",
            expected=True,
        )
        per_host(
            row.get("per_host_package_path_sha256_size_and_member_manifest"),
            field="fleet package parity package identities",
        )
        per_host(
            row.get("per_host_layer_off_and_action_parity"),
            field="fleet package parity layer-off/action parity",
            all_true=True,
        )
        per_host(
            row.get("per_host_engine_smoke_passed"),
            field="fleet package parity engine smoke",
            all_true=True,
        )
    elif kind == "routing":
        if row.get("learner_host") != "inzi":
            raise R303AggregateValidationError("fleet routing does not retain Inzi as sole learner")
        _require_bool(row.get("no_oversubscription"), field="fleet routing oversubscription", expected=True)
        _require_bool(
            row.get("no_unreceipted_fallback"), field="fleet routing fallback", expected=True
        )
        for field in (
            "per_host_simulator_process_count",
            "per_process_environment_count",
            "per_host_policy_leaf_device_uuid_and_worker_ranges",
            "per_host_concurrency_and_resource_caps",
            "capacity_probe_results",
        ):
            per_host(row.get(field), field=f"fleet routing {field}")
        _positive_int(
            row.get("blackwell_training_headroom_bytes"),
            field="fleet routing Blackwell headroom",
            allow_zero=True,
        )
    elif kind == "activation":
        for field in (
            "old_r274_services_inactive_verified",
            "shared_kaggle_queue_service_unchanged",
            "no_old_r274_training_or_collection",
            "no_serving_selector_change",
        ):
            _require_bool(row.get(field), field=f"fleet activation {field}", expected=True)
        if row.get("new_managed_trainer_service") != fleet["new_managed_trainer_service"]:
            raise R303AggregateValidationError("fleet activation names a foreign trainer service")
        if not isinstance(row.get("new_managed_worker_services"), list) or not all(
            isinstance(item, str) and item for item in row["new_managed_worker_services"]
        ):
            raise R303AggregateValidationError("fleet activation worker service inventory is malformed")
        _require_string(row.get("global_collection_id"), field="fleet activation global collection id")
        _require_string(
            row.get("global_game_identity_schema"), field="fleet activation game identity schema"
        )
        _require_string(
            row.get("lease_and_duplicate_suppression_schema"),
            field="fleet activation duplicate-suppression schema",
        )
    else:  # collection_shard
        for field in (
            "selected_action_legality_passed",
            "schema_validation_passed",
            "sealed_append_only",
        ):
            _require_bool(row.get(field), field=f"fleet collection {field}", expected=True)
        if row.get("duplicate_conflict_count") != 0:
            raise R303AggregateValidationError("fleet collection has a conflicting duplicate game")
        for field in ("size_bytes", "duplicate_identical_count", "missing_game_count", "simulator_failure_count"):
            _positive_int(row.get(field), field=f"fleet collection {field}", allow_zero=True)
        if row.get("missing_game_count") != 0 or row.get("simulator_failure_count") != 0:
            raise R303AggregateValidationError("fleet collection has unaccounted missing or failed games")
        game_ids = _rows(row.get("ordered_global_game_ids"), field="fleet collection game identities")
        if not game_ids or any(not isinstance(item, str) or not item for item in game_ids):
            raise R303AggregateValidationError("fleet collection game identities are malformed")
        if len(set(game_ids)) != len(game_ids):
            raise R303AggregateValidationError("fleet collection contains duplicate global game credit")
        counts = per_host(row.get("host_contribution_counts"), field="fleet collection host contributions")
        for host, count in counts.items():
            _positive_int(count, field=f"fleet collection host contribution {host}", allow_zero=True)
        per_host(row.get("resource_peaks_by_host"), field="fleet collection resource peaks")
    return row


def validate_fleet_activation_chain_r303(
    *,
    candidate_receipt: Mapping[str, Any],
    candidate_receipt_identity: ImmutableFileIdentity,
    queue_receipt: Mapping[str, Any],
    queue_receipt_identity: ImmutableFileIdentity,
    inventory_receipt: Mapping[str, Any],
    inventory_receipt_identity: ImmutableFileIdentity,
    package_parity_receipt: Mapping[str, Any],
    package_parity_receipt_identity: ImmutableFileIdentity,
    routing_receipt: Mapping[str, Any],
    routing_receipt_identity: ImmutableFileIdentity,
    activation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate cross-links for a recorded future activation, never enact it."""

    candidate = validate_candidate_validation_receipt_r303(candidate_receipt)
    try:
        queue = handoff.validate_kaggle_receipt("queue", queue_receipt)
    except handoff.R303HandoffError as exc:
        raise R303AggregateValidationError("fleet chain queue receipt is invalid") from exc
    inventory = validate_fleet_receipt_r303("fleet_inventory", inventory_receipt)
    parity = validate_fleet_receipt_r303("package_parity", package_parity_receipt)
    routing = validate_fleet_receipt_r303("routing", routing_receipt)
    activation = validate_fleet_receipt_r303("activation", activation_receipt)
    if activation["candidate_validation_receipt_sha256"] != candidate_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet activation candidate receipt binding drifted")
    if activation["kaggle_queue_receipt_sha256"] != queue_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet activation queue receipt binding drifted")
    if activation["kaggle_queue_entry_status"] != queue["queue_entry_status"]:
        raise R303AggregateValidationError("fleet activation queue status drifted")
    if activation["fleet_inventory_receipt_sha256"] != inventory_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet activation inventory receipt binding drifted")
    if activation["package_parity_receipt_sha256"] != package_parity_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet activation package parity binding drifted")
    if activation["routing_receipt_sha256"] != routing_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet activation routing receipt binding drifted")
    if parity["fleet_inventory_receipt_sha256"] != inventory_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet package parity inventory binding drifted")
    if routing["fleet_inventory_receipt_sha256"] != inventory_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet routing inventory binding drifted")
    if routing["package_parity_receipt_sha256"] != package_parity_receipt_identity.sha256:
        raise R303AggregateValidationError("fleet routing parity binding drifted")
    if parity["candidate_checkpoint_sha256"] != candidate["candidate_checkpoint_sha256"]:
        raise R303AggregateValidationError("fleet package parity candidate checkpoint drifted")
    return {
        "activation_receipt_schema": activation["schema"],
        "activation_receipt_is_only_validated_not_executed": True,
        "service_control_performed_by_validator": False,
        "queue_or_fleet_worker_started_by_validator": False,
    }


__all__ = [
    "ImmutableFileIdentity",
    "MIGRATION_CONSUMER_SOURCES",
    "MIGRATION_TEST_ASSERTIONS",
    "R303AggregateValidationError",
    "R303_CONSUMER_MIGRATION_SCHEMA",
    "R303_CONSUMER_TEST_EVIDENCE_SCHEMA",
    "R303_PREACTIVATION_AGGREGATE_SCHEMA",
    "R303_R241_TYPED_SOURCE_PATH",
    "R303_R241_TYPED_SOURCE_SHA256",
    "build_pre_activation_aggregate",
    "canonical_sha256",
    "inspect_revision_5_consumer_migration_receipt",
    "sha256_file",
    "validate_candidate_validation_receipt_r303",
    "validate_fleet_activation_chain_r303",
    "validate_fleet_receipt_r303",
    "validate_kaggle_queue_chain_r303",
    "validate_pre_activation_aggregate",
    "validate_revision_5_consumer_migration_receipt",
]
