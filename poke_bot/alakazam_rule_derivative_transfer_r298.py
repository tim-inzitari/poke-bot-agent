"""Create-only quarantined shard staging for the isolated r298 experiment.

This module is intentionally separate from all policy, training, selector,
and service code.  It has two safe modes:

* a deterministic, write-free transfer plan that may be made as soon as a
  locally immutable shard has passed schema/SHA/size checks; and
* an explicit transport-protocol executor for the narrow revision-4 staging
  exception.  The executor is inert unless a caller supplies a transport and
  explicitly opts in.  It can only create a private partial object and publish
  a final object by a create-only operation; it never loads, decodes, imports,
  mmaps, evaluates, trains on, or replaces staged data.

The physical transport is deliberately injected.  That keeps credentials and
host-specific SSH mechanics out of the derivative model and makes the safety
checks testable without contacting Inzi.  A conforming transport must honour
the methods' exact create-only contracts below; every result is re-validated
by this module before it receives a receipt.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol

from .alakazam_rule_derivative_training_r298 import (
    MAX_PHYSICAL_SHARD_BYTES,
    R298_CONTRACT_PATH,
    R298_CONTRACT_SHA256,
    R298_GOAL_PATH,
    R298_GOAL_SHA256,
    R298_RAW_WINDOW_END,
    R298_RAW_WINDOW_START,
    R298_REFEATURE_MANIFEST_SCHEMA,
    R298_REFEATURE_RECORD_SCHEMA,
    RuleDerivativeTrainingError,
    validate_raw_corpus_binding,
    verify_raw_source_receipt_files,
)
from .alakazam_collision_census_r298 import (
    canonical_sha256 as collision_canonical_sha256,
    validate_frozen_schema_gate,
)


R298_TRANSFER_PLAN_SCHEMA: Final = (
    "poke_bot.alakazam_elmo_refeaturization_transfer_plan/v1"
)
R298_TRANSFER_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_elmo_refeaturization_shard_transfer_receipt/v1"
)
R298_TRANSFER_PARITY_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_elmo_refeaturization_transfer_parity_receipt/v1"
)
INZI_QUARANTINE_DIRECTORY: Final = (
    "/home/inzi/poke-bot-agent/outputs/quarantine/"
    "alakazam-elmo-rule-derivative/g4-refeaturized-census-shards"
)
INZI_DESTINATION_HOST: Final = "inzi"
MAX_TRANSFER_LANES: Final = 4
_SHA256_RE: Final = re.compile(r"^sha256:([0-9a-f]{64})$")
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_FILE: Final = PROJECT_ROOT / R298_CONTRACT_PATH
GOAL_FILE: Final = PROJECT_ROOT / R298_GOAL_PATH
MAX_RECORD_JSON_BYTES: Final = 64 * 1024 * 1024


class RuleDerivativeTransferError(RuleDerivativeTrainingError):
    """A rev4 staging input or remote observation cannot be trusted."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Use the shared collision-census canonical receipt encoding."""

    try:
        return collision_canonical_sha256(value)
    except Exception as exc:
        raise RuleDerivativeTransferError("value is not canonical r298 JSON") from exc


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_immutable_json(path: Path | str, *, field: str) -> tuple[Path, Mapping[str, Any], str]:
    """Open one receipt/manifest as a regular immutable JSON source.

    Re-featurization and staging are create-only, but we still refuse a
    symlink, special file, malformed JSON, or a path that disappears during a
    read.  The returned digest is the exact source-file SHA, distinct from the
    canonical payload digest used in cross-receipt bindings.
    """

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RuleDerivativeTransferError(f"{field} must be a regular non-symlink file")
    before = target.stat()
    digest = sha256_file(target)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleDerivativeTransferError(f"cannot read {field}") from exc
    after = target.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(target) != digest:
        raise RuleDerivativeTransferError(f"{field} changed while being read")
    return target.resolve(), _mapping(payload, field=field), digest


def _load_contract_staging_binding() -> Mapping[str, Any]:
    """Re-open the sole canonical contract and prove the narrow staging scope."""

    contract_path, contract, digest = _read_immutable_json(CONTRACT_FILE, field="r298 contract")
    if contract_path != CONTRACT_FILE.resolve() or digest != R298_CONTRACT_SHA256:
        raise RuleDerivativeTransferError("on-disk r298 contract identity drifted")
    if GOAL_FILE.is_symlink() or not GOAL_FILE.is_file() or sha256_file(GOAL_FILE) != R298_GOAL_SHA256:
        raise RuleDerivativeTransferError("on-disk r298 goal gateway identity drifted")
    staging = _mapping(contract.get("create_only_inzi_transfer_staging"), field="contract staging")
    if (
        staging.get("owner_goal_revision"),
        staging.get("source_host"),
        staging.get("destination_host"),
        staging.get("destination_directory"),
        staging.get("destination_role"),
        staging.get("staged_objects_training_or_evaluation_eligible_on_inzi"),
        staging.get("staging_or_final_parity_receipt_grants_later_inzi_use"),
    ) != (
        4,
        "elmo",
        INZI_DESTINATION_HOST,
        INZI_QUARANTINE_DIRECTORY,
        "quarantined_non_runtime_create_only_staging",
        False,
        False,
    ):
        raise RuleDerivativeTransferError("canonical contract staging destination/authority drifted")
    transport = _mapping(staging.get("transport"), field="contract staging transport")
    if transport.get("parallel_lanes_maximum") != MAX_TRANSFER_LANES or transport.get(
        "strict_per_object_limit_is_96_mib"
    ) is not True or transport.get("atomic_create_only_final_publication_required") is not True:
        raise RuleDerivativeTransferError("canonical contract staging transport guard drifted")
    return staging


@dataclass(frozen=True)
class TransferEvidence:
    """Re-opened immutable evidence required before planning or staging bytes."""

    refeature_manifest: Mapping[str, Any]
    raw_manifest: Mapping[str, Any]
    raw_receipt: Mapping[str, Any]
    frozen_schema_manifest: Mapping[str, Any]
    zero_bypass_receipt: Mapping[str, Any]
    file_identities: Mapping[str, Mapping[str, Any]]

    @property
    def refeature_manifest_sha256(self) -> str:
        return canonical_sha256(self.refeature_manifest)

    @property
    def raw_manifest_sha256(self) -> str:
        return canonical_sha256(self.raw_manifest)

    @property
    def frozen_schema_manifest_sha256(self) -> str:
        return canonical_sha256(self.frozen_schema_manifest)

    @property
    def zero_bypass_receipt_sha256(self) -> str:
        return canonical_sha256(self.zero_bypass_receipt)


def load_transfer_evidence(
    *,
    refeature_manifest_path: Path | str,
    raw_manifest_path: Path | str,
    raw_receipt_path: Path | str,
    frozen_schema_manifest_path: Path | str,
    zero_bypass_receipt_path: Path | str,
) -> TransferEvidence:
    """Re-open/re-hash every evidence file that a staging plan relies upon.

    Strings carried by a re-feature manifest are never sufficient.  The
    actual typed raw manifest/receipt and frozen schema/zero-bypass files are
    reopened and tied back to one another before any source object is read.
    """

    _load_contract_staging_binding()
    files = {
        "refeature_manifest": _read_immutable_json(
            refeature_manifest_path, field="re-featurization manifest"
        ),
        "raw_manifest": _read_immutable_json(raw_manifest_path, field="raw 30-day manifest"),
        "raw_receipt": _read_immutable_json(raw_receipt_path, field="raw 30-day receipt"),
        "frozen_schema_manifest": _read_immutable_json(
            frozen_schema_manifest_path, field="frozen schema manifest"
        ),
        "zero_bypass_receipt": _read_immutable_json(
            zero_bypass_receipt_path, field="zero-bypass receipt"
        ),
    }
    refeature = files["refeature_manifest"][1]
    raw_manifest = files["raw_manifest"][1]
    raw_receipt = files["raw_receipt"][1]
    frozen = files["frozen_schema_manifest"][1]
    bypass = files["zero_bypass_receipt"][1]
    try:
        validate_raw_corpus_binding(raw_manifest, raw_receipt)
        verify_raw_source_receipt_files(raw_manifest, raw_receipt)
        frozen_sha, bypass_sha = validate_frozen_schema_gate(frozen, bypass)
    except Exception as exc:
        raise RuleDerivativeTransferError("raw/frozen evidence failed its canonical r298 gate") from exc
    if refeature.get("schema") != R298_REFEATURE_MANIFEST_SCHEMA:
        raise RuleDerivativeTransferError("re-featurization manifest schema drifted")
    if refeature.get("validated_30_day_raw_manifest_sha256") != canonical_sha256(raw_manifest):
        raise RuleDerivativeTransferError("re-featurization manifest does not bind reopened raw manifest")
    if refeature.get("frozen_schema_manifest_sha256") != frozen_sha:
        raise RuleDerivativeTransferError("re-featurization manifest does not bind reopened frozen schema")
    if refeature.get("zero_bypass_receipt_sha256") != bypass_sha:
        raise RuleDerivativeTransferError("re-featurization manifest does not bind reopened zero-bypass receipt")
    for field in (
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    ):
        expected = frozen.get(field)
        if not isinstance(expected, str) or refeature.get(field) != expected:
            raise RuleDerivativeTransferError(
                f"re-featurization manifest does not bind reopened frozen {field}"
            )
    identities = {
        name: {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
        for name, (path, _payload, digest) in files.items()
    }
    return TransferEvidence(
        refeature_manifest=refeature,
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        frozen_schema_manifest=frozen,
        zero_bypass_receipt=bypass,
        file_identities=identities,
    )


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuleDerivativeTransferError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuleDerivativeTransferError(f"{field} must be a list")
    return list(value)


def _positive_int(value: Any, *, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuleDerivativeTransferError(f"{field} must be a positive exact integer")
    if maximum is not None and value > maximum:
        raise RuleDerivativeTransferError(f"{field} exceeds its allowed maximum")
    return value


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuleDerivativeTransferError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuleDerivativeTransferError(f"{field} must be a relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuleDerivativeTransferError(f"{field} escapes the materialization root")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def content_addressed_filename(source_sha256: str) -> str:
    """Return the exact rev4 final object name for one physical shard."""

    digest = _sha(source_sha256, field="source shard SHA-256")
    return f"sha256-{digest.removeprefix('sha256:')}.refeaturization-census.shard"


def _required_schema_binding(manifest: Mapping[str, Any]) -> dict[str, str]:
    fields = (
        "frozen_schema_manifest_sha256",
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    )
    result = {field: _sha(manifest.get(field), field=f"refeature manifest {field}") for field in fields}
    return result


def _validate_complete_shard_stream(
    path: Path,
    *,
    expected_record_count: int,
    expected_uncompressed_jsonl_bytes: int,
    expected_bindings: Mapping[str, str],
) -> None:
    """Stream and validate every JSONL record, including a late corrupted row.

    A header-only check is insufficient: an otherwise valid shard could carry
    a mismatched schema/binding in its final record.  This uses bounded lines
    and never materializes a whole gzip payload in memory.  Reaching EOF also
    validates gzip trailers/CRCs rather than merely parsing an initial member.
    """

    count = 0
    uncompressed = 0
    try:
        with gzip.open(path, "rb") as stream:
            while True:
                line = stream.readline(MAX_RECORD_JSON_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_RECORD_JSON_BYTES:
                    raise RuleDerivativeTransferError("physical shard contains an overlong JSONL record")
                if not line.endswith(b"\n"):
                    raise RuleDerivativeTransferError("physical shard has a non-terminated JSONL record")
                uncompressed += len(line)
                if uncompressed > expected_uncompressed_jsonl_bytes:
                    raise RuleDerivativeTransferError("physical shard uncompressed byte count exceeds manifest")
                try:
                    value = json.loads(line)
                except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuleDerivativeTransferError(
                        f"physical shard JSONL record {count} is malformed"
                    ) from exc
                record = _mapping(value, field=f"physical shard record {count}")
                if record.get("schema") != R298_REFEATURE_RECORD_SCHEMA:
                    raise RuleDerivativeTransferError("physical shard record schema drifted")
                bindings = _mapping(
                    record.get("materialization_bindings"),
                    field=f"physical shard record {count} bindings",
                )
                for field, expected in expected_bindings.items():
                    if bindings.get(field) != expected:
                        raise RuleDerivativeTransferError(
                            f"physical shard record {count} does not bind expected {field}"
                        )
                count += 1
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise RuleDerivativeTransferError("physical shard gzip stream is malformed") from exc
    if count != expected_record_count:
        raise RuleDerivativeTransferError("physical shard record count disagrees with manifest")
    if uncompressed != expected_uncompressed_jsonl_bytes:
        raise RuleDerivativeTransferError("physical shard uncompressed byte count disagrees with manifest")


def validate_refeatured_shard(
    output_root: Path | str,
    shard: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable shard against its re-featurization manifest.

    This intentionally re-hashes the byte object instead of trusting a day
    marker.  The check is bounded (header-only decompression) and cannot load
    the artifact into a policy or training process.
    """

    root = Path(output_root)
    row = _mapping(shard, field="physical shard")
    relative = _relative_path(row.get("relative_path"), field="physical shard relative path")
    source = root / relative
    if source.is_symlink() or not source.is_file():
        raise RuleDerivativeTransferError(f"physical shard is unavailable: {source}")
    size = _positive_int(row.get("size_bytes"), field="physical shard size", maximum=MAX_PHYSICAL_SHARD_BYTES)
    if source.stat().st_size != size:
        raise RuleDerivativeTransferError("physical shard exact size drifted")
    digest = _sha(row.get("sha256"), field="physical shard SHA-256")
    if sha256_file(source) != digest:
        raise RuleDerivativeTransferError("physical shard SHA-256 drifted")
    record_count = _positive_int(row.get("record_count"), field="physical shard record count")
    uncompressed = _positive_int(
        row.get("uncompressed_jsonl_bytes"), field="physical shard uncompressed byte count"
    )
    expected_binding = _required_schema_binding(manifest)
    expected_binding["validated_30_day_raw_manifest_sha256"] = _sha(
        manifest.get("validated_30_day_raw_manifest_sha256"),
        field="refeature manifest raw corpus binding",
    )
    _validate_complete_shard_stream(
        source,
        expected_record_count=record_count,
        expected_uncompressed_jsonl_bytes=uncompressed,
        expected_bindings=expected_binding,
    )
    return {
        "relative_path": str(relative),
        "sha256": digest,
        "size_bytes": size,
        "record_count": record_count,
        "uncompressed_jsonl_bytes": uncompressed,
        "source_schema_validation_passed": True,
    }


def validate_refeature_manifest_for_transfer(
    output_root: Path | str,
    manifest: Mapping[str, Any],
    *,
    evidence: TransferEvidence | None = None,
) -> list[dict[str, Any]]:
    """Fail closed unless a complete frozen-schema 30-day shard set is present."""

    row = _mapping(manifest, field="refeature manifest")
    if row.get("schema") != R298_REFEATURE_MANIFEST_SCHEMA:
        raise RuleDerivativeTransferError("transfer source has the wrong re-featurization manifest schema")
    if row.get("status") != "complete_30_day_create_only_refeaturization_pending_census_gate":
        raise RuleDerivativeTransferError("transfer requires a completed immutable 30-day source manifest")
    if row.get("goal_contract_path") != R298_CONTRACT_PATH or row.get("goal_contract_sha256") != R298_CONTRACT_SHA256:
        raise RuleDerivativeTransferError("transfer source binds a stale goal contract")
    if row.get("goal_gateway_sha256") != R298_GOAL_SHA256:
        raise RuleDerivativeTransferError("transfer source binds a stale goal gateway")
    if row.get("legacy_feature_source_sha256") != (
        "sha256:ce0eb08ca74e337fe6ee1eaeb678eb42bd7f413bbfab04f8a228c3cfd3ce3db5"
    ):
        raise RuleDerivativeTransferError("transfer source does not preserve the exact r274 legacy ABI")
    if row.get("utc_day_partitions") != [
        *[f"2026-07-{day:02d}" for day in range(13, 32)],
        *[f"2026-08-{day:02d}" for day in range(1, 12)],
    ] or (row.get("utc_day_partitions") or [None])[0] != R298_RAW_WINDOW_START or (row.get("utc_day_partitions") or [None])[-1] != R298_RAW_WINDOW_END:
        raise RuleDerivativeTransferError("transfer source does not contain the exact 30 UTC day window")
    if row.get("all_physical_shards_at_or_below_96_mib") is not True:
        raise RuleDerivativeTransferError("transfer source has an over-limit shard")
    _required_schema_binding(row)
    _sha(row.get("validated_30_day_raw_manifest_sha256"), field="refeature manifest raw corpus binding")
    if evidence is None:
        raise RuleDerivativeTransferError(
            "transfer validation requires reopened raw/schema/zero-bypass evidence"
        )
    if row is not evidence.refeature_manifest and canonical_sha256(row) != evidence.refeature_manifest_sha256:
        raise RuleDerivativeTransferError("transfer source manifest differs from reopened evidence")
    expected_evidence_bindings = {
        "validated_30_day_raw_manifest_sha256": evidence.raw_manifest_sha256,
        "frozen_schema_manifest_sha256": evidence.frozen_schema_manifest_sha256,
        "zero_bypass_receipt_sha256": evidence.zero_bypass_receipt_sha256,
    }
    for field, expected in expected_evidence_bindings.items():
        if row.get(field) != expected:
            raise RuleDerivativeTransferError(
                f"transfer source manifest does not bind reopened {field}"
            )
    physical = _rows(row.get("physical_shards"), field="refeature physical shards")
    if not physical or row.get("physical_shard_count") != len(physical):
        raise RuleDerivativeTransferError("transfer source physical shard inventory is incomplete")
    seen_sha: set[str] = set()
    seen_logical: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in physical:
        shard = validate_refeatured_shard(output_root, _mapping(raw, field="manifest physical shard"), manifest=row)
        logical_id = raw.get("logical_id")
        if not isinstance(logical_id, str) or not logical_id or logical_id in seen_logical:
            raise RuleDerivativeTransferError("transfer source has duplicate/missing shard logical identities")
        if shard["sha256"] in seen_sha:
            raise RuleDerivativeTransferError("transfer source repeats a physical byte object")
        seen_logical.add(logical_id)
        seen_sha.add(shard["sha256"])
        validated.append({"source_shard_logical_id": logical_id, **shard})
    return validated


def build_transfer_plan(
    output_root: Path | str,
    *,
    refeature_manifest_path: Path | str,
    raw_manifest_path: Path | str,
    raw_receipt_path: Path | str,
    frozen_schema_manifest_path: Path | str,
    zero_bypass_receipt_path: Path | str,
    lane_count: int = MAX_TRANSFER_LANES,
) -> dict[str, Any]:
    """Make a deterministic, dry-run-only rev4 transfer plan.

    This performs no remote I/O and produces no shard receipt.  A plan is not
    parity evidence and cannot make a candidate, BO250, or any Inzi use legal.
    """

    if isinstance(lane_count, bool) or not isinstance(lane_count, int) or not 1 <= lane_count <= MAX_TRANSFER_LANES:
        raise RuleDerivativeTransferError("transfer lane count must be an integer from 1 through 4")
    evidence = load_transfer_evidence(
        refeature_manifest_path=refeature_manifest_path,
        raw_manifest_path=raw_manifest_path,
        raw_receipt_path=raw_receipt_path,
        frozen_schema_manifest_path=frozen_schema_manifest_path,
        zero_bypass_receipt_path=zero_bypass_receipt_path,
    )
    if Path(refeature_manifest_path).resolve().parent != Path(output_root).resolve():
        raise RuleDerivativeTransferError(
            "re-featurization manifest must live at the declared source output root"
        )
    manifest = evidence.refeature_manifest
    validated = validate_refeature_manifest_for_transfer(
        output_root, manifest, evidence=evidence
    )
    binding = _required_schema_binding(manifest)
    raw_binding = _sha(
        manifest.get("validated_30_day_raw_manifest_sha256"),
        field="validated 30-day raw manifest binding",
    )
    entries: list[dict[str, Any]] = []
    for index, shard in enumerate(sorted(validated, key=lambda value: (value["source_shard_logical_id"], value["sha256"]))):
        filename = content_addressed_filename(shard["sha256"])
        entries.append(
            {
                **shard,
                "content_addressed_filename": filename,
                "destination_directory": INZI_QUARANTINE_DIRECTORY,
                "destination_path": f"{INZI_QUARANTINE_DIRECTORY}/{filename}",
                "parallel_lane_id": index % lane_count,
            }
        )
    plan = {
        "schema": R298_TRANSFER_PLAN_SCHEMA,
        "version": 1,
        "status": "planned_no_inzi_bytes_mutated",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "goal_revision": 4,
        "source_manifest_sha256": canonical_sha256(manifest),
        "source_manifest_file_sha256": evidence.file_identities["refeature_manifest"]["sha256"],
        "schema_binding": binding,
        "validated_30_day_raw_manifest_binding": raw_binding,
        "reopened_evidence": dict(evidence.file_identities),
        "raw_corpus_receipt_sha256": canonical_sha256(evidence.raw_receipt),
        "zero_bypass_receipt_sha256": evidence.zero_bypass_receipt_sha256,
        "destination_host": INZI_DESTINATION_HOST,
        "destination_directory": INZI_QUARANTINE_DIRECTORY,
        "max_parallel_lanes": MAX_TRANSFER_LANES,
        "requested_parallel_lanes": lane_count,
        "entries": entries,
        "per_shard_receipts": [],
        "actual_transfer_started": False,
        "remote_io_performed": False,
        "source_schema_validation_passed": True,
        "resumable_private_partial_publication": True,
        "exact_match_disposition": "skipped_exact",
        "conflicting_final_disposition": "refused_no_overwrite_delete_rename_or_replacement",
        "candidate_training_or_bo250_allowed": False,
        "inzi_loading_or_execution_authority": False,
        "inzi_forbidden_uses": [
            "load",
            "decode",
            "import",
            "mmap",
            "cache_warm",
            "training",
            "collection",
            "selector",
            "runtime",
            "worker",
            "managed_service_create_edit_stop_restart_reload_or_replace",
            "checkpoint",
            "package",
            "submission",
            "propagation",
            "activation",
        ],
    }
    return plan


def reopen_plan_evidence(plan: Mapping[str, Any]) -> TransferEvidence:
    """Re-open the plan's recorded source files before staging any bytes."""

    row = _mapping(plan, field="transfer plan")
    evidence_rows = _mapping(row.get("reopened_evidence"), field="transfer plan reopened evidence")
    expected_names = {
        "refeature_manifest",
        "raw_manifest",
        "raw_receipt",
        "frozen_schema_manifest",
        "zero_bypass_receipt",
    }
    if set(evidence_rows) != expected_names:
        raise RuleDerivativeTransferError("transfer plan evidence inventory drifted")
    normalized: dict[str, Mapping[str, Any]] = {}
    for name in sorted(expected_names):
        identity = _mapping(evidence_rows[name], field=f"transfer plan evidence {name}")
        if set(identity) != {"path", "sha256", "size_bytes"}:
            raise RuleDerivativeTransferError("transfer plan evidence identity fields drifted")
        path = identity.get("path")
        if not isinstance(path, str) or not path:
            raise RuleDerivativeTransferError("transfer plan evidence path is missing")
        normalized[name] = identity
    evidence = load_transfer_evidence(
        refeature_manifest_path=str(normalized["refeature_manifest"]["path"]),
        raw_manifest_path=str(normalized["raw_manifest"]["path"]),
        raw_receipt_path=str(normalized["raw_receipt"]["path"]),
        frozen_schema_manifest_path=str(normalized["frozen_schema_manifest"]["path"]),
        zero_bypass_receipt_path=str(normalized["zero_bypass_receipt"]["path"]),
    )
    if dict(evidence.file_identities) != {
        name: dict(normalized[name]) for name in sorted(expected_names)
    }:
        raise RuleDerivativeTransferError("transfer plan evidence file identity drifted")
    if row.get("source_manifest_sha256") != evidence.refeature_manifest_sha256 or row.get(
        "source_manifest_file_sha256"
    ) != evidence.file_identities["refeature_manifest"]["sha256"]:
        raise RuleDerivativeTransferError("transfer plan source manifest identity drifted")
    if row.get("validated_30_day_raw_manifest_binding") != evidence.raw_manifest_sha256 or row.get(
        "raw_corpus_receipt_sha256"
    ) != canonical_sha256(evidence.raw_receipt) or row.get("zero_bypass_receipt_sha256") != evidence.zero_bypass_receipt_sha256:
        raise RuleDerivativeTransferError("transfer plan raw/zero-bypass identity drifted")
    return evidence


def validate_transfer_plan(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Validate a dry-run plan independently before a transport sees it."""

    row = _mapping(plan, field="transfer plan")
    if row.get("schema") != R298_TRANSFER_PLAN_SCHEMA or row.get("status") != "planned_no_inzi_bytes_mutated":
        raise RuleDerivativeTransferError("transfer plan is not an inert r298 dry-run plan")
    if (
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("goal_gateway_sha256"),
        row.get("goal_revision"),
    ) != (R298_CONTRACT_PATH, R298_CONTRACT_SHA256, R298_GOAL_SHA256, 4):
        raise RuleDerivativeTransferError("transfer plan goal binding drifted")
    if (
        row.get("destination_host") != INZI_DESTINATION_HOST
        or row.get("destination_directory") != INZI_QUARANTINE_DIRECTORY
    ):
        raise RuleDerivativeTransferError("transfer plan destination is not the quarantined directory")
    if row.get("max_parallel_lanes") != MAX_TRANSFER_LANES:
        raise RuleDerivativeTransferError("transfer plan lane ceiling drifted")
    lanes = row.get("requested_parallel_lanes")
    if isinstance(lanes, bool) or not isinstance(lanes, int) or not 1 <= lanes <= MAX_TRANSFER_LANES:
        raise RuleDerivativeTransferError("transfer plan requested lane count is invalid")
    if row.get("actual_transfer_started") is not False or row.get("remote_io_performed") is not False:
        raise RuleDerivativeTransferError("a plan cannot claim remote side effects")
    if row.get("inzi_loading_or_execution_authority") is not False:
        raise RuleDerivativeTransferError("transfer plan grants forbidden Inzi authority")
    _sha(row.get("source_manifest_sha256"), field="transfer source manifest identity")
    _sha(row.get("source_manifest_file_sha256"), field="transfer source manifest file identity")
    _sha(row.get("validated_30_day_raw_manifest_binding"), field="transfer raw manifest binding")
    binding = _mapping(row.get("schema_binding"), field="transfer schema binding")
    for field in (
        "frozen_schema_manifest_sha256",
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    ):
        _sha(binding.get(field), field=f"transfer schema binding {field}")
    evidence = reopen_plan_evidence(row)
    if dict(binding) != _required_schema_binding(evidence.refeature_manifest):
        raise RuleDerivativeTransferError("transfer plan schema binding differs from reopened source")
    evidence_root = Path(evidence.file_identities["refeature_manifest"]["path"]).parent
    expected_sources = {
        str(source["source_shard_logical_id"]): source
        for source in validate_refeature_manifest_for_transfer(
            evidence_root,
            evidence.refeature_manifest,
            evidence=evidence,
        )
    }
    entries = _rows(row.get("entries"), field="transfer plan entries")
    if not entries:
        raise RuleDerivativeTransferError("transfer plan has no validated source shards")
    names: set[str] = set()
    shas: set[str] = set()
    for raw in entries:
        entry = _mapping(raw, field="transfer plan entry")
        digest = _sha(entry.get("sha256"), field="transfer plan shard SHA-256")
        filename = content_addressed_filename(digest)
        if entry.get("content_addressed_filename") != filename or filename in names:
            raise RuleDerivativeTransferError("transfer plan content-addressed filename drifted")
        if digest in shas:
            raise RuleDerivativeTransferError("transfer plan repeats a byte object")
        if entry.get("destination_directory") != INZI_QUARANTINE_DIRECTORY or entry.get("destination_path") != f"{INZI_QUARANTINE_DIRECTORY}/{filename}":
            raise RuleDerivativeTransferError("transfer plan entry escapes quarantine")
        _positive_int(entry.get("size_bytes"), field="transfer plan source size", maximum=MAX_PHYSICAL_SHARD_BYTES)
        lane = entry.get("parallel_lane_id")
        if isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane < lanes:
            raise RuleDerivativeTransferError("transfer plan lane assignment is invalid")
        if entry.get("source_schema_validation_passed") is not True:
            raise RuleDerivativeTransferError("transfer plan contains an unvalidated source shard")
        logical_id = entry.get("source_shard_logical_id")
        expected_source = expected_sources.get(str(logical_id))
        if expected_source is None or any(
            entry.get(field) != expected_source.get(field)
            for field in (
                "relative_path",
                "sha256",
                "size_bytes",
                "record_count",
                "uncompressed_jsonl_bytes",
                "source_schema_validation_passed",
            )
        ):
            raise RuleDerivativeTransferError(
                "transfer plan shard does not match reopened full-stream source validation"
            )
        names.add(filename)
        shas.add(digest)
    if set(str(_mapping(entry, field="transfer plan entry").get("source_shard_logical_id")) for entry in entries) != set(expected_sources):
        raise RuleDerivativeTransferError("transfer plan omits or adds a reopened source shard")
    return [_mapping(value, field="transfer plan entry") for value in entries]


def write_transfer_plan_create_only(path: Path | str, plan: Mapping[str, Any]) -> str:
    """Persist an inert plan exactly once, returning its content identity."""

    validate_transfer_plan(plan)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(plan) + "\n").encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuleDerivativeTransferError(f"create-only transfer plan already exists: {destination}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(destination)


def transfer_receipt_filename(source_sha256: str) -> str:
    """Return the contract-required create-only local receipt filename."""

    digest = _sha(source_sha256, field="source shard receipt SHA-256")
    return f"sha256-{digest.removeprefix('sha256:')}.transfer-receipt.json"


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuleDerivativeTransferError(f"create-only artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def write_shard_transfer_receipt_create_only(
    receipt_root: Path | str,
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> tuple[Path, str]:
    """Seal a verified per-shard receipt under its source content address."""

    row = _validate_shard_receipt(plan, receipt)
    source = _sha(row.get("source_sha256"), field="per-shard receipt source SHA-256")
    name = transfer_receipt_filename(source)
    path = Path(receipt_root) / name
    return path, _write_json_create_only(path, row)


def write_transfer_parity_receipt_create_only(
    receipt_root: Path | str,
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_receipts: Sequence[Mapping[str, Any]],
) -> tuple[Path, str]:
    """Seal the one final parity receipt exactly once after full validation."""

    row = _mapping(receipt, field="final transfer parity receipt")
    expected = make_transfer_parity_receipt(plan, shard_receipts)
    if canonical_sha256(row) != canonical_sha256(expected):
        raise RuleDerivativeTransferError("final parity receipt does not match validated shard receipts")
    path = Path(receipt_root) / "transfer-parity-receipt.json"
    return path, _write_json_create_only(path, row)


@dataclass(frozen=True)
class RemoteShardObservation:
    """A post-operation remote identity observation, never a decoded shard."""

    exists: bool
    sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class RemoteDirectoryObservation:
    """Remote quarantine directory identity without inspecting any shard data."""

    exists: bool
    is_directory: bool
    is_symlink: bool


class QuarantinedShardTransport(Protocol):
    """Minimal create-only transport contract for the rev4 exception.

    ``publish_private_partial_create_only`` must atomically create *only* the
    final name and return False when that final name already exists.  It must
    not overwrite, delete, rename, or replace a final object.  Private partial
    objects are not receipts and may never be loaded by the destination.
    """

    def inspect_final(self, *, destination_path: str) -> RemoteShardObservation:
        ...

    def ensure_quarantine_directory_create_only(
        self, *, destination_directory: str
    ) -> RemoteDirectoryObservation:
        """Create/check only the exact contract directory, never a symlink."""
        ...

    def upload_private_partial(
        self,
        *,
        source_path: Path,
        destination_directory: str,
        private_partial_name: str,
    ) -> str:
        ...

    def publish_private_partial_create_only(
        self,
        *,
        destination_directory: str,
        private_partial_name: str,
        final_filename: str,
    ) -> bool:
        ...


def _remote_matches(entry: Mapping[str, Any], remote: RemoteShardObservation) -> bool:
    return (
        remote.exists
        and remote.sha256 == entry["sha256"]
        and remote.size_bytes == entry["size_bytes"]
    )


def make_shard_transfer_receipt(
    plan: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    remote: RemoteShardObservation,
    disposition: str,
    started_at_utc: str,
    validated_at_utc: str,
    conflict_refused: bool,
) -> dict[str, Any]:
    """Build one exact per-object receipt from independently verified facts."""

    entries = validate_transfer_plan(plan)
    row = _mapping(entry, field="transfer entry")
    if not any(canonical_sha256(candidate) == canonical_sha256(row) for candidate in entries):
        raise RuleDerivativeTransferError("transfer receipt entry is not in the validated plan")
    if disposition not in {"copied", "resumed", "skipped_exact"}:
        raise RuleDerivativeTransferError("transfer disposition must be copied, resumed, or skipped_exact")
    exact = _remote_matches(row, remote)
    passed = exact and not conflict_refused
    receipt = {
        "schema": R298_TRANSFER_RECEIPT_SCHEMA,
        "version": 1,
        "status": "passed_source_remote_object_parity" if passed else "failed_or_conflict_refused",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_revision": 4,
        "validated_30_day_raw_manifest_sha256": plan[
            "validated_30_day_raw_manifest_binding"
        ],
        "frozen_schema_manifest_sha256": plan["schema_binding"]["frozen_schema_manifest_sha256"],
        "feature_schema_sha256": plan["schema_binding"]["feature_schema_sha256"],
        "target_schema_sha256": plan["schema_binding"]["target_schema_sha256"],
        "checklist_provenance_schema_sha256": plan["schema_binding"]["checklist_provenance_schema_sha256"],
        "source_shard_logical_id": row["source_shard_logical_id"],
        "content_addressed_filename": row["content_addressed_filename"],
        "source_sha256": row["sha256"],
        "source_size_bytes": row["size_bytes"],
        "source_schema_validation_passed": row["source_schema_validation_passed"],
        "destination_host": INZI_DESTINATION_HOST,
        "destination_directory": INZI_QUARANTINE_DIRECTORY,
        "destination_path": row["destination_path"],
        "destination_sha256": remote.sha256,
        "destination_size_bytes": remote.size_bytes,
        "destination_validation_passed": passed,
        "transfer_disposition_copied_resumed_or_skipped_exact": disposition,
        "parallel_lane_id": row["parallel_lane_id"],
        "started_at_utc": started_at_utc,
        "validated_at_utc": validated_at_utc,
        "conflict_refused": conflict_refused,
        "schema_binding": dict(plan["schema_binding"]),
        "validated_30_day_raw_manifest_binding": plan[
            "validated_30_day_raw_manifest_binding"
        ],
        "inzi_loading_or_execution_authority": False,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    return receipt


def _stage_one(
    plan: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    output_root: Path,
    transport: QuarantinedShardTransport,
) -> dict[str, Any]:
    started = _utc_now()
    row = _mapping(entry, field="transfer plan entry")
    initial = transport.inspect_final(destination_path=str(row["destination_path"]))
    if _remote_matches(row, initial):
        return make_shard_transfer_receipt(
            plan,
            row,
            remote=initial,
            disposition="skipped_exact",
            started_at_utc=started,
            validated_at_utc=_utc_now(),
            conflict_refused=False,
        )
    if initial.exists:
        # Never manipulate a conflicting final object.  Preserve its state in
        # the receipt; it will block the final parity receipt.
        return make_shard_transfer_receipt(
            plan,
            row,
            remote=initial,
            disposition="skipped_exact",
            started_at_utc=started,
            validated_at_utc=_utc_now(),
            conflict_refused=True,
        )
    source_path = output_root / _relative_path(row["relative_path"], field="source shard path")
    if source_path.is_symlink() or not source_path.is_file() or sha256_file(source_path) != row["sha256"]:
        raise RuleDerivativeTransferError("source shard changed after transfer plan validation")
    private_partial_name = f".{row['content_addressed_filename']}.partial"
    raw_disposition = transport.upload_private_partial(
        source_path=source_path,
        destination_directory=INZI_QUARANTINE_DIRECTORY,
        private_partial_name=private_partial_name,
    )
    if raw_disposition not in {"copied", "resumed"}:
        raise RuleDerivativeTransferError("transport returned an invalid partial-upload disposition")
    published = transport.publish_private_partial_create_only(
        destination_directory=INZI_QUARANTINE_DIRECTORY,
        private_partial_name=private_partial_name,
        final_filename=str(row["content_addressed_filename"]),
    )
    final = transport.inspect_final(destination_path=str(row["destination_path"]))
    return make_shard_transfer_receipt(
        plan,
        row,
        remote=final,
        disposition=raw_disposition,
        started_at_utc=started,
        validated_at_utc=_utc_now(),
        conflict_refused=not published,
    )


def execute_transfer_plan_create_only(
    plan: Mapping[str, Any],
    *,
    output_root: Path | str,
    transport: QuarantinedShardTransport,
    explicit_revision4_staging_authorization: bool = False,
) -> list[dict[str, Any]]:
    """Execute only the narrowly authorized byte-staging operation.

    It is deliberately impossible to call by accident: callers must supply a
    real injected transport and set ``explicit_revision4_staging_authorization``.
    This function has no relation to training authority; successful receipts
    remain quarantined, Elmo-training-only evidence.
    """

    if explicit_revision4_staging_authorization is not True:
        raise RuleDerivativeTransferError("actual staging requires explicit revision-4 authorization")
    entries = validate_transfer_plan(plan)
    # Re-open all source evidence again immediately before remote I/O.  The
    # plan itself is not authority if a source receipt/schema changed later.
    evidence = reopen_plan_evidence(plan)
    root = Path(output_root).resolve()
    evidence_root = Path(evidence.file_identities["refeature_manifest"]["path"]).parent
    if root != evidence_root:
        raise RuleDerivativeTransferError(
            "actual staging source root differs from the reopened re-featurization manifest root"
        )
    # Validate the full current gzip stream again immediately before remote
    # I/O.  A file that changed after dry-run planning cannot be staged just
    # because its first record or a cached digest looked valid earlier.
    for entry in entries:
        validate_refeatured_shard(root, entry, manifest=evidence.refeature_manifest)
    directory = transport.ensure_quarantine_directory_create_only(
        destination_directory=INZI_QUARANTINE_DIRECTORY
    )
    if not (
        isinstance(directory, RemoteDirectoryObservation)
        and directory.exists
        and directory.is_directory
        and not directory.is_symlink
    ):
        raise RuleDerivativeTransferError(
            "remote contract quarantine directory is absent, non-directory, or symlinked"
        )
    lane_count = int(_mapping(plan, field="transfer plan")["requested_parallel_lanes"])
    receipts: list[dict[str, Any]] = []
    # This bounded I/O pool has no model/tensor workers and is capped by the
    # contract's four lane maximum.  Source validation occurs before any
    # transport call, so a changed local object is never staged.
    with ThreadPoolExecutor(max_workers=lane_count, thread_name_prefix="r298-stage") as pool:
        futures = {
            pool.submit(
                _stage_one,
                plan,
                entry,
                output_root=root,
                transport=transport,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            receipts.append(future.result())
    return sorted(receipts, key=lambda receipt: str(receipt["source_shard_logical_id"]))


def _validate_shard_receipt(plan: Mapping[str, Any], receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    row = _mapping(receipt, field="per-shard transfer receipt")
    if row.get("schema") != R298_TRANSFER_RECEIPT_SCHEMA:
        raise RuleDerivativeTransferError("per-shard transfer receipt schema drifted")
    expected_payload = dict(row)
    claimed_payload = expected_payload.pop("receipt_payload_sha256", None)
    if claimed_payload != canonical_sha256(expected_payload):
        raise RuleDerivativeTransferError("per-shard transfer receipt checksum drifted")
    required = {
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_revision": 4,
        "destination_host": INZI_DESTINATION_HOST,
        "destination_directory": INZI_QUARANTINE_DIRECTORY,
        "inzi_loading_or_execution_authority": False,
        "status": "passed_source_remote_object_parity",
        "destination_validation_passed": True,
        "conflict_refused": False,
    }
    if any(row.get(field) != expected for field, expected in required.items()):
        raise RuleDerivativeTransferError("per-shard transfer receipt did not pass strict staging checks")
    entries = validate_transfer_plan(plan)
    matching = [
        entry
        for entry in entries
        if entry["source_shard_logical_id"] == row.get("source_shard_logical_id")
    ]
    if len(matching) != 1:
        raise RuleDerivativeTransferError("per-shard receipt source logical identity is ambiguous")
    expected = matching[0]
    for field, plan_field in (
        ("content_addressed_filename", "content_addressed_filename"),
        ("source_sha256", "sha256"),
        ("source_size_bytes", "size_bytes"),
        ("destination_path", "destination_path"),
        ("parallel_lane_id", "parallel_lane_id"),
    ):
        if row.get(field) != expected[plan_field]:
            raise RuleDerivativeTransferError(f"per-shard receipt {field} disagrees with plan")
    if row.get("destination_sha256") != expected["sha256"] or row.get("destination_size_bytes") != expected["size_bytes"]:
        raise RuleDerivativeTransferError("per-shard receipt remote object does not equal source")
    if row.get("source_schema_validation_passed") is not True:
        raise RuleDerivativeTransferError("per-shard receipt lacks local schema validation")
    if row.get("validated_30_day_raw_manifest_sha256") != plan[
        "validated_30_day_raw_manifest_binding"
    ]:
        raise RuleDerivativeTransferError("per-shard receipt raw manifest binding drifted")
    if row.get("schema_binding") != plan["schema_binding"] or row.get("validated_30_day_raw_manifest_binding") != plan["validated_30_day_raw_manifest_binding"]:
        raise RuleDerivativeTransferError("per-shard receipt schema/raw bindings drifted")
    return row


def make_transfer_parity_receipt(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Prove one complete source/remote parity set; reject partial staging."""

    entries = validate_transfer_plan(plan)
    checked = [_validate_shard_receipt(plan, receipt) for receipt in receipts]
    if len(checked) != len(entries):
        raise RuleDerivativeTransferError("final parity requires one passed receipt for every source shard")
    by_logical = {str(row["source_shard_logical_id"]): row for row in checked}
    if len(by_logical) != len(checked) or set(by_logical) != {
        str(entry["source_shard_logical_id"]) for entry in entries
    }:
        raise RuleDerivativeTransferError("final parity receipt set is incomplete or duplicated")
    ordered = [by_logical[str(entry["source_shard_logical_id"])] for entry in entries]
    filenames = [str(row["content_addressed_filename"]) for row in ordered]
    source_pairs = [
        {"filename": row["content_addressed_filename"], "sha256": row["source_sha256"], "size_bytes": row["source_size_bytes"]}
        for row in ordered
    ]
    destination_pairs = [
        {"filename": row["content_addressed_filename"], "sha256": row["destination_sha256"], "size_bytes": row["destination_size_bytes"]}
        for row in ordered
    ]
    if source_pairs != destination_pairs:
        raise RuleDerivativeTransferError("source/remote object identities are not equal")
    dispositions = [str(row["transfer_disposition_copied_resumed_or_skipped_exact"]) for row in ordered]
    receipt = {
        "schema": R298_TRANSFER_PARITY_RECEIPT_SCHEMA,
        "version": 1,
        "status": "passed_complete_source_remote_parity",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_revision": 4,
        "destination_directory": INZI_QUARANTINE_DIRECTORY,
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "destination_manifest_sha256": canonical_sha256(destination_pairs),
        "source_and_destination_object_count": len(ordered),
        "content_addressed_filename_set": filenames,
        "per_object_source_sha256_and_size": source_pairs,
        "per_object_destination_sha256_and_size": destination_pairs,
        "schema_binding": dict(plan["schema_binding"]),
        "validated_30_day_raw_manifest_binding": plan[
            "validated_30_day_raw_manifest_binding"
        ],
        "ordered_per_shard_receipt_sha256s": [canonical_sha256(row) for row in ordered],
        "skip_resume_and_conflict_counts": {
            "copied": dispositions.count("copied"),
            "resumed": dispositions.count("resumed"),
            "skipped_exact": dispositions.count("skipped_exact"),
            "conflict_refused": 0,
        },
        "parallel_lane_maximum_observed": max(
            int(row["parallel_lane_id"]) + 1 for row in ordered
        ),
        "source_remote_parity_passed": True,
        "inzi_loading_or_execution_authority": False,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    return receipt


__all__ = [
    "INZI_DESTINATION_HOST",
    "INZI_QUARANTINE_DIRECTORY",
    "MAX_TRANSFER_LANES",
    "R298_TRANSFER_PARITY_RECEIPT_SCHEMA",
    "R298_TRANSFER_PLAN_SCHEMA",
    "R298_TRANSFER_RECEIPT_SCHEMA",
    "QuarantinedShardTransport",
    "RemoteDirectoryObservation",
    "RemoteShardObservation",
    "RuleDerivativeTransferError",
    "TransferEvidence",
    "build_transfer_plan",
    "canonical_sha256",
    "content_addressed_filename",
    "execute_transfer_plan_create_only",
    "make_shard_transfer_receipt",
    "make_transfer_parity_receipt",
    "load_transfer_evidence",
    "reopen_plan_evidence",
    "sha256_file",
    "transfer_receipt_filename",
    "validate_refeature_manifest_for_transfer",
    "validate_refeatured_shard",
    "validate_transfer_plan",
    "write_shard_transfer_receipt_create_only",
    "write_transfer_parity_receipt_create_only",
    "write_transfer_plan_create_only",
]
