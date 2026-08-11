"""Create-only r260 OwnDeckLedger sidecar evidence materialization.

This module is deliberately outside the trainer, launcher, service, and
runtime-registry paths.  It turns the already immutable r259 daily side store
into a deterministic, independently auditable joined stream and then supports
an exact create-only copy to an Inzi-local sidecar root.  It never starts a
service, fetches corpus data, or makes an artifact training-eligible by itself.

The public data plane is intentionally narrow:

* each of the exact twenty daily ``meta.json`` and gzip shards is rehashed;
* every row is read through the r259 immutable reader and indexed by the
  canonical four-field key;
* joined rows are sorted by that key and compressed deterministically; and
* remote parity is a bounded semantic audit in addition to byte identity.

The final aggregate and Inzi bindings use the ABI validated by
``poke_bot.r241_own_deck_successor``.  They are only written after the full
twenty-day proof and a local/remote parity receipt exist.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

from . import own_deck_rollout_store as rollout_store


R260_SIDECAR_MATERIALIZATION_SCHEMA = (
    "poke_bot.r260_own_deck_sidecar_materialization/v1"
)
R260_JOINED_ROW_SCHEMA = "poke_bot.r260_own_deck_joined_row/v1"
R260_JOINED_DATASET_MANIFEST_SCHEMA = (
    "poke_bot.r260_own_deck_joined_dataset_manifest/v1"
)
R260_JOIN_RECEIPT_SCHEMA = "poke_bot.r260_own_deck_sidecar_join_receipt/v1"
R260_SCHEMA_RECEIPT_SCHEMA = "poke_bot.r260_own_deck_sidecar_schema_receipt/v1"
R260_COUNT_RECEIPT_SCHEMA = "poke_bot.r260_own_deck_sidecar_count_receipt/v1"
R260_DIGEST_RECEIPT_SCHEMA = "poke_bot.r260_own_deck_sidecar_digest_receipt/v1"
R260_COMPLETION_RECEIPT_SCHEMA = (
    "poke_bot.r260_own_deck_sidecar_completion_receipt/v1"
)
R260_PARITY_RECEIPT_SCHEMA = (
    "poke_bot.r260_own_deck_causal_local_remote_parity_receipt/v1"
)
R260_TRANSPORT_RECEIPT_SCHEMA = "poke_bot.r260_own_deck_inzi_transport_receipt/v1"

R260_AGGREGATE_BINDING_SCHEMA = "poke_bot.r241_own_deck_sidecar_binding/v1"
R260_INZI_BINDING_SCHEMA = "poke_bot.r241_own_deck_inzi_dataset_binding/v1"

EVIDENCE_DIRECTORY_NAME = "r260-sidecar-evidence-v1"
SOURCE_MANIFEST_COPY_NAME = "source-manifest.json"
SOURCE_WINDOW_RECEIPT_COPY_NAME = "source-window-receipt.json"
JOINED_DIRECTORY_NAME = "joined"
JOINED_DATASET_NAME = "r260-own-deck-joined.jsonl.gz"
JOINED_MANIFEST_NAME = "r260-own-deck-joined.manifest.json"
RECEIPTS_DIRECTORY_NAME = "receipts"
JOIN_RECEIPT_NAME = "join.json"
SCHEMA_RECEIPT_NAME = "schema.json"
COUNT_RECEIPT_NAME = "count.json"
DIGEST_RECEIPT_NAME = "digest.json"
PARITY_RECEIPT_NAME = "causal-local-remote-parity.json"
COMPLETION_RECEIPT_NAME = "completion.json"
AGGREGATE_BINDING_NAME = "r260-sidecar-binding.json"
TRANSPORT_RECEIPT_NAME = "transport.json"
INZI_BINDING_NAME = "r260-inzi-dataset-binding.json"

RECORD_KEY_FIELDS = (
    "episode_id",
    "seat",
    "env_step",
    "observation_fingerprint",
)
MAX_CAUSAL_PARITY_SAMPLES = 4_096
DEFAULT_CAUSAL_PARITY_SAMPLES = 256


class R260SidecarMaterializationError(RuntimeError):
    """A source, artifact, or evidence invariant is unsafe or incomplete."""


class _OwnerContract(Protocol):
    sha256: str
    source_manifest_sha256: str
    source_window_receipt_sha256: str
    side_store_root: str
    inzi_training_root: str
    inzi_prefix_staging_root: str


@dataclass(frozen=True)
class FileIdentity:
    """Raw byte identity for one regular, non-symlink file."""

    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class DailyArtifact:
    """The two immutable files and row count for one validated source day."""

    day: str
    meta: FileIdentity
    shard: FileIdentity
    row_count: int


@dataclass(frozen=True)
class SidecarAudit:
    """A complete rehash of the exact twenty-day r259 side store."""

    sidecar_root: Path
    source_manifest: FileIdentity
    source_window_receipt: FileIdentity
    daily: tuple[DailyArtifact, ...]
    daily_build_identity: Mapping[str, str]
    source_archive_bytes: int
    validated_episode_count: int

    @property
    def daily_meta_identities(self) -> dict[str, dict[str, object]]:
        return {row.day: row.meta.as_dict() for row in self.daily}

    @property
    def daily_shard_identities(self) -> dict[str, dict[str, object]]:
        return {row.day: row.shard.as_dict() for row in self.daily}


@dataclass(frozen=True)
class MaterializationResult:
    """The local Elmo-or-fixture materialization artifacts, all create-only."""

    audit: SidecarAudit
    evidence_root: Path
    joined_dataset: FileIdentity
    joined_manifest: FileIdentity
    join_receipt: FileIdentity
    schema_receipt: FileIdentity
    count_receipt: FileIdentity
    digest_receipt: FileIdentity
    row_count: int
    joined_rows_sha256: str
    joined_keys_sha256: str


@dataclass(frozen=True)
class TransportResult:
    """The byte-identical Inzi copy before completion is finalized."""

    source: MaterializationResult
    inzi_sidecar_root: Path
    inzi_evidence_root: Path
    joined_dataset: FileIdentity
    joined_manifest: FileIdentity
    transport_receipt: FileIdentity


@dataclass(frozen=True)
class PrefixTransferResult:
    """Append-only, explicitly nonterminal Inzi staging-prefix transfer."""

    source_audit: SidecarAudit
    inzi_staging_root: Path
    committed_days: tuple[str, ...]
    copied_days: tuple[str, ...]
    prefix_receipt: FileIdentity


@dataclass(frozen=True)
class CompletionResult:
    """The full Inzi-local aggregate evidence accepted by the r260 validator."""

    audit: SidecarAudit
    inzi_sidecar_root: Path
    evidence_root: Path
    aggregate_binding: FileIdentity
    completion_receipt: FileIdentity
    parity_receipt: FileIdentity
    joined_dataset: FileIdentity


def expected_days() -> tuple[str, ...]:
    """Return the fixed, gap-free r241 expert window in canonical order."""

    start = date(2026, 7, 22)
    return tuple((start + timedelta(days=offset)).isoformat() for offset in range(20))


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    """Encode canonical JSON and reject NaN/non-serializable evidence."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise R260SidecarMaterializationError("evidence is not canonical JSON") from exc
    return (text + ("\n" if newline else "")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a typed SHA-256 identity for raw bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def semantic_digest(value: Mapping[str, Any], *, field: str) -> str:
    """Hash a receipt/manifest after removing its declared self-digest."""

    detached = dict(value)
    detached.pop(field, None)
    return sha256_bytes(canonical_json_bytes(detached))


def file_identity(path: Path | str, *, label: str) -> FileIdentity:
    """Rehash one regular file; never follow a symlink supplied as evidence."""

    source = _regular_file(path, label=label)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return FileIdentity(
        path=source,
        sha256="sha256:" + digest.hexdigest(),
        size_bytes=int(source.stat().st_size),
    )


def materialize_r260_sidecar(
    *,
    sidecar_root: Path | str,
    source_manifest: Path | str,
    source_window_receipt: Path | str,
    evidence_root: Path | str,
    owner_contract: _OwnerContract | None = None,
    expected_sidecar_root: Path | str | None = None,
    receipt_identity_root: Path | str | None = None,
) -> MaterializationResult:
    """Create a deterministic joined dataset after a strict 20/20 source audit.

    ``evidence_root`` must not already exist.  It receives only new files;
    source daily files and their directory modes are never changed.
    """

    contract = _owner_contract(owner_contract)
    audit = audit_r260_sidecar(
        sidecar_root=sidecar_root,
        source_manifest=source_manifest,
        source_window_receipt=source_window_receipt,
        owner_contract=contract,
        expected_sidecar_root=expected_sidecar_root,
    )
    physical_root = audit.sidecar_root
    root = _new_directory(evidence_root, label="r260 evidence root")
    _require_descendant(root, physical_root, label="r260 evidence root")
    logical_root = _receipt_identity_root(
        physical_root=physical_root,
        receipt_identity_root=receipt_identity_root,
    )
    if receipt_identity_root is not None:
        _require_staging_inzi_root(physical_root, contract)
        _require_final_inzi_root(logical_root, contract)
    joined_dir = _new_directory(root / JOINED_DIRECTORY_NAME, label="r260 joined directory")
    receipts_dir = _new_directory(root / RECEIPTS_DIRECTORY_NAME, label="r260 receipt directory")

    # Keep source receipt bytes available to a later remote re-audit without
    # asking an Inzi process to dereference an Elmo archive path.
    source_manifest_copy = _copy_regular_create_only(
        audit.source_manifest.path,
        root / SOURCE_MANIFEST_COPY_NAME,
        label="r260 source manifest copy",
    )
    source_window_copy = _copy_regular_create_only(
        audit.source_window_receipt.path,
        root / SOURCE_WINDOW_RECEIPT_COPY_NAME,
        label="r260 source-window receipt copy",
    )

    joined_path = joined_dir / JOINED_DATASET_NAME
    joined_dataset, row_count, rows_sha, keys_sha = _write_deterministic_joined_dataset(
        audit,
        joined_path,
    )
    joined_manifest_payload = _seal(
        {
            "schema": R260_JOINED_DATASET_MANIFEST_SCHEMA,
            "status": "complete",
            "owner_contract_sha256": contract.sha256,
            "source_manifest": _project_identity(
                source_manifest_copy, physical_root=physical_root, logical_root=logical_root
            ).as_dict(),
            "source_window_receipt": _project_identity(
                source_window_copy, physical_root=physical_root, logical_root=logical_root
            ).as_dict(),
            "record_key": list(RECORD_KEY_FIELDS),
            "daily_sidecar_meta_files": _project_daily_identities(
                audit.daily_meta_identities, physical_root=physical_root, logical_root=logical_root
            ),
            "daily_sidecar_shard_files": _project_daily_identities(
                audit.daily_shard_identities, physical_root=physical_root, logical_root=logical_root
            ),
            "joined_dataset": _project_identity(
                joined_dataset, physical_root=physical_root, logical_root=logical_root
            ).as_dict(),
            "joined_row_count": row_count,
            "joined_rows_sha256": rows_sha,
            "joined_keys_sha256": keys_sha,
            "daily_build_identity": dict(audit.daily_build_identity),
            "active_r241_training_eligible": False,
        },
        field="manifest_sha256",
    )
    joined_manifest = _write_json_create_only(
        joined_dir / JOINED_MANIFEST_NAME,
        joined_manifest_payload,
        label="r260 joined manifest",
    )

    common = {
        "owner_contract_sha256": contract.sha256,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "record_key": list(RECORD_KEY_FIELDS),
        "daily_sidecar_meta_files": _project_daily_identities(
            audit.daily_meta_identities, physical_root=physical_root, logical_root=logical_root
        ),
        "daily_sidecar_shard_files": _project_daily_identities(
            audit.daily_shard_identities, physical_root=physical_root, logical_root=logical_root
        ),
        "joined_dataset": _project_identity(
            joined_dataset, physical_root=physical_root, logical_root=logical_root
        ).as_dict(),
        "joined_manifest": _project_identity(
            joined_manifest, physical_root=physical_root, logical_root=logical_root
        ).as_dict(),
        "joined_row_count": row_count,
        "joined_rows_sha256": rows_sha,
        "joined_keys_sha256": keys_sha,
        "active_r241_training_eligible": False,
    }
    join_receipt = _write_json_create_only(
        receipts_dir / JOIN_RECEIPT_NAME,
        _seal(
            {
                "schema": R260_JOIN_RECEIPT_SCHEMA,
                "status": "passed",
                **common,
                "one_to_one_key_coverage": True,
                "duplicate_key_count": 0,
                "unmatched_record_count": 0,
            },
            field="receipt_sha256",
        ),
        label="r260 join receipt",
    )
    schema_receipt = _write_json_create_only(
        receipts_dir / SCHEMA_RECEIPT_NAME,
        _seal(
            {
                "schema": R260_SCHEMA_RECEIPT_SCHEMA,
                "status": "passed",
                **common,
                "joined_row_schema": R260_JOINED_ROW_SCHEMA,
                "joined_row_version": 1,
                "sidecar_row_schema": rollout_store.OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
                "sidecar_row_version": rollout_store.OWN_DECK_ROLLOUT_SIDECAR_VERSION,
                "required_joined_fields": [
                    "schema",
                    "version",
                    "key",
                    "source_date",
                    "source_manifest_sha256",
                    "daily_meta_sha256",
                    "sidecar_row",
                ],
            },
            field="receipt_sha256",
        ),
        label="r260 schema receipt",
    )
    daily_row_counts = {row.day: row.row_count for row in audit.daily}
    count_receipt = _write_json_create_only(
        receipts_dir / COUNT_RECEIPT_NAME,
        _seal(
            {
                "schema": R260_COUNT_RECEIPT_SCHEMA,
                "status": "passed",
                **common,
                "day_count": len(audit.daily),
                "expected_dates": list(expected_days()),
                "daily_joined_row_counts": daily_row_counts,
                "validated_episode_count": audit.validated_episode_count,
                "source_archive_bytes": audit.source_archive_bytes,
            },
            field="receipt_sha256",
        ),
        label="r260 count receipt",
    )
    digest_receipt = _write_json_create_only(
        receipts_dir / DIGEST_RECEIPT_NAME,
        _seal(
            {
                "schema": R260_DIGEST_RECEIPT_SCHEMA,
                "status": "passed",
                **common,
                "daily_build_identity": dict(audit.daily_build_identity),
                "source_manifest_copy": _project_identity(
                    source_manifest_copy, physical_root=physical_root, logical_root=logical_root
                ).as_dict(),
                "source_window_receipt_copy": _project_identity(
                    source_window_copy, physical_root=physical_root, logical_root=logical_root
                ).as_dict(),
                "all_daily_artifacts_rehashed": True,
            },
            field="receipt_sha256",
        ),
        label="r260 digest receipt",
    )
    for directory in (joined_dir, receipts_dir):
        _seal_directory(directory)
    # The evidence root intentionally remains writable until Inzi has added its
    # create-only parity/completion/binding receipts.  Existing files are 0444.
    return MaterializationResult(
        audit=audit,
        evidence_root=root,
        joined_dataset=joined_dataset,
        joined_manifest=joined_manifest,
        join_receipt=join_receipt,
        schema_receipt=schema_receipt,
        count_receipt=count_receipt,
        digest_receipt=digest_receipt,
        row_count=row_count,
        joined_rows_sha256=rows_sha,
        joined_keys_sha256=keys_sha,
    )


def audit_r260_sidecar(
    *,
    sidecar_root: Path | str,
    source_manifest: Path | str,
    source_window_receipt: Path | str,
    owner_contract: _OwnerContract | None = None,
    expected_sidecar_root: Path | str | None = None,
) -> SidecarAudit:
    """Rehash and validate exactly the r259 20-day immutable input layout."""

    return _audit_r260_sidecar(
        sidecar_root=sidecar_root,
        source_manifest=source_manifest,
        source_window_receipt=source_window_receipt,
        owner_contract=owner_contract,
        expected_sidecar_root=expected_sidecar_root,
        require_complete=True,
    )


def audit_r260_sidecar_prefix(
    *,
    sidecar_root: Path | str,
    source_manifest: Path | str,
    source_window_receipt: Path | str,
    owner_contract: _OwnerContract | None = None,
    expected_sidecar_root: Path | str | None = None,
) -> SidecarAudit:
    """Rehash a gap-free committed prefix without granting final authority.

    This is the only audit used by revision-262 prefix transfer.  It accepts
    one through twenty consecutive non-dot daily directories beginning on
    July 22, but its result cannot feed materialization, completion, binding,
    training, or final promotion by itself.
    """

    return _audit_r260_sidecar(
        sidecar_root=sidecar_root,
        source_manifest=source_manifest,
        source_window_receipt=source_window_receipt,
        owner_contract=owner_contract,
        expected_sidecar_root=expected_sidecar_root,
        require_complete=False,
    )


def _audit_r260_sidecar(
    *,
    sidecar_root: Path | str,
    source_manifest: Path | str,
    source_window_receipt: Path | str,
    owner_contract: _OwnerContract | None,
    expected_sidecar_root: Path | str | None,
    require_complete: bool,
) -> SidecarAudit:
    """Shared full/prefix implementation; full callers require all 20 days."""

    contract = _owner_contract(owner_contract)
    root = _regular_directory(sidecar_root, label="r260 sidecar root")
    if expected_sidecar_root is not None:
        expected = _regular_directory(
            expected_sidecar_root, label="expected r260 sidecar root"
        )
        if root != expected:
            raise R260SidecarMaterializationError(
                "r260 sidecar root is not the receipt-bound source root"
            )
    manifest_identity = file_identity(source_manifest, label="r260 source manifest")
    if manifest_identity.sha256 != contract.source_manifest_sha256:
        raise R260SidecarMaterializationError("r260 source manifest digest drifted")
    window_identity = file_identity(
        source_window_receipt, label="r260 source-window receipt"
    )
    if window_identity.sha256 != contract.source_window_receipt_sha256:
        raise R260SidecarMaterializationError(
            "r260 source-window receipt digest drifted"
        )
    manifest = _read_json_object(manifest_identity.path, label="r260 source manifest")
    archives = _validate_source_manifest(manifest, contract=contract)
    archive_by_day = {str(row["date"]): row for row in archives}

    daily_root = _regular_directory(
        root / rollout_store.DAILY_DIRECTORY_NAME, label="r260 daily sidecar root"
    )
    non_dot_children = [child for child in daily_root.iterdir() if not child.name.startswith(".")]
    if any(child.is_symlink() or not child.is_dir() for child in non_dot_children):
        raise R260SidecarMaterializationError("r260 daily root has an unsafe non-dot member")
    actual_days = tuple(sorted(child.name for child in non_dot_children))
    if not actual_days or actual_days != expected_days()[: len(actual_days)]:
        raise R260SidecarMaterializationError(
            "r260 sidecar daily layout is not a gap-free canonical prefix"
        )
    if require_complete and actual_days != expected_days():
        raise R260SidecarMaterializationError(
            "r260 sidecar is not the exact complete twenty-day layout"
        )
    daily: list[DailyArtifact] = []
    build_identity: dict[str, str] | None = None
    for day in actual_days:
        directory = _regular_directory(daily_root / day, label=f"r260 daily {day}")
        members = {item.name for item in directory.iterdir()}
        required = {rollout_store.DAILY_META_NAME, rollout_store.DAILY_SHARD_NAME}
        if members != required:
            raise R260SidecarMaterializationError(
                f"r260 daily {day} has an incomplete or unexpected member set"
            )
        meta_path = _regular_file(
            directory / rollout_store.DAILY_META_NAME, label=f"r260 daily {day} metadata"
        )
        shard_path = _regular_file(
            directory / rollout_store.DAILY_SHARD_NAME, label=f"r260 daily {day} shard"
        )
        meta_identity = file_identity(meta_path, label=f"r260 daily {day} metadata")
        shard_identity = file_identity(shard_path, label=f"r260 daily {day} shard")
        try:
            meta = rollout_store.read_daily_meta(root, day)
        except rollout_store.OwnDeckRolloutStoreError as exc:
            raise R260SidecarMaterializationError(
                f"r260 daily {day} immutable metadata validation failed"
            ) from exc
        _validate_daily_meta(
            meta,
            day=day,
            archive=archive_by_day[day],
            contract=contract,
            observed_shard=shard_identity,
        )
        projected = _daily_build_identity(meta, label=f"r260 daily {day} build")
        if build_identity is None:
            build_identity = projected
        elif projected != build_identity:
            raise R260SidecarMaterializationError(
                "r260 daily sidecars do not share one exact build identity"
            )
        row_count = _strict_int(meta.get("row_count"), label=f"r260 daily {day} row_count")
        daily.append(DailyArtifact(day, meta_identity, shard_identity, row_count))
    if build_identity is None:
        raise R260SidecarMaterializationError("r260 sidecar has no daily build identity")
    return SidecarAudit(
        sidecar_root=root,
        source_manifest=manifest_identity,
        source_window_receipt=window_identity,
        daily=tuple(daily),
        daily_build_identity=build_identity,
        source_archive_bytes=sum(_strict_int(row["bytes"], label="archive bytes") for row in archives),
        validated_episode_count=sum(
            _strict_int(row["validated_episode_count"], label="archive episode count")
            for row in archives
        ),
    )


def attest_r260_existing_inzi_transport(
    *,
    elmo_sidecar_root: Path | str,
    elmo_evidence_root: Path | str,
    inzi_sidecar_root: Path | str,
    inzi_evidence_root: Path | str,
    expected_elmo_sidecar_root: Path | str,
    owner_contract: _OwnerContract | None = None,
    receipt_identity_root: Path | str | None = None,
) -> FileIdentity:
    """Receipt an already copied full Inzi tree without copying it again.

    This is the revision-262 completion path: daily directories may have
    arrived append-only in the noneligible staging root, then the deterministic
    join is created there.  The function rehashes both full trees and accepts
    only exact byte-identical transport before writing a create-only receipt.
    ``receipt_identity_root`` precommits the exact final sibling paths that
    become real only after the atomic promotion.
    """

    contract = _owner_contract(owner_contract)
    expected_elmo = _regular_directory(
        expected_elmo_sidecar_root, label="expected Elmo sidecar root"
    )
    local_root = _regular_directory(elmo_sidecar_root, label="Elmo sidecar root")
    if local_root != expected_elmo:
        raise R260SidecarMaterializationError("r260 transport local root is not Elmo")
    local_evidence = _regular_directory(elmo_evidence_root, label="Elmo evidence root")
    _require_descendant(local_evidence, local_root, label="Elmo evidence root")
    remote_text = Path(inzi_sidecar_root).expanduser()
    _reject_elmo_destination(remote_text, expected_elmo)
    remote_root = _regular_directory(remote_text, label="Inzi sidecar root")
    configured_final = _configured_inzi_root(
        contract, "inzi_training_root", label="r260 configured Inzi final root"
    )
    configured_staging = _configured_inzi_root(
        contract, "inzi_prefix_staging_root", label="r260 configured Inzi staging root"
    )
    if remote_root == configured_staging:
        if receipt_identity_root is None or Path(receipt_identity_root).expanduser().resolve() != configured_final:
            raise R260SidecarMaterializationError(
                "staging transport receipts must precommit the configured final Inzi root"
            )
    elif remote_root != configured_final:
        raise R260SidecarMaterializationError(
            "r260 transport receipt root is neither configured staging nor final Inzi root"
        )
    remote_evidence = _regular_directory(inzi_evidence_root, label="Inzi evidence root")
    _require_descendant(remote_evidence, remote_root, label="Inzi evidence root")
    local = _materialization_from_existing(
        sidecar_root=local_root,
        evidence_root=local_evidence,
        owner_contract=contract,
    )
    remote = _materialization_from_existing(
        sidecar_root=remote_root,
        evidence_root=remote_evidence,
        owner_contract=contract,
    )
    return _write_transport_receipt(
        source=local,
        remote=remote,
        inzi_evidence_root=remote_evidence,
        owner_contract=contract,
        receipt_identity_root=receipt_identity_root,
    )


def transport_r260_sidecar_to_inzi(
    *,
    source_sidecar_root: Path | str,
    source_evidence_root: Path | str,
    inzi_sidecar_root: Path | str,
    expected_elmo_sidecar_root: Path | str,
    owner_contract: _OwnerContract | None = None,
) -> TransportResult:
    """Copy the complete sidecar and initial evidence byte-for-byte to Inzi.

    The destination root is create-only and must not be the Elmo source root
    (or any child of it).  This is a local filesystem producer; callers that
    cross a network boundary must mount/stage the source first and still get
    the exact same raw-byte checks on both sides.
    """

    contract = _owner_contract(owner_contract)
    elmo_root = _expected_elmo_sidecar_path(
        expected_elmo_sidecar_root, contract, require_directory=True
    )
    source_root = _regular_directory(source_sidecar_root, label="Elmo sidecar root")
    if source_root != elmo_root:
        raise R260SidecarMaterializationError(
            "r260 transport source is not the declared Elmo sidecar root"
        )
    source_evidence = _regular_directory(
        source_evidence_root, label="Elmo r260 sidecar evidence root"
    )
    _require_descendant(source_evidence, source_root, label="Elmo evidence root")
    source = _materialization_from_existing(
        sidecar_root=source_root,
        evidence_root=source_evidence,
        owner_contract=contract,
    )
    destination = Path(inzi_sidecar_root).expanduser()
    _reject_elmo_destination(destination, elmo_root)
    if destination.resolve() != _configured_inzi_root(
        contract, "inzi_training_root", label="r260 configured Inzi final root"
    ):
        raise R260SidecarMaterializationError(
            "r260 transport destination is not the configured final Inzi root"
        )
    destination_root = _new_directory(destination, label="Inzi sidecar root")
    inzi_daily_root = _new_directory(
        destination_root / rollout_store.DAILY_DIRECTORY_NAME,
        label="Inzi daily sidecar root",
    )
    for daily in source.audit.daily:
        source_day_dir = source_root / rollout_store.DAILY_DIRECTORY_NAME / daily.day
        destination_day_dir = _new_directory(
            inzi_daily_root / daily.day, label=f"Inzi daily {daily.day}"
        )
        copied_shard = _copy_regular_create_only(
            source_day_dir / rollout_store.DAILY_SHARD_NAME,
            destination_day_dir / rollout_store.DAILY_SHARD_NAME,
            label=f"Inzi daily {daily.day} shard",
        )
        copied_meta = _copy_regular_create_only(
            source_day_dir / rollout_store.DAILY_META_NAME,
            destination_day_dir / rollout_store.DAILY_META_NAME,
            label=f"Inzi daily {daily.day} metadata",
        )
        if copied_shard.sha256 != daily.shard.sha256 or copied_meta.sha256 != daily.meta.sha256:
            raise R260SidecarMaterializationError("r260 daily transport byte identity drifted")
        _seal_directory(destination_day_dir)
    _seal_directory(inzi_daily_root)

    inzi_evidence = _new_directory(
        destination_root / EVIDENCE_DIRECTORY_NAME, label="Inzi r260 evidence root"
    )
    _copy_evidence_members(source.evidence_root, inzi_evidence)
    remote = _materialization_from_existing(
        sidecar_root=destination_root,
        evidence_root=inzi_evidence,
        owner_contract=contract,
    )
    transport_receipt = _write_transport_receipt(
        source=source,
        remote=remote,
        inzi_evidence_root=inzi_evidence,
        owner_contract=contract,
        receipt_identity_root=None,
    )
    return TransportResult(
        source=source,
        inzi_sidecar_root=destination_root,
        inzi_evidence_root=inzi_evidence,
        joined_dataset=remote.joined_dataset,
        joined_manifest=remote.joined_manifest,
        transport_receipt=transport_receipt,
    )


def stage_r260_sidecar_prefix_to_inzi(
    *,
    source_sidecar_root: Path | str,
    source_manifest: Path | str,
    source_window_receipt: Path | str,
    inzi_staging_root: Path | str,
    expected_elmo_sidecar_root: Path | str,
    owner_contract: _OwnerContract | None = None,
) -> PrefixTransferResult:
    """Append-only copy of committed daily shards into a nonterminal Inzi root.

    Revision 262 authorizes this transfer before all days are materialized, but
    never lets the staging root become a trainer path.  Each invocation accepts
    only a contiguous non-dot source prefix and either verifies an already
    copied immutable day byte-for-byte or creates the next missing day.  It
    cannot replace, delete, reorder, or make any prefix artifact eligible.
    """

    contract = _owner_contract(owner_contract)
    elmo_root = _expected_elmo_sidecar_path(
        expected_elmo_sidecar_root, contract, require_directory=True
    )
    source_root = _regular_directory(source_sidecar_root, label="Elmo sidecar root")
    if source_root != elmo_root:
        raise R260SidecarMaterializationError(
            "r260 prefix source is not the declared Elmo sidecar root"
        )
    source = audit_r260_sidecar_prefix(
        sidecar_root=source_root,
        source_manifest=source_manifest,
        source_window_receipt=source_window_receipt,
        owner_contract=contract,
        expected_sidecar_root=elmo_root,
    )
    destination_text = Path(inzi_staging_root).expanduser()
    _reject_elmo_destination(destination_text, elmo_root)
    if destination_text.resolve() != _configured_inzi_root(
        contract, "inzi_prefix_staging_root", label="r260 configured Inzi staging root"
    ):
        raise R260SidecarMaterializationError(
            "r260 prefix transfer destination is not the configured Inzi staging root"
        )
    if destination_text.exists() or destination_text.is_symlink():
        destination = _regular_directory(destination_text, label="Inzi sidecar staging root")
    else:
        destination = _new_directory(destination_text, label="Inzi sidecar staging root")
    _require_staging_inzi_root(destination, contract)
    daily_root = destination / rollout_store.DAILY_DIRECTORY_NAME
    if daily_root.exists() or daily_root.is_symlink():
        target_daily = _regular_directory(daily_root, label="Inzi staging daily root")
    else:
        target_daily = _new_directory(daily_root, label="Inzi staging daily root")
    existing_members = [child for child in target_daily.iterdir() if not child.name.startswith(".")]
    if any(child.is_symlink() or not child.is_dir() for child in existing_members):
        raise R260SidecarMaterializationError("Inzi staging daily root has an unsafe non-dot member")
    existing_days = tuple(sorted(child.name for child in existing_members))
    if existing_days != expected_days()[: len(existing_days)]:
        raise R260SidecarMaterializationError("Inzi staging days are not a canonical prefix")
    source_days = tuple(row.day for row in source.daily)
    if len(existing_days) > len(source_days) or existing_days != source_days[: len(existing_days)]:
        raise R260SidecarMaterializationError(
            "Inzi staging prefix is not a prefix of the committed Elmo source"
        )
    copied: list[str] = []
    source_by_day = {row.day: row for row in source.daily}
    for day in source_days:
        expected = source_by_day[day]
        destination_day = target_daily / day
        if destination_day.exists() or destination_day.is_symlink():
            _verify_staged_day(destination_day, expected)
            continue
        source_day = source_root / rollout_store.DAILY_DIRECTORY_NAME / day
        new_day = _new_directory(destination_day, label=f"Inzi staged day {day}")
        copied_meta = _copy_regular_create_only(
            source_day / rollout_store.DAILY_META_NAME,
            new_day / rollout_store.DAILY_META_NAME,
            label=f"Inzi staged {day} metadata",
        )
        copied_shard = _copy_regular_create_only(
            source_day / rollout_store.DAILY_SHARD_NAME,
            new_day / rollout_store.DAILY_SHARD_NAME,
            label=f"Inzi staged {day} shard",
        )
        if copied_meta.sha256 != expected.meta.sha256 or copied_shard.sha256 != expected.shard.sha256:
            raise R260SidecarMaterializationError("Inzi staged day byte identity drifted")
        _seal_directory(new_day)
        copied.append(day)
    prefix_dir = destination / "r260-prefix-receipts"
    if prefix_dir.exists() or prefix_dir.is_symlink():
        prefix_dir = _regular_directory(prefix_dir, label="Inzi prefix receipt directory")
    else:
        prefix_dir = _new_directory(prefix_dir, label="Inzi prefix receipt directory")
    prefix_payload = _seal(
        {
            "schema": R260_SIDECAR_MATERIALIZATION_SCHEMA,
            "status": (
                "complete_staging_not_training_eligible"
                if len(source_days) == len(expected_days())
                else "partial_staging_not_training_eligible"
            ),
            "owner_contract_sha256": contract.sha256,
            "source_manifest_sha256": contract.source_manifest_sha256,
            "source_window_receipt_sha256": contract.source_window_receipt_sha256,
            "source_elmo_sidecar_root": str(source_root),
            "inzi_staging_root": str(destination),
            "record_key": list(RECORD_KEY_FIELDS),
            "committed_days": list(source_days),
            "daily_meta_files": source.daily_meta_identities,
            "daily_shard_files": source.daily_shard_identities,
            "day_count": len(source_days),
            "expected_day_count": len(expected_days()),
            "append_only": True,
            "final_root_promoted": False,
            "active_r241_training_eligible": False,
            "r260_training_eligible": False,
            "selector_change_authorized": False,
        },
        field="receipt_sha256",
    )
    prefix_path = prefix_dir / f"prefix-{len(source_days):02d}-of-20.json"
    if prefix_path.exists() or prefix_path.is_symlink():
        existing = _read_receipt(
            prefix_path,
            R260_SIDECAR_MATERIALIZATION_SCHEMA,
            statuses={
                "partial_staging_not_training_eligible",
                "complete_staging_not_training_eligible",
            },
        )
        if dict(existing.payload) != prefix_payload:
            raise R260SidecarMaterializationError("existing Inzi prefix receipt drifted")
        prefix_identity = file_identity(prefix_path, label="r260 prefix receipt")
    else:
        prefix_identity = _write_json_create_only(
            prefix_path, prefix_payload, label="r260 prefix receipt"
        )
    return PrefixTransferResult(
        source_audit=source,
        inzi_staging_root=destination,
        committed_days=source_days,
        copied_days=tuple(copied),
        prefix_receipt=prefix_identity,
    )


def promote_r260_inzi_staging_root(
    *,
    inzi_staging_root: Path | str,
    inzi_final_root: Path | str,
    expected_elmo_sidecar_root: Path | str,
    owner_contract: _OwnerContract | None = None,
) -> Path:
    """Atomically promote only a fully receipted twenty-day staging tree.

    Revision 262 requires the join, byte-transport, causal-parity, completion,
    and aggregate precommit receipts to pass *before* the rename.  Their
    FileIdentity paths must already name the exact final sibling root, so the
    promotion makes those paths real without rewriting any evidence byte.
    """

    contract = _owner_contract(owner_contract)
    elmo_root = _expected_elmo_sidecar_path(
        expected_elmo_sidecar_root, contract, require_directory=False
    )
    staging_text = Path(inzi_staging_root).expanduser()
    final_text = Path(inzi_final_root).expanduser()
    _reject_elmo_destination(staging_text, elmo_root)
    _reject_elmo_destination(final_text, elmo_root)
    if staging_text.resolve() != _configured_inzi_root(
        contract, "inzi_prefix_staging_root", label="r260 configured Inzi staging root"
    ) or final_text.resolve() != _configured_inzi_root(
        contract, "inzi_training_root", label="r260 configured Inzi final root"
    ):
        raise R260SidecarMaterializationError(
            "r260 promotion roots do not match the owner-configured staging/final paths"
        )
    staging = _regular_directory(staging_text, label="Inzi sidecar staging root")
    if final_text.exists() or final_text.is_symlink():
        raise R260SidecarMaterializationError("Inzi final sidecar root already exists")
    if staging.parent.resolve() != final_text.parent.expanduser().resolve():
        raise R260SidecarMaterializationError(
            "Inzi staging/final roots must be sibling paths for atomic promotion"
        )
    evidence_prefix = staging / "r260-prefix-receipts" / "prefix-20-of-20.json"
    prefix = _read_receipt(
        evidence_prefix,
        R260_SIDECAR_MATERIALIZATION_SCHEMA,
        statuses={"complete_staging_not_training_eligible"},
    )
    if prefix.payload.get("status") != "complete_staging_not_training_eligible":
        raise R260SidecarMaterializationError("Inzi staging prefix is not complete")
    # The source receipt copies may not exist before full evidence
    # materialization.  Validate the exact final day topology directly here.
    daily_root = _regular_directory(
        staging / rollout_store.DAILY_DIRECTORY_NAME, label="Inzi staged daily root"
    )
    days = tuple(sorted(child.name for child in daily_root.iterdir() if not child.name.startswith(".")))
    if days != expected_days():
        raise R260SidecarMaterializationError("Inzi staging root is not 20/20")
    evidence = _regular_directory(
        staging / EVIDENCE_DIRECTORY_NAME, label="Inzi staged r260 evidence root"
    )
    materialized = _materialization_from_existing(
        sidecar_root=staging, evidence_root=evidence, owner_contract=contract
    )
    receipts_dir = _regular_directory(
        evidence / RECEIPTS_DIRECTORY_NAME, label="Inzi staged r260 receipt directory"
    )
    join = _read_receipt(receipts_dir / JOIN_RECEIPT_NAME, R260_JOIN_RECEIPT_SCHEMA)
    schema = _read_receipt(receipts_dir / SCHEMA_RECEIPT_NAME, R260_SCHEMA_RECEIPT_SCHEMA)
    count = _read_receipt(receipts_dir / COUNT_RECEIPT_NAME, R260_COUNT_RECEIPT_SCHEMA)
    digest = _read_receipt(receipts_dir / DIGEST_RECEIPT_NAME, R260_DIGEST_RECEIPT_SCHEMA)
    transport = _read_receipt(
        receipts_dir / TRANSPORT_RECEIPT_NAME, R260_TRANSPORT_RECEIPT_SCHEMA
    )
    parity = _read_receipt(receipts_dir / PARITY_RECEIPT_NAME, R260_PARITY_RECEIPT_SCHEMA)
    completion_receipt = _read_receipt(
        receipts_dir / COMPLETION_RECEIPT_NAME, R260_COMPLETION_RECEIPT_SCHEMA
    )
    _validate_materialization_receipts(
        materialized, contract=contract, receipts=(join, schema, count, digest)
    )
    _validate_precommit_materialization_paths(
        materialized=materialized,
        contract=contract,
        physical_root=staging,
        logical_root=final_text.resolve(),
    )
    _validate_transport_receipt(transport, materialized=materialized, contract=contract)
    if transport.payload.get("inzi_sidecar_root") != str(final_text.resolve()):
        raise R260SidecarMaterializationError(
            "staged transport receipt does not precommit the configured final root"
        )
    _validate_parity_receipt(parity, materialized=materialized, contract=contract)
    expected_joined = _project_identity(
        materialized.joined_dataset,
        physical_root=staging,
        logical_root=final_text.resolve(),
    ).as_dict()
    if parity.payload.get("inzi_joined_dataset") != expected_joined:
        raise R260SidecarMaterializationError(
            "staged parity receipt does not precommit the configured final root"
        )
    completion_identity = file_identity(
        completion_receipt.path, label="r260 staged completion receipt"
    )
    _validate_completion_receipt(
        completion_receipt,
        materialized=materialized,
        contract=contract,
        physical_root=staging,
        logical_root=final_text.resolve(),
    )
    aggregate_payload = _read_json_object(
        evidence / AGGREGATE_BINDING_NAME, label="r260 staged aggregate binding"
    )
    _validate_aggregate_binding_payload(
        aggregate_payload,
        materialized=materialized,
        completion=completion_identity,
        contract=contract,
        physical_root=staging,
        logical_root=final_text.resolve(),
    )
    try:
        os.replace(staging, final_text)
    except OSError as exc:
        raise R260SidecarMaterializationError("could not atomically promote Inzi staging root") from exc
    return _regular_directory(final_text, label="atomically promoted Inzi final root")


def _verify_staged_day(directory: Path, expected: DailyArtifact) -> None:
    target = _regular_directory(directory, label=f"Inzi staged day {expected.day}")
    members = {child.name for child in target.iterdir()}
    if members != {rollout_store.DAILY_META_NAME, rollout_store.DAILY_SHARD_NAME}:
        raise R260SidecarMaterializationError("Inzi staged day has unexpected members")
    meta = file_identity(target / rollout_store.DAILY_META_NAME, label="Inzi staged metadata")
    shard = file_identity(target / rollout_store.DAILY_SHARD_NAME, label="Inzi staged shard")
    if meta.sha256 != expected.meta.sha256 or meta.size_bytes != expected.meta.size_bytes:
        raise R260SidecarMaterializationError("Inzi staged metadata identity drifted")
    if shard.sha256 != expected.shard.sha256 or shard.size_bytes != expected.shard.size_bytes:
        raise R260SidecarMaterializationError("Inzi staged shard identity drifted")


def attest_r260_causal_local_remote_parity(
    *,
    elmo_sidecar_root: Path | str,
    elmo_evidence_root: Path | str,
    inzi_sidecar_root: Path | str,
    inzi_evidence_root: Path | str,
    expected_elmo_sidecar_root: Path | str,
    owner_contract: _OwnerContract | None = None,
    sample_limit: int = DEFAULT_CAUSAL_PARITY_SAMPLES,
    receipt_identity_root: Path | str | None = None,
) -> FileIdentity:
    """Write bounded semantic causal parity evidence in the Inzi evidence root."""

    contract = _owner_contract(owner_contract)
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int):
        raise R260SidecarMaterializationError("r260 parity sample_limit must be an integer")
    if not 1 <= sample_limit <= MAX_CAUSAL_PARITY_SAMPLES:
        raise R260SidecarMaterializationError("r260 parity sample_limit is outside the bounded range")
    expected_elmo = _regular_directory(
        expected_elmo_sidecar_root, label="expected Elmo sidecar root"
    )
    local_root = _regular_directory(elmo_sidecar_root, label="Elmo sidecar root")
    if local_root != expected_elmo:
        raise R260SidecarMaterializationError("r260 parity local root is not Elmo")
    remote_root_text = Path(inzi_sidecar_root).expanduser()
    _reject_elmo_destination(remote_root_text, expected_elmo)
    local = _materialization_from_existing(
        sidecar_root=local_root,
        evidence_root=elmo_evidence_root,
        owner_contract=contract,
    )
    remote = _materialization_from_existing(
        sidecar_root=remote_root_text,
        evidence_root=inzi_evidence_root,
        owner_contract=contract,
    )
    configured_final = _configured_inzi_root(
        contract, "inzi_training_root", label="r260 configured Inzi final root"
    )
    configured_staging = _configured_inzi_root(
        contract, "inzi_prefix_staging_root", label="r260 configured Inzi staging root"
    )
    if remote.audit.sidecar_root == configured_staging:
        if receipt_identity_root is None or Path(receipt_identity_root).expanduser().resolve() != configured_final:
            raise R260SidecarMaterializationError(
                "staging parity receipts must precommit the configured final Inzi root"
            )
    elif remote.audit.sidecar_root != configured_final:
        raise R260SidecarMaterializationError(
            "r260 parity root is neither configured staging nor final Inzi root"
        )
    logical_remote_root = _receipt_identity_root(
        physical_root=remote.audit.sidecar_root,
        receipt_identity_root=receipt_identity_root,
    )
    _assert_transport_identity(local, remote)
    count = local.row_count
    if count <= 0:
        raise R260SidecarMaterializationError("r260 parity has no joined rows")
    indices = _sample_indices(count, sample_limit)
    local_samples = _causal_sample_projection(local.joined_dataset.path, indices)
    remote_samples = _causal_sample_projection(remote.joined_dataset.path, indices)
    if local_samples != remote_samples:
        raise R260SidecarMaterializationError("r260 local/remote causal parity drifted")
    payload = _seal(
        {
            "schema": R260_PARITY_RECEIPT_SCHEMA,
            "status": "passed",
            "owner_contract_sha256": contract.sha256,
            "source_manifest_sha256": contract.source_manifest_sha256,
            "source_window_receipt_sha256": contract.source_window_receipt_sha256,
            "record_key": list(RECORD_KEY_FIELDS),
            "elmo_joined_dataset": local.joined_dataset.as_dict(),
            "inzi_joined_dataset": _project_identity(
                remote.joined_dataset,
                physical_root=remote.audit.sidecar_root,
                logical_root=logical_remote_root,
            ).as_dict(),
            "joined_dataset_byte_identical": True,
            "daily_meta_byte_identical": True,
            "daily_shard_byte_identical": True,
            "sample_limit": sample_limit,
            "sample_count": len(indices),
            "sample_indices": list(indices),
            "sample_key_sha256": sha256_bytes(canonical_json_bytes(local_samples)),
            "local_causal_projection_sha256": sha256_bytes(
                canonical_json_bytes(local_samples)
            ),
            "remote_causal_projection_sha256": sha256_bytes(
                canonical_json_bytes(remote_samples)
            ),
            "active_r241_training_eligible": False,
        },
        field="receipt_sha256",
    )
    return _write_json_create_only(
        Path(inzi_evidence_root) / RECEIPTS_DIRECTORY_NAME / PARITY_RECEIPT_NAME,
        payload,
        label="r260 causal local/remote parity receipt",
    )


def finalize_r260_inzi_sidecar(
    *,
    inzi_sidecar_root: Path | str,
    inzi_evidence_root: Path | str,
    expected_elmo_sidecar_root: Path | str,
    owner_contract: _OwnerContract | None = None,
    validate_with_launcher: bool = True,
    receipt_identity_root: Path | str | None = None,
) -> CompletionResult:
    """Publish completion/aggregate evidence after parity and transport pass.

    With ``receipt_identity_root`` the receipts are precommitted in a complete
    staging tree for the exact final sibling root.  Callers must atomically
    promote that tree and then run :func:`verify_r260_inzi_sidecar_completion`
    before creating the final Inzi dataset binding.
    """

    contract = _owner_contract(owner_contract)
    expected_elmo = _regular_directory(
        expected_elmo_sidecar_root, label="expected Elmo sidecar root"
    )
    root_text = Path(inzi_sidecar_root).expanduser()
    _reject_elmo_destination(root_text, expected_elmo)
    root = _regular_directory(root_text, label="Inzi sidecar root")
    evidence = _regular_directory(inzi_evidence_root, label="Inzi evidence root")
    _require_descendant(evidence, root, label="Inzi evidence root")
    materialized = _materialization_from_existing(
        sidecar_root=root, evidence_root=evidence, owner_contract=contract
    )
    logical_root = _receipt_identity_root(
        physical_root=root, receipt_identity_root=receipt_identity_root
    )
    configured_final = _configured_inzi_root(
        contract, "inzi_training_root", label="r260 configured Inzi final root"
    )
    configured_staging = _configured_inzi_root(
        contract, "inzi_prefix_staging_root", label="r260 configured Inzi staging root"
    )
    if root == configured_staging:
        if logical_root != configured_final:
            raise R260SidecarMaterializationError(
                "staging completion must precommit the configured final Inzi root"
            )
    elif root != configured_final or logical_root != configured_final:
        raise R260SidecarMaterializationError(
            "r260 completion must target configured staging precommit or final Inzi root"
        )
    receipts_dir = _regular_directory(
        evidence / RECEIPTS_DIRECTORY_NAME, label="Inzi r260 receipt directory"
    )
    join = _read_receipt(receipts_dir / JOIN_RECEIPT_NAME, R260_JOIN_RECEIPT_SCHEMA)
    schema = _read_receipt(receipts_dir / SCHEMA_RECEIPT_NAME, R260_SCHEMA_RECEIPT_SCHEMA)
    count = _read_receipt(receipts_dir / COUNT_RECEIPT_NAME, R260_COUNT_RECEIPT_SCHEMA)
    digest = _read_receipt(receipts_dir / DIGEST_RECEIPT_NAME, R260_DIGEST_RECEIPT_SCHEMA)
    transport = _read_receipt(
        receipts_dir / TRANSPORT_RECEIPT_NAME, R260_TRANSPORT_RECEIPT_SCHEMA
    )
    parity = _read_receipt(receipts_dir / PARITY_RECEIPT_NAME, R260_PARITY_RECEIPT_SCHEMA)
    _validate_materialization_receipts(
        materialized,
        contract=contract,
        receipts=(join, schema, count, digest),
    )
    _validate_transport_receipt(transport, materialized=materialized, contract=contract)
    _validate_parity_receipt(parity, materialized=materialized, contract=contract)
    parity_identity = file_identity(parity.path, label="r260 parity receipt")
    completion_payload = _seal(
        {
            "schema": R260_COMPLETION_RECEIPT_SCHEMA,
            "status": "complete",
            "owner_contract_sha256": contract.sha256,
            "source_manifest_sha256": contract.source_manifest_sha256,
            "source_window_receipt_sha256": contract.source_window_receipt_sha256,
            "day_count": len(materialized.audit.daily),
            "validated_episode_count": materialized.audit.validated_episode_count,
            "source_archive_bytes": materialized.audit.source_archive_bytes,
            "joined_dataset": _project_identity(
                materialized.joined_dataset, physical_root=root, logical_root=logical_root
            ).as_dict(),
            "join_receipt": _project_identity(
                file_identity(join.path, label="r260 join receipt"),
                physical_root=root,
                logical_root=logical_root,
            ).as_dict(),
            "schema_receipt": _project_identity(
                file_identity(schema.path, label="r260 schema receipt"),
                physical_root=root,
                logical_root=logical_root,
            ).as_dict(),
            "count_receipt": _project_identity(
                file_identity(count.path, label="r260 count receipt"),
                physical_root=root,
                logical_root=logical_root,
            ).as_dict(),
            "digest_receipt": _project_identity(
                file_identity(digest.path, label="r260 digest receipt"),
                physical_root=root,
                logical_root=logical_root,
            ).as_dict(),
            "causal_local_remote_parity_receipt": _project_identity(
                parity_identity, physical_root=root, logical_root=logical_root
            ).as_dict(),
            "complete_twenty_day_sidecar": True,
            "active_r241_training_eligible": False,
        },
        field="receipt_sha256",
    )
    completion = _write_json_create_only(
        receipts_dir / COMPLETION_RECEIPT_NAME,
        completion_payload,
        label="r260 completion receipt",
    )
    terminal = {
        "completion": _project_identity(
            completion, physical_root=root, logical_root=logical_root
        ).as_dict(),
        "join": _project_identity(
            file_identity(join.path, label="r260 join receipt"),
            physical_root=root,
            logical_root=logical_root,
        ).as_dict(),
        "schema": _project_identity(
            file_identity(schema.path, label="r260 schema receipt"),
            physical_root=root,
            logical_root=logical_root,
        ).as_dict(),
        "count": _project_identity(
            file_identity(count.path, label="r260 count receipt"),
            physical_root=root,
            logical_root=logical_root,
        ).as_dict(),
        "digest": _project_identity(
            file_identity(digest.path, label="r260 digest receipt"),
            physical_root=root,
            logical_root=logical_root,
        ).as_dict(),
        "causal_local_remote_parity": _project_identity(
            parity_identity, physical_root=root, logical_root=logical_root
        ).as_dict(),
    }
    binding_payload = {
        "schema": R260_AGGREGATE_BINDING_SCHEMA,
        "status": "complete_training_eligible",
        "owner_contract_sha256": contract.sha256,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "day_count": len(materialized.audit.daily),
        "validated_episode_count": materialized.audit.validated_episode_count,
        "source_archive_bytes": materialized.audit.source_archive_bytes,
        "daily_sidecar_meta_receipts": _project_daily_identities(
            materialized.audit.daily_meta_identities,
            physical_root=root,
            logical_root=logical_root,
        ),
        "terminal_receipts": terminal,
        "daily_build_identity": dict(materialized.audit.daily_build_identity),
        "partial_or_unreceipted_side_store_training_eligible": False,
    }
    binding_payload["binding_sha256"] = semantic_digest(
        binding_payload, field="binding_sha256"
    )
    aggregate = _write_json_create_only(
        evidence / AGGREGATE_BINDING_NAME,
        binding_payload,
        label="r260 aggregate sidecar binding",
    )
    _validate_completion_receipt(
        _read_receipt(completion.path, R260_COMPLETION_RECEIPT_SCHEMA),
        materialized=materialized,
        contract=contract,
        physical_root=root,
        logical_root=logical_root,
    )
    _validate_aggregate_binding_payload(
        _read_json_object(aggregate.path, label="r260 aggregate binding"),
        materialized=materialized,
        completion=completion,
        contract=contract,
        physical_root=root,
        logical_root=logical_root,
    )
    # Keep production coupled to the actual launch validator rather than a
    # hand-maintained lookalike.  The opt-out exists only for lightweight
    # receipt-fixture tests running in an environment deliberately without the
    # model runtime; it is never exposed by the CLI or deployment path.
    if validate_with_launcher and logical_root == root:
        try:
            from .r241_own_deck_successor import validate_r260_sidecar_binding

            validate_r260_sidecar_binding(
                _read_json_object(aggregate.path, label="r260 aggregate binding"),
                owner_contract=contract,  # type: ignore[arg-type]
                verify_daily_receipt_files=True,
            )
        except Exception as exc:
            raise R260SidecarMaterializationError(
                "r260 aggregate binding is not accepted by its launch validator"
            ) from exc
    _seal_directory(receipts_dir)
    # The top-level evidence directory stays writable for exactly one final
    # create-only Inzi binding.  ``bind_r260_inzi_dataset`` seals it after that
    # file has been published; reopening it is never an option.
    return CompletionResult(
        audit=materialized.audit,
        inzi_sidecar_root=root,
        evidence_root=evidence,
        aggregate_binding=aggregate,
        completion_receipt=completion,
        parity_receipt=parity_identity,
        joined_dataset=materialized.joined_dataset,
    )


def verify_r260_inzi_sidecar_completion(
    *,
    inzi_sidecar_root: Path | str,
    inzi_evidence_root: Path | str,
    expected_elmo_sidecar_root: Path | str,
    owner_contract: _OwnerContract | None = None,
    validate_with_launcher: bool = True,
) -> CompletionResult:
    """Rehash and accept a fully promoted Inzi completion without writing.

    This is the mandatory post-rename step for revision-262 precommit
    receipts: all projected final FileIdentity paths must now exist and pass
    the actual r241 aggregate validator before an Inzi binding can be made.
    """

    contract = _owner_contract(owner_contract)
    expected_elmo = _regular_directory(
        expected_elmo_sidecar_root, label="expected Elmo sidecar root"
    )
    root_text = Path(inzi_sidecar_root).expanduser()
    _reject_elmo_destination(root_text, expected_elmo)
    root = _regular_directory(root_text, label="Inzi sidecar root")
    _require_final_inzi_root(root, contract)
    evidence = _regular_directory(inzi_evidence_root, label="Inzi evidence root")
    _require_descendant(evidence, root, label="Inzi evidence root")
    materialized = _materialization_from_existing(
        sidecar_root=root, evidence_root=evidence, owner_contract=contract
    )
    receipts_dir = _regular_directory(
        evidence / RECEIPTS_DIRECTORY_NAME, label="Inzi r260 receipt directory"
    )
    join = _read_receipt(receipts_dir / JOIN_RECEIPT_NAME, R260_JOIN_RECEIPT_SCHEMA)
    schema = _read_receipt(receipts_dir / SCHEMA_RECEIPT_NAME, R260_SCHEMA_RECEIPT_SCHEMA)
    count = _read_receipt(receipts_dir / COUNT_RECEIPT_NAME, R260_COUNT_RECEIPT_SCHEMA)
    digest = _read_receipt(receipts_dir / DIGEST_RECEIPT_NAME, R260_DIGEST_RECEIPT_SCHEMA)
    transport = _read_receipt(
        receipts_dir / TRANSPORT_RECEIPT_NAME, R260_TRANSPORT_RECEIPT_SCHEMA
    )
    parity = _read_receipt(receipts_dir / PARITY_RECEIPT_NAME, R260_PARITY_RECEIPT_SCHEMA)
    completion_receipt = _read_receipt(
        receipts_dir / COMPLETION_RECEIPT_NAME, R260_COMPLETION_RECEIPT_SCHEMA
    )
    _validate_materialization_receipts(
        materialized, contract=contract, receipts=(join, schema, count, digest)
    )
    _validate_transport_receipt(transport, materialized=materialized, contract=contract)
    _validate_parity_receipt(parity, materialized=materialized, contract=contract)
    completion_identity = file_identity(
        completion_receipt.path, label="r260 completion receipt"
    )
    _validate_completion_receipt(
        completion_receipt,
        materialized=materialized,
        contract=contract,
        physical_root=root,
        logical_root=root,
    )
    aggregate = file_identity(
        evidence / AGGREGATE_BINDING_NAME, label="r260 aggregate sidecar binding"
    )
    aggregate_payload = _read_json_object(
        aggregate.path, label="r260 aggregate sidecar binding"
    )
    _validate_aggregate_binding_payload(
        aggregate_payload,
        materialized=materialized,
        completion=completion_identity,
        contract=contract,
        physical_root=root,
        logical_root=root,
    )
    if validate_with_launcher:
        try:
            from .r241_own_deck_successor import validate_r260_sidecar_binding

            validate_r260_sidecar_binding(
                aggregate_payload,
                owner_contract=contract,  # type: ignore[arg-type]
                verify_daily_receipt_files=True,
            )
        except Exception as exc:
            raise R260SidecarMaterializationError(
                "promoted r260 aggregate is not accepted by its launch validator"
            ) from exc
    return CompletionResult(
        audit=materialized.audit,
        inzi_sidecar_root=root,
        evidence_root=evidence,
        aggregate_binding=aggregate,
        completion_receipt=completion_identity,
        parity_receipt=file_identity(parity.path, label="r260 parity receipt"),
        joined_dataset=materialized.joined_dataset,
    )


def bind_r260_inzi_dataset(
    *,
    completion: CompletionResult,
    source_transport_receipt: Path | str,
    owner_contract: _OwnerContract | None = None,
    validate_with_launcher: bool = True,
) -> FileIdentity:
    """Write the final Inzi dataset binding consumed by the r260 launch gate."""

    contract = _owner_contract(owner_contract)
    root = _regular_directory(completion.inzi_sidecar_root, label="Inzi sidecar root")
    evidence = _regular_directory(completion.evidence_root, label="Inzi evidence root")
    if root != completion.inzi_sidecar_root.resolve():
        raise R260SidecarMaterializationError("Inzi sidecar root drifted before binding")
    _reject_elmo_text_path(str(root), str(contract.side_store_root))
    _require_final_inzi_root(root, contract)
    _require_descendant(evidence, root, label="Inzi evidence root")
    transport = _read_receipt(source_transport_receipt, R260_TRANSPORT_RECEIPT_SCHEMA)
    aggregate_payload = _read_json_object(
        completion.aggregate_binding.path, label="r260 aggregate sidecar binding"
    )
    aggregate_semantic = str(aggregate_payload.get("binding_sha256") or "")
    _require_sha256(aggregate_semantic, label="r260 aggregate binding semantic digest")
    aggregate_file = file_identity(
        completion.aggregate_binding.path, label="r260 aggregate binding file"
    )
    joined = file_identity(
        completion.joined_dataset.path, label="r260 Inzi joined dataset"
    )
    source_joined = _identity_from_mapping(
        transport.payload.get("source_joined_dataset"), label="r260 source joined dataset"
    )
    if source_joined.sha256 != joined.sha256 or source_joined.size_bytes != joined.size_bytes:
        raise R260SidecarMaterializationError("r260 Inzi dataset is not byte-identical to Elmo")
    receipts_dir = evidence / RECEIPTS_DIRECTORY_NAME
    joined_receipt = _read_receipt(receipts_dir / JOIN_RECEIPT_NAME, R260_JOIN_RECEIPT_SCHEMA)
    schema_receipt = _read_receipt(receipts_dir / SCHEMA_RECEIPT_NAME, R260_SCHEMA_RECEIPT_SCHEMA)
    count_receipt = _read_receipt(receipts_dir / COUNT_RECEIPT_NAME, R260_COUNT_RECEIPT_SCHEMA)
    digest_receipt = _read_receipt(receipts_dir / DIGEST_RECEIPT_NAME, R260_DIGEST_RECEIPT_SCHEMA)
    parity_receipt = _read_receipt(receipts_dir / PARITY_RECEIPT_NAME, R260_PARITY_RECEIPT_SCHEMA)

    # The current launcher validator historically called this semantic field
    # ``sidecar_binding_sha256``.  It must remain the binding's semantic digest;
    # the raw FileIdentity is carried separately so no impossible SHA fixed
    # point is required.  Newer validators require the explicit raw field;
    # callers on the older ABI receive a fail-closed explanation rather than a
    # forged binding.
    payload: dict[str, Any] = {
        "schema": R260_INZI_BINDING_SCHEMA,
        "status": "complete_transport_ready",
        "owner_contract_sha256": contract.sha256,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "sidecar_binding_sha256": aggregate_semantic,
        "sidecar_binding": aggregate_file.as_dict(),
        "transport_kind": "create_only_copy",
        "elmo_joined_dataset_sha256": source_joined.sha256,
        "inzi_joined_dataset_sha256": joined.sha256,
        "join_receipt_sha256": _receipt_semantic_sha(joined_receipt),
        "schema_receipt_sha256": _receipt_semantic_sha(schema_receipt),
        "count_receipt_sha256": _receipt_semantic_sha(count_receipt),
        "digest_receipt_sha256": _receipt_semantic_sha(digest_receipt),
        "causal_local_remote_parity_receipt_sha256": _receipt_semantic_sha(parity_receipt),
        "transport_receipt_sha256": _receipt_semantic_sha(transport),
        "day_count": len(completion.audit.daily),
        "validated_episode_count": completion.audit.validated_episode_count,
        "source_archive_bytes": completion.audit.source_archive_bytes,
        "inzi_joined_dataset": joined.as_dict(),
        "inzi_sidecar_root": str(root),
        # Corrected r260 validator ABI: raw file SHA is not the semantic
        # binding digest.  Kept explicit to make the cross-host proof auditable.
        "sidecar_binding_file_sha256": aggregate_file.sha256,
    }
    payload["binding_sha256"] = semantic_digest(payload, field="binding_sha256")
    target = evidence / INZI_BINDING_NAME
    result = _write_json_create_only(target, payload, label="r260 Inzi dataset binding")
    if validate_with_launcher:
        try:
            from .r241_own_deck_successor import validate_r260_inzi_dataset_binding

            validate_r260_inzi_dataset_binding(
                _read_json_object(result.path, label="r260 Inzi dataset binding"),
                sidecar_binding=aggregate_payload,
                owner_contract=contract,  # type: ignore[arg-type]
                require_local_dataset=True,
            )
        except Exception as exc:
            raise R260SidecarMaterializationError(
                "r260 Inzi dataset binding is not accepted by its launch validator"
            ) from exc
    _seal_directory(evidence)
    return result


def _owner_contract(value: _OwnerContract | None) -> _OwnerContract:
    if value is not None:
        _require_sha256(value.sha256, label="r260 owner contract")
        _require_sha256(value.source_manifest_sha256, label="r260 source manifest")
        _require_sha256(
            value.source_window_receipt_sha256, label="r260 source-window receipt"
        )
        if not str(value.side_store_root or ""):
            raise R260SidecarMaterializationError("r260 owner side-store root is missing")
        for attribute, label in (
            ("inzi_training_root", "r260 owner Inzi final root"),
            ("inzi_prefix_staging_root", "r260 owner Inzi staging root"),
        ):
            path_text = str(getattr(value, attribute, "") or "")
            if not path_text or not Path(path_text).expanduser().is_absolute():
                raise R260SidecarMaterializationError(f"{label} is missing or not absolute")
        return value
    from .r241_own_deck_successor import load_r260_owner_contract

    return load_r260_owner_contract()


def _validate_source_manifest(
    value: Mapping[str, Any], *, contract: _OwnerContract
) -> list[Mapping[str, Any]]:
    if value.get("schema") != rollout_store.ARCHIVE_RECEIPT_SCHEMA or value.get("status") != "ready":
        raise R260SidecarMaterializationError("r260 source manifest is not a ready latest20 receipt")
    if (
        value.get("window_policy") != "exact_20_consecutive_calendar_days"
        or value.get("window_start") != expected_days()[0]
        or value.get("window_end") != expected_days()[-1]
        or _strict_int(value.get("days"), label="r260 source manifest days") != 20
        or _strict_int(value.get("total_episodes"), label="r260 source manifest episodes")
        != 91_253
    ):
        raise R260SidecarMaterializationError("r260 source manifest window drifted")
    archives = value.get("archives")
    if not isinstance(archives, list) or len(archives) != 20:
        raise R260SidecarMaterializationError("r260 source manifest does not list 20 archives")
    dates: list[str] = []
    total_bytes = 0
    total_episodes = 0
    for row in archives:
        if not isinstance(row, Mapping):
            raise R260SidecarMaterializationError("r260 archive row is not an object")
        day = row.get("date")
        if not isinstance(day, str):
            raise R260SidecarMaterializationError("r260 archive date is invalid")
        dates.append(day)
        _require_sha256(row.get("sha256"), label=f"r260 archive {day} checksum")
        if row.get("validated") is not True:
            raise R260SidecarMaterializationError(f"r260 archive {day} is not validated")
        if row.get("dataset_slug") != f"pokemon-tcg-ai-battle-episodes-{day}":
            raise R260SidecarMaterializationError(f"r260 archive {day} slug drifted")
        total_bytes += _strict_int(row.get("bytes"), label=f"r260 archive {day} bytes")
        total_episodes += _strict_int(
            row.get("validated_episode_count"), label=f"r260 archive {day} episodes"
        )
    if tuple(dates) != expected_days():
        raise R260SidecarMaterializationError("r260 source manifest dates are incomplete")
    if total_episodes != 91_253 or total_bytes != 14_842_033_482:
        raise R260SidecarMaterializationError("r260 source manifest totals drifted")
    versioned = value.get("versioned_receipt")
    if not isinstance(versioned, str) or not versioned:
        raise R260SidecarMaterializationError("r260 source manifest lacks its versioned receipt")
    # The separate FileIdentity was already rehashed by the caller.  Retain the
    # contract argument here so a future source-manifest schema cannot quietly
    # lose its explicit versioned-receipt pointer.
    _require_sha256(contract.source_window_receipt_sha256, label="r260 source window")
    return [dict(row) for row in archives]


def _validate_daily_meta(
    meta: Mapping[str, Any],
    *,
    day: str,
    archive: Mapping[str, Any],
    contract: _OwnerContract,
    observed_shard: FileIdentity,
) -> None:
    if (
        meta.get("schema") != rollout_store.OWN_DECK_ROLLOUT_DAILY_META_SCHEMA
        or meta.get("version") != rollout_store.OWN_DECK_ROLLOUT_DAILY_META_VERSION
        or meta.get("status") != "complete_immutable_sidecar"
        or meta.get("day") != day
    ):
        raise R260SidecarMaterializationError(f"r260 daily {day} metadata schema drifted")
    source = _mapping(meta.get("source"), label=f"r260 daily {day} source")
    manifest = _mapping(source.get("manifest"), label=f"r260 daily {day} manifest")
    versioned = _mapping(
        source.get("versioned_receipt"), label=f"r260 daily {day} versioned receipt"
    )
    archive_meta = _mapping(source.get("archive"), label=f"r260 daily {day} archive")
    if (
        manifest.get("sha256") != contract.source_manifest_sha256
        or manifest.get("window_start") != expected_days()[0]
        or manifest.get("window_end") != expected_days()[-1]
        or _strict_int(manifest.get("days"), label=f"r260 daily {day} days") != 20
        or _strict_int(
            manifest.get("total_episodes"), label=f"r260 daily {day} source episodes"
        )
        != 91_253
        or versioned.get("sha256") != contract.source_window_receipt_sha256
    ):
        raise R260SidecarMaterializationError(f"r260 daily {day} source binding drifted")
    for name in ("date", "sha256", "bytes", "validated_episode_count"):
        if archive_meta.get(name) != archive.get(name):
            raise R260SidecarMaterializationError(
                f"r260 daily {day} archive identity drifted at {name}"
            )
    if archive_meta.get("source_slug") != archive.get("dataset_slug"):
        raise R260SidecarMaterializationError(
            f"r260 daily {day} archive identity drifted at source_slug"
        )
    shard = _mapping(meta.get("shard"), label=f"r260 daily {day} shard metadata")
    if (
        meta.get("shard_sha256") != observed_shard.sha256
        or shard.get("sha256") != observed_shard.sha256
        or _strict_int(shard.get("bytes"), label=f"r260 daily {day} shard bytes")
        != observed_shard.size_bytes
        or shard.get("path") != rollout_store.DAILY_SHARD_NAME
        or shard.get("compression") != "gzip"
        or shard.get("format") != "jsonl"
        or shard.get("row_schema") != rollout_store.OWN_DECK_ROLLOUT_SIDECAR_SCHEMA
        or shard.get("row_version") != rollout_store.OWN_DECK_ROLLOUT_SIDECAR_VERSION
    ):
        raise R260SidecarMaterializationError(f"r260 daily {day} shard identity drifted")
    eligibility = _mapping(
        meta.get("training_eligibility"), label=f"r260 daily {day} training eligibility"
    )
    if eligibility.get("active_r241") is not False or eligibility.get("sidecar_only") is not True:
        raise R260SidecarMaterializationError(f"r260 daily {day} has unsafe training authority")


def _daily_build_identity(meta: Mapping[str, Any], *, label: str) -> dict[str, str]:
    build = _mapping(meta.get("build"), label=label)
    if build.get("mode") != "archive_native":
        raise R260SidecarMaterializationError(f"{label} is not archive-native")
    code = _mapping(build.get("code"), label=f"{label} code")
    if not code:
        raise R260SidecarMaterializationError(f"{label} code inventory is empty")
    for name, digest in code.items():
        if not isinstance(name, str) or not name:
            raise R260SidecarMaterializationError(f"{label} code identity name is invalid")
        _require_sha256(digest, label=f"{label} code {name}")
    snapshot = _mapping(build.get("source_snapshot"), label=f"{label} source snapshot")
    image = _mapping(build.get("image"), label=f"{label} image")
    classifier = _mapping(build.get("classifier"), label=f"{label} classifier")
    snapshot_sha = snapshot.get("tree_sha256")
    image_sha = image.get("id")
    _require_sha256(snapshot_sha, label=f"{label} source snapshot identity")
    _require_sha256(image_sha, label=f"{label} image identity")
    code_sha = sha256_bytes(
        canonical_json_bytes({"r259_build_code": dict(sorted(code.items()))})
    )
    classifier_sha = sha256_bytes(canonical_json_bytes(dict(classifier)))
    return {
        "sidecar_build_code_sha256": code_sha,
        "source_snapshot_tree_sha256": str(snapshot_sha),
        "container_image_id": str(image_sha),
        "archive_native_classifier_sha256": classifier_sha,
    }


def _write_deterministic_joined_dataset(
    audit: SidecarAudit, target: Path
) -> tuple[FileIdentity, int, str, str]:
    """External-sort sidecar rows without holding the corpus in process memory."""

    if target.exists() or target.is_symlink():
        raise R260SidecarMaterializationError("r260 joined dataset target already exists")
    work_fd, work_name = tempfile.mkstemp(
        prefix=".r260-sidecar-index.", suffix=".sqlite3", dir=target.parent
    )
    os.close(work_fd)
    work = Path(work_name)
    try:
        connection = sqlite3.connect(work)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE rows (episode_id TEXT NOT NULL, seat INTEGER NOT NULL, "
                "env_step INTEGER NOT NULL, observation_fingerprint TEXT NOT NULL, "
                "source_date TEXT NOT NULL, daily_meta_sha256 TEXT NOT NULL, payload BLOB NOT NULL, "
                "PRIMARY KEY(episode_id, seat, env_step, observation_fingerprint))"
            )
            for daily in audit.daily:
                try:
                    rows = rollout_store.iter_daily_sidecar_rows(
                        audit.sidecar_root,
                        daily.day,
                        expected_meta_sha256=_daily_meta_semantic_digest(daily.meta.path),
                    )
                    for row in rows:
                        key = _row_key(row, label=f"r260 daily {daily.day} row")
                        if str(row.get("source_date") or "") != daily.day:
                            raise R260SidecarMaterializationError(
                                "r260 daily row source date drifted"
                            )
                        payload = canonical_json_bytes(dict(row), newline=False)
                        try:
                            connection.execute(
                                "INSERT INTO rows VALUES (?,?,?,?,?,?,?)",
                                (*key, daily.day, _daily_meta_semantic_digest(daily.meta.path), payload),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise R260SidecarMaterializationError(
                                "r260 sidecar has a duplicate canonical four-field key"
                            ) from exc
                except rollout_store.OwnDeckRolloutStoreError as exc:
                    raise R260SidecarMaterializationError(
                        f"r260 daily {daily.day} row validation failed"
                    ) from exc
            connection.commit()
            row_count = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
            if row_count <= 0:
                raise R260SidecarMaterializationError("r260 complete sidecar contains no joinable rows")
            row_digest = hashlib.sha256()
            key_digest = hashlib.sha256()
            temp_name: str | None = None
            try:
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
                )
                with os.fdopen(descriptor, "wb") as raw:
                    with gzip.GzipFile(
                        fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9
                    ) as zipped:
                        cursor = connection.execute(
                            "SELECT episode_id, seat, env_step, observation_fingerprint, "
                            "source_date, daily_meta_sha256, payload FROM rows "
                            "ORDER BY episode_id, seat, env_step, observation_fingerprint"
                        )
                        for episode_id, seat, env_step, fingerprint, source_date, meta_sha, payload in cursor:
                            sidecar_row = json.loads(bytes(payload).decode("utf-8"))
                            joined = {
                                "schema": R260_JOINED_ROW_SCHEMA,
                                "version": 1,
                                "key": {
                                    "episode_id": episode_id,
                                    "seat": seat,
                                    "env_step": env_step,
                                    "observation_fingerprint": fingerprint,
                                },
                                "source_date": source_date,
                                "source_manifest_sha256": sidecar_row["source_manifest_sha256"],
                                "daily_meta_sha256": meta_sha,
                                "sidecar_row": sidecar_row,
                            }
                            encoded = canonical_json_bytes(joined)
                            zipped.write(encoded)
                            row_digest.update(encoded)
                            key_digest.update(
                                canonical_json_bytes(joined["key"], newline=False)
                            )
                            key_digest.update(b"\n")
                    raw.flush()
                    os.fsync(raw.fileno())
                _publish_temp_create_only(Path(temp_name), target, label="r260 joined dataset")
                temp_name = None
            finally:
                if temp_name is not None:
                    Path(temp_name).unlink(missing_ok=True)
        finally:
            connection.close()
    finally:
        work.unlink(missing_ok=True)
    joined = file_identity(target, label="r260 joined dataset")
    _make_readonly(target)
    return (
        joined,
        row_count,
        "sha256:" + row_digest.hexdigest(),
        "sha256:" + key_digest.hexdigest(),
    )


def _materialization_from_existing(
    *, sidecar_root: Path | str, evidence_root: Path | str, owner_contract: _OwnerContract
) -> MaterializationResult:
    """Read and cross-check a previously create-only materialization tree."""

    evidence = _regular_directory(evidence_root, label="r260 evidence root")
    manifest_copy = evidence / SOURCE_MANIFEST_COPY_NAME
    window_copy = evidence / SOURCE_WINDOW_RECEIPT_COPY_NAME
    audit = audit_r260_sidecar(
        sidecar_root=sidecar_root,
        source_manifest=manifest_copy,
        source_window_receipt=window_copy,
        owner_contract=owner_contract,
        expected_sidecar_root=None,
    )
    joined_dir = _regular_directory(evidence / JOINED_DIRECTORY_NAME, label="r260 joined directory")
    joined_dataset = file_identity(joined_dir / JOINED_DATASET_NAME, label="r260 joined dataset")
    manifest = _read_joined_manifest(joined_dir / JOINED_MANIFEST_NAME, audit, owner_contract)
    joined_manifest = file_identity(
        joined_dir / JOINED_MANIFEST_NAME, label="r260 joined manifest"
    )
    receipts_dir = _regular_directory(evidence / RECEIPTS_DIRECTORY_NAME, label="r260 receipts")
    join = file_identity(receipts_dir / JOIN_RECEIPT_NAME, label="r260 join receipt")
    schema = file_identity(receipts_dir / SCHEMA_RECEIPT_NAME, label="r260 schema receipt")
    count = file_identity(receipts_dir / COUNT_RECEIPT_NAME, label="r260 count receipt")
    digest = file_identity(receipts_dir / DIGEST_RECEIPT_NAME, label="r260 digest receipt")
    for identity, expected_schema in (
        (join, R260_JOIN_RECEIPT_SCHEMA),
        (schema, R260_SCHEMA_RECEIPT_SCHEMA),
        (count, R260_COUNT_RECEIPT_SCHEMA),
        (digest, R260_DIGEST_RECEIPT_SCHEMA),
    ):
        _read_receipt(identity.path, expected_schema)
    rows = _strict_int(manifest.get("joined_row_count"), label="r260 joined row count")
    if rows <= 0:
        raise R260SidecarMaterializationError("r260 joined dataset has no rows")
    if not _content_identity_matches(
        manifest.get("joined_dataset"), joined_dataset, label="r260 manifest dataset"
    ):
        raise R260SidecarMaterializationError("r260 joined manifest dataset identity drifted")
    return MaterializationResult(
        audit=audit,
        evidence_root=evidence,
        joined_dataset=joined_dataset,
        joined_manifest=joined_manifest,
        join_receipt=join,
        schema_receipt=schema,
        count_receipt=count,
        digest_receipt=digest,
        row_count=rows,
        joined_rows_sha256=str(manifest["joined_rows_sha256"]),
        joined_keys_sha256=str(manifest["joined_keys_sha256"]),
    )


def _read_joined_manifest(
    path: Path, audit: SidecarAudit, contract: _OwnerContract
) -> dict[str, Any]:
    manifest = _read_json_object(path, label="r260 joined manifest")
    if (
        manifest.get("schema") != R260_JOINED_DATASET_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("manifest_sha256") != semantic_digest(manifest, field="manifest_sha256")
        or manifest.get("owner_contract_sha256") != contract.sha256
        or manifest.get("source_manifest", {}).get("sha256") != contract.source_manifest_sha256
        or manifest.get("source_window_receipt", {}).get("sha256")
        != contract.source_window_receipt_sha256
        or manifest.get("record_key") != list(RECORD_KEY_FIELDS)
        or not _content_identity_map_matches(
            manifest.get("daily_sidecar_meta_files"),
            audit.daily_meta_identities,
            label="r260 joined manifest daily metadata",
        )
        or not _content_identity_map_matches(
            manifest.get("daily_sidecar_shard_files"),
            audit.daily_shard_identities,
            label="r260 joined manifest daily shards",
        )
        or manifest.get("daily_build_identity") != dict(audit.daily_build_identity)
        or manifest.get("active_r241_training_eligible") is not False
    ):
        raise R260SidecarMaterializationError("r260 joined manifest identity drifted")
    _identity_from_mapping(manifest.get("joined_dataset"), label="r260 joined manifest dataset")
    _require_sha256(manifest.get("joined_rows_sha256"), label="r260 joined rows digest")
    _require_sha256(manifest.get("joined_keys_sha256"), label="r260 joined keys digest")
    return manifest


def _copy_evidence_members(source: Path, destination: Path) -> None:
    """Copy only materialization members whose identities are independently checked."""

    relative_files = (
        Path(SOURCE_MANIFEST_COPY_NAME),
        Path(SOURCE_WINDOW_RECEIPT_COPY_NAME),
        Path(JOINED_DIRECTORY_NAME) / JOINED_DATASET_NAME,
        Path(JOINED_DIRECTORY_NAME) / JOINED_MANIFEST_NAME,
        Path(RECEIPTS_DIRECTORY_NAME) / JOIN_RECEIPT_NAME,
        Path(RECEIPTS_DIRECTORY_NAME) / SCHEMA_RECEIPT_NAME,
        Path(RECEIPTS_DIRECTORY_NAME) / COUNT_RECEIPT_NAME,
        Path(RECEIPTS_DIRECTORY_NAME) / DIGEST_RECEIPT_NAME,
    )
    _new_directory(destination / JOINED_DIRECTORY_NAME, label="Inzi joined directory")
    _new_directory(destination / RECEIPTS_DIRECTORY_NAME, label="Inzi receipts directory")
    for relative in relative_files:
        copied = _copy_regular_create_only(
            source / relative,
            destination / relative,
            label=f"r260 copied evidence {relative.as_posix()}",
        )
        original = file_identity(source / relative, label=f"r260 source evidence {relative.as_posix()}")
        if copied.sha256 != original.sha256 or copied.size_bytes != original.size_bytes:
            raise R260SidecarMaterializationError("r260 evidence transport byte identity drifted")
    _seal_directory(destination / JOINED_DIRECTORY_NAME)
    # The receipt directory must remain writable until local parity/finalization.


def _write_transport_receipt(
    *,
    source: MaterializationResult,
    remote: MaterializationResult,
    inzi_evidence_root: Path,
    owner_contract: _OwnerContract,
    receipt_identity_root: Path | str | None,
) -> FileIdentity:
    """Write one transport receipt after a complete source/remote re-audit."""

    _assert_transport_identity(source, remote)
    logical_root = _receipt_identity_root(
        physical_root=remote.audit.sidecar_root,
        receipt_identity_root=receipt_identity_root,
    )
    transport_payload = _seal(
        {
            "schema": R260_TRANSPORT_RECEIPT_SCHEMA,
            "status": "passed",
            "owner_contract_sha256": owner_contract.sha256,
            "transport_kind": "create_only_copy",
            "source_elmo_sidecar_root": str(source.audit.sidecar_root),
            "inzi_sidecar_root": str(logical_root),
            "source_joined_dataset": source.joined_dataset.as_dict(),
            "inzi_joined_dataset": _project_identity(
                remote.joined_dataset,
                physical_root=remote.audit.sidecar_root,
                logical_root=logical_root,
            ).as_dict(),
            "joined_dataset_byte_identical": True,
            "source_joined_manifest": source.joined_manifest.as_dict(),
            "inzi_joined_manifest": _project_identity(
                remote.joined_manifest,
                physical_root=remote.audit.sidecar_root,
                logical_root=logical_root,
            ).as_dict(),
            "daily_meta_source": source.audit.daily_meta_identities,
            "daily_meta_inzi": _project_daily_identities(
                remote.audit.daily_meta_identities,
                physical_root=remote.audit.sidecar_root,
                logical_root=logical_root,
            ),
            "daily_shard_source": source.audit.daily_shard_identities,
            "daily_shard_inzi": _project_daily_identities(
                remote.audit.daily_shard_identities,
                physical_root=remote.audit.sidecar_root,
                logical_root=logical_root,
            ),
            "source_manifest_sha256": owner_contract.source_manifest_sha256,
            "source_window_receipt_sha256": owner_contract.source_window_receipt_sha256,
        },
        field="receipt_sha256",
    )
    return _write_json_create_only(
        inzi_evidence_root / RECEIPTS_DIRECTORY_NAME / TRANSPORT_RECEIPT_NAME,
        transport_payload,
        label="r260 Inzi transport receipt",
    )


def _assert_transport_identity(local: MaterializationResult, remote: MaterializationResult) -> None:
    if (
        local.joined_dataset.sha256 != remote.joined_dataset.sha256
        or local.joined_dataset.size_bytes != remote.joined_dataset.size_bytes
        or local.row_count != remote.row_count
        or local.joined_rows_sha256 != remote.joined_rows_sha256
        or local.joined_keys_sha256 != remote.joined_keys_sha256
        or not _content_identity_map_matches(
            local.audit.daily_meta_identities,
            remote.audit.daily_meta_identities,
            label="r260 local/remote daily metadata",
        )
        or not _content_identity_map_matches(
            local.audit.daily_shard_identities,
            remote.audit.daily_shard_identities,
            label="r260 local/remote daily shards",
        )
    ):
        raise R260SidecarMaterializationError("r260 local/remote byte identity drifted")


def _sample_indices(count: int, limit: int) -> tuple[int, ...]:
    if count <= limit:
        return tuple(range(count))
    if limit == 1:
        return (0,)
    return tuple(sorted({(index * (count - 1)) // (limit - 1) for index in range(limit)}))


def _causal_sample_projection(path: Path, indices: Sequence[int]) -> list[dict[str, Any]]:
    selected = set(indices)
    found: list[dict[str, Any]] = []
    with gzip.open(_regular_file(path, label="r260 joined dataset"), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in selected:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise R260SidecarMaterializationError("r260 joined row is not JSON") from exc
            _validate_joined_row(row)
            sidecar = _mapping(row["sidecar_row"], label="r260 parity sidecar row")
            found.append(
                {
                    "key": row["key"],
                    "deck_fingerprint": sidecar.get("deck_fingerprint"),
                    "ledger_snapshot": sidecar.get("ledger_snapshot"),
                    "board_feature_fingerprint": sidecar.get("board_feature_fingerprint"),
                    "policy_stage_option_features": sidecar.get("policy_stage_option_features"),
                    "supervision": sidecar.get("supervision"),
                }
            )
    if len(found) != len(indices):
        raise R260SidecarMaterializationError("r260 parity joined dataset is shorter than declared")
    return found


def _validate_joined_row(value: object) -> None:
    row = _mapping(value, label="r260 joined row")
    if row.get("schema") != R260_JOINED_ROW_SCHEMA or row.get("version") != 1:
        raise R260SidecarMaterializationError("r260 joined row schema drifted")
    key = _mapping(row.get("key"), label="r260 joined row key")
    _row_key(key, label="r260 joined row key")
    sidecar = _mapping(row.get("sidecar_row"), label="r260 joined sidecar row")
    if _row_key(sidecar, label="r260 joined sidecar row") != _row_key(
        key, label="r260 joined key"
    ):
        raise R260SidecarMaterializationError("r260 joined row key does not match sidecar row")
    if row.get("source_date") != sidecar.get("source_date") or row.get(
        "source_manifest_sha256"
    ) != sidecar.get("source_manifest_sha256"):
        raise R260SidecarMaterializationError("r260 joined row source binding drifted")
    _require_sha256(row.get("daily_meta_sha256"), label="r260 joined daily metadata")


def _logical_identity_for_path(
    path: Path,
    *,
    label: str,
    physical_root: Path,
    logical_root: Path,
) -> dict[str, object]:
    return _project_identity(
        file_identity(path, label=label),
        physical_root=physical_root,
        logical_root=logical_root,
    ).as_dict()


def _validate_completion_receipt(
    receipt: "_Receipt",
    *,
    materialized: MaterializationResult,
    contract: _OwnerContract,
    physical_root: Path,
    logical_root: Path,
) -> None:
    """Check completion evidence against physical bytes and named logical paths."""

    payload = receipt.payload
    receipts_dir = materialized.evidence_root / RECEIPTS_DIRECTORY_NAME
    expected = {
        "schema": R260_COMPLETION_RECEIPT_SCHEMA,
        "status": "complete",
        "owner_contract_sha256": contract.sha256,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "day_count": len(materialized.audit.daily),
        "validated_episode_count": materialized.audit.validated_episode_count,
        "source_archive_bytes": materialized.audit.source_archive_bytes,
        "joined_dataset": _project_identity(
            materialized.joined_dataset,
            physical_root=physical_root,
            logical_root=logical_root,
        ).as_dict(),
        "join_receipt": _logical_identity_for_path(
            receipts_dir / JOIN_RECEIPT_NAME,
            label="r260 join receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "schema_receipt": _logical_identity_for_path(
            receipts_dir / SCHEMA_RECEIPT_NAME,
            label="r260 schema receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "count_receipt": _logical_identity_for_path(
            receipts_dir / COUNT_RECEIPT_NAME,
            label="r260 count receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "digest_receipt": _logical_identity_for_path(
            receipts_dir / DIGEST_RECEIPT_NAME,
            label="r260 digest receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "causal_local_remote_parity_receipt": _logical_identity_for_path(
            receipts_dir / PARITY_RECEIPT_NAME,
            label="r260 causal parity receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "complete_twenty_day_sidecar": True,
        "active_r241_training_eligible": False,
    }
    unsigned = dict(payload)
    receipt_sha = unsigned.pop("receipt_sha256", None)
    if (
        set(payload) != set(expected) | {"receipt_sha256"}
        or payload.get("receipt_sha256") != semantic_digest(payload, field="receipt_sha256")
        or receipt_sha is None
        or unsigned != expected
    ):
        raise R260SidecarMaterializationError("r260 completion receipt drifted")


def _validate_aggregate_binding_payload(
    payload: Mapping[str, Any],
    *,
    materialized: MaterializationResult,
    completion: FileIdentity,
    contract: _OwnerContract,
    physical_root: Path,
    logical_root: Path,
) -> None:
    """Validate the strict aggregate ABI before/after staging promotion."""

    receipts_dir = materialized.evidence_root / RECEIPTS_DIRECTORY_NAME
    terminal = {
        "completion": _project_identity(
            completion, physical_root=physical_root, logical_root=logical_root
        ).as_dict(),
        "join": _logical_identity_for_path(
            receipts_dir / JOIN_RECEIPT_NAME,
            label="r260 join receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "schema": _logical_identity_for_path(
            receipts_dir / SCHEMA_RECEIPT_NAME,
            label="r260 schema receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "count": _logical_identity_for_path(
            receipts_dir / COUNT_RECEIPT_NAME,
            label="r260 count receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "digest": _logical_identity_for_path(
            receipts_dir / DIGEST_RECEIPT_NAME,
            label="r260 digest receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "causal_local_remote_parity": _logical_identity_for_path(
            receipts_dir / PARITY_RECEIPT_NAME,
            label="r260 causal parity receipt",
            physical_root=physical_root,
            logical_root=logical_root,
        ),
    }
    expected = {
        "schema": R260_AGGREGATE_BINDING_SCHEMA,
        "status": "complete_training_eligible",
        "owner_contract_sha256": contract.sha256,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "day_count": len(materialized.audit.daily),
        "validated_episode_count": materialized.audit.validated_episode_count,
        "source_archive_bytes": materialized.audit.source_archive_bytes,
        "daily_sidecar_meta_receipts": _project_daily_identities(
            materialized.audit.daily_meta_identities,
            physical_root=physical_root,
            logical_root=logical_root,
        ),
        "terminal_receipts": terminal,
        "daily_build_identity": dict(materialized.audit.daily_build_identity),
        "partial_or_unreceipted_side_store_training_eligible": False,
    }
    unsigned = dict(payload)
    digest = unsigned.pop("binding_sha256", None)
    if (
        set(payload) != set(expected) | {"binding_sha256"}
        or payload.get("binding_sha256") != semantic_digest(payload, field="binding_sha256")
        or digest is None
        or unsigned != expected
    ):
        raise R260SidecarMaterializationError("r260 aggregate sidecar binding drifted")


def _validate_materialization_receipts(
    materialized: MaterializationResult,
    *,
    contract: _OwnerContract,
    receipts: Sequence[FileIdentity],
) -> None:
    expected = (
        R260_JOIN_RECEIPT_SCHEMA,
        R260_SCHEMA_RECEIPT_SCHEMA,
        R260_COUNT_RECEIPT_SCHEMA,
        R260_DIGEST_RECEIPT_SCHEMA,
    )
    for identity, schema in zip(receipts, expected, strict=True):
        payload = _read_receipt(identity.path, schema).payload
        if (
            payload.get("owner_contract_sha256") != contract.sha256
            or payload.get("source_manifest_sha256") != contract.source_manifest_sha256
            or payload.get("source_window_receipt_sha256")
            != contract.source_window_receipt_sha256
            or payload.get("record_key") != list(RECORD_KEY_FIELDS)
            or not _content_identity_matches(
                payload.get("joined_dataset"),
                materialized.joined_dataset,
                label="r260 materialization receipt dataset",
            )
            or not _content_identity_matches(
                payload.get("joined_manifest"),
                materialized.joined_manifest,
                label="r260 materialization receipt manifest",
            )
            or not _content_identity_map_matches(
                payload.get("daily_sidecar_meta_files"),
                materialized.audit.daily_meta_identities,
                label="r260 materialization receipt daily metadata",
            )
            or not _content_identity_map_matches(
                payload.get("daily_sidecar_shard_files"),
                materialized.audit.daily_shard_identities,
                label="r260 materialization receipt daily shards",
            )
            or _strict_int(payload.get("joined_row_count"), label="r260 receipt row count")
            != materialized.row_count
            or payload.get("active_r241_training_eligible") is not False
        ):
            raise R260SidecarMaterializationError("r260 materialization receipt drifted")


def _validate_precommit_materialization_paths(
    *,
    materialized: MaterializationResult,
    contract: _OwnerContract,
    physical_root: Path,
    logical_root: Path,
) -> None:
    """Require staged join evidence to name only its future final paths."""

    evidence = materialized.evidence_root
    expected_dataset = _project_identity(
        materialized.joined_dataset,
        physical_root=physical_root,
        logical_root=logical_root,
    ).as_dict()
    expected_manifest = _project_identity(
        materialized.joined_manifest,
        physical_root=physical_root,
        logical_root=logical_root,
    ).as_dict()
    expected_meta = _project_daily_identities(
        materialized.audit.daily_meta_identities,
        physical_root=physical_root,
        logical_root=logical_root,
    )
    expected_shards = _project_daily_identities(
        materialized.audit.daily_shard_identities,
        physical_root=physical_root,
        logical_root=logical_root,
    )
    joined_manifest = _read_json_object(
        evidence / JOINED_DIRECTORY_NAME / JOINED_MANIFEST_NAME,
        label="r260 staged joined manifest",
    )
    if (
        joined_manifest.get("source_manifest")
        != _logical_identity_for_path(
            evidence / SOURCE_MANIFEST_COPY_NAME,
            label="r260 staged source manifest copy",
            physical_root=physical_root,
            logical_root=logical_root,
        )
        or joined_manifest.get("source_window_receipt")
        != _logical_identity_for_path(
            evidence / SOURCE_WINDOW_RECEIPT_COPY_NAME,
            label="r260 staged source-window receipt copy",
            physical_root=physical_root,
            logical_root=logical_root,
        )
        or joined_manifest.get("joined_dataset") != expected_dataset
        or joined_manifest.get("daily_sidecar_meta_files") != expected_meta
        or joined_manifest.get("daily_sidecar_shard_files") != expected_shards
        or joined_manifest.get("owner_contract_sha256") != contract.sha256
    ):
        raise R260SidecarMaterializationError(
            "staged joined manifest does not precommit final FileIdentity paths"
        )
    receipts_dir = evidence / RECEIPTS_DIRECTORY_NAME
    for name, schema in (
        (JOIN_RECEIPT_NAME, R260_JOIN_RECEIPT_SCHEMA),
        (SCHEMA_RECEIPT_NAME, R260_SCHEMA_RECEIPT_SCHEMA),
        (COUNT_RECEIPT_NAME, R260_COUNT_RECEIPT_SCHEMA),
        (DIGEST_RECEIPT_NAME, R260_DIGEST_RECEIPT_SCHEMA),
    ):
        payload = _read_receipt(receipts_dir / name, schema).payload
        if (
            payload.get("joined_dataset") != expected_dataset
            or payload.get("joined_manifest") != expected_manifest
            or payload.get("daily_sidecar_meta_files") != expected_meta
            or payload.get("daily_sidecar_shard_files") != expected_shards
        ):
            raise R260SidecarMaterializationError(
                "staged materialization receipt does not precommit final FileIdentity paths"
            )


def _validate_transport_receipt(
    receipt: "_Receipt", *, materialized: MaterializationResult, contract: _OwnerContract
) -> None:
    """Validate transport proof against the local Inzi tree by raw content."""

    payload = receipt.payload
    required = {
        "schema",
        "status",
        "receipt_sha256",
        "owner_contract_sha256",
        "transport_kind",
        "source_elmo_sidecar_root",
        "inzi_sidecar_root",
        "source_joined_dataset",
        "inzi_joined_dataset",
        "joined_dataset_byte_identical",
        "source_joined_manifest",
        "inzi_joined_manifest",
        "daily_meta_source",
        "daily_meta_inzi",
        "daily_shard_source",
        "daily_shard_inzi",
        "source_manifest_sha256",
        "source_window_receipt_sha256",
    }
    if set(payload) != required:
        raise R260SidecarMaterializationError("r260 transport receipt key shape drifted")
    if (
        payload.get("owner_contract_sha256") != contract.sha256
        or payload.get("transport_kind") != "create_only_copy"
        or payload.get("source_elmo_sidecar_root") != str(contract.side_store_root)
        or not isinstance(payload.get("inzi_sidecar_root"), str)
        or not str(payload.get("inzi_sidecar_root") or "")
        or str(payload.get("inzi_sidecar_root")).startswith(
            str(contract.side_store_root).rstrip("/") + "/"
        )
        or payload.get("source_manifest_sha256") != contract.source_manifest_sha256
        or payload.get("source_window_receipt_sha256")
        != contract.source_window_receipt_sha256
        or payload.get("joined_dataset_byte_identical") is not True
        or not _content_identity_matches(
            payload.get("source_joined_dataset"),
            materialized.joined_dataset,
            label="r260 transport source joined dataset",
        )
        or not _content_identity_matches(
            payload.get("inzi_joined_dataset"),
            materialized.joined_dataset,
            label="r260 transport Inzi joined dataset",
        )
        or not _content_identity_matches(
            payload.get("source_joined_manifest"),
            materialized.joined_manifest,
            label="r260 transport source joined manifest",
        )
        or not _content_identity_matches(
            payload.get("inzi_joined_manifest"),
            materialized.joined_manifest,
            label="r260 transport Inzi joined manifest",
        )
        or not _content_identity_map_matches(
            payload.get("daily_meta_source"),
            materialized.audit.daily_meta_identities,
            label="r260 transport source daily metadata",
        )
        or not _content_identity_map_matches(
            payload.get("daily_meta_inzi"),
            materialized.audit.daily_meta_identities,
            label="r260 transport Inzi daily metadata",
        )
        or not _content_identity_map_matches(
            payload.get("daily_shard_source"),
            materialized.audit.daily_shard_identities,
            label="r260 transport source daily shards",
        )
        or not _content_identity_map_matches(
            payload.get("daily_shard_inzi"),
            materialized.audit.daily_shard_identities,
            label="r260 transport Inzi daily shards",
        )
    ):
        raise R260SidecarMaterializationError("r260 transport receipt drifted")


def _validate_parity_receipt(
    receipt: "_Receipt", *, materialized: MaterializationResult, contract: _OwnerContract
) -> None:
    payload = receipt.payload
    if (
        payload.get("owner_contract_sha256") != contract.sha256
        or payload.get("source_manifest_sha256") != contract.source_manifest_sha256
        or payload.get("source_window_receipt_sha256") != contract.source_window_receipt_sha256
        or payload.get("record_key") != list(RECORD_KEY_FIELDS)
        or not _content_identity_matches(
            payload.get("inzi_joined_dataset"),
            materialized.joined_dataset,
            label="r260 parity Inzi dataset",
        )
        or payload.get("joined_dataset_byte_identical") is not True
        or payload.get("daily_meta_byte_identical") is not True
        or payload.get("daily_shard_byte_identical") is not True
        or payload.get("active_r241_training_eligible") is not False
    ):
        raise R260SidecarMaterializationError("r260 parity receipt drifted")
    sample_count = _strict_int(payload.get("sample_count"), label="r260 parity sample count")
    limit = _strict_int(payload.get("sample_limit"), label="r260 parity sample limit")
    if not 1 <= limit <= MAX_CAUSAL_PARITY_SAMPLES or not 1 <= sample_count <= limit:
        raise R260SidecarMaterializationError("r260 parity receipt sample bounds drifted")
    if payload.get("local_causal_projection_sha256") != payload.get(
        "remote_causal_projection_sha256"
    ):
        raise R260SidecarMaterializationError("r260 parity receipt projections differ")


@dataclass(frozen=True)
class _Receipt:
    path: Path
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, object]:
        return file_identity(self.path, label="r260 receipt").as_dict()


def _read_receipt(
    path: Path | str,
    schema: str,
    *,
    statuses: set[str] | None = None,
) -> _Receipt:
    source = _regular_file(path, label="r260 receipt")
    try:
        info = source.stat()
    except OSError as exc:  # pragma: no cover - already regular above
        raise R260SidecarMaterializationError("r260 receipt is unreadable") from exc
    if info.st_mode & 0o222:
        raise R260SidecarMaterializationError("r260 receipt is not immutable")
    payload = _read_json_object(source, label="r260 receipt")
    accepted_statuses = statuses or {"passed", "complete"}
    if (
        payload.get("schema") != schema
        or payload.get("status") not in accepted_statuses
        or payload.get("receipt_sha256") != semantic_digest(payload, field="receipt_sha256")
    ):
        raise R260SidecarMaterializationError("r260 receipt schema/status/digest drifted")
    return _Receipt(source, payload)


def _receipt_semantic_sha(receipt: _Receipt) -> str:
    result = receipt.payload.get("receipt_sha256")
    _require_sha256(result, label="r260 receipt semantic digest")
    return str(result)


def _daily_meta_semantic_digest(path: Path) -> str:
    payload = _read_json_object(path, label="r260 daily metadata")
    digest = payload.get("meta_sha256")
    _require_sha256(digest, label="r260 daily metadata semantic digest")
    return str(digest)


def _row_key(value: Mapping[str, Any], *, label: str) -> tuple[str, int, int, str]:
    episode = value.get("episode_id")
    seat = value.get("seat")
    step = value.get("env_step")
    fingerprint = value.get("observation_fingerprint")
    if not isinstance(episode, str) or not episode:
        raise R260SidecarMaterializationError(f"{label} episode_id is invalid")
    if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
        raise R260SidecarMaterializationError(f"{label} seat is invalid")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise R260SidecarMaterializationError(f"{label} env_step is invalid")
    _require_sha256(fingerprint, label=f"{label} observation_fingerprint")
    return episode, seat, step, str(fingerprint)


def _seal(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    payload = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    payload.pop(field, None)
    payload[field] = semantic_digest(payload, field=field)
    return payload


def _write_json_create_only(path: Path, payload: Mapping[str, Any], *, label: str) -> FileIdentity:
    return _write_bytes_create_only(path, canonical_json_bytes(dict(payload)), label=label)


def _write_bytes_create_only(path: Path, body: bytes, *, label: str) -> FileIdentity:
    target = Path(path)
    _ensure_directory(target.parent, label=f"{label} parent")
    if target.exists() or target.is_symlink():
        raise R260SidecarMaterializationError(f"{label} target already exists")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        _publish_temp_create_only(temporary, target, label=label)
        temporary = None
        return file_identity(target, label=label)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_regular_create_only(source: Path | str, target: Path | str, *, label: str) -> FileIdentity:
    origin = _regular_file(source, label=f"{label} source")
    destination = Path(target)
    _ensure_directory(destination.parent, label=f"{label} parent")
    if destination.exists() or destination.is_symlink():
        raise R260SidecarMaterializationError(f"{label} destination already exists")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(name)
        with origin.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=4 * 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
            os.fchmod(output_stream.fileno(), 0o444)
        _publish_temp_create_only(temporary, destination, label=label)
        temporary = None
        identity = file_identity(destination, label=label)
        original = file_identity(origin, label=f"{label} source")
        if identity.sha256 != original.sha256 or identity.size_bytes != original.size_bytes:
            raise R260SidecarMaterializationError(f"{label} byte identity drifted")
        return identity
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_temp_create_only(temporary: Path, target: Path, *, label: str) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise R260SidecarMaterializationError(f"{label} target already exists") from exc
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(target.parent)


def _new_directory(path: Path | str, *, label: str) -> Path:
    target = Path(path).expanduser()
    if target.exists() or target.is_symlink():
        raise R260SidecarMaterializationError(f"{label} must be a new directory")
    _ensure_directory(target.parent, label=f"{label} parent")
    try:
        target.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise R260SidecarMaterializationError(f"{label} already exists") from exc
    return _regular_directory(target, label=label)


def _ensure_directory(path: Path | str, *, label: str) -> Path:
    target = Path(path).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise R260SidecarMaterializationError(f"could not create {label}") from exc
    return _regular_directory(target, label=label)


def _seal_directory(path: Path | str) -> None:
    directory = _regular_directory(path, label="r260 immutable directory")
    for child in directory.iterdir():
        if child.is_symlink():
            raise R260SidecarMaterializationError("r260 immutable directory contains a symlink")
        if child.is_file():
            _make_readonly(child)
    try:
        directory.chmod(0o555)
    except OSError as exc:
        raise R260SidecarMaterializationError("could not seal r260 directory") from exc


def _make_readonly(path: Path) -> None:
    try:
        path.chmod(0o444)
    except OSError as exc:
        raise R260SidecarMaterializationError("could not seal r260 artifact") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise R260SidecarMaterializationError(f"{label} is unreadable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise R260SidecarMaterializationError(f"{label} must be a regular non-symlink file")
    return candidate.resolve()


def _regular_directory(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise R260SidecarMaterializationError(f"{label} is unreadable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise R260SidecarMaterializationError(f"{label} must be a directory without symlinks")
    return candidate.resolve()


def _read_json_object(path: Path | str, *, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label)

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise R260SidecarMaterializationError(
                    f"{label} has a duplicate JSON key: {key}"
                )
            output[key] = value
        return output

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R260SidecarMaterializationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise R260SidecarMaterializationError(f"{label} must be a JSON object")
    return value


def _identity_from_mapping(value: object, *, label: str) -> FileIdentity:
    row = _mapping(value, label=label)
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise R260SidecarMaterializationError(f"{label} must be an exact FileIdentity")
    path_text = str(row.get("path") or "")
    if not path_text:
        raise R260SidecarMaterializationError(f"{label} path is missing")
    path = Path(path_text)
    if not path.is_absolute():
        raise R260SidecarMaterializationError(f"{label} path must be absolute")
    _require_sha256(row.get("sha256"), label=f"{label} sha256")
    size = _strict_int(row.get("size_bytes"), label=f"{label} size")
    if size <= 0:
        raise R260SidecarMaterializationError(f"{label} size is invalid")
    return FileIdentity(path=path, sha256=str(row["sha256"]), size_bytes=size)


def _content_identity_matches(
    expected: object, actual: FileIdentity, *, label: str
) -> bool:
    """Compare portable file evidence without confusing host-local paths.

    The materialization receipts are byte-identical when transported from Elmo
    to Inzi, so their embedded FileIdentity paths deliberately remain the
    Elmo paths.  A remote re-audit must therefore compare raw digest and size,
    while the final Inzi aggregate independently records and rehashes its own
    local paths.  This is intentionally narrower than FileIdentity equality.
    """

    reference = _identity_from_mapping(expected, label=label)
    return reference.sha256 == actual.sha256 and reference.size_bytes == actual.size_bytes


def _content_identity_map_matches(
    expected: object,
    actual: Mapping[str, Mapping[str, object]],
    *,
    label: str,
) -> bool:
    """Compare an exact day map by content identities, never by path text."""

    try:
        reference = _mapping(expected, label=label)
        if set(reference) != set(actual):
            return False
        return all(
            _content_identity_matches(
                reference[day],
                _identity_from_mapping(actual[day], label=f"{label} {day} actual"),
                label=f"{label} {day}",
            )
            for day in sorted(actual)
        )
    except R260SidecarMaterializationError:
        return False


def _receipt_identity_root(
    *, physical_root: Path, receipt_identity_root: Path | str | None
) -> Path:
    """Return the root whose absolute paths receipts are expected to name.

    Normally this is the physical root.  Revision 262 additionally permits a
    complete, sealed staging tree to precommit receipts for its exact sibling
    final root immediately before an atomic rename.  Only paths below the
    physical tree may be projected, preserving digest and byte-size evidence.
    """

    if receipt_identity_root is None:
        return physical_root
    candidate = Path(receipt_identity_root).expanduser().resolve()
    if candidate == physical_root:
        return physical_root
    if candidate.parent != physical_root.parent:
        raise R260SidecarMaterializationError(
            "precommit receipt identity root must be a sibling of the physical root"
        )
    return candidate


def _project_identity(
    identity: FileIdentity, *, physical_root: Path, logical_root: Path
) -> FileIdentity:
    """Keep byte identity while projecting a staged path to its final sibling."""

    try:
        relative = identity.path.relative_to(physical_root)
    except ValueError as exc:
        raise R260SidecarMaterializationError(
            "receipt FileIdentity lies outside the physical sidecar root"
        ) from exc
    return FileIdentity(
        path=(logical_root / relative),
        sha256=identity.sha256,
        size_bytes=identity.size_bytes,
    )


def _project_daily_identities(
    identities: Mapping[str, Mapping[str, object]],
    *,
    physical_root: Path,
    logical_root: Path,
) -> dict[str, dict[str, object]]:
    return {
        day: _project_identity(
            _identity_from_mapping(identity, label=f"r260 daily {day} FileIdentity"),
            physical_root=physical_root,
            logical_root=logical_root,
        ).as_dict()
        for day, identity in identities.items()
    }


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R260SidecarMaterializationError(f"{label} must be an object")
    return value


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R260SidecarMaterializationError(f"{label} must be a nonnegative integer")
    return int(value)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise R260SidecarMaterializationError(f"{label} must be a SHA-256 identity")
    return text


def _require_descendant(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise R260SidecarMaterializationError(f"{label} must be below its sidecar root") from exc


def _reject_elmo_destination(path: Path, elmo_root: Path) -> None:
    raw = path.absolute()
    try:
        raw.relative_to(elmo_root)
    except ValueError:
        return
    raise R260SidecarMaterializationError("Inzi destination may not point to the Elmo side-store")


def _reject_elmo_text_path(path: str, elmo_root: str) -> None:
    if path == elmo_root or path.startswith(elmo_root.rstrip("/") + "/"):
        raise R260SidecarMaterializationError("Inzi binding may not name an Elmo side-store path")


def _configured_inzi_root(contract: _OwnerContract, attribute: str, *, label: str) -> Path:
    value = str(getattr(contract, attribute, "") or "")
    if not value:
        raise R260SidecarMaterializationError(f"{label} is absent from the owner contract")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise R260SidecarMaterializationError(f"{label} must be absolute")
    return path.resolve()


def _configured_elmo_sidecar_root(contract: _OwnerContract) -> Path:
    value = str(contract.side_store_root or "")
    path = Path(value).expanduser()
    if not value or not path.is_absolute():
        raise R260SidecarMaterializationError("r260 configured Elmo side-store root is invalid")
    return path.resolve()


def _expected_elmo_sidecar_path(
    value: Path | str, contract: _OwnerContract, *, require_directory: bool
) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise R260SidecarMaterializationError("expected Elmo sidecar root must be absolute")
    expected = _configured_elmo_sidecar_root(contract)
    resolved = candidate.resolve()
    if resolved != expected:
        raise R260SidecarMaterializationError(
            "expected Elmo sidecar root does not match the owner contract"
        )
    if require_directory:
        return _regular_directory(resolved, label="expected Elmo sidecar root")
    return resolved


def _require_final_inzi_root(root: Path, contract: _OwnerContract) -> None:
    expected = _configured_inzi_root(
        contract, "inzi_training_root", label="r260 configured Inzi final root"
    )
    if root != expected:
        raise R260SidecarMaterializationError(
            "r260 operation requires the configured final Inzi training root"
        )


def _require_staging_inzi_root(root: Path, contract: _OwnerContract) -> None:
    expected = _configured_inzi_root(
        contract, "inzi_prefix_staging_root", label="r260 configured Inzi staging root"
    )
    if root != expected:
        raise R260SidecarMaterializationError(
            "r260 operation requires the configured Inzi prefix staging root"
        )


__all__ = [
    "AGGREGATE_BINDING_NAME",
    "CompletionResult",
    "DEFAULT_CAUSAL_PARITY_SAMPLES",
    "EVIDENCE_DIRECTORY_NAME",
    "FileIdentity",
    "INZI_BINDING_NAME",
    "MaterializationResult",
    "PrefixTransferResult",
    "R260SidecarMaterializationError",
    "R260_AGGREGATE_BINDING_SCHEMA",
    "R260_INZI_BINDING_SCHEMA",
    "R260_JOINED_ROW_SCHEMA",
    "R260_PARITY_RECEIPT_SCHEMA",
    "TransportResult",
    "attest_r260_causal_local_remote_parity",
    "audit_r260_sidecar",
    "audit_r260_sidecar_prefix",
    "bind_r260_inzi_dataset",
    "canonical_json_bytes",
    "expected_days",
    "file_identity",
    "finalize_r260_inzi_sidecar",
    "materialize_r260_sidecar",
    "promote_r260_inzi_staging_root",
    "semantic_digest",
    "sha256_bytes",
    "transport_r260_sidecar_to_inzi",
    "stage_r260_sidecar_prefix_to_inzi",
]
