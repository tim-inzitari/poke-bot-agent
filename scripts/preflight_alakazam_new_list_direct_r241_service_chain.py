"""Read-only preflight for the isolated r241 Inzi terminal service chain.

Every r241 service template calls this helper before it performs its own
bounded action.  The helper has no service-manager, training, queue, upload,
or network authority.  It only admits an explicit checksum-bound activation
overlay, the immutable source snapshot it names, and the separately mounted
baseline snapshot.  In particular, it never treats a writable checkout,
ready-looking runtime registry, or a self-declared ``immutable`` field as an
activation authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTIVATION_OVERLAY_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_activation_overlay/v1"
ACTIVATION_OVERLAY_MIRROR_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirror/v1"
)
RUNTIME_REGISTRY_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241_runtime_registry/v1"
SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1"
SOURCE_STAGING_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_source_snapshot_staging/v1"
BASELINE_PAYLOAD_SNAPSHOT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_baseline_payload_snapshot/v1"
)
BASELINE_PAYLOAD_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_baseline_payload_snapshot_staging/v1"
)
CANONICAL_BASELINE_ROSTER_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_canonical_baseline_roster/v1"
)
OWNER_START_AUTHORIZATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization/v1"
)
OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization_generator/v1"
)
OVERLAY_MIRRORS_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirrors/v1"
)
CHECKPOINT_TRANSPORT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport/v1"
)
WORKER_IMAGE_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_elmo_worker_image/v1"
OWNER_CONTRACT_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241/v1"

R241_REVISION = 241
CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
RUN_NAME = "alakazam_new_list_direct_policy_r241"
REGISTRY_RELATIVE_PATH = "state/alakazam-new-list-direct-r241-runtime-registry.json"
OWNER_CONTRACT_RELATIVE_PATH = "state/alakazam-new-list-direct-policy-r241.json"
SOURCE_MANIFEST_FILENAME = "r241-source-snapshot-manifest.json"
BASELINE_MANIFEST_FILENAME = "r241-baseline-payload-manifest.json"
SOURCE_ROOT_PREFIX = "alakazam-new-list-direct-r241-src-"
BASELINE_ROOT_PREFIX = "alakazam-new-list-direct-r241-baselines-"

TRAINER_UNIT = "pokebot-alakazam-new-list-direct-r241.service"
FINALIZER_UNIT = "pokebot-alakazam-new-list-direct-r241-finalize.service"
QUEUE_UNIT = "pokebot-alakazam-new-list-direct-r241-submission-queue.service"
UPLOADER_UNIT = "pokebot-alakazam-new-list-direct-r241-upload.service"

UNIT_BY_STAGE = {
    "trainer": TRAINER_UNIT,
    "finalizer": FINALIZER_UNIT,
    "queue": QUEUE_UNIT,
    "uploader": UPLOADER_UNIT,
}
ON_SUCCESS = {
    "trainer": FINALIZER_UNIT,
    "finalizer": QUEUE_UNIT,
    "queue": UPLOADER_UNIT,
    "uploader": None,
}

# This is deliberately a much smaller set than the launcher closure.  The
# trainer's own checksum-bound ``--check`` validates the complete closure;
# these are the service-chain members that must never disappear between that
# check and this template's exec handoff.
REQUIRED_SERVICE_CHAIN_SOURCE_MEMBERS = frozenset(
    {
        "scripts/launch_alakazam_new_list_direct_r241.py",
        "scripts/preflight_alakazam_new_list_direct_r241_service_chain.py",
        "scripts/finalize_alakazam_new_list_direct_r241.py",
        "scripts/process_alakazam_new_list_direct_r241_submission_queue.py",
        "scripts/upload_alakazam_new_list_direct_r241_submission_queue.py",
        REGISTRY_RELATIVE_PATH,
        OWNER_CONTRACT_RELATIVE_PATH,
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-finalize.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-submission-queue.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-upload.service.template",
    }
)

_SHA256_PREFIX = "sha256:"


class R241ServiceChainPreflightError(RuntimeError):
    """Raised when the inert r241 chain cannot be safely armed."""


@dataclass(frozen=True)
class StaticIdentity:
    """Identity obtained only from the sealed source snapshot."""

    registry: Path
    owner_contract: Path
    owner_contract_sha256: str
    run_root: Path
    official_cg_root: Path


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return _SHA256_PREFIX + digest.hexdigest()


def _source_tree_digest(rows: Sequence[Mapping[str, object]]) -> str:
    canonical = [
        {
            "path": str(row["path"]),
            "sha256": _require_sha256(row["sha256"], label="source tree member sha"),
            "size_bytes": _exact_int(row["size_bytes"], label="source tree member size"),
        }
        for row in sorted(rows, key=lambda item: str(item["path"]))
    ]
    return _SHA256_PREFIX + hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R241ServiceChainPreflightError(f"{label} must be an object")
    return dict(value)


def _require_string(value: object, *, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise R241ServiceChainPreflightError(f"{label} is required")
    return result


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241ServiceChainPreflightError(f"{label} must be an exact integer")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    result = _require_string(value, label=label)
    if not result.startswith(_SHA256_PREFIX) or len(result) != 71:
        raise R241ServiceChainPreflightError(f"{label} must be a SHA-256 digest")
    try:
        int(result.removeprefix(_SHA256_PREFIX), 16)
    except ValueError as exc:
        raise R241ServiceChainPreflightError(f"{label} must be hexadecimal") from exc
    return result


def _absolute_path(value: object, *, label: str, reject_symlink: bool = True) -> Path:
    raw = Path(_require_string(value, label=label)).expanduser()
    if not raw.is_absolute():
        raise R241ServiceChainPreflightError(f"{label} must be absolute")
    if reject_symlink and raw.is_symlink():
        raise R241ServiceChainPreflightError(f"{label} must not be a symlink")
    return raw.resolve()


def _environment_path(name: str, *, reject_symlink: bool = True) -> Path:
    return _absolute_path(
        os.environ.get(name), label=f"environment {name}", reject_symlink=reject_symlink
    )


def _environment_sha256(name: str) -> str:
    return _require_sha256(os.environ.get(name), label=f"environment {name}")


def _regular_file(path: Path, *, label: str, readonly: bool = True) -> Path:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241ServiceChainPreflightError(f"{label} must be a regular file: {raw}")
    resolved = raw.resolve()
    if readonly and resolved.stat().st_mode & 0o222:
        raise R241ServiceChainPreflightError(f"{label} must be create-only/read-only: {resolved}")
    return resolved


def _regular_directory(path: Path, *, label: str, readonly: bool = False) -> Path:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241ServiceChainPreflightError(f"{label} must be a real directory: {raw}")
    resolved = raw.resolve()
    if readonly and resolved.stat().st_mode & 0o222:
        raise R241ServiceChainPreflightError(f"{label} must be read-only: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str, canonical: bool = False) -> dict[str, Any]:
    file_path = _regular_file(path, label=label)
    try:
        raw = file_path.read_bytes()
        decoded = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241ServiceChainPreflightError(f"{label} is not readable JSON: {file_path}") from exc
    payload = _require_mapping(decoded, label=label)
    if canonical and raw != _canonical_json(payload):
        raise R241ServiceChainPreflightError(f"{label} is not canonical create-only JSON")
    return payload


def _expect_same_path(actual: object, expected: Path, *, label: str) -> None:
    if _absolute_path(actual, label=label) != expected:
        raise R241ServiceChainPreflightError(f"{label} drifted from the immutable binding")


def _expect_same_value(actual: object, expected: str, *, label: str) -> None:
    if str(actual or "") != expected:
        raise R241ServiceChainPreflightError(f"{label} drifted from the immutable binding")


def _under_outputs(path: Path, outputs_root: Path, *, label: str) -> None:
    if not _is_within(path, outputs_root):
        raise R241ServiceChainPreflightError(
            f"{label} must be under the external r241 outputs root"
        )


def _safe_relative(value: object, *, label: str) -> Path:
    rendered = _require_string(value, label=label)
    candidate = Path(rendered)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise R241ServiceChainPreflightError(f"{label} must be a safe relative path")
    return candidate


def _snapshot_member(source_root: Path, relative: object, *, label: str) -> Path:
    candidate = source_root / _safe_relative(relative, label=label)
    if candidate.is_symlink() or not candidate.is_file():
        raise R241ServiceChainPreflightError(f"{label} is not a regular source member")
    resolved = candidate.resolve()
    if not _is_within(resolved, source_root):
        raise R241ServiceChainPreflightError(f"{label} escapes the source snapshot")
    if resolved.stat().st_mode & 0o222:
        raise R241ServiceChainPreflightError(f"{label} is writable inside the source snapshot")
    return resolved


def _expect_sha256(
    path: Path, expected: object, *, label: str, outputs_root: Path | None = None
) -> Path:
    expected_sha256 = _require_sha256(expected, label=f"{label} digest")
    resolved = _regular_file(path, label=label)
    if outputs_root is not None:
        _under_outputs(resolved, outputs_root, label=label)
    if _sha256_file(resolved) != expected_sha256:
        raise R241ServiceChainPreflightError(f"{label} checksum drifted")
    return resolved


def _separate(left: Path, right: Path, *, label: str) -> None:
    if left == right or _is_within(left, right) or _is_within(right, left):
        raise R241ServiceChainPreflightError(f"{label} must be separate")


def _require_no_inherited_overrides() -> None:
    # The launcher will install the verified r236 CG_LIB_PATH itself.  Any
    # inherited override would make the source-snapshot check non-reproducible.
    for name in (
        "CG_LIB_PATH",
        "POKEBOT_LIBCG_PATH",
        "POKEBOT_BATCH_LIBCG",
        "POKEBOT_ALLOW_ORACLE_DECK",
        "POKEBOT_BASELINES_DIR",
    ):
        if os.environ.get(name):
            raise R241ServiceChainPreflightError(
                f"service chain inherited forbidden {name}"
            )


def _validate_source_manifest(
    *, source_root: Path, manifest: Path, owner_contract_sha256: str, expected_tree_sha256: str
) -> dict[str, Any]:
    payload = _read_json(manifest, label="r241 source snapshot manifest", canonical=True)
    if (
        payload.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("owner_contract_sha256") != owner_contract_sha256
        or payload.get("source_tree_sha256") != expected_tree_sha256
        or payload.get("external_outputs_required") is not True
        or payload.get("baseline_payloads_separate_and_receipted") is not True
        or payload.get("authenticated") is not True
        or payload.get("status") != "authenticated_immutable_source_snapshot"
    ):
        raise R241ServiceChainPreflightError("r241 source manifest contract drifted")
    raw_rows = payload.get("files")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise R241ServiceChainPreflightError("r241 source manifest lacks a file inventory")

    rows: list[dict[str, object]] = []
    inventory_paths: set[str] = set()
    for raw in raw_rows:
        row = _require_mapping(raw, label="r241 source manifest inventory row")
        relative = str(row.get("path") or "")
        if relative == SOURCE_MANIFEST_FILENAME or relative in inventory_paths:
            raise R241ServiceChainPreflightError("r241 source manifest inventory is malformed")
        member = _snapshot_member(source_root, relative, label="r241 source inventory member")
        expected_sha = _require_sha256(row.get("sha256"), label=f"source member {relative}")
        expected_size = _exact_int(row.get("size_bytes"), label=f"source member {relative}")
        if expected_size < 0 or member.stat().st_size != expected_size:
            raise R241ServiceChainPreflightError("r241 source manifest member size drifted")
        if _sha256_file(member) != expected_sha:
            raise R241ServiceChainPreflightError("r241 source manifest member checksum drifted")
        inventory_paths.add(relative)
        rows.append({"path": relative, "sha256": expected_sha, "size_bytes": expected_size})
    if _source_tree_digest(rows) != expected_tree_sha256:
        raise R241ServiceChainPreflightError("r241 source manifest tree checksum drifted")
    missing = sorted(REQUIRED_SERVICE_CHAIN_SOURCE_MEMBERS - inventory_paths)
    if missing:
        raise R241ServiceChainPreflightError(
            "r241 source manifest omits service-chain closure: " + ", ".join(missing)
        )

    observed_paths: set[str] = set()
    for directory_text, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        directory = Path(directory_text)
        if directory.is_symlink() or directory.stat().st_mode & 0o222:
            raise R241ServiceChainPreflightError("r241 source snapshot contains a writable directory")
        for name in directory_names:
            child = directory / name
            if child.is_symlink() or child.stat().st_mode & 0o222:
                raise R241ServiceChainPreflightError("r241 source snapshot contains an unsafe directory")
        for name in file_names:
            child = directory / name
            relative = child.relative_to(source_root).as_posix()
            if relative == SOURCE_MANIFEST_FILENAME:
                continue
            if child.is_symlink() or not child.is_file() or child.stat().st_mode & 0o222:
                raise R241ServiceChainPreflightError("r241 source snapshot contains an unsafe file")
            observed_paths.add(relative)
    if observed_paths != inventory_paths:
        raise R241ServiceChainPreflightError(
            "r241 source snapshot has unbound or missing source members"
        )
    return payload


def _validate_static_identity(
    *, source_root: Path, manifest: Path, manifest_sha256: str, source_tree_sha256: str
) -> StaticIdentity:
    registry = _snapshot_member(
        source_root, REGISTRY_RELATIVE_PATH, label="r241 snapshot runtime registry"
    )
    if _environment_path("R241_INZI_RUNTIME_REGISTRY") != registry:
        raise R241ServiceChainPreflightError(
            "R241_INZI_RUNTIME_REGISTRY must name the pending registry inside the source snapshot"
        )
    registry_payload = _read_json(registry, label="r241 snapshot runtime registry")
    if (
        registry_payload.get("schema") != RUNTIME_REGISTRY_SCHEMA
        or _exact_int(registry_payload.get("revision"), label="snapshot registry revision")
        != R241_REVISION
    ):
        raise R241ServiceChainPreflightError("r241 snapshot runtime registry schema drifted")

    owner = _require_mapping(registry_payload.get("owner_contract"), label="owner contract")
    owner_sha256 = _require_sha256(owner.get("sha256"), label="owner contract checksum")
    if owner.get("path") != OWNER_CONTRACT_RELATIVE_PATH:
        raise R241ServiceChainPreflightError("r241 snapshot owner contract path drifted")
    owner_contract = _snapshot_member(
        source_root, OWNER_CONTRACT_RELATIVE_PATH, label="r241 snapshot owner contract"
    )
    if _sha256_file(owner_contract) != owner_sha256:
        raise R241ServiceChainPreflightError("r241 owner contract checksum drifted")
    contract = _read_json(owner_contract, label="r241 snapshot owner contract")
    if (
        contract.get("schema") != OWNER_CONTRACT_SCHEMA
        or _exact_int(contract.get("owner_decision_revision"), label="owner contract revision")
        != R241_REVISION
        or contract.get("candidate_id") != CANDIDATE_ID
        or registry_payload.get("owner_clarification_revision")
        != contract.get("latest_owner_clarification_revision")
    ):
        raise R241ServiceChainPreflightError("r241 snapshot owner contract identity drifted")

    source = _require_mapping(registry_payload.get("source_snapshot"), label="source snapshot")
    baseline = _require_mapping(registry_payload.get("baseline_payloads"), label="baseline payloads")
    run = _require_mapping(registry_payload.get("run"), label="run")
    if (
        source.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or source.get("candidate_id") != CANDIDATE_ID
        or source.get("owner_contract_sha256") != owner_sha256
        or source.get("status") != "pending_immutable_source_snapshot"
        or str(source.get("manifest_sha256") or "")
        or str(source.get("source_tree_sha256") or "")
        or baseline.get("status") != "pending_external_baseline_payload_snapshot"
        or baseline.get("source_snapshot_fallback_allowed") is not False
        or run.get("name") != RUN_NAME
        or run.get("external_activation_overlay_required") is not True
        or run.get("activation_overlay_schema") != ACTIVATION_OVERLAY_SCHEMA
    ):
        raise R241ServiceChainPreflightError(
            "r241 snapshot registry is not the required pending immutable base projection"
        )
    hosts = _require_mapping(source.get("hosts"), label="snapshot source hosts")
    if set(hosts) != {"inzi", "elmo"}:
        raise R241ServiceChainPreflightError("r241 snapshot source hosts drifted")
    inzi_source = _require_mapping(hosts.get("inzi"), label="snapshot Inzi source")
    outputs_root = _environment_path("R241_INZI_OUTPUTS_ROOT")
    _expect_same_path(inzi_source.get("outputs_root"), outputs_root, label="snapshot outputs root")

    run_root = _absolute_path(run.get("inzi_root"), label="snapshot Inzi run root")
    official = _require_mapping(registry_payload.get("official_libcg"), label="official libcg")
    official_hosts = _require_mapping(official.get("hosts"), label="official libcg hosts")
    official_inzi = _require_mapping(official_hosts.get("inzi"), label="official Inzi libcg")
    official_cg_root = _absolute_path(
        official_inzi.get("runtime_root"), label="official Inzi libcg root"
    )

    # The complete source-manifest audit follows the owner identity checks, so
    # the owner hash is derived from the sealed registry + contract rather than
    # baked into this service template.
    _validate_source_manifest(
        source_root=source_root,
        manifest=manifest,
        owner_contract_sha256=owner_sha256,
        expected_tree_sha256=source_tree_sha256,
    )
    if _sha256_file(manifest) != manifest_sha256:
        raise R241ServiceChainPreflightError("r241 source manifest checksum drifted")
    return StaticIdentity(
        registry=registry,
        owner_contract=owner_contract,
        owner_contract_sha256=owner_sha256,
        run_root=run_root,
        official_cg_root=official_cg_root,
    )


def _validate_source_overlay(
    overlay: Mapping[str, Any], *, outputs_root: Path
) -> tuple[Path, Path, str, str, StaticIdentity]:
    source_root = _regular_directory(
        _environment_path("R241_INZI_SOURCE_SNAPSHOT_ROOT"),
        label="r241 immutable source root",
        readonly=True,
    )
    manifest = _regular_file(
        _environment_path("R241_INZI_SOURCE_SNAPSHOT_MANIFEST"),
        label="r241 source manifest",
        readonly=True,
    )
    manifest_sha256 = _environment_sha256("R241_INZI_SOURCE_SNAPSHOT_MANIFEST_SHA256")
    source_tree_sha256 = _environment_sha256("R241_INZI_SOURCE_TREE_SHA256")
    if not _is_within(manifest, source_root) or manifest.name != SOURCE_MANIFEST_FILENAME:
        raise R241ServiceChainPreflightError("r241 source manifest escapes its source root")
    if not source_root.name.startswith(SOURCE_ROOT_PREFIX):
        raise R241ServiceChainPreflightError("r241 source root is not content-addressed")
    _separate(source_root, outputs_root, label="r241 source root and outputs root")

    source = _require_mapping(overlay.get("source_snapshot"), label="activation source snapshot")
    if (
        source.get("status") != "ready"
        or source.get("owner_contract_sha256") != overlay.get("owner_contract_sha256")
        or source.get("manifest_sha256") != manifest_sha256
        or source.get("source_tree_sha256") != source_tree_sha256
    ):
        raise R241ServiceChainPreflightError("activation source snapshot identity drifted")
    hosts = _require_mapping(source.get("hosts"), label="activation source hosts")
    if set(hosts) != {"inzi", "elmo"}:
        raise R241ServiceChainPreflightError("activation source hosts drifted")
    inzi = _require_mapping(hosts.get("inzi"), label="activation Inzi source")
    _expect_same_path(inzi.get("root"), source_root, label="activation Inzi source root")
    _expect_same_path(inzi.get("manifest"), manifest, label="activation Inzi source manifest")
    _expect_same_path(inzi.get("outputs_root"), outputs_root, label="activation Inzi outputs root")

    expected_root_suffix = manifest_sha256.removeprefix(_SHA256_PREFIX)[:16]
    if source_root.name != SOURCE_ROOT_PREFIX + expected_root_suffix:
        raise R241ServiceChainPreflightError("r241 source root is not manifest-content-addressed")
    _expect_sha256(manifest, manifest_sha256, label="r241 source manifest")

    staging = _expect_sha256(
        _absolute_path(inzi.get("staging_receipt"), label="activation Inzi source staging receipt"),
        inzi.get("staging_receipt_sha256"),
        label="activation Inzi source staging receipt",
        outputs_root=outputs_root,
    )
    staging_payload = _read_json(staging, label="activation Inzi source staging receipt", canonical=True)
    binding = _require_mapping(staging_payload.get("source_snapshot"), label="source staging binding")
    expected_binding = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "host": "inzi",
        "root": str(source_root),
        "source_execution_root": str(source_root),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
        "owner_contract_sha256": overlay.get("owner_contract_sha256"),
        "outputs_root": str(outputs_root),
    }
    if (
        staging_payload.get("schema") != SOURCE_STAGING_SCHEMA
        or staging_payload.get("revision") != R241_REVISION
        or staging_payload.get("candidate_id") != CANDIDATE_ID
        or staging_payload.get("status") != "passed"
        or staging_payload.get("passed") is not True
        or any(binding.get(key) != value for key, value in expected_binding.items())
    ):
        raise R241ServiceChainPreflightError("activation Inzi source staging receipt drifted")

    identity = _validate_static_identity(
        source_root=source_root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        source_tree_sha256=source_tree_sha256,
    )
    if overlay.get("owner_contract_sha256") != identity.owner_contract_sha256:
        raise R241ServiceChainPreflightError(
            "activation overlay owner contract differs from source manifest/contract"
        )
    return source_root, manifest, manifest_sha256, source_tree_sha256, identity


def _validate_baseline_overlay(
    overlay: Mapping[str, Any], *, source_root: Path, outputs_root: Path, owner_contract_sha256: str
) -> tuple[Path, Path, str, str, str, str]:
    baseline = _require_mapping(overlay.get("baseline_payloads"), label="activation baseline payloads")
    if baseline.get("status") != "ready":
        raise R241ServiceChainPreflightError("activation baseline payload is not ready")
    root = _regular_directory(
        _environment_path("R241_INZI_BASELINES_ROOT"),
        label="r241 external baseline root",
        readonly=True,
    )
    manifest = _regular_file(
        _environment_path("R241_INZI_BASELINE_PAYLOAD_MANIFEST"),
        label="r241 baseline payload manifest",
        readonly=True,
    )
    manifest_sha256 = _environment_sha256("R241_INZI_BASELINE_PAYLOAD_MANIFEST_SHA256")
    tree_sha256 = _environment_sha256("R241_INZI_BASELINE_TREE_SHA256")
    if manifest.parent != root or manifest.name != BASELINE_MANIFEST_FILENAME:
        raise R241ServiceChainPreflightError("r241 baseline manifest escapes its baseline root")
    if not root.name.startswith(BASELINE_ROOT_PREFIX):
        raise R241ServiceChainPreflightError("r241 baseline root is not content-addressed")
    _separate(root, source_root, label="r241 baseline root and source root")
    _separate(root, outputs_root, label="r241 baseline root and outputs root")

    hosts = _require_mapping(baseline.get("hosts"), label="activation baseline hosts")
    if set(hosts) != {"inzi", "elmo"}:
        raise R241ServiceChainPreflightError("activation baseline hosts drifted")
    inzi = _require_mapping(hosts.get("inzi"), label="activation Inzi baseline")
    elmo = _require_mapping(hosts.get("elmo"), label="activation Elmo baseline")
    if "canonical_roster_receipt" in baseline:
        raise R241ServiceChainPreflightError(
            "activation baseline payload may not carry an unscoped canonical-roster path"
        )
    _expect_same_path(inzi.get("root"), root, label="activation Inzi baseline root")
    _expect_same_path(inzi.get("manifest"), manifest, label="activation Inzi baseline manifest")
    _expect_same_value(
        inzi.get("manifest_sha256"), manifest_sha256, label="activation Inzi baseline manifest digest"
    )
    _expect_same_value(
        inzi.get("baseline_tree_sha256"), tree_sha256, label="activation Inzi baseline tree digest"
    )
    expected_root_suffix = manifest_sha256.removeprefix(_SHA256_PREFIX)[:16]
    if root.name != BASELINE_ROOT_PREFIX + expected_root_suffix:
        raise R241ServiceChainPreflightError("r241 baseline root is not manifest-content-addressed")
    _expect_sha256(manifest, manifest_sha256, label="r241 baseline payload manifest")
    payload = _read_json(manifest, label="r241 baseline payload manifest", canonical=True)
    if (
        payload.get("schema") != BASELINE_PAYLOAD_SNAPSHOT_SCHEMA
        or payload.get("revision") != R241_REVISION
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("owner_contract_sha256") != owner_contract_sha256
        or payload.get("baseline_tree_sha256") != tree_sha256
        or payload.get("authenticated") is not True
        or payload.get("status") != "authenticated_immutable_baseline_payload_snapshot"
    ):
        raise R241ServiceChainPreflightError("r241 baseline payload snapshot schema/identity drifted")

    canonical_path = _expect_sha256(
        _absolute_path(
            inzi.get("canonical_roster_receipt"),
            label="activation Inzi canonical baseline roster receipt",
        ),
        baseline.get("canonical_roster_receipt_sha256"),
        label="activation Inzi canonical baseline roster receipt",
        outputs_root=outputs_root,
    )
    elmo_canonical_path = _absolute_path(
        elmo.get("canonical_roster_receipt"),
        label="activation Elmo canonical baseline roster receipt",
    )
    if elmo_canonical_path == canonical_path:
        raise R241ServiceChainPreflightError(
            "activation baseline roster receipt paths must remain host-scoped"
        )
    canonical_sha256 = _require_sha256(
        baseline.get("canonical_roster_receipt_sha256"), label="canonical baseline roster digest"
    )
    canonical = _read_json(
        canonical_path, label="activation canonical baseline roster receipt", canonical=True
    )
    canonical_manifest_sha256 = _require_sha256(
        baseline.get("canonical_baseline_manifest_sha256"), label="canonical baseline manifest digest"
    )
    canonical_roster_sha256 = _require_sha256(
        baseline.get("canonical_baseline_roster_sha256"), label="canonical baseline roster digest"
    )
    if (
        canonical.get("schema") != CANONICAL_BASELINE_ROSTER_SCHEMA
        or canonical.get("revision") != R241_REVISION
        or canonical.get("candidate_id") != CANDIDATE_ID
        or canonical.get("status") != "passed"
        or canonical.get("passed") is not True
        or canonical.get("owner_contract_sha256") != owner_contract_sha256
        or canonical.get("baseline_manifest_sha256") != canonical_manifest_sha256
        or canonical.get("baseline_roster_sha256") != canonical_roster_sha256
        or not isinstance(canonical.get("baseline_roster"), list)
        or not canonical.get("baseline_roster")
    ):
        raise R241ServiceChainPreflightError("canonical baseline roster receipt drifted")

    staging = _expect_sha256(
        _absolute_path(inzi.get("staging_receipt"), label="activation Inzi baseline staging receipt"),
        inzi.get("staging_receipt_sha256"),
        label="activation Inzi baseline staging receipt",
        outputs_root=outputs_root,
    )
    staging_payload = _read_json(staging, label="activation Inzi baseline staging receipt", canonical=True)
    snapshot_binding = _require_mapping(
        staging_payload.get("baseline_payload_snapshot"), label="baseline staging snapshot binding"
    )
    canonical_binding = _require_mapping(
        staging_payload.get("canonical_roster_receipt"), label="baseline staging canonical binding"
    )
    expected_snapshot = {
        "schema": BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": "inzi",
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "baseline_tree_sha256": tree_sha256,
        "baseline_manifest_sha256": canonical_manifest_sha256,
        "baseline_roster_sha256": canonical_roster_sha256,
        "owner_contract_sha256": owner_contract_sha256,
    }
    if (
        staging_payload.get("schema") != BASELINE_PAYLOAD_STAGING_SCHEMA
        or staging_payload.get("revision") != R241_REVISION
        or staging_payload.get("candidate_id") != CANDIDATE_ID
        or staging_payload.get("status") != "passed"
        or staging_payload.get("passed") is not True
        or staging_payload.get("receipt_outside_source_and_baseline_snapshot") is not True
        or any(snapshot_binding.get(key) != value for key, value in expected_snapshot.items())
        or _absolute_path(canonical_binding.get("path"), label="baseline staging canonical receipt")
        != canonical_path
        or canonical_binding.get("sha256") != canonical_sha256
        or canonical_binding.get("baseline_manifest_sha256") != canonical_manifest_sha256
        or canonical_binding.get("baseline_roster_sha256") != canonical_roster_sha256
    ):
        raise R241ServiceChainPreflightError("activation Inzi baseline staging receipt drifted")
    return (
        root,
        manifest,
        manifest_sha256,
        tree_sha256,
        canonical_manifest_sha256,
        canonical_roster_sha256,
    )


def _validate_owner_start_authorization(
    overlay: Mapping[str, Any],
    *,
    outputs_root: Path,
    owner_contract_sha256: str,
    source_manifest_sha256: str,
    source_tree_sha256: str,
    baseline_manifest_sha256: str,
    baseline_roster_sha256: str,
) -> tuple[Path, str]:
    binding = _require_mapping(
        overlay.get("owner_start_authorization"), label="activation owner-start authorization"
    )
    if set(binding) != {"schema", "sha256", "byte_identical_mirrors_required", "hosts"}:
        raise R241ServiceChainPreflightError(
            "activation owner-start authorization has an unsupported shared-mirror shape"
        )
    if (
        binding.get("schema") != OWNER_START_AUTHORIZATION_SCHEMA
        or binding.get("byte_identical_mirrors_required") is not True
    ):
        raise R241ServiceChainPreflightError(
            "activation owner-start authorization lacks its shared-mirror contract"
        )
    authorization_sha256 = _require_sha256(
        binding.get("sha256"), label="activation owner-start authorization digest"
    )
    authorization_hosts = _require_mapping(
        binding.get("hosts"), label="activation owner-start authorization hosts"
    )
    if set(authorization_hosts) != {"inzi", "elmo"}:
        raise R241ServiceChainPreflightError(
            "activation owner-start authorization must bind Inzi and Elmo"
        )
    host_binding = _require_mapping(
        authorization_hosts.get("inzi"), label="activation Inzi owner-start authorization"
    )
    if set(host_binding) != {"path"}:
        raise R241ServiceChainPreflightError(
            "activation Inzi owner-start authorization must contain only path"
        )
    path = _expect_sha256(
        _absolute_path(host_binding.get("path"), label="r241 owner-start authorization receipt"),
        authorization_sha256,
        label="r241 owner-start authorization receipt",
        outputs_root=outputs_root,
    )
    receipt = _read_json(path, label="r241 owner-start authorization receipt", canonical=True)
    expected = {
        "schema": OWNER_START_AUTHORIZATION_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "authorized",
        "authorized": True,
        "owner_contract_sha256": owner_contract_sha256,
        "allowed_actions": ["managed_r241_training_start"],
        "source_snapshot_manifest_sha256": source_manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
        "canonical_baseline_manifest_sha256": baseline_manifest_sha256,
        "canonical_baseline_roster_sha256": baseline_roster_sha256,
        "submission_boundary": {
            "exact_count": 1,
            "checkpoint_source": "expert_before_iter_00010.pt",
            "intermediate_iteration_5_submission_allowed": False,
            "retry_copy_or_duplicate_allowed": False,
        },
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise R241ServiceChainPreflightError("r241 owner-start authorization binding drifted")
    provenance = _require_mapping(
        receipt.get("authorization_provenance"), label="r241 owner-start authorization provenance"
    )
    if provenance != {
        "schema": OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA,
        "create_only": True,
        "explicit_operator_intent": "authorize_managed_r241_training_start",
    }:
        raise R241ServiceChainPreflightError(
            "r241 owner-start authorization was not emitted by the explicit create-only generator"
        )
    return path, authorization_sha256


def _validate_activation_overlay_mirror_receipt(
    *,
    requested_path: Path,
    requested_sha256: str,
    overlay_path: Path,
    overlay_sha256: str,
    authorization_path: Path,
    authorization_sha256: str,
    outputs_root: Path,
) -> tuple[Path, str]:
    """Bind Inzi's installed copy to the one logical external overlay.

    A shared overlay digest proves the transaction identity, while this
    host-local create-only receipt proves the exact local copy and the local
    generated authorization mirror.  Neither path has an implicit default.
    """

    configured_path = _environment_path("R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT")
    configured_sha256 = _environment_sha256(
        "R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256"
    )
    if configured_path != requested_path:
        raise R241ServiceChainPreflightError("activation-overlay mirror receipt path drifted")
    if configured_sha256 != requested_sha256:
        raise R241ServiceChainPreflightError("activation-overlay mirror receipt checksum argument drifted")
    mirror_path = _expect_sha256(
        configured_path,
        requested_sha256,
        label="r241 activation-overlay mirror receipt",
        outputs_root=outputs_root,
    )
    receipt = _read_json(
        mirror_path, label="r241 activation-overlay mirror receipt", canonical=True
    )
    expected = {
        "schema": ACTIVATION_OVERLAY_MIRROR_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "host": "inzi",
        "logical_overlay": {
            "path": str(overlay_path),
            "sha256": overlay_sha256,
        },
        "owner_start_authorization": {
            "path": str(authorization_path),
            "sha256": authorization_sha256,
        },
        "outputs_root": str(outputs_root),
        "byte_identical_copy_verified": True,
    }
    if receipt != expected:
        raise R241ServiceChainPreflightError(
            "r241 activation-overlay mirror receipt does not bind this host-local byte-identical install"
        )
    return mirror_path, requested_sha256


def _validate_worker_image_overlay(
    overlay: Mapping[str, Any],
    *,
    owner_contract_sha256: str,
    source_manifest_sha256: str,
    source_tree_sha256: str,
) -> None:
    """Keep the Elmo-only image receipt checksum-bound in the shared overlay."""

    image = _require_mapping(overlay.get("worker_image"), label="activation worker image")
    allowed = {"schema", "image_id_sha256", "receipt", "source_snapshot", "tag"}
    if set(image) not in (allowed, allowed - {"tag"}):
        raise R241ServiceChainPreflightError(
            "activation worker image has an unsupported shape"
        )
    if image.get("schema") != WORKER_IMAGE_SCHEMA:
        raise R241ServiceChainPreflightError("activation worker image schema drifted")
    _require_sha256(image.get("image_id_sha256"), label="activation worker image ID")
    receipt = _require_mapping(image.get("receipt"), label="activation worker image receipt")
    if set(receipt) != {"path", "sha256"}:
        raise R241ServiceChainPreflightError(
            "activation worker image receipt has an unsupported shape"
        )
    receipt_path = _require_string(receipt.get("path"), label="activation worker image receipt path")
    if not Path(receipt_path).is_absolute():
        raise R241ServiceChainPreflightError(
            "activation worker image receipt path must be an absolute Elmo path"
        )
    _require_sha256(receipt.get("sha256"), label="activation worker image receipt digest")
    expected_source = {
        "owner_contract_sha256": owner_contract_sha256,
        "manifest_sha256": source_manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
    }
    if _require_mapping(
        image.get("source_snapshot"), label="activation worker image source snapshot"
    ) != expected_source:
        raise R241ServiceChainPreflightError(
            "activation worker image does not bind this immutable source snapshot"
        )
    if "tag" in image and not _require_string(
        image.get("tag"), label="activation worker image informational tag"
    ):
        raise R241ServiceChainPreflightError("activation worker image informational tag drifted")


def _validate_overlay_shape(
    overlay: Mapping[str, Any],
    *,
    static: StaticIdentity,
    source_manifest_sha256: str,
    source_tree_sha256: str,
) -> None:
    base = _require_mapping(overlay.get("base_registry"), label="activation base registry")
    if (
        overlay.get("schema") != ACTIVATION_OVERLAY_SCHEMA
        or overlay.get("revision") != R241_REVISION
        or overlay.get("candidate_id") != CANDIDATE_ID
        or overlay.get("status") != "ready"
        or overlay.get("passed") is not True
        or overlay.get("owner_contract_sha256") != static.owner_contract_sha256
        or base.get("path") != REGISTRY_RELATIVE_PATH
        or base.get("sha256") != _sha256_file(static.registry)
    ):
        raise R241ServiceChainPreflightError(
            "activation overlay does not bind this immutable pending registry"
        )
    mirrors = _require_mapping(overlay.get("mirrors"), label="activation overlay mirrors")
    if mirrors != {
        "schema": OVERLAY_MIRRORS_SCHEMA,
        "hosts": ["inzi", "elmo"],
        "byte_identical_required": True,
    }:
        raise R241ServiceChainPreflightError(
            "activation overlay is not the required shared Inzi/Elmo logical mirror"
        )
    preservation = _require_mapping(
        overlay.get("peak_r195_preservation"), label="activation peak-r195 preservation"
    )
    if set(preservation) != {"receipt_sha256_inzi", "receipt_sha256_elmo"}:
        raise R241ServiceChainPreflightError(
            "activation overlay may only replace peak-r195 receipt hashes"
        )
    for name, value in preservation.items():
        _require_sha256(value, label=f"activation preservation {name}")
    remote = _require_mapping(overlay.get("remote_collection"), label="activation remote collection")
    if set(remote) != {
        "endpoint_id",
        "manifest_sha256",
        "host_receipt_sha256",
        "runtime_receipt_sha256",
        "gameplay_receipt_sha256",
        "checkpoint_transport",
    }:
        raise R241ServiceChainPreflightError(
            "activation overlay may only replace remote receipt hashes"
        )
    if remote.get("endpoint_id") != "elmo-r241-official-r236-direct-policy-8767":
        raise R241ServiceChainPreflightError("activation overlay names an ineligible remote endpoint")
    for name in (
        "manifest_sha256",
        "host_receipt_sha256",
        "runtime_receipt_sha256",
        "gameplay_receipt_sha256",
    ):
        _require_sha256(remote.get(name), label=f"activation remote {name}")
    _validate_ready_checkpoint_transport(
        _require_mapping(
            remote.get("checkpoint_transport"), label="activation remote checkpoint transport"
        )
    )
    if not source_manifest_sha256:
        raise R241ServiceChainPreflightError("activation source manifest digest is missing")
    _validate_worker_image_overlay(
        overlay,
        owner_contract_sha256=static.owner_contract_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
    )


def _validate_ready_checkpoint_transport(transport: Mapping[str, Any]) -> None:
    """Validate the overlay-only promotion of Elmo's digest-addressed route.

    The staging receipt itself belongs to Elmo and can be unavailable at Inzi;
    this layer therefore binds its declared absolute location and checksum,
    while the immutable publisher/launcher validates the receipt's contents.
    """

    expected = {
        "schema": CHECKPOINT_TRANSPORT_SCHEMA,
        "status": "ready",
        "endpoint_id": "elmo-r241-official-r236-direct-policy-8767",
        "host_role": "elmo",
        "verification_endpoint": "elmo:8767",
        "verification_port": 8767,
        "container_root": "/workspace/checkpoint",
        "environment_key": "POKEBOT_REMOTE_CHECKPOINT_ROOT",
        "remote_path_prefix": "/workspace/checkpoint/",
        "content_addressing": {
            "algorithm": "sha256",
            "filename_scheme": "poke_bot.remote_jobs.digest_addressed_basename/v1",
        },
        "read_only_container_mount": True,
        "same_absolute_source_and_baseline_paths_preserved": True,
    }
    if any(transport.get(key) != value for key, value in expected.items()):
        raise R241ServiceChainPreflightError("r241 checkpoint transport identity drifted")
    host_root = _require_string(transport.get("host_root"), label="Elmo checkpoint host root")
    trainer_root = _require_string(
        transport.get("trainer_visible_root"), label="trainer-visible checkpoint root"
    )
    staging_receipt = _require_string(
        transport.get("staging_receipt"), label="Elmo checkpoint transport staging receipt"
    )
    for value, label in (
        (host_root, "Elmo checkpoint host root"),
        (trainer_root, "trainer-visible checkpoint root"),
        (staging_receipt, "Elmo checkpoint transport staging receipt"),
    ):
        candidate = Path(value)
        if not candidate.is_absolute() or value == "/workspace" or value.startswith("/workspace/"):
            raise R241ServiceChainPreflightError(f"{label} must be an external absolute path")
    _require_sha256(
        transport.get("staging_receipt_sha256"),
        label="Elmo checkpoint transport staging receipt digest",
    )
    initial = _require_mapping(
        transport.get("initial_checkpoint"), label="Elmo checkpoint transport initial checkpoint"
    )
    if set(initial) != {"container_path", "sha256"}:
        raise R241ServiceChainPreflightError(
            "Elmo checkpoint transport initial checkpoint has an unsupported shape"
        )
    checkpoint_path = _require_string(
        initial.get("container_path"), label="Elmo checkpoint transport initial checkpoint path"
    )
    if (
        not Path(checkpoint_path).is_absolute()
        or Path(checkpoint_path).parent != Path("/workspace/checkpoint")
    ):
        raise R241ServiceChainPreflightError(
            "Elmo checkpoint transport initial checkpoint is outside /workspace/checkpoint"
        )
    _require_sha256(
        initial.get("sha256"), label="Elmo checkpoint transport initial checkpoint digest"
    )


def _external_path(name: str, *, outputs_root: Path) -> Path:
    value = _environment_path(name)
    _under_outputs(value, outputs_root, label=name)
    return value


def _validate_stage_paths(
    *, stage: str, outputs_root: Path, static: StaticIdentity
) -> dict[str, Path]:
    names = (
        "RUN_ROOT",
        "RUNTIME_DIR",
        "OFFICIAL_CG_DIR",
        "MATCHUP_TREE",
        "MATCHUP_RUNTIME_ACTIVATION",
        "MODEL_RUNTIME_ACTIVATION",
        "FINALIZER_OUTPUT_DIR",
        "FINALIZER_RECEIPT",
        "QUEUE_AUTHORIZATION",
        "QUEUE",
        "QUEUE_RECEIPTS_DIR",
    )
    paths = {name.lower(): _external_path(f"R241_INZI_{name}", outputs_root=outputs_root) for name in names}
    if paths["run_root"] != static.run_root:
        raise R241ServiceChainPreflightError("r241 run root differs from the sealed source registry")
    if paths["runtime_dir"] != paths["run_root"] / "runtime":
        raise R241ServiceChainPreflightError("r241 runtime directory must remain under the isolated run root")
    if paths["official_cg_dir"] != static.official_cg_root:
        raise R241ServiceChainPreflightError("r241 official libcg root differs from the sealed source registry")
    if paths["official_cg_dir"] != paths["runtime_dir"] / "cg-r236":
        raise R241ServiceChainPreflightError("r241 official libcg root must remain in the isolated runtime")

    if stage == "trainer":
        _regular_directory(paths["run_root"], label="r241 external run root")
        _regular_directory(paths["runtime_dir"], label="r241 external runtime directory")
        _regular_directory(paths["official_cg_dir"], label="r241 official cg runtime")
    elif stage == "finalizer":
        for name in ("run_root", "runtime_dir", "official_cg_dir"):
            _regular_directory(paths[name], label=f"r241 {name}")
        for name in (
            "matchup_tree",
            "matchup_runtime_activation",
            "model_runtime_activation",
        ):
            _regular_file(paths[name], label=f"r241 {name}")
    elif stage == "queue":
        for name in ("finalizer_receipt", "queue_authorization"):
            _regular_file(paths[name], label=f"r241 {name}")
        if paths["queue"].exists() or paths["queue"].is_symlink():
            raise R241ServiceChainPreflightError(
                "r241 isolated queue must be absent before its single enqueue handoff"
            )
    else:
        for name in ("finalizer_receipt", "queue_authorization", "queue"):
            _regular_file(paths[name], label=f"r241 {name}")
    return paths


def _validate_executable(name: str, *, label: str) -> None:
    path = _environment_path(name, reject_symlink=False)
    if not path.is_file() or not path.stat().st_mode & stat.S_IXUSR:
        raise R241ServiceChainPreflightError(f"{label} must be an executable file")


def _validate_executables(*, stage: str) -> None:
    _validate_executable("R241_INZI_PYTHON", label="r241 Python")
    if stage == "uploader":
        _validate_executable("R241_INZI_KAGGLE_BIN", label="r241 Kaggle client")


def validate_activation_overlay(
    *,
    stage: str,
    overlay_path: Path,
    overlay_sha256: str,
    overlay_mirror_receipt: Path,
    overlay_mirror_receipt_sha256: str,
) -> dict[str, object]:
    """Validate one service stage without starting work or writing an artifact."""

    if stage not in UNIT_BY_STAGE:
        raise R241ServiceChainPreflightError(f"unsupported r241 service-chain stage: {stage}")
    _require_no_inherited_overrides()
    outputs_root = _regular_directory(
        _environment_path("R241_INZI_OUTPUTS_ROOT"), label="r241 external outputs root"
    )
    requested_overlay = _absolute_path(overlay_path, label="requested activation overlay")
    configured_overlay = _environment_path("R241_INZI_ACTIVATION_OVERLAY")
    if configured_overlay != requested_overlay:
        raise R241ServiceChainPreflightError("activation overlay path drifted")
    configured_sha256 = _environment_sha256("R241_INZI_ACTIVATION_OVERLAY_SHA256")
    requested_sha256 = _require_sha256(overlay_sha256, label="requested activation overlay")
    if configured_sha256 != requested_sha256:
        raise R241ServiceChainPreflightError("activation overlay checksum argument drifted")
    # Read-only permission and a self-claimed immutable flag are not enough:
    # the operator must present the exact external digest for every stage.
    checked_overlay = _expect_sha256(
        configured_overlay,
        requested_sha256,
        label="r241 external activation overlay",
        outputs_root=outputs_root,
    )
    overlay = _read_json(checked_overlay, label="r241 external activation overlay", canonical=True)
    requested_mirror_receipt = _absolute_path(
        overlay_mirror_receipt, label="requested activation-overlay mirror receipt"
    )
    requested_mirror_receipt_sha256 = _require_sha256(
        overlay_mirror_receipt_sha256,
        label="requested activation-overlay mirror receipt",
    )

    source_root, source_manifest, source_manifest_sha256, source_tree_sha256, static = (
        _validate_source_overlay(overlay, outputs_root=outputs_root)
    )
    _validate_overlay_shape(
        overlay,
        static=static,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
    )
    (
        baseline_root,
        baseline_manifest,
        baseline_manifest_sha256,
        baseline_tree_sha256,
        canonical_baseline_manifest_sha256,
        canonical_baseline_roster_sha256,
    ) = _validate_baseline_overlay(
        overlay,
        source_root=source_root,
        outputs_root=outputs_root,
        owner_contract_sha256=static.owner_contract_sha256,
    )
    authorization_path, authorization_sha256 = _validate_owner_start_authorization(
        overlay,
        outputs_root=outputs_root,
        owner_contract_sha256=static.owner_contract_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        baseline_manifest_sha256=canonical_baseline_manifest_sha256,
        baseline_roster_sha256=canonical_baseline_roster_sha256,
    )
    mirror_receipt, mirror_receipt_sha256 = _validate_activation_overlay_mirror_receipt(
        requested_path=requested_mirror_receipt,
        requested_sha256=requested_mirror_receipt_sha256,
        overlay_path=checked_overlay,
        overlay_sha256=requested_sha256,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        outputs_root=outputs_root,
    )
    paths = _validate_stage_paths(stage=stage, outputs_root=outputs_root, static=static)
    _validate_executables(stage=stage)
    return {
        "schema": "poke_bot.alakazam_new_list_direct_r241_inzi_service_chain_preflight/v1",
        "status": "passed",
        "passed": True,
        "stage": stage,
        "unit": UNIT_BY_STAGE[stage],
        "next_on_success_unit": ON_SUCCESS[stage],
        "activation_overlay": str(checked_overlay),
        "activation_overlay_sha256": requested_sha256,
        "activation_overlay_mirror_receipt": str(mirror_receipt),
        "activation_overlay_mirror_receipt_sha256": mirror_receipt_sha256,
        "owner_contract": {
            "path": str(static.owner_contract),
            "sha256": static.owner_contract_sha256,
        },
        "source_snapshot": {
            "root": str(source_root),
            "manifest": str(source_manifest),
            "manifest_sha256": source_manifest_sha256,
            "source_tree_sha256": source_tree_sha256,
        },
        "baseline_payload": {
            "root": str(baseline_root),
            "manifest": str(baseline_manifest),
            "manifest_sha256": baseline_manifest_sha256,
            "baseline_tree_sha256": baseline_tree_sha256,
        },
        "outputs_root": str(outputs_root),
        "runtime_registry": str(static.registry),
        "run_root": str(paths["run_root"]),
        "performed_service_action": False,
        "performed_training": False,
        "performed_submission": False,
        "network_io_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(UNIT_BY_STAGE), required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--overlay-sha256", required=True)
    parser.add_argument("--overlay-mirror-receipt", type=Path, required=True)
    parser.add_argument("--overlay-mirror-receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_activation_overlay(
        stage=args.stage,
        overlay_path=args.overlay,
        overlay_sha256=args.overlay_sha256,
        overlay_mirror_receipt=args.overlay_mirror_receipt,
        overlay_mirror_receipt_sha256=args.overlay_mirror_receipt_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R241ServiceChainPreflightError as exc:
        print(f"r241 Inzi service-chain preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(78)
