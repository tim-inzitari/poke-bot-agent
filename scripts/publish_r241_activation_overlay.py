#!/usr/bin/env python3
"""Create the external, receipt-bound r241 activation overlay.

The immutable r241 source snapshot deliberately contains a *pending* runtime
registry.  This publisher is the only supported way to bind that snapshot to
completed source/baseline/host receipts without editing it in place.  It does
not stage a host, start a service, collect games, or submit anything.

Run it once after the no-work preflight receipts exist.  It builds one
canonical, create-only logical overlay from locally inspectable copies of the
host receipts.  A separate mirror installer copies those exact bytes and the
host-neutral owner authorization to both hosts and re-hashes them.  Host-
specific paths remain nested in the common JSON object, so every service
consumes the same logical overlay SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import r241_baseline_payload_snapshot as baseline_payload  # noqa: E402


OVERLAY_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_activation_overlay/v1"
SOURCE_STAGING_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_source_snapshot_staging/v1"
OWNER_AUTH_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization/v1"
PRESERVATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_peak_r195_preservation/v2"
)
REMOTE_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_manifest/v1"
)
REMOTE_HOST_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_host/v1"
)
REMOTE_RUNTIME_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_runtime/v1"
)
REMOTE_GAMEPLAY_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_gameplay/v1"
)
# The four remote preflight receipts intentionally do not share one textual
# status.  Host/runtime/manifest complete their offline checks as ``passed``;
# gameplay proves the strictly no-game preflight state instead.  Keep this map
# keyed by the exact schema so a gameplay-only state cannot be replayed as a
# successful host/runtime/manifest receipt (or vice versa).
REMOTE_RECEIPT_EXPECTED_STATUS = {
    REMOTE_MANIFEST_SCHEMA: "passed",
    REMOTE_HOST_SCHEMA: "passed",
    REMOTE_RUNTIME_SCHEMA: "passed",
    REMOTE_GAMEPLAY_SCHEMA: "ready_no_games_started",
}
CHECKPOINT_TRANSPORT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport/v1"
)
CHECKPOINT_TRANSPORT_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport_staging/v1"
)
WORKER_IMAGE_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_elmo_worker_image/v1"
ELMO_ENDPOINT_ID = "elmo-r241-official-r236-direct-policy-8767"
ELMO_ENDPOINT = "elmo:8767"
ELMO_CHECKPOINT_CONTAINER_ROOT = "/workspace/checkpoint"
ELMO_CHECKPOINT_ENVIRONMENT_KEY = "POKEBOT_REMOTE_CHECKPOINT_ROOT"
ELMO_CHECKPOINT_FILENAME_SCHEME = "poke_bot.remote_jobs.digest_addressed_basename/v1"
CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
REVISION = 241
OWNER_CONTRACT_SIZE_BYTES = 14_235
REGISTRY_RELATIVE_PATH = "state/alakazam-new-list-direct-r241-runtime-registry.json"


class R241ActivationOverlayError(RuntimeError):
    """The activation transaction cannot be safely published."""


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R241ActivationOverlayError("overlay payload is not canonical JSON") from exc


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _valid_sha256(value: object, *, label: str) -> str:
    rendered = str(value or "")
    if not rendered.startswith("sha256:") or len(rendered) != 71:
        raise R241ActivationOverlayError(f"{label} lacks a canonical SHA-256")
    try:
        int(rendered.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise R241ActivationOverlayError(f"{label} is not hexadecimal") from exc
    return rendered


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241ActivationOverlayError(f"{label} must be a regular non-symlink file: {raw}")
    return raw.resolve()


def _regular_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241ActivationOverlayError(f"{label} must be a real non-symlink directory: {raw}")
    return raw.resolve()


def _read_json(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    file = _regular_file(path, label=label)
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241ActivationOverlayError(f"{label} is unreadable JSON: {file}") from exc
    if not isinstance(value, dict):
        raise R241ActivationOverlayError(f"{label} must contain a JSON object")
    return file, value


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R241ActivationOverlayError(f"{label} must be an object")
    return dict(value)


def _expect_file_sha(path: Path | str, digest: str, *, label: str) -> Path:
    file = _regular_file(path, label=label)
    expected = _valid_sha256(digest, label=label)
    actual = _sha256_file(file)
    if actual != expected:
        raise R241ActivationOverlayError(
            f"{label} checksum mismatch: expected={expected} actual={actual}"
        )
    return file


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _declared_absolute_path(value: str | Path, *, label: str) -> str:
    """Accept a host-declared path without requiring it on this controller."""

    rendered = str(value or "").strip()
    if not rendered or not Path(rendered).is_absolute():
        raise R241ActivationOverlayError(f"{label} must be an absolute host path")
    return rendered


def _create_only_json(path: Path, value: Mapping[str, object]) -> None:
    raw = path.expanduser()
    if raw.is_symlink():
        raise R241ActivationOverlayError(f"overlay output may not be a symlink: {raw}")
    parent = _regular_directory(raw.parent, label="overlay output parent")
    target = parent / raw.name
    encoded = _canonical_json(value)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
            raise R241ActivationOverlayError(
                f"overlay output already exists with different bytes: {target}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
                raise R241ActivationOverlayError(
                    f"overlay output already exists with different bytes: {target}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _pending_registry(root: Path, *, owner_contract_sha256: str) -> tuple[Path, dict[str, Any]]:
    registry_path, registry = _read_json(
        root / REGISTRY_RELATIVE_PATH, label="pending source-snapshot registry"
    )
    owner = _require_mapping(registry.get("owner_contract"), label="registry owner")
    source = _require_mapping(registry.get("source_snapshot"), label="registry source snapshot")
    baseline = _require_mapping(registry.get("baseline_payloads"), label="registry baseline payload")
    preservation = _require_mapping(
        registry.get("peak_r195_preservation"), label="registry preservation"
    )
    remote = _require_mapping(registry.get("remote_collection"), label="registry remote")
    run = _require_mapping(registry.get("run"), label="registry run")
    endpoints = remote.get("eligible_endpoints")
    endpoint = (
        dict(endpoints[0])
        if isinstance(endpoints, list) and len(endpoints) == 1 and isinstance(endpoints[0], Mapping)
        else {}
    )
    checkpoint_transport = _require_mapping(
        remote.get("checkpoint_transport"), label="registry checkpoint transport"
    )
    if (
        registry.get("schema")
        != "poke_bot.alakazam_new_list_direct_policy_r241_runtime_registry/v1"
        or registry.get("revision") != REVISION
        or owner.get("sha256") != owner_contract_sha256
        or source.get("status") != "pending_immutable_source_snapshot"
        or source.get("manifest_sha256") not in (None, "")
        or source.get("source_tree_sha256") not in (None, "")
        or baseline.get("status") != "pending_external_baseline_payload_snapshot"
        or any(
            baseline.get(key) not in (None, "")
            for key in (
                "canonical_roster_receipt",
                "canonical_roster_receipt_sha256",
                "canonical_baseline_manifest_sha256",
                "canonical_baseline_roster_sha256",
            )
        )
        or preservation.get("receipt_sha256_inzi") not in (None, "")
        or preservation.get("receipt_sha256_elmo") not in (None, "")
        or any(
            endpoint.get(key) not in (None, "")
            for key in (
                "manifest_sha256",
                "host_receipt_sha256",
                "runtime_receipt_sha256",
                "gameplay_receipt_sha256",
            )
        )
        or run.get("external_activation_overlay_required") is not True
        or run.get("activation_overlay_schema") != OVERLAY_SCHEMA
        or run.get("managed_service_start_authorized") is not False
        or run.get("submission_authorized") is not False
        or checkpoint_transport.get("schema") != CHECKPOINT_TRANSPORT_SCHEMA
        or checkpoint_transport.get("status") != "pending_external_checkpoint_transport"
        or checkpoint_transport.get("endpoint_id") != ELMO_ENDPOINT_ID
        or checkpoint_transport.get("host_role") != "elmo"
        or checkpoint_transport.get("verification_endpoint") != ELMO_ENDPOINT
        or checkpoint_transport.get("verification_port") != 8767
        or checkpoint_transport.get("host_root") not in (None, "")
        or checkpoint_transport.get("trainer_visible_root") not in (None, "")
        or checkpoint_transport.get("staging_receipt") not in (None, "")
        or checkpoint_transport.get("staging_receipt_sha256") not in (None, "")
        or _require_mapping(
            checkpoint_transport.get("initial_checkpoint"),
            label="pending checkpoint transport initial checkpoint",
        )
        != {"container_path": "", "sha256": ""}
    ):
        raise R241ActivationOverlayError(
            "source-snapshot registry is not the required pending immutable intent"
        )
    return registry_path, registry


def _source_staging(
    path: Path,
    digest: str,
    *,
    declared_receipt_path: str | Path,
    host: str,
    owner_contract_sha256: str,
) -> dict[str, Any]:
    receipt_path = _expect_file_sha(path, digest, label=f"{host} source staging receipt")
    _, receipt = _read_json(receipt_path, label=f"{host} source staging receipt")
    snapshot = _require_mapping(receipt.get("source_snapshot"), label="source staging snapshot")
    required = {
        "schema": "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1",
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "host": host,
        "owner_contract_sha256": owner_contract_sha256,
    }
    if (
        receipt.get("schema") != SOURCE_STAGING_SCHEMA
        or receipt.get("revision") != REVISION
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or any(snapshot.get(key) != value for key, value in required.items())
    ):
        raise R241ActivationOverlayError(f"{host} source staging receipt identity drifted")
    for key in (
        "root",
        "source_execution_root",
        "manifest",
        "outputs_root",
    ):
        if not str(snapshot.get(key) or "").strip() or not Path(str(snapshot[key])).is_absolute():
            raise R241ActivationOverlayError(f"{host} source staging receipt lacks {key}")
    for key in ("manifest_sha256", "source_tree_sha256", "file_inventory_sha256"):
        _valid_sha256(snapshot.get(key), label=f"{host} source staging {key}")
    return {
        "root": str(snapshot["root"]),
        "manifest": str(snapshot["manifest"]),
        "outputs_root": str(snapshot["outputs_root"]),
        "staging_receipt": _declared_absolute_path(
            declared_receipt_path, label=f"{host} declared source staging receipt"
        ),
        "staging_receipt_sha256": digest,
        "manifest_sha256": str(snapshot["manifest_sha256"]),
        "source_tree_sha256": str(snapshot["source_tree_sha256"]),
        "file_inventory_sha256": str(snapshot["file_inventory_sha256"]),
    }


def _baseline_staging(
    path: Path,
    digest: str,
    *,
    declared_receipt_path: str | Path,
    canonical_roster_receipt: tuple[Path, Mapping[str, Any]],
    declared_canonical_roster_receipt_path: str | Path,
    host: str,
    owner_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = _expect_file_sha(path, digest, label=f"{host} baseline staging receipt")
    _, receipt = _read_json(receipt_path, label=f"{host} baseline staging receipt")
    snapshot = _require_mapping(
        receipt.get("baseline_payload_snapshot"), label="baseline staging snapshot"
    )
    canonical = _require_mapping(
        receipt.get("canonical_roster_receipt"), label="baseline staging canonical roster"
    )
    if (
        receipt.get("schema") != baseline_payload.BASELINE_PAYLOAD_STAGING_SCHEMA
        or receipt.get("revision") != REVISION
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("receipt_outside_source_and_baseline_snapshot") is not True
        or snapshot.get("schema") != baseline_payload.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA
        or snapshot.get("revision") != REVISION
        or snapshot.get("candidate_id") != CANDIDATE_ID
        or snapshot.get("status") != "authenticated_immutable_baseline_payload_snapshot"
        or snapshot.get("authenticated") is not True
        or snapshot.get("host") != host
        or snapshot.get("owner_contract_sha256") != owner_contract_sha256
    ):
        raise R241ActivationOverlayError(f"{host} baseline staging receipt identity drifted")
    for key in ("root", "manifest"):
        if not str(snapshot.get(key) or "").strip() or not Path(str(snapshot[key])).is_absolute():
            raise R241ActivationOverlayError(f"{host} baseline staging receipt lacks {key}")
    for key in (
        "manifest_sha256",
        "baseline_tree_sha256",
        "baseline_manifest_sha256",
        "baseline_roster_sha256",
    ):
        _valid_sha256(snapshot.get(key), label=f"{host} baseline staging {key}")
    canonical_path, canonical_payload = canonical_roster_receipt
    declared_canonical = _declared_absolute_path(
        declared_canonical_roster_receipt_path,
        label=f"{host} declared canonical roster receipt",
    )
    if (
        str(canonical.get("path") or "") != declared_canonical
        or str(canonical.get("sha256") or "") != _sha256_file(canonical_path)
    ):
        raise R241ActivationOverlayError(
            f"{host} baseline staging receipt canonical-roster path/identity drifted"
        )
    if (
        canonical.get("baseline_manifest_sha256")
        != canonical_payload.get("baseline_manifest_sha256")
        or canonical.get("baseline_roster_sha256")
        != canonical_payload.get("baseline_roster_sha256")
        or snapshot.get("baseline_manifest_sha256")
        != canonical_payload.get("baseline_manifest_sha256")
        or snapshot.get("baseline_roster_sha256")
        != canonical_payload.get("baseline_roster_sha256")
        or snapshot.get("baseline_roster") != canonical_payload.get("baseline_roster")
    ):
        raise R241ActivationOverlayError(
            f"{host} baseline staging receipt canonical roster binding drifted"
        )
    return (
        {
            "root": str(snapshot["root"]),
            "manifest": str(snapshot["manifest"]),
            "manifest_sha256": str(snapshot["manifest_sha256"]),
            "baseline_tree_sha256": str(snapshot["baseline_tree_sha256"]),
            "staging_receipt": _declared_absolute_path(
                declared_receipt_path, label=f"{host} declared baseline staging receipt"
            ),
            "staging_receipt_sha256": digest,
            "canonical_roster_receipt": declared_canonical,
        },
        {
            "path": declared_canonical,
            "sha256": _sha256_file(canonical_path),
            "baseline_manifest_sha256": str(canonical_payload["baseline_manifest_sha256"]),
            "baseline_roster_sha256": str(canonical_payload["baseline_roster_sha256"]),
            "public_contract_sha256s": dict(canonical_payload["public_contract_sha256s"]),
        },
    )


def _passed_preservation(path: Path, digest: str, *, host: str, owner_contract_sha256: str) -> None:
    receipt_path = _expect_file_sha(path, digest, label=f"{host} peak-r195 preservation receipt")
    _, receipt = _read_json(receipt_path, label=f"{host} peak-r195 preservation receipt")
    contract = _require_mapping(receipt.get("contract"), label="preservation contract")
    contract_path = str(contract.get("path") or "").strip()
    contract_sha256 = _valid_sha256(
        contract.get("sha256"), label="preservation contract sha256"
    )
    contract_size = contract.get("size_bytes")
    if (
        set(contract) != {"path", "sha256", "size_bytes"}
        or not contract_path
        or not Path(contract_path).is_absolute()
        or isinstance(contract_size, bool)
        or not isinstance(contract_size, int)
        or contract_size != OWNER_CONTRACT_SIZE_BYTES
    ):
        raise R241ActivationOverlayError(
            "preservation contract is not a typed immutable file identity"
        )
    if (
        receipt.get("schema") != PRESERVATION_SCHEMA
        or receipt.get("revision") != REVISION
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("derived_not_self_asserted") is not True
        or contract_sha256 != owner_contract_sha256
    ):
        raise R241ActivationOverlayError(f"{host} peak-r195 preservation receipt drifted")


def _passed_remote_receipt(path: Path, digest: str, *, schema: str, label: str) -> None:
    expected_status = REMOTE_RECEIPT_EXPECTED_STATUS.get(schema)
    if expected_status is None:
        raise R241ActivationOverlayError(f"{label} has an unsupported remote receipt schema")
    receipt_path = _expect_file_sha(path, digest, label=label)
    _, receipt = _read_json(receipt_path, label=label)
    if (
        receipt.get("schema") != schema
        or receipt.get("revision") != REVISION
        or receipt.get("status") != expected_status
        or receipt.get("passed") is not True
        or receipt.get("deployment_action") != "not_started"
    ):
        raise R241ActivationOverlayError(
            f"{label} does not match its canonical r241 preflight receipt state"
        )


def _checkpoint_transport_staging(
    path: Path,
    digest: str,
    *,
    declared_receipt_path: str | Path,
    trainer_visible_root: str | Path,
    owner_contract_sha256: str,
) -> dict[str, Any]:
    """Derive the shared :8767 checkpoint transport from a local receipt copy.

    The controller need not see Elmo's native root.  It inspects a copied,
    checksum-pinned staging receipt, retains the declared Elmo path in the
    canonical overlay, and separately binds the Inzi-visible mounted view used
    by the ordinary ``RemoteJobClient`` staging code.
    """

    receipt_path = _expect_file_sha(
        path, digest, label="Elmo checkpoint transport staging receipt"
    )
    _, receipt = _read_json(
        receipt_path, label="Elmo checkpoint transport staging receipt"
    )
    transport = _require_mapping(
        receipt.get("checkpoint_transport"), label="checkpoint transport"
    )
    expected = {
        "schema": CHECKPOINT_TRANSPORT_SCHEMA,
        "endpoint_id": ELMO_ENDPOINT_ID,
        "host_role": "elmo",
        "verification_endpoint": ELMO_ENDPOINT,
        "verification_port": 8767,
        "container_root": ELMO_CHECKPOINT_CONTAINER_ROOT,
        "environment_key": ELMO_CHECKPOINT_ENVIRONMENT_KEY,
        "remote_path_prefix": f"{ELMO_CHECKPOINT_CONTAINER_ROOT}/",
        "content_addressing": {
            "algorithm": "sha256",
            "filename_scheme": ELMO_CHECKPOINT_FILENAME_SCHEME,
        },
        "read_only_container_mount": True,
        "same_absolute_source_and_baseline_paths_preserved": True,
    }
    if (
        receipt.get("schema") != CHECKPOINT_TRANSPORT_STAGING_SCHEMA
        or receipt.get("revision") != REVISION
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("owner_contract_sha256") != owner_contract_sha256
        or any(transport.get(key) != value for key, value in expected.items())
    ):
        raise R241ActivationOverlayError(
            "Elmo checkpoint transport staging receipt identity drifted"
        )
    host_root = str(transport.get("host_root") or "").strip()
    if (
        not host_root
        or not Path(host_root).is_absolute()
        or host_root == "/workspace"
        or host_root.startswith("/workspace/")
    ):
        raise R241ActivationOverlayError(
            "Elmo checkpoint transport staging receipt lacks an external host root"
        )
    visible_root = _declared_absolute_path(
        trainer_visible_root, label="trainer-visible checkpoint transport root"
    )
    if visible_root == "/workspace" or visible_root.startswith("/workspace/"):
        raise R241ActivationOverlayError(
            "trainer-visible checkpoint transport root may not be a container path"
        )
    initial = _require_mapping(
        receipt.get("initial_checkpoint"), label="checkpoint transport initial checkpoint"
    )
    if set(initial) != {"container_path", "sha256"}:
        raise R241ActivationOverlayError(
            "checkpoint transport initial checkpoint has an unsupported shape"
        )
    container_path = str(initial.get("container_path") or "").strip()
    if (
        not Path(container_path).is_absolute()
        or Path(container_path).parent != Path(ELMO_CHECKPOINT_CONTAINER_ROOT)
    ):
        raise R241ActivationOverlayError(
            "checkpoint transport initial checkpoint is outside /workspace/checkpoint"
        )
    _valid_sha256(initial.get("sha256"), label="checkpoint transport initial checkpoint")
    return {
        "schema": CHECKPOINT_TRANSPORT_SCHEMA,
        "status": "ready",
        "endpoint_id": ELMO_ENDPOINT_ID,
        "host_role": "elmo",
        "verification_endpoint": ELMO_ENDPOINT,
        "verification_port": 8767,
        "host_root": host_root,
        "trainer_visible_root": visible_root,
        "container_root": ELMO_CHECKPOINT_CONTAINER_ROOT,
        "environment_key": ELMO_CHECKPOINT_ENVIRONMENT_KEY,
        "remote_path_prefix": f"{ELMO_CHECKPOINT_CONTAINER_ROOT}/",
        "content_addressing": dict(expected["content_addressing"]),
        "read_only_container_mount": True,
        "same_absolute_source_and_baseline_paths_preserved": True,
        "staging_receipt": _declared_absolute_path(
            declared_receipt_path,
            label="declared Elmo checkpoint transport staging receipt",
        ),
        "staging_receipt_sha256": digest,
        "initial_checkpoint": {
            "container_path": container_path,
            "sha256": str(initial["sha256"]),
        },
    }


def _worker_image(
    path: Path,
    digest: str,
    *,
    declared_receipt_path: str | Path,
    owner_contract_sha256: str,
    source_manifest_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    """Bind the dedicated Elmo image receipt to this exact source snapshot.

    Docker tags are intentionally only informational.  The activation overlay
    carries the local Docker content ID and a checksum-pinned, create-only
    receipt whose source identities must equal the source snapshot that the
    overlay promotes.
    """

    receipt_path = _expect_file_sha(path, digest, label="Elmo worker image receipt")
    _, receipt = _read_json(receipt_path, label="Elmo worker image receipt")
    image = _require_mapping(receipt.get("image"), label="Elmo worker image")
    source = _require_mapping(
        receipt.get("source_snapshot"), label="Elmo worker image source snapshot"
    )
    authority = _require_mapping(
        receipt.get("activation_authority"), label="Elmo worker image authority"
    )
    smoke = _require_mapping(
        receipt.get("noncanonical_network_disabled_one_shot_smoke"),
        label="Elmo worker image no-network smoke",
    )
    expected_source = {
        "owner_contract_sha256": owner_contract_sha256,
        "manifest_sha256": source_manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
    }
    if (
        receipt.get("schema") != WORKER_IMAGE_SCHEMA
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("status") != "sealed_noncanonical_no_network_smoke_passed"
        or receipt.get("create_only") is not True
        or any(source.get(key) != value for key, value in expected_source.items())
        or authority.get("external_activation_overlay_created") is not False
        or authority.get("managed_service_start_authorized") is not False
        or authority.get("listener_started") is not False
        or authority.get("training_started") is not False
        or smoke.get("validated_external_d162") is not True
        or smoke.get("simulator_battles_started") != 0
        or smoke.get("native_function_calls") != 0
        or smoke.get("search_calls") != 0
    ):
        raise R241ActivationOverlayError("Elmo worker image receipt identity drifted")
    image_id = _valid_sha256(
        image.get("image_id_sha256"), label="Elmo worker image ID"
    )
    tag = str(image.get("tag") or "").strip()
    binding: dict[str, Any] = {
        "schema": WORKER_IMAGE_SCHEMA,
        "image_id_sha256": image_id,
        "receipt": {
            "path": _declared_absolute_path(
                declared_receipt_path, label="declared Elmo worker image receipt"
            ),
            "sha256": digest,
        },
        "source_snapshot": expected_source,
    }
    if tag:
        binding["tag"] = tag
    return binding


def _owner_start_authorization(
    path: Path,
    digest: str,
    *,
    declared_path: str | Path,
    owner_contract_sha256: str,
    source_manifest_sha256: str,
    source_tree_sha256: str,
    baseline_manifest_sha256: str,
    baseline_roster_sha256: str,
) -> dict[str, str]:
    receipt_path = _expect_file_sha(path, digest, label="owner-start authorization receipt")
    _, receipt = _read_json(receipt_path, label="owner-start authorization receipt")
    expected = {
        "schema": OWNER_AUTH_SCHEMA,
        "revision": REVISION,
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
        raise R241ActivationOverlayError("owner-start authorization receipt binding drifted")
    if receipt.get("authorization_provenance") != {
        "schema": (
            "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization_generator/v1"
        ),
        "create_only": True,
        "explicit_operator_intent": "authorize_managed_r241_training_start",
    }:
        raise R241ActivationOverlayError(
            "owner-start authorization was not emitted by the explicit create-only generator"
        )
    return {
        "path": _declared_absolute_path(
            declared_path, label="declared owner-start authorization receipt"
        ),
        "sha256": digest,
    }


def _stage(
    *,
    source_snapshot_root: Path,
    source_snapshot_manifest: Path,
    owner_contract_sha256: str,
    inzi_source_staging_receipt: Path,
    inzi_source_staging_receipt_sha256: str,
    inzi_source_staging_receipt_declared_path: str,
    elmo_source_staging_receipt: Path,
    elmo_source_staging_receipt_sha256: str,
    elmo_source_staging_receipt_declared_path: str,
    inzi_baseline_staging_receipt: Path,
    inzi_baseline_staging_receipt_sha256: str,
    inzi_baseline_staging_receipt_declared_path: str,
    elmo_baseline_staging_receipt: Path,
    elmo_baseline_staging_receipt_sha256: str,
    elmo_baseline_staging_receipt_declared_path: str,
    canonical_roster_receipt: Path,
    canonical_roster_receipt_sha256: str,
    inzi_canonical_roster_receipt_declared_path: str,
    elmo_canonical_roster_receipt_declared_path: str,
    inzi_preservation_receipt: Path,
    inzi_preservation_receipt_sha256: str,
    elmo_preservation_receipt: Path,
    elmo_preservation_receipt_sha256: str,
    remote_manifest: Path,
    remote_manifest_sha256: str,
    remote_host_receipt: Path,
    remote_host_receipt_sha256: str,
    remote_runtime_receipt: Path,
    remote_runtime_receipt_sha256: str,
    remote_gameplay_receipt: Path,
    remote_gameplay_receipt_sha256: str,
    elmo_worker_image_receipt: Path,
    elmo_worker_image_receipt_sha256: str,
    elmo_worker_image_receipt_declared_path: str,
    remote_checkpoint_transport_staging_receipt: Path,
    remote_checkpoint_transport_staging_receipt_sha256: str,
    remote_checkpoint_transport_staging_receipt_declared_path: str,
    checkpoint_transport_trainer_visible_root: str,
    owner_start_authorization: Path,
    owner_start_authorization_sha256: str,
    inzi_owner_start_authorization_declared_path: str,
    elmo_owner_start_authorization_declared_path: str,
    output: Path,
) -> dict[str, object]:
    owner_contract_sha256 = _valid_sha256(
        owner_contract_sha256, label="owner contract"
    )
    root = _regular_directory(source_snapshot_root, label="source snapshot root")
    manifest = _regular_file(source_snapshot_manifest, label="source snapshot manifest")
    if manifest.parent != root or manifest.name != "r241-source-snapshot-manifest.json":
        raise R241ActivationOverlayError("source snapshot manifest does not belong to the supplied root")
    registry_path, registry = _pending_registry(root, owner_contract_sha256=owner_contract_sha256)
    canonical_path = _expect_file_sha(
        canonical_roster_receipt,
        canonical_roster_receipt_sha256,
        label="canonical baseline roster receipt",
    )
    try:
        canonical_path, canonical_payload = baseline_payload.validate_canonical_roster_receipt(
            canonical_path,
            expected_sha256=canonical_roster_receipt_sha256,
            owner_contract_sha256=owner_contract_sha256,
        )
    except baseline_payload.R241BaselinePayloadError as exc:
        raise R241ActivationOverlayError(
            f"canonical baseline roster receipt failed validation: {exc}"
        ) from exc
    inzi_source = _source_staging(
        inzi_source_staging_receipt,
        inzi_source_staging_receipt_sha256,
        declared_receipt_path=inzi_source_staging_receipt_declared_path,
        host="inzi",
        owner_contract_sha256=owner_contract_sha256,
    )
    elmo_source = _source_staging(
        elmo_source_staging_receipt,
        elmo_source_staging_receipt_sha256,
        declared_receipt_path=elmo_source_staging_receipt_declared_path,
        host="elmo",
        owner_contract_sha256=owner_contract_sha256,
    )
    if (
        inzi_source["manifest_sha256"] != elmo_source["manifest_sha256"]
        or inzi_source["source_tree_sha256"] != elmo_source["source_tree_sha256"]
    ):
        raise R241ActivationOverlayError("Inzi and Elmo source snapshots are not byte-identical")
    if _sha256_file(manifest) != inzi_source["manifest_sha256"]:
        raise R241ActivationOverlayError(
            "controller-inspected source snapshot manifest does not match host staging receipts"
        )
    _, local_manifest = _read_json(manifest, label="controller-inspected source snapshot manifest")
    if (
        local_manifest.get("schema")
        != "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1"
        or local_manifest.get("owner_contract_sha256") != owner_contract_sha256
        or local_manifest.get("source_tree_sha256") != inzi_source["source_tree_sha256"]
    ):
        raise R241ActivationOverlayError(
            "controller-inspected source snapshot does not bind the staged owner/tree identity"
        )

    inzi_baseline, inzi_canonical = _baseline_staging(
        inzi_baseline_staging_receipt,
        inzi_baseline_staging_receipt_sha256,
        declared_receipt_path=inzi_baseline_staging_receipt_declared_path,
        canonical_roster_receipt=(canonical_path, canonical_payload),
        declared_canonical_roster_receipt_path=(
            inzi_canonical_roster_receipt_declared_path
        ),
        host="inzi",
        owner_contract_sha256=owner_contract_sha256,
    )
    elmo_baseline, elmo_canonical = _baseline_staging(
        elmo_baseline_staging_receipt,
        elmo_baseline_staging_receipt_sha256,
        declared_receipt_path=elmo_baseline_staging_receipt_declared_path,
        canonical_roster_receipt=(canonical_path, canonical_payload),
        declared_canonical_roster_receipt_path=(
            elmo_canonical_roster_receipt_declared_path
        ),
        host="elmo",
        owner_contract_sha256=owner_contract_sha256,
    )
    if (
        inzi_canonical["sha256"] != elmo_canonical["sha256"]
        or inzi_canonical["baseline_manifest_sha256"]
        != elmo_canonical["baseline_manifest_sha256"]
        or inzi_canonical["baseline_roster_sha256"]
        != elmo_canonical["baseline_roster_sha256"]
    ):
        raise R241ActivationOverlayError("Inzi and Elmo baseline receipts bind different rosters")
    _passed_preservation(
        inzi_preservation_receipt,
        inzi_preservation_receipt_sha256,
        host="inzi",
        owner_contract_sha256=owner_contract_sha256,
    )
    _passed_preservation(
        elmo_preservation_receipt,
        elmo_preservation_receipt_sha256,
        host="elmo",
        owner_contract_sha256=owner_contract_sha256,
    )
    _passed_remote_receipt(
        remote_manifest,
        remote_manifest_sha256,
        schema=REMOTE_MANIFEST_SCHEMA,
        label="Elmo remote manifest",
    )
    _passed_remote_receipt(
        remote_host_receipt,
        remote_host_receipt_sha256,
        schema=REMOTE_HOST_SCHEMA,
        label="Elmo remote host receipt",
    )
    _passed_remote_receipt(
        remote_runtime_receipt,
        remote_runtime_receipt_sha256,
        schema=REMOTE_RUNTIME_SCHEMA,
        label="Elmo remote runtime receipt",
    )
    _passed_remote_receipt(
        remote_gameplay_receipt,
        remote_gameplay_receipt_sha256,
        schema=REMOTE_GAMEPLAY_SCHEMA,
        label="Elmo remote gameplay receipt",
    )
    worker_image = _worker_image(
        elmo_worker_image_receipt,
        elmo_worker_image_receipt_sha256,
        declared_receipt_path=elmo_worker_image_receipt_declared_path,
        owner_contract_sha256=owner_contract_sha256,
        source_manifest_sha256=str(inzi_source["manifest_sha256"]),
        source_tree_sha256=str(inzi_source["source_tree_sha256"]),
    )
    checkpoint_transport = _checkpoint_transport_staging(
        remote_checkpoint_transport_staging_receipt,
        remote_checkpoint_transport_staging_receipt_sha256,
        declared_receipt_path=(
            remote_checkpoint_transport_staging_receipt_declared_path
        ),
        trainer_visible_root=checkpoint_transport_trainer_visible_root,
        owner_contract_sha256=owner_contract_sha256,
    )
    authorization = _owner_start_authorization(
        owner_start_authorization,
        owner_start_authorization_sha256,
        declared_path=inzi_owner_start_authorization_declared_path,
        owner_contract_sha256=owner_contract_sha256,
        source_manifest_sha256=str(inzi_source["manifest_sha256"]),
        source_tree_sha256=str(inzi_source["source_tree_sha256"]),
        baseline_manifest_sha256=str(inzi_canonical["baseline_manifest_sha256"]),
        baseline_roster_sha256=str(inzi_canonical["baseline_roster_sha256"]),
    )
    elmo_authorization_path = _declared_absolute_path(
        elmo_owner_start_authorization_declared_path,
        label="declared Elmo owner-start authorization receipt",
    )
    _regular_directory(output.expanduser().parent, label="canonical overlay output parent")

    source_projection = {
        "schema": "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1",
        "candidate_id": CANDIDATE_ID,
        "owner_contract_sha256": owner_contract_sha256,
        "status": "ready",
        "manifest_sha256": inzi_source["manifest_sha256"],
        "source_tree_sha256": inzi_source["source_tree_sha256"],
        "hosts": {
            "inzi": {
                key: inzi_source[key]
                for key in ("root", "manifest", "outputs_root", "staging_receipt", "staging_receipt_sha256")
            },
            "elmo": {
                key: elmo_source[key]
                for key in ("root", "manifest", "outputs_root", "staging_receipt", "staging_receipt_sha256")
            },
        },
    }
    baseline_projection = {
        "schema": baseline_payload.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA.replace(
            "_snapshot/v1", "_registry/v1"
        ),
        "candidate_id": CANDIDATE_ID,
        "status": "ready",
        "separately_mounted_and_receipted": True,
        "source_snapshot_fallback_allowed": False,
        "canonical_roster_receipt_sha256": inzi_canonical["sha256"],
        "canonical_baseline_manifest_sha256": inzi_canonical["baseline_manifest_sha256"],
        "canonical_baseline_roster_sha256": inzi_canonical["baseline_roster_sha256"],
        "hosts": {"inzi": inzi_baseline, "elmo": elmo_baseline},
    }
    # The schema transformation above is intentionally explicit in the output
    # rather than copied from a mutable registry.  Keep the literal assertion
    # next to it so a future snapshot-schema rename cannot quietly publish a
    # different runtime registry schema.
    if baseline_projection["schema"] != (
        "poke_bot.alakazam_new_list_direct_r241_baseline_payload_registry/v1"
    ):
        raise R241ActivationOverlayError("baseline payload registry schema derivation drifted")
    overlay = {
        "schema": OVERLAY_SCHEMA,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "ready",
        "passed": True,
        "owner_contract_sha256": owner_contract_sha256,
        "base_registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "sha256": _sha256_file(registry_path),
        },
        "source_snapshot": source_projection,
        "baseline_payloads": baseline_projection,
        "peak_r195_preservation": {
            "receipt_sha256_inzi": inzi_preservation_receipt_sha256,
            "receipt_sha256_elmo": elmo_preservation_receipt_sha256,
        },
        "remote_collection": {
            "endpoint_id": "elmo-r241-official-r236-direct-policy-8767",
            "manifest_sha256": remote_manifest_sha256,
            "host_receipt_sha256": remote_host_receipt_sha256,
            "runtime_receipt_sha256": remote_runtime_receipt_sha256,
            "gameplay_receipt_sha256": remote_gameplay_receipt_sha256,
            "checkpoint_transport": checkpoint_transport,
        },
        "worker_image": worker_image,
        "mirrors": {
            "schema": "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirrors/v1",
            "hosts": ["inzi", "elmo"],
            "byte_identical_required": True,
        },
        "owner_start_authorization": {
            "schema": OWNER_AUTH_SCHEMA,
            "sha256": authorization["sha256"],
            "byte_identical_mirrors_required": True,
            "hosts": {
                "inzi": {"path": authorization["path"]},
                "elmo": {"path": elmo_authorization_path},
            },
        },
    }
    _create_only_json(output, overlay)
    output_path = _regular_file(output, label="published canonical activation overlay")
    return {
        "overlay_path": str(output_path),
        "overlay_sha256": _sha256_file(output_path),
        **overlay,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot-root", type=Path, required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--owner-contract-sha256", required=True)
    for name in ("inzi-source", "elmo-source", "inzi-baseline", "elmo-baseline"):
        parser.add_argument(f"--{name}-staging-receipt", type=Path, required=True)
        parser.add_argument(f"--{name}-staging-receipt-sha256", required=True)
        parser.add_argument(f"--{name}-staging-receipt-declared-path", required=True)
    parser.add_argument("--canonical-roster-receipt", type=Path, required=True)
    parser.add_argument("--canonical-roster-receipt-sha256", required=True)
    for name in ("inzi", "elmo"):
        parser.add_argument(
            f"--{name}-canonical-roster-receipt-declared-path", required=True
        )
    for name in ("inzi", "elmo"):
        parser.add_argument(f"--{name}-preservation-receipt", type=Path, required=True)
        parser.add_argument(f"--{name}-preservation-receipt-sha256", required=True)
    for name in ("remote-manifest", "remote-host", "remote-runtime", "remote-gameplay"):
        parser.add_argument(f"--{name}-receipt", type=Path, required=True)
        parser.add_argument(f"--{name}-receipt-sha256", required=True)
    parser.add_argument("--elmo-worker-image-receipt", type=Path, required=True)
    parser.add_argument("--elmo-worker-image-receipt-sha256", required=True)
    parser.add_argument("--elmo-worker-image-receipt-declared-path", required=True)
    parser.add_argument(
        "--remote-checkpoint-transport-staging-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--remote-checkpoint-transport-staging-receipt-sha256", required=True
    )
    parser.add_argument(
        "--remote-checkpoint-transport-staging-receipt-declared-path",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-transport-trainer-visible-root",
        required=True,
        help=(
            "absolute Inzi-visible mount of the receipt-bound Elmo checkpoint "
            "transport root"
        ),
    )
    parser.add_argument("--owner-start-authorization", type=Path, required=True)
    parser.add_argument("--owner-start-authorization-sha256", required=True)
    for name in ("inzi", "elmo"):
        parser.add_argument(
            f"--{name}-owner-start-authorization-declared-path", required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _stage(
        source_snapshot_root=args.source_snapshot_root,
        source_snapshot_manifest=args.source_snapshot_manifest,
        owner_contract_sha256=args.owner_contract_sha256,
        inzi_source_staging_receipt=args.inzi_source_staging_receipt,
        inzi_source_staging_receipt_sha256=args.inzi_source_staging_receipt_sha256,
        inzi_source_staging_receipt_declared_path=(
            args.inzi_source_staging_receipt_declared_path
        ),
        elmo_source_staging_receipt=args.elmo_source_staging_receipt,
        elmo_source_staging_receipt_sha256=args.elmo_source_staging_receipt_sha256,
        elmo_source_staging_receipt_declared_path=(
            args.elmo_source_staging_receipt_declared_path
        ),
        inzi_baseline_staging_receipt=args.inzi_baseline_staging_receipt,
        inzi_baseline_staging_receipt_sha256=args.inzi_baseline_staging_receipt_sha256,
        inzi_baseline_staging_receipt_declared_path=(
            args.inzi_baseline_staging_receipt_declared_path
        ),
        elmo_baseline_staging_receipt=args.elmo_baseline_staging_receipt,
        elmo_baseline_staging_receipt_sha256=args.elmo_baseline_staging_receipt_sha256,
        elmo_baseline_staging_receipt_declared_path=(
            args.elmo_baseline_staging_receipt_declared_path
        ),
        canonical_roster_receipt=args.canonical_roster_receipt,
        canonical_roster_receipt_sha256=args.canonical_roster_receipt_sha256,
        inzi_canonical_roster_receipt_declared_path=(
            args.inzi_canonical_roster_receipt_declared_path
        ),
        elmo_canonical_roster_receipt_declared_path=(
            args.elmo_canonical_roster_receipt_declared_path
        ),
        inzi_preservation_receipt=args.inzi_preservation_receipt,
        inzi_preservation_receipt_sha256=args.inzi_preservation_receipt_sha256,
        elmo_preservation_receipt=args.elmo_preservation_receipt,
        elmo_preservation_receipt_sha256=args.elmo_preservation_receipt_sha256,
        remote_manifest=args.remote_manifest_receipt,
        remote_manifest_sha256=args.remote_manifest_receipt_sha256,
        remote_host_receipt=args.remote_host_receipt,
        remote_host_receipt_sha256=args.remote_host_receipt_sha256,
        remote_runtime_receipt=args.remote_runtime_receipt,
        remote_runtime_receipt_sha256=args.remote_runtime_receipt_sha256,
        remote_gameplay_receipt=args.remote_gameplay_receipt,
        remote_gameplay_receipt_sha256=args.remote_gameplay_receipt_sha256,
        elmo_worker_image_receipt=args.elmo_worker_image_receipt,
        elmo_worker_image_receipt_sha256=args.elmo_worker_image_receipt_sha256,
        elmo_worker_image_receipt_declared_path=(
            args.elmo_worker_image_receipt_declared_path
        ),
        remote_checkpoint_transport_staging_receipt=(
            args.remote_checkpoint_transport_staging_receipt
        ),
        remote_checkpoint_transport_staging_receipt_sha256=(
            args.remote_checkpoint_transport_staging_receipt_sha256
        ),
        remote_checkpoint_transport_staging_receipt_declared_path=(
            args.remote_checkpoint_transport_staging_receipt_declared_path
        ),
        checkpoint_transport_trainer_visible_root=(
            args.checkpoint_transport_trainer_visible_root
        ),
        owner_start_authorization=args.owner_start_authorization,
        owner_start_authorization_sha256=args.owner_start_authorization_sha256,
        inzi_owner_start_authorization_declared_path=(
            args.inzi_owner_start_authorization_declared_path
        ),
        elmo_owner_start_authorization_declared_path=(
            args.elmo_owner_start_authorization_declared_path
        ),
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R241ActivationOverlayError as exc:
        print(f"r241 activation-overlay publication failed: {exc}", file=sys.stderr)
        raise SystemExit(78)
