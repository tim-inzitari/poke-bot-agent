"""Fail-closed ctypes support for r198 true-RNG pairing snapshots.

This adapter is intentionally separate from the ordinary competition ``cg``
path.  It talks only to the private, evaluation-only native extension declared
in :mod:`engine_patches/RtpPairingSnapshotExport.cpp`.  A caller cannot turn a
requested seed into a pairing claim: native capture accepts only the versioned
post-start snapshot boundary with ``deviceRand=false`` and later restores fresh
``ApiData`` objects for the A/B/C arms.

The module never selects a production runtime, starts a service, or packages a
competition artifact.  Its receipts are evidence for a later, separate review.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SNAPSHOT_ABI_NAME = "poke_bot.rtp_pairing_snapshot_abi"
SNAPSHOT_ABI_VERSION = 2
SNAPSHOT_CAPTURE_BOUNDARY = "post_battle_start_first_external_selection"
SNAPSHOT_BOUNDARY_TAG = 1
CAPABILITY_SCHEMA = "poke_bot.recursive_turn_planner.true_rng_pairing_capability/v2"
PROBE_SCHEMA = "poke_bot.recursive_turn_planner.true_rng_pairing_probe/v1"
BUILD_SCHEMA = "poke_bot.recursive_turn_planner.true_rng_pairing_build/v1"
SNAPSHOT_SEAL_SCHEMA = "poke_bot.recursive_turn_planner.true_rng_pairing_snapshot_seal/v1"
SUPPORTED_RNG_KINDS = ("snapshot",)


class RTPPairingSnapshotError(RuntimeError):
    """The native snapshot ABI or its evidence was not trustworthy."""


_INITIALIZED_LIBRARY_PATHS: set[str] = set()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    """Make an absolute path without resolving symlinks or hiding ``..``."""

    raw = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    return Path(raw)


def _reject_symlink_components(path: str | Path, *, label: str) -> Path:
    """Reject every existing lexical component that is a symlink.

    This runs before ``resolve``/``stat`` so a caller cannot smuggle a source,
    engine, receipt, or sealed snapshot through a symlinked ancestor.  We walk
    literal ``..`` components too: normalizing first could otherwise erase a
    symlink traversal from the audit path.
    """

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        if component in ("", "."):
            continue
        if component == "..":
            current = current.parent
            continue
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RTPPairingSnapshotError(
                f"cannot inspect {label} path component: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RTPPairingSnapshotError(f"{label} traverses a symlink: {current}")
    return absolute


def _existing_regular_file(path: str | Path, *, label: str) -> Path:
    lexical = _reject_symlink_components(path, label=label)
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise RTPPairingSnapshotError(f"cannot access {label}: {lexical}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RTPPairingSnapshotError(f"{label} is not a regular file: {resolved}")
    return resolved


def _output_path_without_symlinks(path: str | Path, *, label: str) -> Path:
    lexical = _reject_symlink_components(path, label=label)
    try:
        lexical.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RTPPairingSnapshotError(f"cannot create {label} parent: {lexical.parent}") from exc
    _reject_symlink_components(lexical.parent, label=f"{label} parent")
    try:
        target_metadata = os.lstat(lexical)
    except FileNotFoundError:
        target_metadata = None
    except OSError as exc:
        raise RTPPairingSnapshotError(f"cannot inspect {label}: {lexical}") from exc
    if target_metadata is not None and stat.S_ISLNK(target_metadata.st_mode):
        raise RTPPairingSnapshotError(f"{label} target is a symlink: {lexical}")
    # The parent was just checked lexically and has no symlink components, so
    # resolving only it gives a stable absolute target without following a
    # user-controlled target link.
    return lexical.parent.resolve(strict=True) / lexical.name


def file_digest(path: str | Path) -> str:
    source = _existing_regular_file(path, label="artifact")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _valid_digest(value: Any, *, label: str) -> str:
    result = str(value or "")
    if not result.startswith("sha256:") or len(result) != 71:
        raise RTPPairingSnapshotError(f"{label} must be a SHA-256 digest")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise RTPPairingSnapshotError(f"{label} must be a SHA-256 digest") from exc
    return result


def frozen_file_identity(path: str | Path) -> dict[str, Any]:
    source = _existing_regular_file(path, label="frozen artifact")
    metadata = source.stat(follow_symlinks=False)
    return {
        "path": str(source),
        "sha256": file_digest(source),
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def verify_frozen_file_identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RTPPairingSnapshotError(f"{label} must be a file identity")
    path = _existing_regular_file(str(value.get("path") or ""), label=label)
    expected = _valid_digest(value.get("sha256"), label=f"{label}.sha256")
    observed = frozen_file_identity(path)
    if observed["sha256"] != expected:
        raise RTPPairingSnapshotError(
            f"{label} checksum mismatch: expected {expected}, got {observed['sha256']}"
        )
    declared_bytes = value.get("bytes")
    if declared_bytes is not None and (
        isinstance(declared_bytes, bool) or int(declared_bytes) != observed["bytes"]
    ):
        raise RTPPairingSnapshotError(f"{label}.bytes does not match the file")
    declared_mode = value.get("mode")
    if declared_mode is not None and int(declared_mode) != observed["mode"]:
        raise RTPPairingSnapshotError(f"{label}.mode does not match the file")
    return observed


def snapshot_abi_contract() -> dict[str, Any]:
    """The canonical ABI shape that build/probe/capability records must bind."""

    return {
        "name": SNAPSHOT_ABI_NAME,
        "version": SNAPSHOT_ABI_VERSION,
        "initialize_symbol": "RtpPairingSnapshotInitialize",
        "start_symbol": "RtpPairingBattleStartSeededOut",
        "observation_symbol": "RtpPairingSnapshotGetBattleJsonOut",
        "capture_symbol": "RtpPairingSnapshotCapture",
        "restore_symbol": "RtpPairingSnapshotRestore",
        "restore_serialized_symbol": "RtpPairingSnapshotRestoreSerialized",
        "release_symbol": "RtpPairingSnapshotRelease",
        "serialized_size_symbol": "RtpPairingSnapshotSerializedSize",
        "serialize_symbol": "RtpPairingSnapshotSerialize",
        "fingerprint_size_symbol": "RtpPairingSnapshotFingerprintSize",
        "fingerprint_symbol": "RtpPairingSnapshotFingerprint",
        "last_error_symbol": "RtpPairingSnapshotLastError",
        "requires_device_rand_false": True,
        "requires_time_limit_zero": True,
        "capture_boundary": SNAPSHOT_CAPTURE_BOUNDARY,
        "boundary_tag": SNAPSHOT_BOUNDARY_TAG,
        "full_state_game_rng_config_counters": True,
        "serialization_compatibility": "exact_engine_artifact_only",
        "requires_pristine_process_initialization": True,
        "serialized_restore_requires_sealed_sha256": True,
        "serialized_restore_sealed_sha256_bytes": 32,
        "observation_lifetime": "until_next_mutation_or_battle_finish",
    }


def snapshot_abi_sha256() -> str:
    return canonical_digest(snapshot_abi_contract())


@dataclass(frozen=True)
class PairingArtifactSet:
    """Hash-verified evidence needed to attest one private native build."""

    engine_artifact: Mapping[str, Any]
    source_artifact: Mapping[str, Any]
    patch_artifact: Mapping[str, Any]
    build_artifact: Mapping[str, Any]

    @classmethod
    def from_paths(
        cls,
        *,
        engine_path: str | Path,
        source_manifest_path: str | Path,
        patch_path: str | Path,
        build_receipt_path: str | Path,
    ) -> "PairingArtifactSet":
        return cls(
            engine_artifact=frozen_file_identity(engine_path),
            source_artifact=frozen_file_identity(source_manifest_path),
            patch_artifact=frozen_file_identity(patch_path),
            build_artifact=frozen_file_identity(build_receipt_path),
        )

    def verified(self) -> "PairingArtifactSet":
        return PairingArtifactSet(
            engine_artifact=verify_frozen_file_identity(
                self.engine_artifact, label="engine_artifact"
            ),
            source_artifact=verify_frozen_file_identity(
                self.source_artifact, label="source_artifact"
            ),
            patch_artifact=verify_frozen_file_identity(
                self.patch_artifact, label="patch_artifact"
            ),
            build_artifact=verify_frozen_file_identity(
                self.build_artifact, label="build_artifact"
            ),
        )


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _existing_regular_file(path, label=label)
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RTPPairingSnapshotError(f"cannot read {label}: {source}") from exc
    if not isinstance(loaded, Mapping):
        raise RTPPairingSnapshotError(f"{label} must be a JSON object")
    return dict(loaded)


def verify_build_receipt(artifacts: PairingArtifactSet) -> PairingArtifactSet:
    """Rehash all build inputs and cross-bind the immutable build receipt."""

    checked = artifacts.verified()
    receipt = _read_json(checked.build_artifact["path"], label="build receipt")
    if receipt.get("schema") != BUILD_SCHEMA or receipt.get("status") != "success":
        raise RTPPairingSnapshotError("build receipt is not a successful pairing build")
    expected = {
        "engine_artifact_sha256": checked.engine_artifact["sha256"],
        "source_artifact_sha256": checked.source_artifact["sha256"],
        "patch_artifact_sha256": checked.patch_artifact["sha256"],
        "canonical_abi_sha256": snapshot_abi_sha256(),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RTPPairingSnapshotError(f"build receipt does not bind {key}")
    required_modes = {
        "engine_artifact": 0o555,
        "source_artifact": 0o444,
        "patch_artifact": 0o444,
        "build_artifact": 0o444,
    }
    for name, mode in required_modes.items():
        identity = getattr(checked, name)
        if identity.get("mode") != mode:
            raise RTPPairingSnapshotError(
                f"{name} must be immutable mode {mode:04o}"
            )
    for name in ("engine_artifact", "source_artifact", "patch_artifact"):
        if receipt.get(name) != dict(getattr(checked, name)):
            raise RTPPairingSnapshotError(
                f"build receipt does not bind the full {name} identity"
            )
    return checked


def _immutable_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish a 0444 JSON record once, never replacing mismatched evidence."""

    target = _output_path_without_symlinks(path, label="immutable record")
    material = dict(payload)
    identity = canonical_digest(
        {key: value for key, value in material.items() if key != "created_at_utc"}
    )
    material["record_sha256"] = identity
    encoded = json.dumps(material, sort_keys=True, indent=2) + "\n"
    if target.exists():
        existing_path = _existing_regular_file(target, label="existing immutable record")
        if stat.S_IMODE(existing_path.stat(follow_symlinks=False).st_mode) != 0o444:
            raise RTPPairingSnapshotError(
                f"existing immutable record is not mode 0444: {existing_path}"
            )
        existing = _read_json(target, label="existing immutable record")
        if existing.get("record_sha256") != identity:
            raise RTPPairingSnapshotError(
                f"immutable record already exists with different content: {target}"
            )
        return target
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing_path = _existing_regular_file(target, label="racing immutable record")
            if stat.S_IMODE(existing_path.stat(follow_symlinks=False).st_mode) != 0o444:
                raise RTPPairingSnapshotError(
                    f"racing immutable record is not mode 0444: {existing_path}"
                )
            existing = _read_json(target, label="racing immutable record")
            if existing.get("record_sha256") != identity:
                raise RTPPairingSnapshotError(
                    f"immutable record appeared with different content: {target}"
                )
        _existing_regular_file(target, label="published immutable record")
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_pairing_case_binding(
    case_binding_artifact: Mapping[str, Any], *, expected_debug_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the immutable per-cell identity carried by a snapshot seal."""

    identity = verify_frozen_file_identity(
        case_binding_artifact, label="pairing case binding"
    )
    if identity["mode"] != 0o444:
        raise RTPPairingSnapshotError("pairing case binding must be immutable mode 0444")
    binding = _read_json(identity["path"], label="pairing case binding")
    if binding.get("schema") != "poke_bot.recursive_turn_planner.r198_pairing_case_binding/v1":
        raise RTPPairingSnapshotError("pairing case binding has the wrong schema")
    if binding.get("status") != "sealed":
        raise RTPPairingSnapshotError("pairing case binding is not sealed")
    for key in ("cell_id", "case_id", "opponent_id"):
        if not isinstance(binding.get(key), str) or not binding[key]:
            raise RTPPairingSnapshotError(f"pairing case binding lacks {key}")
    for key, minimum, maximum in (("seat", 0, 1), ("replicate", 0, 2**31 - 1)):
        value = binding.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise RTPPairingSnapshotError(f"pairing case binding has invalid {key}")
    debug_seed = binding.get("debug_seed")
    if (
        isinstance(debug_seed, bool)
        or not isinstance(debug_seed, int)
        or not 0 <= debug_seed <= 0xFFFFFFFF
        or debug_seed != int(expected_debug_seed)
    ):
        raise RTPPairingSnapshotError("pairing case binding debug_seed does not match capture")
    decks = binding.get("ordered_deck_identities")
    if not isinstance(decks, list) or len(decks) != 2:
        raise RTPPairingSnapshotError(
            "pairing case binding requires exactly two ordered deck identities"
        )
    for index, deck in enumerate(decks):
        if not isinstance(deck, Mapping):
            raise RTPPairingSnapshotError(
                f"pairing case binding deck identity {index} is invalid"
            )
        _valid_digest(deck.get("sha256"), label=f"ordered_deck_identities[{index}].sha256")
    for key in ("cohort_identity", "source_exclusion_identity"):
        value = binding.get(key)
        if not isinstance(value, Mapping):
            raise RTPPairingSnapshotError(f"pairing case binding lacks {key}")
        _valid_digest(value.get("sha256"), label=f"{key}.sha256")
    return identity, binding


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def emit_true_rng_pairing_probe(
    *,
    output_path: str | Path,
    artifacts: PairingArtifactSet,
    deterministic_probe: Mapping[str, Any],
    divergent_policy_true_pairing_passed: bool,
    all_arms_restored_or_replayed: bool,
) -> Path:
    """Write the immutable duplicate-restore probe required by capability v2."""

    checked = verify_build_receipt(artifacts)
    probe = dict(deterministic_probe)
    if probe.get("passed") is not True:
        raise RTPPairingSnapshotError("duplicate-restore deterministic probe did not pass")
    required_probe_fields = (
        "initial_snapshot_fingerprint_sha256",
        "initial_snapshot_fingerprint_bytes",
        "deterministic_transcript_sha256",
        "transcript_steps",
        "duplicate_restore_independent_handles",
        "device_rand_false_verified",
        "requested_seed_only_rejected",
        "delayed_restore_transcript_passed",
        "cross_process_restore_passed",
    )
    for key in required_probe_fields:
        if key not in probe:
            raise RTPPairingSnapshotError(f"deterministic probe lacks {key}")
    if probe["duplicate_restore_independent_handles"] is not True:
        raise RTPPairingSnapshotError("duplicate restores did not have independent handles")
    if probe["device_rand_false_verified"] is not True:
        raise RTPPairingSnapshotError("probe did not verify deviceRand=false")
    if probe["requested_seed_only_rejected"] is not True:
        raise RTPPairingSnapshotError("probe did not reject seed-only pairing")
    if probe["delayed_restore_transcript_passed"] is not True:
        raise RTPPairingSnapshotError("probe did not pass a delayed-restore transcript")
    if probe["cross_process_restore_passed"] is not True:
        raise RTPPairingSnapshotError("probe did not pass cross-process restore")
    _valid_digest(
        probe["initial_snapshot_fingerprint_sha256"],
        label="initial_snapshot_fingerprint_sha256",
    )
    _valid_digest(
        probe["deterministic_transcript_sha256"],
        label="deterministic_transcript_sha256",
    )
    if int(probe["initial_snapshot_fingerprint_bytes"]) < 1:
        raise RTPPairingSnapshotError("snapshot fingerprint must have bytes")
    if int(probe["transcript_steps"]) < 1:
        raise RTPPairingSnapshotError("probe must compare at least one transcript step")
    payload = {
        "schema": PROBE_SCHEMA,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "engine_artifact_sha256": checked.engine_artifact["sha256"],
        "source_artifact_sha256": checked.source_artifact["sha256"],
        "patch_artifact_sha256": checked.patch_artifact["sha256"],
        "build_artifact_sha256": checked.build_artifact["sha256"],
        "canonical_abi_sha256": snapshot_abi_sha256(),
        "verified_rng_kinds": list(SUPPORTED_RNG_KINDS),
        "device_rand_false_verified": True,
        "requested_seed_only_rejected": True,
        "duplicate_restore_independent_handles": True,
        "delayed_restore_transcript_passed": True,
        "cross_process_restore_passed": True,
        "all_arms_restored_or_replayed": bool(all_arms_restored_or_replayed),
        "divergent_policy_true_pairing_passed": bool(
            divergent_policy_true_pairing_passed
        ),
        "deterministic_restore_probe": probe,
    }
    if payload["all_arms_restored_or_replayed"] is not True:
        raise RTPPairingSnapshotError("A/B/C restore proof is required")
    if payload["divergent_policy_true_pairing_passed"] is not True:
        raise RTPPairingSnapshotError("divergent-policy fresh-restore proof is required")
    return _immutable_json(output_path, payload)


def emit_true_rng_pairing_capability(
    *,
    output_path: str | Path,
    artifacts: PairingArtifactSet,
    probe_path: str | Path,
) -> Path:
    """Emit a capability v2 record only after all native evidence cross-binds."""

    checked = verify_build_receipt(artifacts)
    probe_artifact = frozen_file_identity(probe_path)
    probe = _read_json(probe_artifact["path"], label="true-RNG pairing probe")
    if probe.get("schema") != PROBE_SCHEMA or probe.get("status") != "passed":
        raise RTPPairingSnapshotError("true-RNG pairing probe did not pass")
    required = {
        "engine_artifact_sha256": checked.engine_artifact["sha256"],
        "source_artifact_sha256": checked.source_artifact["sha256"],
        "patch_artifact_sha256": checked.patch_artifact["sha256"],
        "build_artifact_sha256": checked.build_artifact["sha256"],
        "canonical_abi_sha256": snapshot_abi_sha256(),
    }
    for key, expected in required.items():
        if probe.get(key) != expected:
            raise RTPPairingSnapshotError(f"true-RNG probe does not bind {key}")
    for key in (
        "device_rand_false_verified",
        "requested_seed_only_rejected",
        "duplicate_restore_independent_handles",
        "delayed_restore_transcript_passed",
        "cross_process_restore_passed",
        "all_arms_restored_or_replayed",
        "divergent_policy_true_pairing_passed",
    ):
        if probe.get(key) is not True:
            raise RTPPairingSnapshotError(f"true-RNG probe lacks passing {key}")
    if probe.get("verified_rng_kinds") != list(SUPPORTED_RNG_KINDS):
        raise RTPPairingSnapshotError("true-RNG probe verified an unexpected RNG kind")
    abi = snapshot_abi_contract()
    payload = {
        "schema": CAPABILITY_SCHEMA,
        "status": "available",
        "created_at_utc": _utc_now(),
        "true_rng_pairing_available": True,
        "supported_rng_kinds": list(SUPPORTED_RNG_KINDS),
        "engine_artifact": dict(checked.engine_artifact),
        "source_artifact": dict(checked.source_artifact),
        "patch_artifact": dict(checked.patch_artifact),
        "build_artifact": dict(checked.build_artifact),
        "abi": {**abi, "canonical_abi_sha256": snapshot_abi_sha256()},
        "probe": probe_artifact,
    }
    return _immutable_json(output_path, payload)


def _configure_library(library: Any) -> None:
    library.BattleFinish.argtypes = [ctypes.c_void_p]
    library.BattleFinish.restype = None
    library.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    library.Select.restype = ctypes.c_int

    library.RtpPairingSnapshotAbiVersion.argtypes = []
    library.RtpPairingSnapshotAbiVersion.restype = ctypes.c_int
    library.RtpPairingSnapshotLastError.argtypes = []
    library.RtpPairingSnapshotLastError.restype = ctypes.c_char_p
    library.RtpPairingSnapshotInitialize.argtypes = []
    library.RtpPairingSnapshotInitialize.restype = ctypes.c_int
    library.RtpPairingBattleStartSeededOut.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.RtpPairingBattleStartSeededOut.restype = ctypes.c_int
    library.RtpPairingSnapshotGetBattleJsonOut.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.RtpPairingSnapshotGetBattleJsonOut.restype = ctypes.c_int
    library.RtpPairingSnapshotCapture.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.RtpPairingSnapshotCapture.restype = ctypes.c_int
    library.RtpPairingSnapshotRestore.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.RtpPairingSnapshotRestore.restype = ctypes.c_int
    library.RtpPairingSnapshotRestoreSerialized.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.RtpPairingSnapshotRestoreSerialized.restype = ctypes.c_int
    library.RtpPairingSnapshotRelease.argtypes = [ctypes.c_void_p]
    library.RtpPairingSnapshotRelease.restype = None
    library.RtpPairingSnapshotSerializedSize.argtypes = [ctypes.c_void_p]
    library.RtpPairingSnapshotSerializedSize.restype = ctypes.c_int
    library.RtpPairingSnapshotSerialize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
    ]
    library.RtpPairingSnapshotSerialize.restype = ctypes.c_int
    library.RtpPairingSnapshotFingerprintSize.argtypes = [ctypes.c_void_p]
    library.RtpPairingSnapshotFingerprintSize.restype = ctypes.c_int
    library.RtpPairingSnapshotFingerprint.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
    ]
    library.RtpPairingSnapshotFingerprint.restype = ctypes.c_int


def load_rtp_pairing_snapshot_library(
    path: str | Path, *, initialize: bool = True
) -> Any:
    """Load the private ABI and fail closed on missing/version-mismatched symbols."""

    resolved = str(_existing_regular_file(path, label="pairing engine"))
    try:
        library = ctypes.CDLL(resolved)
        _configure_library(library)
    except (AttributeError, OSError) as exc:
        raise RTPPairingSnapshotError(
            f"engine lacks the private RTP pairing snapshot ABI: {resolved}"
        ) from exc
    version = int(library.RtpPairingSnapshotAbiVersion())
    if version != SNAPSHOT_ABI_VERSION:
        raise RTPPairingSnapshotError(
            f"unsupported pairing snapshot ABI {version}; expected {SNAPSHOT_ABI_VERSION}"
        )
    if initialize and resolved not in _INITIALIZED_LIBRARY_PATHS:
        result = int(library.RtpPairingSnapshotInitialize())
        if result != 0:
            raise RTPPairingSnapshotError(
                "private pairing engine initialization failed "
                f"({result}): {_native_error(library)}"
            )
        _INITIALIZED_LIBRARY_PATHS.add(resolved)
    return library


def _native_error(library: Any) -> str:
    value = library.RtpPairingSnapshotLastError()
    if not value:
        return "native extension provided no error detail"
    try:
        return bytes(value).decode("utf-8", errors="replace")
    except TypeError:
        return str(value)


def _canonical_observation_digest(observation: Mapping[str, Any]) -> str:
    return canonical_digest(dict(observation))


@dataclass
class RtpPairingSnapshot:
    """Opaque native snapshot retained only for the lifetime of a cell."""

    _engine: "RtpPairingSnapshotEngine"
    _pointer: ctypes.c_void_p
    fingerprint_bytes: bytes
    requested_seed: int
    _closed: bool = False

    @property
    def fingerprint_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.fingerprint_bytes).hexdigest()

    @property
    def snapshot_id(self) -> str:
        return "snapshot-" + self.fingerprint_sha256[7:31]

    @property
    def bytes(self) -> int:
        return len(self.fingerprint_bytes)

    @property
    def serialized_bytes(self) -> bytes:
        """Private opaque material for an isolated arm worker, never a seed."""

        return self.fingerprint_bytes

    def close(self) -> None:
        if not self._closed and self._pointer.value:
            self._engine._library.RtpPairingSnapshotRelease(self._pointer)
        self._pointer = ctypes.c_void_p()
        self._closed = True

    def __enter__(self) -> "RtpPairingSnapshot":
        if self._closed:
            raise RTPPairingSnapshotError("pairing snapshot is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort native cleanup
        try:
            self.close()
        except Exception:
            pass


class RtpPairingBattle:
    """One fresh restore of a pairing snapshot, with an auditable transcript."""

    def __init__(self, engine: "RtpPairingSnapshotEngine", pointer: ctypes.c_void_p):
        self._engine = engine
        self._pointer = pointer
        self._closed = False
        self._observation: dict[str, Any] | None = None
        self.transcript_events: list[dict[str, Any]] = []

    @property
    def native_handle(self) -> int:
        return int(self._pointer.value or 0)

    def _ensure_open(self) -> None:
        if self._closed or not self._pointer.value:
            raise RTPPairingSnapshotError("pairing battle is closed")

    def observation(self) -> dict[str, Any]:
        self._ensure_open()
        if self._observation is None:
            json_pointer = ctypes.c_char_p()
            json_count = ctypes.c_int(0)
            select_player = ctypes.c_int(-1)
            result = int(
                self._engine._library.RtpPairingSnapshotGetBattleJsonOut(
                    self._pointer,
                    ctypes.byref(json_pointer),
                    ctypes.byref(json_count),
                    ctypes.byref(select_player),
                )
            )
            if result != 0 or not json_pointer.value or json_count.value <= 0:
                raise RTPPairingSnapshotError(
                    "native restore returned an empty observation "
                    f"({result}): {_native_error(self._engine._library)}"
                )
            try:
                raw_json = ctypes.string_at(json_pointer, int(json_count.value))
                parsed = json.loads(raw_json.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RTPPairingSnapshotError("native restore returned malformed JSON") from exc
            if not isinstance(parsed, Mapping):
                raise RTPPairingSnapshotError("native restore observation is not an object")
            self._observation = dict(parsed)
            self.transcript_events.append(
                {
                    "ordinal": len(self.transcript_events),
                    "kind": "observation",
                    "observation_sha256": _canonical_observation_digest(parsed),
                    "select_player": int(select_player.value),
                }
            )
        return dict(self._observation)

    def step(self, action: Sequence[int]) -> dict[str, Any]:
        self._ensure_open()
        selected = list(action)
        current_observation = self.observation()
        selection = current_observation.get("select") or {}
        minimum = selection.get("minCount")
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in selected)
            or (not selected and minimum != 0)
        ):
            raise RTPPairingSnapshotError("native selection is not legal for this boundary")
        values = (ctypes.c_int * len(selected))(*selected)
        error = int(self._engine._library.Select(self._pointer, values, len(selected)))
        if error:
            raise RTPPairingSnapshotError(f"native Select failed with code {error}")
        self._observation = None
        observation = self.observation()
        self.transcript_events.append(
            {
                "ordinal": len(self.transcript_events),
                "kind": "selection",
                "action": selected,
                "observation_sha256": _canonical_observation_digest(observation),
            }
        )
        return observation

    @property
    def finished(self) -> bool:
        result = ((self.observation().get("current") or {}).get("result"))
        return result is not None and int(result) != -1

    @property
    def winner(self) -> int | None:
        result = ((self.observation().get("current") or {}).get("result"))
        if result is None or int(result) == -1:
            return None
        return int(result)

    @property
    def transcript_sha256(self) -> str:
        return canonical_digest(self.transcript_events)

    def close(self) -> None:
        if not self._closed and self._pointer.value:
            self._engine._library.BattleFinish(self._pointer)
        self._pointer = ctypes.c_void_p()
        self._closed = True

    def __enter__(self) -> "RtpPairingBattle":
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort native cleanup
        try:
            self.close()
        except Exception:
            pass


class RtpPairingSnapshotEngine:
    """Private engine wrapper that can capture and restore exact A/B/C starts."""

    def __init__(
        self,
        library_path: str | Path,
        *,
        library: Any | None = None,
        initialize: bool = True,
    ) -> None:
        self.library_identity = frozen_file_identity(library_path)
        self._bound_artifacts: PairingArtifactSet | None = None
        self._library = (
            load_rtp_pairing_snapshot_library(library_path, initialize=initialize)
            if library is None
            else library
        )
        if library is not None:
            _configure_library(self._library)
            version = int(self._library.RtpPairingSnapshotAbiVersion())
            if version != SNAPSHOT_ABI_VERSION:
                raise RTPPairingSnapshotError("injected engine has unsupported snapshot ABI")
            if initialize:
                resolved = self.library_identity["path"]
                if resolved not in _INITIALIZED_LIBRARY_PATHS:
                    result = int(self._library.RtpPairingSnapshotInitialize())
                    if result != 0:
                        raise RTPPairingSnapshotError(
                            "injected pairing engine initialization failed "
                            f"({result}): {_native_error(self._library)}"
                        )
                    _INITIALIZED_LIBRARY_PATHS.add(resolved)

    @property
    def abi(self) -> dict[str, Any]:
        return {**snapshot_abi_contract(), "canonical_abi_sha256": snapshot_abi_sha256()}

    @property
    def identity(self) -> dict[str, Any]:
        """Exact receipt-bound native engine identity for the evaluation runner."""

        return dict(self._require_bound_artifacts().engine_artifact)

    def require_bound_artifacts(self, artifacts: PairingArtifactSet) -> PairingArtifactSet:
        """Bind this loaded binary to the exact private build evidence."""

        checked = verify_build_receipt(artifacts)
        if self.library_identity["sha256"] != checked.engine_artifact["sha256"]:
            raise RTPPairingSnapshotError(
                "loaded pairing engine does not match the receipt-bound engine artifact"
            )
        self._bound_artifacts = checked
        return checked

    def _require_bound_artifacts(self) -> PairingArtifactSet:
        if self._bound_artifacts is None:
            raise RTPPairingSnapshotError(
                "capture/restore is forbidden until exact private build artifacts are bound"
            )
        # Rehash immediately before native use so a mutable private path cannot
        # silently inherit an earlier successful attestation.
        return self.require_bound_artifacts(self._bound_artifacts)

    def capture_cell_snapshot(
        self,
        deck0: Sequence[int],
        deck1: Sequence[int],
        requested_seed: int,
    ) -> RtpPairingSnapshot:
        self._require_bound_artifacts()
        if len(deck0) != 60 or len(deck1) != 60:
            raise RTPPairingSnapshotError("pairing snapshot requires two 60-card decks")
        if isinstance(requested_seed, bool) or not 0 <= int(requested_seed) <= 0xFFFFFFFF:
            raise RTPPairingSnapshotError("requested seed must fit uint32")
        cards = list(deck0) + list(deck1)
        if any(isinstance(card, bool) or not isinstance(card, int) for card in cards):
            raise RTPPairingSnapshotError("deck cards must be integers")
        native_cards = (ctypes.c_int * 120)(*cards)
        battle = ctypes.c_void_p()
        error_player = ctypes.c_int(-1)
        error_type = ctypes.c_int(-1)
        start_result = int(
            self._library.RtpPairingBattleStartSeededOut(
                native_cards,
                ctypes.c_uint32(int(requested_seed)),
                ctypes.byref(battle),
                ctypes.byref(error_player),
                ctypes.byref(error_type),
            )
        )
        if start_result != 0 or not battle.value:
            raise RTPPairingSnapshotError(
                "native deterministic pairing start failed "
                f"player={int(error_player.value)} type={int(error_type.value)}: "
                f"{_native_error(self._library)}"
            )
        try:
            snapshot_ptr = ctypes.c_void_p()
            result = int(self._library.RtpPairingSnapshotCapture(battle, ctypes.byref(snapshot_ptr)))
            if result != 0 or not snapshot_ptr.value:
                raise RTPPairingSnapshotError(
                    f"native snapshot capture failed ({result}): {_native_error(self._library)}"
                )
        finally:
            self._library.BattleFinish(battle)
        try:
            fingerprint = self._snapshot_fingerprint(snapshot_ptr)
            return RtpPairingSnapshot(
                self,
                snapshot_ptr,
                fingerprint,
                int(requested_seed),
            )
        except Exception:
            self._library.RtpPairingSnapshotRelease(snapshot_ptr)
            raise

    def _snapshot_fingerprint(self, snapshot: ctypes.c_void_p) -> bytes:
        size = int(self._library.RtpPairingSnapshotFingerprintSize(snapshot))
        if size < 1 or size > 64 * 1024 * 1024:
            raise RTPPairingSnapshotError(
                f"native snapshot returned invalid fingerprint size {size}: {_native_error(self._library)}"
            )
        buffer = (ctypes.c_ubyte * size)()
        copied = int(
            self._library.RtpPairingSnapshotFingerprint(snapshot, buffer, size)
        )
        if copied != size:
            raise RTPPairingSnapshotError(
                f"native snapshot fingerprint copy mismatch {copied}/{size}: "
                f"{_native_error(self._library)}"
            )
        return bytes(buffer)

    def restore_snapshot(self, snapshot: RtpPairingSnapshot) -> RtpPairingBattle:
        self._require_bound_artifacts()
        if snapshot._engine is not self:
            raise RTPPairingSnapshotError("snapshot belongs to another engine wrapper")
        if snapshot._closed or not snapshot._pointer.value:
            raise RTPPairingSnapshotError("snapshot is closed")
        battle = ctypes.c_void_p()
        result = int(
            self._library.RtpPairingSnapshotRestore(snapshot._pointer, ctypes.byref(battle))
        )
        if result != 0 or not battle.value:
            raise RTPPairingSnapshotError(
                f"native snapshot restore failed ({result}): {_native_error(self._library)}"
            )
        return RtpPairingBattle(self, battle)

    @staticmethod
    def _sealed_snapshot_identity(snapshot_artifact: Mapping[str, Any]) -> dict[str, Any]:
        identity = verify_frozen_file_identity(
            snapshot_artifact, label="sealed snapshot artifact"
        )
        if identity["mode"] != 0o444:
            raise RTPPairingSnapshotError(
                "sealed snapshot artifact must be immutable mode 0444"
            )
        return identity

    def _restore_verified_sealed_snapshot(
        self, identity: Mapping[str, Any], material: bytes
    ) -> RtpPairingBattle:
        """Call the unsafe native State decoder only after seal verification."""

        self._require_bound_artifacts()
        if not material or len(material) > 64 * 1024 * 1024:
            raise RTPPairingSnapshotError("serialized pairing snapshot has invalid size")
        expected = _valid_digest(identity.get("sha256"), label="sealed snapshot sha256")
        actual = "sha256:" + hashlib.sha256(material).hexdigest()
        if actual != expected:
            raise RTPPairingSnapshotError(
                f"sealed snapshot checksum mismatch: expected {expected}, got {actual}"
            )
        payload = (ctypes.c_ubyte * len(material)).from_buffer_copy(material)
        sealed_digest = (ctypes.c_ubyte * 32).from_buffer_copy(
            bytes.fromhex(expected[7:])
        )
        battle = ctypes.c_void_p()
        result = int(
            self._library.RtpPairingSnapshotRestoreSerialized(
                payload,
                len(material),
                sealed_digest,
                len(sealed_digest),
                ctypes.byref(battle),
            )
        )
        if result != 0 or not battle.value:
            raise RTPPairingSnapshotError(
                f"native serialized snapshot restore failed ({result}): "
                f"{_native_error(self._library)}"
            )
        return RtpPairingBattle(self, battle)

    def restore_serialized_snapshot(
        self,
        serialized: bytes | bytearray,
        *,
        snapshot_artifact: Mapping[str, Any],
    ) -> RtpPairingBattle:
        """Restore bytes only when they exactly equal a sealed artifact.

        The raw native endpoint can invoke the upstream unchecked State reader.
        Therefore this public wrapper requires the immutable 0444 snapshot
        artifact—not a caller-provided hash string—and checks the supplied
        bytes against that artifact before invoking native code.
        """

        identity = self._sealed_snapshot_identity(snapshot_artifact)
        sealed_material = _existing_regular_file(
            identity["path"], label="sealed snapshot artifact"
        ).read_bytes()
        material = bytes(serialized)
        if material != sealed_material:
            raise RTPPairingSnapshotError(
                "serialized snapshot bytes do not equal the sealed artifact"
            )
        return self._restore_verified_sealed_snapshot(identity, material)

    def restore_sealed_snapshot_artifact(
        self, snapshot_artifact: Mapping[str, Any]
    ) -> RtpPairingBattle:
        """Open only a receipt-bound sealed snapshot file for an arm worker."""

        identity = self._sealed_snapshot_identity(snapshot_artifact)
        material = _existing_regular_file(
            identity["path"], label="sealed snapshot artifact"
        ).read_bytes()
        return self._restore_verified_sealed_snapshot(identity, material)

    def restore_sealed_snapshot_manifest(
        self, snapshot_seal_path: str | Path
    ) -> RtpPairingBattle:
        """Restore only a receipt-bound physical snapshot seal.

        Arm workers receive the seal path from the evaluation's immutable RNG
        materials manifest.  The seal binds the snapshot's exact bytes to this
        private engine/source/patch/build/ABI tuple before native restore.
        """

        checked = self._require_bound_artifacts()
        seal_identity = frozen_file_identity(snapshot_seal_path)
        if seal_identity["mode"] != 0o444:
            raise RTPPairingSnapshotError("snapshot seal must be immutable mode 0444")
        seal = _read_json(seal_identity["path"], label="snapshot seal")
        if seal.get("schema") != SNAPSHOT_SEAL_SCHEMA or seal.get("status") != "sealed":
            raise RTPPairingSnapshotError("snapshot seal is not a sealed pairing artifact")
        required = {
            "engine_artifact_sha256": checked.engine_artifact["sha256"],
            "source_artifact_sha256": checked.source_artifact["sha256"],
            "patch_artifact_sha256": checked.patch_artifact["sha256"],
            "build_artifact_sha256": checked.build_artifact["sha256"],
            "canonical_abi_sha256": snapshot_abi_sha256(),
            "capture_boundary": SNAPSHOT_CAPTURE_BOUNDARY,
            "boundary_tag": SNAPSHOT_BOUNDARY_TAG,
        }
        for key, expected in required.items():
            if seal.get(key) != expected:
                raise RTPPairingSnapshotError(f"snapshot seal does not bind {key}")
        if seal.get("requested_seed_is_pairing_proof") is not False:
            raise RTPPairingSnapshotError(
                "snapshot seal must state that its audit seed is not pairing proof"
            )
        requested_seed = seal.get("requested_seed_audit_only")
        if (
            isinstance(requested_seed, bool)
            or not isinstance(requested_seed, int)
            or not 0 <= requested_seed <= 0xFFFFFFFF
        ):
            raise RTPPairingSnapshotError("snapshot seal has an invalid audit seed")
        case_binding_artifact = seal.get("case_binding_artifact")
        if not isinstance(case_binding_artifact, Mapping):
            raise RTPPairingSnapshotError("snapshot seal lacks a case binding artifact")
        case_identity, _ = verify_pairing_case_binding(
            case_binding_artifact, expected_debug_seed=requested_seed
        )
        if seal.get("case_binding_artifact_sha256") != case_identity["sha256"]:
            raise RTPPairingSnapshotError("snapshot seal does not bind its case binding")
        artifact = seal.get("snapshot_artifact")
        if not isinstance(artifact, Mapping):
            raise RTPPairingSnapshotError("snapshot seal lacks a snapshot artifact identity")
        return self.restore_sealed_snapshot_artifact(artifact)

    def duplicate_restore_probe(
        self,
        snapshot: RtpPairingSnapshot,
        action_selector: Callable[[Mapping[str, Any], int], Sequence[int]],
        *,
        steps: int,
        delay_seconds: float,
    ) -> dict[str, Any]:
        """Prove duplicate restores are independent and transcript-identical.

        The selector sees identical observations for clone A/B and must return
        exactly the same legal action.  A third clone is left untouched until
        the end, proving mutation of A/B did not alias the snapshot or C.
        """

        if steps < 1:
            raise RTPPairingSnapshotError("duplicate-restore probe needs at least one step")
        if delay_seconds <= 0.0:
            raise RTPPairingSnapshotError("duplicate-restore probe requires a deliberate delay")
        left = self.restore_snapshot(snapshot)
        right: RtpPairingBattle | None = None
        untouched: RtpPairingBattle | None = None
        try:
            time.sleep(delay_seconds)
            right = self.restore_snapshot(snapshot)
            untouched = self.restore_snapshot(snapshot)
            handles = {left.native_handle, right.native_handle, untouched.native_handle}
            if 0 in handles or len(handles) != 3:
                raise RTPPairingSnapshotError("native restore reused a battle handle")
            initial_untouched = _canonical_observation_digest(untouched.observation())
            executed_steps = 0
            for index in range(steps):
                left_observation = left.observation()
                right_observation = right.observation()
                if _canonical_observation_digest(left_observation) != _canonical_observation_digest(
                    right_observation
                ):
                    raise RTPPairingSnapshotError("duplicate restores diverged before selection")
                action = list(action_selector(left_observation, index))
                other_action = list(action_selector(right_observation, index))
                if action != other_action:
                    raise RTPPairingSnapshotError("probe selector is not deterministic")
                left.step(action)
                right.step(other_action)
                executed_steps += 1
                if left.transcript_sha256 != right.transcript_sha256:
                    raise RTPPairingSnapshotError("duplicate restores diverged after selection")
                if left.finished or right.finished:
                    if index + 1 < steps:
                        raise RTPPairingSnapshotError("game finished before requested probe steps")
                    break
            if _canonical_observation_digest(untouched.observation()) != initial_untouched:
                raise RTPPairingSnapshotError("restored clone mutated an independent handle")
            return {
                "passed": True,
                "initial_snapshot_fingerprint_sha256": snapshot.fingerprint_sha256,
                "initial_snapshot_fingerprint_bytes": snapshot.bytes,
                "deterministic_transcript_sha256": left.transcript_sha256,
                "transcript_steps": executed_steps,
                "duplicate_restore_independent_handles": True,
                "device_rand_false_verified": True,
                "requested_seed_only_rejected": True,
                "delayed_restore_transcript_passed": True,
                # A standalone child-process proof is recorded by the staging
                # runner.  This in-process probe cannot truthfully claim it.
                "cross_process_restore_passed": False,
                "delayed_restore_seconds": float(delay_seconds),
                "capture_boundary": SNAPSHOT_CAPTURE_BOUNDARY,
                "boundary_tag": SNAPSHOT_BOUNDARY_TAG,
            }
        finally:
            if untouched is not None:
                untouched.close()
            if right is not None:
                right.close()
            left.close()


__all__ = [
    "BUILD_SCHEMA",
    "CAPABILITY_SCHEMA",
    "PairingArtifactSet",
    "PROBE_SCHEMA",
    "RTPPairingSnapshotError",
    "RtpPairingBattle",
    "RtpPairingSnapshot",
    "RtpPairingSnapshotEngine",
    "SNAPSHOT_ABI_NAME",
    "SNAPSHOT_ABI_VERSION",
    "SNAPSHOT_SEAL_SCHEMA",
    "canonical_digest",
    "emit_true_rng_pairing_capability",
    "emit_true_rng_pairing_probe",
    "file_digest",
    "frozen_file_identity",
    "load_rtp_pairing_snapshot_library",
    "snapshot_abi_contract",
    "snapshot_abi_sha256",
    "verify_build_receipt",
    "verify_pairing_case_binding",
]
