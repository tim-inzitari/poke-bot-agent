"""Freeze the r198 three-arm evaluation inputs before any game is run.

This module has deliberately narrow authority.  It creates an evaluation-only
official-control cohort, captures and seals the native post-start pairing
snapshots that the A/B/C runner restores, runs the independent planner-pass
preflight, and prepares the evaluator-v2 manifest.  It never trains, serves,
selects, promotes, or submits a model.

The important distinction is that a requested seed starts a native capture but
is retained only as audit metadata.  The evidence consumed by the evaluator is
the immutable snapshot byte artifact and its separately immutable native seal.
No code here can recreate a paired cell from a seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poke_bot.engine_rebuild.rtp_pairing_snapshot import (
    CAPABILITY_SCHEMA,
    PROBE_SCHEMA,
    SNAPSHOT_ABI_NAME,
    SNAPSHOT_ABI_VERSION,
    SNAPSHOT_BOUNDARY_TAG,
    SNAPSHOT_CAPTURE_BOUNDARY,
    SNAPSHOT_SEAL_SCHEMA,
    PairingArtifactSet,
    RTPPairingSnapshotError,
    RtpPairingSnapshotEngine,
    snapshot_abi_contract,
    snapshot_abi_sha256,
    verify_build_receipt,
)


REQUEST_SCHEMA = "poke_bot.recursive_turn_planner.r198_evaluation_input_request/v1"
COHORT_SCHEMA = "poke_bot.recursive_turn_planner.r197_evaluation_only_cohort/v1"
SOURCE_EXCLUSION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r197_evaluation_only_source_exclusion/v1"
)
MATERIALS_SCHEMA = "poke_bot.recursive_turn_planner.r198_evaluation_rng_materials/v1"
PREFLIGHT_INPUT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_planner_pass_preflight_input/v1"
)
PREFLIGHT_SCHEMA = "poke_bot.recursive_turn_planner.r198_planner_pass_preflight/v1"
EVALUATION_AUTHORITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_authorization/v1"
)
R197_COMPLETION_SCHEMA = "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
R197_SELECTION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r197_training_selection_plan/v1"
)
R197_WHOLE_EPISODE_SELECTION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r197_whole_episode_selection/v1"
)

R198_CANDIDATE_CONTRACT_SHA256 = (
    "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R198_REGISTRY_SHA256 = (
    "sha256:78fd8e52df1464db94e74a49247a67ced41b5d164dc86fafec3229f2c1e47edc"
)
R198_REGISTRY_BYTES = 2117
R198_CANDIDATE_EVALUATION_BINDING_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_candidate_evaluation_binding/v1"
)
R198_PARENT_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R198_SIDECAR_SHA256 = (
    "sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31"
)
R198_SIDECAR_CONFIG_SHA256 = (
    "sha256:7fb0658f0358c93636524a40ddd52f9f76199de261963a85dbf5946901a9f676"
)
R198_DECK_FILE_SHA256 = (
    "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
)
R198_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R198_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R198_MATCHUP_ADAPTER_REGISTRY_RELATIVE = Path("state/matchup_adapter_roster.json")
R198_MATCHUP_ADAPTER_REGISTRY_SHA256 = (
    "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
)
R198_MATCHUP_ADAPTER_REGISTRY_BYTES = 11_899
R198_MATCHUP_ADAPTER_REGISTRY_MODE = 0o444
R198_MATCHUP_ADAPTER_SLOT_REGISTRY_DIGEST = (
    "sha256:444c42c1235c19d3d95b10e80a12a84f35c9fb803967096736446eac1a5e225a"
)
MATCHUP_ADAPTER_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_roster/v1"
MATCHUP_ADAPTER_SLOT_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_slot_registry/v1"
OFFICIAL_PANEL: tuple[tuple[str, str], ...] = (
    (
        "iono",
        "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    ),
    (
        "dragapult-ex",
        "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    ),
    (
        "mega-abomasnow-ex",
        "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    ),
    (
        "mega-lucario-ex",
        "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
    ),
)
REPLICATES_PER_SEAT = 125
PAIRED_CELL_COUNT = len(OFFICIAL_PANEL) * 2 * REPLICATES_PER_SEAT
R198_ARMS = ("no_rtp", "direct_bridge_recursive_disabled", "recursive_rtp")


class R198EvaluationInputError(RuntimeError):
    """Raised when immutable r198 evidence cannot be produced honestly."""


@dataclass(frozen=True)
class CapturedSnapshot:
    """Opaque bytes returned by one bound native post-start capture."""

    serialized_bytes: bytes
    snapshot_id: str
    fingerprint_sha256: str
    fingerprint_bytes: int


@dataclass(frozen=True)
class _CapabilityBinding:
    receipt: dict[str, Any]
    engine_artifact: dict[str, Any]
    source_artifact: dict[str, Any]
    patch_artifact: dict[str, Any]
    build_artifact: dict[str, Any]
    abi_sha256: str


SnapshotCapturer = Callable[[Sequence[int], Sequence[int], int], CapturedSnapshot]
PreflightRunner = Callable[[Mapping[str, Any], Path], Path | str]
SeedProvider = Callable[[], int]
# Production uses `_native_fixture_observation_extractor`, which restores the
# just-written *seal* through the capability-bound wrapper.  The injectable
# form keeps focused tests independent of a private native shared library.
FixtureObservationExtractor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _r197_canonical_json_digest(value: Any) -> str:
    """Match the r197 selection planner's canonical JSON digest exactly."""

    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _valid_digest(value: Any, label: str) -> str:
    result = str(value or "")
    if not result.startswith("sha256:") or len(result) != 71:
        raise R198EvaluationInputError(f"{label} must be a SHA-256 identity")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise R198EvaluationInputError(f"{label} must be a SHA-256 identity") from exc
    return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R198EvaluationInputError(f"{label} must be a JSON object")
    return dict(value)


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise R198EvaluationInputError(f"{label} is required")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise R198EvaluationInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise R198EvaluationInputError(f"{label} must be an integer") from exc
    if result < minimum:
        raise R198EvaluationInputError(f"{label} must be at least {minimum}")
    return result


def _absolute_no_parent(path: str | Path, label: str) -> Path:
    raw = Path(os.path.expanduser(os.fspath(path)))
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if ".." in raw.parts:
        raise R198EvaluationInputError(f"{label} may not contain '..'")
    return Path(os.path.abspath(os.fspath(raw)))


def _physical_existing(path: str | Path, label: str, *, directory: bool = False) -> Path:
    target = _absolute_no_parent(path, label)
    current = Path(target.anchor)
    for index, component in enumerate(target.parts[1:]):
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise R198EvaluationInputError(f"{label} does not exist: {target}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise R198EvaluationInputError(f"{label} traverses a symlink: {current}")
        if index < len(target.parts) - 2 and not stat.S_ISDIR(metadata.st_mode):
            raise R198EvaluationInputError(f"{label} has a non-directory ancestor: {current}")
    mode = os.lstat(target).st_mode
    if directory:
        if not stat.S_ISDIR(mode):
            raise R198EvaluationInputError(f"{label} is not a physical directory: {target}")
    elif not stat.S_ISREG(mode):
        raise R198EvaluationInputError(f"{label} is not a physical regular file: {target}")
    return target


def _ensure_physical_directory(path: str | Path, label: str) -> Path:
    target = _absolute_no_parent(path, label)
    current = Path(target.anchor)
    for component in target.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
        except OSError as exc:
            raise R198EvaluationInputError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise R198EvaluationInputError(f"{label} has a symlink or non-directory: {current}")
    return target


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _identity(path: str | Path, label: str, *, immutable: bool = True) -> dict[str, Any]:
    source = _physical_existing(path, label)
    mode = stat.S_IMODE(os.lstat(source).st_mode)
    if immutable and mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise R198EvaluationInputError(f"{label} must be immutable: {source}")
    return {"path": str(source), "sha256": _file_digest(source), "bytes": source.stat().st_size}


def _verify_identity(raw: Any, label: str, *, immutable: bool = True) -> dict[str, Any]:
    declared = _mapping(raw, label)
    path = _physical_existing(_text(declared.get("path"), f"{label}.path"), label)
    observed = _identity(path, label, immutable=immutable)
    if observed["sha256"] != _valid_digest(declared.get("sha256"), f"{label}.sha256"):
        raise R198EvaluationInputError(f"{label} SHA-256 changed")
    if "bytes" in declared and _integer(declared["bytes"], f"{label}.bytes") != observed["bytes"]:
        raise R198EvaluationInputError(f"{label} byte length changed")
    return observed


def _canonical_matchup_adapter_registry_digest(payload: Mapping[str, Any]) -> str:
    """Match the V6 registry digest embedded in the r195 parent contract."""

    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _snapshot_local_matchup_adapter_registry(
    production_factory: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash the exact V6 roster before the materializer can capture a cell.

    The factory is responsible for staging this source-snapshot-local file,
    but capture must not begin merely because a later arm child would reject
    an absent or ambient registry.  The raw JSON bytes and the registry's
    canonical slot digest are distinct identities and both are fixed by r198.
    """

    factory = _mapping(production_factory, "production factory")
    source_root = _physical_existing(
        _text(
            factory.get("source_snapshot_root"),
            "production_factory.source_snapshot_root",
        ),
        "production factory source snapshot root",
        directory=True,
    )
    raw = _mapping(
        factory.get("matchup_adapter_registry"),
        "production_factory.matchup_adapter_registry",
    )
    expected_keys = {"path", "sha256", "bytes", "mode"}
    if set(raw) != expected_keys:
        raise R198EvaluationInputError(
            "production_factory.matchup_adapter_registry must contain exactly "
            "path, sha256, bytes, and mode"
        )
    declared_mode = _integer(
        raw.get("mode"), "production_factory.matchup_adapter_registry.mode", minimum=0
    )
    if declared_mode != R198_MATCHUP_ADAPTER_REGISTRY_MODE:
        raise R198EvaluationInputError(
            "production_factory.matchup_adapter_registry must declare mode 0444"
        )
    observed = _verify_identity(
        raw,
        "production_factory.matchup_adapter_registry",
        immutable=True,
    )
    registry_path = Path(observed["path"])
    expected_path = source_root / R198_MATCHUP_ADAPTER_REGISTRY_RELATIVE
    if registry_path != expected_path:
        raise R198EvaluationInputError(
            "production_factory.matchup_adapter_registry is not the exact "
            "snapshot-local state/matchup_adapter_roster.json"
        )
    actual_mode = stat.S_IMODE(os.lstat(registry_path).st_mode)
    if actual_mode != R198_MATCHUP_ADAPTER_REGISTRY_MODE:
        raise R198EvaluationInputError(
            "production_factory.matchup_adapter_registry must have mode 0444"
        )
    if (
        observed["sha256"] != R198_MATCHUP_ADAPTER_REGISTRY_SHA256
        or observed["bytes"] != R198_MATCHUP_ADAPTER_REGISTRY_BYTES
    ):
        raise R198EvaluationInputError(
            "production_factory.matchup_adapter_registry does not bind the exact r198 roster bytes"
        )
    payload = _read_json(registry_path, "snapshot-local matchup adapter registry")
    if (
        payload.get("schema") != MATCHUP_ADAPTER_REGISTRY_SCHEMA
        or payload.get("slot_schema") != MATCHUP_ADAPTER_SLOT_REGISTRY_SCHEMA
        or _integer(payload.get("slot_capacity"), "snapshot-local roster slot_capacity")
        != 64
    ):
        raise R198EvaluationInputError(
            "snapshot-local matchup adapter registry is not the exact V6 slot registry"
        )
    if _canonical_matchup_adapter_registry_digest(payload) != (
        R198_MATCHUP_ADAPTER_SLOT_REGISTRY_DIGEST
    ):
        raise R198EvaluationInputError(
            "snapshot-local matchup adapter registry canonical slot digest changed"
        )
    return {**observed, "mode": actual_mode}


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = _physical_existing(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R198EvaluationInputError(f"cannot read {label}: {source}") from exc
    return _mapping(value, label)


def _write_immutable_bytes(path: Path, payload: bytes, label: str) -> dict[str, Any]:
    parent = _ensure_physical_directory(path.parent, f"{label} parent")
    target = parent / path.name
    try:
        existing = os.lstat(target)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        raise R198EvaluationInputError(f"refusing to clobber {label}: {target}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(target, 0o444)
    except Exception:
        # The partially written O_EXCL file is intentionally retained as
        # forensic evidence.  Its parent will be frozen by the caller.
        raise
    identity = _identity(target, label)
    if identity["sha256"] != _sha256_bytes(payload):
        raise R198EvaluationInputError(f"{label} did not retain its written bytes")
    return identity


def _write_immutable_json(path: Path, payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    material = dict(payload)
    material["record_sha256"] = canonical_digest(
        {key: value for key, value in material.items() if key not in {"created_at_utc", "record_sha256"}}
    )
    encoded = (json.dumps(material, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return _write_immutable_bytes(path, encoded, label)


def _write_immutable_observation(
    path: Path, observation: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Seal a native observation without adding materializer metadata.

    The production factory restores the fixture seal independently and
    canonical-compares that native observation to this JSON.  In particular,
    a normal immutable-record ``record_sha256`` field would alter the gameplay
    observation and make that comparison dishonest.
    """

    payload = _mapping(observation, label)
    try:
        encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R198EvaluationInputError(f"{label} is not JSON serializable") from exc
    return _write_immutable_bytes(path, encoded, label)


def _freeze_tree(root: Path) -> None:
    """Make a completed or failed materialization tree read-only without links."""

    physical = _physical_existing(root, "evaluation input output root", directory=True)
    for current_text, directories, filenames in os.walk(physical, topdown=False, followlinks=False):
        current = Path(current_text)
        for name in filenames:
            entry = current / name
            mode = os.lstat(entry).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise R198EvaluationInputError(f"output contains an unsafe file: {entry}")
            os.chmod(entry, 0o444)
        for name in directories:
            entry = current / name
            mode = os.lstat(entry).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise R198EvaluationInputError(f"output contains an unsafe directory: {entry}")
            os.chmod(entry, 0o555)
        os.chmod(current, 0o555)


def _deck_cards(identity: Mapping[str, Any], label: str) -> list[int]:
    artifact = _verify_identity(identity, label)
    try:
        cards = [int(row.strip()) for row in Path(artifact["path"]).read_text(encoding="utf-8").splitlines() if row.strip()]
    except ValueError as exc:
        raise R198EvaluationInputError(f"{label} must contain one integer card ID per line") from exc
    if len(cards) != 60:
        raise R198EvaluationInputError(f"{label} must contain exactly 60 cards")
    return cards


def _r197_selection_side(selection: Mapping[str, Any], side: str) -> tuple[str, list[str]]:
    record = _mapping(selection.get(side), f"r197 selection.{side}")
    candidate = _mapping(record.get("candidate_selection"), f"r197 selection.{side}.candidate_selection")
    if candidate.get("schema") != R197_WHOLE_EPISODE_SELECTION_SCHEMA or candidate.get("split") != side:
        raise R198EvaluationInputError(f"r197 selection.{side} is not a whole-episode {side} selection")
    capped = _mapping(record.get("batch_cap_selection"), f"r197 selection.{side}.batch_cap_selection")
    if capped.get("row_level_sampling") is not False or capped.get("cross_window_dynamics_target") is not False:
        raise R198EvaluationInputError(f"r197 selection.{side} permits prohibited sampling")
    ids_raw = record.get("retained_episode_ids")
    if not isinstance(ids_raw, Sequence) or isinstance(ids_raw, (str, bytes)):
        raise R198EvaluationInputError(f"r197 selection.{side} lacks retained episode IDs")
    ids = [_text(value, f"r197 {side} episode") for value in ids_raw]
    if not ids or len(set(ids)) != len(ids):
        raise R198EvaluationInputError(f"r197 selection.{side} must retain unique episodes")
    if _integer(capped.get("retained_episode_count"), f"r197 {side} retained count", minimum=1) != len(ids):
        raise R198EvaluationInputError(f"r197 selection.{side} retained count mismatch")
    digest = _valid_digest(capped.get("retained_episode_ids_sha256"), f"r197 {side} retained digest")
    if _r197_canonical_json_digest(ids) != digest:
        raise R198EvaluationInputError(
            f"r197 selection.{side} retained episode digest does not rehash its exact IDs"
        )
    return digest, ids


def _candidate_provenance(completion_receipt: str | Path) -> dict[str, Any]:
    identity = _identity(completion_receipt, "r197 completion receipt")
    receipt = _read_json(identity["path"], "r197 completion receipt")
    if receipt.get("schema") != R197_COMPLETION_SCHEMA or receipt.get("status") != "completed_shadow_only":
        raise R198EvaluationInputError("completion receipt is not the completed r198 shadow candidate")
    if receipt.get("candidate_contract_sha256") != R198_CANDIDATE_CONTRACT_SHA256:
        raise R198EvaluationInputError("completion receipt does not bind the exact r198 candidate")
    authority = _mapping(receipt.get("authority"), "r197 completion authority")
    if authority.get("shadow_only") is not True:
        raise R198EvaluationInputError("r197 candidate is not shadow-only")
    for key in (
        "serving_eligible",
        "action_authority_enabled",
        "selector_authority",
        "live_checkpoint_publication",
        "submission_eligible",
    ):
        if authority.get(key) is not False:
            raise R198EvaluationInputError(f"r197 completion unexpectedly grants {key}")
    contract = _mapping(receipt.get("contract"), "r197 completion contract")
    corpus = _mapping(contract.get("complete_action_corpus"), "r197 complete-action corpus")
    if corpus.get("schema") != "poke_bot.rtp_complete_action_shadow_corpus/v1":
        raise R198EvaluationInputError("r197 corpus schema is invalid")
    split = _mapping(corpus.get("split"), "r197 corpus split")
    if split.get("source_disjoint") is not True or split.get("unit") != "episode_id" or _integer(split.get("seed"), "r197 split seed") != 5_000_000:
        raise R198EvaluationInputError("r197 corpus does not prove source-disjoint episodes")
    selection = _mapping(corpus.get("selection"), "r197 selection")
    if selection.get("schema") != R197_SELECTION_SCHEMA:
        raise R198EvaluationInputError("r197 selection schema is invalid")
    if selection.get("row_level_sampling") is not False or selection.get("cross_window_dynamics_target") is not False:
        raise R198EvaluationInputError("r197 selection permits prohibited sampling")
    train_digest, train_ids = _r197_selection_side(selection, "train")
    heldout_digest, heldout_ids = _r197_selection_side(selection, "heldout")
    if set(train_ids).intersection(heldout_ids):
        raise R198EvaluationInputError("r197 train and heldout episodes overlap")
    if _valid_digest(selection.get("train_selection_sha256"), "r197 train selection digest") != train_digest:
        raise R198EvaluationInputError("r197 train selection digest mismatch")
    if _valid_digest(selection.get("heldout_selection_sha256"), "r197 heldout selection digest") != heldout_digest:
        raise R198EvaluationInputError("r197 heldout selection digest mismatch")
    training = _mapping(receipt.get("training"), "r197 completion training")
    if training.get("heldout_is_source_excluded") is not True:
        raise R198EvaluationInputError("r197 heldout does not remain source-excluded")
    target_wiring = _mapping(training.get("candidate_target_wiring"), "r197 candidate target wiring")
    if target_wiring.get("status") != "masked_absent_no_fabrication":
        raise R198EvaluationInputError("r197 candidate target status unexpectedly changed")
    for key, expected in (
        ("latent_lookahead_targets", "not_wired_future_input"),
        ("unobserved_action_returns", "not_fabricated"),
        ("value_of_planning_target", "not_heuristic_labeled"),
    ):
        if target_wiring.get(key) != expected:
            raise R198EvaluationInputError(f"r197 target wiring changed at {key}")
    metrics = _mapping(training.get("metrics"), "r197 completion metrics")
    heldout_metrics = _mapping(metrics.get("rtp_heldout"), "r197 heldout metrics")
    for key in (
        "mean_candidate_calibration_target_count",
        "mean_candidate_ranking_pair_count",
        "mean_candidate_return_target_count",
    ):
        if float(heldout_metrics.get(key, -1.0)) != 0.0:
            raise R198EvaluationInputError(f"r197 unexpectedly reports a trusted target at {key}")
    return {
        "completion_receipt": identity,
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "corpus_manifest_sha256": _valid_digest(corpus.get("manifest_sha256"), "r197 corpus manifest"),
        "corpus_receipt_sha256": _valid_digest(corpus.get("receipt_sha256"), "r197 corpus receipt"),
        "selection_plan_sha256": _valid_digest(selection.get("selection_plan_sha256"), "r197 selection plan"),
        "train_selection_sha256": train_digest,
        "heldout_selection_sha256": heldout_digest,
        "train_episode_ids": train_ids,
        "heldout_episode_ids": heldout_ids,
    }


def _official_registry(registry_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = _identity(registry_path, "research-control registry")
    if identity["sha256"] != R198_REGISTRY_SHA256 or identity["bytes"] != R198_REGISTRY_BYTES:
        raise R198EvaluationInputError("research-control registry is not the exact frozen r198 panel")
    registry = _read_json(identity["path"], "research-control registry")
    if registry.get("schema") != "poke_bot.research_control_registry/v1" or registry.get("registry_id") != "alakazam-research-controls" or _integer(registry.get("version"), "research-control registry version", minimum=1) != 1:
        raise R198EvaluationInputError("research-control registry schema is invalid")
    controls = registry.get("controls")
    if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)):
        raise R198EvaluationInputError("research-control registry controls must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(controls):
        row = _mapping(raw, f"research-control row {index}")
        identifier = _text(row.get("opponent_id"), f"research-control row {index}.opponent_id")
        if identifier in by_id:
            raise R198EvaluationInputError("research-control registry repeats an opponent")
        by_id[identifier] = row
    if tuple(by_id) != tuple(identifier for identifier, _ in OFFICIAL_PANEL):
        raise R198EvaluationInputError("research-control registry order is not the official r198 panel")
    rows: list[dict[str, Any]] = []
    for identifier, content_digest in OFFICIAL_PANEL:
        row = by_id[identifier]
        if (
            row.get("content_digest") != content_digest
            or row.get("training_eligible") is not False
            or row.get("formal_eval") is not False
            or row.get("included_in_gate_pass") is not False
            or float(row.get("gate_weight", -1.0)) != 0.0
        ):
            raise R198EvaluationInputError(f"official control {identifier} is not evaluation-only")
        rows.append({"id": identifier, "content_digest": content_digest, "training_eligible": False})
    return identity, rows


def _cohort_and_proof(
    provenance: Mapping[str, Any], registry_identity: Mapping[str, Any], registry_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for opponent_id, content_digest in OFFICIAL_PANEL:
        for candidate_seat in (0, 1):
            for replicate in range(REPLICATES_PER_SEAT):
                cases.append(
                    {
                        "case_id": f"r198-{opponent_id}-seat{candidate_seat}-rep{replicate:03d}",
                        "opponent_id": opponent_id,
                        "content_digest": content_digest,
                        "candidate_seat": candidate_seat,
                        "replicate": replicate,
                        "evaluation_only": True,
                        "training_eligible": False,
                        "replay_eligible": False,
                    }
                )
    bindings = sorted(
        [
            {
                "case_id": row["case_id"],
                "opponent_id": row["opponent_id"],
                "content_digest": row["content_digest"],
                "candidate_seat": row["candidate_seat"],
                "replicate": row["replicate"],
            }
            for row in cases
        ],
        key=lambda row: str(row["case_id"]),
    )
    bindings_sha256 = canonical_digest(bindings)
    train_ids = list(provenance["train_episode_ids"])
    heldout_ids = list(provenance["heldout_episode_ids"])
    # The generated evaluation cells intentionally live in a disjoint source
    # namespace.  The identifiers below are still computed for every case and
    # intersected against the exact rehashed r197 retention sets; a registry
    # flag alone is not evidence of source exclusion.
    evaluation_case_source_ids = sorted(
        "r198-evaluation-only-case/v1/"
        + canonical_digest(
            {
                "case_id": row["case_id"],
                "opponent_id": row["opponent_id"],
                "content_digest": row["content_digest"],
                "candidate_seat": row["candidate_seat"],
                "replicate": row["replicate"],
                "domain": "r198-official-control-evaluation-only",
            }
        )[7:]
        for row in bindings
    )
    if len(evaluation_case_source_ids) != PAIRED_CELL_COUNT or len(set(evaluation_case_source_ids)) != PAIRED_CELL_COUNT:
        raise R198EvaluationInputError("r198 synthetic evaluation source identities are not exactly one-to-one")
    union_ids = sorted(set(train_ids).union(heldout_ids))
    intersection_ids = sorted(set(union_ids).intersection(evaluation_case_source_ids))
    computation = {
        "method": "exact_source_id_set_intersection",
        "evaluation_case_source_kind": "r198_official_control_synthetic_case_identity_v1",
        "r197_train_episode_ids_sha256": _r197_canonical_json_digest(train_ids),
        "r197_train_episode_count": len(train_ids),
        "r197_heldout_episode_ids_sha256": _r197_canonical_json_digest(heldout_ids),
        "r197_heldout_episode_count": len(heldout_ids),
        "r197_union_episode_ids_sha256": canonical_digest(union_ids),
        "r197_union_episode_count": len(union_ids),
        "evaluation_case_source_ids_sha256": canonical_digest(evaluation_case_source_ids),
        "evaluation_case_source_count": len(evaluation_case_source_ids),
        "intersection_episode_ids_sha256": canonical_digest(intersection_ids),
        "intersection_episode_count": len(intersection_ids),
    }
    if computation["intersection_episode_count"] != 0:
        raise R198EvaluationInputError("r198 synthetic cases overlap r197 retained source identities")
    source_identity = canonical_digest(
        {
            "schema": "poke_bot.recursive_turn_planner.r198_evaluation_only_source_identity/v1",
            "candidate_contract_sha256": provenance["candidate_contract_sha256"],
            "r197_completion_receipt_sha256": provenance["completion_receipt"]["sha256"],
            "r197_corpus_manifest_sha256": provenance["corpus_manifest_sha256"],
            "r197_corpus_receipt_sha256": provenance["corpus_receipt_sha256"],
            "r197_selection_plan_sha256": provenance["selection_plan_sha256"],
            "r197_train_selection_sha256": provenance["train_selection_sha256"],
            "r197_heldout_selection_sha256": provenance["heldout_selection_sha256"],
            "research_control_registry_sha256": registry_identity["sha256"],
            "registry_rows": [dict(row) for row in registry_rows],
            "source_exclusion_computation": computation,
            "evaluation_only": True,
            "replay_eligible": False,
            "training_eligible": False,
            "source_identity_overlap_count": 0,
        }
    )
    cohort = {
        "schema": COHORT_SCHEMA,
        "status": "frozen",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "source_identity_sha256": source_identity,
        "registry_rows": [dict(row) for row in registry_rows],
        "cases": cases,
        "case_bindings_sha256": bindings_sha256,
        "source_exclusion_computation": computation,
    }
    proof = {
        "schema": SOURCE_EXCLUSION_SCHEMA,
        "status": "verified",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "all_registry_rows_training_eligible": False,
        "r197_supervised_heldout_calibration_only": True,
        "r197_completion_receipt_sha256": provenance["completion_receipt"]["sha256"],
        "candidate_contract_sha256": provenance["candidate_contract_sha256"],
        "r197_corpus_manifest_sha256": provenance["corpus_manifest_sha256"],
        "r197_corpus_receipt_sha256": provenance["corpus_receipt_sha256"],
        "r197_selection_plan_sha256": provenance["selection_plan_sha256"],
        "r197_train_selection_sha256": provenance["train_selection_sha256"],
        "r197_heldout_selection_sha256": provenance["heldout_selection_sha256"],
        "source_identity_sha256": source_identity,
        "evaluation_case_bindings_sha256": bindings_sha256,
        "source_identity_overlap_count": 0,
        "registry_rows": [dict(row) for row in registry_rows],
        "source_exclusion_computation": computation,
    }
    return cohort, proof


def _validate_capability(capability_path: str | Path) -> _CapabilityBinding:
    receipt = _identity(capability_path, "pairing capability receipt")
    payload = _read_json(receipt["path"], "pairing capability receipt")
    if payload.get("schema") != CAPABILITY_SCHEMA or payload.get("status") != "available" or payload.get("true_rng_pairing_available") is not True:
        raise R198EvaluationInputError("true-RNG pairing capability is unavailable")
    kinds = payload.get("supported_rng_kinds")
    if kinds != ["snapshot"]:
        raise R198EvaluationInputError("r198 pairing capability must authorize snapshot only")
    engine = _verify_identity(payload.get("engine_artifact"), "pairing capability engine")
    source = _verify_identity(payload.get("source_artifact"), "pairing capability source")
    patch = _verify_identity(payload.get("patch_artifact"), "pairing capability patch")
    build = _verify_identity(payload.get("build_artifact"), "pairing capability build")
    artifacts = PairingArtifactSet(
        engine_artifact=engine,
        source_artifact=source,
        patch_artifact=patch,
        build_artifact=build,
    )
    try:
        checked = verify_build_receipt(artifacts)
    except RTPPairingSnapshotError as exc:
        raise R198EvaluationInputError(f"pairing build evidence is invalid: {exc}") from exc
    abi = _mapping(payload.get("abi"), "pairing capability ABI")
    abi_digest = _valid_digest(abi.get("canonical_abi_sha256"), "pairing capability ABI digest")
    abi_without_digest = dict(abi)
    abi_without_digest.pop("canonical_abi_sha256", None)
    if (
        abi.get("name") != SNAPSHOT_ABI_NAME
        or _integer(abi.get("version"), "pairing capability ABI version", minimum=1) != SNAPSHOT_ABI_VERSION
        or abi_digest != snapshot_abi_sha256()
        or canonical_digest(abi_without_digest) != abi_digest
    ):
        raise R198EvaluationInputError("pairing capability ABI is not exact v2")
    for key, expected in snapshot_abi_contract().items():
        if abi_without_digest.get(key) != expected:
            raise R198EvaluationInputError(f"pairing capability ABI differs at {key}")
    probe = _verify_identity(payload.get("probe"), "pairing capability probe")
    probe_payload = _read_json(probe["path"], "pairing capability probe")
    if probe_payload.get("schema") != PROBE_SCHEMA or probe_payload.get("status") != "passed":
        raise R198EvaluationInputError("pairing capability lacks a passing probe")
    for field, identity in (
        ("engine_artifact_sha256", checked.engine_artifact),
        ("source_artifact_sha256", checked.source_artifact),
        ("patch_artifact_sha256", checked.patch_artifact),
        ("build_artifact_sha256", checked.build_artifact),
    ):
        if probe_payload.get(field) != identity["sha256"]:
            raise R198EvaluationInputError(f"pairing probe differs at {field}")
    if probe_payload.get("canonical_abi_sha256") != abi_digest:
        raise R198EvaluationInputError("pairing probe ABI differs from capability")
    for field in (
        "device_rand_false_verified",
        "requested_seed_only_rejected",
        "duplicate_restore_independent_handles",
        "all_arms_restored_or_replayed",
        "divergent_policy_true_pairing_passed",
        "delayed_restore_transcript_passed",
        "cross_process_restore_passed",
    ):
        if probe_payload.get(field) is not True:
            raise R198EvaluationInputError(f"pairing probe did not pass {field}")
    return _CapabilityBinding(
        receipt=receipt,
        engine_artifact=dict(checked.engine_artifact),
        source_artifact=dict(checked.source_artifact),
        patch_artifact=dict(checked.patch_artifact),
        build_artifact=dict(checked.build_artifact),
        abi_sha256=abi_digest,
    )


def _native_capturer(binding: _CapabilityBinding) -> SnapshotCapturer:
    artifacts = PairingArtifactSet(
        engine_artifact=binding.engine_artifact,
        source_artifact=binding.source_artifact,
        patch_artifact=binding.patch_artifact,
        build_artifact=binding.build_artifact,
    )
    try:
        engine = RtpPairingSnapshotEngine(binding.engine_artifact["path"])
        engine.require_bound_artifacts(artifacts)
    except RTPPairingSnapshotError as exc:
        raise R198EvaluationInputError(f"cannot load exact v2 pairing engine: {exc}") from exc

    def capture(deck0: Sequence[int], deck1: Sequence[int], requested_seed: int) -> CapturedSnapshot:
        try:
            with engine.capture_cell_snapshot(deck0, deck1, requested_seed) as snapshot:
                blob = bytes(snapshot.serialized_bytes)
                return CapturedSnapshot(
                    serialized_bytes=blob,
                    snapshot_id=snapshot.snapshot_id,
                    fingerprint_sha256=snapshot.fingerprint_sha256,
                    fingerprint_bytes=snapshot.bytes,
                )
        except RTPPairingSnapshotError as exc:
            raise R198EvaluationInputError(f"native r198 snapshot capture failed: {exc}") from exc

    return capture


def _native_fixture_observation_extractor(
    binding: _CapabilityBinding,
) -> FixtureObservationExtractor:
    """Restore a preflight fixture only through its immutable native seal."""

    artifacts = PairingArtifactSet(
        engine_artifact=binding.engine_artifact,
        source_artifact=binding.source_artifact,
        patch_artifact=binding.patch_artifact,
        build_artifact=binding.build_artifact,
    )
    try:
        engine = RtpPairingSnapshotEngine(binding.engine_artifact["path"])
        engine.require_bound_artifacts(artifacts)
    except RTPPairingSnapshotError as exc:
        raise R198EvaluationInputError(
            "cannot load exact v2 pairing engine for planner preflight fixtures"
        ) from exc

    def extract(snapshot_seal: Mapping[str, Any]) -> Mapping[str, Any]:
        seal = _verify_identity(snapshot_seal, "planner preflight snapshot seal")
        try:
            with engine.restore_sealed_snapshot_manifest(seal["path"]) as battle:
                observation = battle.observation()
        except RTPPairingSnapshotError as exc:
            raise R198EvaluationInputError(
                "cannot restore sealed r198 planner preflight fixture"
            ) from exc
        return _mapping(observation, "restored r198 planner preflight observation")

    return extract


def _check_captured_snapshot(captured: CapturedSnapshot, label: str) -> None:
    if not captured.serialized_bytes or len(captured.serialized_bytes) > 64 * 1024 * 1024:
        raise R198EvaluationInputError(f"{label} has invalid opaque snapshot bytes")
    if not captured.snapshot_id or any(token in captured.snapshot_id for token in ("/", "\\", "..")):
        raise R198EvaluationInputError(f"{label} has an unsafe snapshot ID")
    if captured.fingerprint_sha256 != _sha256_bytes(captured.serialized_bytes):
        raise R198EvaluationInputError(f"{label} fingerprint does not bind captured bytes")
    if captured.fingerprint_bytes != len(captured.serialized_bytes):
        raise R198EvaluationInputError(f"{label} fingerprint byte length does not bind captured bytes")


def _seal_snapshot(
    *,
    target: Path,
    blob_target: Path,
    case_binding_target: Path,
    material_id: str,
    cell_id: str,
    case: Mapping[str, Any],
    candidate_deck: Mapping[str, Any],
    opponent_deck: Mapping[str, Any],
    candidate_seat: int,
    capability: _CapabilityBinding,
    cohort_identity: Mapping[str, Any],
    source_proof_identity: Mapping[str, Any],
    evaluation_case_bindings_sha256: str,
    requested_seed: int,
    captured: CapturedSnapshot,
    purpose: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _check_captured_snapshot(captured, material_id)
    snapshot_identity = _write_immutable_bytes(blob_target, captured.serialized_bytes, f"snapshot {material_id}")
    if snapshot_identity["sha256"] != captured.fingerprint_sha256 or snapshot_identity["bytes"] != captured.fingerprint_bytes:
        raise R198EvaluationInputError(f"snapshot {material_id} identity differs from its native fingerprint")
    opponent_id = _text(case.get("opponent_id"), f"{material_id} opponent")
    opponent_content_digest = _valid_digest(case.get("content_digest"), f"{material_id} opponent digest")
    replicate = _integer(case.get("replicate"), f"{material_id} replicate")
    candidate_case_seat = _integer(case.get("candidate_seat"), f"{material_id} candidate seat")
    if candidate_case_seat != candidate_seat or candidate_seat not in {0, 1}:
        raise R198EvaluationInputError(f"{material_id} candidate seat differs from its cohort case")
    case_id = _text(case.get("case_id"), f"{material_id} case id")
    seat0_deck = candidate_deck if candidate_seat == 0 else opponent_deck
    seat1_deck = opponent_deck if candidate_seat == 0 else candidate_deck
    # The native restore wrapper treats this as a distinct immutable evidence
    # object.  Keeping it separate from the seal makes the gameplay-cell
    # identity visible before the opaque snapshot decoder is ever reached.
    case_binding = {
        "schema": "poke_bot.recursive_turn_planner.r198_pairing_case_binding/v1",
        "status": "sealed",
        "created_at_utc": _utc_now(),
        "cell_id": cell_id,
        "case_id": case_id,
        "opponent_id": opponent_id,
        "opponent_content_digest": opponent_content_digest,
        "seat": candidate_seat,
        "candidate_seat": candidate_seat,
        "replicate": replicate,
        "debug_seed": requested_seed,
        "ordered_deck_identities": [seat0_deck, seat1_deck],
        "candidate_deck_identity": candidate_deck,
        "opponent_deck_identity": opponent_deck,
        "cohort_identity": cohort_identity,
        "source_exclusion_identity": source_proof_identity,
        "evaluation_case_bindings_sha256": _valid_digest(
            evaluation_case_bindings_sha256,
            f"{material_id} evaluation case bindings digest",
        ),
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "promotion_eligible": False,
        "kaggle_submission_authorized": False,
    }
    case_binding_identity = _write_immutable_json(
        case_binding_target, case_binding, f"pairing case binding {material_id}"
    )
    seal = {
        "schema": SNAPSHOT_SEAL_SCHEMA,
        "status": "sealed",
        "created_at_utc": _utc_now(),
        "purpose": purpose,
        "rng_material_id": material_id,
        "cell_id": cell_id,
        # This is the evaluator's unique material identity.  Preserve the
        # native fingerprint-derived label separately for diagnostics; it must
        # never substitute for the schedule's cell identity.
        "snapshot_id": material_id,
        "native_snapshot_id": captured.snapshot_id,
        "snapshot_artifact": snapshot_identity,
        "snapshot_artifact_sha256": snapshot_identity["sha256"],
        "snapshot_artifact_bytes": snapshot_identity["bytes"],
        "snapshot_fingerprint_sha256": captured.fingerprint_sha256,
        "snapshot_fingerprint_bytes": captured.fingerprint_bytes,
        "engine_artifact_sha256": capability.engine_artifact["sha256"],
        "source_artifact_sha256": capability.source_artifact["sha256"],
        "patch_artifact_sha256": capability.patch_artifact["sha256"],
        "build_artifact_sha256": capability.build_artifact["sha256"],
        "canonical_abi_sha256": capability.abi_sha256,
        "capture_boundary": SNAPSHOT_CAPTURE_BOUNDARY,
        "boundary_tag": SNAPSHOT_BOUNDARY_TAG,
        "rng_kind": "snapshot",
        "candidate_deck_sha256": candidate_deck["sha256"],
        "candidate_deck_order_sha256": candidate_deck["sha256"],
        "opponent_id": opponent_id,
        "opponent_content_digest": opponent_content_digest,
        "opponent_deck_sha256": opponent_deck["sha256"],
        "opponent_deck_order_sha256": opponent_deck["sha256"],
        "candidate_seat": candidate_seat,
        "seat0_deck_sha256": seat0_deck["sha256"],
        "seat1_deck_sha256": seat1_deck["sha256"],
        "replicate": replicate,
        "evaluation_case_id": case_id,
        "evaluation_only_cohort_sha256": cohort_identity["sha256"],
        "source_exclusion_proof_sha256": source_proof_identity["sha256"],
        "case_binding_artifact": case_binding_identity,
        "case_binding_artifact_sha256": case_binding_identity["sha256"],
        "requested_seed_audit_only": requested_seed,
        "requested_seed_is_pairing_proof": False,
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "promotion_eligible": False,
        "kaggle_submission_authorized": False,
    }
    seal_identity = _write_immutable_json(target, seal, f"snapshot seal {material_id}")
    material = {
        "id": material_id,
        "kind": "snapshot",
        "snapshot_artifact": snapshot_identity,
        "seal": seal_identity,
        "opponent_id": opponent_id,
        "candidate_seat": candidate_seat,
        "replicate": replicate,
        "evaluation_case_id": case_id,
        "requested_seed_audit_only": requested_seed,
    }
    return material, seal_identity


def _candidate_evaluation_binding(raw: Any) -> dict[str, Any]:
    """Require evaluator-v2's exact semantic candidate binding before capture."""

    value = _mapping(raw, "base specification candidate_evaluation_binding")
    expected: dict[str, Any] = {
        "schema": R198_CANDIDATE_EVALUATION_BINDING_SCHEMA,
        "status": "bound",
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "parent_checkpoint_sha256": R198_PARENT_CHECKPOINT_SHA256,
        "sidecar_sha256": R198_SIDECAR_SHA256,
        "sidecar_config_sha256": R198_SIDECAR_CONFIG_SHA256,
        "deck_file_sha256": R198_DECK_FILE_SHA256,
        "deck_cards_sha256": R198_DECK_CARDS_SHA256,
        "matchup_tree_sha256": R198_MATCHUP_TREE_SHA256,
        "sizing_profile": "pure_rl_r197",
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "required_neural_passes": {"normal": 6, "forced_replan": 5},
    }
    if set(value) != set(expected):
        raise R198EvaluationInputError(
            "candidate_evaluation_binding must contain exactly the evaluator-v2 semantic keys"
        )
    for key, required in expected.items():
        if value.get(key) != required:
            raise R198EvaluationInputError(
                f"candidate_evaluation_binding differs at {key}"
            )
    return value


def _canonical_runtime_profile_payload(arm: str, raw: Any) -> dict[str, Any]:
    """Accept only the production-factory profile payload for one arm.

    Profile construction belongs to the factory.  The materializer's role is
    limited to sealing those exact payload bytes under its own immutable input
    root, never independently rebuilding an r198 runtime profile.
    """

    payload = _mapping(raw, f"base {arm} runtime_profile_payload")
    try:
        from poke_bot.rtp_r198_production_factory import r198_runtime_profile_payload
    except ImportError as exc:  # pragma: no cover - factory is required in production
        raise R198EvaluationInputError("r198 production factory profile builder is unavailable") from exc
    expected = r198_runtime_profile_payload(arm)
    if payload != expected:
        raise R198EvaluationInputError(
            f"base {arm} runtime_profile_payload differs from the production factory"
        )
    return payload


def _base_spec(base_spec_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, list[int]]]:
    base_identity = _identity(base_spec_path, "sealed evaluator base specification")
    base = _read_json(base_identity["path"], "sealed evaluator base specification")
    shared = _mapping(base.get("shared_artifacts"), "base specification shared_artifacts")
    for name in ("parent_checkpoint", "deck", "matchup_tree"):
        if name not in shared:
            raise R198EvaluationInputError(f"base specification lacks shared_artifacts.{name}")
        shared[name] = _verify_identity(shared[name], f"base shared artifact {name}")
    candidate_deck = shared["deck"]
    candidate_cards = _deck_cards(candidate_deck, "candidate deck")
    production_factory = _mapping(base.get("production_factory"), "base specification production_factory")
    production_factory["matchup_adapter_registry"] = (
        _snapshot_local_matchup_adapter_registry(production_factory)
    )
    factory_artifacts = _mapping(production_factory.get("artifacts"), "production factory artifacts")
    factory_deck = _verify_identity(factory_artifacts.get("deck"), "production factory candidate deck")
    if factory_deck != candidate_deck:
        raise R198EvaluationInputError("production factory candidate deck differs from shared evaluator deck")
    candidate_binding = _candidate_evaluation_binding(
        base.get("candidate_evaluation_binding")
    )
    evaluation_cg_closure = _mapping(
        base.get("evaluation_cg_closure"), "base specification evaluation_cg_closure"
    )
    if set(evaluation_cg_closure) != {"receipt", "runtime_library"}:
        raise R198EvaluationInputError(
            "base specification evaluation_cg_closure must contain exactly receipt and runtime_library"
        )
    evaluation_cg_closure["receipt"] = _verify_identity(
        evaluation_cg_closure.get("receipt"), "base evaluation CG closure receipt"
    )
    runtime_library = _verify_identity(
        evaluation_cg_closure.get("runtime_library"),
        "base evaluation CG snapshot-local runtime library",
    )
    if (
        Path(runtime_library["path"]).name != "libcg.so"
        or stat.S_IMODE(os.lstat(runtime_library["path"]).st_mode) != 0o444
    ):
        raise R198EvaluationInputError(
            "base evaluation CG runtime_library must be the physical 0444 snapshot-local libcg.so"
        )
    factory_cg = _mapping(
        production_factory.get("evaluation_cg"), "production factory evaluation_cg"
    )
    factory_library = _verify_identity(
        factory_cg.get("library"), "production factory evaluation CG library"
    )
    if factory_library != runtime_library:
        raise R198EvaluationInputError(
            "base evaluation CG runtime_library differs from production factory snapshot-local library"
        )
    evaluation_cg_closure["runtime_library"] = runtime_library
    raw_arms = _mapping(base.get("arms"), "base specification arms")
    if set(raw_arms) != set(R198_ARMS):
        raise R198EvaluationInputError("base specification does not contain the exact three r198 arms")
    arms: dict[str, dict[str, Any]] = {}
    for arm in R198_ARMS:
        row = _mapping(raw_arms.get(arm), f"base {arm} arm")
        if "runtime_profile" in row:
            raise R198EvaluationInputError(
                "base specification must provide factory runtime_profile_payloads, not profile identities"
            )
        row["runtime_profile_payload"] = _canonical_runtime_profile_payload(
            arm, row.get("runtime_profile_payload")
        )
        arms[arm] = row
    raw_opponents = base.get("opponents")
    if not isinstance(raw_opponents, Sequence) or isinstance(raw_opponents, (str, bytes)):
        raise R198EvaluationInputError("base specification opponents must be a list")
    expected = dict(OFFICIAL_PANEL)
    decks: dict[str, dict[str, Any]] = {}
    cards: dict[str, list[int]] = {"candidate": candidate_cards}
    seen: set[str] = set()
    for index, raw in enumerate(raw_opponents):
        opponent = _mapping(raw, f"base opponent {index}")
        opponent_id = _text(opponent.get("id"), f"base opponent {index}.id")
        if opponent_id in seen or expected.get(opponent_id) != opponent.get("content_digest"):
            raise R198EvaluationInputError("base specification opponent panel is not the official r198 panel")
        seen.add(opponent_id)
        deck = _verify_identity(opponent.get("deck"), f"base opponent {opponent_id} deck")
        decks[opponent_id] = deck
        cards[opponent_id] = _deck_cards(deck, f"opponent {opponent_id} deck")
        opponent["deck"] = deck
        raw_opponents[index] = opponent
    if seen != set(expected):
        raise R198EvaluationInputError("base specification does not include the exact four official opponents")
    base["shared_artifacts"] = shared
    base["opponents"] = list(raw_opponents)
    base["production_factory"] = production_factory
    base["candidate_evaluation_binding"] = candidate_binding
    base["evaluation_cg_closure"] = evaluation_cg_closure
    base["arms"] = arms
    return base, candidate_deck, decks, cards


def _default_seed_provider() -> int:
    return secrets.randbits(32)


def _preflight_runner_default(
    preflight_input: Mapping[str, Any] | str | Path, output_path: Path
) -> Path:
    try:
        from poke_bot.rtp_r198_production_factory import run_r198_planner_preflight
    except ImportError as exc:  # pragma: no cover - factory lands separately
        raise R198EvaluationInputError("r198 production factory preflight is unavailable") from exc
    result = run_r198_planner_preflight(preflight_input, output_path)
    return Path(result)


def _validate_preflight_receipt(
    path: str | Path,
    *,
    direct_profile: Mapping[str, Any],
    recursive_profile: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    matchup_adapter_registry: Mapping[str, Any],
    preflight_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = _identity(path, "planner pass preflight receipt")
    payload = _read_json(identity["path"], "planner pass preflight receipt")
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("status") != "passed":
        raise R198EvaluationInputError("planner pass preflight did not pass")
    expected = {
        "sidecar_sha256": sidecar["sha256"],
        "direct_runtime_profile_sha256": direct_profile["sha256"],
        "recursive_runtime_profile_sha256": recursive_profile["sha256"],
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "normal_probe_observed_neural_passes": 6,
        "forced_replan_probe_observed_neural_passes": 5,
        "neural_budget_failures": 0,
        "matchup_adapter_registry_sha256": matchup_adapter_registry["sha256"],
        "matchup_adapter_slot_registry_digest": (
            R198_MATCHUP_ADAPTER_SLOT_REGISTRY_DIGEST
        ),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise R198EvaluationInputError(f"planner pass preflight differs at {key}")
    if payload.get("normal_probe_completed") is not True or payload.get("forced_replan_probe_completed") is not True:
        raise R198EvaluationInputError("planner pass preflight did not complete both probes")
    if preflight_input is not None:
        bound = _verify_identity(
            payload.get("preflight_input"), "planner pass preflight bound input"
        )
        if bound != dict(preflight_input):
            raise R198EvaluationInputError(
                "planner pass preflight receipt is not bound to its sealed input"
            )
    return identity


def _authority_payload(
    *,
    manifest: Mapping[str, Any],
    materials: Mapping[str, Any],
    cohort: Mapping[str, Any],
    proof: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_AUTHORITY_SCHEMA,
        "status": "authorized_evaluation_only",
        # The runner takes the manifest as a separate immutable input, so this
        # digest is the authority's exact binding rather than an authority to
        # alter the manifest or any production runtime.
        "manifest_sha256": manifest["sha256"],
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }


def materialize_r198_evaluation_inputs(
    *,
    completion_receipt: str | Path,
    research_control_registry: str | Path,
    pairing_capability: str | Path,
    evaluator_base_spec: str | Path,
    output_root: str | Path,
    run_nonce: str | None = None,
    snapshot_capturer: SnapshotCapturer | None = None,
    preflight_runner: PreflightRunner | None = None,
    seed_provider: SeedProvider | None = None,
    fixture_observation_extractor: FixtureObservationExtractor | None = None,
) -> dict[str, Any]:
    """Materialize exactly 1,000 sealed r198 cell inputs plus two preflight fixtures.

    ``snapshot_capturer``, ``fixture_observation_extractor``, and
    ``preflight_runner`` are injection seams for focused tests only.
    Production callers omit them: the exact capability's native engine captures
    snapshots, restores the preflight fixture seals to obtain the observations,
    and the sealed production factory runs the actual normal/forced planner
    preflight.
    """

    provenance = _candidate_provenance(completion_receipt)
    registry_identity, registry_rows = _official_registry(research_control_registry)
    capability = _validate_capability(pairing_capability)
    base, candidate_deck, opponent_decks, deck_cards = _base_spec(evaluator_base_spec)
    matchup_adapter_registry = _mapping(
        _mapping(base["production_factory"], "production factory").get(
            "matchup_adapter_registry"
        ),
        "production_factory.matchup_adapter_registry",
    )
    cohort_payload, proof_payload = _cohort_and_proof(provenance, registry_identity, registry_rows)
    if run_nonce is None:
        run_nonce = secrets.token_hex(12)
    if not run_nonce or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_nonce):
        raise R198EvaluationInputError("run_nonce must be a non-empty safe path token")
    request_identity = canonical_digest(
        {
            "schema": REQUEST_SCHEMA,
            "run_nonce": run_nonce,
            "completion_receipt_sha256": provenance["completion_receipt"]["sha256"],
            "registry_sha256": registry_identity["sha256"],
            "capability_sha256": capability.receipt["sha256"],
            "base_spec_sha256": _identity(evaluator_base_spec, "sealed evaluator base specification")["sha256"],
        }
    )
    base_root = _ensure_physical_directory(output_root, "r198 evaluation input root")
    target = base_root / f"r198-evaluation-inputs-{request_identity[7:31]}"
    if os.path.lexists(target):
        raise R198EvaluationInputError(f"refusing to reuse or overwrite evaluation input root: {target}")
    # Construct any capability-bound native helpers before an output directory
    # exists.  If the exact engine cannot load, there is no partially mutable
    # evidence tree to leave behind.
    capture = snapshot_capturer or _native_capturer(capability)
    extract_fixture_observation = (
        fixture_observation_extractor
        or _native_fixture_observation_extractor(capability)
    )
    next_seed = seed_provider or _default_seed_provider
    try:
        target.mkdir(mode=0o755)
    except OSError as exc:
        raise R198EvaluationInputError(f"cannot create evaluation input root: {target}") from exc

    try:
        cohort_identity = _write_immutable_json(target / "cohort" / "evaluation-only-cohort.json", cohort_payload, "evaluation-only cohort")
        proof_payload["evaluation_only_cohort_sha256"] = cohort_identity["sha256"]
        proof_payload["evaluation_only_cohort_bytes"] = cohort_identity["bytes"]
        proof_identity = _write_immutable_json(target / "cohort" / "source-exclusion-proof.json", proof_payload, "source-exclusion proof")

        cases = {(
            str(row["opponent_id"]), int(row["candidate_seat"]), int(row["replicate"])
        ): row for row in cohort_payload["cases"]}
        if len(cases) != PAIRED_CELL_COUNT:
            raise R198EvaluationInputError("frozen cohort does not contain exactly 1,000 unique cases")
        materials: list[dict[str, Any]] = []
        scored_seeds: set[int] = set()
        material_index = 0
        for opponent_id, _ in OFFICIAL_PANEL:
            for candidate_seat in (0, 1):
                for replicate in range(REPLICATES_PER_SEAT):
                    case = cases[(opponent_id, candidate_seat, replicate)]
                    seed = _integer(next_seed(), "requested snapshot seed")
                    if seed > 0xFFFFFFFF:
                        raise R198EvaluationInputError("requested snapshot seed exceeds uint32")
                    if seed in scored_seeds:
                        raise R198EvaluationInputError(
                            "r198 evaluation snapshot debug seeds must be globally unique"
                        )
                    scored_seeds.add(seed)
                    deck0 = deck_cards["candidate"] if candidate_seat == 0 else deck_cards[opponent_id]
                    deck1 = deck_cards[opponent_id] if candidate_seat == 0 else deck_cards["candidate"]
                    captured = capture(deck0, deck1, seed)
                    material_id = f"rng-cell-{material_index:04d}"
                    cell_id = f"cell-{material_index:06d}"
                    material, _ = _seal_snapshot(
                        target=target / ".private" / "rng-seals" / f"{material_id}.seal.json",
                        blob_target=target / ".private" / "rng-snapshots" / f"{material_id}.bin",
                        case_binding_target=target / ".private" / "case-bindings" / f"{material_id}.json",
                        material_id=material_id,
                        cell_id=cell_id,
                        case=case,
                        candidate_deck=candidate_deck,
                        opponent_deck=opponent_decks[opponent_id],
                        candidate_seat=candidate_seat,
                        capability=capability,
                        cohort_identity=cohort_identity,
                        source_proof_identity=proof_identity,
                        evaluation_case_bindings_sha256=str(cohort_payload["case_bindings_sha256"]),
                        requested_seed=seed,
                        captured=captured,
                        purpose="r198_three_arm_evaluation_cell",
                    )
                    materials.append(material)
                    material_index += 1
        if len(materials) != PAIRED_CELL_COUNT:
            raise R198EvaluationInputError("materializer did not create exactly 1,000 paired snapshot materials")
        if len(scored_seeds) != PAIRED_CELL_COUNT:
            raise R198EvaluationInputError("r198 evaluation snapshot seed set is not exactly 1,000")
        if len({row["snapshot_artifact"]["sha256"] for row in materials}) != PAIRED_CELL_COUNT or len({row["seal"]["sha256"] for row in materials}) != PAIRED_CELL_COUNT:
            raise R198EvaluationInputError("r198 cell snapshots or seals are not one-to-one")

        arms = _mapping(base.get("arms"), "base specification arms")
        direct = _mapping(
            arms.get("direct_bridge_recursive_disabled"), "direct bridge arm"
        )
        recursive = _mapping(arms.get("recursive_rtp"), "recursive arm")
        direct_runtime_artifact = _verify_identity(
            direct.get("runtime_artifact"), "direct runtime artifact"
        )
        recursive_runtime_artifact = _verify_identity(
            recursive.get("runtime_artifact"), "recursive runtime artifact"
        )
        direct_sidecar = _verify_identity(direct.get("rtp_sidecar"), "direct RTP sidecar")
        recursive_sidecar = _verify_identity(
            recursive.get("rtp_sidecar"), "recursive RTP sidecar"
        )
        if direct_sidecar["sha256"] != recursive_sidecar["sha256"]:
            raise R198EvaluationInputError("direct and recursive arms do not share the exact sidecar")

        # Preflight fixtures are deliberately separate from the statistical
        # 1,000-cell cohort.  They have no score, replay, training, promotion,
        # or serving authority.
        preflight_cases: dict[str, dict[str, Any]] = {}
        preflight_seeds: set[int] = set()
        preflight_fixture_root = target / "preflight-fixture-inputs"
        fixture_case = cases[(OFFICIAL_PANEL[0][0], 0, 0)]
        fixture_opponent_id = OFFICIAL_PANEL[0][0]
        # The factory verifies preflight evidence against an independent
        # read-only evaluation-input root.  Re-publish the exact official deck
        # bytes there (same SHA/byte identity) rather than passing a writable
        # package path or a path outside that verifier's root.
        fixture_opponent_deck = _write_immutable_bytes(
            preflight_fixture_root / "opponent-decks" / f"{fixture_opponent_id}.deck",
            Path(opponent_decks[fixture_opponent_id]["path"]).read_bytes(),
            "planner preflight official opponent deck",
        )
        if (
            fixture_opponent_deck["sha256"] != opponent_decks[fixture_opponent_id]["sha256"]
            or fixture_opponent_deck["bytes"] != opponent_decks[fixture_opponent_id]["bytes"]
        ):
            raise R198EvaluationInputError(
                "planner preflight opponent deck copy differs from the frozen official deck"
            )
        sealed_arms: dict[str, dict[str, Any]] = {}
        for arm in R198_ARMS:
            arm_row = copy.deepcopy(_mapping(arms[arm], f"base {arm} arm"))
            profile_payload = arm_row.pop("runtime_profile_payload", None)
            profile_identity = _write_immutable_observation(
                preflight_fixture_root / "runtime-profiles" / f"{arm}.json",
                _canonical_runtime_profile_payload(arm, profile_payload),
                f"planner preflight {arm} runtime profile",
            )
            arm_row["runtime_profile"] = profile_identity
            sealed_arms[arm] = arm_row
        direct_profile = sealed_arms["direct_bridge_recursive_disabled"]["runtime_profile"]
        recursive_profile = sealed_arms["recursive_rtp"]["runtime_profile"]
        for name, expected_mode in (("normal", "recursive_plan"), ("forced_replan", "forced_replan")):
            seed = _integer(next_seed(), f"{name} preflight snapshot seed")
            if seed > 0xFFFFFFFF:
                raise R198EvaluationInputError("preflight snapshot seed exceeds uint32")
            if seed in scored_seeds or seed in preflight_seeds:
                raise R198EvaluationInputError(
                    "preflight snapshot debug seeds must be distinct from all scored-cell seeds"
                )
            preflight_seeds.add(seed)
            captured = capture(deck_cards["candidate"], deck_cards[fixture_opponent_id], seed)
            fixture_id = f"planner-preflight-{name}"
            fixture, fixture_seal = _seal_snapshot(
                target=preflight_fixture_root / "seals" / f"{fixture_id}.seal.json",
                blob_target=preflight_fixture_root / "snapshots" / f"{fixture_id}.bin",
                case_binding_target=preflight_fixture_root / "case-bindings" / f"{fixture_id}.json",
                material_id=fixture_id,
                cell_id=fixture_id,
                case=fixture_case,
                candidate_deck=candidate_deck,
                opponent_deck=fixture_opponent_deck,
                candidate_seat=0,
                capability=capability,
                cohort_identity=cohort_identity,
                source_proof_identity=proof_identity,
                evaluation_case_bindings_sha256=str(cohort_payload["case_bindings_sha256"]),
                requested_seed=seed,
                captured=captured,
                purpose="r198_planner_pass_preflight_only_non_scored",
            )
            observation = _mapping(
                extract_fixture_observation(fixture_seal),
                f"{name} restored planner preflight observation",
            )
            observation_identity = _write_immutable_observation(
                preflight_fixture_root / "observations" / f"{fixture_id}.json",
                observation,
                f"{name} restored planner preflight observation",
            )
            preflight_cases[name] = {
                "snapshot_artifact": fixture["snapshot_artifact"],
                "snapshot_seal": fixture_seal,
                "observation": observation_identity,
                "observation_sha256": observation_identity["sha256"],
                "observation_source_snapshot_seal_sha256": fixture_seal["sha256"],
                "candidate_deck": candidate_deck,
                "opponent_deck": fixture_opponent_deck,
                "candidate_seat": 0,
                "opponent_id": fixture_opponent_id,
                "replicate": int(fixture_case["replicate"]),
                "expected_mode": expected_mode,
                "evaluation_only": True,
                "training_eligible": False,
                "replay_eligible": False,
                "scored": False,
            }
        if len(preflight_seeds) != 2:
            raise R198EvaluationInputError("r198 requires exactly two distinct planner preflight seeds")
        if len(scored_seeds.union(preflight_seeds)) != PAIRED_CELL_COUNT + 2:
            raise R198EvaluationInputError(
                "r198 scored and preflight snapshot debug seeds are not globally unique"
            )
        # The production factory receives this sealed, read-only subtree as
        # its only evaluation-input root.  Keeping it separate allows the
        # materializer to write the later manifest/authority records without
        # making a fixture writable again.
        _freeze_tree(preflight_fixture_root)
        production_factory = copy.deepcopy(_mapping(base["production_factory"], "production factory"))
        production_factory["evaluation_inputs_root"] = str(preflight_fixture_root)

        preflight_input = {
            "schema": PREFLIGHT_INPUT_SCHEMA,
            "status": "sealed",
            "created_at_utc": _utc_now(),
            "candidate": {
                "candidate_contract_sha256": provenance["candidate_contract_sha256"],
                "r197_completion_receipt": provenance["completion_receipt"],
                "parent_checkpoint": base["shared_artifacts"]["parent_checkpoint"],
                "deck": candidate_deck,
                "matchup_tree": base["shared_artifacts"]["matchup_tree"],
            },
            "production_factory": production_factory,
            "arms": {
                "direct_bridge_recursive_disabled": {
                    "runtime_profile": direct_profile,
                    "runtime_artifact": direct_runtime_artifact,
                    "rtp_sidecar": direct_sidecar,
                },
                "recursive_rtp": {
                    "runtime_profile": recursive_profile,
                    "runtime_artifact": recursive_runtime_artifact,
                    "rtp_sidecar": recursive_sidecar,
                },
            },
            "pairing_capability": {"receipt": capability.receipt},
            "fixtures": preflight_cases,
            "evaluation_only": True,
            "training_eligible": False,
            "replay_eligible": False,
            "serving_eligible": False,
            "action_authority_enabled": False,
            "promotion_eligible": False,
            "kaggle_submission_authorized": False,
        }
        preflight_input_identity = _write_immutable_json(target / "preflight" / "planner-pass-preflight-input.json", preflight_input, "planner preflight input")
        preflight_output = target / "preflight" / "planner-pass-preflight.json"
        if preflight_runner is None:
            # The actual factory must re-open the immutable input rather than
            # trust the materializer's in-memory mapping.  Test seams retain a
            # mapping for small focused fake preflight runners.
            preflight_path = _preflight_runner_default(
                preflight_input_identity["path"], preflight_output
            )
        else:
            preflight_path = Path(preflight_runner(preflight_input, preflight_output))
        preflight_identity = _validate_preflight_receipt(
            preflight_path,
            direct_profile=direct_profile,
            recursive_profile=recursive_profile,
            sidecar=direct_sidecar,
            matchup_adapter_registry=matchup_adapter_registry,
            preflight_input=(
                preflight_input_identity if preflight_runner is None else None
            ),
        )

        materials_payload = {
            "schema": MATERIALS_SCHEMA,
            "status": "sealed",
            "created_at_utc": _utc_now(),
            "candidate_contract_sha256": provenance["candidate_contract_sha256"],
            "evaluation_only_cohort": cohort_identity,
            "source_exclusion_proof": proof_identity,
            "pairing_capability": {"receipt": capability.receipt},
            "planner_preflight_input": preflight_input_identity,
            "planner_preflight_receipt": preflight_identity,
            "rng_materials": materials,
            "preflight_fixtures": preflight_cases,
            "paired_cell_count": PAIRED_CELL_COUNT,
            "scored_requested_seed_set_sha256": canonical_digest(sorted(scored_seeds)),
            "scored_requested_seed_count": len(scored_seeds),
            "preflight_requested_seed_set_sha256": canonical_digest(sorted(preflight_seeds)),
            "preflight_requested_seed_count": len(preflight_seeds),
            "all_requested_seed_set_sha256": canonical_digest(sorted(scored_seeds.union(preflight_seeds))),
            "all_requested_seed_count": len(scored_seeds.union(preflight_seeds)),
            "training_eligible": False,
            "replay_eligible": False,
            "serving_eligible": False,
            "action_authority_enabled": False,
            "selector_authority": False,
            "promotion_eligible": False,
            "submission_eligible": False,
            "kaggle_submission_authorized": False,
        }
        materials_identity = _write_immutable_json(target / "rng-materials.json", materials_payload, "r198 RNG materials manifest")

        final_spec = copy.deepcopy(base)
        final_spec["production_factory"] = production_factory
        final_spec["arms"] = sealed_arms
        shared = _mapping(final_spec.get("shared_artifacts"), "final evaluator shared artifacts")
        shared.update(
            {
                "evaluation_only_cohort": cohort_identity,
                "r197_completion_receipt": provenance["completion_receipt"],
                "planner_preflight_receipt": preflight_identity,
                "research_control_registry": registry_identity,
            }
        )
        final_spec["shared_artifacts"] = shared
        final_spec["rng_materials"] = materials
        final_spec["pairing_capability"] = {"receipt": capability.receipt}
        final_spec["source_exclusion_proof"] = {"receipt": proof_identity}
        final_spec["replicates_per_seat"] = REPLICATES_PER_SEAT
        final_spec["evaluation_input_materials"] = materials_identity
        final_spec["evaluation_isolation"] = {
            "training_eligible": False,
            "replay_eligible": False,
            "serving_eligible": False,
            "action_authority_enabled": False,
            "selector_authority": False,
            "promotion_eligible": False,
            "submission_eligible": False,
            "kaggle_submission_authorized": False,
            "requested_seed_is_pairing_proof": False,
        }
        final_spec_identity = _write_immutable_json(target / "prepared-evaluator-spec.json", final_spec, "prepared evaluator specification")
        try:
            from poke_bot.rtp_three_arm_evaluation import prepare_three_arm_manifest_from_spec
        except ImportError as exc:  # pragma: no cover - repository module is required in production
            raise R198EvaluationInputError("evaluator-v2 manifest compiler is unavailable") from exc
        try:
            prepared_manifest_path = prepare_three_arm_manifest_from_spec(
                final_spec,
                output_path=target / "prepared-evaluator-manifest-v2.json",
            )
        except Exception as exc:
            raise R198EvaluationInputError(f"evaluator-v2 manifest preparation failed: {exc}") from exc
        prepared_manifest_identity = _identity(prepared_manifest_path, "prepared evaluator-v2 manifest")
        authority_payload = _authority_payload(
            manifest=prepared_manifest_identity,
            materials=materials_identity,
            cohort=cohort_identity,
            proof=proof_identity,
            preflight=preflight_identity,
        )
        authority_identity = _write_immutable_json(target / "evaluation-only-authority.json", authority_payload, "evaluation-only authority")
        _freeze_tree(target)
        return {
            "status": "materialized_evaluation_only",
            "output_dir": str(target),
            "request_sha256": request_identity,
            "evaluation_only_cohort": cohort_identity,
            "source_exclusion_proof": proof_identity,
            "rng_materials_manifest": materials_identity,
            "planner_preflight_input": preflight_input_identity,
            "planner_preflight_receipt": preflight_identity,
            "prepared_evaluator_spec": final_spec_identity,
            "prepared_evaluator_manifest": prepared_manifest_identity,
            "evaluation_only_authority": authority_identity,
            "paired_cell_count": PAIRED_CELL_COUNT,
            "authority": {
                "training_eligible": False,
                "replay_eligible": False,
                "serving_eligible": False,
                "action_authority_enabled": False,
                "selector_authority": False,
                "promotion_eligible": False,
                "submission_eligible": False,
                "kaggle_submission_authorized": False,
            },
        }
    except Exception:
        # A failed publication is never retried in place.  Freeze what was
        # written so it remains inspectable while the caller chooses a new
        # output root/run nonce for a later attempt.
        try:
            _freeze_tree(target)
        except Exception:
            pass
        raise


__all__ = [
    "CapturedSnapshot",
    "PAIRED_CELL_COUNT",
    "R198EvaluationInputError",
    "materialize_r198_evaluation_inputs",
]
