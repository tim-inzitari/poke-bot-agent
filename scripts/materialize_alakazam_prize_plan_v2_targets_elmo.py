"""Create-only, dry-run-first Elmo controller for r23 Prize-plan-v2 labels.

This controller is intentionally a narrow data-materialization boundary.  It
never starts, stops, reloads, or inspects a managed trainer/service, and it
never mounts a model, checkpoint, optimizer, or training corpus.  Its only
mutating mode is ``--execute``.  That mode writes an immutable source snapshot
and a new target-set root after a train-only Phi fit, four bounded day lanes,
and aggregate validation complete successfully.

The source snapshot contains only the public target builder, its stdlib-only
dependencies, and the current r23 contract.  Every input mount is read-only;
the sole writable mount is a new private stage below the declared output
parent.  Failed stages are deliberately retained as evidence: this controller
does not retry, resume, overwrite, delete, or clean up a run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

WINDOW_DAYS: tuple[str, ...] = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 12)]
)
SPLIT_BY_DAY: dict[str, str] = {
    **{day: "train" for day in WINDOW_DAYS[:14]},
    **{day: "validation" for day in WINDOW_DAYS[14:17]},
    **{day: "evaluation" for day in WINDOW_DAYS[17:]},
}
TRAIN_DAYS = WINDOW_DAYS[:14]
VALIDATION_DAYS = WINDOW_DAYS[14:17]
EVALUATION_DAYS = WINDOW_DAYS[17:]

CONTRACT_OWNER_REVISION = 23
CONTRACT_AUTHORITY_KEY = "revision_23_prize_plan_v2_h3_actor_canary"
OVERLAY_MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1"
BASE_COMPLETION_SCHEMA = "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"
TARGET_ROW_SCHEMA = "poke_bot.alakazam_prize_plan_target_overlay/v2"
PHI_MANIFEST_SCHEMA = "poke_bot.alakazam_prize_plan_phi_fit_manifest/v2"
PHI_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_phi_fit_receipt/v2"
TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_prize_plan_target_set_manifest/v2"
TARGET_SET_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_target_set_materialization_receipt/v2"
EXPECTED_COMPLETE_ACTION_PROGRAMS = 2_081_530
LANE_COUNT = 4
DAYS_PER_LANE = 5
CONTAINER_UID_GID = "950:950"
ELMO_IMAGE_SHA256 = (
    "sha256:0bcf2305438f8feecd9420cc37af8da4e3a2d81986e112597ad38fbe1e3f1aa3"
)
DEFAULT_CONTRACT_SHA256 = (
    "sha256:57a297db37af56fcc5ea418683b88f093720580a0b3acf6709ef5bac0eb297c2"
)
FIT_CONFIGURATION: dict[str, Any] = {
    "algorithm": "alternating_weighted_2d_isotonic_pava/v1",
    "smoothing_prior_strength": 8.0,
    "max_iterations": 10_000,
    "convergence_tolerance": 1e-10,
}
GAMMA = 1.0

DEFAULT_OVERLAY_ROOT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-recent20-rtp-overlay-v1-attempt4"
)
DEFAULT_OVERLAY_MANIFEST = (
    DEFAULT_OVERLAY_ROOT
    + "/manifests/sha256-081e40d9b9cc98714abaa8945c8d176a9143bdb8e87aeeee0327878642b118bd"
    ".overlay-manifest.json"
)
DEFAULT_OVERLAY_MANIFEST_SHA256 = (
    "sha256:081e40d9b9cc98714abaa8945c8d176a9143bdb8e87aeeee0327878642b118bd"
)
DEFAULT_BASE_PACK_COMPLETION = (
    DEFAULT_OVERLAY_ROOT
    + "/bindings/sha256-e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba"
    ".base-completion.json"
)
DEFAULT_BASE_PACK_COMPLETION_SHA256 = (
    "sha256:e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba"
)
DEFAULT_RAW_EPISODE_ROOT = "/srv/poke-bot-agent/archive/episode-days"
DEFAULT_SNAPSHOT_PARENT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-prize-plan-v2-r23-source-snapshots"
)
DEFAULT_PRIVATE_STAGE_PARENT = "/srv/poke-bot-agent/outputs/experiments"
DEFAULT_TARGET_ROOT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-prize-plan-v2-targets-r23"
)

_SHA256_PREFIX = "sha256:"
_SOURCE_RELATIVE_PATHS = (
    "goals/alakazam-elmo-rule-derivative/contract.json",
    "poke_bot/__init__.py",
    "poke_bot/prize_plan_targets_v2.py",
    "scripts/build_alakazam_prize_plan_targets_v2.py",
)


class PrizePlanMaterializationError(RuntimeError):
    """A source, contract, input, or no-replace invariant failed."""


@dataclass(frozen=True)
class ContractBinding:
    path: str
    sha256: str
    canonical_goal_revision: int
    authority_key: str
    authority_owner_goal_revision: int


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    sha256: str
    size_bytes: int
    contents_base64: str


@dataclass(frozen=True)
class SourceSnapshot:
    manifest: dict[str, Any]
    manifest_sha256: str
    files: tuple[SnapshotFile, ...]


@dataclass(frozen=True)
class OverlayDay:
    utc_day: str
    split: str
    overlay_relative_path: str
    overlay_sha256: str
    overlay_size_bytes: int
    complete_action_programs: int
    raw_episode_filename: str


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _sha_hex(value: object, *, label: str) -> str:
    raw = str(value).removeprefix(_SHA256_PREFIX)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise PrizePlanMaterializationError(f"{label} is not a full lowercase SHA-256")
    return raw


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrizePlanMaterializationError(f"{label} is not a non-negative integer")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PrizePlanMaterializationError(f"{label} is absent")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
        raise PrizePlanMaterializationError(f"{label} is not a portable relative path")
    return str(path)


def _require_regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise PrizePlanMaterializationError(f"{label} is not a regular file: {resolved}")
    return resolved


def _read_contract_bytes(path: Path) -> tuple[Path, bytes, Mapping[str, Any]]:
    contract_path = _require_regular_file(path, label="canonical goal contract")
    try:
        body = contract_path.read_bytes()
        contract = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrizePlanMaterializationError("canonical goal contract is invalid JSON") from error
    if not isinstance(contract, Mapping):
        raise PrizePlanMaterializationError("canonical goal contract is not an object")
    return contract_path, body, contract


def read_contract_binding(path: Path, *, expected_sha256: str) -> ContractBinding:
    """Bind the current wrapper and preserve the embedded r23 authority."""

    _sha_hex(expected_sha256, label="expected canonical goal contract SHA-256")
    contract_path, body, contract = _read_contract_bytes(path)
    actual_sha = sha256_bytes(body)
    if actual_sha != expected_sha256:
        raise PrizePlanMaterializationError(
            "canonical goal contract SHA-256 drifted; refresh the explicit binding"
        )
    revision = contract.get("goal_revision")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < CONTRACT_OWNER_REVISION
    ):
        raise PrizePlanMaterializationError(
            "canonical goal contract is not a current wrapper with embedded r23 authority"
        )
    authority = contract.get(CONTRACT_AUTHORITY_KEY)
    if not isinstance(authority, Mapping):
        raise PrizePlanMaterializationError("canonical goal contract lacks Prize-plan-v2 authority")
    if authority.get("owner_goal_revision") != CONTRACT_OWNER_REVISION:
        raise PrizePlanMaterializationError("Prize-plan-v2 authority owner revision drifted")
    sidecar = authority.get("sidecar_strategy")
    target = authority.get("public_prize_plan_target")
    activation = authority.get("activation_and_failure_boundary")
    if not isinstance(sidecar, Mapping) or sidecar.get("elmo_role") != (
        "read_only_public_target_label_materialization_only_no_critic_model_optimizer_or_training"
    ):
        raise PrizePlanMaterializationError("contract does not preserve Elmo label-only isolation")
    if not isinstance(target, Mapping) or target.get("row_schema") != TARGET_ROW_SCHEMA:
        raise PrizePlanMaterializationError("contract Prize-plan target schema drifted")
    if target.get("manifest_schema") != TARGET_SET_MANIFEST_SCHEMA:
        raise PrizePlanMaterializationError("contract Prize-plan target-set manifest schema drifted")
    if not isinstance(activation, Mapping) or activation.get("current_activation_allowed") is not False:
        raise PrizePlanMaterializationError("contract unexpectedly grants activation authority")
    if activation.get("implementation_and_staging_authorized_now") is not True:
        raise PrizePlanMaterializationError("contract does not authorize target staging")
    return ContractBinding(
        path=str(contract_path),
        sha256=actual_sha,
        canonical_goal_revision=revision,
        authority_key=CONTRACT_AUTHORITY_KEY,
        authority_owner_goal_revision=CONTRACT_OWNER_REVISION,
    )


def build_source_snapshot(
    contract_path: Path,
    *,
    expected_contract_sha256: str,
    project_root: Path = ROOT,
) -> SourceSnapshot:
    """Create a minimal immutable source payload for the container only."""

    contract = read_contract_binding(contract_path, expected_sha256=expected_contract_sha256)
    root = project_root.resolve()
    files: list[SnapshotFile] = []
    for relative in _SOURCE_RELATIVE_PATHS:
        candidate = _require_regular_file(root / relative, label=f"source snapshot member {relative}")
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise PrizePlanMaterializationError("source snapshot member escaped project root") from error
        body = candidate.read_bytes()
        files.append(
            SnapshotFile(
                path=relative,
                sha256=sha256_bytes(body),
                size_bytes=len(body),
                contents_base64=base64.b64encode(body).decode("ascii"),
            )
        )
    if tuple(item.path for item in files) != _SOURCE_RELATIVE_PATHS:
        raise PrizePlanMaterializationError("source snapshot inventory drifted")
    if files[0].sha256 != contract.sha256:
        raise PrizePlanMaterializationError("source snapshot contract bytes drifted during preparation")
    portable_contract = {**asdict(contract), "path": _SOURCE_RELATIVE_PATHS[0]}
    manifest = {
        "schema": "poke_bot.alakazam_prize_plan_v2_elmo_source_snapshot/v1",
        "owner_goal_revision": CONTRACT_OWNER_REVISION,
        "required_authority": CONTRACT_AUTHORITY_KEY,
        "canonical_goal_contract": portable_contract,
        "members": [
            {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in files
        ],
        "minimal_member_count": len(_SOURCE_RELATIVE_PATHS),
        "public_target_materialization_only": True,
        "model_optimizer_or_training_members_present": False,
        "create_only": True,
    }
    return SourceSnapshot(
        manifest=manifest,
        manifest_sha256=sha256_bytes(canonical_bytes(manifest)),
        files=tuple(files),
    )


def parse_overlay_days(
    manifest: Mapping[str, Any],
    *,
    expected_row_count: int = EXPECTED_COMPLETE_ACTION_PROGRAMS,
) -> tuple[OverlayDay, ...]:
    """Validate the fixed 14/3/3 recent-20 complete-action inventory."""

    if manifest.get("schema") != OVERLAY_MANIFEST_SCHEMA:
        raise PrizePlanMaterializationError("foreign complete-action overlay manifest schema")
    raw_shards = manifest.get("overlay_shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != len(WINDOW_DAYS):
        raise PrizePlanMaterializationError("overlay manifest does not contain exactly twenty day shards")
    observed: dict[str, OverlayDay] = {}
    for entry in raw_shards:
        if not isinstance(entry, Mapping):
            raise PrizePlanMaterializationError("overlay manifest contains malformed day data")
        day = entry.get("utc_day")
        if not isinstance(day, str) or day not in SPLIT_BY_DAY or day in observed:
            raise PrizePlanMaterializationError("overlay manifest day inventory drifted")
        if entry.get("split") != SPLIT_BY_DAY[day]:
            raise PrizePlanMaterializationError(f"overlay split drifted for {day}")
        overlay_sha = str(entry.get("sha256") or "")
        _sha_hex(overlay_sha, label=f"overlay SHA-256 for {day}")
        observed[day] = OverlayDay(
            utc_day=day,
            split=SPLIT_BY_DAY[day],
            overlay_relative_path=_safe_relative_path(entry.get("path"), label=f"overlay path for {day}"),
            overlay_sha256=overlay_sha,
            overlay_size_bytes=_nonnegative_int(entry.get("size_bytes"), label=f"overlay size for {day}"),
            complete_action_programs=_nonnegative_int(
                entry.get("complete_action_programs"), label=f"program count for {day}"
            ),
            raw_episode_filename=f"pokemon-tcg-ai-battle-episodes-{day}.zip",
        )
    if tuple(sorted(observed)) != WINDOW_DAYS:
        raise PrizePlanMaterializationError("overlay manifest is not the exact contiguous recent-20 window")
    days = tuple(observed[day] for day in WINDOW_DAYS)
    if sum(day.complete_action_programs for day in days) != expected_row_count:
        raise PrizePlanMaterializationError("complete-action program total drifted")
    return days


def allocate_lanes(days: Sequence[OverlayDay]) -> tuple[tuple[OverlayDay, ...], ...]:
    """Allocate exactly five unique days to each of exactly four lanes."""

    if len(days) != len(WINDOW_DAYS) or {item.utc_day for item in days} != set(WINDOW_DAYS):
        raise PrizePlanMaterializationError("lane allocation requires the exact twenty unique days")
    ordered = sorted(
        days,
        key=lambda item: (-max(item.overlay_size_bytes, item.complete_action_programs), item.utc_day),
    )
    lanes: list[list[OverlayDay]] = [[] for _ in range(LANE_COUNT)]
    loads = [0] * LANE_COUNT
    for item in ordered:
        allowed = [index for index in range(LANE_COUNT) if len(lanes[index]) < DAYS_PER_LANE]
        index = min(allowed, key=lambda candidate: (loads[candidate], candidate))
        lanes[index].append(item)
        loads[index] += item.overlay_size_bytes + item.complete_action_programs
    result = tuple(tuple(sorted(lane, key=lambda item: item.utc_day)) for lane in lanes)
    if len(result) != LANE_COUNT or any(len(lane) != DAYS_PER_LANE for lane in result):
        raise PrizePlanMaterializationError("controller did not create four five-day lanes")
    if sorted(item.utc_day for lane in result for item in lane) != list(WINDOW_DAYS):
        raise PrizePlanMaterializationError("lane allocation has a duplicate or missing day")
    return result


def _validate_remote_absolute(path: str, *, label: str) -> str:
    value = PurePosixPath(path)
    if not value.is_absolute() or ".." in value.parts or str(value) in {"/", "."}:
        raise PrizePlanMaterializationError(f"{label} must be a bounded absolute Elmo path")
    artifact_root = PurePosixPath("/srv/poke-bot-agent")
    try:
        value.relative_to(artifact_root)
    except ValueError as error:
        raise PrizePlanMaterializationError(f"{label} escaped the Elmo artifact root") from error
    return str(value)


def build_remote_config(
    *,
    execute: bool,
    snapshot: SourceSnapshot,
    overlay_root: str,
    overlay_manifest: str,
    expected_overlay_manifest_sha256: str,
    base_pack_completion: str,
    expected_base_pack_completion_sha256: str,
    raw_episode_root: str,
    snapshot_parent: str,
    private_stage_parent: str,
    target_root: str,
    image_sha256: str,
    stage_nonce: str,
    expected_row_count: int = EXPECTED_COMPLETE_ACTION_PROGRAMS,
) -> dict[str, Any]:
    """Build the sealed remote plan.  The remote code accepts no shell input."""

    for label, value in (
        ("overlay root", overlay_root),
        ("overlay manifest", overlay_manifest),
        ("base-pack completion", base_pack_completion),
        ("raw episode root", raw_episode_root),
        ("source snapshot parent", snapshot_parent),
        ("private-stage parent", private_stage_parent),
        ("target root", target_root),
    ):
        _validate_remote_absolute(value, label=label)
    _sha_hex(expected_overlay_manifest_sha256, label="expected overlay manifest SHA-256")
    _sha_hex(expected_base_pack_completion_sha256, label="expected base-pack completion SHA-256")
    _sha_hex(image_sha256, label="Elmo Docker image SHA-256")
    if not stage_nonce or any(character not in "0123456789abcdef-" for character in stage_nonce):
        raise PrizePlanMaterializationError("private-stage nonce is malformed")
    snapshot_hex = _sha_hex(snapshot.manifest_sha256, label="source snapshot manifest SHA-256")
    snapshot_root = f"{snapshot_parent}/sha256-{snapshot_hex}"
    stage_root = f"{private_stage_parent}/.alakazam-prize-plan-v2-r23-private-{stage_nonce}"
    return {
        "schema": "poke_bot.alakazam_prize_plan_v2_elmo_materialization_controller/v1",
        "execute": bool(execute),
        "owner_goal_revision": CONTRACT_OWNER_REVISION,
        "required_authority": CONTRACT_AUTHORITY_KEY,
        "expected_complete_action_programs": expected_row_count,
        "lane_count_exact": LANE_COUNT,
        "days_per_lane_exact": DAYS_PER_LANE,
        "fit": {"gamma": GAMMA, "configuration": FIT_CONFIGURATION},
        "container": {
            "image_sha256": image_sha256,
            "user": CONTAINER_UID_GID,
            "cpus": 1,
            "memory": "4g",
            "network": "none",
            "read_only_root": True,
            "low_priority_nice": 15,
            "low_priority_ionice_class": 3,
        },
        "contract": snapshot.manifest["canonical_goal_contract"],
        "source_snapshot": {
            "root": snapshot_root,
            "manifest": snapshot.manifest,
            "manifest_sha256": snapshot.manifest_sha256,
            "files": [asdict(item) for item in snapshot.files],
        },
        "inputs": {
            "overlay_root": overlay_root,
            "overlay_manifest": overlay_manifest,
            "expected_overlay_manifest_sha256": expected_overlay_manifest_sha256,
            "base_pack_completion": base_pack_completion,
            "expected_base_pack_completion_sha256": expected_base_pack_completion_sha256,
            "raw_episode_root": raw_episode_root,
        },
        "publication": {
            "private_stage_root": stage_root,
            "stage_nonce": stage_nonce,
            "target_root": target_root,
            "create_only": True,
            "atomic_no_clobber_publish": True,
            "cleanup_retry_or_overwrite_allowed": False,
        },
    }


# The remote program is stdlib-only and carried through SSH stdin.  It uses no
# interpolated remote paths or untrusted shell fragments.  The r23 target CLI
# is invoked only inside the immutable, no-network, read-only-root container.
_REMOTE_PROGRAM_TEMPLATE = r'''
import base64
import ctypes
import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

CONFIG = json.loads(base64.b64decode(__CONFIG_BASE64__).decode("utf-8"))
SHA_PREFIX = "sha256:"
WINDOW_DAYS = tuple([f"2026-07-{number:02d}" for number in range(23, 32)] + [f"2026-08-{number:02d}" for number in range(1, 12)])
SPLIT_BY_DAY = {**{day: "train" for day in WINDOW_DAYS[:14]}, **{day: "validation" for day in WINDOW_DAYS[14:17]}, **{day: "evaluation" for day in WINDOW_DAYS[17:]}}
TRAIN_DAYS = WINDOW_DAYS[:14]
VALIDATION_DAYS = WINDOW_DAYS[14:17]
EVALUATION_DAYS = WINDOW_DAYS[17:]
OVERLAY_MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1"
BASE_COMPLETION_SCHEMA = "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"
PHI_MANIFEST_SCHEMA = "poke_bot.alakazam_prize_plan_phi_fit_manifest/v2"
PHI_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_phi_fit_receipt/v2"
TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_prize_plan_target_set_manifest/v2"
TARGET_SET_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_target_set_materialization_receipt/v2"
TARGET_DAY_MANIFEST_SCHEMA = "poke_bot.alakazam_prize_plan_target_day_manifest/v2"
TARGET_DAY_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_target_day_materialization_receipt/v2"

def fail(message):
    raise RuntimeError(message)

def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def sha_bytes(value):
    return SHA_PREFIX + hashlib.sha256(value).hexdigest()

def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return SHA_PREFIX + digest.hexdigest()

def sha_hex(value, label):
    raw = str(value).removeprefix(SHA_PREFIX)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        fail(f"{label} is not a full lowercase SHA-256")
    return raw

def bounded_absolute(value, label):
    path = PurePosixPath(value)
    root = PurePosixPath("/srv/poke-bot-agent")
    if not path.is_absolute() or ".." in path.parts or str(path) in {"/", "."}:
        fail(f"{label} is not a bounded absolute path")
    try:
        path.relative_to(root)
    except ValueError:
        fail(f"{label} escaped the Elmo artifact root")
    return Path(str(path))

def portable_relative(value, label):
    if not isinstance(value, str) or not value:
        fail(f"{label} is absent")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
        fail(f"{label} is not portable")
    return str(path)

def no_symlink_path(path, label):
    path = Path(path)
    if not path.is_absolute():
        fail(f"{label} is not absolute")
    current = Path("/")
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            fail(f"{label} is absent: {current}")
        if stat.S_ISLNK(mode):
            fail(f"{label} traverses a symlink: {current}")
    return path

def regular_file(path, label):
    path = no_symlink_path(path, label)
    if not stat.S_ISREG(os.stat(path).st_mode):
        fail(f"{label} is not a regular file: {path}")
    return path

def existing_directory(path, label):
    path = no_symlink_path(path, label)
    if not stat.S_ISDIR(os.stat(path).st_mode):
        fail(f"{label} is not a directory: {path}")
    return path

def child_under(root, relative, label):
    root = existing_directory(root, f"{label} root")
    relative = portable_relative(relative, label)
    child = root / relative
    no_symlink_path(child, label)
    try:
        child.relative_to(root)
    except ValueError:
        fail(f"{label} escaped its root")
    return child

def required_new(path, label):
    path = Path(path)
    if path.exists() or path.is_symlink():
        fail(f"{label} already exists: {path}")
    existing_directory(path.parent, f"{label} parent")
    return path

def planned_new(path, label, allow_create_parent=False):
    path = Path(path)
    if path.exists() or path.is_symlink():
        fail(f"{label} already exists: {path}")
    parent = path.parent
    if parent.exists() or parent.is_symlink():
        existing_directory(parent, f"{label} parent")
    elif allow_create_parent:
        existing_directory(parent.parent, f"{label} trusted parent")
    else:
        fail(f"{label} parent is absent: {parent}")
    return path

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def ensure_create_only_parent(path, label):
    path = Path(path)
    if path.exists() or path.is_symlink():
        return existing_directory(path, label)
    existing_directory(path.parent, f"{label} trusted parent")
    try:
        os.mkdir(path, 0o755)
    except FileExistsError:
        fail(f"refusing racing or pre-existing {label}: {path}")
    fsync_directory(path.parent)
    return existing_directory(path, label)

def write_exclusive(path, body, mode=0o444):
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(fd, body[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)

def fsync_tree(root):
    for current, directories, _files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            fail("publication tree contains a symlink")
        fsync_directory(current_path)

def rename_no_replace(source, destination):
    source = Path(source).resolve()
    destination_input = Path(destination)
    destination_parent = destination_input.parent.resolve()
    destination = destination_parent / destination_input.name
    publication_root = Path("/srv/poke-bot-agent/outputs/experiments")
    try:
        source.relative_to(publication_root)
        destination.relative_to(publication_root)
    except ValueError:
        fail("atomic publication escaped the bounded experiment root")
    if source.is_symlink() or not source.is_dir():
        fail("atomic publication source is not a regular directory")
    existing_directory(source.parent, "atomic publication source parent")
    existing_directory(destination_parent, "atomic publication destination parent")
    if destination_input.exists() or destination_input.is_symlink():
        fail(f"refusing to replace existing publication target: {destination}")
    if os.stat(source).st_dev != os.stat(destination_parent).st_dev:
        fail("atomic publication source and destination are on different filesystems")
    fsync_tree(source)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        fail("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        problem = ctypes.get_errno()
        if problem == errno.EEXIST:
            fail(f"refusing to replace existing publication target: {destination}")
        fail(f"renameat2 publication failed for {destination}: errno={problem}")
    fsync_directory(source.parent)
    fsync_directory(destination.parent)

def positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} is not a non-negative integer")
    return value

def one_json(directory, suffix, label):
    directory = existing_directory(directory, label)
    members = sorted(path for path in directory.iterdir() if path.is_file() and not path.is_symlink() and path.name.endswith(suffix))
    if len(members) != 1:
        fail(f"{label} must contain exactly one {suffix} file")
    return regular_file(members[0], label)

def read_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"{label} is invalid JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    return value

def read_json_list(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"{label} is invalid JSON: {error}")
    if not isinstance(value, list):
        fail(f"{label} is not an array")
    return value

def contract_preflight(contract):
    if not isinstance(contract, dict):
        fail("snapshot contract binding is malformed")
    revision = contract.get("canonical_goal_revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 23:
        fail("snapshot contract is not a current wrapper with embedded r23 authority")
    if contract.get("authority_key") != "revision_23_prize_plan_v2_h3_actor_canary":
        fail("snapshot contract required authority drifted")
    if contract.get("authority_owner_goal_revision") != 23:
        fail("snapshot contract semantic owner revision drifted")
    sha_hex(contract.get("sha256"), "snapshot contract SHA-256")

def overlay_preflight(inputs, expected_rows):
    overlay_root = existing_directory(bounded_absolute(inputs["overlay_root"], "overlay root"), "overlay root")
    manifest_path = regular_file(bounded_absolute(inputs["overlay_manifest"], "overlay manifest"), "overlay manifest")
    try:
        manifest_path.relative_to(overlay_root)
    except ValueError:
        fail("overlay manifest escaped overlay root")
    expected_manifest_sha = inputs["expected_overlay_manifest_sha256"]
    sha_hex(expected_manifest_sha, "expected overlay manifest SHA-256")
    if sha_file(manifest_path) != expected_manifest_sha:
        fail("overlay manifest SHA-256 mismatch")
    manifest = read_json(manifest_path, "overlay manifest")
    if manifest.get("schema") != OVERLAY_MANIFEST_SCHEMA:
        fail("foreign overlay manifest schema")
    raw_shards = manifest.get("overlay_shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 20:
        fail("overlay manifest does not contain exactly twenty shards")
    raw_root = existing_directory(bounded_absolute(inputs["raw_episode_root"], "raw episode root"), "raw episode root")
    seen = set()
    jobs = []
    total = 0
    for entry in raw_shards:
        if not isinstance(entry, dict):
            fail("overlay manifest contains malformed day data")
        day = entry.get("utc_day")
        if day not in SPLIT_BY_DAY or day in seen:
            fail("overlay manifest day inventory drifted")
        split = entry.get("split")
        if split != SPLIT_BY_DAY[day]:
            fail(f"overlay split drifted for {day}")
        relative = portable_relative(entry.get("path"), f"overlay path for {day}")
        overlay = child_under(overlay_root, relative, f"overlay object for {day}")
        regular_file(overlay, f"overlay object for {day}")
        overlay_sha = entry.get("sha256")
        sha_hex(overlay_sha, f"overlay SHA-256 for {day}")
        overlay_size = positive_int(entry.get("size_bytes"), f"overlay size for {day}")
        if overlay.stat().st_size != overlay_size or sha_file(overlay) != overlay_sha:
            fail(f"overlay object SHA-256 or size mismatch for {day}")
        raw_name = f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        raw = child_under(raw_root, raw_name, f"raw ZIP for {day}")
        regular_file(raw, f"raw ZIP for {day}")
        raw_sha = sha_file(raw)
        jobs.append({
            "utc_day": day,
            "split": split,
            "overlay_relative_path": relative,
            "overlay_sha256": overlay_sha,
            "overlay_size_bytes": overlay_size,
            "complete_action_programs": positive_int(entry.get("complete_action_programs"), f"program count for {day}"),
            "raw_episode_filename": raw_name,
            "raw_episode_sha256": raw_sha,
            "raw_episode_size_bytes": raw.stat().st_size,
        })
        total += jobs[-1]["complete_action_programs"]
        seen.add(day)
    if tuple(sorted(seen)) != WINDOW_DAYS:
        fail("overlay manifest is not the exact recent-20 day window")
    if total != expected_rows:
        fail("complete-action program total drifted")
    base = regular_file(bounded_absolute(inputs["base_pack_completion"], "base-pack completion"), "base-pack completion")
    expected_base_sha = inputs["expected_base_pack_completion_sha256"]
    sha_hex(expected_base_sha, "expected base-pack completion SHA-256")
    if sha_file(base) != expected_base_sha:
        fail("base-pack completion SHA-256 mismatch")
    if read_json(base, "base-pack completion").get("schema") != BASE_COMPLETION_SCHEMA:
        fail("foreign base-pack completion schema")
    return {
        "overlay_manifest_path": str(manifest_path),
        "overlay_manifest_sha256": expected_manifest_sha,
        "base_pack_completion_path": str(base),
        "base_pack_completion_sha256": expected_base_sha,
        "jobs": sorted(jobs, key=lambda item: item["utc_day"]),
        "raw_episode_sha256s": [{"utc_day": item["utc_day"], "sha256": item["raw_episode_sha256"], "size_bytes": item["raw_episode_size_bytes"]} for item in sorted(jobs, key=lambda item: item["utc_day"])],
    }

def verify_image(container):
    expected = container["image_sha256"]
    sha_hex(expected, "Docker image SHA-256")
    try:
        inspected = subprocess.run(["sudo", "-n", "docker", "image", "inspect", "--format", "{{.Id}}", expected], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as error:
        fail(f"Docker image inspection could not start: {error}")
    if inspected.returncode != 0 or inspected.stdout.strip() != expected:
        fail("required immutable Docker image is unavailable or mismatched")

def allocate_lanes(jobs):
    if len(jobs) != 20:
        fail("exactly twenty jobs are required")
    ordered = sorted(jobs, key=lambda item: (-max(item["overlay_size_bytes"], item["complete_action_programs"]), item["utc_day"]))
    lanes = [[] for _ in range(4)]
    loads = [0, 0, 0, 0]
    for job in ordered:
        eligible = [index for index in range(4) if len(lanes[index]) < 5]
        index = min(eligible, key=lambda candidate: (loads[candidate], candidate))
        lanes[index].append(job)
        loads[index] += job["overlay_size_bytes"] + job["complete_action_programs"]
    lanes = [sorted(lane, key=lambda item: item["utc_day"]) for lane in lanes]
    if len(lanes) != 4 or any(len(lane) != 5 for lane in lanes):
        fail("four five-day lanes were not produced")
    if sorted(job["utc_day"] for lane in lanes for job in lane) != list(WINDOW_DAYS):
        fail("lane inventory has a duplicate or missing day")
    return lanes

def snapshot_source(snapshot):
    root = Path(snapshot["root"])
    ensure_create_only_parent(root.parent, "source snapshot parent")
    required_new(root, "source snapshot root")
    private = root.parent / ("." + root.name + ".private-" + CONFIG["publication"]["stage_nonce"])
    required_new(private, "source snapshot private root")
    os.mkdir(private, 0o755)
    manifest = snapshot["manifest"]
    if sha_bytes(canonical_bytes(manifest)) != snapshot["manifest_sha256"]:
        fail("source snapshot manifest SHA-256 drifted")
    members = manifest.get("members")
    files = snapshot.get("files")
    expected_names = ["goals/alakazam-elmo-rule-derivative/contract.json", "poke_bot/__init__.py", "poke_bot/prize_plan_targets_v2.py", "scripts/build_alakazam_prize_plan_targets_v2.py"]
    if not isinstance(members, list) or not isinstance(files, list) or len(members) != 4 or len(files) != 4:
        fail("source snapshot must contain exactly four members")
    if [item.get("path") for item in members] != expected_names or [item.get("path") for item in files] != expected_names:
        fail("source snapshot member inventory drifted")
    for expected, supplied in zip(members, files, strict=True):
        relative = portable_relative(supplied.get("path"), "source snapshot member")
        destination = private / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            body = base64.b64decode(supplied.get("contents_base64", ""), validate=True)
        except Exception as error:
            fail(f"source snapshot payload is not valid base64: {error}")
        if len(body) != expected.get("size_bytes") or sha_bytes(body) != expected.get("sha256"):
            fail("source snapshot member bytes drifted")
        write_exclusive(destination, body)
    write_exclusive(private / "SOURCE_MANIFEST.json", canonical_bytes(manifest))
    for expected in members:
        member = regular_file(private / expected["path"], f"source snapshot member {expected['path']}")
        if member.stat().st_size != expected["size_bytes"] or sha_file(member) != expected["sha256"]:
            fail("source snapshot post-write verification failed")
    for path in sorted(private.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(private, 0o555)
    fsync_directory(private)
    rename_no_replace(private, root)
    return root

def create_stage(root, lanes, preflight):
    required_new(root, "private Prize-plan materialization stage")
    os.mkdir(root, 0o755)
    directories = {name: root / name for name in ("fit", "days", "lanes", "logs", "resource_peaks", "finalization")}
    for directory in directories.values():
        os.mkdir(directory, 0o755)
    day_inputs = []
    for job in preflight["jobs"]:
        day_inputs.append({
            "utc_day": job["utc_day"], "split": job["split"],
            "complete_action_overlay_path": "/overlay/" + job["overlay_relative_path"],
            "complete_action_overlay_sha256": job["overlay_sha256"],
            "raw_episode_zip_path": "/raw/" + job["raw_episode_filename"],
            "raw_episode_zip_sha256": job["raw_episode_sha256"],
        })
    write_exclusive(directories["lanes"] / "fit-day-inputs.json", canonical_bytes(day_inputs))
    for index, lane in enumerate(lanes):
        write_exclusive(directories["lanes"] / f"lane-{index:02d}.json", canonical_bytes(lane))
    fsync_directory(directories["lanes"])
    fsync_directory(root)
    return {"root": root, **directories}

def mount(source, target, readonly=False):
    result = f"type=bind,src={source},dst={target}"
    if readonly:
        result += ",readonly"
    return result

def docker_base(container, name):
    return ["sudo", "-n", "docker", "run", "--name", name, "--pull", "never", "--network", "none", "--read-only", "--user", container["user"], "--cpus", str(container["cpus"]), "--memory", container["memory"], "--memory-swap", container["memory"], "--pids-limit", "256", "--cpu-shares", "128", "--blkio-weight", "100", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=/source", "--entrypoint", "/usr/bin/ionice"]

def low_priority_python(container, script, *arguments):
    return [container["image_sha256"], "-c", str(container["low_priority_ionice_class"]), "/usr/bin/nice", "-n", str(container["low_priority_nice"]), "python3", script, *arguments]

def parse_memory_bytes(value):
    text = str(value).strip()
    if not text:
        fail("Docker stats memory value is absent")
    units = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4, "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}
    position = 0
    while position < len(text) and (text[position].isdigit() or text[position] == "."):
        position += 1
    if position == 0:
        fail("Docker stats memory value is malformed")
    number = float(text[:position])
    unit = text[position:].strip().lower() or "b"
    if unit not in units or number < 0.0:
        fail("Docker stats memory unit is malformed")
    return int(number * units[unit])

def docker_stat_sample(name):
    try:
        result = subprocess.run(["sudo", "-n", "docker", "stats", "--no-stream", "--format", "{{json .}}", name], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        fail(f"Docker stats could not sample {name}: {error}")
    if result.returncode != 0:
        return None
    try:
        row = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        fail(f"Docker stats returned malformed JSON for {name}")
    if not isinstance(row, dict):
        fail("Docker stats row is not an object")
    cpu_text = str(row.get("CPUPerc", "")).strip()
    if not cpu_text.endswith("%"):
        fail("Docker stats CPU percentage is malformed")
    try:
        cpu_percent = float(cpu_text[:-1])
        pids = int(str(row.get("PIDs", "")).strip())
    except ValueError:
        fail("Docker stats CPU/PIDs value is malformed")
    mem_usage = str(row.get("MemUsage", "")).split("/", 1)[0].strip()
    if cpu_percent < 0.0 or pids < 0:
        fail("Docker stats contains a negative resource value")
    return {"sampled_at_unix_seconds": time.time(), "cpu_utilization_percent": cpu_percent, "process_count": pids, "memory_bytes": parse_memory_bytes(mem_usage)}

def write_resource_receipt(stage, *, isolated_job, container_name, command, samples, returncode, started_at, completed_at):
    if not samples:
        fail(f"{isolated_job} produced no Docker resource sample")
    receipt = {
        "schema": "poke_bot.alakazam_prize_plan_v2_elmo_resource_peak_receipt/v1",
        "owner_goal_revision": 23,
        "required_authority": "revision_23_prize_plan_v2_h3_actor_canary",
        "isolated_job": isolated_job,
        "container_name": container_name,
        "container_image_sha256": CONFIG["container"]["image_sha256"],
        "resource_limits": {"cpus": CONFIG["container"]["cpus"], "memory": CONFIG["container"]["memory"], "network": "none", "read_only_root": True},
        "measurement_method": "docker_stats_no_stream_poll_v1",
        "sample_interval_seconds": 1.0,
        "sample_count": len(samples),
        "peak_memory_bytes": max(sample["memory_bytes"] for sample in samples),
        "peak_cpu_utilization_percent": max(sample["cpu_utilization_percent"] for sample in samples),
        "peak_process_count": max(sample["process_count"] for sample in samples),
        "experiment_ram_measurement_method": "docker_stats_no_stream_poll_v1",
        "experiment_ram_peak_bytes": max(sample["memory_bytes"] for sample in samples),
        "cpu_process_or_utilization_peak": {
            "measurement_method": "docker_stats_no_stream_poll_v1",
            "peak_cpu_utilization_percent": max(sample["cpu_utilization_percent"] for sample in samples),
            "peak_process_count": max(sample["process_count"] for sample in samples),
        },
        "gpu_memory_peak_bytes_per_device_or_not_applicable": "not_applicable_no_gpu_device_mounted",
        "gpu_utilization_peak_per_device_or_not_applicable": "not_applicable_no_gpu_device_mounted",
        "gpu": {"applicable": False, "reason": "no_gpu_device_is_mounted_or_authorized_for_public_label_materialization"},
        "command_sha256": sha_bytes(canonical_bytes(command)),
        "container_exit_code": returncode,
        "started_at_unix_seconds": started_at,
        "completed_at_unix_seconds": completed_at,
        "model_optimizer_or_training_performed": False,
    }
    destination = stage["resource_peaks"] / f"{isolated_job}.resource-peak.json"
    write_exclusive(destination, canonical_bytes(receipt))
    return destination

def run_monitored_container(stage, *, isolated_job, container_name, command, log_path, label):
    started_at = time.time()
    with log_path.open("xb") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            fail(f"could not start {label}: {error}")
        samples = []
        while process.poll() is None:
            sample = docker_stat_sample(container_name)
            if sample is not None:
                samples.append(sample)
            if process.poll() is None:
                time.sleep(1.0)
        returncode = process.wait()
    completed_at = time.time()
    write_resource_receipt(stage, isolated_job=isolated_job, container_name=container_name, command=command, samples=samples, returncode=returncode, started_at=started_at, completed_at=completed_at)
    if returncode != 0:
        fail(f"{label} failed without retry")

def phi_manifest_and_receipt(root):
    manifest_path = one_json(root / "manifests", ".phi-fit-manifest.json", "Phi manifests")
    receipt_path = one_json(root / "receipts", ".phi-fit-receipt.json", "Phi receipts")
    manifest = read_json(manifest_path, "Phi fit manifest")
    receipt = read_json(receipt_path, "Phi fit receipt")
    if manifest.get("schema") != PHI_MANIFEST_SCHEMA or receipt.get("schema") != PHI_RECEIPT_SCHEMA:
        fail("foreign Phi fit artifact schema")
    if manifest.get("owner_goal_revision") != 23 or receipt.get("owner_goal_revision") != 23:
        fail("Phi fit owner revision drifted")
    if manifest.get("goal_contract", {}).get("sha256") != CONFIG["contract"]["sha256"]:
        fail("Phi fit contract binding drifted")
    if manifest.get("fit_scope") != "sealed_train_split_only":
        fail("Phi fit scope drifted")
    isolation = manifest.get("split_isolation", {})
    if isolation.get("fit_days") != list(TRAIN_DAYS) or isolation.get("validation_days_not_opened_for_fit") != list(VALIDATION_DAYS) or isolation.get("evaluation_days_not_opened_for_fit") != list(EVALUATION_DAYS):
        fail("Phi fit split isolation drifted")
    if manifest.get("fit_configuration") != CONFIG["fit"]["configuration"]:
        fail("Phi fit configuration drifted")
    if receipt.get("fit_configuration_sha256") != manifest.get("fit_configuration_sha256"):
        fail("Phi fit receipt configuration binding drifted")
    if receipt.get("phi_fit_manifest_sha256") != sha_file(manifest_path):
        fail("Phi fit receipt does not bind manifest")
    return manifest_path, sha_file(manifest_path), receipt_path, sha_file(receipt_path)

def run_phi_fit(stage, source_root, inputs, container):
    output = stage["fit"] / "phi-fit"
    required_new(output, "Phi fit output")
    name = "alakazam-prize-plan-v2-r23-" + CONFIG["publication"]["stage_nonce"] + "-fit"
    command = docker_base(container, name) + [
        "--mount", mount(source_root, "/source", readonly=True),
        "--mount", mount(inputs["overlay_root"], "/overlay", readonly=True),
        "--mount", mount(inputs["raw_episode_root"], "/raw", readonly=True),
        "--mount", mount(stage["lanes"], "/lanes", readonly=True),
        "--mount", mount(stage["fit"], "/outputs", readonly=False),
    ] + low_priority_python(container, "/source/scripts/build_alakazam_prize_plan_targets_v2.py", "fit-phi", "--output-root", "/outputs/phi-fit", "--goal-contract", "/source/goals/alakazam-elmo-rule-derivative/contract.json", "--expected-goal-contract-sha256", CONFIG["contract"]["sha256"], "--fit-configuration", json.dumps(CONFIG["fit"]["configuration"], sort_keys=True, separators=(",", ":")))
    day_inputs = read_json_list(stage["lanes"] / "fit-day-inputs.json", "Phi day inputs")
    if len(day_inputs) != 20 or any(not isinstance(item, dict) for item in day_inputs):
        fail("Phi day input inventory is malformed")
    for item in day_inputs:
        command += ["--day-input", json.dumps(item, sort_keys=True, separators=(",", ":"))]
    run_monitored_container(stage, isolated_job="fit-phi", container_name=name, command=command, log_path=stage["logs"] / "fit-phi.log", label="train-only Phi fit")
    return phi_manifest_and_receipt(output)

LANE_RUNNER = r"""\
import argparse
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", required=True)
parser.add_argument("--source-root", required=True)
parser.add_argument("--overlay-root", required=True)
parser.add_argument("--raw-root", required=True)
parser.add_argument("--phi-fit-manifest", required=True)
parser.add_argument("--expected-phi-fit-manifest-sha256", required=True)
parser.add_argument("--output-parent", required=True)
parser.add_argument("--goal-contract-sha256", required=True)
parser.add_argument("--gamma", required=True)
args = parser.parse_args()
with open(args.jobs, "r", encoding="utf-8") as handle:
    jobs = json.load(handle)
if not isinstance(jobs, list) or len(jobs) != 5:
    raise SystemExit("lane job manifest must contain exactly five days")
for job in jobs:
    command = [
        sys.executable, os.path.join(args.source_root, "scripts/build_alakazam_prize_plan_targets_v2.py"),
        "build-day",
        "--complete-action-overlay", os.path.join(args.overlay_root, job["overlay_relative_path"]),
        "--raw-episode-zip", os.path.join(args.raw_root, job["raw_episode_filename"]),
        "--output-root", os.path.join(args.output_parent, job["utc_day"]),
        "--utc-day", job["utc_day"], "--split", job["split"],
        "--goal-contract", os.path.join(args.source_root, "goals/alakazam-elmo-rule-derivative/contract.json"),
        "--expected-goal-contract-sha256", args.goal_contract_sha256,
        "--phi-fit-manifest", args.phi_fit_manifest,
        "--expected-phi-fit-manifest-sha256", args.expected_phi_fit_manifest_sha256,
        "--gamma", args.gamma,
        "--expected-complete-action-overlay-sha256", job["overlay_sha256"],
        "--expected-raw-episode-zip-sha256", job["raw_episode_sha256"],
    ]
    if subprocess.run(command, check=False).returncode != 0:
        raise SystemExit(1)
"""

def run_day_lanes(stage, source_root, inputs, container, lanes, phi_manifest_sha):
    runner = stage["lanes"] / "lane_runner.py"
    write_exclusive(runner, LANE_RUNNER.encode("utf-8"))
    processes = []
    launch_error = None
    for index, _lane in enumerate(lanes):
        name = "alakazam-prize-plan-v2-r23-" + CONFIG["publication"]["stage_nonce"] + f"-lane-{index:02d}"
        command = docker_base(container, name) + [
            "--mount", mount(source_root, "/source", readonly=True),
            "--mount", mount(inputs["overlay_root"], "/overlay", readonly=True),
            "--mount", mount(inputs["raw_episode_root"], "/raw", readonly=True),
            "--mount", mount(stage["lanes"], "/lanes", readonly=True),
            "--mount", mount(stage["fit"] / "phi-fit", "/phi", readonly=True),
            "--mount", mount(stage["days"], "/outputs", readonly=False),
        ] + low_priority_python(container, "/lanes/lane_runner.py", "--jobs", f"/lanes/lane-{index:02d}.json", "--source-root", "/source", "--overlay-root", "/overlay", "--raw-root", "/raw", "--phi-fit-manifest", "/phi/manifests/" + one_json(stage["fit"] / "phi-fit" / "manifests", ".phi-fit-manifest.json", "Phi manifests").name, "--expected-phi-fit-manifest-sha256", phi_manifest_sha, "--output-parent", "/outputs", "--goal-contract-sha256", CONFIG["contract"]["sha256"], "--gamma", str(CONFIG["fit"]["gamma"]))
        log = (stage["logs"] / f"lane-{index:02d}.log").open("xb")
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            log.close()
            launch_error = f"could not start isolated day lane {index}: {error}"
            break
        processes.append({"index": index, "process": process, "log": log, "command": command, "name": name, "started_at": time.time(), "samples": []})
    if len(processes) != 4 and launch_error is None:
        launch_error = "controller did not launch exactly four concurrent day containers"
    pending = list(processes)
    failures = []
    receipt_errors = []
    while pending:
        for item in list(pending):
            process = item["process"]
            if process.poll() is None:
                try:
                    sample = docker_stat_sample(item["name"])
                except Exception as error:
                    receipt_errors.append(f"lane-{item['index']:02d} Docker stats failure: {error}")
                    sample = None
                if sample is not None:
                    item["samples"].append(sample)
            if process.poll() is not None:
                result = process.wait()
                item["log"].close()
                completed_at = time.time()
                try:
                    write_resource_receipt(stage, isolated_job=f"lane-{item['index']:02d}", container_name=item["name"], command=item["command"], samples=item["samples"], returncode=result, started_at=item["started_at"], completed_at=completed_at)
                except Exception as error:
                    receipt_errors.append(f"lane-{item['index']:02d} resource receipt failure: {error}")
                if result != 0:
                    failures.append(item["index"])
                pending.remove(item)
        if pending:
            time.sleep(1.0)
    if launch_error is not None:
        fail(launch_error)
    if receipt_errors:
        fail("day-lane resource-peak receipt failure: " + "; ".join(receipt_errors))
    if failures:
        fail("one or more isolated day lanes failed without retry: " + ",".join(map(str, failures)))

def run_finalizer(stage, source_root, inputs, container, phi_manifest_sha):
    output = stage["finalization"] / "target-set"
    required_new(output, "aggregate target-set output")
    name = "alakazam-prize-plan-v2-r23-" + CONFIG["publication"]["stage_nonce"] + "-finalize"
    command = docker_base(container, name) + [
        "--mount", mount(source_root, "/source", readonly=True),
        "--mount", mount(inputs["overlay_root"], "/overlay", readonly=True),
        "--mount", mount(stage["fit"] / "phi-fit", "/phi", readonly=True),
        "--mount", mount(stage["days"], "/days", readonly=True),
        "--mount", mount(stage["finalization"], "/publish", readonly=False),
    ] + low_priority_python(container, "/source/scripts/build_alakazam_prize_plan_targets_v2.py", "finalize", "--output-root", "/publish/target-set", "--goal-contract", "/source/goals/alakazam-elmo-rule-derivative/contract.json", "--expected-goal-contract-sha256", CONFIG["contract"]["sha256"], "--phi-fit-manifest", "/phi/manifests/" + one_json(stage["fit"] / "phi-fit" / "manifests", ".phi-fit-manifest.json", "Phi manifests").name, "--expected-phi-fit-manifest-sha256", phi_manifest_sha, "--gamma", str(CONFIG["fit"]["gamma"]), "--complete-action-overlay-manifest", "/overlay/" + str(Path(inputs["overlay_manifest"]).relative_to(Path(inputs["overlay_root"]))), "--expected-complete-action-overlay-manifest-sha256", inputs["expected_overlay_manifest_sha256"])
    for day in WINDOW_DAYS:
        command += ["--day-artifact-root", f"/days/{day}"]
    run_monitored_container(stage, isolated_job="finalize", container_name=name, command=command, log_path=stage["logs"] / "finalize.log", label="aggregate target-set finalizer")
    return output

def assert_portable_paths(value, label="target set"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("path") or key.endswith("root"):
                if isinstance(child, str):
                    portable_relative(child, f"{label}.{key}")
            assert_portable_paths(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_portable_paths(child, f"{label}[{index}]")

def validate_final_target_set(root, preflight, phi_manifest_sha, phi_receipt_sha):
    root = existing_directory(root, "aggregate target-set root")
    manifest_path = one_json(root / "manifests", ".prize-plan-target-set-manifest.json", "aggregate manifests")
    receipt_path = one_json(root / "receipts", ".prize-plan-target-set-receipt.json", "aggregate receipts")
    manifest = read_json(manifest_path, "aggregate target-set manifest")
    receipt = read_json(receipt_path, "aggregate target-set receipt")
    if manifest.get("schema") != TARGET_SET_MANIFEST_SCHEMA or receipt.get("schema") != TARGET_SET_RECEIPT_SCHEMA:
        fail("foreign aggregate target-set schema")
    if manifest.get("owner_goal_revision") != 23 or receipt.get("owner_goal_revision") != 23:
        fail("aggregate target-set owner revision drifted")
    if manifest.get("required_authority") != "revision_23_prize_plan_v2_h3_actor_canary" or receipt.get("required_authority") != "revision_23_prize_plan_v2_h3_actor_canary":
        fail("aggregate target-set authority binding drifted")
    if manifest.get("goal_contract_goal_revision") != CONFIG["contract"]["canonical_goal_revision"] or receipt.get("goal_contract_goal_revision") != CONFIG["contract"]["canonical_goal_revision"]:
        fail("aggregate target-set wrapper revision binding drifted")
    goal = manifest.get("goal_contract")
    if not isinstance(goal, dict) or goal.get("sha256") != CONFIG["contract"]["sha256"] or goal.get("goal_revision") != CONFIG["contract"]["canonical_goal_revision"] or goal.get("required_authority") != "revision_23_prize_plan_v2_h3_actor_canary" or goal.get("semantic_owner_goal_revision") != 23:
        fail("aggregate target-set contract binding drifted")
    portable_relative(goal.get("path"), "aggregate goal contract path")
    if sha_file(child_under(root, goal.get("path"), "aggregate goal contract")) != CONFIG["contract"]["sha256"]:
        fail("aggregate target-set goal contract object drifted")
    if receipt.get("goal_contract_sha256") != CONFIG["contract"]["sha256"] or receipt.get("target_set_manifest_sha256") != sha_file(manifest_path):
        fail("aggregate target-set receipt does not bind manifest")
    phi = manifest.get("phi_fit")
    if not isinstance(phi, dict):
        fail("aggregate target-set lacks Phi binding")
    fit_manifest = phi.get("fit_manifest")
    fit_receipt = phi.get("fit_receipt")
    if not isinstance(fit_manifest, dict) or not isinstance(fit_receipt, dict) or phi.get("fit_scope") != "sealed_train_split_only":
        fail("aggregate target-set Phi binding is malformed")
    if fit_manifest.get("sha256") != phi_manifest_sha or fit_receipt.get("sha256") != phi_receipt_sha:
        fail("aggregate target-set Phi binding drifted")
    portable_relative(phi.get("portable_root"), "aggregate Phi portable root")
    portable_relative(fit_manifest.get("path"), "aggregate Phi manifest path")
    portable_relative(fit_receipt.get("path"), "aggregate Phi receipt path")
    if sha_file(child_under(root, fit_manifest.get("path"), "aggregate Phi manifest")) != phi_manifest_sha or sha_file(child_under(root, fit_receipt.get("path"), "aggregate Phi receipt")) != phi_receipt_sha:
        fail("aggregate target-set Phi artifact path binding drifted")
    if receipt.get("phi_fit_manifest_sha256") != phi_manifest_sha or receipt.get("phi_fit_receipt_sha256") != phi_receipt_sha:
        fail("aggregate target-set receipt Phi binding drifted")
    transform = manifest.get("target_value_transform", {})
    transform_path = transform.get("path") if isinstance(transform, dict) else None
    transform_sha = transform.get("sha256") if isinstance(transform, dict) else None
    portable_relative(transform_path, "target value transform path")
    transform_file = child_under(root, transform_path, "target value transform")
    if sha_file(transform_file) != transform_sha:
        fail("aggregate target-set target-value transform binding drifted")
    transform_document = read_json(transform_file, "target value transform")
    if transform.get("schema") != "poke_bot.alakazam_prize_plan_target_value_transform/v2" or transform_document.get("gamma") != CONFIG["fit"]["gamma"]:
        fail("aggregate target-set gamma drifted")
    if receipt.get("target_value_transform_sha256") != transform_sha:
        fail("aggregate target-set receipt transform binding drifted")
    days = manifest.get("target_days")
    shards = manifest.get("all_20_target_shards")
    raws = manifest.get("all_20_raw_episode_zip_sha256s")
    if not isinstance(days, list) or not isinstance(shards, list) or not isinstance(raws, list) or len(days) != 20 or len(shards) != 20 or len(raws) != 20:
        fail("aggregate target-set lacks exact twenty-day inventories")
    if [item.get("utc_day") for item in days] != list(WINDOW_DAYS):
        fail("aggregate target-set day order drifted")
    if [item.get("split") for item in days] != [SPLIT_BY_DAY[day] for day in WINDOW_DAYS]:
        fail("aggregate target-set split inventory drifted")
    if sum(positive_int(item.get("row_count"), "aggregate target row count") for item in shards) != CONFIG["expected_complete_action_programs"]:
        fail("aggregate target-set row count drifted")
    raw_by_day = {item["utc_day"]: item for item in preflight["raw_episode_sha256s"]}
    job_by_day = {item["utc_day"]: item for item in preflight["jobs"]}
    for day, item, shard, raw in zip(WINDOW_DAYS, days, shards, raws, strict=True):
        if item.get("utc_day") != day or shard.get("utc_day") != day or raw.get("utc_day") != day:
            fail("aggregate target-set day identities are misaligned")
        if raw.get("sha256") != raw_by_day[day]["sha256"] or raw.get("size_bytes") != raw_by_day[day]["size_bytes"]:
            fail(f"aggregate raw ZIP binding drifted for {day}")
        if item.get("complete_action_overlay", {}).get("sha256") != job_by_day[day]["overlay_sha256"]:
            fail(f"aggregate overlay binding drifted for {day}")
        day_root = item.get("day_artifact_root")
        target_path = item.get("target_shard", {}).get("path")
        portable_relative(day_root, f"day artifact root for {day}")
        portable_relative(target_path, f"target shard path for {day}")
        target = child_under(root, target_path, f"published target shard for {day}")
        regular_file(target, f"published target shard for {day}")
        item_target = item.get("target_shard")
        if not isinstance(item_target, dict) or item_target.get("sha256") != shard.get("sha256") or item_target.get("size_bytes") != shard.get("size_bytes") or item_target.get("row_count") != shard.get("row_count"):
            fail(f"aggregate target-shard binding drifted for {day}")
        if target.stat().st_size != shard.get("size_bytes") or sha_file(target) != shard.get("sha256"):
            fail(f"published target shard parity failed for {day}")
    assert_portable_paths(manifest)
    assert_portable_paths(receipt)
    return {"manifest_path": str(manifest_path), "manifest_sha256": sha_file(manifest_path), "receipt_path": str(receipt_path), "receipt_sha256": sha_file(receipt_path)}

def validate_resource_peak_receipts(stage):
    expected = ["fit-phi", "lane-00", "lane-01", "lane-02", "lane-03", "finalize"]
    records = []
    for isolated_job in expected:
        path = regular_file(stage["resource_peaks"] / f"{isolated_job}.resource-peak.json", f"resource peak receipt for {isolated_job}")
        receipt = read_json(path, f"resource peak receipt for {isolated_job}")
        if receipt.get("schema") != "poke_bot.alakazam_prize_plan_v2_elmo_resource_peak_receipt/v1" or receipt.get("isolated_job") != isolated_job:
            fail("resource peak receipt schema/job identity drifted")
        if receipt.get("owner_goal_revision") != 23 or receipt.get("required_authority") != "revision_23_prize_plan_v2_h3_actor_canary":
            fail("resource peak receipt owner binding drifted")
        if receipt.get("container_image_sha256") != CONFIG["container"]["image_sha256"] or receipt.get("container_exit_code") != 0:
            fail("resource peak receipt container identity/result drifted")
        if receipt.get("measurement_method") != "docker_stats_no_stream_poll_v1" or positive_int(receipt.get("sample_count"), "resource peak sample count") < 1:
            fail("resource peak receipt measurement evidence is absent")
        if positive_int(receipt.get("peak_memory_bytes"), "peak memory bytes") < 0 or positive_int(receipt.get("experiment_ram_peak_bytes"), "experiment RAM peak bytes") < 0 or positive_int(receipt.get("peak_process_count"), "peak process count") < 0:
            fail("resource peak receipt has invalid peak values")
        if receipt.get("experiment_ram_measurement_method") != "docker_stats_no_stream_poll_v1" or receipt.get("experiment_ram_peak_bytes") != receipt.get("peak_memory_bytes"):
            fail("resource peak receipt misses contract RAM measurement method")
        cpu_peak = receipt.get("peak_cpu_utilization_percent")
        if not isinstance(cpu_peak, (int, float)) or isinstance(cpu_peak, bool) or cpu_peak < 0.0:
            fail("resource peak receipt has invalid CPU peak")
        cpu_evidence = receipt.get("cpu_process_or_utilization_peak")
        if not isinstance(cpu_evidence, dict) or cpu_evidence.get("measurement_method") != "docker_stats_no_stream_poll_v1" or cpu_evidence.get("peak_cpu_utilization_percent") != cpu_peak or cpu_evidence.get("peak_process_count") != receipt.get("peak_process_count"):
            fail("resource peak receipt misses contract CPU peak evidence")
        if receipt.get("gpu", {}).get("applicable") is not False or receipt.get("gpu_memory_peak_bytes_per_device_or_not_applicable") != "not_applicable_no_gpu_device_mounted" or receipt.get("gpu_utilization_peak_per_device_or_not_applicable") != "not_applicable_no_gpu_device_mounted" or receipt.get("model_optimizer_or_training_performed") is not False:
            fail("resource peak receipt isolation drifted")
        records.append({"relative_path": str(path.relative_to(stage["root"])), "sha256": sha_file(path)})
    if len(list(stage["resource_peaks"].iterdir())) != len(expected):
        fail("resource peak receipt inventory drifted")
    return records

def seal_controller_completion(stage, *, preflight, phi_manifest_sha, phi_receipt_sha, publication):
    """Seal controller-owned resource evidence without changing target-set bytes."""
    resource_peaks = validate_resource_peak_receipts(stage)
    completion = {
        "schema": "poke_bot.alakazam_prize_plan_v2_elmo_materialization_completion/v1",
        "owner_goal_revision": 23,
        "required_authority": "revision_23_prize_plan_v2_h3_actor_canary",
        "goal_contract_sha256": CONFIG["contract"]["sha256"],
        "phi_fit_manifest_sha256": phi_manifest_sha,
        "phi_fit_receipt_sha256": phi_receipt_sha,
        "expected_complete_action_programs": CONFIG["expected_complete_action_programs"],
        "complete_action_overlay_manifest_sha256": preflight["overlay_manifest_sha256"],
        "gamma": CONFIG["fit"]["gamma"],
        "fit_configuration": CONFIG["fit"]["configuration"],
        "publication": publication,
        "resource_peak_receipts": resource_peaks,
        "resource_peak_receipt_count": len(resource_peaks),
        "exact_isolated_jobs": ["fit-phi", "lane-00", "lane-01", "lane-02", "lane-03", "finalize"],
        "model_optimizer_or_training_performed": False,
        "created_at_unix_seconds": time.time(),
    }
    path = stage["root"] / "completion.json"
    write_exclusive(path, canonical_bytes(completion))
    return {"relative_path": str(path.relative_to(stage["root"])), "sha256": sha_file(path), "resource_peak_receipts": resource_peaks}

def main():
    if CONFIG.get("schema") != "poke_bot.alakazam_prize_plan_v2_elmo_materialization_controller/v1":
        fail("foreign controller configuration")
    if CONFIG.get("owner_goal_revision") != 23 or CONFIG.get("required_authority") != "revision_23_prize_plan_v2_h3_actor_canary":
        fail("controller semantic owner drifted")
    if CONFIG.get("lane_count_exact") != 4 or CONFIG.get("days_per_lane_exact") != 5:
        fail("controller lane topology drifted")
    if CONFIG.get("fit", {}).get("gamma") != 1.0:
        fail("r23 first target-set gamma must be explicit 1.0")
    contract_preflight(CONFIG["contract"])
    preflight = overlay_preflight(CONFIG["inputs"], CONFIG["expected_complete_action_programs"])
    verify_image(CONFIG["container"])
    lanes = allocate_lanes(preflight["jobs"])
    snapshot_root = Path(CONFIG["source_snapshot"]["root"])
    stage_root = Path(CONFIG["publication"]["private_stage_root"])
    target_root = Path(CONFIG["publication"]["target_root"])
    planned_new(snapshot_root, "source snapshot root", allow_create_parent=True)
    required_new(stage_root, "private target-materialization stage")
    required_new(target_root, "published target-set root")
    report = {
        "schema": "poke_bot.alakazam_prize_plan_v2_elmo_materialization_plan/v1",
        "execute": bool(CONFIG["execute"]),
        "canonical_goal_contract_sha256": CONFIG["contract"]["sha256"],
        "canonical_goal_revision": CONFIG["contract"]["canonical_goal_revision"],
        "required_authority": CONFIG["required_authority"],
        "expected_complete_action_programs": CONFIG["expected_complete_action_programs"],
        "fit": CONFIG["fit"],
        "overlay_manifest_sha256": preflight["overlay_manifest_sha256"],
        "base_pack_completion_sha256": preflight["base_pack_completion_sha256"],
        "raw_episode_zip_sha256s": preflight["raw_episode_sha256s"],
        "source_snapshot_root": str(snapshot_root),
        "source_snapshot_manifest_sha256": CONFIG["source_snapshot"]["manifest_sha256"],
        "private_stage_root": str(stage_root),
        "target_root": str(target_root),
        "lane_count": len(lanes),
        "lanes": [[job["utc_day"] for job in lane] for lane in lanes],
        "container": CONFIG["container"],
        "no_model_optimizer_or_training": True,
        "no_overwrite_retry_or_cleanup": True,
    }
    if not CONFIG["execute"]:
        print(json.dumps(report, sort_keys=True))
        return
    source_root = snapshot_source(CONFIG["source_snapshot"])
    stage = create_stage(stage_root, lanes, preflight)
    phi_manifest_path, phi_manifest_sha, _phi_receipt_path, phi_receipt_sha = run_phi_fit(stage, source_root, CONFIG["inputs"], CONFIG["container"])
    run_day_lanes(stage, source_root, CONFIG["inputs"], CONFIG["container"], lanes, phi_manifest_sha)
    final_stage = run_finalizer(stage, source_root, CONFIG["inputs"], CONFIG["container"], phi_manifest_sha)
    final_report = validate_final_target_set(final_stage, preflight, phi_manifest_sha, phi_receipt_sha)
    completion = seal_controller_completion(stage, preflight=preflight, phi_manifest_sha=phi_manifest_sha, phi_receipt_sha=phi_receipt_sha, publication=final_report)
    rename_no_replace(final_stage, target_root)
    published_report = validate_final_target_set(target_root, preflight, phi_manifest_sha, phi_receipt_sha)
    if published_report["manifest_sha256"] != final_report["manifest_sha256"] or published_report["receipt_sha256"] != final_report["receipt_sha256"]:
        fail("post-publication target-set identity changed")
    report.update({"published": True, "publication": published_report, "phi_fit_manifest_path": str(phi_manifest_path), "phi_fit_manifest_sha256": phi_manifest_sha, "phi_fit_receipt_sha256": phi_receipt_sha, "controller_completion": completion, "resource_peak_receipts": completion["resource_peak_receipts"]})
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"schema": "poke_bot.alakazam_prize_plan_v2_elmo_materialization_error/v1", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise
'''


def render_remote_program(config: Mapping[str, Any]) -> str:
    """Return the self-contained, injection-safe remote program."""

    encoded = base64.b64encode(canonical_bytes(dict(config))).decode("ascii")
    return _REMOTE_PROGRAM_TEMPLATE.replace("__CONFIG_BASE64__", repr(encoded))


def run_remote_program(host: str, program: str) -> subprocess.CompletedProcess[str]:
    """Run the fixed low-priority remote interpreter command only."""

    if not host or any(character.isspace() for character in host):
        raise PrizePlanMaterializationError("Elmo SSH host is malformed")
    try:
        return subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                host,
                "/usr/bin/ionice -c 3 /usr/bin/nice -n 15 python3 -",
            ],
            input=program,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise PrizePlanMaterializationError(f"could not invoke Elmo SSH: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="perform the one create-only Elmo run")
    parser.add_argument("--host", default="elmo")
    parser.add_argument("--goal-contract", type=Path, default=ROOT / "goals/alakazam-elmo-rule-derivative/contract.json")
    parser.add_argument("--expected-goal-contract-sha256", default=DEFAULT_CONTRACT_SHA256)
    parser.add_argument("--overlay-root", default=DEFAULT_OVERLAY_ROOT)
    parser.add_argument("--overlay-manifest", default=DEFAULT_OVERLAY_MANIFEST)
    parser.add_argument("--expected-overlay-manifest-sha256", default=DEFAULT_OVERLAY_MANIFEST_SHA256)
    parser.add_argument("--base-pack-completion", default=DEFAULT_BASE_PACK_COMPLETION)
    parser.add_argument("--expected-base-pack-completion-sha256", default=DEFAULT_BASE_PACK_COMPLETION_SHA256)
    parser.add_argument("--raw-episode-root", default=DEFAULT_RAW_EPISODE_ROOT)
    parser.add_argument("--source-snapshot-parent", default=DEFAULT_SNAPSHOT_PARENT)
    parser.add_argument("--private-stage-parent", default=DEFAULT_PRIVATE_STAGE_PARENT)
    parser.add_argument("--target-root", default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--image-sha256", default=ELMO_IMAGE_SHA256)
    parser.add_argument("--expected-complete-action-programs", type=int, default=EXPECTED_COMPLETE_ACTION_PROGRAMS)
    parser.add_argument("--stage-nonce", default=None, help="optional UUID-like nonce for a deterministic inspected plan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_complete_action_programs != EXPECTED_COMPLETE_ACTION_PROGRAMS:
        raise PrizePlanMaterializationError("r23 controller requires exactly 2,081,530 complete-action programs")
    snapshot = build_source_snapshot(args.goal_contract, expected_contract_sha256=args.expected_goal_contract_sha256)
    nonce = args.stage_nonce or str(uuid.uuid4())
    config = build_remote_config(
        execute=bool(args.execute),
        snapshot=snapshot,
        overlay_root=args.overlay_root,
        overlay_manifest=args.overlay_manifest,
        expected_overlay_manifest_sha256=args.expected_overlay_manifest_sha256,
        base_pack_completion=args.base_pack_completion,
        expected_base_pack_completion_sha256=args.expected_base_pack_completion_sha256,
        raw_episode_root=args.raw_episode_root,
        snapshot_parent=args.source_snapshot_parent,
        private_stage_parent=args.private_stage_parent,
        target_root=args.target_root,
        image_sha256=args.image_sha256,
        stage_nonce=nonce,
        expected_row_count=args.expected_complete_action_programs,
    )
    result = run_remote_program(args.host, render_remote_program(config))
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise PrizePlanMaterializationError(f"Elmo controller refused the run: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PrizePlanMaterializationError("Elmo controller did not return JSON") from error
    if not isinstance(payload, Mapping) or payload.get("execute") is not bool(args.execute):
        raise PrizePlanMaterializationError("Elmo controller response did not bind execution mode")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrizePlanMaterializationError as error:
        raise SystemExit(f"Prize-plan-v2 Elmo materialization refused: {error}") from error
