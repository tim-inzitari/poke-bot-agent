"""Safely materialize the revision-21 Alakazam critic target set on Elmo.

This controller is deliberately dry-run by default.  Its only mutating mode is
``--execute`` and it performs one create-only Elmo run: it seals a minimal
source snapshot, starts exactly four bounded Docker *day lanes*, aggregates the
twenty day artifacts, validates the portable aggregate, and atomically
publishes the result without replacing an existing target set.

It is intentionally not a service manager and does not start, stop, reload,
or otherwise interact with the active trainer, RTP shadow service, or remote
worker.  Failed runs leave their unique private stage intact for inspection;
there is no cleanup, overwrite, retry, or resume path in this controller.
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

OVERLAY_MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1"
TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_set_manifest/v1"
TARGET_SET_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_set_receipt/v1"
TARGET_OWNER_GOAL_REVISION = 21
EXPECTED_TARGET_ROW_COUNT = 2_081_530
LANE_COUNT = 4
CONTAINER_UID_GID = "950:950"
ELMO_IMAGE_SHA256 = (
    "sha256:0bcf2305438f8feecd9420cc37af8da4e3a2d81986e112597ad38fbe1e3f1aa3"
)

DEFAULT_OVERLAY_ROOT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-recent20-rtp-overlay-v1-attempt4"
)
DEFAULT_OVERLAY_MANIFEST = (
    DEFAULT_OVERLAY_ROOT
    +
    "/manifests/sha256-081e40d9b9cc98714abaa8945c8d176a9143bdb8e87aeeee0327878642b118bd"
    ".overlay-manifest.json"
)
DEFAULT_OVERLAY_MANIFEST_SHA256 = (
    "sha256:081e40d9b9cc98714abaa8945c8d176a9143bdb8e87aeeee0327878642b118bd"
)
DEFAULT_BASE_PACK_COMPLETION = (
    DEFAULT_OVERLAY_ROOT
    +
    "/bindings/sha256-e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba"
    ".base-completion.json"
)
DEFAULT_BASE_PACK_COMPLETION_SHA256 = (
    "sha256:e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba"
)
DEFAULT_RAW_EPISODE_ROOT = "/srv/poke-bot-agent/archive/episode-days"
DEFAULT_SNAPSHOT_PARENT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-action-critic-r21-source-snapshots"
)
DEFAULT_PRIVATE_STAGE_PARENT = "/srv/poke-bot-agent/outputs/experiments"
DEFAULT_TARGET_ROOT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-action-critic-targets-r21"
)

_SHA256_PREFIX = "sha256:"
_SOURCE_RELATIVE_PATHS = (
    "goals/alakazam-elmo-rule-derivative/contract.json",
    "poke_bot/__init__.py",
    "poke_bot/action_critic_targets.py",
    "scripts/build_alakazam_action_critic_targets.py",
)


class TargetMaterializationError(RuntimeError):
    """An immutable source or publication invariant was violated."""


@dataclass(frozen=True)
class ContractBinding:
    path: str
    sha256: str
    canonical_goal_revision: int
    target_owner_goal_revision: int


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
        raise TargetMaterializationError(f"{label} is not a full lowercase SHA-256")
    return raw


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TargetMaterializationError(f"{label} is not a non-negative integer")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TargetMaterializationError(f"{label} is absent")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
        raise TargetMaterializationError(f"{label} is not a portable relative path")
    return str(path)


def _require_regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise TargetMaterializationError(f"{label} is not a regular file: {resolved}")
    return resolved


def read_contract_binding(path: Path) -> ContractBinding:
    """Load the current canonical contract without pinning its top-level revision.

    Revision 22 adds unrelated fleet staging.  The r21 critic semantics remain
    authoritative, so this accepts any later canonical revision that retains
    that exact typed authority rather than incorrectly freezing the whole goal
    at top-level revision 21.
    """

    contract_path = _require_regular_file(path, label="canonical goal contract")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TargetMaterializationError("canonical goal contract is invalid JSON") from error
    if not isinstance(contract, Mapping):
        raise TargetMaterializationError("canonical goal contract is not an object")
    revision = contract.get("goal_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < TARGET_OWNER_GOAL_REVISION:
        raise TargetMaterializationError("canonical goal contract predates revision-21 authority")
    critic = contract.get("revision_21_draw_safe_critic_actor_canary")
    if not isinstance(critic, Mapping) or critic.get("owner_goal_revision") != TARGET_OWNER_GOAL_REVISION:
        raise TargetMaterializationError("canonical goal contract lacks revision-21 critic authority")
    target = critic.get("target_overlay")
    if not isinstance(target, Mapping) or target.get("manifest_schema") != TARGET_SET_MANIFEST_SCHEMA:
        raise TargetMaterializationError("canonical goal contract target-overlay schema drifted")
    return ContractBinding(
        path=str(contract_path),
        sha256=sha256_file(contract_path),
        canonical_goal_revision=revision,
        target_owner_goal_revision=TARGET_OWNER_GOAL_REVISION,
    )


def build_source_snapshot(contract_path: Path, *, project_root: Path = ROOT) -> SourceSnapshot:
    """Bind the exact four-file Elmo source snapshot before any remote write."""

    contract = read_contract_binding(contract_path)
    root = project_root.resolve()
    files: list[SnapshotFile] = []
    for relative in _SOURCE_RELATIVE_PATHS:
        candidate = _require_regular_file(root / relative, label=f"source snapshot member {relative}")
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise TargetMaterializationError("source snapshot member escaped the project root") from error
        contents = candidate.read_bytes()
        files.append(
            SnapshotFile(
                path=relative,
                sha256=sha256_bytes(contents),
                size_bytes=len(contents),
                contents_base64=base64.b64encode(contents).decode("ascii"),
            )
        )
    if tuple(item.path for item in files) != _SOURCE_RELATIVE_PATHS:
        raise TargetMaterializationError("source snapshot member inventory drifted")
    if files[0].sha256 != contract.sha256:
        raise TargetMaterializationError("source snapshot contract bytes drifted during preparation")
    # The snapshot moves between hosts; retain the canonical in-snapshot path
    # rather than leaking the controller machine's absolute workspace path.
    portable_contract = {
        **asdict(contract),
        "path": _SOURCE_RELATIVE_PATHS[0],
    }
    manifest = {
        "schema": "poke_bot.alakazam_action_critic_elmo_source_snapshot/v1",
        "target_owner_goal_revision": TARGET_OWNER_GOAL_REVISION,
        "canonical_goal_contract": portable_contract,
        "members": [
            {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in files
        ],
        "minimal_member_count": len(_SOURCE_RELATIVE_PATHS),
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
    expected_row_count: int = EXPECTED_TARGET_ROW_COUNT,
) -> tuple[OverlayDay, ...]:
    """Validate the canonical complete-action overlay inventory deterministically."""

    if manifest.get("schema") != OVERLAY_MANIFEST_SCHEMA:
        raise TargetMaterializationError("foreign complete-action overlay manifest schema")
    raw_shards = manifest.get("overlay_shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != len(WINDOW_DAYS):
        raise TargetMaterializationError("overlay manifest does not contain exactly twenty day shards")
    observed: dict[str, OverlayDay] = {}
    for entry in raw_shards:
        if not isinstance(entry, Mapping):
            raise TargetMaterializationError("overlay manifest contains a malformed day shard")
        day = entry.get("utc_day")
        if not isinstance(day, str) or day not in SPLIT_BY_DAY or day in observed:
            raise TargetMaterializationError("overlay manifest day inventory drifted")
        split = entry.get("split")
        if split != SPLIT_BY_DAY[day]:
            raise TargetMaterializationError(f"overlay manifest split drifted for {day}")
        overlay_sha = str(entry.get("sha256") or "")
        _sha_hex(overlay_sha, label=f"overlay SHA-256 for {day}")
        path = _safe_relative_path(entry.get("path"), label=f"overlay path for {day}")
        size = _positive_int(entry.get("size_bytes"), label=f"overlay size for {day}")
        programs = _positive_int(
            entry.get("complete_action_programs"),
            label=f"complete-action program count for {day}",
        )
        observed[day] = OverlayDay(
            utc_day=day,
            split=split,
            overlay_relative_path=path,
            overlay_sha256=overlay_sha,
            overlay_size_bytes=size,
            complete_action_programs=programs,
            raw_episode_filename=f"pokemon-tcg-ai-battle-episodes-{day}.zip",
        )
    if tuple(sorted(observed)) != WINDOW_DAYS:
        raise TargetMaterializationError("overlay manifest is not the exact contiguous recent-20 window")
    days = tuple(observed[day] for day in WINDOW_DAYS)
    if sum(day.complete_action_programs for day in days) != expected_row_count:
        raise TargetMaterializationError(
            "complete-action program total does not match the sealed critic target row count"
        )
    return days


def allocate_lanes(days: Sequence[OverlayDay]) -> tuple[tuple[OverlayDay, ...], ...]:
    """Allocate the twenty unique days to exactly four balanced, deterministic lanes."""

    if len(days) != len(WINDOW_DAYS) or {day.utc_day for day in days} != set(WINDOW_DAYS):
        raise TargetMaterializationError("lane allocation requires the exact twenty unique days")
    # A byte-and-row proxy keeps the four isolated jobs roughly balanced while
    # the count cap makes the resource topology unambiguous: five days/lane.
    ordered = sorted(
        days,
        key=lambda day: (-max(day.overlay_size_bytes, day.complete_action_programs), day.utc_day),
    )
    lanes: list[list[OverlayDay]] = [[] for _ in range(LANE_COUNT)]
    loads = [0] * LANE_COUNT
    for day in ordered:
        candidates = [index for index in range(LANE_COUNT) if len(lanes[index]) < 5]
        index = min(candidates, key=lambda candidate: (loads[candidate], candidate))
        lanes[index].append(day)
        loads[index] += day.overlay_size_bytes + day.complete_action_programs
    result = tuple(tuple(sorted(lane, key=lambda item: item.utc_day)) for lane in lanes)
    if len(result) != LANE_COUNT or any(len(lane) != 5 for lane in result):
        raise TargetMaterializationError("controller failed to build four five-day lanes")
    flattened = [item.utc_day for lane in result for item in lane]
    if sorted(flattened) != list(WINDOW_DAYS):
        raise TargetMaterializationError("controller allocated a duplicate or missing day")
    return result


def _validate_remote_absolute(path: str, *, label: str) -> str:
    value = PurePosixPath(path)
    if not value.is_absolute() or ".." in value.parts or str(value) in {"/", "."}:
        raise TargetMaterializationError(f"{label} must be a bounded absolute Elmo path")
    required_prefix = PurePosixPath("/srv/poke-bot-agent")
    try:
        value.relative_to(required_prefix)
    except ValueError as error:
        raise TargetMaterializationError(f"{label} escaped the Elmo artifact root") from error
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
    expected_row_count: int = EXPECTED_TARGET_ROW_COUNT,
) -> dict[str, Any]:
    """Build the only JSON input accepted by the remote single-run program."""

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
        raise TargetMaterializationError("private-stage nonce is malformed")
    snapshot_hex = _sha_hex(snapshot.manifest_sha256, label="source snapshot manifest SHA-256")
    snapshot_root = f"{snapshot_parent}/sha256-{snapshot_hex}"
    private_stage_root = f"{private_stage_parent}/.alakazam-action-critic-targets-r21-private-{stage_nonce}"
    return {
        "schema": "poke_bot.alakazam_action_critic_elmo_materialization_controller/v1",
        "execute": execute,
        "target_owner_goal_revision": TARGET_OWNER_GOAL_REVISION,
        "expected_target_row_count": expected_row_count,
        "lane_count_exact": LANE_COUNT,
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
            "private_stage_root": private_stage_root,
            "stage_nonce": stage_nonce,
            "target_root": target_root,
            "create_only": True,
            "atomic_no_clobber_publish": True,
            "cleanup_retry_or_overwrite_allowed": False,
        },
    }


# The remote program deliberately uses only Python's standard library.  It is
# streamed through `ssh elmo python3 -`, avoiding an interpolated remote shell
# command and making every path/data byte an explicit JSON input.
_REMOTE_PROGRAM_TEMPLATE = r"""
import base64
import ctypes
import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

CONFIG = json.loads(base64.b64decode(__CONFIG_BASE64__).decode("utf-8"))
SHA_PREFIX = "sha256:"
WINDOW_DAYS = tuple([f"2026-07-{number:02d}" for number in range(23, 32)] + [f"2026-08-{number:02d}" for number in range(1, 12)])
SPLIT_BY_DAY = {**{day: "train" for day in WINDOW_DAYS[:14]}, **{day: "validation" for day in WINDOW_DAYS[14:17]}, **{day: "evaluation" for day in WINDOW_DAYS[17:]}}
OVERLAY_MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1"
TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_set_manifest/v1"
TARGET_SET_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_set_receipt/v1"

def fail(message):
    raise RuntimeError(message)

def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def sha_bytes(value):
    return SHA_PREFIX + hashlib.sha256(value).hexdigest()

def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    # Validate a new path without mutating it during dry-run.  The
    # content-addressed source-snapshot parent is created only by --execute;
    # its trusted /experiments parent must already exist.
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

def ensure_create_only_parent(path, label):
    # Create a missing one-level artifact parent without replacement.
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

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def rename_no_replace(source, destination):
    # Linux renameat2(RENAME_NOREPLACE) gives the publication operation its
    # no-clobber guarantee.  Do not fall back to os.rename: that can replace
    # an empty pre-existing directory and would violate this controller.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        fail("renameat2(RENAME_NOREPLACE) is unavailable; refusing non-atomic publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        problem = ctypes.get_errno()
        if problem == errno.EEXIST:
            fail(f"refusing to replace existing publication target: {destination}")
        fail(f"renameat2 publication failed for {destination}: errno={problem}")
    fsync_directory(Path(destination).parent)

def positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} is not a non-negative integer")
    return value

def contract_preflight(contract):
    if not isinstance(contract, dict):
        fail("snapshot contract binding is malformed")
    revision = contract.get("canonical_goal_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 21:
        fail("snapshot contract is older than revision-21 critic authority")
    if contract.get("target_owner_goal_revision") != 21:
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
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"overlay manifest is invalid JSON: {error}")
    if not isinstance(manifest, dict) or manifest.get("schema") != OVERLAY_MANIFEST_SCHEMA:
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
            fail("overlay manifest contains a malformed day entry")
        day = entry.get("utc_day")
        if day not in SPLIT_BY_DAY or day in seen:
            fail("overlay manifest day inventory drifted")
        split = entry.get("split")
        if split != SPLIT_BY_DAY[day]:
            fail(f"overlay split drifted for {day}")
        rel = portable_relative(entry.get("path"), f"overlay path for {day}")
        overlay_path = child_under(overlay_root, rel, f"overlay object for {day}")
        regular_file(overlay_path, f"overlay object for {day}")
        overlay_sha = entry.get("sha256")
        sha_hex(overlay_sha, f"overlay SHA-256 for {day}")
        overlay_size = positive_int(entry.get("size_bytes"), f"overlay size for {day}")
        if overlay_path.stat().st_size != overlay_size or sha_file(overlay_path) != overlay_sha:
            fail(f"overlay object SHA-256 or size mismatch for {day}")
        rows = positive_int(entry.get("complete_action_programs"), f"overlay row count for {day}")
        raw_filename = f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        raw_path = child_under(raw_root, raw_filename, f"raw ZIP for {day}")
        regular_file(raw_path, f"raw ZIP for {day}")
        raw_sha = sha_file(raw_path)
        jobs.append({
            "utc_day": day,
            "split": split,
            "overlay_relative_path": rel,
            "overlay_sha256": overlay_sha,
            "overlay_size_bytes": overlay_size,
            "complete_action_programs": rows,
            "raw_episode_filename": raw_filename,
            "raw_episode_sha256": raw_sha,
            "raw_episode_size_bytes": raw_path.stat().st_size,
        })
        total += rows
        seen.add(day)
    if tuple(sorted(seen)) != WINDOW_DAYS:
        fail("overlay manifest is not the exact recent-20 day window")
    if total != expected_rows:
        fail("overlay row total does not match the sealed critic target count")
    base_path = regular_file(bounded_absolute(inputs["base_pack_completion"], "base-pack completion"), "base-pack completion")
    expected_base_sha = inputs["expected_base_pack_completion_sha256"]
    sha_hex(expected_base_sha, "expected base-pack completion SHA-256")
    if sha_file(base_path) != expected_base_sha:
        fail("base-pack completion SHA-256 mismatch")
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"base-pack completion is invalid JSON: {error}")
    if not isinstance(base, dict) or base.get("schema") != "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1":
        fail("foreign base-pack completion schema")
    return {
        "overlay_manifest_path": str(manifest_path),
        "overlay_manifest_sha256": expected_manifest_sha,
        "base_pack_completion_path": str(base_path),
        "base_pack_completion_sha256": expected_base_sha,
        "jobs": sorted(jobs, key=lambda item: item["utc_day"]),
        "raw_episode_sha256s": [
            {"utc_day": item["utc_day"], "sha256": item["raw_episode_sha256"], "size_bytes": item["raw_episode_size_bytes"]}
            for item in sorted(jobs, key=lambda item: item["utc_day"])
        ],
    }

def verify_image(container):
    expected = container["image_sha256"]
    sha_hex(expected, "Docker image SHA-256")
    try:
        inspected = subprocess.run(
            ["sudo", "-n", "docker", "image", "inspect", "--format", "{{.Id}}", expected],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        fail(f"Docker image inspection could not start: {error}")
    if inspected.returncode != 0 or inspected.stdout.strip() != expected:
        fail("required immutable Docker image is unavailable or mismatched")

def allocate_lanes(jobs):
    if len(jobs) != 20:
        fail("exactly twenty jobs are required for four lane allocation")
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
    inventory = sorted(job["utc_day"] for lane in lanes for job in lane)
    if inventory != list(WINDOW_DAYS):
        fail("lane inventory has a duplicate or missing day")
    return lanes

LANE_RUNNER = r'''
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
parser.add_argument("--output-parent", required=True)
args = parser.parse_args()
with open(args.jobs, "r", encoding="utf-8") as handle:
    jobs = json.load(handle)
if not isinstance(jobs, list) or len(jobs) != 5:
    raise SystemExit("lane job manifest must contain exactly five days")
for job in jobs:
    command = [
        sys.executable,
        os.path.join(args.source_root, "scripts/build_alakazam_action_critic_targets.py"),
        "--complete-action-overlay", os.path.join(args.overlay_root, job["overlay_relative_path"]),
        "--raw-episode-zip", os.path.join(args.raw_root, job["raw_episode_filename"]),
        "--output-root", os.path.join(args.output_parent, job["utc_day"]),
        "--utc-day", job["utc_day"],
        "--split", job["split"],
        "--goal-contract", os.path.join(args.source_root, "goals/alakazam-elmo-rule-derivative/contract.json"),
        "--expected-goal-contract-sha256", job["goal_contract_sha256"],
        "--expected-complete-action-overlay-sha256", job["overlay_sha256"],
        "--expected-raw-episode-zip-sha256", job["raw_episode_sha256"],
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
'''

def snapshot_source(snapshot):
    root = Path(snapshot["root"])
    ensure_create_only_parent(root.parent, "source snapshot parent")
    required_new(root, "source snapshot root")
    private = root.parent / ("." + root.name + ".private-" + CONFIG["publication"]["stage_nonce"])
    required_new(private, "source snapshot private root")
    os.mkdir(private, 0o755)
    manifest = snapshot["manifest"]
    body = canonical_bytes(manifest)
    if sha_bytes(body) != snapshot["manifest_sha256"]:
        fail("source snapshot manifest SHA-256 drifted")
    members = manifest.get("members")
    files = snapshot.get("files")
    if not isinstance(members, list) or not isinstance(files, list) or len(members) != 4 or len(files) != 4:
        fail("source snapshot must contain exactly four members")
    expected_names = [
        "goals/alakazam-elmo-rule-derivative/contract.json",
        "poke_bot/__init__.py",
        "poke_bot/action_critic_targets.py",
        "scripts/build_alakazam_action_critic_targets.py",
    ]
    if [item.get("path") for item in members] != expected_names or [item.get("path") for item in files] != expected_names:
        fail("source snapshot member inventory drifted")
    for expected, supplied in zip(members, files, strict=True):
        relative = portable_relative(supplied.get("path"), "source snapshot member")
        destination = private / relative
        destination.parent.mkdir(parents=True, exist_ok=False) if not destination.parent.exists() else None
        try:
            body = base64.b64decode(supplied.get("contents_base64", ""), validate=True)
        except Exception as error:
            fail(f"source snapshot payload is not valid base64: {error}")
        if len(body) != expected.get("size_bytes") or sha_bytes(body) != expected.get("sha256"):
            fail("source snapshot member bytes drifted")
        write_exclusive(destination, body)
    write_exclusive(private / "SOURCE_MANIFEST.json", canonical_bytes(manifest))
    # Verify every source byte after it is durable on Elmo, then make the tree
    # immutable before the no-clobber publication.
    for expected in members:
        member = regular_file(private / expected["path"], f"source snapshot member {expected['path']}")
        if member.stat().st_size != expected["size_bytes"] or sha_file(member) != expected["sha256"]:
            fail("source snapshot post-write verification failed")
    for path in sorted(private.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            os.chmod(path, 0o555)
        else:
            os.chmod(path, 0o444)
    os.chmod(private, 0o555)
    fsync_directory(private)
    rename_no_replace(private, root)
    return root

def create_stage(root, lanes, contract_sha):
    required_new(root, "private target-materialization stage")
    os.mkdir(root, 0o755)
    days = root / "days"
    lanes_dir = root / "lanes"
    logs = root / "logs"
    finalization = root / "finalization"
    for directory, mode in ((days, 0o755), (lanes_dir, 0o755), (logs, 0o755), (finalization, 0o755)):
        os.mkdir(directory, mode)
    write_exclusive(lanes_dir / "lane_runner.py", LANE_RUNNER.encode("utf-8"), 0o444)
    for index, lane in enumerate(lanes):
        jobs = []
        for job in lane:
            record = dict(job)
            record["goal_contract_sha256"] = contract_sha
            jobs.append(record)
        write_exclusive(lanes_dir / f"lane-{index:02d}.json", canonical_bytes(jobs), 0o444)
    fsync_directory(lanes_dir)
    fsync_directory(root)
    return {"root": root, "days": days, "lanes": lanes_dir, "logs": logs, "finalization": finalization}

def mount(source, target, readonly=False):
    result = f"type=bind,src={source},dst={target}"
    if readonly:
        result += ",readonly"
    return result

def docker_base(container, name):
    return [
        "sudo", "-n", "docker", "run", "--name", name,
        "--pull", "never",
        "--network", "none",
        "--read-only",
        "--user", container["user"],
        "--cpus", str(container["cpus"]),
        "--memory", container["memory"],
        "--memory-swap", container["memory"],
        "--pids-limit", "256",
        "--cpu-shares", "128",
        "--blkio-weight", "100",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONPATH=/source",
        "--entrypoint", "/usr/bin/ionice",
    ]

def low_priority_python(container, script, *arguments):
    # Docker's client priority does not reliably become the container-init
    # priority.  Make the immutable container entrypoint apply both controls
    # to the actual Python worker instead.
    return [
        container["image_sha256"],
        "-c", str(container["low_priority_ionice_class"]),
        "/usr/bin/nice", "-n", str(container["low_priority_nice"]),
        "python3", script, *arguments,
    ]

def run_day_lanes(stage, source_root, inputs, container, lanes):
    processes = []
    for index, _lane in enumerate(lanes):
        name = "alakazam-action-critic-r21-" + CONFIG["publication"]["stage_nonce"] + f"-lane-{index:02d}"
        command = docker_base(container, name) + [
            "--mount", mount(source_root, "/source", readonly=True),
            "--mount", mount(inputs["overlay_root"], "/overlay", readonly=True),
            "--mount", mount(inputs["raw_episode_root"], "/raw", readonly=True),
            "--mount", mount(stage["lanes"], "/lanes", readonly=True),
            "--mount", mount(stage["days"], "/outputs", readonly=False),
        ] + low_priority_python(
            container,
            "/lanes/lane_runner.py",
            "--jobs", f"/lanes/lane-{index:02d}.json",
            "--source-root", "/source",
            "--overlay-root", "/overlay",
            "--raw-root", "/raw",
            "--output-parent", "/outputs",
        )
        log = (stage["logs"] / f"lane-{index:02d}.log").open("xb")
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            log.close()
            # Do not signal already-started lanes.  Wait below so their own
            # immutable evidence is not disturbed by a separate launch fault.
            for running, opened, _ in processes:
                running.wait()
                opened.close()
            fail(f"could not start isolated day lane {index}: {error}")
        processes.append((process, log, command))
    if len(processes) != 4:
        fail("controller did not launch exactly four concurrent day containers")
    failures = []
    for index, (process, log, _command) in enumerate(processes):
        result = process.wait()
        log.close()
        if result != 0:
            failures.append(index)
    if failures:
        fail("one or more isolated day lanes failed without retry: " + ",".join(map(str, failures)))

def run_finalizer(stage, source_root, inputs, container):
    output = stage["finalization"] / "target-set"
    required_new(output, "aggregate target-set output")
    name = "alakazam-action-critic-r21-" + CONFIG["publication"]["stage_nonce"] + "-finalize"
    command = docker_base(container, name) + [
        "--mount", mount(source_root, "/source", readonly=True),
        "--mount", mount(inputs["overlay_root"], "/overlay", readonly=True),
        "--mount", mount(stage["days"], "/days", readonly=True),
        "--mount", mount(stage["finalization"], "/publish", readonly=False),
    ] + low_priority_python(
        container,
        "/source/scripts/build_alakazam_action_critic_targets.py",
        "finalize",
        "--output-root", "/publish/target-set",
        "--goal-contract", "/source/goals/alakazam-elmo-rule-derivative/contract.json",
        "--expected-goal-contract-sha256", CONFIG["contract"]["sha256"],
        "--base-pack-completion", "/overlay/" + str(Path(inputs["base_pack_completion"]).relative_to(Path(inputs["overlay_root"]))),
        "--expected-base-pack-completion-sha256", inputs["expected_base_pack_completion_sha256"],
        "--complete-action-overlay-manifest", "/overlay/" + str(Path(inputs["overlay_manifest"]).relative_to(Path(inputs["overlay_root"]))),
        "--expected-complete-action-overlay-manifest-sha256", inputs["expected_overlay_manifest_sha256"],
    )
    for day in WINDOW_DAYS:
        command += ["--day-artifact-root", f"/days/{day}"]
    log = (stage["logs"] / "finalize.log").open("xb")
    try:
        result = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT)
    finally:
        log.close()
    if result.returncode != 0:
        fail("aggregate target-set finalizer failed without retry")
    return output

def one_json(directory, suffix, label):
    directory = existing_directory(directory, label)
    members = sorted(path for path in directory.iterdir() if path.is_file() and not path.is_symlink() and path.name.endswith(suffix))
    if len(members) != 1:
        fail(f"{label} must contain exactly one {suffix} file")
    return members[0]

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

def validate_final_target_set(root, preflight):
    root = existing_directory(root, "aggregate target-set root")
    manifest_path = one_json(root / "manifests", ".target-set-manifest.json", "aggregate manifests")
    receipt_path = one_json(root / "receipts", ".target-set-receipt.json", "aggregate receipts")
    manifest_path = regular_file(manifest_path, "aggregate target-set manifest")
    receipt_path = regular_file(receipt_path, "aggregate target-set receipt")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"aggregate target set is invalid JSON: {error}")
    if not isinstance(manifest, dict) or manifest.get("schema") != TARGET_SET_MANIFEST_SCHEMA:
        fail("foreign aggregate target-set manifest schema")
    if not isinstance(receipt, dict) or receipt.get("schema") != TARGET_SET_RECEIPT_SCHEMA:
        fail("foreign aggregate target-set receipt schema")
    if manifest.get("owner_goal_revision") != 21 or manifest.get("goal_contract", {}).get("sha256") != CONFIG["contract"]["sha256"]:
        fail("aggregate target-set contract binding drifted")
    if receipt.get("target_set_manifest_sha256") != sha_file(manifest_path):
        fail("aggregate target-set receipt does not bind its manifest")
    days = manifest.get("target_days")
    shards = manifest.get("all_20_target_shards")
    raws = manifest.get("all_20_raw_episode_zip_sha256s")
    if not isinstance(days, list) or not isinstance(shards, list) or not isinstance(raws, list) or len(days) != 20 or len(shards) != 20 or len(raws) != 20:
        fail("aggregate target-set does not contain exact twenty-day inventories")
    if [item.get("utc_day") for item in days] != list(WINDOW_DAYS):
        fail("aggregate target-set day order drifted")
    if [item.get("split") for item in days] != [SPLIT_BY_DAY[day] for day in WINDOW_DAYS]:
        fail("aggregate target-set split inventory drifted")
    if sum(positive_int(item.get("row_count"), "aggregate target row count") for item in shards) != CONFIG["expected_target_row_count"]:
        fail("aggregate target-set row count differs from 2,081,530")
    raw_by_day = {item["utc_day"]: item for item in preflight["raw_episode_sha256s"]}
    overlay_by_day = {item["utc_day"]: item for item in preflight["jobs"]}
    for day, item, shard, raw in zip(WINDOW_DAYS, days, shards, raws, strict=True):
        if item.get("utc_day") != day or shard.get("utc_day") != day or raw.get("utc_day") != day:
            fail("aggregate target-set day identities are misaligned")
        if raw.get("sha256") != raw_by_day[day]["sha256"] or raw.get("size_bytes") != raw_by_day[day]["size_bytes"]:
            fail(f"aggregate target-set raw ZIP binding drifted for {day}")
        if item.get("complete_action_overlay", {}).get("sha256") != overlay_by_day[day]["overlay_sha256"]:
            fail(f"aggregate target-set overlay binding drifted for {day}")
        day_root = item.get("day_artifact_root")
        portable_relative(day_root, f"day artifact root for {day}")
        target_path = item.get("target_shard", {}).get("path")
        portable_relative(target_path, f"target shard path for {day}")
        resolved = child_under(root, str(PurePosixPath(day_root) / target_path), f"published target shard for {day}")
        regular_file(resolved, f"published target shard for {day}")
        if resolved.stat().st_size != shard.get("size_bytes") or sha_file(resolved) != shard.get("sha256"):
            fail(f"published target shard parity failed for {day}")
    assert_portable_paths(manifest)
    assert_portable_paths(receipt)
    return {"manifest_path": str(manifest_path), "manifest_sha256": sha_file(manifest_path), "receipt_path": str(receipt_path), "receipt_sha256": sha_file(receipt_path)}

def main():
    if CONFIG.get("schema") != "poke_bot.alakazam_action_critic_elmo_materialization_controller/v1":
        fail("foreign controller configuration")
    if CONFIG.get("target_owner_goal_revision") != 21 or CONFIG.get("lane_count_exact") != 4:
        fail("controller semantic owner or lane count drifted")
    contract_preflight(CONFIG["contract"])
    preflight = overlay_preflight(CONFIG["inputs"], CONFIG["expected_target_row_count"])
    verify_image(CONFIG["container"])
    lanes = allocate_lanes(preflight["jobs"])
    snapshot_root = Path(CONFIG["source_snapshot"]["root"])
    stage_root = Path(CONFIG["publication"]["private_stage_root"])
    target_root = Path(CONFIG["publication"]["target_root"])
    planned_new(snapshot_root, "source snapshot root", allow_create_parent=True)
    required_new(stage_root, "private target-materialization stage")
    required_new(target_root, "published target-set root")
    report = {
        "schema": "poke_bot.alakazam_action_critic_elmo_materialization_plan/v1",
        "execute": bool(CONFIG["execute"]),
        "canonical_goal_contract_sha256": CONFIG["contract"]["sha256"],
        "canonical_goal_revision": CONFIG["contract"]["canonical_goal_revision"],
        "target_owner_goal_revision": 21,
        "overlay_manifest_sha256": preflight["overlay_manifest_sha256"],
        "base_pack_completion_sha256": preflight["base_pack_completion_sha256"],
        "raw_episode_zip_sha256s": preflight["raw_episode_sha256s"],
        "complete_action_program_total": sum(item["complete_action_programs"] for item in preflight["jobs"]),
        "expected_target_row_count": CONFIG["expected_target_row_count"],
        "source_snapshot_root": str(snapshot_root),
        "source_snapshot_manifest_sha256": CONFIG["source_snapshot"]["manifest_sha256"],
        "private_stage_root": str(stage_root),
        "target_root": str(target_root),
        "lane_count": len(lanes),
        "lanes": [[job["utc_day"] for job in lane] for lane in lanes],
        "container": CONFIG["container"],
        "no_overwrite_retry_or_cleanup": True,
    }
    if not CONFIG["execute"]:
        print(json.dumps(report, sort_keys=True))
        return
    source_root = snapshot_source(CONFIG["source_snapshot"])
    stage = create_stage(stage_root, lanes, CONFIG["contract"]["sha256"])
    run_day_lanes(stage, source_root, CONFIG["inputs"], CONFIG["container"], lanes)
    final_stage = run_finalizer(stage, source_root, CONFIG["inputs"], CONFIG["container"])
    final_report = validate_final_target_set(final_stage, preflight)
    rename_no_replace(final_stage, target_root)
    published_report = validate_final_target_set(target_root, preflight)
    if (
        published_report["manifest_sha256"] != final_report["manifest_sha256"]
        or published_report["receipt_sha256"] != final_report["receipt_sha256"]
    ):
        fail("post-publication target-set identity changed")
    report.update({"published": True, "publication": published_report})
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"schema": "poke_bot.alakazam_action_critic_elmo_materialization_error/v1", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise
"""


def render_remote_program(config: Mapping[str, Any]) -> str:
    """Return the self-contained, injection-safe remote program for inspection/tests."""

    encoded = base64.b64encode(canonical_bytes(dict(config))).decode("ascii")
    return _REMOTE_PROGRAM_TEMPLATE.replace("__CONFIG_BASE64__", repr(encoded))


def run_remote_program(host: str, program: str) -> subprocess.CompletedProcess[str]:
    """Run only the fixed ``python3 -`` remote command; all data travels on stdin."""

    if not host or any(character.isspace() for character in host):
        raise TargetMaterializationError("Elmo SSH host is malformed")
    try:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3 -"],
            input=program,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise TargetMaterializationError(f"could not invoke Elmo SSH: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="perform the one create-only Elmo run")
    parser.add_argument("--host", default="elmo")
    parser.add_argument(
        "--goal-contract",
        type=Path,
        default=ROOT / "goals/alakazam-elmo-rule-derivative/contract.json",
    )
    parser.add_argument("--overlay-root", default=DEFAULT_OVERLAY_ROOT)
    parser.add_argument("--overlay-manifest", default=DEFAULT_OVERLAY_MANIFEST)
    parser.add_argument(
        "--expected-overlay-manifest-sha256", default=DEFAULT_OVERLAY_MANIFEST_SHA256
    )
    parser.add_argument("--base-pack-completion", default=DEFAULT_BASE_PACK_COMPLETION)
    parser.add_argument(
        "--expected-base-pack-completion-sha256",
        default=DEFAULT_BASE_PACK_COMPLETION_SHA256,
    )
    parser.add_argument("--raw-episode-root", default=DEFAULT_RAW_EPISODE_ROOT)
    parser.add_argument("--source-snapshot-parent", default=DEFAULT_SNAPSHOT_PARENT)
    parser.add_argument("--private-stage-parent", default=DEFAULT_PRIVATE_STAGE_PARENT)
    parser.add_argument("--target-root", default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--image-sha256", default=ELMO_IMAGE_SHA256)
    parser.add_argument("--expected-target-row-count", type=int, default=EXPECTED_TARGET_ROW_COUNT)
    parser.add_argument(
        "--stage-nonce",
        default=None,
        help="optional UUID-like nonce for a deterministic inspected dry-run plan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_target_row_count != EXPECTED_TARGET_ROW_COUNT:
        raise TargetMaterializationError("revision-21 controller requires exactly 2,081,530 rows")
    snapshot = build_source_snapshot(args.goal_contract)
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
        expected_row_count=args.expected_target_row_count,
    )
    result = run_remote_program(args.host, render_remote_program(config))
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise TargetMaterializationError(f"Elmo controller refused the run: {detail}")
    # The remote process returns exactly one structured plan or receipt.  Do
    # not fabricate success from transport exit status alone.
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise TargetMaterializationError("Elmo controller did not return JSON") from error
    if not isinstance(payload, Mapping) or payload.get("execute") is not bool(args.execute):
        raise TargetMaterializationError("Elmo controller response did not bind execution mode")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TargetMaterializationError as error:
        raise SystemExit(f"action-critic Elmo materialization refused: {error}") from error
