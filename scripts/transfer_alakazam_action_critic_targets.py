#!/usr/bin/env python3
"""Create-only four-stream transfer of the sealed action-critic target view.

This controller moves only the small, target-only revision-21 overlay from
Elmo to Bert.  It deliberately does *not* move replay ZIPs, start a trainer,
or alter a runtime.  The default mode is read-only planning; ``--execute``
uses exactly four independent lanes and writes only under the requested Bert
destination.

The portable target-set manifest itself remains byte-identical.  A separate
local target-view pointer records where its copied day artifacts live on Bert,
so an offline critic trainer can join it without treating Elmo paths as local
authority.  Every source and destination object is SHA-256/size checked.

OpenRsync on Bert has no ``--append-verify``.  A resumable private ``.part``
is therefore prefix-hashed against Elmo before ``rsync --append`` runs, and is
never promoted until its full identity matches the sealed target-set manifest.
No final file is overwritten; a verified existing final is skipped and every
other conflict fails closed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol


TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_set_manifest/v1"
TARGET_SET_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_set_receipt/v1"
TARGET_DAY_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_day_manifest/v1"
TARGET_DAY_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_day_receipt/v1"
TARGET_OVERLAY_SCHEMA = "poke_bot.alakazam_action_critic_target_overlay/v1"
BASE_COMPLETION_SCHEMA = "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"
OVERLAY_MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1"

TRANSFER_PLAN_SCHEMA = "poke_bot.alakazam_action_critic_target_transfer_plan/v1"
FILE_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_file_transfer_receipt/v1"
TARGET_VIEW_SCHEMA = "poke_bot.alakazam_action_critic_target_view/v1"
COMPLETION_SCHEMA = "poke_bot.alakazam_action_critic_target_transfer_completion/v1"

SOURCE_HOST = "elmo"
LANE_COUNT = 4
DEFAULT_DISK_FLOOR_BYTES = 20 * 1024 * 1024 * 1024
DEFAULT_DESTINATION_ROOT = Path(
    "/Users/example/Documents/poke-agent-critic-bootstrap/recent20-target-view-r21"
)
DEFAULT_CONTRACT_PATH = Path(
    "/Users/example/Documents/poke-agent-codex/"
    "goals/alakazam-elmo-rule-derivative/contract.json"
)
DEFAULT_EXPECTED_BASE_COMPLETION_SHA256 = (
    "sha256:e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba"
)
DEFAULT_EXPECTED_OVERLAY_MANIFEST_SHA256 = (
    "sha256:081e40d9b9cc98714abaa8945c8d176a9143bdb8e87aeeee0327878642b118bd"
)

WINDOW_DAYS = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 12)]
)
SPLIT_BY_DAY = {
    **{day: "train" for day in WINDOW_DAYS[:14]},
    **{day: "validation" for day in WINDOW_DAYS[14:17]},
    **{day: "evaluation" for day in WINDOW_DAYS[17:]},
}

SHA256_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
CONTENT_ADDRESS_RE = re.compile(r"^sha256-([0-9a-f]{64})")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SAFE_ABSOLUTE_POSIX_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TargetTransferError(RuntimeError):
    """A sealed target artifact cannot safely be transferred."""


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    """One regular, non-symlink immutable object."""

    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


class SourceReader(Protocol):
    """Read-only source interface used by the planner and transfer lanes."""

    host: str

    def read_bytes(self, path: str) -> bytes: ...

    def identities(self, paths: Sequence[str]) -> dict[str, FileIdentity]: ...

    def prefix_sha256(self, path: str, length: int) -> str: ...

    def append_to_part(self, path: str, part: Path) -> None: ...


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with Path(path).open("rb") as stream:
        while True:
            block_size = 8 * 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                block_size = min(block_size, remaining)
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining is not None and remaining != 0:
        raise TargetTransferError("file ended before the declared partial prefix")
    return "sha256:" + digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetTransferError(message)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetTransferError(f"{label} must be an object")
    return value


def _rows(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TargetTransferError(f"{label} must be an array")
    return list(value)


def _sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if SHA256_RE.fullmatch(text) is None:
        raise TargetTransferError(f"{label} must be a lowercase sha256:<hex> identity")
    return text


def _nonnegative_int(value: object, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetTransferError(f"{label} must be an exact integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise TargetTransferError(f"{label} must be positive")
    return int(value)


def _safe_relative(value: object, *, label: str) -> str:
    text = str(value or "")
    candidate = PurePosixPath(text)
    if (
        not text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise TargetTransferError(f"{label} must be a safe relative path")
    return "/".join(candidate.parts)


def _safe_source_root(value: Path | str, *, label: str) -> str:
    text = str(value)
    candidate = PurePosixPath(text)
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or SAFE_ABSOLUTE_POSIX_RE.fullmatch(text) is None
    ):
        raise TargetTransferError(f"{label} must be a safe absolute POSIX directory")
    return str(candidate)


def _source_member(root: str, relative: object, *, label: str) -> str:
    safe = _safe_relative(relative, label=label)
    return str(PurePosixPath(root).joinpath(*PurePosixPath(safe).parts))


def _source_bound_path(root: str, value: object, *, label: str) -> str:
    """Resolve a source binding path without permitting traversal.

    Portable target-set members must be relative.  The immutable source
    contract/base/overlay bindings may be absolute Elmo paths, because they
    predate the portable target artifact; those paths are never copied.
    """

    text = str(value or "")
    if not text:
        raise TargetTransferError(f"{label} path is absent")
    candidate = PurePosixPath(text)
    if candidate.is_absolute():
        return _safe_source_root(text, label=label)
    return _source_member(root, text, label=label)


def _destination_member(root: Path, relative: object, *, label: str) -> Path:
    safe = _safe_relative(relative, label=label)
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise TargetTransferError(f"{label} escapes destination root") from exc
    return candidate


def _regular_local_identity(path: Path | str, *, label: str) -> FileIdentity:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise TargetTransferError(f"{label} is unavailable: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TargetTransferError(f"{label} must be a regular non-symlink file: {candidate}")
    digest = sha256_file(candidate)
    try:
        after = candidate.lstat()
    except OSError as exc:
        raise TargetTransferError(f"{label} disappeared while being hashed: {candidate}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise TargetTransferError(f"{label} changed while being hashed: {candidate}")
    return FileIdentity(str(candidate.resolve()), digest, int(after.st_size))


class LocalSourceReader:
    """Fixture implementation with the same non-symlink rules as SSH."""

    host = "local"

    def read_bytes(self, path: str) -> bytes:
        identity = _regular_local_identity(path, label="local source object")
        body = Path(identity.path).read_bytes()
        if sha256_bytes(body) != identity.sha256 or len(body) != identity.size_bytes:
            raise TargetTransferError("local source object changed while being read")
        return body

    def identities(self, paths: Sequence[str]) -> dict[str, FileIdentity]:
        result: dict[str, FileIdentity] = {}
        for path in paths:
            if path not in result:
                result[path] = _regular_local_identity(path, label="local source object")
        return result

    def prefix_sha256(self, path: str, length: int) -> str:
        identity = _regular_local_identity(path, label="local source object")
        if length < 0 or length > identity.size_bytes:
            raise TargetTransferError("local source prefix length is invalid")
        return sha256_file(identity.path, limit=length)

    def append_to_part(self, path: str, part: Path) -> None:
        identity = _regular_local_identity(path, label="local source object")
        existing = 0
        if part.exists() or part.is_symlink():
            part_identity = _regular_local_identity(part, label="private transfer partial")
            existing = part_identity.size_bytes
        if existing > identity.size_bytes:
            raise TargetTransferError("private transfer partial exceeds source object")
        part.parent.mkdir(parents=True, exist_ok=True)
        with Path(identity.path).open("rb") as source:
            source.seek(existing)
            descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    written = 0
                    while written < len(block):
                        written += os.write(descriptor, block[written:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


_REMOTE_HELPER = r'''
import base64
import hashlib
import json
import os
import stat
import sys

def regular(path):
    state = os.lstat(path)
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
        raise ValueError("not a regular non-symlink file")
    return state

def digest(path, limit=None):
    value = hashlib.sha256()
    remaining = limit
    with open(path, "rb") as stream:
        while True:
            amount = 8 * 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                amount = min(amount, remaining)
            block = stream.read(amount)
            if not block:
                break
            value.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining is not None and remaining != 0:
        raise ValueError("short prefix")
    return "sha256:" + value.hexdigest()

try:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    operation = request["op"]
    if operation == "read":
        path = request["path"]
        regular(path)
        with open(path, "rb") as stream:
            result = {"body_b64": base64.b64encode(stream.read()).decode("ascii")}
    elif operation == "identities":
        rows = []
        for path in request["paths"]:
            before = regular(path)
            checksum = digest(path)
            after = regular(path)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("source changed while hashing")
            rows.append({"path": path, "sha256": checksum, "size_bytes": after.st_size})
        result = {"identities": rows}
    elif operation == "prefix":
        path = request["path"]
        length = request["length"]
        state = regular(path)
        if not isinstance(length, int) or isinstance(length, bool) or length < 0 or length > state.st_size:
            raise ValueError("invalid prefix length")
        result = {"sha256": digest(path, length)}
    else:
        raise ValueError("unknown operation")
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
except Exception as exc:
    sys.stderr.write("target-transfer read-only helper failed: %s\\n" % (exc,))
    raise SystemExit(2)
'''


class SSHSourceReader:
    """A read-only SSH source.  The only writer remains local rsync on Bert."""

    def __init__(self, host: str) -> None:
        if not SAFE_HOST_RE.fullmatch(host) or host == "local":
            raise TargetTransferError("source host must be a safe non-local SSH host")
        self.host = host
        encoded = base64.b64encode(_REMOTE_HELPER.encode("utf-8")).decode("ascii")
        self._remote_command = (
            "python3 -c \"import base64;exec(compile(base64.b64decode('"
            + encoded
            + "'),'target_transfer_remote_helper','exec'))\""
        )

    def _run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=15",
                    self.host,
                    self._remote_command,
                ],
                input=canonical_bytes(dict(request)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise TargetTransferError("could not launch read-only SSH probe") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise TargetTransferError(f"read-only source probe failed: {detail}")
        try:
            parsed = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TargetTransferError("source probe emitted invalid JSON") from exc
        return dict(_mapping(parsed, label="source probe response"))

    def read_bytes(self, path: str) -> bytes:
        safe = _safe_source_root(path, label="source object")
        response = self._run({"op": "read", "path": safe})
        encoded = response.get("body_b64")
        if not isinstance(encoded, str):
            raise TargetTransferError("source probe omitted object body")
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except ValueError as exc:
            raise TargetTransferError("source probe emitted invalid object encoding") from exc

    def identities(self, paths: Sequence[str]) -> dict[str, FileIdentity]:
        wanted = list(dict.fromkeys(_safe_source_root(path, label="source object") for path in paths))
        response = self._run({"op": "identities", "paths": wanted})
        rows = _rows(response.get("identities"), label="source object identities")
        if len(rows) != len(wanted):
            raise TargetTransferError("source probe identity count drifted")
        result: dict[str, FileIdentity] = {}
        for expected, raw in zip(wanted, rows, strict=True):
            row = _mapping(raw, label="source object identity")
            if row.get("path") != expected:
                raise TargetTransferError("source probe identity path drifted")
            result[expected] = FileIdentity(
                path=expected,
                sha256=_sha256(row.get("sha256"), label="source object SHA-256"),
                size_bytes=_nonnegative_int(row.get("size_bytes"), label="source object size"),
            )
        return result

    def prefix_sha256(self, path: str, length: int) -> str:
        safe = _safe_source_root(path, label="source object")
        response = self._run({"op": "prefix", "path": safe, "length": int(length)})
        return _sha256(response.get("sha256"), label="source prefix SHA-256")

    def append_to_part(self, path: str, part: Path) -> None:
        safe = _safe_source_root(path, label="source object")
        part.parent.mkdir(parents=True, exist_ok=True)
        if part.exists() or part.is_symlink():
            state = part.lstat()
            if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
                raise TargetTransferError("private transfer partial is not a regular file")
            # OpenRsync inherits the immutable source object's read-only mode
            # even with --no-perms.  Restore the controller-owned partial's
            # private writable mode before a verified resume.
            os.chmod(part, 0o600)
        # ``--append`` only writes after the already prefix-verified partial.
        # ``--no-links`` guards the destination even if a source changes after
        # its read-only identity probe; the subsequent full hash remains the
        # definitive integrity check.  Source host/path already pass the strict
        # shell-safe allowlists above, so do not require ``--protect-args``:
        # Bert's system OpenRsync 2.6.9 does not implement that newer option.
        command = [
            "rsync",
            "--append",
            "--partial",
            "--compress",
            "--compress-level=1",
            "--no-links",
            "--no-owner",
            "--no-group",
            "--no-perms",
            "--",
            f"{self.host}:{safe}",
            str(part),
        ]
        try:
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            except OSError as exc:
                raise TargetTransferError("could not launch rsync target transfer") from exc
        finally:
            if part.exists() and not part.is_symlink() and part.is_file():
                os.chmod(part, 0o600)
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise TargetTransferError(f"rsync target transfer failed: {detail}")


def _read_json(
    source: SourceReader,
    path: str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str, int, bytes]:
    body = source.read_bytes(path)
    digest = sha256_bytes(body)
    if expected_sha256 is not None and digest != expected_sha256:
        raise TargetTransferError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} actual={digest}"
        )
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetTransferError(f"{label} is not valid JSON") from exc
    return dict(_mapping(parsed, label=label)), digest, len(body), body


def _assert_content_addressed(path: str, body: bytes, *, label: str) -> None:
    match = CONTENT_ADDRESS_RE.match(PurePosixPath(path).name)
    if match is None:
        raise TargetTransferError(f"{label} lacks a content-addressed filename")
    if match.group(1) != sha256_bytes(body).removeprefix("sha256:"):
        raise TargetTransferError(f"{label} filename does not match its bytes")


def _read_local_contract(
    path: Path | str, *, expected_sha256: str | None
) -> tuple[Path, dict[str, Any], str]:
    identity = _regular_local_identity(path, label="local canonical contract")
    if expected_sha256 is not None and identity.sha256 != expected_sha256:
        raise TargetTransferError("local canonical contract SHA-256 mismatch")
    try:
        parsed = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetTransferError("local canonical contract is not valid JSON") from exc
    contract = dict(_mapping(parsed, label="local canonical contract"))
    _assert_revision_21_contract(contract)
    return Path(identity.path), contract, identity.sha256


def _assert_revision_21_contract(contract: Mapping[str, Any]) -> None:
    """Accept current contract revisions while requiring the r21 owner block.

    The active typed contract can advance for unrelated work (currently r22),
    so it must not be forced back to top-level revision 21.  The embedded
    revision-21 critic owner is the semantic authority for this transfer.
    """

    revision = _nonnegative_int(contract.get("goal_revision"), label="contract goal revision")
    if revision < 21:
        raise TargetTransferError("contract predates the revision-21 critic authority")
    owner = _mapping(
        contract.get("revision_21_draw_safe_critic_actor_canary"),
        label="revision-21 critic authority",
    )
    if owner.get("owner_goal_revision") != 21:
        raise TargetTransferError("revision-21 critic authority owner revision drifted")
    overlay = _mapping(owner.get("target_overlay"), label="revision-21 target overlay authority")
    actor = _mapping(owner.get("actor_advantage"), label="revision-21 actor authority")
    if (
        overlay.get("schema") != TARGET_OVERLAY_SCHEMA
        or overlay.get("manifest_schema") != TARGET_SET_MANIFEST_SCHEMA
        or overlay.get("row_join_identity")
        != [
            "utc_day",
            "source_archive_sha256",
            "source_member",
            "episode_id",
            "acting_seat",
            "env_step",
            "program_identity",
        ]
        or overlay.get("group_key")
        != ["source_archive_sha256", "episode_id", "acting_seat"]
        or overlay.get("group_order") != "strictly_increasing_env_step_no_duplicates"
        or overlay.get("required_terminal_fields")
        != ["z", "z_mask", "win_target_one_only_for_z_plus1", "win_target_mask"]
        or overlay.get("required_per_horizon_fields")
        != [
            "h",
            "mask",
            "unavailable_reason",
            "future_program_identity",
            "future_env_step",
            "own_remaining_before",
            "own_remaining_after",
            "opponent_remaining_before",
            "opponent_remaining_after",
            "own_taken",
            "opponent_taken",
            "differential",
        ]
        or overlay.get("hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed")
        is not False
    ):
        raise TargetTransferError("revision-21 target-overlay semantics drifted")
    prize_count = _mapping(overlay.get("prize_count"), label="revision-21 prize count authority")
    horizon = _mapping(overlay.get("horizon_definition"), label="revision-21 horizon authority")
    if (
        prize_count.get("valid_inclusive_range") != [1, 6]
        or prize_count.get("zero_behavior")
        != "mask_as_setup_or_uninitialized_never_treat_as_real_zero_progress"
        or horizon.get("values") != [1, 2, 3]
        or horizon.get("target") != "clip((own_taken-opponent_taken)/3,-1,+1)"
        or actor.get("enabled_formula")
        != "(z-V_existing(s))+0.05*m1*(Q_prize^1(s,a)-V_prize^1(s))"
        or actor.get("complete_action_value_broadcast_identically_across_selected_factorized_stages")
        is not True
        or actor.get("actor_gradient_into_sidecar_allowed") is not False
    ):
        raise TargetTransferError("revision-21 critic actor semantics drifted")
    transfer = _mapping(
        _mapping(
            contract.get("revision_20_conservative_critic_actor_canary"),
            label="revision-20 preserved critic authority",
        ).get("elmo_to_bert_bootstrap_transfer"),
        label="revision-20 four-stream transfer authority",
    )
    if (
        transfer.get("source_host") != "elmo"
        or transfer.get("destination_host") != "bert"
        or transfer.get("source_read_only") is not True
        or transfer.get("parallel_lanes_exact") != LANE_COUNT
        or transfer.get("unique_source_object_per_lane") is not True
        or transfer.get("raw_episode_zip_transfer_required") is not False
        or transfer.get("bert_disk_free_floor_bytes") != DEFAULT_DISK_FLOOR_BYTES
    ):
        raise TargetTransferError("revision-20 preserved target-transfer authority drifted")


def _entry(
    *,
    source_path: str,
    destination_relative: str,
    sha256: str,
    size_bytes: int,
    role: str,
    utc_day: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source_path": _safe_source_root(source_path, label="entry source path"),
        "destination_relative": _safe_relative(
            destination_relative, label="entry destination relative path"
        ),
        "sha256": _sha256(sha256, label="entry SHA-256"),
        "size_bytes": _nonnegative_int(size_bytes, label="entry size"),
        "role": str(role),
    }
    if utc_day is not None:
        if DAY_RE.fullmatch(utc_day) is None:
            raise TargetTransferError("entry UTC day is malformed")
        value["utc_day"] = utc_day
    if split is not None:
        if split not in {"train", "validation", "evaluation"}:
            raise TargetTransferError("entry split is malformed")
        value["split"] = split
    return value


def _assign_lpt_lanes(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(entries) < LANE_COUNT:
        raise TargetTransferError("exactly four streams require at least four unique objects")
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(LANE_COUNT)]
    totals = [0] * LANE_COUNT
    for raw in sorted(
        (dict(item) for item in entries),
        key=lambda item: (-int(item["size_bytes"]), str(item["destination_relative"])),
    ):
        lane_id = min(range(LANE_COUNT), key=lambda index: (totals[index], index))
        raw["lane_id"] = lane_id
        lanes[lane_id].append(raw)
        totals[lane_id] += int(raw["size_bytes"])
    output = [
        item
        for lane in lanes
        for item in sorted(
            lane,
            key=lambda row: (-int(row["size_bytes"]), str(row["destination_relative"])),
        )
    ]
    lane_rows = [
        {
            "lane_id": lane_id,
            "entry_count": len(lanes[lane_id]),
            "total_size_bytes": totals[lane_id],
            "source_paths": [
                str(item["source_path"])
                for item in sorted(
                    lanes[lane_id],
                    key=lambda row: (-int(row["size_bytes"]), str(row["destination_relative"])),
                )
            ],
        }
        for lane_id in range(LANE_COUNT)
    ]
    return output, lane_rows


def _exact_day_map(
    rows: object, *, label: str, require_split: bool = True
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _rows(rows, label=label):
        item = _mapping(raw, label=f"{label} item")
        day = str(item.get("utc_day") or "")
        if day not in SPLIT_BY_DAY or day in result:
            raise TargetTransferError(f"{label} day inventory drifted")
        if require_split and item.get("split") != SPLIT_BY_DAY[day]:
            raise TargetTransferError(f"{label} split drifted for {day}")
        result[day] = item
    if set(result) != set(WINDOW_DAYS):
        raise TargetTransferError(f"{label} is not the exact 20-day window")
    return result


def _assert_declared_identity(
    actual: FileIdentity,
    declared: Mapping[str, Any],
    *,
    label: str,
    require_size: bool = True,
) -> tuple[str, int]:
    expected_sha = _sha256(declared.get("sha256"), label=f"{label} SHA-256")
    expected_size = _nonnegative_int(declared.get("size_bytes"), label=f"{label} size")
    if actual.sha256 != expected_sha or (require_size and actual.size_bytes != expected_size):
        raise TargetTransferError(f"{label} SHA-256 or size mismatch")
    return expected_sha, expected_size


def _target_day_entries(
    *,
    source: SourceReader,
    source_root: str,
    day: str,
    set_item: Mapping[str, Any],
    expected_contract_sha: str,
    overlay_item: Mapping[str, Any],
    raw_item: Mapping[str, Any],
    aggregate_shard_item: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify one portable day root and derive its three exact copied objects."""

    split = SPLIT_BY_DAY[day]
    root_relative = _safe_relative(set_item.get("day_artifact_root"), label="day artifact root")
    if root_relative != f"days/{day}":
        raise TargetTransferError("portable target day root must be exactly days/<utc-day>")
    manifest_relative = _safe_relative(
        set_item.get("day_manifest_path"), label="target day manifest path"
    )
    receipt_relative = _safe_relative(
        set_item.get("day_receipt_path"), label="target day receipt path"
    )
    prefix = root_relative + "/"
    if not manifest_relative.startswith(prefix) or not receipt_relative.startswith(prefix):
        raise TargetTransferError("portable target day metadata escaped its day root")
    manifest_path = _source_member(source_root, manifest_relative, label="day manifest source")
    receipt_path = _source_member(source_root, receipt_relative, label="day receipt source")
    manifest_sha = _sha256(set_item.get("day_manifest_sha256"), label="target day manifest SHA-256")
    receipt_sha = _sha256(set_item.get("day_receipt_sha256"), label="target day receipt SHA-256")
    manifest, actual_manifest_sha, manifest_size, manifest_body = _read_json(
        source, manifest_path, label=f"target day manifest {day}", expected_sha256=manifest_sha
    )
    receipt, actual_receipt_sha, receipt_size, receipt_body = _read_json(
        source, receipt_path, label=f"target day receipt {day}", expected_sha256=receipt_sha
    )
    _assert_content_addressed(manifest_path, manifest_body, label=f"target day manifest {day}")
    _assert_content_addressed(receipt_path, receipt_body, label=f"target day receipt {day}")
    if (
        manifest.get("schema") != TARGET_DAY_MANIFEST_SCHEMA
        or manifest.get("owner_goal_revision") != 21
        or manifest.get("utc_day") != day
        or manifest.get("split") != split
    ):
        raise TargetTransferError(f"target day manifest schema/day/split drifted for {day}")
    contract = _mapping(manifest.get("goal_contract"), label="target day goal contract")
    if (
        _sha256(contract.get("sha256"), label="target day contract SHA-256")
        != expected_contract_sha
        or not isinstance(contract.get("goal_revision"), int)
        or contract.get("goal_revision") < 21
        or contract.get("critic_semantic_owner_goal_revision") != 21
        or contract.get("required_authority")
        != "revision_21_draw_safe_critic_actor_canary"
    ):
        raise TargetTransferError(f"target day contract binding drifted for {day}")
    complete = _mapping(manifest.get("complete_action_overlay"), label="target day overlay binding")
    if _sha256(complete.get("sha256"), label="target day overlay SHA-256") != _sha256(
        overlay_item.get("sha256"), label="overlay shard SHA-256"
    ):
        raise TargetTransferError(f"target day overlay binding drifted for {day}")
    raw = _mapping(manifest.get("raw_episode_zip"), label="target day raw ZIP binding")
    raw_sha = _sha256(raw.get("sha256"), label="target day raw ZIP SHA-256")
    raw_size = _nonnegative_int(raw.get("size_bytes"), label="target day raw ZIP size")
    if (
        raw_sha != _sha256(raw_item.get("sha256"), label="aggregate raw ZIP SHA-256")
        or raw_size != _nonnegative_int(raw_item.get("size_bytes"), label="aggregate raw ZIP size")
        or raw.get("source_archive_sha256_verified") is not True
    ):
        raise TargetTransferError(f"target day raw ZIP identity drifted for {day}")
    target = _mapping(manifest.get("target_shard"), label="target day shard")
    declared_target = _mapping(set_item.get("target_shard"), label="aggregate target shard")
    target_sha = _sha256(target.get("sha256"), label="target shard SHA-256")
    target_size = _nonnegative_int(target.get("size_bytes"), label="target shard size")
    target_rows = _nonnegative_int(target.get("row_count"), label="target shard row count", allow_zero=True)
    if (
        target_sha != _sha256(declared_target.get("sha256"), label="aggregate target shard SHA-256")
        or target_size != _nonnegative_int(declared_target.get("size_bytes"), label="aggregate target shard size")
        or target_rows != _nonnegative_int(declared_target.get("row_count"), label="aggregate target shard row count", allow_zero=True)
        or target_sha != _sha256(aggregate_shard_item.get("sha256"), label="all-target-shards SHA-256")
        or target_size != _nonnegative_int(aggregate_shard_item.get("size_bytes"), label="all-target-shards size")
        or target_rows != _nonnegative_int(aggregate_shard_item.get("row_count"), label="all-target-shards row count", allow_zero=True)
        or aggregate_shard_item.get("split") != split
        or aggregate_shard_item.get("utc_day") != day
    ):
        raise TargetTransferError(f"target day shard binding drifted for {day}")
    shard_subpath = _safe_relative(target.get("path"), label="target shard path")
    shard_relative = _safe_relative(
        f"{root_relative}/{shard_subpath}", label="target shard source relative path"
    )
    if not shard_relative.startswith(root_relative + "/objects/"):
        raise TargetTransferError("target shard must remain under its portable objects directory")
    shard_path = _source_member(source_root, shard_relative, label="target shard source")
    schema_subpath = _safe_relative(
        manifest.get("target_schema_path"), label="target day schema path"
    )
    schema_relative = _safe_relative(
        f"{root_relative}/{schema_subpath}", label="target schema source relative path"
    )
    if not schema_relative.startswith(root_relative + "/schemas/"):
        raise TargetTransferError("target schema must remain under its portable schemas directory")
    schema_path = _source_member(source_root, schema_relative, label="target schema source")
    schema_sha = _sha256(manifest.get("target_schema_sha256"), label="target schema SHA-256")
    schema_body = source.read_bytes(schema_path)
    if sha256_bytes(schema_body) != schema_sha:
        raise TargetTransferError(f"target day schema SHA-256 drifted for {day}")
    _assert_content_addressed(schema_path, schema_body, label=f"target day schema {day}")
    if (
        receipt.get("schema") != TARGET_DAY_RECEIPT_SCHEMA
        or receipt.get("owner_goal_revision") != 21
        or receipt.get("goal_contract_goal_revision") != contract.get("goal_revision")
        or receipt.get("critic_semantic_owner_goal_revision") != 21
        or _sha256(receipt.get("goal_contract_sha256"), label="target day receipt contract SHA-256")
        != expected_contract_sha
        or _sha256(receipt.get("manifest_sha256"), label="target day receipt manifest SHA-256")
        != actual_manifest_sha
        or receipt.get("manifest_path") != manifest_relative[len(prefix) :]
        or _sha256(receipt.get("complete_action_overlay_sha256"), label="target day receipt overlay SHA-256")
        != _sha256(complete.get("sha256"), label="target day overlay SHA-256")
        or _sha256(receipt.get("raw_episode_zip_sha256"), label="target day receipt raw SHA-256") != raw_sha
        or _sha256(receipt.get("target_shard_sha256"), label="target day receipt shard SHA-256") != target_sha
        or _sha256(receipt.get("target_schema_sha256"), label="target day receipt schema SHA-256")
        != schema_sha
        or _nonnegative_int(receipt.get("target_shard_size_bytes"), label="target day receipt shard size")
        != target_size
        or _nonnegative_int(receipt.get("target_row_count"), label="target day receipt row count", allow_zero=True)
        != target_rows
        or receipt.get("coverage") != manifest.get("coverage")
    ):
        raise TargetTransferError(f"target day receipt binding drifted for {day}")
    entries = [
        _entry(
            source_path=shard_path,
            destination_relative=shard_relative,
            sha256=target_sha,
            size_bytes=target_size,
            role="target_shard",
            utc_day=day,
            split=split,
        ),
        _entry(
            source_path=manifest_path,
            destination_relative=manifest_relative,
            sha256=actual_manifest_sha,
            size_bytes=manifest_size,
            role="target_day_manifest",
            utc_day=day,
            split=split,
        ),
        _entry(
            source_path=receipt_path,
            destination_relative=receipt_relative,
            sha256=actual_receipt_sha,
            size_bytes=receipt_size,
            role="target_day_receipt",
            utc_day=day,
            split=split,
        ),
        _entry(
            source_path=schema_path,
            destination_relative=schema_relative,
            sha256=schema_sha,
            size_bytes=len(schema_body),
            role="target_day_schema",
            utc_day=day,
            split=split,
        ),
    ]
    summary = {
        "utc_day": day,
        "split": split,
        "day_artifact_root_relative": root_relative,
        "day_manifest_relative": manifest_relative,
        "day_manifest_sha256": actual_manifest_sha,
        "day_receipt_relative": receipt_relative,
        "day_receipt_sha256": actual_receipt_sha,
        "target_shard_relative": shard_relative,
        "target_shard_sha256": target_sha,
        "target_shard_size_bytes": target_size,
        "target_row_count": target_rows,
        "target_schema_relative": schema_relative,
        "target_schema_sha256": schema_sha,
        "raw_episode_zip_sha256": raw_sha,
        "raw_episode_zip_size_bytes": raw_size,
        "complete_action_overlay_sha256": _sha256(complete.get("sha256"), label="target day overlay SHA-256"),
    }
    return entries, summary


def build_target_transfer_plan(
    *,
    source: SourceReader,
    source_root: Path | str,
    destination_root: Path | str,
    target_set_manifest_relative: str,
    target_set_receipt_relative: str,
    local_contract_path: Path | str,
    expected_contract_sha256: str | None = None,
    expected_base_completion_sha256: str = DEFAULT_EXPECTED_BASE_COMPLETION_SHA256,
    expected_overlay_manifest_sha256: str = DEFAULT_EXPECTED_OVERLAY_MANIFEST_SHA256,
    disk_floor_bytes: int = DEFAULT_DISK_FLOOR_BYTES,
) -> tuple[dict[str, Any], str]:
    """Read and validate the sealed portable target set, then make a 4-lane plan."""

    root = _safe_source_root(source_root, label="source target-set root")
    manifest_relative = _safe_relative(
        target_set_manifest_relative, label="target-set manifest relative path"
    )
    receipt_relative = _safe_relative(
        target_set_receipt_relative, label="target-set receipt relative path"
    )
    manifest_path = _source_member(root, manifest_relative, label="target-set manifest source")
    receipt_path = _source_member(root, receipt_relative, label="target-set receipt source")
    floor = _nonnegative_int(disk_floor_bytes, label="Bert disk floor")
    expected_base_sha = _sha256(
        expected_base_completion_sha256, label="expected base completion SHA-256"
    )
    expected_overlay_sha = _sha256(
        expected_overlay_manifest_sha256, label="expected overlay manifest SHA-256"
    )
    expected_contract = (
        None
        if expected_contract_sha256 is None or not str(expected_contract_sha256)
        else _sha256(expected_contract_sha256, label="expected contract SHA-256")
    )
    local_contract, contract, local_contract_sha = _read_local_contract(
        local_contract_path, expected_sha256=expected_contract
    )
    manifest, manifest_sha, manifest_size, manifest_body = _read_json(
        source, manifest_path, label="target-set manifest"
    )
    receipt, receipt_sha, receipt_size, receipt_body = _read_json(
        source, receipt_path, label="target-set receipt"
    )
    _assert_content_addressed(manifest_path, manifest_body, label="target-set manifest")
    _assert_content_addressed(receipt_path, receipt_body, label="target-set receipt")
    if manifest.get("schema") != TARGET_SET_MANIFEST_SCHEMA or manifest.get("owner_goal_revision") != 21:
        raise TargetTransferError("target-set manifest schema or owner revision drifted")
    if receipt.get("schema") != TARGET_SET_RECEIPT_SCHEMA or receipt.get("owner_goal_revision") != 21:
        raise TargetTransferError("target-set receipt schema or owner revision drifted")
    goal_binding = _mapping(manifest.get("goal_contract"), label="target-set goal contract")
    declared_contract_sha = _sha256(goal_binding.get("sha256"), label="target-set contract SHA-256")
    if declared_contract_sha != local_contract_sha:
        raise TargetTransferError("target-set contract is not the current local canonical contract")
    if (
        manifest.get("goal_contract_goal_revision") != contract.get("goal_revision")
        or manifest.get("critic_semantic_owner_goal_revision") != 21
        or manifest.get("required_critic_authority")
        != "revision_21_draw_safe_critic_actor_canary"
    ):
        raise TargetTransferError("target-set current contract/embedded critic authority drifted")
    source_contract_path = _source_bound_path(
        root, goal_binding.get("path"), label="target-set source contract"
    )
    source_contract, source_contract_sha, source_contract_size, _source_contract_body = _read_json(
        source,
        source_contract_path,
        label="target-set source contract",
        expected_sha256=declared_contract_sha,
    )
    _assert_revision_21_contract(source_contract)
    if source_contract != contract:
        # Equal hashes are decisive, but this explicit check makes a malformed
        # source decoding failure easier to diagnose and prevents accidental
        # binding to a non-canonical JSON serialization.
        raise TargetTransferError("source and local canonical contracts differ")
    base_binding = _mapping(manifest.get("base_pack_completion"), label="target-set base pack")
    base_sha = _sha256(base_binding.get("sha256"), label="target-set base completion SHA-256")
    if base_sha != expected_base_sha:
        raise TargetTransferError("target-set base completion identity is not the sealed recent-20 pack")
    base_path = _source_bound_path(root, base_binding.get("path"), label="target-set source base completion")
    base, actual_base_sha, actual_base_size, _base_body = _read_json(
        source, base_path, label="target-set source base completion", expected_sha256=base_sha
    )
    if base.get("schema") != BASE_COMPLETION_SCHEMA:
        raise TargetTransferError("target-set base completion schema drifted")
    if actual_base_size != _nonnegative_int(base_binding.get("size_bytes"), label="target-set base completion size"):
        raise TargetTransferError("target-set base completion size drifted")
    overlay_binding = _mapping(
        manifest.get("complete_action_overlay_manifest"), label="target-set complete-action overlay"
    )
    overlay_sha = _sha256(
        overlay_binding.get("sha256"), label="target-set complete-action overlay SHA-256"
    )
    if overlay_sha != expected_overlay_sha:
        raise TargetTransferError("target-set overlay identity is not the sealed recent-20 overlay")
    overlay_path = _source_bound_path(
        root, overlay_binding.get("path"), label="target-set source complete-action overlay"
    )
    overlay, actual_overlay_sha, actual_overlay_size, _overlay_body = _read_json(
        source,
        overlay_path,
        label="target-set source complete-action overlay",
        expected_sha256=overlay_sha,
    )
    if overlay.get("schema") != OVERLAY_MANIFEST_SCHEMA:
        raise TargetTransferError("target-set complete-action overlay schema drifted")
    if actual_overlay_size != _nonnegative_int(
        overlay_binding.get("size_bytes"), label="target-set complete-action overlay size"
    ):
        raise TargetTransferError("target-set complete-action overlay size drifted")
    overlay_by_day = _exact_day_map(overlay.get("overlay_shards"), label="complete-action overlay shards")
    if list(manifest.get("source_days") or []) != list(WINDOW_DAYS):
        raise TargetTransferError("target-set source-day order is not the sealed recent-20 window")
    split_days = _mapping(manifest.get("split_days"), label="target-set split days")
    if {
        split: list(split_days.get(split) or [])
        for split in ("train", "validation", "evaluation")
    } != {
        split: [day for day in WINDOW_DAYS if SPLIT_BY_DAY[day] == split]
        for split in ("train", "validation", "evaluation")
    }:
        raise TargetTransferError("target-set fixed 14/3/3 split inventory drifted")
    if manifest.get("episode_and_seat_group_split_disjoint") is not True:
        raise TargetTransferError("target-set does not attest episode/seat split disjointness")
    info = _mapping(manifest.get("information_boundary"), label="target-set information boundary")
    if info.get("hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed") is not False:
        raise TargetTransferError("target-set information boundary drifted")
    raw_by_day = _exact_day_map(
        manifest.get("all_20_raw_episode_zip_sha256s"),
        label="target-set raw ZIP identities",
        require_split=False,
    )
    for day, raw in raw_by_day.items():
        _sha256(raw.get("sha256"), label=f"raw ZIP SHA-256 for {day}")
        _nonnegative_int(raw.get("size_bytes"), label=f"raw ZIP size for {day}")
    descriptor_by_day = _exact_day_map(
        manifest.get("all_20_target_shards"), label="target-set target-shard identities"
    )
    set_days = _exact_day_map(manifest.get("target_days"), label="target-set day artifacts")
    if (
        _sha256(receipt.get("target_set_manifest_sha256"), label="target-set receipt manifest SHA-256")
        != manifest_sha
        or receipt.get("target_set_manifest_path") != manifest_relative
        or _sha256(receipt.get("goal_contract_sha256"), label="target-set receipt contract SHA-256")
        != declared_contract_sha
        or _sha256(receipt.get("base_pack_completion_sha256"), label="target-set receipt base SHA-256")
        != actual_base_sha
        or _sha256(
            receipt.get("complete_action_overlay_manifest_sha256"),
            label="target-set receipt overlay SHA-256",
        )
        != actual_overlay_sha
        or receipt.get("day_count") != len(WINDOW_DAYS)
        or receipt.get("goal_contract_goal_revision") != contract.get("goal_revision")
        or receipt.get("critic_semantic_owner_goal_revision") != 21
        or receipt.get("required_critic_authority")
        != "revision_21_draw_safe_critic_actor_canary"
        or receipt.get("coverage") != manifest.get("coverage")
        or receipt.get("episode_and_seat_group_split_disjoint") is not True
    ):
        raise TargetTransferError("target-set aggregate receipt binding drifted")
    entries: list[dict[str, Any]] = [
        _entry(
            source_path=manifest_path,
            destination_relative=manifest_relative,
            sha256=manifest_sha,
            size_bytes=manifest_size,
            role="target_set_manifest",
        ),
        _entry(
            source_path=receipt_path,
            destination_relative=receipt_relative,
            sha256=receipt_sha,
            size_bytes=receipt_size,
            role="target_set_receipt",
        ),
        _entry(
            source_path=source_contract_path,
            destination_relative=_safe_relative(
                goal_binding.get("path"), label="target-set contract binding path"
            ),
            sha256=declared_contract_sha,
            size_bytes=source_contract_size,
            role="target_set_binding",
        ),
        _entry(
            source_path=base_path,
            destination_relative=_safe_relative(
                base_binding.get("path"), label="target-set base binding path"
            ),
            sha256=actual_base_sha,
            size_bytes=actual_base_size,
            role="target_set_binding",
        ),
        _entry(
            source_path=overlay_path,
            destination_relative=_safe_relative(
                overlay_binding.get("path"), label="target-set overlay binding path"
            ),
            sha256=actual_overlay_sha,
            size_bytes=actual_overlay_size,
            role="target_set_binding",
        ),
    ]
    local_days: list[dict[str, Any]] = []
    for day in WINDOW_DAYS:
        day_entries, day_summary = _target_day_entries(
            source=source,
            source_root=root,
            day=day,
            set_item=set_days[day],
            expected_contract_sha=declared_contract_sha,
            overlay_item=overlay_by_day[day],
            raw_item=raw_by_day[day],
            aggregate_shard_item=descriptor_by_day[day],
        )
        entries.extend(day_entries)
        local_days.append(day_summary)
    # Every item is checked again in one source-side identity batch immediately
    # before it becomes a planned transfer object.  No raw ZIP path can enter
    # the plan because entries are only emitted above for shard/manifest/receipt
    # objects under the portable root.
    source_identities = source.identities([str(item["source_path"]) for item in entries])
    for item in entries:
        identity = source_identities.get(str(item["source_path"]))
        if identity is None or identity.sha256 != item["sha256"] or identity.size_bytes != item["size_bytes"]:
            raise TargetTransferError("planned source object SHA-256 or size drifted")
    source_paths = [str(item["source_path"]) for item in entries]
    destination_paths = [str(item["destination_relative"]) for item in entries]
    if len(set(source_paths)) != len(source_paths) or len(set(destination_paths)) != len(destination_paths):
        raise TargetTransferError("target transfer has duplicate source or destination objects")
    if any("raw" in str(item["role"]).lower() for item in entries):
        raise TargetTransferError("raw episode ZIP entered target-only transfer plan")
    assigned_entries, lanes = _assign_lpt_lanes(entries)
    destination = Path(destination_root).expanduser().resolve(strict=False)
    plan = {
        "schema": TRANSFER_PLAN_SCHEMA,
        "owner_goal_revision": 21,
        "parallel_lanes_exact": LANE_COUNT,
        "source": {
            "host": source.host,
            "read_only": True,
            "target_set_root": root,
            "target_set_manifest_relative": manifest_relative,
            "target_set_manifest_sha256": manifest_sha,
            "target_set_receipt_relative": receipt_relative,
            "target_set_receipt_sha256": receipt_sha,
            "goal_contract_sha256": declared_contract_sha,
            "source_contract_path": source_contract_path,
            "base_pack_completion_sha256": actual_base_sha,
            "complete_action_overlay_manifest_sha256": actual_overlay_sha,
            "raw_episode_zip_identities": [
                {
                    "utc_day": day,
                    "sha256": _sha256(raw_by_day[day].get("sha256"), label="raw ZIP SHA-256"),
                    "size_bytes": _nonnegative_int(raw_by_day[day].get("size_bytes"), label="raw ZIP size"),
                }
                for day in WINDOW_DAYS
            ],
        },
        "local_contract": {
            "path": str(local_contract),
            "sha256": local_contract_sha,
            "goal_revision": contract.get("goal_revision"),
            "embedded_critic_owner_goal_revision": 21,
        },
        "destination_root": str(destination),
        "bert_disk_free_floor_bytes": floor,
        "target_only": True,
        "raw_episode_zip_transfer_required": False,
        "raw_episode_zip_objects_transferred": False,
        "canonical_target_set_manifest_remains_byte_identical": True,
        "target_days": local_days,
        "entries": assigned_entries,
        "lanes": lanes,
        "total_size_bytes": sum(int(item["size_bytes"]) for item in assigned_entries),
    }
    plan_sha = sha256_bytes(canonical_bytes(plan))
    validate_target_transfer_plan(plan, expected_sha256=plan_sha)
    return plan, plan_sha


def validate_target_transfer_plan(
    plan: Mapping[str, Any], *, expected_sha256: str | None = None
) -> str:
    if plan.get("schema") != TRANSFER_PLAN_SCHEMA:
        raise TargetTransferError("target transfer plan schema drifted")
    if plan.get("owner_goal_revision") != 21 or plan.get("parallel_lanes_exact") != LANE_COUNT:
        raise TargetTransferError("target transfer plan authority/topology drifted")
    if plan.get("target_only") is not True or plan.get("raw_episode_zip_objects_transferred") is not False:
        raise TargetTransferError("target transfer plan information boundary drifted")
    entries = _rows(plan.get("entries"), label="target transfer plan entries")
    lanes = _rows(plan.get("lanes"), label="target transfer plan lanes")
    if len(lanes) != LANE_COUNT or len(entries) < LANE_COUNT:
        raise TargetTransferError("target transfer plan does not have exactly four lanes")
    source_paths: set[str] = set()
    destination_paths: set[str] = set()
    assigned: set[int] = set()
    total = 0
    for raw in entries:
        item = _mapping(raw, label="target transfer plan entry")
        source = _safe_source_root(item.get("source_path"), label="plan source path")
        destination = _safe_relative(item.get("destination_relative"), label="plan destination path")
        if source in source_paths or destination in destination_paths:
            raise TargetTransferError("target transfer plan has duplicate object paths")
        source_paths.add(source)
        destination_paths.add(destination)
        _sha256(item.get("sha256"), label="plan entry SHA-256")
        total += _nonnegative_int(item.get("size_bytes"), label="plan entry size")
        lane = item.get("lane_id")
        if lane not in range(LANE_COUNT):
            raise TargetTransferError("target transfer plan entry lane is malformed")
        assigned.add(int(lane))
        if item.get("role") not in {
            "target_set_manifest",
            "target_set_receipt",
            "target_set_binding",
            "target_day_manifest",
            "target_day_receipt",
            "target_day_schema",
            "target_shard",
        }:
            raise TargetTransferError("target transfer plan has a non-target object role")
    if assigned != set(range(LANE_COUNT)) or total != plan.get("total_size_bytes"):
        raise TargetTransferError("target transfer plan lane or byte totals drifted")
    seen_lane_ids: set[int] = set()
    for raw in lanes:
        lane = _mapping(raw, label="target transfer lane")
        lane_id = lane.get("lane_id")
        if lane_id not in range(LANE_COUNT) or lane_id in seen_lane_ids:
            raise TargetTransferError("target transfer lane identity drifted")
        seen_lane_ids.add(int(lane_id))
        source_list = _rows(lane.get("source_paths"), label="target transfer lane source paths")
        if len(source_list) != lane.get("entry_count") or not source_list:
            raise TargetTransferError("target transfer lane is empty or count drifted")
    digest = sha256_bytes(canonical_bytes(dict(plan)))
    if expected_sha256 is not None and digest != expected_sha256:
        raise TargetTransferError("target transfer plan SHA-256 mismatch")
    return digest


def _assert_directory_not_symlink(path: Path, *, create: bool) -> None:
    """Create a bounded local directory tree without ever traversing a link."""

    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            state = current.lstat()
        except FileNotFoundError:
            if not create:
                raise TargetTransferError(f"destination parent is absent: {current}")
            try:
                current.mkdir()
            except FileExistsError:
                # Four independent lanes can concurrently create their common
                # private parent.  Re-inspect rather than treating that benign
                # race as permission to traverse an unverified path.
                pass
            try:
                state = current.lstat()
            except OSError as exc:
                raise TargetTransferError(
                    f"destination directory appeared but cannot be verified: {current}"
                ) from exc
        except OSError as exc:
            raise TargetTransferError(f"cannot inspect destination directory: {current}") from exc
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise TargetTransferError(f"destination path component is not a real directory: {current}")


def _write_create_only_or_verify(path: Path, body: bytes) -> str:
    _assert_directory_not_symlink(path.parent, create=True)
    expected = sha256_bytes(body)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        actual = _regular_local_identity(path, label="existing create-only receipt")
        if actual.sha256 != expected or actual.size_bytes != len(body):
            raise TargetTransferError(f"existing create-only artifact conflicts: {path}")
        return expected
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return expected


def _entry_receipt_relative(entry: Mapping[str, Any], plan_sha256: str) -> str:
    key = sha256_bytes(
        canonical_bytes(
            {
                "plan_sha256": plan_sha256,
                "source_path": entry["source_path"],
                "destination_relative": entry["destination_relative"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
        )
    ).removeprefix("sha256:")
    return f"transfer/receipts/sha256-{key}.target-file-transfer-receipt.json"


def _part_path(root: Path, plan_sha256: str, entry: Mapping[str, Any]) -> Path:
    plan_key = plan_sha256.removeprefix("sha256:")
    return _destination_member(
        root,
        f"transfer/private-partials/{plan_key}/{entry['destination_relative']}.part",
        label="private partial path",
    )


def _final_state(root: Path, entry: Mapping[str, Any]) -> tuple[Path, FileIdentity | None]:
    final = _destination_member(root, entry["destination_relative"], label="final destination")
    if not final.exists() and not final.is_symlink():
        return final, None
    return final, _regular_local_identity(final, label="existing destination final")


def _verify_or_copy_entry(
    *,
    source: SourceReader,
    root: Path,
    entry: Mapping[str, Any],
    plan_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify one final object or append-copy its private verified partial."""

    expected_sha = _sha256(entry.get("sha256"), label="entry SHA-256")
    expected_size = _nonnegative_int(entry.get("size_bytes"), label="entry size")
    final, existing = _final_state(root, entry)
    if existing is not None:
        if existing.sha256 != expected_sha or existing.size_bytes != expected_size:
            raise TargetTransferError(f"existing final conflicts and will not be overwritten: {final}")
    else:
        part = _part_path(root, plan_sha256, entry)
        existing_bytes = 0
        if part.exists() or part.is_symlink():
            partial = _regular_local_identity(part, label="private transfer partial")
            existing_bytes = partial.size_bytes
            if existing_bytes > expected_size:
                raise TargetTransferError("private transfer partial is larger than its source object")
            local_prefix = sha256_file(part, limit=existing_bytes)
            source_prefix = source.prefix_sha256(str(entry["source_path"]), existing_bytes)
            if local_prefix != source_prefix:
                raise TargetTransferError("private transfer partial prefix does not match source")
        if existing_bytes < expected_size:
            _assert_directory_not_symlink(part.parent, create=True)
            source.append_to_part(str(entry["source_path"]), part)
        partial = _regular_local_identity(part, label="completed private transfer partial")
        if partial.sha256 != expected_sha or partial.size_bytes != expected_size:
            raise TargetTransferError("completed private transfer partial SHA-256 or size mismatch")
        _assert_directory_not_symlink(final.parent, create=True)
        try:
            os.link(part, final)
        except FileExistsError:
            raced = _regular_local_identity(final, label="raced destination final")
            if raced.sha256 != expected_sha or raced.size_bytes != expected_size:
                raise TargetTransferError("raced destination final conflicts and will not be overwritten")
        except OSError as exc:
            raise TargetTransferError("could not create-only promote verified target object") from exc
        final_identity = _regular_local_identity(final, label="promoted destination final")
        if final_identity.sha256 != expected_sha or final_identity.size_bytes != expected_size:
            raise TargetTransferError("promoted destination final SHA-256 or size mismatch")
    receipt = {
        "schema": FILE_RECEIPT_SCHEMA,
        "owner_goal_revision": 21,
        "plan_sha256": plan_sha256,
        "source": {
            "path": str(entry["source_path"]),
            "sha256": expected_sha,
            "size_bytes": expected_size,
        },
        "destination": {
            "relative_path": str(entry["destination_relative"]),
            "sha256": expected_sha,
            "size_bytes": expected_size,
            "regular_non_symlink": True,
        },
        "role": str(entry["role"]),
        "utc_day": entry.get("utc_day"),
        "split": entry.get("split"),
        "source_destination_identity_match": True,
        "raw_episode_zip_transferred": False,
        "create_only": True,
    }
    receipt_relative = _entry_receipt_relative(entry, plan_sha256)
    receipt_sha = _write_create_only_or_verify(
        _destination_member(root, receipt_relative, label="file receipt"), canonical_bytes(receipt)
    )
    return receipt, receipt_sha


def _remaining_copy_bytes(root: Path, plan: Mapping[str, Any], plan_sha256: str) -> int:
    remaining = 0
    for raw in _rows(plan.get("entries"), label="transfer plan entries"):
        entry = _mapping(raw, label="transfer plan entry")
        expected_size = _nonnegative_int(entry.get("size_bytes"), label="entry size")
        _final, final_identity = _final_state(root, entry)
        if final_identity is not None:
            expected_sha = _sha256(entry.get("sha256"), label="entry SHA-256")
            if final_identity.sha256 != expected_sha or final_identity.size_bytes != expected_size:
                raise TargetTransferError("existing final conflicts and will not be overwritten")
            continue
        part = _part_path(root, plan_sha256, entry)
        if part.exists() or part.is_symlink():
            partial = _regular_local_identity(part, label="private transfer partial")
            if partial.size_bytes > expected_size:
                raise TargetTransferError("private transfer partial is larger than source")
            remaining += expected_size - partial.size_bytes
        else:
            remaining += expected_size
    return remaining


def _verified_source_execution_preflight(source: SourceReader, plan: Mapping[str, Any]) -> None:
    entries = _rows(plan.get("entries"), label="target transfer plan entries")
    identities = source.identities([str(_mapping(item, label="plan entry")["source_path"]) for item in entries])
    for raw in entries:
        entry = _mapping(raw, label="plan entry")
        source_path = str(entry["source_path"])
        identity = identities.get(source_path)
        if identity is None or identity.sha256 != entry["sha256"] or identity.size_bytes != entry["size_bytes"]:
            raise TargetTransferError("source object drifted after target transfer planning")


def execute_target_transfer_plan(
    plan: Mapping[str, Any],
    *,
    source: SourceReader,
    plan_sha256: str | None = None,
    free_bytes: Callable[[Path], int] | None = None,
) -> dict[str, Any]:
    """Execute exactly four lanes after all capacity and source checks pass."""

    calculated_sha = validate_target_transfer_plan(plan, expected_sha256=plan_sha256)
    destination = Path(str(plan["destination_root"])).expanduser().resolve(strict=False)
    parent = destination.parent
    _assert_directory_not_symlink(parent, create=False)
    if destination.exists() or destination.is_symlink():
        state = destination.lstat()
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise TargetTransferError("destination root is not a real directory")
    remaining = _remaining_copy_bytes(destination, plan, calculated_sha)
    free = (free_bytes or (lambda path: shutil.disk_usage(path).free))(parent)
    floor = _nonnegative_int(plan.get("bert_disk_free_floor_bytes"), label="Bert disk floor")
    if int(free) < floor + remaining:
        raise TargetTransferError("Bert free space would fall below the required 20GiB floor")
    # No destination directory is created until the strict capacity reservation
    # and a fresh read-only source identity batch have both passed.
    _verified_source_execution_preflight(source, plan)
    _assert_directory_not_symlink(destination, create=True)
    plan_relative = (
        "transfer/plans/sha256-"
        + calculated_sha.removeprefix("sha256:")
        + ".target-transfer-plan.json"
    )
    _write_create_only_or_verify(
        _destination_member(destination, plan_relative, label="transfer plan"),
        canonical_bytes(dict(plan)),
    )
    lanes = _rows(plan.get("lanes"), label="target transfer plan lanes")
    entries_by_lane: dict[int, list[Mapping[str, Any]]] = {lane: [] for lane in range(LANE_COUNT)}
    for raw in _rows(plan.get("entries"), label="target transfer plan entries"):
        entry = _mapping(raw, label="target transfer plan entry")
        entries_by_lane[int(entry["lane_id"])].append(entry)

    def run_lane(lane_id: int) -> list[tuple[dict[str, Any], str]]:
        # One lane owns a disjoint sequence of source objects.  No nested pool
        # is used, so ``max_workers=4`` is the exact transfer topology.
        return [
            _verify_or_copy_entry(
                source=source,
                root=destination,
                entry=entry,
                plan_sha256=calculated_sha,
            )
            for entry in entries_by_lane[lane_id]
        ]

    lane_results: dict[int, list[tuple[dict[str, Any], str]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=LANE_COUNT) as pool:
        futures = {pool.submit(run_lane, lane_id): lane_id for lane_id in range(LANE_COUNT)}
        for future in concurrent.futures.as_completed(futures):
            lane_results[futures[future]] = future.result()
    if set(lane_results) != set(range(LANE_COUNT)) or len(lanes) != LANE_COUNT:
        raise TargetTransferError("target transfer did not complete exactly four lanes")
    receipt_by_destination: dict[str, dict[str, str]] = {}
    for lane_id in range(LANE_COUNT):
        for receipt, receipt_sha in lane_results[lane_id]:
            relative = str(_mapping(receipt.get("destination"), label="file receipt destination")["relative_path"])
            receipt_by_destination[relative] = {
                "receipt_relative": _entry_receipt_relative(
                    next(
                        item
                        for item in entries_by_lane[lane_id]
                        if item["destination_relative"] == relative
                    ),
                    calculated_sha,
                ),
                "receipt_sha256": receipt_sha,
            }
    source_binding = _mapping(plan.get("source"), label="target transfer source binding")
    target_view = {
        "schema": TARGET_VIEW_SCHEMA,
        "owner_goal_revision": 21,
        "status": "verified_target_only_offline_input",
        "target_set_root_relative": ".",
        "canonical_target_set_manifest": {
            "relative_path": source_binding["target_set_manifest_relative"],
            "sha256": source_binding["target_set_manifest_sha256"],
            "remains_byte_identical": True,
        },
        "canonical_target_set_receipt": {
            "relative_path": source_binding["target_set_receipt_relative"],
            "sha256": source_binding["target_set_receipt_sha256"],
        },
        "source_binding": {
            "goal_contract_sha256": source_binding["goal_contract_sha256"],
            "base_pack_completion_sha256": source_binding["base_pack_completion_sha256"],
            "complete_action_overlay_manifest_sha256": source_binding[
                "complete_action_overlay_manifest_sha256"
            ],
            "all_20_raw_episode_zip_sha256s": source_binding["raw_episode_zip_identities"],
        },
        "target_days": [dict(item) for item in _rows(plan.get("target_days"), label="target days")],
        "file_receipts": [
            {
                "destination_relative": str(entry["destination_relative"]),
                **receipt_by_destination[str(entry["destination_relative"])],
            }
            for entry in sorted(
                (_mapping(item, label="transfer entry") for item in _rows(plan.get("entries"), label="transfer entries")),
                key=lambda item: str(item["destination_relative"]),
            )
        ],
        "plan_relative": plan_relative,
        "plan_sha256": calculated_sha,
        "raw_episode_zip_transferred": False,
        "runtime_or_training_started": False,
        "create_only": True,
    }
    target_view_body = canonical_bytes(target_view)
    target_view_sha = sha256_bytes(target_view_body)
    target_view_relative = (
        "transfer/target-view/sha256-"
        + target_view_sha.removeprefix("sha256:")
        + ".target-view.json"
    )
    _write_create_only_or_verify(
        _destination_member(destination, target_view_relative, label="target view"), target_view_body
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "owner_goal_revision": 21,
        "status": "complete_verified_target_only_transfer",
        "plan_relative": plan_relative,
        "plan_sha256": calculated_sha,
        "target_view_relative": target_view_relative,
        "target_view_sha256": target_view_sha,
        "entry_count": len(_rows(plan.get("entries"), label="transfer entries")),
        "parallel_lanes_exact": LANE_COUNT,
        "source_destination_sha256_size_verified": True,
        "raw_episode_zip_transferred": False,
        "runtime_or_training_started": False,
        "create_only": True,
    }
    completion_body = canonical_bytes(completion)
    completion_sha = sha256_bytes(completion_body)
    completion_relative = (
        "transfer/completion/sha256-"
        + completion_sha.removeprefix("sha256:")
        + ".target-transfer-completion.json"
    )
    _write_create_only_or_verify(
        _destination_member(destination, completion_relative, label="target transfer completion"),
        completion_body,
    )
    return {
        "destination_root": str(destination),
        "plan_sha256": calculated_sha,
        "target_view_path": str(_destination_member(destination, target_view_relative, label="target view")),
        "target_view_sha256": target_view_sha,
        "completion_path": str(
            _destination_member(destination, completion_relative, label="target transfer completion")
        ),
        "completion_sha256": completion_sha,
        "entry_count": completion["entry_count"],
        "remaining_copy_bytes_reserved": remaining,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-host", default=SOURCE_HOST)
    parser.add_argument("--source-root", required=True, type=str)
    parser.add_argument("--target-set-manifest-relative", required=True)
    parser.add_argument("--target-set-receipt-relative", required=True)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--expected-contract-sha256", default="")
    parser.add_argument(
        "--expected-base-completion-sha256", default=DEFAULT_EXPECTED_BASE_COMPLETION_SHA256
    )
    parser.add_argument(
        "--expected-overlay-manifest-sha256", default=DEFAULT_EXPECTED_OVERLAY_MANIFEST_SHA256
    )
    parser.add_argument("--bert-disk-floor-bytes", type=int, default=DEFAULT_DISK_FLOOR_BYTES)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="copy only after dry-run-equivalent source, capacity, and identity checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        source = SSHSourceReader(args.source_host)
        plan, plan_sha = build_target_transfer_plan(
            source=source,
            source_root=args.source_root,
            destination_root=args.destination_root,
            target_set_manifest_relative=args.target_set_manifest_relative,
            target_set_receipt_relative=args.target_set_receipt_relative,
            local_contract_path=args.contract,
            expected_contract_sha256=args.expected_contract_sha256 or None,
            expected_base_completion_sha256=args.expected_base_completion_sha256,
            expected_overlay_manifest_sha256=args.expected_overlay_manifest_sha256,
            disk_floor_bytes=args.bert_disk_floor_bytes,
        )
        if not args.execute:
            print(
                json.dumps(
                    {
                        "phase": "dry_run",
                        "plan_sha256": plan_sha,
                        "entry_count": len(plan["entries"]),
                        "total_size_bytes": plan["total_size_bytes"],
                        "lanes": plan["lanes"],
                        "raw_episode_zip_transferred": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = execute_target_transfer_plan(plan, source=source, plan_sha256=plan_sha)
    except TargetTransferError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"phase": "complete", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
