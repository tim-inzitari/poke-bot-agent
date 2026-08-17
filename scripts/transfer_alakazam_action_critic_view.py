#!/usr/bin/env python3
"""Create-only four-stream transfer for the compact Alakazam critic view.

The action critic trains from the sealed 40-wide semantic tensor pack plus
the sealed complete-action overlay.  This tool deliberately transfers that
compact view rather than the much larger collision-census corpus.  It has two
modes:

* the default dry-run probes Elmo read-only, validates the canonical objects,
  and prints a deterministic four-lane plan; and
* ``--execute`` writes only private ``.part`` objects, verified final files,
  and create-only plan/receipt artifacts under the requested Bert root.

It does not start training, alter a trainer, rewrite the canonical overlay
manifest, or attach a sidecar to a runtime.  The source remains read-only.

OpenRsync on Bert lacks ``--append-verify``.  A resumed private partial is
therefore prefix-SHA-verified against Elmo before this tool ever invokes
``rsync --append``.  Every completed object is independently full-SHA and
size verified before a hard-link based create-only promotion.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol


TRANSFER_PLAN_SCHEMA = "poke_bot.alakazam_action_critic_training_view_transfer_plan/v1"
FILE_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_training_view_file_receipt/v1"
COMPLETION_SCHEMA = "poke_bot.alakazam_action_critic_training_view_completion/v1"
TRAINING_VIEW_SCHEMA = "poke_bot.alakazam_action_critic_training_view/v1"

BASE_COMPLETION_SCHEMA = "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"
BASE_PACK_SCHEMA = "poke_bot.alakazam_recent20_semantic_tensor_pack/v1"
OVERLAY_MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1"
OVERLAY_COMPLETION_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_completion/v1"

SOURCE_HOST = "elmo"
DEFAULT_BASE_ROOT = (
    "/srv/poke-bot-agent/archive/alakazam-rule-derivative/.incoming/"
    "r11-inzi-sealed-20260813T141600Z-d74152bc/home/pokebot/poke-bot-agent/"
    "outputs/bootstrap/alakazam-rule-derivative-r10-semantic-pack-all20-v3"
)
DEFAULT_OVERLAY_ROOT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-recent20-rtp-overlay-v1-attempt4"
)
DEFAULT_DESTINATION_ROOT = Path(
    "/Users/example/Documents/poke-agent-critic-bootstrap/"
    "recent20-training-view-r20"
)

EXPECTED_BASE_COMPLETION_SHA256 = (
    "sha256:e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba"
)
EXPECTED_BASE_SCHEMA_SHA256 = (
    "sha256:3a528138e819b10691e8a7ed917c55e4000b9ec039562cf859cc2e00706bb3fa"
)
EXPECTED_CORPUS_MANIFEST_SHA256 = (
    "sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1"
)
EXPECTED_OVERLAY_SCHEMA_SHA256 = (
    "sha256:29de1530768f1b3f8b9be7e02fe2dfef3eeb64475d1ba0ff146026d5c54d6a37"
)
EXPECTED_OVERLAY_MANIFEST_SHA256 = (
    "sha256:081e40d9b9cc98714abaa8945c8d176a9143bdb8e87aeeee0327878642b118bd"
)
EXPECTED_OVERLAY_COMPLETION_SHA256 = (
    "sha256:c7a9392a1c91adfa27730963d867ee88069c41585d3fd2027df96d2301edfd91"
)
EXPECTED_OVERLAY_VALIDATION_SHA256 = (
    "sha256:4b1611013154f27f4be7f097ba2cd692504f00a2c91c65313e8d3a2cb2bf069b"
)

EXPECTED_COMPACT_ENTRY_COUNT = 128
EXPECTED_COMPACT_TOTAL_BYTES = 8_692_555_652
LANE_COUNT = 4
DEFAULT_DISK_FLOOR_BYTES = 20 * 1024 * 1024 * 1024

WINDOW_DAYS = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 12)]
)
SPLIT_BY_DAY = {
    **{day: "train" for day in WINDOW_DAYS[:14]},
    **{day: "validation" for day in WINDOW_DAYS[14:17]},
    **{day: "evaluation" for day in WINDOW_DAYS[17:]},
}
BASE_FILE_NAMES = {
    "features_f32": "features.f32",
    "decision_offsets_u64": "decision_offsets.u64",
    "selected_option_u32": "selected_option.u32",
    "decision_key_sha256": "decision_keys.sha256",
}
SHA256_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
CONTENT_ADDRESS_RE = re.compile(r"^sha256-([0-9a-f]{64})")
SAFE_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class CriticViewTransferError(RuntimeError):
    """The compact critic view cannot safely be planned or transferred."""


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    """A checked regular, non-symlink object identity."""

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
    """Read-only source interface; implementations never mutate Elmo."""

    host: str

    def read_bytes(self, path: str) -> bytes: ...

    def identities(self, paths: Sequence[str]) -> dict[str, FileIdentity]: ...

    def prefix_sha256(self, path: str, length: int) -> str: ...


def canonical_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used by all receipts."""

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


def sha256_bytes(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def sha256_file(path: Path | str, *, limit: int | None = None) -> str:
    """Hash a whole regular file or exactly its leading ``limit`` bytes."""

    digest = hashlib.sha256()
    remaining = limit
    with Path(path).open("rb") as stream:
        while True:
            read_size = 8 * 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                read_size = min(read_size, remaining)
            block = stream.read(read_size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining is not None and remaining != 0:
        raise CriticViewTransferError("partial file ended before its declared prefix")
    return "sha256:" + digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticViewTransferError(message)


def _sha256(value: object, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if SHA256_RE.fullmatch(text) is None:
        raise CriticViewTransferError(f"{field} must be a lowercase SHA-256 identity")
    return text


def _positive_int(value: object, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CriticViewTransferError(f"{field} must be an exact integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise CriticViewTransferError(f"{field} must be positive")
    return int(value)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CriticViewTransferError(f"{field} must be an object")
    return value


def _rows(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CriticViewTransferError(f"{field} must be an array")
    return list(value)


def _safe_relative(value: object, *, field: str) -> str:
    text = str(value or "")
    candidate = PurePosixPath(text)
    if (
        not text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise CriticViewTransferError(f"{field} must be a safe relative path")
    return "/".join(candidate.parts)


def _absolute_source_root(value: Path | str, *, field: str) -> str:
    text = str(value)
    candidate = PurePosixPath(text)
    if not candidate.is_absolute() or ".." in candidate.parts or not SAFE_REMOTE_PATH_RE.fullmatch(text):
        raise CriticViewTransferError(f"{field} must be a safe absolute POSIX source path")
    return str(candidate)


def _source_join(root: str, relative: str) -> str:
    safe = _safe_relative(relative, field="source relative path")
    return str(PurePosixPath(root).joinpath(*PurePosixPath(safe).parts))


def _destination_path(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative, field="destination relative path")
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise CriticViewTransferError("destination path escapes destination root") from exc
    return candidate


def _identity_from_local_path(path: Path | str, *, label: str) -> FileIdentity:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise CriticViewTransferError(f"{label} is unavailable: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CriticViewTransferError(f"{label} must be a regular non-symlink file: {candidate}")
    digest = sha256_file(candidate)
    try:
        after = candidate.lstat()
    except OSError as exc:
        raise CriticViewTransferError(f"{label} disappeared while being hashed") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CriticViewTransferError(f"{label} changed while being hashed")
    return FileIdentity(str(candidate.resolve()), digest, int(after.st_size))


class LocalSourceReader:
    """Read a local fixture/source while applying the same regular-file rules."""

    host = "local"

    def read_bytes(self, path: str) -> bytes:
        identity = _identity_from_local_path(path, label="local source object")
        try:
            body = Path(identity.path).read_bytes()
        except OSError as exc:
            raise CriticViewTransferError(f"cannot read local source object: {path}") from exc
        if sha256_bytes(body) != identity.sha256 or len(body) != identity.size_bytes:
            raise CriticViewTransferError("local source object changed while being read")
        return body

    def identities(self, paths: Sequence[str]) -> dict[str, FileIdentity]:
        result: dict[str, FileIdentity] = {}
        for path in paths:
            if path in result:
                continue
            result[path] = _identity_from_local_path(path, label="local source object")
        return result

    def prefix_sha256(self, path: str, length: int) -> str:
        identity = _identity_from_local_path(path, label="local source object")
        if length < 0 or length > identity.size_bytes:
            raise CriticViewTransferError("invalid requested local source prefix length")
        return sha256_file(identity.path, limit=length)


_REMOTE_HELPER = r'''
import base64
import hashlib
import json
import os
import stat
import sys

def digest(path, limit=None):
    h = hashlib.sha256()
    remaining = limit
    with open(path, "rb") as f:
        while True:
            size = 8 * 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                size = min(size, remaining)
            block = f.read(size)
            if not block:
                break
            h.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining is not None and remaining != 0:
        raise ValueError("short prefix")
    return "sha256:" + h.hexdigest()

def regular(path):
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise ValueError("not a regular non-symlink file")
    return st

try:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    op = request["op"]
    if op == "read":
        path = request["path"]
        regular(path)
        with open(path, "rb") as stream:
            body = stream.read()
        response = {"body_b64": base64.b64encode(body).decode("ascii")}
    elif op == "identities":
        rows = []
        for path in request["paths"]:
            before = regular(path)
            value = digest(path)
            after = regular(path)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("source changed while hashing")
            rows.append({"path": path, "sha256": value, "size_bytes": after.st_size})
        response = {"identities": rows}
    elif op == "prefix":
        path = request["path"]
        length = request["length"]
        st = regular(path)
        if not isinstance(length, int) or isinstance(length, bool) or length < 0 or length > st.st_size:
            raise ValueError("invalid prefix length")
        response = {"sha256": digest(path, length)}
    else:
        raise ValueError("unknown operation")
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
except Exception as exc:
    sys.stderr.write("critic-view remote helper failed: %s\n" % (exc,))
    raise SystemExit(2)
'''


class SSHSourceReader:
    """Read only exact paths from the named source host through SSH."""

    def __init__(self, host: str) -> None:
        if not host or host == "local":
            raise ValueError("SSHSourceReader requires a non-local host")
        self.host = host
        encoded = base64.b64encode(_REMOTE_HELPER.encode("utf-8")).decode("ascii")
        self._remote_command = (
            "python3 -c \"import base64;exec(compile(base64.b64decode('"
            + encoded
            + "'),'critic_view_remote_helper','exec'))\""
        )

    def _run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        body = canonical_bytes(dict(request))
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
                input=body,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise CriticViewTransferError("could not launch read-only SSH probe") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise CriticViewTransferError(f"read-only source probe failed: {detail}")
        try:
            value = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CriticViewTransferError("source probe emitted invalid JSON") from exc
        return dict(_mapping(value, field="source probe response"))

    def read_bytes(self, path: str) -> bytes:
        response = self._run({"op": "read", "path": _absolute_source_root(path, field="source path")})
        encoded = response.get("body_b64")
        if not isinstance(encoded, str):
            raise CriticViewTransferError("source probe omitted object body")
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except ValueError as exc:
            raise CriticViewTransferError("source probe emitted invalid object encoding") from exc

    def identities(self, paths: Sequence[str]) -> dict[str, FileIdentity]:
        wanted = list(dict.fromkeys(_absolute_source_root(path, field="source path") for path in paths))
        response = self._run({"op": "identities", "paths": wanted})
        rows = _rows(response.get("identities"), field="source probe identities")
        if len(rows) != len(wanted):
            raise CriticViewTransferError("source probe identity count drifted")
        result: dict[str, FileIdentity] = {}
        for expected, raw in zip(wanted, rows, strict=True):
            row = _mapping(raw, field="source probe identity")
            path = str(row.get("path") or "")
            if path != expected:
                raise CriticViewTransferError("source probe identity path drifted")
            result[path] = FileIdentity(
                path=path,
                sha256=_sha256(row.get("sha256"), field="source object SHA-256"),
                size_bytes=_positive_int(row.get("size_bytes"), field="source object size"),
            )
        return result

    def prefix_sha256(self, path: str, length: int) -> str:
        response = self._run(
            {
                "op": "prefix",
                "path": _absolute_source_root(path, field="source path"),
                "length": _positive_int(length, field="source prefix length", allow_zero=True),
            }
        )
        return _sha256(response.get("sha256"), field="source prefix SHA-256")


def _read_json(source: SourceReader, path: str, *, label: str, expected_sha256: str | None = None) -> tuple[dict[str, Any], str, int, bytes]:
    body = source.read_bytes(path)
    digest = sha256_bytes(body)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CriticViewTransferError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} actual={digest}"
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CriticViewTransferError(f"{label} is not valid JSON") from exc
    return dict(_mapping(payload, field=label)), digest, len(body), body


def _content_addressed_digest(path: str, body: bytes, *, label: str) -> str:
    match = CONTENT_ADDRESS_RE.match(PurePosixPath(path).name)
    if match is None:
        raise CriticViewTransferError(f"{label} lacks a content-addressed file name")
    actual = sha256_bytes(body)
    if actual.removeprefix("sha256:") != match.group(1):
        raise CriticViewTransferError(f"{label} content-addressed file name disagrees with bytes")
    return actual


def _derive_day(pack: Mapping[str, Any]) -> str:
    candidates: list[str] = []
    for field in ("receipt_path", "source_path"):
        raw = str(pack.get(field) or "")
        for part in PurePosixPath(raw).parts:
            if DAY_RE.fullmatch(part):
                candidates.append(part)
    files = _mapping(pack.get("files"), field="base pack files")
    for value in files.values():
        item = _mapping(value, field="base file declaration")
        parent = PurePosixPath(str(item.get("path") or "")).parent.name
        if DAY_RE.fullmatch(parent):
            candidates.append(parent)
    if not candidates or len(set(candidates)) != 1:
        raise CriticViewTransferError("base completion does not identify one UTC day per pack")
    return candidates[0]


def _metadata_path(root: str, relative: str) -> str:
    return _source_join(root, relative)


def _metadata_name(digest: str, suffix: str) -> str:
    return f"sha256-{digest.removeprefix('sha256:')}{suffix}"


def _strict_overlay_receipt_paths(overlay_root: str) -> tuple[str, str]:
    return (
        _metadata_path(
            overlay_root,
            "receipts/"
            + _metadata_name(EXPECTED_OVERLAY_COMPLETION_SHA256, ".completion-receipt.json"),
        ),
        _metadata_path(
            overlay_root,
            "validation-receipts/"
            + _metadata_name(EXPECTED_OVERLAY_VALIDATION_SHA256, ".pipeline-loader-validation.json"),
        ),
    )


def _assert_named_identity(path: str, identity: FileIdentity, *, label: str, expected_sha256: str | None = None) -> None:
    if expected_sha256 is not None and identity.sha256 != expected_sha256:
        raise CriticViewTransferError(f"{label} SHA-256 drifted")
    match = CONTENT_ADDRESS_RE.match(PurePosixPath(path).name)
    if match is None or identity.sha256.removeprefix("sha256:") != match.group(1):
        raise CriticViewTransferError(f"{label} content-addressed path drifted")


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
    result: dict[str, Any] = {
        "source_path": _absolute_source_root(source_path, field="entry source path"),
        "destination_relative": _safe_relative(destination_relative, field="entry destination path"),
        "sha256": _sha256(sha256, field="entry SHA-256"),
        "size_bytes": _positive_int(size_bytes, field="entry size"),
        "role": str(role),
    }
    if utc_day is not None:
        if DAY_RE.fullmatch(utc_day) is None:
            raise CriticViewTransferError("entry UTC day is malformed")
        result["utc_day"] = utc_day
    if split is not None:
        if split not in {"train", "validation", "evaluation"}:
            raise CriticViewTransferError("entry split is malformed")
        result["split"] = split
    return result


def _assign_lpt_lanes(entries: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically assign every object with longest-processing-time bin packing."""

    if len(entries) < LANE_COUNT:
        raise CriticViewTransferError("four-stream transfer requires at least four distinct objects")
    sorted_entries = sorted(
        (dict(entry) for entry in entries),
        key=lambda entry: (-int(entry["size_bytes"]), str(entry["destination_relative"])),
    )
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(LANE_COUNT)]
    totals = [0] * LANE_COUNT
    for entry in sorted_entries:
        lane = min(range(LANE_COUNT), key=lambda index: (totals[index], index))
        entry["lane_id"] = lane
        lanes[lane].append(entry)
        totals[lane] += int(entry["size_bytes"])
    output_entries = [
        entry
        for lane in lanes
        for entry in sorted(
            lane,
            key=lambda item: (-int(item["size_bytes"]), str(item["destination_relative"])),
        )
    ]
    lane_rows = [
        {
            "lane_id": lane,
            "entry_count": len(lanes[lane]),
            "total_size_bytes": totals[lane],
            "destination_relatives": [
                str(entry["destination_relative"])
                for entry in sorted(
                    lanes[lane],
                    key=lambda item: (-int(item["size_bytes"]), str(item["destination_relative"])),
                )
            ],
        }
        for lane in range(LANE_COUNT)
    ]
    return output_entries, lane_rows


def _find_generic_single_file(source: SourceReader, root: str, relative_directory: str, suffix: str) -> str:
    """Generic test-fixture helper; production strict mode never enumerates remotely."""

    if not isinstance(source, LocalSourceReader):
        raise CriticViewTransferError("noncanonical source requires explicit receipt paths")
    directory = Path(_source_join(root, relative_directory))
    matches = sorted(path for path in directory.glob(f"*{suffix}") if path.is_file() and not path.is_symlink())
    if len(matches) != 1:
        raise CriticViewTransferError(f"expected exactly one generic {suffix} artifact")
    return str(matches[0].resolve())


def build_transfer_plan(
    *,
    source: SourceReader,
    base_root: Path | str,
    overlay_root: Path | str,
    destination_root: Path | str,
    disk_floor_bytes: int = DEFAULT_DISK_FLOOR_BYTES,
    strict: bool = True,
    overlay_manifest_path: Path | str | None = None,
    overlay_completion_receipt_path: Path | str | None = None,
    overlay_validation_receipt_path: Path | str | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate Elmo source objects and return a deterministic exact four-lane plan.

    ``strict=True`` is the production path and pins every authoritative
    recent-20 identity.  The non-strict mode exists solely for hermetic unit
    fixtures; it still validates all manifest-declared object identities.
    """

    base_root_text = _absolute_source_root(base_root, field="base root")
    overlay_root_text = _absolute_source_root(overlay_root, field="overlay root")
    floor = _positive_int(disk_floor_bytes, field="Bert disk floor")
    destination_text = str(Path(destination_root).expanduser().resolve(strict=False))

    if overlay_manifest_path is None:
        if not strict:
            raise CriticViewTransferError("noncanonical plan requires an explicit overlay manifest path")
        manifest_path = _metadata_path(
            overlay_root_text,
            "manifests/"
            + _metadata_name(EXPECTED_OVERLAY_MANIFEST_SHA256, ".overlay-manifest.json"),
        )
    else:
        manifest_path = _absolute_source_root(overlay_manifest_path, field="overlay manifest path")
    manifest, manifest_sha, _manifest_size, manifest_body = _read_json(
        source,
        manifest_path,
        label="complete-action overlay manifest",
        expected_sha256=EXPECTED_OVERLAY_MANIFEST_SHA256 if strict else None,
    )
    _content_addressed_digest(manifest_path, manifest_body, label="complete-action overlay manifest")
    if manifest.get("schema") != OVERLAY_MANIFEST_SCHEMA:
        raise CriticViewTransferError("complete-action overlay manifest schema drifted")

    base_completion_path = _source_join(base_root_text, "COMPLETE.json")
    base_completion, base_sha, _base_size, base_body = _read_json(
        source,
        base_completion_path,
        label="base pack completion",
        expected_sha256=EXPECTED_BASE_COMPLETION_SHA256 if strict else None,
    )
    if base_completion.get("schema") != BASE_COMPLETION_SCHEMA:
        raise CriticViewTransferError("base completion schema drifted")

    base_pack = _mapping(manifest.get("base_pack"), field="overlay base pack")
    declared_base_sha = _sha256(base_pack.get("completion_sha256"), field="overlay base completion SHA-256")
    if declared_base_sha != base_sha:
        raise CriticViewTransferError("overlay does not bind the copied base completion")
    if strict and base_sha != EXPECTED_BASE_COMPLETION_SHA256:
        raise CriticViewTransferError("base completion identity drifted")
    if base_pack.get("feature_tensors_copied_into_overlay") is not False:
        raise CriticViewTransferError("overlay incorrectly claims to contain feature tensors")

    declared_corpus_sha = _sha256(base_pack.get("corpus_manifest_sha256"), field="overlay corpus SHA-256")
    declared_base_schema_sha = _sha256(base_pack.get("schema_sha256"), field="overlay base schema SHA-256")
    declared_overlay_schema_sha = _sha256(manifest.get("overlay_schema_sha256"), field="overlay schema SHA-256")
    if strict and (
        declared_corpus_sha != EXPECTED_CORPUS_MANIFEST_SHA256
        or declared_base_schema_sha != EXPECTED_BASE_SCHEMA_SHA256
        or declared_overlay_schema_sha != EXPECTED_OVERLAY_SCHEMA_SHA256
    ):
        raise CriticViewTransferError("overlay canonical metadata identity drifted")

    completion_binding_path = _metadata_path(
        overlay_root_text,
        "bindings/" + _metadata_name(declared_base_sha, ".base-completion.json"),
    )
    corpus_binding_path = _metadata_path(
        overlay_root_text,
        "bindings/" + _metadata_name(declared_corpus_sha, ".corpus-manifest.json"),
    )
    base_schema_path = _metadata_path(
        overlay_root_text,
        "schemas/" + _metadata_name(declared_base_schema_sha, ".base-schema.json"),
    )
    overlay_schema_path = _metadata_path(
        overlay_root_text,
        "schemas/" + _metadata_name(declared_overlay_schema_sha, ".overlay-schema.json"),
    )
    _require(
        PurePosixPath(str(base_pack.get("completion_path") or "")).name
        == PurePosixPath(completion_binding_path).name,
        "overlay completion binding path drifted",
    )
    _require(
        PurePosixPath(str(base_pack.get("corpus_manifest_path") or "")).name
        == PurePosixPath(corpus_binding_path).name,
        "overlay corpus binding path drifted",
    )
    _require(
        PurePosixPath(str(base_pack.get("schema_path") or "")).name
        == PurePosixPath(base_schema_path).name,
        "overlay base schema path drifted",
    )

    binding_completion, binding_sha, _size, binding_body = _read_json(
        source,
        completion_binding_path,
        label="overlay base completion binding",
        expected_sha256=base_sha,
    )
    if binding_body != base_body or binding_completion != base_completion:
        raise CriticViewTransferError("base completion binding is not byte-identical")
    corpus_binding, corpus_sha, _size, corpus_body = _read_json(
        source,
        corpus_binding_path,
        label="overlay corpus binding",
        expected_sha256=declared_corpus_sha,
    )
    base_schema, base_schema_sha, _size, base_schema_body = _read_json(
        source,
        base_schema_path,
        label="base schema",
        expected_sha256=declared_base_schema_sha,
    )
    overlay_schema, overlay_schema_sha, _size, overlay_schema_body = _read_json(
        source,
        overlay_schema_path,
        label="overlay schema",
        expected_sha256=declared_overlay_schema_sha,
    )
    for path, body, label in (
        (completion_binding_path, binding_body, "overlay base completion binding"),
        (corpus_binding_path, corpus_body, "overlay corpus binding"),
        (base_schema_path, base_schema_body, "base schema"),
        (overlay_schema_path, overlay_schema_body, "overlay schema"),
    ):
        _content_addressed_digest(path, body, label=label)
    if corpus_binding.get("schema") is None or base_schema.get("schema") != BASE_PACK_SCHEMA:
        raise CriticViewTransferError("base corpus/schema binding is malformed")
    if overlay_schema.get("schema") is None:
        raise CriticViewTransferError("overlay schema binding is malformed")
    schema = _mapping(base_pack.get("schema"), field="overlay base schema descriptor")
    if (
        schema.get("schema") != BASE_PACK_SCHEMA
        or schema.get("feature_width") != 40
        or schema.get("feature_dtype") != "float32_le"
    ):
        raise CriticViewTransferError("overlay base feature ABI drifted")

    if overlay_completion_receipt_path is None or overlay_validation_receipt_path is None:
        if strict:
            strict_completion, strict_validation = _strict_overlay_receipt_paths(overlay_root_text)
            overlay_completion_receipt_path = overlay_completion_receipt_path or strict_completion
            overlay_validation_receipt_path = overlay_validation_receipt_path or strict_validation
        else:
            overlay_completion_receipt_path = overlay_completion_receipt_path or _find_generic_single_file(
                source, overlay_root_text, "receipts", ".completion-receipt.json"
            )
            overlay_validation_receipt_path = overlay_validation_receipt_path or _find_generic_single_file(
                source, overlay_root_text, "validation-receipts", ".pipeline-loader-validation.json"
            )
    overlay_completion_path = _absolute_source_root(
        overlay_completion_receipt_path, field="overlay completion receipt path"
    )
    overlay_validation_path = _absolute_source_root(
        overlay_validation_receipt_path, field="overlay validation receipt path"
    )
    overlay_completion, overlay_completion_sha, _size, overlay_completion_body = _read_json(
        source,
        overlay_completion_path,
        label="overlay completion receipt",
        expected_sha256=EXPECTED_OVERLAY_COMPLETION_SHA256 if strict else None,
    )
    overlay_validation, overlay_validation_sha, _size, overlay_validation_body = _read_json(
        source,
        overlay_validation_path,
        label="overlay pipeline validation receipt",
        expected_sha256=EXPECTED_OVERLAY_VALIDATION_SHA256 if strict else None,
    )
    _content_addressed_digest(overlay_completion_path, overlay_completion_body, label="overlay completion receipt")
    _content_addressed_digest(overlay_validation_path, overlay_validation_body, label="overlay pipeline validation receipt")
    if overlay_completion.get("schema") != OVERLAY_COMPLETION_SCHEMA:
        raise CriticViewTransferError("overlay completion receipt schema drifted")
    if (
        overlay_completion.get("manifest_sha256") != manifest_sha
        or overlay_completion.get("base_pack_completion_sha256") != base_sha
    ):
        raise CriticViewTransferError("overlay completion receipt does not bind source identities")
    if overlay_validation.get("manifest_sha256") not in {None, manifest_sha}:
        raise CriticViewTransferError("overlay validation receipt manifest binding drifted")

    packs = _rows(base_completion.get("packs"), field="base completion packs")
    if not packs:
        raise CriticViewTransferError("base completion has no packs")
    seen_days: set[str] = set()
    seen_sources: set[str] = set()
    entries: list[dict[str, Any]] = [
        _entry(
            source_path=base_completion_path,
            destination_relative="base/COMPLETE.json",
            sha256=base_sha,
            size_bytes=len(base_body),
            role="base_pack_completion",
        )
    ]
    for raw_pack in packs:
        pack = _mapping(raw_pack, field="base completion pack")
        day = _derive_day(pack)
        if day in seen_days:
            raise CriticViewTransferError("base completion repeats a day")
        seen_days.add(day)
        source_sha = _sha256(pack.get("source_sha256"), field="base pack source SHA-256")
        if source_sha in seen_sources:
            raise CriticViewTransferError("base completion repeats a source shard")
        seen_sources.add(source_sha)
        files = _mapping(pack.get("files"), field="base pack files")
        if set(files) != set(BASE_FILE_NAMES):
            raise CriticViewTransferError("base pack file role inventory drifted")
        for role, name in BASE_FILE_NAMES.items():
            declaration = _mapping(files.get(role), field=f"base pack {day} {role}")
            declared_name = PurePosixPath(str(declaration.get("path") or "")).name
            if declared_name != name:
                raise CriticViewTransferError("base completion file name drifted")
            entries.append(
                _entry(
                    source_path=_source_join(base_root_text, f"{day}/{name}"),
                    destination_relative=f"base/{day}/{name}",
                    sha256=_sha256(declaration.get("sha256"), field=f"base {day} {role} SHA-256"),
                    size_bytes=_positive_int(declaration.get("size_bytes"), field=f"base {day} {role} size"),
                    role=f"base_{role}",
                    utc_day=day,
                    split=SPLIT_BY_DAY.get(day),
                )
            )
        entries.append(
            _entry(
                source_path=_source_join(base_root_text, f"{day}/receipt.json"),
                destination_relative=f"base/{day}/receipt.json",
                sha256=_sha256(pack.get("receipt_sha256"), field=f"base {day} receipt SHA-256"),
                size_bytes=1,  # Replaced with a measured, verified source identity below.
                role="base_day_receipt",
                utc_day=day,
                split=SPLIT_BY_DAY.get(day),
            )
        )
    if strict and (tuple(sorted(seen_days)) != WINDOW_DAYS or len(packs) != len(WINDOW_DAYS)):
        raise CriticViewTransferError("base completion does not cover the sealed recent-20 day window")

    shards = _rows(manifest.get("overlay_shards"), field="overlay shards")
    seen_overlay_days: set[str] = set()
    seen_overlay_paths: set[str] = set()
    for raw_shard in shards:
        shard = _mapping(raw_shard, field="overlay shard")
        day = str(shard.get("utc_day") or "")
        if DAY_RE.fullmatch(day) is None or day in seen_overlay_days:
            raise CriticViewTransferError("overlay shard day inventory is malformed")
        seen_overlay_days.add(day)
        split = str(shard.get("split") or "")
        if split not in {"train", "validation", "evaluation"}:
            raise CriticViewTransferError("overlay shard split is malformed")
        if strict and SPLIT_BY_DAY.get(day) != split:
            raise CriticViewTransferError("overlay shard split drifted from sealed day split")
        source_sha = _sha256(shard.get("base_source_shard_sha256"), field="overlay base source SHA-256")
        if source_sha not in seen_sources:
            raise CriticViewTransferError("overlay references a base source shard absent from completion")
        relative = _safe_relative(shard.get("path"), field="overlay shard path")
        if not relative.startswith("objects/") or relative in seen_overlay_paths:
            raise CriticViewTransferError("overlay shard object path inventory is malformed")
        seen_overlay_paths.add(relative)
        entries.append(
            _entry(
                source_path=_source_join(overlay_root_text, relative),
                destination_relative=f"overlay/{relative}",
                sha256=_sha256(shard.get("sha256"), field="overlay shard SHA-256"),
                size_bytes=_positive_int(shard.get("size_bytes"), field="overlay shard size"),
                role="complete_action_overlay_shard",
                utc_day=day,
                split=split,
            )
        )
    source_days = [str(value) for value in _rows(manifest.get("source_days"), field="overlay source days")]
    if len(source_days) != len(set(source_days)) or set(source_days) != seen_overlay_days:
        raise CriticViewTransferError("overlay source-day declaration disagrees with shards")
    if strict and (tuple(source_days) != WINDOW_DAYS or seen_overlay_days != set(WINDOW_DAYS)):
        raise CriticViewTransferError("overlay does not cover the sealed recent-20 window")

    metadata_rows = [
        (manifest_path, "overlay/manifests/" + PurePosixPath(manifest_path).name, manifest_sha, len(manifest_body), "overlay_manifest"),
        (completion_binding_path, "overlay/bindings/" + PurePosixPath(completion_binding_path).name, binding_sha, len(binding_body), "overlay_base_completion_binding"),
        (corpus_binding_path, "overlay/bindings/" + PurePosixPath(corpus_binding_path).name, corpus_sha, len(corpus_body), "overlay_corpus_binding"),
        (base_schema_path, "overlay/schemas/" + PurePosixPath(base_schema_path).name, base_schema_sha, len(base_schema_body), "base_schema"),
        (overlay_schema_path, "overlay/schemas/" + PurePosixPath(overlay_schema_path).name, overlay_schema_sha, len(overlay_schema_body), "overlay_schema"),
        (overlay_completion_path, "overlay/receipts/" + PurePosixPath(overlay_completion_path).name, overlay_completion_sha, len(overlay_completion_body), "overlay_completion_receipt"),
        (overlay_validation_path, "overlay/validation-receipts/" + PurePosixPath(overlay_validation_path).name, overlay_validation_sha, len(overlay_validation_body), "overlay_validation_receipt"),
    ]
    for source_path, destination_relative, digest, size, role in metadata_rows:
        entries.append(
            _entry(
                source_path=source_path,
                destination_relative=destination_relative,
                sha256=digest,
                size_bytes=size,
                role=role,
            )
        )

    destination_paths = [str(entry["destination_relative"]) for entry in entries]
    source_paths = [str(entry["source_path"]) for entry in entries]
    if len(destination_paths) != len(set(destination_paths)):
        raise CriticViewTransferError("transfer plan repeats a destination object")
    if len(source_paths) != len(set(source_paths)):
        raise CriticViewTransferError("transfer plan repeats a source object")

    # Rehash all large source objects in one source-reader pass before assigning
    # lanes.  In particular, this fills the deliberately unknown day-receipt
    # byte lengths without trusting a completion field that does not carry it.
    actual = source.identities(source_paths)
    normalized_entries: list[dict[str, Any]] = []
    for entry in entries:
        identity = actual.get(str(entry["source_path"]))
        if identity is None:
            raise CriticViewTransferError("source identity probe omitted a planned object")
        expected_sha = str(entry["sha256"])
        if identity.sha256 != expected_sha:
            raise CriticViewTransferError(
                f"source object SHA-256 mismatch for {entry['destination_relative']}"
            )
        expected_size = int(entry["size_bytes"])
        if entry["role"] == "base_day_receipt":
            entry = dict(entry)
            entry["size_bytes"] = identity.size_bytes
        elif identity.size_bytes != expected_size:
            raise CriticViewTransferError(
                f"source object size mismatch for {entry['destination_relative']}"
            )
        normalized_entries.append(dict(entry))
    entries, lanes = _assign_lpt_lanes(normalized_entries)
    total_bytes = sum(int(entry["size_bytes"]) for entry in entries)
    if strict and (
        len(entries) != EXPECTED_COMPACT_ENTRY_COUNT
        or total_bytes != EXPECTED_COMPACT_TOTAL_BYTES
        or len(shards) != len(WINDOW_DAYS)
    ):
        raise CriticViewTransferError("canonical compact critic-view inventory drifted")

    plan = {
        "schema": TRANSFER_PLAN_SCHEMA,
        "source": {
            "host": source.host,
            "base_root": base_root_text,
            "overlay_root": overlay_root_text,
            "base_completion_sha256": base_sha,
            "base_schema_sha256": declared_base_schema_sha,
            "corpus_manifest_sha256": declared_corpus_sha,
            "overlay_schema_sha256": declared_overlay_schema_sha,
            "overlay_manifest_sha256": manifest_sha,
            "overlay_completion_receipt_sha256": overlay_completion_sha,
            "overlay_validation_receipt_sha256": overlay_validation_sha,
            "read_only": True,
        },
        "destination_root": destination_text,
        "parallel_lanes_exact": LANE_COUNT,
        "unique_source_object_per_lane": True,
        "disk_free_floor_bytes": floor,
        "resumable_private_partials_only": True,
        "final_conflict_behavior": "fail_closed_no_overwrite",
        "entries": entries,
        "lanes": lanes,
        "entry_count": len(entries),
        "total_size_bytes": total_bytes,
    }
    return plan, sha256_bytes(canonical_bytes(plan))


def validate_transfer_plan(plan: Mapping[str, Any], *, expected_sha256: str | None = None) -> str:
    """Validate plan shape, four lanes, LPT assignment, and object uniqueness."""

    body = canonical_bytes(dict(plan))
    digest = sha256_bytes(body)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CriticViewTransferError("transfer plan SHA-256 drifted")
    if plan.get("schema") != TRANSFER_PLAN_SCHEMA:
        raise CriticViewTransferError("foreign transfer plan")
    if plan.get("parallel_lanes_exact") != LANE_COUNT:
        raise CriticViewTransferError("transfer plan does not require exactly four lanes")
    if plan.get("unique_source_object_per_lane") is not True:
        raise CriticViewTransferError("transfer plan does not require unique source objects")
    _positive_int(plan.get("disk_free_floor_bytes"), field="plan disk floor")
    entries = _rows(plan.get("entries"), field="transfer plan entries")
    lanes = _rows(plan.get("lanes"), field="transfer plan lanes")
    if len(lanes) != LANE_COUNT or len(entries) < LANE_COUNT:
        raise CriticViewTransferError("transfer plan lane inventory is malformed")
    source_paths: set[str] = set()
    destination_paths: set[str] = set()
    expected_lane_entries: dict[int, list[dict[str, Any]]] = {lane: [] for lane in range(LANE_COUNT)}
    for raw_entry in entries:
        entry = dict(_mapping(raw_entry, field="transfer plan entry"))
        source_path = _absolute_source_root(entry.get("source_path"), field="planned source path")
        destination_relative = _safe_relative(entry.get("destination_relative"), field="planned destination path")
        _sha256(entry.get("sha256"), field="planned object SHA-256")
        _positive_int(entry.get("size_bytes"), field="planned object size")
        lane = _positive_int(entry.get("lane_id"), field="planned lane", allow_zero=True)
        if lane not in expected_lane_entries:
            raise CriticViewTransferError("planned lane is outside the exact four-lane range")
        if source_path in source_paths or destination_relative in destination_paths:
            raise CriticViewTransferError("transfer plan duplicates a source or destination object")
        source_paths.add(source_path)
        destination_paths.add(destination_relative)
        expected_lane_entries[lane].append(entry)
    for index, raw_lane in enumerate(lanes):
        lane = _mapping(raw_lane, field="transfer plan lane")
        if lane.get("lane_id") != index:
            raise CriticViewTransferError("transfer plan lanes are not canonical")
        expected = sorted(
            expected_lane_entries[index],
            key=lambda entry: (-int(entry["size_bytes"]), str(entry["destination_relative"])),
        )
        expected_paths = [str(entry["destination_relative"]) for entry in expected]
        if lane.get("destination_relatives") != expected_paths:
            raise CriticViewTransferError("transfer plan lane ordering is not deterministic LPT")
        if lane.get("entry_count") != len(expected) or lane.get("total_size_bytes") != sum(
            int(entry["size_bytes"]) for entry in expected
        ):
            raise CriticViewTransferError("transfer plan lane totals drifted")
    # A syntactically well-formed set of lane rows is insufficient: an edited
    # plan could move a large object to another lane and still preserve all
    # simple per-lane sums.  Re-run the deterministic LPT allocator and demand
    # exact lane placement for every sealed source object.
    unassigned = []
    for entry in entries:
        row = dict(_mapping(entry, field="transfer plan entry"))
        row.pop("lane_id", None)
        unassigned.append(row)
    expected_entries, expected_lanes = _assign_lpt_lanes(unassigned)
    observed_assignment = {
        str(_mapping(entry, field="transfer plan entry")["destination_relative"]): int(
            _mapping(entry, field="transfer plan entry")["lane_id"]
        )
        for entry in entries
    }
    expected_assignment = {
        str(entry["destination_relative"]): int(entry["lane_id"])
        for entry in expected_entries
    }
    if observed_assignment != expected_assignment or [dict(row) for row in lanes] != expected_lanes:
        raise CriticViewTransferError("transfer plan is not the deterministic four-lane LPT allocation")
    if plan.get("entry_count") != len(entries) or plan.get("total_size_bytes") != sum(
        int(_mapping(entry, field="entry")["size_bytes"]) for entry in entries
    ):
        raise CriticViewTransferError("transfer plan total inventory drifted")
    return digest


def _write_create_only(path: Path, body: bytes, *, label: str, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as exc:
        raise CriticViewTransferError(f"create-only {label} already exists: {path}") from exc
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_bytes(body)


def _write_create_only_or_verify(path: Path, body: bytes, *, label: str) -> str:
    if path.exists() or path.is_symlink():
        identity = _identity_from_local_path(path, label=label)
        if identity.sha256 != sha256_bytes(body) or identity.size_bytes != len(body):
            raise CriticViewTransferError(f"existing create-only {label} conflicts: {path}")
        return identity.sha256
    return _write_create_only(path, body, label=label)


def _validated_destination_target(destination_root: Path) -> Path:
    target = Path(destination_root).expanduser()
    if target == Path("/") or target == target.home() or not target.name:
        raise CriticViewTransferError("destination root is too broad")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CriticViewTransferError("destination parent must be an existing non-symlink directory")
    return target


def _ensure_destination_root(destination_root: Path) -> Path:
    target = _validated_destination_target(destination_root)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise CriticViewTransferError("destination root must be a non-symlink directory")
    else:
        target.mkdir(mode=0o755)
    return target.resolve()


@contextlib.contextmanager
def _execution_lock(destination_root: Path, plan_sha256: str) -> Iterable[None]:
    lock_path = destination_root / "transfer" / "locks" / f"sha256-{plan_sha256.removeprefix('sha256:')}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CriticViewTransferError("another controller owns this transfer plan") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _final_identity_if_exact(final_path: Path, entry: Mapping[str, Any]) -> bool:
    if not final_path.exists() and not final_path.is_symlink():
        return False
    identity = _identity_from_local_path(final_path, label="existing final destination")
    if identity.sha256 != entry["sha256"] or identity.size_bytes != int(entry["size_bytes"]):
        raise CriticViewTransferError(f"existing final object conflicts: {final_path}")
    return True


def _part_path(destination_root: Path, plan_sha256: str, entry: Mapping[str, Any]) -> Path:
    safe = _safe_relative(entry["destination_relative"], field="staging destination path")
    return destination_root / "transfer" / "staging" / plan_sha256.removeprefix("sha256:") / (
        Path(*PurePosixPath(safe).parts).as_posix() + ".part"
    )


def _assert_part_is_resumable(part: Path, entry: Mapping[str, Any]) -> int:
    if not part.exists() and not part.is_symlink():
        return 0
    try:
        info = part.lstat()
    except OSError as exc:
        raise CriticViewTransferError("private partial cannot be statted") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CriticViewTransferError("private partial must be a regular non-symlink file")
    size = int(info.st_size)
    if size > int(entry["size_bytes"]):
        raise CriticViewTransferError("private partial exceeds its sealed object size")
    return size


def _append_local_source(source_path: str, part: Path, offset: int) -> None:
    source = _identity_from_local_path(source_path, label="local transfer source")
    if offset > source.size_bytes:
        raise CriticViewTransferError("local private partial exceeds source object")
    part.parent.mkdir(parents=True, exist_ok=True)
    with Path(source.path).open("rb") as origin, part.open("ab") as destination:
        origin.seek(offset)
        while True:
            block = origin.read(8 * 1024 * 1024)
            if not block:
                break
            destination.write(block)
        destination.flush()
        os.fsync(destination.fileno())


def _rsync_into_part(source: SourceReader, source_path: str, part: Path, *, resume: bool) -> None:
    rsync = shutil.which("rsync")
    if rsync is None:
        raise CriticViewTransferError("rsync is required for a remote source transfer")
    part.parent.mkdir(parents=True, exist_ok=True)
    command = [rsync, "-az", "--compress-level=1", "--partial"]
    if resume:
        command.append("--append")
    if isinstance(source, SSHSourceReader):
        command.extend(
            [
                "--rsync-path",
                "/usr/bin/nice -n 10 /usr/bin/rsync",
                f"{source.host}:{_absolute_source_root(source_path, field='rsync source path')}",
                str(part),
            ]
        )
    else:
        command.extend([source_path, str(part)])
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise CriticViewTransferError("could not start rsync") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CriticViewTransferError(f"rsync failed for private partial: {detail}")


def _copy_into_part(source: SourceReader, entry: Mapping[str, Any], part: Path, offset: int) -> None:
    if offset:
        local_prefix = sha256_file(part, limit=offset)
        remote_prefix = source.prefix_sha256(str(entry["source_path"]), offset)
        if local_prefix != remote_prefix:
            raise CriticViewTransferError("private partial prefix does not match the immutable source")
    if isinstance(source, LocalSourceReader):
        _append_local_source(str(entry["source_path"]), part, offset)
    else:
        _rsync_into_part(source, str(entry["source_path"]), part, resume=bool(offset))


def _promote_part_create_only(part: Path, final_path: Path, entry: Mapping[str, Any]) -> None:
    if _final_identity_if_exact(final_path, entry):
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _identity_from_local_path(part, label="verified private partial")
    try:
        os.link(part, final_path, follow_symlinks=False)
    except FileExistsError:
        if _final_identity_if_exact(final_path, entry):
            return
        raise CriticViewTransferError("create-only promotion encountered a conflicting final")
    except OSError as exc:
        raise CriticViewTransferError("create-only hard-link promotion failed") from exc
    final_identity = _identity_from_local_path(final_path, label="promoted final object")
    if final_identity.sha256 != entry["sha256"] or final_identity.size_bytes != int(entry["size_bytes"]):
        raise CriticViewTransferError("promoted final object identity drifted")
    os.chmod(final_path, 0o444)
    try:
        part.unlink()
    except OSError as exc:
        # Both names now point at the exact immutable final bytes.  Refusing to
        # clean a private duplicate is safer than broadening to a delete retry.
        raise CriticViewTransferError("promoted final exists but private partial could not be unlinked") from exc


def _receipt_for_entry(
    plan_sha256: str,
    entry: Mapping[str, Any],
    *,
    source_host: str,
) -> tuple[dict[str, Any], str]:
    receipt = {
        "schema": FILE_RECEIPT_SCHEMA,
        "transfer_plan_sha256": plan_sha256,
        "source": {
            "host": source_host,
            "path": str(entry["source_path"]),
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        },
        "destination_relative": str(entry["destination_relative"]),
        "destination_sha256": str(entry["sha256"]),
        "destination_size_bytes": int(entry["size_bytes"]),
        "lane_id": int(entry["lane_id"]),
        "role": str(entry["role"]),
        "source_and_destination_sha256_size_verified": True,
        "promotion": "same_filesystem_hard_link_create_only",
        "private_partial_only_before_promotion": True,
    }
    return receipt, sha256_bytes(canonical_bytes(receipt))


class _CapacityLedger:
    """Conservative floor guard shared by exactly four worker lanes."""

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        destination_root: Path,
        floor_bytes: int,
        external_reserved_bytes: int,
        free_bytes: Callable[[Path], int],
    ) -> None:
        self._entries = {str(entry["destination_relative"]): dict(entry) for entry in entries}
        self._verified: set[str] = set()
        self.destination_root = destination_root
        self.floor_bytes = floor_bytes
        self.external_reserved_bytes = external_reserved_bytes
        self.free_bytes = free_bytes
        self._lock = threading.Lock()

    def _required_unverified(self) -> int:
        return sum(
            int(entry["size_bytes"])
            for relative, entry in self._entries.items()
            if relative not in self._verified
        )

    def initial_check(self) -> int:
        with self._lock:
            required = self._required_unverified()
            available = self.free_bytes(self.destination_root)
            if available < self.floor_bytes + self.external_reserved_bytes + required:
                raise CriticViewTransferError(
                    "Bert free space cannot reserve the complete compact view above the required floor"
                )
            return required

    def before_object(self) -> None:
        with self._lock:
            required = self._required_unverified()
            available = self.free_bytes(self.destination_root)
            if available < self.floor_bytes + self.external_reserved_bytes + required:
                raise CriticViewTransferError(
                    "Bert free space fell below the reserved floor before a new object transfer"
                )

    def mark_verified(self, entry: Mapping[str, Any]) -> None:
        with self._lock:
            self._verified.add(str(entry["destination_relative"]))


def _reservation_payload(
    *, plan_sha256: str,
    floor_bytes: int,
    required_bytes: int,
    external_reserved_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": "poke_bot.alakazam_action_critic_training_view_disk_reservation/v1",
        "transfer_plan_sha256": plan_sha256,
        "floor_bytes": floor_bytes,
        "compact_view_reserved_bytes": required_bytes,
        "external_reserved_bytes": external_reserved_bytes,
        "reservation_is_advisory_for_this_controller_only": True,
        "release_requires_completion_receipt": True,
    }


def _write_plan_and_reservation(
    *,
    destination_root: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
    required_bytes: int,
    external_reserved_bytes: int,
) -> None:
    plan_path = destination_root / "transfer" / "plans" / f"sha256-{plan_sha256.removeprefix('sha256:')}.json"
    _write_create_only_or_verify(plan_path, canonical_bytes(dict(plan)), label="transfer plan")
    reservation = _reservation_payload(
        plan_sha256=plan_sha256,
        floor_bytes=int(plan["disk_free_floor_bytes"]),
        required_bytes=required_bytes,
        external_reserved_bytes=external_reserved_bytes,
    )
    reservation_sha = sha256_bytes(canonical_bytes(reservation))
    reservation_path = destination_root / "transfer" / "reservations" / f"sha256-{reservation_sha.removeprefix('sha256:')}.json"
    _write_create_only_or_verify(
        reservation_path,
        canonical_bytes(reservation),
        label="disk reservation",
    )


def _verify_source_plan_identities(source: SourceReader, entries: Sequence[Mapping[str, Any]]) -> None:
    identities = source.identities([str(entry["source_path"]) for entry in entries])
    for entry in entries:
        source_path = str(entry["source_path"])
        identity = identities.get(source_path)
        if identity is None or identity.sha256 != entry["sha256"] or identity.size_bytes != int(entry["size_bytes"]):
            raise CriticViewTransferError("immutable source object changed after the transfer plan was sealed")


def _transfer_one(
    *,
    source: SourceReader,
    destination_root: Path,
    plan_sha256: str,
    entry: Mapping[str, Any],
    ledger: _CapacityLedger,
) -> tuple[str, str]:
    final_path = _destination_path(destination_root, str(entry["destination_relative"]))
    if not _final_identity_if_exact(final_path, entry):
        ledger.before_object()
        part = _part_path(destination_root, plan_sha256, entry)
        offset = _assert_part_is_resumable(part, entry)
        if offset == int(entry["size_bytes"]):
            part_identity = _identity_from_local_path(part, label="complete private partial")
            if part_identity.sha256 != entry["sha256"]:
                raise CriticViewTransferError("complete private partial conflicts with immutable source")
        else:
            _copy_into_part(source, entry, part, offset)
            part_identity = _identity_from_local_path(part, label="completed private partial")
            if part_identity.sha256 != entry["sha256"] or part_identity.size_bytes != int(entry["size_bytes"]):
                raise CriticViewTransferError("private partial did not finish at its sealed SHA-256 and size")
        _promote_part_create_only(part, final_path, entry)
    if not _final_identity_if_exact(final_path, entry):
        raise CriticViewTransferError("final object is not exact after transfer")
    ledger.mark_verified(entry)
    receipt, receipt_sha = _receipt_for_entry(
        plan_sha256,
        entry,
        source_host=source.host,
    )
    receipt_path = destination_root / "transfer" / "receipts" / f"sha256-{receipt_sha.removeprefix('sha256:')}.transfer-receipt.json"
    _write_create_only_or_verify(receipt_path, canonical_bytes(receipt), label="per-object transfer receipt")
    return str(entry["destination_relative"]), receipt_sha


def _training_view_payload(plan: Mapping[str, Any], plan_sha256: str) -> dict[str, Any]:
    source = _mapping(plan.get("source"), field="plan source")
    return {
        "schema": TRAINING_VIEW_SCHEMA,
        "transfer_plan_sha256": plan_sha256,
        "canonical_overlay_manifest": {
            "relative_path": "overlay/manifests/"
            + _metadata_name(str(source["overlay_manifest_sha256"]), ".overlay-manifest.json"),
            "sha256": source["overlay_manifest_sha256"],
        },
        "canonical_base_completion": {
            "relative_path": "base/COMPLETE.json",
            "sha256": source["base_completion_sha256"],
        },
        "base_pack_root_relative": "base",
        "overlay_root_relative": "overlay",
        "canonical_manifest_remains_byte_identical": True,
        "base_completion_path_override_required": True,
        "runtime_or_training_activation_authority": False,
    }


def execute_transfer_plan(
    plan: Mapping[str, Any],
    *,
    source: SourceReader,
    destination_root: Path | str | None = None,
    plan_sha256: str | None = None,
    external_reserved_bytes: int = 0,
    free_bytes: Callable[[Path], int] = _disk_free_bytes,
) -> dict[str, Any]:
    """Execute exactly four transfer lanes after all fail-closed preflights."""

    digest = validate_transfer_plan(plan, expected_sha256=plan_sha256)
    requested_destination = _validated_destination_target(
        Path(destination_root) if destination_root is not None else Path(str(plan["destination_root"]))
    )
    if str(requested_destination.resolve(strict=False)) != str(plan["destination_root"]):
        raise CriticViewTransferError("execute destination does not match the sealed transfer plan")
    external = _positive_int(
        external_reserved_bytes,
        field="external reserved bytes",
        allow_zero=True,
    )
    entries = [dict(_mapping(entry, field="transfer entry")) for entry in _rows(plan["entries"], field="transfer entries")]
    _verify_source_plan_identities(source, entries)
    # This intentionally happens before creating even the destination root:
    # below-floor capacity is a no-write failure, not an empty output tree.
    full_reservation = sum(int(entry["size_bytes"]) for entry in entries)
    if free_bytes(requested_destination.parent) < int(plan["disk_free_floor_bytes"]) + external + full_reservation:
        raise CriticViewTransferError(
            "Bert free space cannot reserve the complete compact view above the required floor"
        )
    destination = _ensure_destination_root(requested_destination)
    with _execution_lock(destination, digest):
        ledger = _CapacityLedger(
            entries,
            destination_root=destination,
            floor_bytes=int(plan["disk_free_floor_bytes"]),
            external_reserved_bytes=external,
            free_bytes=free_bytes,
        )
        required_bytes = ledger.initial_check()
        _write_plan_and_reservation(
            destination_root=destination,
            plan=plan,
            plan_sha256=digest,
            required_bytes=required_bytes,
            external_reserved_bytes=external,
        )
        by_lane: dict[int, list[dict[str, Any]]] = {lane: [] for lane in range(LANE_COUNT)}
        for entry in entries:
            by_lane[int(entry["lane_id"])].append(entry)

        def run_lane(lane: int) -> list[tuple[str, str]]:
            outcomes: list[tuple[str, str]] = []
            ordered = sorted(
                by_lane[lane],
                key=lambda item: (-int(item["size_bytes"]), str(item["destination_relative"])),
            )
            for item in ordered:
                outcomes.append(
                    _transfer_one(
                        source=source,
                        destination_root=destination,
                        plan_sha256=digest,
                        entry=item,
                        ledger=ledger,
                    )
                )
            return outcomes

        results: list[tuple[str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=LANE_COUNT) as executor:
            futures = [executor.submit(run_lane, lane) for lane in range(LANE_COUNT)]
            for future in futures:
                results.extend(future.result())

        receipt_by_destination = dict(results)
        if set(receipt_by_destination) != {str(entry["destination_relative"]) for entry in entries}:
            raise CriticViewTransferError("not every source object has a per-object receipt")
        for entry in entries:
            final_path = _destination_path(destination, str(entry["destination_relative"]))
            if not _final_identity_if_exact(final_path, entry):
                raise CriticViewTransferError("final parity check failed before completion receipt")

        training_view = _training_view_payload(plan, digest)
        training_view_sha = sha256_bytes(canonical_bytes(training_view))
        training_view_path = destination / "transfer" / "training-view" / f"sha256-{training_view_sha.removeprefix('sha256:')}.json"
        _write_create_only_or_verify(training_view_path, canonical_bytes(training_view), label="local training-view pointer")
        completion = {
            "schema": COMPLETION_SCHEMA,
            "transfer_plan_sha256": digest,
            "source": dict(_mapping(plan["source"], field="plan source")),
            "destination_root": str(destination),
            "entry_count": len(entries),
            "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
            "ordered_per_object_transfer_receipt_sha256s": [
                receipt_by_destination[str(entry["destination_relative"])]
                for entry in sorted(entries, key=lambda item: str(item["destination_relative"]))
            ],
            "training_view_path": str(training_view_path.relative_to(destination)),
            "training_view_sha256": training_view_sha,
            "all_source_destination_sha256_size_verified": True,
            "private_partials_not_training_eligible": True,
            "canonical_overlay_manifest_rewritten": False,
            "runtime_or_trainer_activation_performed": False,
        }
        completion_sha = sha256_bytes(canonical_bytes(completion))
        completion_path = destination / "transfer" / "completion" / f"sha256-{completion_sha.removeprefix('sha256:')}.json"
        _write_create_only_or_verify(completion_path, canonical_bytes(completion), label="transfer completion receipt")
    return {
        "plan_sha256": digest,
        "completion_path": str(completion_path),
        "completion_sha256": completion_sha,
        "training_view_path": str(training_view_path),
        "entry_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
    }


def _source_reader(host: str) -> SourceReader:
    return LocalSourceReader() if host == "local" else SSHSourceReader(host)


def _argument_nonnegative(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-host", default=SOURCE_HOST, help="Elmo SSH host, or local for hermetic use")
    parser.add_argument("--base-root", default=DEFAULT_BASE_ROOT)
    parser.add_argument("--overlay-root", default=DEFAULT_OVERLAY_ROOT)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    parser.add_argument("--disk-floor-bytes", type=_argument_nonnegative, default=DEFAULT_DISK_FLOOR_BYTES)
    parser.add_argument("--external-reserved-bytes", type=_argument_nonnegative, default=0)
    parser.add_argument("--execute", action="store_true", help="perform the create-only transfer; default is read-only dry-run")
    parser.add_argument("--overlay-manifest-path", type=Path)
    parser.add_argument("--overlay-completion-receipt-path", type=Path)
    parser.add_argument("--overlay-validation-receipt-path", type=Path)
    parser.add_argument(
        "--allow-noncanonical-fixture",
        action="store_true",
        help="disable fixed production identities; intended only for hermetic tests",
    )
    args = parser.parse_args(argv)
    source = _source_reader(args.source_host)
    plan, plan_sha = build_transfer_plan(
        source=source,
        base_root=args.base_root,
        overlay_root=args.overlay_root,
        destination_root=args.destination_root,
        disk_floor_bytes=args.disk_floor_bytes,
        strict=not args.allow_noncanonical_fixture,
        overlay_manifest_path=args.overlay_manifest_path,
        overlay_completion_receipt_path=args.overlay_completion_receipt_path,
        overlay_validation_receipt_path=args.overlay_validation_receipt_path,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry_run_read_only",
                    "transfer_plan_sha256": plan_sha,
                    "plan": plan,
                },
                sort_keys=True,
            )
        )
        return 0
    result = execute_transfer_plan(
        plan,
        source=source,
        plan_sha256=plan_sha,
        external_reserved_bytes=args.external_reserved_bytes,
    )
    print(json.dumps({"mode": "executed", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
