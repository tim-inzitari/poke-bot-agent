#!/usr/bin/env python3
"""Fail-closed, checksum-first transfer of the sealed r241 Alakazam corpus.

This tool is intentionally inert unless ``--execute`` is supplied.  When an
operator runs it on Inzi it reads the one sealed Elmo r241 exact-20 output,
copies only the Alakazam feature corpus plus the exact archive receipt, and
publishes a new top-level protected pointer.  It never edits the Elmo source,
never starts a service, and never replaces an existing destination identity.

The source pointer remains byte-identical as
``SOURCE_PROTECTED_EXPERT_CORPUS.json``.  The new top-level pointer adds the
host-local r241 archive binding required by the launcher and checkpoint
receipt generator without claiming that the source pointer itself changed.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import dataclasses
from datetime import date, timedelta
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Iterator, Mapping, Sequence


R241_CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
TRANSFER_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_exact20_transfer/v1"
ARCHIVE_SCHEMA = "poke_bot.expert_latest20_receipt/v1"
FINAL_READY_SCHEMA = "poke_bot.latest20_specialist_corpora/v1"
POINTER_SCHEMA = "poke_bot.pinned_expert_corpus/v1"

SOURCE_HOST = "elmo"
INZI_HOST = "inzi"
REMOTE_INZI_FINALIZE_FLAG = "--remote-inzi-finalize"
REMOTE_ELMO_METADATA_HANDOFF_FLAG = "--remote-elmo-metadata-handoff"
REMOTE_ELMO_METADATA_PLAN_FLAG = "--remote-elmo-metadata-plan"
SOURCE_WINDOW_ROOT = (
    "/mnt/Main/main/poke-bot-agent/archive/expert-r241-derived/windows/"
    "2026-07-22_2026-08-10/roster18-v6-strategic"
)
SOURCE_ARCHIVE_RECEIPT = (
    "/mnt/Main/main/poke-bot-agent/archive/"
    "expert-r241-20260722-20260810/current.json"
)
DESTINATION_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "alakazam_new_list_direct_policy_r241/runtime/expert"
)
DESTINATION_ARCHIVE_RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "expert-r241-20260722-20260810-current.json"
)
ELMO_METADATA_HANDOFF_ROOT = Path(
    "/mnt/Main/main/poke-bot-agent/outputs/pure_rl/"
    "alakazam_new_list_direct_policy_r241/runtime/expert"
)
INZI_TRANSFER_RECEIPT_PATH = (
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "alakazam_new_list_direct_policy_r241/runtime/expert/"
    "R241_EXACT20_CORPUS_TRANSFER_READY.json"
)
# This is the sealed Inzi receipt emitted by the authorized exact20 transfer.
# The Elmo projection is derived from, and preserves a byte-identical copy of,
# this receipt; it does not re-transfer or project any feature payload bytes.
INZI_TRANSFER_RECEIPT_SHA256 = (
    "sha256:b4b9cfdfcca444a8e09ee83db7836c2384075d8d5c4186c642c9423d72f888a3"
)
INZI_TRANSFER_RECEIPT_SIZE_BYTES = 9_890
ELMO_METADATA_HANDOFF_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_metadata_handoff/v1"
)
ELMO_METADATA_STREAM_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_metadata_stream/v1"
)
ELMO_METADATA_HANDOFF_ENV = "R241_EXACT20_ELMO_METADATA_HANDOFF"
ELMO_METADATA_INZI_RECEIPT_ENV = "R241_EXACT20_INZI_TRANSFER_RECEIPT_B64"

# TrueNAS sudo rejects a large environment assignment passed through ``env``.
# Keep the command line fixed and send the exact script/receipt envelope over
# stdin to this tiny root-side bootstrap instead.  The bootstrap sets the two
# process-local guards only after it validates the fixed stream shape; no
# receipt bytes appear in the SSH or sudo argument vector.
_ELMO_METADATA_REMOTE_BOOTSTRAP = r"""
import base64
import hashlib
import json
import os
import sys

_STREAM_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_elmo_metadata_stream/v1"
_FLAGS = {"--remote-elmo-metadata-handoff", "--remote-elmo-metadata-plan"}
if len(sys.argv) != 2 or sys.argv[1] not in _FLAGS:
    raise SystemExit("r241 Elmo metadata bootstrap received an invalid mode")
try:
    _payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    if set(_payload) != {
        "schema",
        "script_b64",
        "script_sha256",
        "inzi_transfer_receipt_b64",
        "inzi_transfer_receipt_sha256",
    } or _payload["schema"] != _STREAM_SCHEMA:
        raise ValueError("invalid stream envelope")
    _script = base64.b64decode(_payload["script_b64"].encode("ascii"), validate=True)
    _receipt_b64 = _payload["inzi_transfer_receipt_b64"]
    _receipt = base64.b64decode(_receipt_b64.encode("ascii"), validate=True)
    if _payload["script_sha256"] != "sha256:" + hashlib.sha256(_script).hexdigest():
        raise ValueError("script digest mismatch")
    if _payload["inzi_transfer_receipt_sha256"] != "sha256:" + hashlib.sha256(_receipt).hexdigest():
        raise ValueError("receipt digest mismatch")
except (KeyError, TypeError, UnicodeDecodeError, ValueError) as _exc:
    raise SystemExit("r241 Elmo metadata bootstrap rejected its stream") from _exc
os.environ["R241_EXACT20_ELMO_METADATA_HANDOFF"] = "1"
os.environ["R241_EXACT20_INZI_TRANSFER_RECEIPT_B64"] = _receipt_b64
_flag = sys.argv[1]
sys.argv = ["transfer_r241_exact20_alakazam_corpus.py", _flag]
exec(compile(_script, "transfer_r241_exact20_alakazam_corpus.py", "exec"), {"__name__": "__main__", "__file__": "transfer_r241_exact20_alakazam_corpus.py"})
""".strip()

SOURCE_READY_NAME = "LATEST20_SPECIALIST_CORPORA_READY.json"
SOURCE_POINTER_NAME = "SOURCE_PROTECTED_EXPERT_CORPUS.json"
SOURCE_READY_COPY_NAME = "SOURCE_LATEST20_SPECIALIST_CORPORA_READY.json"
ARCHIVE_COPY_NAME = "EXACT20_ARCHIVE_RECEIPT.json"
TRANSFER_RECEIPT_NAME = "R241_EXACT20_CORPUS_TRANSFER_READY.json"
INZI_TRANSFER_RECEIPT_COPY_NAME = "INZI_R241_EXACT20_CORPUS_TRANSFER_READY.json"
TOP_LEVEL_POINTER_NAME = "PROTECTED_EXPERT_CORPUS.json"
MANIFEST_NAME = "manifest.json"

WINDOW_START = "2026-07-22"
WINDOW_END = "2026-08-10"
WINDOW_DAYS = 20
WINDOW_TOTAL_EPISODES = 91_253


def _window_dates() -> tuple[str, ...]:
    start = date.fromisoformat(WINDOW_START)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(WINDOW_DAYS))


EXPECTED_DATES = _window_dates()


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    """A non-symlink regular-file identity."""

    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self, *, relative_to: Path | None = None) -> dict[str, object]:
        path = self.path
        if relative_to is not None:
            path = path.relative_to(relative_to)
        return {
            "path": str(path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclasses.dataclass(frozen=True)
class Exact20Contract:
    """The immutable Elmo identities that r241 is allowed to import."""

    ready_sha256: str
    ready_size_bytes: int
    source_pointer_sha256: str
    source_pointer_size_bytes: int
    manifest_sha256: str
    manifest_size_bytes: int
    archive_sha256: str
    archive_size_bytes: int
    shard_bytes: int
    records: int
    decisions: int
    dates: tuple[str, ...] = EXPECTED_DATES


DEFAULT_CONTRACT = Exact20Contract(
    ready_sha256="sha256:7da3523ede3b065e1335ded4630b810f4cbec3857fe78a7b55f53bf1e3ff8d37",
    ready_size_bytes=8_891,
    source_pointer_sha256="sha256:bfb2f77cc17ba29b450bc9f81e7cca223b035feb64034ba89e6e9985573bacde",
    source_pointer_size_bytes=2_132,
    manifest_sha256="sha256:d23e38ba14e004fbaa74921eea94ce63f96a7ec953342eda69582f7ebcbbccd6",
    manifest_size_bytes=118_026,
    archive_sha256="sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16",
    archive_size_bytes=15_298,
    shard_bytes=5_471_162_566,
    records=26_704,
    decisions=2_040_911,
)


@dataclasses.dataclass(frozen=True)
class CorpusIdentity:
    """The fully rehashed, source-preserving corpus identity."""

    ready: FileIdentity
    source_pointer: FileIdentity
    manifest: FileIdentity
    archive: FileIdentity
    shards: tuple[FileIdentity, ...]
    sidecars: tuple[FileIdentity, ...]
    records: int
    decisions: int
    shard_bytes: int


@dataclasses.dataclass(frozen=True)
class ElmoMetadataHandoff:
    """The small create-only Elmo projection of the sealed Inzi handoff.

    This object intentionally contains no feature-shard paths.  The Elmo
    projection carries only the small receipts required by the shared launch
    validator, while the checksum-exact Inzi receipt retains the complete
    twenty-shard inventory and is copied byte-for-byte as provenance.
    """

    root: Path
    pointer: FileIdentity
    transfer_receipt: FileIdentity
    inzi_transfer_receipt: FileIdentity
    archive: FileIdentity


class R241Exact20TransferError(RuntimeError):
    """The exact20 corpus cannot safely be transferred or finalized."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise R241Exact20TransferError(message)


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R241Exact20TransferError(
            f"{label} must be a regular non-symlink file: {candidate}"
        )
    return candidate.resolve()


def file_identity(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> FileIdentity:
    source = _regular(path, label=label)
    identity = FileIdentity(
        path=source,
        sha256=sha256_file(source),
        size_bytes=int(source.stat().st_size),
    )
    if expected_sha256 is not None and identity.sha256 != expected_sha256:
        raise R241Exact20TransferError(
            f"{label} checksum mismatch: expected={expected_sha256} actual={identity.sha256}"
        )
    if expected_size_bytes is not None and identity.size_bytes != expected_size_bytes:
        raise R241Exact20TransferError(
            f"{label} size mismatch: expected={expected_size_bytes} actual={identity.size_bytes}"
        )
    return identity


def _read_json(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241Exact20TransferError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise R241Exact20TransferError(f"{label} must contain a JSON object")
    return source, payload


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241Exact20TransferError(f"{label} must be an exact integer")
    return int(value)


def _safe_member(root: Path, relative: object, *, label: str) -> Path:
    text = str(relative or "")
    candidate = PurePosixPath(text)
    if (
        not text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise R241Exact20TransferError(f"{label} is not a safe relative path")
    path = root.joinpath(*candidate.parts)
    # Each r241 manifest member is a direct child today, but keeping this
    # containment test makes a malicious future nested member fail closed too.
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise R241Exact20TransferError(f"{label} escapes the corpus root") from exc
    return path


def _same_identity(left: FileIdentity, right: Mapping[str, object], *, label: str) -> None:
    _require(str(right.get("sha256") or "") == left.sha256, f"{label} checksum drifted")
    _require(
        _exact_int(right.get("size_bytes"), label=f"{label} size") == left.size_bytes,
        f"{label} size drifted",
    )


def _same_content(left: FileIdentity, right: FileIdentity) -> bool:
    """Compare content identity while allowing an intentional local path copy."""

    return left.sha256 == right.sha256 and left.size_bytes == right.size_bytes


def _validate_archive(
    archive_path: Path,
    *,
    contract: Exact20Contract,
) -> tuple[FileIdentity, dict[str, Any]]:
    identity = file_identity(
        archive_path,
        label="r241 exact20 archive receipt",
        expected_sha256=contract.archive_sha256,
        expected_size_bytes=contract.archive_size_bytes,
    )
    _, archive = _read_json(identity.path, label="r241 exact20 archive receipt")
    dates = [str(row.get("date") or "") for row in archive.get("archives") or ()]
    _require(
        archive.get("schema") == ARCHIVE_SCHEMA
        and archive.get("status") == "ready"
        and archive.get("window_policy") == "exact_20_consecutive_calendar_days"
        and archive.get("window_start") == WINDOW_START
        and archive.get("window_end") == WINDOW_END
        and _exact_int(archive.get("days"), label="archive days") == WINDOW_DAYS
        and archive.get("all_dates_represented") is True
        and dates == list(contract.dates)
        and _exact_int(archive.get("total_episodes"), label="archive episodes")
        == WINDOW_TOTAL_EPISODES,
        "r241 exact20 archive receipt is not the sealed Jul22-Aug10 window",
    )
    _require(
        all(
            isinstance(row, Mapping)
            and row.get("validated") is True
            and _is_sha256(row.get("sha256"))
            for row in archive.get("archives") or ()
        ),
        "r241 exact20 archive receipt contains an invalid archive row",
    )
    return identity, archive


def _validate_metadata(
    root: Path,
    *,
    archive_path: Path,
    contract: Exact20Contract,
) -> tuple[
    FileIdentity,
    FileIdentity,
    FileIdentity,
    FileIdentity,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate the four small immutable roots before accepting any shard."""

    ready = file_identity(
        root / SOURCE_READY_COPY_NAME,
        label="r241 source final READY receipt",
        expected_sha256=contract.ready_sha256,
        expected_size_bytes=contract.ready_size_bytes,
    )
    source_pointer = file_identity(
        root / SOURCE_POINTER_NAME,
        label="r241 source protected pointer",
        expected_sha256=contract.source_pointer_sha256,
        expected_size_bytes=contract.source_pointer_size_bytes,
    )
    manifest = file_identity(
        root / MANIFEST_NAME,
        label="r241 source Alakazam manifest",
        expected_sha256=contract.manifest_sha256,
        expected_size_bytes=contract.manifest_size_bytes,
    )
    archive, archive_payload = _validate_archive(archive_path, contract=contract)
    _, ready_payload = _read_json(ready.path, label="r241 source final READY receipt")
    _, pointer_payload = _read_json(source_pointer.path, label="r241 source protected pointer")
    _, manifest_payload = _read_json(manifest.path, label="r241 source Alakazam manifest")

    result_rows = [
        dict(row)
        for row in ready_payload.get("results") or ()
        if isinstance(row, Mapping) and row.get("archetype") == "alakazam"
    ]
    _require(
        ready_payload.get("schema") == FINAL_READY_SCHEMA
        and ready_payload.get("status") == "ready"
        and ready_payload.get("archive_receipt_sha256") == contract.archive_sha256
        and list(ready_payload.get("dates") or ()) == list(contract.dates)
        and len(result_rows) == 1
        and result_rows[0].get("status") == "ready"
        and result_rows[0].get("protected_corpus")
        == "alakazam/PROTECTED_EXPERT_CORPUS.json"
        and result_rows[0].get("manifest_sha256") == contract.manifest_sha256
        and _exact_int(result_rows[0].get("records"), label="READY Alakazam records")
        == contract.records
        and _exact_int(result_rows[0].get("decisions"), label="READY Alakazam decisions")
        == contract.decisions,
        "r241 source final READY receipt no longer binds the sealed Alakazam corpus",
    )

    selection = dict(pointer_payload.get("selection") or {})
    totals = dict(pointer_payload.get("totals") or {})
    _require(
        pointer_payload.get("schema") == POINTER_SCHEMA
        and pointer_payload.get("protected") is True
        and pointer_payload.get("manifest") == MANIFEST_NAME
        and pointer_payload.get("manifest_sha256") == contract.manifest_sha256
        and selection
        == {
            "field": "GameSequence.archetype",
            "operator": "exact_casefold",
            "opponent_routes_only": False,
            "seat_semantics": "acting_seat_only",
            "value": "alakazam",
        }
        and _exact_int(totals.get("bytes"), label="source pointer shard bytes")
        == contract.shard_bytes
        and _exact_int(totals.get("records_kept"), label="source pointer records")
        == contract.records
        and _exact_int(totals.get("decisions_kept"), label="source pointer decisions")
        == contract.decisions,
        "r241 source protected pointer identity is not the sealed Alakazam corpus",
    )

    manifest_totals = dict(manifest_payload.get("totals") or {})
    _require(
        manifest_payload.get("format") == "pokebot-bootstrap-feature-manifest"
        and manifest_payload.get("format_version") == 1
        and manifest_payload.get("compact_mode") == "temporal-expert-v1"
        and manifest_payload.get("date_start") == WINDOW_START
        and manifest_payload.get("date_end") == WINDOW_END
        and list(manifest_payload.get("dates") or ()) == list(contract.dates)
        and dict(manifest_payload.get("selection") or {}) == selection
        and _exact_int(manifest_totals.get("bytes"), label="manifest shard bytes")
        == contract.shard_bytes
        and _exact_int(manifest_totals.get("records_kept"), label="manifest records")
        == contract.records
        and _exact_int(manifest_totals.get("decisions_kept"), label="manifest decisions")
        == contract.decisions,
        "r241 source manifest does not preserve the sealed Alakazam identity",
    )
    return (
        ready,
        source_pointer,
        manifest,
        archive,
        ready_payload,
        pointer_payload,
        manifest_payload,
    )


def validate_staged_source(
    root: Path | str,
    *,
    archive_path: Path | str | None = None,
    contract: Exact20Contract = DEFAULT_CONTRACT,
) -> CorpusIdentity:
    """Rehash and semantically validate a staged source or landed destination.

    ``root`` must contain the raw source READY receipt, raw source pointer,
    manifest, each feature shard and its sidecar.  No input is repaired or
    rewritten; one invalid byte raises before a transfer is published.
    """

    source_root = Path(root).expanduser().resolve()
    _require(source_root.is_dir() and not source_root.is_symlink(), "staged r241 root is unsafe")
    archive_file = (
        Path(archive_path).expanduser().resolve()
        if archive_path is not None
        else source_root / ARCHIVE_COPY_NAME
    )
    (
        ready,
        source_pointer,
        manifest,
        archive,
        _ready_payload,
        _pointer_payload,
        manifest_payload,
    ) = _validate_metadata(source_root, archive_path=archive_file, contract=contract)
    archive_payload = _read_json(archive.path, label="r241 exact20 archive receipt")[1]
    archive_by_date = {
        str(row["date"]): str(row["sha256"])
        for row in archive_payload.get("archives") or ()
        if isinstance(row, Mapping)
    }
    rows = list(manifest_payload.get("shards") or ())
    _require(len(rows) == WINDOW_DAYS, "r241 manifest must contain exactly 20 shards")
    shards: list[FileIdentity] = []
    sidecars: list[FileIdentity] = []
    seen_dates: list[str] = []
    seen_paths: set[str] = set()
    decisions = 0
    records = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise R241Exact20TransferError("r241 manifest shard row is not an object")
        relative = str(row.get("path") or "")
        _require(relative not in seen_paths, "r241 manifest repeats a shard path")
        seen_paths.add(relative)
        shard_path = _safe_member(source_root, relative, label="r241 manifest shard")
        expected_sha = str(row.get("sha256") or "")
        _require(_is_sha256(expected_sha), "r241 manifest shard digest is invalid")
        expected_bytes = _exact_int(row.get("bytes"), label=f"r241 shard bytes {relative}")
        shard = file_identity(
            shard_path,
            label=f"r241 shard {relative}",
            expected_sha256=expected_sha,
            expected_size_bytes=expected_bytes,
        )
        sidecar_path = _safe_member(source_root, relative + ".json", label="r241 shard sidecar")
        sidecar = file_identity(sidecar_path, label=f"r241 shard sidecar {relative}")
        try:
            with shard.path.open("rb") as stream:
                header = pickle.load(stream)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError) as exc:
            raise R241Exact20TransferError(f"r241 shard header is unreadable: {relative}") from exc
        if not isinstance(header, Mapping):
            raise R241Exact20TransferError(f"r241 shard header is not an object: {relative}")
        _, sidecar_payload = _read_json(sidecar.path, label=f"r241 shard sidecar {relative}")
        dates = list(row.get("source_dates") or ())
        _require(len(dates) == 1 and isinstance(dates[0], str), "r241 shard source date is invalid")
        shard_date = str(dates[0])
        seen_dates.append(shard_date)
        expected_archive = archive_by_date.get(shard_date)
        _require(expected_archive is not None, "r241 shard date is outside the archive window")
        _require(
            row.get("required_archetype") == "alakazam"
            and row.get("selection_archetype") == "alakazam"
            and row.get("compact_mode") == "temporal-expert-v1"
            and _exact_int(row.get("dataset_schema"), label="r241 shard dataset schema") == 6
            and _exact_int(row.get("feature_schema"), label="r241 shard feature schema") == 5
            and _exact_int(row.get("max_context"), label="r241 shard max context") == 320
            and row.get("source_archive_sha256") == expected_archive,
            f"r241 manifest shard provenance drifted: {relative}",
        )
        _require(
            list(header.get("source_dates") or ()) == [shard_date]
            and header.get("source_archive_sha256") == expected_archive
            and header.get("required_archetype") == "alakazam"
            and header.get("compact_mode") == "temporal-expert-v1"
            and _exact_int(header.get("dataset_schema"), label="r241 header dataset schema") == 6
            and _exact_int(header.get("feature_schema"), label="r241 header feature schema") == 5
            and _exact_int(header.get("max_context"), label="r241 header max context") == 320,
            f"r241 shard header provenance drifted: {relative}",
        )
        _require(
            list(sidecar_payload.get("source_dates") or ()) == [shard_date]
            and sidecar_payload.get("source_archive_sha256") == expected_archive
            and _exact_int(sidecar_payload.get("dataset_schema"), label="r241 sidecar dataset schema") == 6
            and _exact_int(sidecar_payload.get("feature_schema"), label="r241 sidecar feature schema") == 5,
            f"r241 shard sidecar provenance drifted: {relative}",
        )
        stats = dict(row.get("stats") or {})
        decisions += _exact_int(stats.get("decisions_kept"), label="r241 shard decisions")
        records += _exact_int(stats.get("records_kept"), label="r241 shard records")
        shards.append(shard)
        sidecars.append(sidecar)
    _require(seen_dates == list(contract.dates), "r241 manifest shard dates are not exact and ordered")
    _require(
        sum(item.size_bytes for item in shards) == contract.shard_bytes
        and decisions == contract.decisions
        and records == contract.records,
        "r241 staged shard totals drifted from the sealed source pointer",
    )
    return CorpusIdentity(
        ready=ready,
        source_pointer=source_pointer,
        manifest=manifest,
        archive=archive,
        shards=tuple(shards),
        sidecars=tuple(sidecars),
        records=records,
        decisions=decisions,
        shard_bytes=sum(item.size_bytes for item in shards),
    )


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _create_only_bytes(path: Path, body: bytes, *, label: str) -> FileIdentity:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise R241Exact20TransferError(f"{label} path is a symlink: {path}")
    if path.exists():
        existing = file_identity(path, label=label)
        expected_sha = "sha256:" + hashlib.sha256(body).hexdigest()
        _require(
            existing.sha256 == expected_sha and existing.size_bytes == len(body),
            f"{label} already exists with another identity: {path}",
        )
        return existing
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            # link(2) is an atomic create-only publish and never replaces an
            # existing pointer/receipt even when another process races us.
            os.link(temporary, path)
        except FileExistsError:
            existing = file_identity(path, label=label)
            expected_sha = "sha256:" + hashlib.sha256(body).hexdigest()
            _require(
                existing.sha256 == expected_sha and existing.size_bytes == len(body),
                f"{label} appeared with another identity: {path}",
            )
            return existing
    finally:
        temporary.unlink(missing_ok=True)
    return file_identity(path, label=label)


def _create_only_json(path: Path, payload: Mapping[str, object], *, label: str) -> FileIdentity:
    return _create_only_bytes(path, _canonical_json(payload), label=label)


def _copy_create_only(source: Path, destination: Path, *, label: str) -> FileIdentity:
    source_identity = file_identity(source, label=f"{label} source")
    if destination.is_symlink():
        raise R241Exact20TransferError(f"{label} destination is a symlink: {destination}")
    if destination.exists():
        existing = file_identity(destination, label=label)
        _require(_same_content(existing, source_identity), f"{label} destination identity differs")
        return existing
    return _create_only_bytes(destination, source.read_bytes(), label=label)


def _transfer_receipt(
    identity: CorpusIdentity,
    *,
    destination: Path,
    archive_destination: Path,
    source_host: str,
    source_root: str,
    source_archive_receipt: str,
) -> dict[str, object]:
    return {
        "schema": TRANSFER_SCHEMA,
        "status": "ready",
        "candidate_id": R241_CANDIDATE_ID,
        "source": {
            "host": source_host,
            "window_root": source_root,
            "archive_receipt_path": source_archive_receipt,
            "final_ready": identity.ready.as_dict(relative_to=destination),
            "protected_pointer": identity.source_pointer.as_dict(relative_to=destination),
            "manifest": identity.manifest.as_dict(relative_to=destination),
        },
        "destination": {
            "root": str(destination),
            "archive_receipt_path": str(archive_destination),
            "archive_receipt": identity.archive.as_dict(relative_to=destination),
        },
        "corpus": {
            "records": identity.records,
            "decisions": identity.decisions,
            "shard_bytes": identity.shard_bytes,
            "shards": [row.as_dict(relative_to=destination) for row in identity.shards],
            "sidecars": [row.as_dict(relative_to=destination) for row in identity.sidecars],
        },
        "source_mutated": False,
        "active_training_modified": False,
    }


def _top_level_pointer(
    source_pointer_payload: Mapping[str, object],
    identity: CorpusIdentity,
    *,
    destination: Path,
    archive_destination: Path,
    transfer_receipt: FileIdentity,
    source_host: str,
    source_pointer_path: str,
    source_ready_path: str,
    source_archive_receipt: str,
) -> dict[str, object]:
    """Preserve the source pointer body while adding local r241 provenance."""

    pointer = dict(source_pointer_payload)
    pointer["r241_source_finalization"] = {
        "host": source_host,
        "source_ready_path": source_ready_path,
        "ready": identity.ready.as_dict(relative_to=destination),
        "source_pointer_path": source_pointer_path,
        "source_pointer": identity.source_pointer.as_dict(relative_to=destination),
        "source_manifest": identity.manifest.as_dict(relative_to=destination),
    }
    pointer["r241_archive_binding"] = {
        "archive_receipt_path": str(archive_destination),
        "archive_receipt_sha256": identity.archive.sha256,
        "archive_receipt_size_bytes": identity.archive.size_bytes,
        "copied_archive_receipt": identity.archive.as_dict(relative_to=destination),
        "source_host": source_host,
        "source_archive_receipt_path": source_archive_receipt,
        "source_archive_receipt_sha256": identity.archive.sha256,
        "source_archive_receipt_size_bytes": identity.archive.size_bytes,
    }
    pointer["r241_exact20_transfer"] = {
        "schema": TRANSFER_SCHEMA,
        "receipt": transfer_receipt.as_dict(relative_to=destination),
        "source_mutated": False,
    }
    return pointer


def _validate_finalized_destination(
    destination: Path | str,
    *,
    archive_destination: Path | str,
    contract: Exact20Contract = DEFAULT_CONTRACT,
) -> CorpusIdentity:
    """Validate the complete landed tree and its external archive binding."""

    root = Path(destination).expanduser().resolve()
    archive_path = Path(archive_destination).expanduser().resolve()
    identity = validate_staged_source(root, archive_path=root / ARCHIVE_COPY_NAME, contract=contract)
    external_archive, _archive_payload = _validate_archive(archive_path, contract=contract)
    _require(
        _same_content(external_archive, identity.archive),
        "external archive receipt differs from copied r241 archive",
    )
    pointer_path, pointer = _read_json(root / TOP_LEVEL_POINTER_NAME, label="r241 finalized protected pointer")
    pointer_identity = file_identity(pointer_path, label="r241 finalized protected pointer")
    _require(
        pointer.get("schema") == POINTER_SCHEMA
        and pointer.get("protected") is True
        and pointer.get("manifest") == MANIFEST_NAME
        and pointer.get("manifest_sha256") == identity.manifest.sha256,
        "r241 finalized protected pointer lost its source manifest identity",
    )
    source = dict(pointer.get("r241_source_finalization") or {})
    source_ready = dict(source.get("ready") or {})
    source_pointer = dict(source.get("source_pointer") or {})
    source_manifest = dict(source.get("source_manifest") or {})
    _same_identity(identity.ready, source_ready, label="r241 source READY provenance")
    _same_identity(identity.source_pointer, source_pointer, label="r241 source pointer provenance")
    _same_identity(identity.manifest, source_manifest, label="r241 source manifest provenance")
    _require(
        source_ready.get("path") == SOURCE_READY_COPY_NAME
        and source_pointer.get("path") == SOURCE_POINTER_NAME
        and source_manifest.get("path") == MANIFEST_NAME,
        "r241 finalized pointer does not preserve exact source file locations",
    )
    binding = dict(pointer.get("r241_archive_binding") or {})
    copied_archive = dict(binding.get("copied_archive_receipt") or {})
    _same_identity(identity.archive, copied_archive, label="r241 copied archive provenance")
    _require(
        binding.get("archive_receipt_path") == str(archive_path)
        and binding.get("archive_receipt_sha256") == external_archive.sha256
        and _exact_int(binding.get("archive_receipt_size_bytes"), label="r241 archive binding size")
        == external_archive.size_bytes
        and binding.get("source_archive_receipt_sha256") == external_archive.sha256
        and _exact_int(binding.get("source_archive_receipt_size_bytes"), label="r241 source archive binding size")
        == external_archive.size_bytes,
        "r241 finalized pointer lacks the exact copied archive binding",
    )
    transfer = dict(pointer.get("r241_exact20_transfer") or {})
    receipt_row = dict(transfer.get("receipt") or {})
    receipt_path = _safe_member(root, receipt_row.get("path"), label="r241 transfer receipt")
    receipt_identity = file_identity(receipt_path, label="r241 transfer receipt")
    _same_identity(receipt_identity, receipt_row, label="r241 transfer receipt")
    _, receipt = _read_json(receipt_path, label="r241 transfer receipt")
    _require(
        transfer.get("schema") == TRANSFER_SCHEMA
        and receipt.get("schema") == TRANSFER_SCHEMA
        and receipt.get("status") == "ready"
        and receipt.get("candidate_id") == R241_CANDIDATE_ID
        and receipt.get("source_mutated") is False
        and receipt.get("active_training_modified") is False,
        "r241 transfer receipt is not an inert exact20 receipt",
    )
    # The raw pointer itself must remain available and byte-identical.  This is
    # the critical distinction between adding a host-local binding and
    # rewriting the Elmo source provenance.
    raw_pointer = _read_json(root / SOURCE_POINTER_NAME, label="r241 copied source pointer")[1]
    pointer_without_extensions = {
        key: value
        for key, value in pointer.items()
        if not key.startswith("r241_")
    }
    _require(
        pointer_without_extensions == raw_pointer,
        "r241 top-level pointer altered the source pointer body",
    )
    _require(pointer_identity.size_bytes > identity.source_pointer.size_bytes, "r241 finalized pointer lacks binding fields")
    return identity


def _make_stage(
    parent: Path,
    *,
    destination_name: str,
) -> Path:
    """Return the one durable, non-runtime partial root for this handoff.

    A failed rsync must not make the next operator re-download multi-gigabyte
    shards.  This path remains outside the runtime name and has no top-level
    pointer until every hash check has passed.  It is therefore safe to reuse
    under the private destination lock, while a malformed or foreign partial
    still fails closed during validation.
    """

    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{destination_name}.r241-exact20.partial"
    if stage.is_symlink():
        raise R241Exact20TransferError(f"r241 partial root is a symlink: {stage}")
    if stage.exists():
        _require(stage.is_dir(), f"r241 partial root is not a directory: {stage}")
        return stage.resolve()
    try:
        stage.mkdir(mode=0o700)
    except FileExistsError:
        # Another producer is not supported, but the destination lock makes a
        # same-process race impossible.  Never treat an unknown entry as a
        # usable partial root.
        _require(
            stage.is_dir() and not stage.is_symlink(),
            f"r241 partial root appeared unsafely: {stage}",
        )
    return stage.resolve()


@contextlib.contextmanager
def _destination_lock(destination: Path) -> Iterator[None]:
    """Serialize only this create-only handoff; it never controls training."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.r241-exact20.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Promote a staged directory without replacing any existing root."""

    _require(not destination.exists() and not destination.is_symlink(), "r241 destination appeared before promotion")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is not None and os.name == "posix":
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        # Linux's AT_FDCWD / RENAME_NOREPLACE.  Inzi is the production caller;
        # the guarded fallback keeps focused macOS tests portable.
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result == 0:
            return
        failure = ctypes.get_errno()
        if failure == errno.EEXIST:
            raise R241Exact20TransferError("r241 destination appeared during promotion")
        if failure not in {errno.ENOSYS, errno.EINVAL}:
            raise OSError(failure, os.strerror(failure), str(destination))
    # Tests run on hosts without Linux renameat2.  The caller holds a private
    # lock and has already rejected any destination, so this is still
    # fail-closed for the supported single-writer transfer procedure.
    if destination.exists() or destination.is_symlink():
        raise R241Exact20TransferError("r241 destination appeared during fallback promotion")
    os.rename(source, destination)


def _seal_read_only(root: Path, *, seal_root: bool) -> None:
    """Make staged contents immutable without preventing final rename(2).

    POSIX rename needs write permission on the source directory on macOS.  The
    staging root therefore remains writable until it has been promoted under
    its final create-only name; every child is already sealed before that
    boundary.
    """

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise R241Exact20TransferError(f"r241 transferred tree contains a symlink: {path}")
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
        else:
            raise R241Exact20TransferError(f"r241 transferred tree contains a non-file entry: {path}")
    if seal_root:
        root.chmod(0o555)


def _recover_published_stage(
    stage: Path,
    *,
    destination: Path,
    archive_destination: Path,
    contract: Exact20Contract,
) -> CorpusIdentity | None:
    """Finish a crash-interrupted publish without rewriting any receipt.

    The top-level pointer is created last.  Its presence in the durable partial
    root means that a previous invocation finished all content construction;
    we only rehash the complete handoff and perform the create-only rename.
    An invalid published partial is deliberately not repaired in place.
    """

    pointer = stage / TOP_LEVEL_POINTER_NAME
    if pointer.is_symlink():
        raise R241Exact20TransferError(
            f"r241 partial top-level pointer is a symlink: {pointer}"
        )
    if not pointer.exists():
        return None
    _validate_finalized_destination(
        stage,
        archive_destination=archive_destination,
        contract=contract,
    )
    _rename_noreplace(stage, destination)
    _seal_read_only(destination, seal_root=True)
    return _validate_finalized_destination(
        destination,
        archive_destination=archive_destination,
        contract=contract,
    )


def _assert_safe_partial_for_writes(stage: Path) -> None:
    """Reject links/devices in a retained partial before any copy can follow it."""

    _require(
        stage.is_dir() and not stage.is_symlink(),
        f"r241 partial root is unsafe: {stage}",
    )
    for entry in stage.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise R241Exact20TransferError(
                f"r241 partial root contains an unsafe entry: {entry}"
            )


def _publish_stage(
    stage: Path,
    *,
    destination: Path,
    archive_destination: Path,
    contract: Exact20Contract,
    source_host: str,
    source_root: str,
    source_archive_receipt: str,
) -> CorpusIdentity:
    identity = validate_staged_source(stage, contract=contract)
    _, source_pointer_payload = _read_json(stage / SOURCE_POINTER_NAME, label="r241 source protected pointer")
    archive_identity = _copy_create_only(
        identity.archive.path,
        archive_destination,
        label="r241 copied archive receipt",
    )
    _require(_same_content(archive_identity, identity.archive), "r241 archive copy identity drifted")
    receipt_payload = _transfer_receipt(
        identity,
        destination=stage,
        archive_destination=archive_destination.resolve(),
        source_host=source_host,
        source_root=source_root,
        source_archive_receipt=source_archive_receipt,
    )
    receipt_identity = _create_only_json(
        stage / TRANSFER_RECEIPT_NAME,
        receipt_payload,
        label="r241 transfer receipt",
    )
    top_pointer = _top_level_pointer(
        source_pointer_payload,
        identity,
        destination=stage,
        archive_destination=archive_destination.resolve(),
        transfer_receipt=receipt_identity,
        source_host=source_host,
        source_pointer_path=f"{source_root}/specialist-corpora/alakazam/PROTECTED_EXPERT_CORPUS.json",
        source_ready_path=f"{source_root}/{SOURCE_READY_NAME}",
        source_archive_receipt=source_archive_receipt,
    )
    _create_only_json(
        stage / TOP_LEVEL_POINTER_NAME,
        top_pointer,
        label="r241 top-level protected pointer",
    )
    # The receipt intentionally records the pre-promotion staging root.  Its
    # contents are a source-transfer audit, while the top-level pointer is the
    # only runtime marker and is revalidated after the final rename.
    _validate_finalized_destination(
        stage,
        archive_destination=archive_destination,
        contract=contract,
    )
    _seal_read_only(stage, seal_root=False)
    return identity


def finalize_local_copy(
    *,
    source_root: Path | str,
    source_archive_receipt: Path | str,
    destination: Path | str,
    archive_destination: Path | str,
    contract: Exact20Contract = DEFAULT_CONTRACT,
    source_host: str = SOURCE_HOST,
    source_root_label: str = SOURCE_WINDOW_ROOT,
    source_archive_label: str = SOURCE_ARCHIVE_RECEIPT,
) -> CorpusIdentity:
    """Testable local implementation of the exact same create-only finalizer.

    It is deliberately useful only for isolated test fixtures.  The CLI below
    pins its production remote paths and requires ``--execute``.
    """

    source = Path(source_root).expanduser().resolve()
    archive = Path(source_archive_receipt).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    target_archive = Path(archive_destination).expanduser().resolve()
    _require(source.is_dir() and not source.is_symlink(), "local r241 source root is unsafe")
    with _destination_lock(target):
        if target.exists() or target.is_symlink():
            return _validate_finalized_destination(
                target,
                archive_destination=target_archive,
                contract=contract,
            )
        stage = _make_stage(target.parent, destination_name=target.name)
        recovered = _recover_published_stage(
            stage,
            destination=target,
            archive_destination=target_archive,
            contract=contract,
        )
        if recovered is not None:
            return recovered
        _assert_safe_partial_for_writes(stage)
        try:
            shutil.copy2(source / SOURCE_READY_COPY_NAME, stage / SOURCE_READY_COPY_NAME)
            shutil.copy2(source / SOURCE_POINTER_NAME, stage / SOURCE_POINTER_NAME)
            shutil.copy2(source / MANIFEST_NAME, stage / MANIFEST_NAME)
            shutil.copy2(archive, stage / ARCHIVE_COPY_NAME)
            _validate_metadata(stage, archive_path=stage / ARCHIVE_COPY_NAME, contract=contract)
            manifest = _read_json(stage / MANIFEST_NAME, label="r241 source Alakazam manifest")[1]
            for row in manifest.get("shards") or ():
                relative = str(dict(row).get("path") or "")
                shard_source = _safe_member(source, relative, label="local r241 source shard")
                shard_target = _safe_member(stage, relative, label="local r241 staged shard")
                shard_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shard_source, shard_target)
                shutil.copy2(
                    _safe_member(source, relative + ".json", label="local r241 source sidecar"),
                    _safe_member(stage, relative + ".json", label="local r241 staged sidecar"),
                )
            identity = _publish_stage(
                stage,
                destination=target,
                archive_destination=target_archive,
                contract=contract,
                source_host=source_host,
                source_root=source_root_label,
                source_archive_receipt=source_archive_label,
            )
            _rename_noreplace(stage, target)
            _seal_read_only(target, seal_root=True)
            # Return identities rooted at the landed runtime path, never at
            # the now-nonexistent staging directory.  This also makes an
            # idempotent second invocation return the same object identity.
            return _validate_finalized_destination(
                target,
                archive_destination=target_archive,
                contract=contract,
            )
        except BaseException:
            # Preserve only this deterministic, non-runtime partial root so a
            # later operator can resume verified copies.  It has no top-level
            # protected pointer on ordinary failures, so it cannot be used by
            # training; deleting it would throw away safely resumable shards.
            raise


def _metadata_handoff_stage(parent: Path, *, destination_name: str) -> Path:
    """Return the separate small, non-runtime Elmo projection staging root."""

    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{destination_name}.r241-elmo-metadata-handoff.partial"
    if stage.is_symlink():
        raise R241Exact20TransferError(
            f"r241 Elmo metadata handoff partial root is a symlink: {stage}"
        )
    if stage.exists():
        _require(
            stage.is_dir(),
            f"r241 Elmo metadata handoff partial root is not a directory: {stage}",
        )
    else:
        try:
            stage.mkdir(mode=0o700)
        except FileExistsError:
            _require(
                stage.is_dir() and not stage.is_symlink(),
                f"r241 Elmo metadata handoff partial root appeared unsafely: {stage}",
            )
    return stage.resolve()


def _receipt_row(
    value: object,
    *,
    label: str,
    expected_path: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> dict[str, object]:
    """Validate one receipt identity row without resolving its local path."""

    _require(isinstance(value, Mapping), f"{label} must be an object")
    row = dict(value)
    _require(row.get("path") == expected_path, f"{label} path drifted")
    _require(_is_sha256(row.get("sha256")), f"{label} sha is malformed")
    size = _exact_int(row.get("size_bytes"), label=f"{label} size")
    _require(size >= 0, f"{label} size is negative")
    if expected_sha256 is not None:
        _require(row.get("sha256") == expected_sha256, f"{label} sha drifted")
    if expected_size_bytes is not None:
        _require(size == expected_size_bytes, f"{label} size drifted")
    return row


def _validate_inzi_transfer_receipt_bytes(
    body: bytes,
    *,
    contract: Exact20Contract = DEFAULT_CONTRACT,
    expected_sha256: str | None = INZI_TRANSFER_RECEIPT_SHA256,
    expected_size_bytes: int | None = INZI_TRANSFER_RECEIPT_SIZE_BYTES,
    expected_archive_destination: Path | str = DESTINATION_ARCHIVE_RECEIPT,
) -> dict[str, object]:
    """Validate the sealed Inzi receipt that backs an Elmo metadata projection."""

    actual_sha256 = "sha256:" + hashlib.sha256(body).hexdigest()
    if expected_sha256 is not None:
        _require(
            actual_sha256 == expected_sha256,
            "r241 Elmo metadata handoff Inzi receipt checksum drifted",
        )
    if expected_size_bytes is not None:
        _require(
            len(body) == expected_size_bytes,
            "r241 Elmo metadata handoff Inzi receipt size drifted",
        )
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241Exact20TransferError(
            "r241 Elmo metadata handoff Inzi receipt is not JSON"
        ) from exc
    _require(isinstance(value, Mapping), "r241 Elmo metadata handoff Inzi receipt is not an object")
    receipt = dict(value)
    _require(
        receipt.get("schema") == TRANSFER_SCHEMA
        and receipt.get("status") == "ready"
        and receipt.get("candidate_id") == R241_CANDIDATE_ID
        and receipt.get("source_mutated") is False
        and receipt.get("active_training_modified") is False,
        "r241 Elmo metadata handoff Inzi receipt is not inert ready evidence",
    )
    _require(isinstance(receipt.get("source"), Mapping), "r241 Inzi receipt source is not an object")
    source = dict(receipt["source"])
    _require(
        source.get("host") == SOURCE_HOST
        and source.get("window_root") == SOURCE_WINDOW_ROOT
        and source.get("archive_receipt_path") == SOURCE_ARCHIVE_RECEIPT,
        "r241 Inzi receipt source provenance drifted",
    )
    _receipt_row(
        source.get("final_ready"),
        label="r241 Inzi receipt READY",
        expected_path=SOURCE_READY_COPY_NAME,
        expected_sha256=contract.ready_sha256,
        expected_size_bytes=contract.ready_size_bytes,
    )
    _receipt_row(
        source.get("protected_pointer"),
        label="r241 Inzi receipt source pointer",
        expected_path=SOURCE_POINTER_NAME,
        expected_sha256=contract.source_pointer_sha256,
        expected_size_bytes=contract.source_pointer_size_bytes,
    )
    _receipt_row(
        source.get("manifest"),
        label="r241 Inzi receipt manifest",
        expected_path=MANIFEST_NAME,
        expected_sha256=contract.manifest_sha256,
        expected_size_bytes=contract.manifest_size_bytes,
    )
    _require(
        isinstance(receipt.get("destination"), Mapping),
        "r241 Inzi receipt destination is not an object",
    )
    destination = dict(receipt["destination"])
    _require(
        destination.get("archive_receipt_path")
        == str(Path(expected_archive_destination).expanduser()),
        "r241 Inzi receipt archive destination drifted",
    )
    _receipt_row(
        destination.get("archive_receipt"),
        label="r241 Inzi receipt copied archive",
        expected_path=ARCHIVE_COPY_NAME,
        expected_sha256=contract.archive_sha256,
        expected_size_bytes=contract.archive_size_bytes,
    )
    _require(isinstance(receipt.get("corpus"), Mapping), "r241 Inzi receipt corpus is not an object")
    corpus = dict(receipt["corpus"])
    _require(
        _exact_int(corpus.get("records"), label="r241 Inzi receipt records") == contract.records
        and _exact_int(corpus.get("decisions"), label="r241 Inzi receipt decisions")
        == contract.decisions
        and _exact_int(corpus.get("shard_bytes"), label="r241 Inzi receipt shard bytes")
        == contract.shard_bytes,
        "r241 Inzi receipt corpus totals drifted",
    )
    shards = list(corpus.get("shards") or ())
    sidecars = list(corpus.get("sidecars") or ())
    _require(
        len(shards) == len(contract.dates) and len(sidecars) == len(contract.dates),
        "r241 Inzi receipt does not retain exactly twenty shard identities",
    )
    shard_total = 0
    for day, shard, sidecar in zip(contract.dates, shards, sidecars, strict=True):
        relative = f"all-recognized-{day}.alakazam.features"
        shard_row = _receipt_row(
            shard,
            label=f"r241 Inzi receipt shard {day}",
            expected_path=relative,
        )
        _require(
            _exact_int(shard_row.get("size_bytes"), label=f"r241 Inzi shard {day} size") > 0,
            f"r241 Inzi receipt shard {day} is empty",
        )
        shard_total += _exact_int(shard_row.get("size_bytes"), label=f"r241 Inzi shard {day} size")
        _receipt_row(
            sidecar,
            label=f"r241 Inzi receipt sidecar {day}",
            expected_path=f"{relative}.json",
        )
    _require(
        shard_total == contract.shard_bytes,
        "r241 Inzi receipt shard rows do not add to the sealed total",
    )
    return receipt


def _elmo_metadata_corpus_identity(
    root: Path,
    *,
    archive_path: Path,
    contract: Exact20Contract,
) -> tuple[CorpusIdentity, dict[str, Any]]:
    """Validate only the small metadata files used by an Elmo projection."""

    (
        ready,
        source_pointer,
        manifest,
        archive,
        _ready_payload,
        pointer_payload,
        _manifest_payload,
    ) = (
        _validate_metadata(root, archive_path=archive_path, contract=contract)
    )
    return (
        CorpusIdentity(
            ready=ready,
            source_pointer=source_pointer,
            manifest=manifest,
            archive=archive,
            shards=(),
            sidecars=(),
            records=contract.records,
            decisions=contract.decisions,
            shard_bytes=contract.shard_bytes,
        ),
        pointer_payload,
    )


def _elmo_metadata_transfer_receipt(
    *,
    inzi_receipt: Mapping[str, object],
    inzi_receipt_identity: FileIdentity,
    destination: Path,
    metadata_root: Path,
    archive_destination: Path,
    identity: CorpusIdentity,
    inzi_receipt_path: str,
) -> dict[str, object]:
    """Create a local receipt while retaining the sealed Inzi receipt verbatim."""

    return {
        "schema": TRANSFER_SCHEMA,
        "status": "ready",
        "candidate_id": R241_CANDIDATE_ID,
        "source": dict(inzi_receipt["source"]),
        "destination": {
            "root": str(destination),
            "archive_receipt_path": str(archive_destination),
            "archive_receipt": identity.archive.as_dict(relative_to=metadata_root),
        },
        "corpus": dict(inzi_receipt["corpus"]),
        "source_mutated": False,
        "active_training_modified": False,
        "r241_elmo_metadata_handoff": {
            "schema": ELMO_METADATA_HANDOFF_SCHEMA,
            "metadata_only": True,
            "feature_shards_copied": False,
            "feature_sidecars_copied": False,
            "source_archive_reused_without_mutation": True,
            "inzi_transfer_receipt": {
                "host": INZI_HOST,
                "remote_path": inzi_receipt_path,
                "local_copy": inzi_receipt_identity.as_dict(relative_to=metadata_root),
            },
        },
    }


def _validate_elmo_metadata_handoff_destination(
    destination: Path | str,
    *,
    archive_destination: Path | str,
    inzi_receipt_body: bytes,
    contract: Exact20Contract = DEFAULT_CONTRACT,
    expected_inzi_receipt_sha256: str | None = INZI_TRANSFER_RECEIPT_SHA256,
    expected_inzi_receipt_size_bytes: int | None = INZI_TRANSFER_RECEIPT_SIZE_BYTES,
    expected_inzi_archive_destination: Path | str = DESTINATION_ARCHIVE_RECEIPT,
    inzi_receipt_path: str = INZI_TRANSFER_RECEIPT_PATH,
    published_destination: Path | str | None = None,
) -> ElmoMetadataHandoff:
    """Validate an Elmo metadata-only root without requiring shard copies."""

    root = Path(destination).expanduser().resolve()
    external_archive = Path(archive_destination).expanduser().resolve()
    public_root = Path(published_destination or root).expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), "r241 Elmo metadata handoff root is unsafe")
    expected_members = {
        SOURCE_READY_COPY_NAME,
        SOURCE_POINTER_NAME,
        MANIFEST_NAME,
        ARCHIVE_COPY_NAME,
        INZI_TRANSFER_RECEIPT_COPY_NAME,
        TRANSFER_RECEIPT_NAME,
        TOP_LEVEL_POINTER_NAME,
    }
    actual_members = {entry.name for entry in root.iterdir()}
    _require(
        actual_members == expected_members,
        "r241 Elmo metadata handoff must contain only its seven small receipt files",
    )
    _require(
        all(entry.is_file() and not entry.is_symlink() for entry in root.iterdir()),
        "r241 Elmo metadata handoff contains a non-file or symlink",
    )
    identity, source_pointer_payload = _elmo_metadata_corpus_identity(
        root,
        archive_path=root / ARCHIVE_COPY_NAME,
        contract=contract,
    )
    archive_identity, _archive_payload = _validate_archive(external_archive, contract=contract)
    _require(
        _same_content(identity.archive, archive_identity),
        "r241 Elmo metadata copied archive differs from the bound source archive",
    )
    copied_inzi = file_identity(
        root / INZI_TRANSFER_RECEIPT_COPY_NAME,
        label="r241 copied sealed Inzi transfer receipt",
        expected_sha256=expected_inzi_receipt_sha256,
        expected_size_bytes=expected_inzi_receipt_size_bytes,
    )
    _require(
        copied_inzi.path.read_bytes() == inzi_receipt_body,
        "r241 Elmo metadata handoff did not preserve the exact Inzi receipt bytes",
    )
    inzi_receipt = _validate_inzi_transfer_receipt_bytes(
        inzi_receipt_body,
        contract=contract,
        expected_sha256=expected_inzi_receipt_sha256,
        expected_size_bytes=expected_inzi_receipt_size_bytes,
        expected_archive_destination=expected_inzi_archive_destination,
    )
    transfer_identity = file_identity(
        root / TRANSFER_RECEIPT_NAME,
        label="r241 Elmo metadata transfer receipt",
    )
    _, transfer_receipt = _read_json(
        transfer_identity.path, label="r241 Elmo metadata transfer receipt"
    )
    _require(
        transfer_receipt.get("schema") == TRANSFER_SCHEMA
        and transfer_receipt.get("status") == "ready"
        and transfer_receipt.get("candidate_id") == R241_CANDIDATE_ID
        and transfer_receipt.get("source_mutated") is False
        and transfer_receipt.get("active_training_modified") is False
        and transfer_receipt.get("source") == inzi_receipt.get("source")
        and transfer_receipt.get("corpus") == inzi_receipt.get("corpus"),
        "r241 Elmo metadata transfer receipt drifted from the sealed Inzi evidence",
    )
    receipt_destination = dict(transfer_receipt.get("destination") or {})
    _require(
        receipt_destination.get("root") == str(public_root)
        and receipt_destination.get("archive_receipt_path") == str(external_archive)
        and receipt_destination.get("archive_receipt")
        == identity.archive.as_dict(relative_to=root),
        "r241 Elmo metadata transfer receipt has another destination binding",
    )
    projection = dict(transfer_receipt.get("r241_elmo_metadata_handoff") or {})
    origin = dict(projection.get("inzi_transfer_receipt") or {})
    _require(
        projection.get("schema") == ELMO_METADATA_HANDOFF_SCHEMA
        and projection.get("metadata_only") is True
        and projection.get("feature_shards_copied") is False
        and projection.get("feature_sidecars_copied") is False
        and projection.get("source_archive_reused_without_mutation") is True
        and origin.get("host") == INZI_HOST
        and origin.get("remote_path") == inzi_receipt_path
        and origin.get("local_copy") == copied_inzi.as_dict(relative_to=root),
        "r241 Elmo metadata handoff provenance is incomplete",
    )
    pointer_identity, pointer = _read_json(
        root / TOP_LEVEL_POINTER_NAME, label="r241 Elmo top-level protected pointer"
    )
    pointer_file = file_identity(pointer_identity, label="r241 Elmo top-level protected pointer")
    expected_pointer = _top_level_pointer(
        source_pointer_payload,
        identity,
        destination=root,
        archive_destination=external_archive,
        transfer_receipt=transfer_identity,
        source_host=SOURCE_HOST,
        source_pointer_path=(
            f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/"
            "PROTECTED_EXPERT_CORPUS.json"
        ),
        source_ready_path=f"{SOURCE_WINDOW_ROOT}/{SOURCE_READY_NAME}",
        source_archive_receipt=SOURCE_ARCHIVE_RECEIPT,
    )
    _require(
        pointer == expected_pointer,
        "r241 Elmo metadata top-level pointer does not exactly preserve source provenance",
    )
    return ElmoMetadataHandoff(
        root=root,
        pointer=pointer_file,
        transfer_receipt=transfer_identity,
        inzi_transfer_receipt=copied_inzi,
        archive=archive_identity,
    )


def finalize_elmo_metadata_handoff_local_copy(
    *,
    source_ready: Path | str,
    source_pointer: Path | str,
    source_manifest: Path | str,
    source_archive_receipt: Path | str,
    inzi_transfer_receipt_body: bytes,
    destination: Path | str,
    contract: Exact20Contract = DEFAULT_CONTRACT,
    expected_inzi_receipt_sha256: str | None = INZI_TRANSFER_RECEIPT_SHA256,
    expected_inzi_receipt_size_bytes: int | None = INZI_TRANSFER_RECEIPT_SIZE_BYTES,
    expected_inzi_archive_destination: Path | str = DESTINATION_ARCHIVE_RECEIPT,
    inzi_receipt_path: str = INZI_TRANSFER_RECEIPT_PATH,
) -> ElmoMetadataHandoff:
    """Create a metadata-only Elmo handoff from sealed source and Inzi bytes.

    This helper is used by focused tests and the guarded Elmo SSH process.  It
    never reads or writes a feature shard; the complete shard inventory remains
    in the checksum-pinned Inzi receipt copied as one small provenance file.
    """

    ready_source = Path(source_ready).expanduser().resolve()
    pointer_source = Path(source_pointer).expanduser().resolve()
    manifest_source = Path(source_manifest).expanduser().resolve()
    archive_source = Path(source_archive_receipt).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    inzi_receipt = _validate_inzi_transfer_receipt_bytes(
        inzi_transfer_receipt_body,
        contract=contract,
        expected_sha256=expected_inzi_receipt_sha256,
        expected_size_bytes=expected_inzi_receipt_size_bytes,
        expected_archive_destination=expected_inzi_archive_destination,
    )
    _validate_archive(archive_source, contract=contract)
    with _destination_lock(target):
        if target.exists() or target.is_symlink():
            return _validate_elmo_metadata_handoff_destination(
                target,
                archive_destination=archive_source,
                inzi_receipt_body=inzi_transfer_receipt_body,
                contract=contract,
                expected_inzi_receipt_sha256=expected_inzi_receipt_sha256,
                expected_inzi_receipt_size_bytes=expected_inzi_receipt_size_bytes,
                expected_inzi_archive_destination=expected_inzi_archive_destination,
                inzi_receipt_path=inzi_receipt_path,
            )
        stage = _metadata_handoff_stage(target.parent, destination_name=target.name)
        if (stage / TOP_LEVEL_POINTER_NAME).exists():
            handoff = _validate_elmo_metadata_handoff_destination(
                stage,
                archive_destination=archive_source,
                inzi_receipt_body=inzi_transfer_receipt_body,
                contract=contract,
                expected_inzi_receipt_sha256=expected_inzi_receipt_sha256,
                expected_inzi_receipt_size_bytes=expected_inzi_receipt_size_bytes,
                expected_inzi_archive_destination=expected_inzi_archive_destination,
                inzi_receipt_path=inzi_receipt_path,
                published_destination=target,
            )
            _rename_noreplace(stage, target)
            _seal_read_only(target, seal_root=True)
            return _validate_elmo_metadata_handoff_destination(
                target,
                archive_destination=archive_source,
                inzi_receipt_body=inzi_transfer_receipt_body,
                contract=contract,
                expected_inzi_receipt_sha256=expected_inzi_receipt_sha256,
                expected_inzi_receipt_size_bytes=expected_inzi_receipt_size_bytes,
                expected_inzi_archive_destination=expected_inzi_archive_destination,
                inzi_receipt_path=inzi_receipt_path,
            )
        _assert_safe_partial_for_writes(stage)
        _copy_create_only(
            ready_source, stage / SOURCE_READY_COPY_NAME, label="r241 Elmo copied source READY"
        )
        _copy_create_only(
            pointer_source,
            stage / SOURCE_POINTER_NAME,
            label="r241 Elmo copied source protected pointer",
        )
        _copy_create_only(
            manifest_source, stage / MANIFEST_NAME, label="r241 Elmo copied source manifest"
        )
        _copy_create_only(
            archive_source,
            stage / ARCHIVE_COPY_NAME,
            label="r241 Elmo copied source archive receipt",
        )
        identity, source_pointer_payload = _elmo_metadata_corpus_identity(
            stage, archive_path=stage / ARCHIVE_COPY_NAME, contract=contract
        )
        external_archive = file_identity(
            archive_source,
            label="r241 Elmo bound source archive receipt",
            expected_sha256=contract.archive_sha256,
            expected_size_bytes=contract.archive_size_bytes,
        )
        _require(
            _same_content(identity.archive, external_archive),
            "r241 Elmo metadata archive copy drifted from the source archive",
        )
        copied_inzi = _create_only_bytes(
            stage / INZI_TRANSFER_RECEIPT_COPY_NAME,
            inzi_transfer_receipt_body,
            label="r241 copied sealed Inzi transfer receipt",
        )
        receipt_payload = _elmo_metadata_transfer_receipt(
            inzi_receipt=inzi_receipt,
            inzi_receipt_identity=copied_inzi,
            destination=target,
            metadata_root=stage,
            archive_destination=archive_source,
            identity=identity,
            inzi_receipt_path=inzi_receipt_path,
        )
        receipt_identity = _create_only_json(
            stage / TRANSFER_RECEIPT_NAME,
            receipt_payload,
            label="r241 Elmo metadata transfer receipt",
        )
        pointer_payload = _top_level_pointer(
            source_pointer_payload,
            identity,
            destination=stage,
            archive_destination=archive_source,
            transfer_receipt=receipt_identity,
            source_host=SOURCE_HOST,
            source_pointer_path=(
                f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/"
                "PROTECTED_EXPERT_CORPUS.json"
            ),
            source_ready_path=f"{SOURCE_WINDOW_ROOT}/{SOURCE_READY_NAME}",
            source_archive_receipt=SOURCE_ARCHIVE_RECEIPT,
        )
        _create_only_json(
            stage / TOP_LEVEL_POINTER_NAME,
            pointer_payload,
            label="r241 Elmo metadata top-level protected pointer",
        )
        _validate_elmo_metadata_handoff_destination(
            stage,
            archive_destination=archive_source,
            inzi_receipt_body=inzi_transfer_receipt_body,
            contract=contract,
            expected_inzi_receipt_sha256=expected_inzi_receipt_sha256,
            expected_inzi_receipt_size_bytes=expected_inzi_receipt_size_bytes,
            expected_inzi_archive_destination=expected_inzi_archive_destination,
            inzi_receipt_path=inzi_receipt_path,
            published_destination=target,
        )
        _seal_read_only(stage, seal_root=False)
        _rename_noreplace(stage, target)
        _seal_read_only(target, seal_root=True)
        return _validate_elmo_metadata_handoff_destination(
            target,
            archive_destination=archive_source,
            inzi_receipt_body=inzi_transfer_receipt_body,
            contract=contract,
            expected_inzi_receipt_sha256=expected_inzi_receipt_sha256,
            expected_inzi_receipt_size_bytes=expected_inzi_receipt_size_bytes,
            expected_inzi_archive_destination=expected_inzi_archive_destination,
            inzi_receipt_path=inzi_receipt_path,
        )


def _remote_readonly_command(host: str, command: str, *, label: str) -> bytes:
    """Run one explicitly read-only source inspection command over SSH."""

    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise R241Exact20TransferError(
            f"r241 read-only Elmo plan failed for {label}{(': ' + detail) if detail else ''}"
        ) from exc
    return bytes(result.stdout)


def _remote_readonly_bytes(
    host: str,
    source: str,
    *,
    label: str,
    limit_bytes: int | None = None,
) -> bytes:
    """Read a source file (or harmless prefix) without creating a destination."""

    quoted = shlex.quote(source)
    command = (
        f"exec sudo -n cat -- {quoted}"
        if limit_bytes is None
        else f"exec sudo -n head -c {int(limit_bytes)} -- {quoted}"
    )
    return _remote_readonly_command(host, command, label=label)


def _remote_file_identity(host: str, source: str, *, label: str) -> FileIdentity:
    """Obtain an Elmo file SHA-256 and byte count without copying the file."""

    quoted = shlex.quote(source)
    body = _remote_readonly_command(
        host,
        (
            "set -eu; "
            f"sudo -n sha256sum -- {quoted}; "
            f"sudo -n stat -c '%s' -- {quoted}"
        ),
        label=label,
    ).decode("utf-8", errors="strict")
    rows = body.splitlines()
    if len(rows) != 2:
        raise R241Exact20TransferError(
            f"r241 read-only Elmo identity output is malformed for {label}"
        )
    digest = rows[0].split(maxsplit=1)[0].strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise R241Exact20TransferError(
            f"r241 read-only Elmo digest is malformed for {label}"
        )
    try:
        size = int(rows[1].strip())
    except ValueError as exc:
        raise R241Exact20TransferError(
            f"r241 read-only Elmo size is malformed for {label}"
        ) from exc
    _require(size >= 0, f"r241 read-only Elmo size is negative for {label}")
    return FileIdentity(Path(source), f"sha256:{digest}", size)


def _remote_json(
    host: str,
    source: str,
    *,
    label: str,
) -> tuple[FileIdentity, dict[str, Any], bytes]:
    """Read only a small JSON receipt and prove it matches remote hashing."""

    identity = _remote_file_identity(host, source, label=label)
    body = _remote_readonly_bytes(host, source, label=label)
    _require(
        identity.sha256 == "sha256:" + hashlib.sha256(body).hexdigest()
        and identity.size_bytes == len(body),
        f"r241 read-only Elmo JSON changed while being inspected: {label}",
    )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241Exact20TransferError(
            f"r241 read-only Elmo JSON is malformed: {label}"
        ) from exc
    if not isinstance(payload, dict):
        raise R241Exact20TransferError(
            f"r241 read-only Elmo JSON is not an object: {label}"
        )
    return identity, payload, body


def _remote_alakazam_member(relative: object, *, label: str) -> str:
    """Turn a manifest member into one fixed, shell-quoted-safe Elmo path."""

    text = str(relative or "")
    member = PurePosixPath(text)
    if (
        not text
        or member.is_absolute()
        or ".." in member.parts
        or any(part in {"", "."} for part in member.parts)
    ):
        raise R241Exact20TransferError(f"{label} is not a safe remote member path")
    return f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/{member.as_posix()}"


def _require_remote_identity(
    identity: FileIdentity,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> None:
    _require(
        identity.sha256 == expected_sha256 and identity.size_bytes == expected_size_bytes,
        f"r241 read-only Elmo identity drifted for {label}: "
        f"expected={expected_sha256}/{expected_size_bytes} "
        f"actual={identity.sha256}/{identity.size_bytes}",
    )


def plan_from_elmo(
    *,
    contract: Exact20Contract = DEFAULT_CONTRACT,
) -> dict[str, object]:
    """Read-only, source-side preflight for the one immutable corpus handoff.

    This intentionally uses SSH only for ``cat``, ``head``, ``sha256sum`` and
    ``stat`` under ``sudo -n``.  It never invokes rsync, creates neither Inzi
    destination, and returns the exact members that a later ``--execute`` may
    copy after separate approval.
    """

    ready_source = f"{SOURCE_WINDOW_ROOT}/{SOURCE_READY_NAME}"
    pointer_source = (
        f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/"
        "PROTECTED_EXPERT_CORPUS.json"
    )
    manifest_source = f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/{MANIFEST_NAME}"
    ready_identity, ready_payload, ready_bytes = _remote_json(
        SOURCE_HOST, ready_source, label="r241 source final READY receipt"
    )
    pointer_identity, pointer_payload, pointer_bytes = _remote_json(
        SOURCE_HOST, pointer_source, label="r241 source protected pointer"
    )
    manifest_identity, manifest_payload, manifest_bytes = _remote_json(
        SOURCE_HOST, manifest_source, label="r241 source Alakazam manifest"
    )
    archive_identity, archive_payload, archive_bytes = _remote_json(
        SOURCE_HOST, SOURCE_ARCHIVE_RECEIPT, label="r241 exact20 archive receipt"
    )
    _require_remote_identity(
        ready_identity,
        expected_sha256=contract.ready_sha256,
        expected_size_bytes=contract.ready_size_bytes,
        label="source final READY receipt",
    )
    _require_remote_identity(
        pointer_identity,
        expected_sha256=contract.source_pointer_sha256,
        expected_size_bytes=contract.source_pointer_size_bytes,
        label="source protected pointer",
    )
    _require_remote_identity(
        manifest_identity,
        expected_sha256=contract.manifest_sha256,
        expected_size_bytes=contract.manifest_size_bytes,
        label="source Alakazam manifest",
    )
    _require_remote_identity(
        archive_identity,
        expected_sha256=contract.archive_sha256,
        expected_size_bytes=contract.archive_size_bytes,
        label="source exact20 archive receipt",
    )

    # Reuse the same strict metadata gate as a real transfer without writing
    # anything beneath either production target.  The temporary directory only
    # contains four small receipt files and vanishes before this function
    # returns; every corpus feature remains on Elmo during plan mode.
    with tempfile.TemporaryDirectory(prefix="r241-exact20-plan-") as temporary:
        metadata_root = Path(temporary)
        (metadata_root / SOURCE_READY_COPY_NAME).write_bytes(ready_bytes)
        (metadata_root / SOURCE_POINTER_NAME).write_bytes(pointer_bytes)
        (metadata_root / MANIFEST_NAME).write_bytes(manifest_bytes)
        archive_local = metadata_root / ARCHIVE_COPY_NAME
        archive_local.write_bytes(archive_bytes)
        _validate_metadata(metadata_root, archive_path=archive_local, contract=contract)

    archive_by_date = {
        str(row["date"]): str(row["sha256"])
        for row in archive_payload.get("archives") or ()
        if isinstance(row, Mapping)
    }
    rows = list(manifest_payload.get("shards") or ())
    _require(len(rows) == WINDOW_DAYS, "r241 source plan manifest must contain 20 shards")
    planned_members: list[dict[str, object]] = [
        {
            "role": "source_final_ready",
            "source_path": ready_source,
            "destination_paths": [str(DESTINATION_ROOT / SOURCE_READY_COPY_NAME)],
            **ready_identity.as_dict(),
        },
        {
            "role": "source_protected_pointer",
            "source_path": pointer_source,
            "destination_paths": [str(DESTINATION_ROOT / SOURCE_POINTER_NAME)],
            **pointer_identity.as_dict(),
        },
        {
            "role": "source_manifest",
            "source_path": manifest_source,
            "destination_paths": [str(DESTINATION_ROOT / MANIFEST_NAME)],
            **manifest_identity.as_dict(),
        },
        {
            "role": "exact20_archive_receipt",
            "source_path": SOURCE_ARCHIVE_RECEIPT,
            "destination_paths": [
                str(DESTINATION_ROOT / ARCHIVE_COPY_NAME),
                str(DESTINATION_ARCHIVE_RECEIPT),
            ],
            **archive_identity.as_dict(),
        },
    ]
    seen_paths: set[str] = set()
    shard_bytes = 0
    records = 0
    decisions = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise R241Exact20TransferError("r241 source plan manifest shard row is not an object")
        relative = str(row.get("path") or "")
        _require(relative not in seen_paths, "r241 source plan manifest repeats a shard path")
        seen_paths.add(relative)
        source = _remote_alakazam_member(relative, label="r241 source plan shard")
        expected_sha = str(row.get("sha256") or "")
        expected_size = _exact_int(row.get("bytes"), label=f"r241 source plan bytes {relative}")
        _require(_is_sha256(expected_sha), "r241 source plan shard digest is invalid")
        shard_identity = _remote_file_identity(SOURCE_HOST, source, label=f"r241 shard {relative}")
        _require_remote_identity(
            shard_identity,
            expected_sha256=expected_sha,
            expected_size_bytes=expected_size,
            label=f"source shard {relative}",
        )
        sidecar_source = _remote_alakazam_member(
            relative + ".json", label="r241 source plan sidecar"
        )
        sidecar_identity, sidecar_payload, _sidecar_bytes = _remote_json(
            SOURCE_HOST, sidecar_source, label=f"r241 shard sidecar {relative}"
        )
        try:
            header = pickle.load(
                io.BytesIO(
                    _remote_readonly_bytes(
                        SOURCE_HOST,
                        source,
                        label=f"r241 shard header {relative}",
                        limit_bytes=1_048_576,
                    )
                )
            )
        except (pickle.PickleError, EOFError, AttributeError, ImportError) as exc:
            raise R241Exact20TransferError(
                f"r241 read-only Elmo shard header is unreadable: {relative}"
            ) from exc
        if not isinstance(header, Mapping):
            raise R241Exact20TransferError(
                f"r241 read-only Elmo shard header is not an object: {relative}"
            )
        dates = list(row.get("source_dates") or ())
        _require(
            len(dates) == 1 and isinstance(dates[0], str),
            "r241 source plan shard source date is invalid",
        )
        shard_date = str(dates[0])
        expected_archive = archive_by_date.get(shard_date)
        _require(expected_archive is not None, "r241 source plan shard date is outside archive")
        _require(
            row.get("required_archetype") == "alakazam"
            and row.get("selection_archetype") == "alakazam"
            and row.get("compact_mode") == "temporal-expert-v1"
            and _exact_int(row.get("dataset_schema"), label="r241 source plan dataset schema") == 6
            and _exact_int(row.get("feature_schema"), label="r241 source plan feature schema") == 5
            and _exact_int(row.get("max_context"), label="r241 source plan max context") == 320
            and row.get("source_archive_sha256") == expected_archive,
            f"r241 source plan shard provenance drifted: {relative}",
        )
        _require(
            list(header.get("source_dates") or ()) == [shard_date]
            and header.get("source_archive_sha256") == expected_archive
            and header.get("required_archetype") == "alakazam"
            and header.get("compact_mode") == "temporal-expert-v1"
            and _exact_int(header.get("dataset_schema"), label="r241 source plan header dataset schema") == 6
            and _exact_int(header.get("feature_schema"), label="r241 source plan header feature schema") == 5
            and _exact_int(header.get("max_context"), label="r241 source plan header max context") == 320,
            f"r241 source plan shard header provenance drifted: {relative}",
        )
        _require(
            list(sidecar_payload.get("source_dates") or ()) == [shard_date]
            and sidecar_payload.get("source_archive_sha256") == expected_archive
            and _exact_int(sidecar_payload.get("dataset_schema"), label="r241 source plan sidecar dataset schema") == 6
            and _exact_int(sidecar_payload.get("feature_schema"), label="r241 source plan sidecar feature schema") == 5,
            f"r241 source plan sidecar provenance drifted: {relative}",
        )
        stats = dict(row.get("stats") or {})
        shard_bytes += shard_identity.size_bytes
        records += _exact_int(stats.get("records_kept"), label="r241 source plan records")
        decisions += _exact_int(stats.get("decisions_kept"), label="r241 source plan decisions")
        planned_members.extend(
            (
                {
                    "role": "alakazam_feature_shard",
                    "source_path": source,
                    "destination_paths": [str(DESTINATION_ROOT / relative)],
                    **shard_identity.as_dict(),
                },
                {
                    "role": "alakazam_feature_sidecar",
                    "source_path": sidecar_source,
                    "destination_paths": [str(DESTINATION_ROOT / f"{relative}.json")],
                    **sidecar_identity.as_dict(),
                },
            )
        )
    _require(
        [str(dict(row).get("source_dates", [""])[0]) for row in rows]
        == list(contract.dates),
        "r241 source plan shard dates are not exact and ordered",
    )
    _require(
        shard_bytes == contract.shard_bytes
        and records == contract.records
        and decisions == contract.decisions,
        "r241 source plan shard totals drifted from exact20 contract",
    )
    return {
        "schema": TRANSFER_SCHEMA,
        "status": "source_validated_plan_only",
        "source_host": SOURCE_HOST,
        "source_read_only": True,
        "rsync_invoked": False,
        "transfer_executed": False,
        "source_mutated": False,
        "service_action": "none",
        "source_window_root": SOURCE_WINDOW_ROOT,
        "source_archive_receipt": SOURCE_ARCHIVE_RECEIPT,
        "destination": str(DESTINATION_ROOT),
        "archive_destination": str(DESTINATION_ARCHIVE_RECEIPT),
        "planned_member_count": len(planned_members),
        "planned_source_file_bytes": sum(
            _exact_int(member["size_bytes"], label="planned source file size")
            for member in planned_members
        ),
        "corpus": {
            "records": records,
            "decisions": decisions,
            "shard_bytes": shard_bytes,
            "feature_shard_count": len(rows),
            "sidecar_count": len(rows),
        },
        "members": planned_members,
    }


def _remote_rsync(host: str, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--no-links",
            "--partial",
            "--append-verify",
            "--protect-args",
            "--rsync-path=sudo -n rsync",
            f"{host}:{source}",
            str(destination),
        ],
        check=True,
        # The controller parses one compact completion JSON line from remote
        # stdout.  Rsync diagnostics remain on stderr, but its normal chatter
        # must never be mistaken for a finalization receipt.
        stdout=subprocess.DEVNULL,
    )


def _transfer_from_elmo_on_inzi(
    *,
    destination: Path = DESTINATION_ROOT,
    archive_destination: Path = DESTINATION_ARCHIVE_RECEIPT,
    contract: Exact20Contract = DEFAULT_CONTRACT,
) -> CorpusIdentity:
    """Perform the create-only transfer inside the Inzi host namespace only."""

    _require(
        os.environ.get("R241_EXACT20_INZI_FINALIZER") == "1",
        "r241 destination finalization is permitted only inside the Inzi SSH process",
    )

    target = destination.expanduser().resolve()
    target_archive = archive_destination.expanduser().resolve()
    with _destination_lock(target):
        if target.exists() or target.is_symlink():
            return _validate_finalized_destination(
                target,
                archive_destination=target_archive,
                contract=contract,
            )
        stage = _make_stage(target.parent, destination_name=target.name)
        recovered = _recover_published_stage(
            stage,
            destination=target,
            archive_destination=target_archive,
            contract=contract,
        )
        if recovered is not None:
            return recovered
        _assert_safe_partial_for_writes(stage)
        try:
            _remote_rsync(SOURCE_HOST, f"{SOURCE_WINDOW_ROOT}/{SOURCE_READY_NAME}", stage / SOURCE_READY_COPY_NAME)
            _remote_rsync(
                SOURCE_HOST,
                f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/PROTECTED_EXPERT_CORPUS.json",
                stage / SOURCE_POINTER_NAME,
            )
            _remote_rsync(
                SOURCE_HOST,
                f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/{MANIFEST_NAME}",
                stage / MANIFEST_NAME,
            )
            _remote_rsync(SOURCE_HOST, SOURCE_ARCHIVE_RECEIPT, stage / ARCHIVE_COPY_NAME)
            _validate_metadata(stage, archive_path=stage / ARCHIVE_COPY_NAME, contract=contract)
            manifest = _read_json(stage / MANIFEST_NAME, label="r241 source Alakazam manifest")[1]
            for row in manifest.get("shards") or ():
                relative = str(dict(row).get("path") or "")
                _safe_member(stage, relative, label="r241 staged shard")
                _remote_rsync(
                    SOURCE_HOST,
                    f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/{relative}",
                    stage / relative,
                )
                _remote_rsync(
                    SOURCE_HOST,
                    f"{SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/{relative}.json",
                    stage / f"{relative}.json",
                )
            identity = _publish_stage(
                stage,
                destination=target,
                archive_destination=target_archive,
                contract=contract,
                source_host=SOURCE_HOST,
                source_root=SOURCE_WINDOW_ROOT,
                source_archive_receipt=SOURCE_ARCHIVE_RECEIPT,
            )
            _rename_noreplace(stage, target)
            _seal_read_only(target, seal_root=True)
            return _validate_finalized_destination(
                target,
                archive_destination=target_archive,
                contract=contract,
            )
        except BaseException:
            # Keep the non-runtime partial root for rsync --append-verify on a
            # later explicit invocation.  No source tree or runtime pointer is
            # changed by this failure path.
            raise


def _inzi_completion_payload(identity: CorpusIdentity) -> dict[str, object]:
    """The only stdout payload emitted by the remote Inzi finalizer."""

    return {
        "schema": TRANSFER_SCHEMA,
        "status": "ready",
        "execution_host": INZI_HOST,
        "destination": str(DESTINATION_ROOT),
        "archive_destination": str(DESTINATION_ARCHIVE_RECEIPT),
        "manifest_sha256": identity.manifest.sha256,
        "source_pointer_sha256": identity.source_pointer.sha256,
        "archive_receipt_sha256": identity.archive.sha256,
        "records": identity.records,
        "decisions": identity.decisions,
        "shard_bytes": identity.shard_bytes,
        "source_mutated": False,
        "service_action": "none",
    }


def _parse_inzi_completion(output: bytes) -> dict[str, object]:
    """Accept exactly one ready JSON line from the streamed remote process."""

    candidates: list[dict[str, object]] = []
    for raw_line in output.decode("utf-8", errors="replace").splitlines():
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(dict(value))
    if len(candidates) != 1:
        raise R241Exact20TransferError(
            "r241 Inzi finalizer did not emit one unambiguous completion receipt"
        )
    payload = candidates[0]
    if (
        payload.get("schema") != TRANSFER_SCHEMA
        or payload.get("status") != "ready"
        or payload.get("execution_host") != INZI_HOST
        or payload.get("destination") != str(DESTINATION_ROOT)
        or payload.get("archive_destination") != str(DESTINATION_ARCHIVE_RECEIPT)
        or payload.get("manifest_sha256") != DEFAULT_CONTRACT.manifest_sha256
        or payload.get("source_pointer_sha256") != DEFAULT_CONTRACT.source_pointer_sha256
        or payload.get("archive_receipt_sha256") != DEFAULT_CONTRACT.archive_sha256
        or _exact_int(payload.get("records"), label="Inzi transfer records")
        != DEFAULT_CONTRACT.records
        or _exact_int(payload.get("decisions"), label="Inzi transfer decisions")
        != DEFAULT_CONTRACT.decisions
        or _exact_int(payload.get("shard_bytes"), label="Inzi transfer shard bytes")
        != DEFAULT_CONTRACT.shard_bytes
        or payload.get("source_mutated") is not False
        or payload.get("service_action") != "none"
    ):
        raise R241Exact20TransferError(
            "r241 Inzi finalizer completion receipt does not match the exact20 contract"
        )
    return payload


def transfer_from_elmo(
    *,
    destination: Path = DESTINATION_ROOT,
    archive_destination: Path = DESTINATION_ARCHIVE_RECEIPT,
    contract: Exact20Contract = DEFAULT_CONTRACT,
) -> dict[str, object]:
    """Stream the finalizer to Inzi; never touch Inzi paths from the controller.

    The controller has no `/home/inzi` mount.  It sends this exact source file
    over SSH to ``python3 - --remote-inzi-finalize`` so all directory locking,
    partial resumption, rsync reads, checksum verification and atomic rename
    occur in Inzi's own filesystem namespace.  No controller-side ``Path``
    operation is allowed on either production destination.
    """

    if (
        Path(destination).expanduser() != DESTINATION_ROOT
        or Path(archive_destination).expanduser() != DESTINATION_ARCHIVE_RECEIPT
        or contract != DEFAULT_CONTRACT
    ):
        raise R241Exact20TransferError(
            "r241 controller transfer accepts only the immutable Inzi exact20 endpoints"
        )
    script_source = _regular(Path(__file__), label="r241 Inzi finalizer source").read_bytes()
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        INZI_HOST,
        "env",
        "R241_EXACT20_INZI_FINALIZER=1",
        "python3",
        "-",
        REMOTE_INZI_FINALIZE_FLAG,
    ]
    try:
        result = subprocess.run(
            command,
            input=script_source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise R241Exact20TransferError(
            "r241 Inzi-only finalizer failed before an accepted completion receipt"
            + (f": {detail}" if detail else "")
        ) from exc
    return _parse_inzi_completion(bytes(result.stdout))


def _read_sealed_inzi_transfer_receipt() -> bytes:
    """Fetch only the small, already-sealed Inzi receipt for Elmo provenance."""

    body = _remote_readonly_command(
        INZI_HOST,
        f"exec cat -- {shlex.quote(INZI_TRANSFER_RECEIPT_PATH)}",
        label="sealed Inzi exact20 transfer receipt",
    )
    _validate_inzi_transfer_receipt_bytes(body)
    return body


def _elmo_handoff_source_paths() -> tuple[Path, Path, Path, Path]:
    """Return the immutable small Elmo roots used by the metadata projection."""

    window = Path(SOURCE_WINDOW_ROOT)
    return (
        window / SOURCE_READY_NAME,
        window / "specialist-corpora" / "alakazam" / "PROTECTED_EXPERT_CORPUS.json",
        window / "specialist-corpora" / "alakazam" / MANIFEST_NAME,
        Path(SOURCE_ARCHIVE_RECEIPT),
    )


def _elmo_metadata_receipt_from_environment() -> bytes:
    """Decode and authenticate controller-provided sealed Inzi receipt bytes."""

    encoded = os.environ.get(ELMO_METADATA_INZI_RECEIPT_ENV)
    _require(encoded is not None, "r241 Elmo metadata handoff lacks the sealed Inzi receipt")
    _require(
        len(encoded) <= 131_072,
        "r241 Elmo metadata handoff receipt environment is unexpectedly large",
    )
    try:
        body = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise R241Exact20TransferError(
            "r241 Elmo metadata handoff receipt environment is not base64"
        ) from exc
    _validate_inzi_transfer_receipt_bytes(body)
    return body


def _elmo_metadata_handoff_on_elmo(*, plan_only: bool) -> dict[str, object]:
    """Create/validate the small Elmo handoff inside Elmo's filesystem only."""

    _require(
        os.environ.get(ELMO_METADATA_HANDOFF_ENV) == "1",
        "r241 Elmo metadata handoff is permitted only inside the guarded Elmo SSH process",
    )
    inzi_body = _elmo_metadata_receipt_from_environment()
    ready_source, pointer_source, manifest_source, archive_source = _elmo_handoff_source_paths()
    ready = file_identity(
        ready_source,
        label="r241 Elmo source READY receipt",
        expected_sha256=DEFAULT_CONTRACT.ready_sha256,
        expected_size_bytes=DEFAULT_CONTRACT.ready_size_bytes,
    )
    pointer = file_identity(
        pointer_source,
        label="r241 Elmo raw source protected pointer",
        expected_sha256=DEFAULT_CONTRACT.source_pointer_sha256,
        expected_size_bytes=DEFAULT_CONTRACT.source_pointer_size_bytes,
    )
    manifest = file_identity(
        manifest_source,
        label="r241 Elmo source manifest",
        expected_sha256=DEFAULT_CONTRACT.manifest_sha256,
        expected_size_bytes=DEFAULT_CONTRACT.manifest_size_bytes,
    )
    archive, _archive_payload = _validate_archive(archive_source, contract=DEFAULT_CONTRACT)
    if plan_only:
        state = "absent"
        if ELMO_METADATA_HANDOFF_ROOT.exists() or ELMO_METADATA_HANDOFF_ROOT.is_symlink():
            _validate_elmo_metadata_handoff_destination(
                ELMO_METADATA_HANDOFF_ROOT,
                archive_destination=archive_source,
                inzi_receipt_body=inzi_body,
            )
            state = "existing_valid"
        return {
            "schema": ELMO_METADATA_HANDOFF_SCHEMA,
            "status": "source_and_inzi_receipt_validated_plan_only",
            "execution_host": SOURCE_HOST,
            "destination": str(ELMO_METADATA_HANDOFF_ROOT),
            "destination_state": state,
            "archive_destination": str(archive_source),
            "inzi_transfer_receipt_path": INZI_TRANSFER_RECEIPT_PATH,
            "inzi_transfer_receipt_sha256": INZI_TRANSFER_RECEIPT_SHA256,
            "source_ready_sha256": ready.sha256,
            "source_pointer_sha256": pointer.sha256,
            "manifest_sha256": manifest.sha256,
            "archive_receipt_sha256": archive.sha256,
            "metadata_only": True,
            "feature_shards_copied": False,
            "source_mutated": False,
            "service_action": "none",
        }
    handoff = finalize_elmo_metadata_handoff_local_copy(
        source_ready=ready_source,
        source_pointer=pointer_source,
        source_manifest=manifest_source,
        source_archive_receipt=archive_source,
        inzi_transfer_receipt_body=inzi_body,
        destination=ELMO_METADATA_HANDOFF_ROOT,
    )
    return {
        "schema": ELMO_METADATA_HANDOFF_SCHEMA,
        "status": "ready",
        "execution_host": SOURCE_HOST,
        "destination": str(handoff.root),
        "archive_destination": str(archive_source),
        "inzi_transfer_receipt_path": INZI_TRANSFER_RECEIPT_PATH,
        "pointer_sha256": handoff.pointer.sha256,
        "transfer_receipt_sha256": handoff.transfer_receipt.sha256,
        "inzi_transfer_receipt_sha256": handoff.inzi_transfer_receipt.sha256,
        "archive_receipt_sha256": handoff.archive.sha256,
        "source_ready_sha256": ready.sha256,
        "source_pointer_sha256": pointer.sha256,
        "manifest_sha256": manifest.sha256,
        "metadata_only": True,
        "feature_shards_copied": False,
        "source_mutated": False,
        "service_action": "none",
    }


def _parse_elmo_metadata_completion(
    output: bytes,
    *,
    expected_status: str,
) -> dict[str, object]:
    """Accept exactly one completion JSON object from the guarded Elmo process."""

    candidates: list[dict[str, object]] = []
    for raw_line in output.decode("utf-8", errors="replace").splitlines():
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(dict(value))
    if len(candidates) != 1:
        raise R241Exact20TransferError(
            "r241 Elmo metadata handoff did not emit one unambiguous completion receipt"
        )
    payload = candidates[0]
    _require(
        payload.get("schema") == ELMO_METADATA_HANDOFF_SCHEMA
        and payload.get("status") == expected_status
        and payload.get("execution_host") == SOURCE_HOST
        and payload.get("destination") == str(ELMO_METADATA_HANDOFF_ROOT)
        and payload.get("archive_destination") == SOURCE_ARCHIVE_RECEIPT
        and payload.get("inzi_transfer_receipt_path")
        == INZI_TRANSFER_RECEIPT_PATH
        and payload.get("inzi_transfer_receipt_sha256")
        == INZI_TRANSFER_RECEIPT_SHA256
        and payload.get("source_ready_sha256") == DEFAULT_CONTRACT.ready_sha256
        and payload.get("source_pointer_sha256") == DEFAULT_CONTRACT.source_pointer_sha256
        and payload.get("manifest_sha256") == DEFAULT_CONTRACT.manifest_sha256
        and payload.get("archive_receipt_sha256") == DEFAULT_CONTRACT.archive_sha256
        and payload.get("metadata_only") is True
        and payload.get("feature_shards_copied") is False
        and payload.get("source_mutated") is False
        and payload.get("service_action") == "none",
        "r241 Elmo metadata handoff completion does not match the sealed contract",
    )
    if expected_status == "ready":
        _require(
            _is_sha256(payload.get("pointer_sha256"))
            and _is_sha256(payload.get("transfer_receipt_sha256")),
            "r241 Elmo metadata handoff completion omits local receipt identities",
        )
    return payload


def _run_elmo_metadata_handoff(*, plan_only: bool) -> dict[str, object]:
    """Stream this exact code to Elmo without touching Elmo paths locally."""

    inzi_receipt = _read_sealed_inzi_transfer_receipt()
    script_source = _regular(Path(__file__), label="r241 Elmo metadata handoff source").read_bytes()
    stream = json.dumps(
        {
            "schema": ELMO_METADATA_STREAM_SCHEMA,
            "script_b64": base64.b64encode(script_source).decode("ascii"),
            "script_sha256": "sha256:" + hashlib.sha256(script_source).hexdigest(),
            "inzi_transfer_receipt_b64": base64.b64encode(inzi_receipt).decode("ascii"),
            "inzi_transfer_receipt_sha256": "sha256:"
            + hashlib.sha256(inzi_receipt).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    remote_command = " ".join(
        shlex.quote(value)
        for value in (
            "sudo",
            "-n",
            "python3",
            "-c",
            _ELMO_METADATA_REMOTE_BOOTSTRAP,
            REMOTE_ELMO_METADATA_PLAN_FLAG
            if plan_only
            else REMOTE_ELMO_METADATA_HANDOFF_FLAG,
        )
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        SOURCE_HOST,
        remote_command,
    ]
    try:
        result = subprocess.run(
            command,
            input=stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise R241Exact20TransferError(
            "r241 guarded Elmo metadata handoff failed before an accepted completion receipt"
            + (f": {detail}" if detail else "")
        ) from exc
    return _parse_elmo_metadata_completion(
        bytes(result.stdout),
        expected_status=("source_and_inzi_receipt_validated_plan_only" if plan_only else "ready"),
    )


def _production_args(args: argparse.Namespace) -> None:
    if Path(args.destination).expanduser() != DESTINATION_ROOT:
        raise R241Exact20TransferError("r241 transfer destination is fixed by the runtime registry")
    if Path(args.archive_destination).expanduser() != DESTINATION_ARCHIVE_RECEIPT:
        raise R241Exact20TransferError("r241 archive receipt destination is fixed by the runtime registry")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="read-only source-side Elmo hash/metadata preflight; never copies",
    )
    mode.add_argument("--execute", action="store_true", help="perform the one create-only transfer")
    mode.add_argument(
        "--elmo-metadata-plan",
        action="store_true",
        help="read-only plan for the small Elmo exact20 handoff projection",
    )
    mode.add_argument(
        "--elmo-metadata-execute",
        action="store_true",
        help="create-only Elmo metadata handoff; never copies feature shards",
    )
    mode.add_argument(REMOTE_INZI_FINALIZE_FLAG, action="store_true", help=argparse.SUPPRESS)
    mode.add_argument(REMOTE_ELMO_METADATA_HANDOFF_FLAG, action="store_true", help=argparse.SUPPRESS)
    mode.add_argument(REMOTE_ELMO_METADATA_PLAN_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--destination", type=Path, default=DESTINATION_ROOT)
    parser.add_argument("--archive-destination", type=Path, default=DESTINATION_ARCHIVE_RECEIPT)
    args = parser.parse_args(argv)
    _production_args(args)
    if args.remote_inzi_finalize:
        identity = _transfer_from_elmo_on_inzi(
            destination=args.destination,
            archive_destination=args.archive_destination,
        )
        print(json.dumps(_inzi_completion_payload(identity), separators=(",", ":")))
        return 0
    if args.remote_elmo_metadata_handoff:
        print(
            json.dumps(
                _elmo_metadata_handoff_on_elmo(plan_only=False), separators=(",", ":")
            )
        )
        return 0
    if args.remote_elmo_metadata_plan:
        print(
            json.dumps(
                _elmo_metadata_handoff_on_elmo(plan_only=True), separators=(",", ":")
            )
        )
        return 0
    if args.plan:
        print(json.dumps(plan_from_elmo(), indent=2, sort_keys=True))
        return 0
    if args.elmo_metadata_plan:
        print(json.dumps(_run_elmo_metadata_handoff(plan_only=True), indent=2, sort_keys=True))
        return 0
    if not args.execute and not args.elmo_metadata_execute:
        print(
            json.dumps(
                {
                    "schema": TRANSFER_SCHEMA,
                    "status": "inert_plan_only",
                    "source_host": SOURCE_HOST,
                    "source_window_root": SOURCE_WINDOW_ROOT,
                    "source_archive_receipt": SOURCE_ARCHIVE_RECEIPT,
                    "destination": str(DESTINATION_ROOT),
                    "archive_destination": str(DESTINATION_ARCHIVE_RECEIPT),
                    "execute_required": True,
                    "source_mutated": False,
                    "service_action": "none",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    completion = (
        _run_elmo_metadata_handoff(plan_only=False)
        if args.elmo_metadata_execute
        else transfer_from_elmo(
            destination=args.destination,
            archive_destination=args.archive_destination,
        )
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
