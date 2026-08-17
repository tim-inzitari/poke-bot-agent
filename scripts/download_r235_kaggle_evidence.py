#!/usr/bin/env python3
"""Collect immutable R235 replay evidence with both agent-seat logs.

``materialize`` is entirely offline: it seals already saved metadata, replay,
and seat-0/seat-1 log files.  ``download`` is deliberately explicit and makes
exactly one Kaggle API request for the selected episode list, replay, and each
of the two seat logs.  It has no retry, queue, upload, or submission code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r235_direct_submission_tooling as support


SCHEMA = "poke_bot.r235_kaggle_both_seat_evidence/v1"


class R235EvidenceError(RuntimeError):
    """Downloaded or supplied evidence did not prove the exact R235 episode."""


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file() or raw.stat().st_size <= 0:
        raise R235EvidenceError(f"{label} must be a non-empty regular non-symlink file")
    return raw.resolve()


def _read_json(path: Path | str, *, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235EvidenceError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R235EvidenceError(f"{label} must be a JSON object")
    return value


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> Path:
    if path.exists() or path.is_symlink():
        raise R235EvidenceError(f"evidence target already exists: {path}")
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        created = True
        position = 0
        while position < len(payload):
            amount = os.write(descriptor, payload[position:])
            if amount <= 0:
                raise OSError("short evidence write")
            position += amount
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(path, mode)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise R235EvidenceError(f"cannot write evidence target: {path}") from exc
    return path.resolve()


def _copy_immutable(source: Path, target: Path) -> Path:
    source = _regular_file(source, label="evidence source")
    if target.exists() or target.is_symlink():
        raise R235EvidenceError(f"evidence target already exists: {target}")
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        created = True
        with source.open("rb") as input_stream:
            while True:
                block = input_stream.read(4 * 1024 * 1024)
                if not block:
                    break
                position = 0
                while position < len(block):
                    amount = os.write(descriptor, block[position:])
                    if amount <= 0:
                        raise OSError("short evidence copy")
                    position += amount
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(target, 0o444)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                target.unlink()
            except OSError:
                pass
        raise R235EvidenceError(f"cannot copy evidence target: {target}") from exc
    return target.resolve()


def _reserve_output(path: Path) -> Path:
    target = Path(path).expanduser()
    if target.exists() or target.is_symlink():
        raise R235EvidenceError(f"evidence output already exists: {target}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise R235EvidenceError("evidence output parent must already be a real directory")
    try:
        target.mkdir(mode=0o700)
    except OSError as exc:
        raise R235EvidenceError(f"cannot reserve evidence output: {target}") from exc
    return target.resolve()


def _value(value: object, *names: str) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
    else:
        for name in names:
            candidate = getattr(value, name, None)
            if candidate is not None:
                return candidate
    return None


def _integer(value: object, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise R235EvidenceError(f"{label} is not a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise R235EvidenceError(f"{label} is not a positive integer") from exc
    if result < minimum:
        raise R235EvidenceError(f"{label} is not a positive integer")
    return result


def _normalise_episode(value: object, *, submission_id: int, episode_id: int) -> dict[str, Any]:
    observed_id = _integer(_value(value, "episode_id", "id"), label="episode id")
    if observed_id != episode_id:
        raise R235EvidenceError("episode metadata ID does not match the selected episode")
    agents_raw = _value(value, "agents")
    if not isinstance(agents_raw, list):
        raise R235EvidenceError("episode metadata lacks agents")
    agents: list[dict[str, Any]] = []
    for agent in agents_raw:
        index = _integer(_value(agent, "index"), label="agent seat", minimum=0)
        agent_submission = _integer(
            _value(agent, "submission_id", "submissionId"), label="agent submission ID"
        )
        agents.append({"index": index, "submission_id": agent_submission})
    if {agent["index"] for agent in agents} != {0, 1} or len(agents) != 2:
        raise R235EvidenceError("episode metadata must identify exactly seats 0 and 1")
    own = [agent for agent in agents if agent["submission_id"] == submission_id]
    if len(own) != 1:
        raise R235EvidenceError("episode metadata does not identify the R235 submission exactly once")
    return {
        "episode_id": episode_id,
        "submission_id": submission_id,
        "submitted_seat": own[0]["index"],
        "seats": agents,
        "state": str(_value(value, "state") or ""),
        "type": str(_value(value, "type") or ""),
    }


def _context_and_submission_id(
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    binding_path: Path,
    r225_contract: Path,
    r236_contract: Path,
    submission_id_receipt_path: Path,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    try:
        context = support.validate_r235_binding(
            archive_path=archive_path,
            manifest_path=manifest_path,
            manifest_member=manifest_member,
            binding_path=binding_path,
            r225_contract=r225_contract,
            r236_contract=r236_contract,
        )
        id_receipt = support.validate_submission_id_receipt(context, submission_id_receipt_path)
        submission = id_receipt["payload"]["submission"]
        submission_id = _integer(submission.get("id_text", submission.get("id")), label="R235 submission ID")
    except (OSError, support.R235SupportError, KeyError, TypeError) as exc:
        raise R235EvidenceError(f"R235 provenance validation failed: {exc}") from exc
    return context, submission_id, id_receipt


def _file_row(source: Path, archived: Path) -> dict[str, Any]:
    return {
        "source_path": str(source),
        "source_sha256": support.sha256_file(source),
        "path": archived.name,
        "sha256": support.sha256_file(archived),
        "bytes": archived.stat().st_size,
    }


def _seal_reserved(
    *,
    output: Path,
    context: Mapping[str, Any],
    submission_id: int,
    id_receipt: Mapping[str, Any],
    metadata_source: Path,
    replay_source: Path,
    seat_log_sources: Mapping[int, Path],
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_payload = _read_json(metadata_source, label="episode metadata")
    episode_id = _integer(metadata_payload.get("episode_id", metadata_payload.get("id")), label="episode ID")
    episode = _normalise_episode(metadata_payload, submission_id=submission_id, episode_id=episode_id)
    if set(seat_log_sources) != {0, 1}:
        raise R235EvidenceError("both seat-0 and seat-1 logs are required")
    metadata_archived = _copy_immutable(metadata_source, output / "episode-metadata.json")
    replay_archived = _copy_immutable(
        replay_source, output / f"episode-{episode_id}-replay{Path(replay_source).suffix or '.json'}"
    )
    seat_archived = {
        seat: _copy_immutable(
            seat_log_sources[seat],
            output / f"episode-{episode_id}-seat-{seat}-logs{Path(seat_log_sources[seat]).suffix or '.json'}",
        )
        for seat in (0, 1)
    }
    files = {
        "episode_metadata": _file_row(_regular_file(metadata_source, label="episode metadata"), metadata_archived),
        "replay": _file_row(_regular_file(replay_source, label="replay"), replay_archived),
        "seat_0_logs": _file_row(_regular_file(seat_log_sources[0], label="seat 0 logs"), seat_archived[0]),
        "seat_1_logs": _file_row(_regular_file(seat_log_sources[1], label="seat 1 logs"), seat_archived[1]),
    }
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "immutable": True,
        "submission": {
            "id": submission_id,
            "competition": support.COMPETITION,
            "message": support.LABEL,
        },
        "episode": episode,
        "r235_binding": {"path": context["path"], "sha256": context["sha256"]},
        "submission_id_receipt": {"path": id_receipt["path"], "sha256": id_receipt["sha256"]},
        "acquisition": dict(acquisition),
        "files": files,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    receipt_path = _write_new_bytes(output / "R235_BOTH_SEAT_EVIDENCE.json", _canonical_json(receipt))
    os.chmod(output, 0o555)
    return {**receipt, "path": str(receipt_path), "sha256": support.sha256_file(receipt_path)}


def materialize_evidence(
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    binding_path: Path,
    r225_contract: Path,
    r236_contract: Path,
    submission_id_receipt_path: Path,
    episode_metadata_path: Path,
    replay_path: Path,
    seat_0_log_path: Path,
    seat_1_log_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Seal previously saved offline evidence; no remote client is involved."""

    context, submission_id, id_receipt = _context_and_submission_id(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest_member=manifest_member,
        binding_path=binding_path,
        r225_contract=r225_contract,
        r236_contract=r236_contract,
        submission_id_receipt_path=submission_id_receipt_path,
    )
    output = _reserve_output(output_path)
    return _seal_reserved(
        output=output,
        context=context,
        submission_id=submission_id,
        id_receipt=id_receipt,
        metadata_source=_regular_file(episode_metadata_path, label="episode metadata"),
        replay_source=_regular_file(replay_path, label="replay"),
        seat_log_sources={
            0: _regular_file(seat_0_log_path, label="seat 0 logs"),
            1: _regular_file(seat_1_log_path, label="seat 1 logs"),
        },
        acquisition={"mode": "offline_materialized", "network_calls": 0, "retry_allowed": False},
    )


def _find_download(directory: Path, expected: str, *, label: str) -> Path:
    candidate = directory / expected
    if not candidate.is_file() or candidate.is_symlink() or candidate.stat().st_size <= 0:
        raise R235EvidenceError(f"{label} is missing after the one API call: {candidate.name}")
    return candidate.resolve()


def download_evidence_once(
    api: Any,
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    binding_path: Path,
    r225_contract: Path,
    r236_contract: Path,
    submission_id_receipt_path: Path,
    episode_id: int,
    output_path: Path,
) -> dict[str, Any]:
    """Use exactly four API calls: list, replay, seat-0 logs, seat-1 logs."""

    context, submission_id, id_receipt = _context_and_submission_id(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest_member=manifest_member,
        binding_path=binding_path,
        r225_contract=r225_contract,
        r236_contract=r236_contract,
        submission_id_receipt_path=submission_id_receipt_path,
    )
    selected_episode = _integer(episode_id, label="selected episode ID")
    output = _reserve_output(output_path)
    staging = output / ".download"
    staging.mkdir(mode=0o700)
    # No retries: each direct API call is intentionally singular.
    rows = api.competition_list_episodes(submission_id)
    matching = [row for row in list(rows) if _integer(_value(row, "id", "episode_id"), label="API episode ID") == selected_episode]
    if len(matching) != 1:
        raise R235EvidenceError("API episode list does not resolve exactly the selected episode")
    metadata = _normalise_episode(matching[0], submission_id=submission_id, episode_id=selected_episode)
    # Keep the source shape consumable by the common offline sealing path.
    metadata_source = _write_new_bytes(
        staging / "episode-metadata.json",
        _canonical_json(
            {
                "episode_id": metadata["episode_id"],
                "submission_id": metadata["submission_id"],
                "agents": metadata["seats"],
                "state": metadata["state"],
                "type": metadata["type"],
            }
        ),
        mode=0o600,
    )
    api.competition_episode_replay(selected_episode, str(staging))
    api.competition_episode_agent_logs(selected_episode, 0, str(staging))
    api.competition_episode_agent_logs(selected_episode, 1, str(staging))
    replay = _find_download(staging, f"episode-{selected_episode}-replay.json", label="replay")
    seat_0 = _find_download(staging, f"episode-{selected_episode}-agent-0-logs.json", label="seat 0 logs")
    seat_1 = _find_download(staging, f"episode-{selected_episode}-agent-1-logs.json", label="seat 1 logs")
    expected_staging_files = {
        "episode-metadata.json",
        replay.name,
        seat_0.name,
        seat_1.name,
    }
    observed_staging_files = set()
    for child in staging.iterdir():
        if child.is_symlink() or not child.is_file() or child.stat().st_size <= 0:
            raise R235EvidenceError("API staging produced an unsafe non-file artifact")
        observed_staging_files.add(child.name)
    if observed_staging_files != expected_staging_files:
        raise R235EvidenceError("API staging produced unexpected unbound evidence files")
    result = _seal_reserved(
        output=output,
        context=context,
        submission_id=submission_id,
        id_receipt=id_receipt,
        metadata_source=metadata_source,
        replay_source=replay,
        seat_log_sources={0: seat_0, 1: seat_1},
        acquisition={
            "mode": "kaggle_api_single_fetch",
            "network_calls": 4,
            "retry_allowed": False,
            "episode_id": selected_episode,
        },
    )
    # The API staging directory is retained as forensic source material, but
    # no file under the final evidence directory remains mutable.
    for child in staging.iterdir():
        os.chmod(child, 0o444)
    os.chmod(staging, 0o555)
    return result


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-member", required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--r225-contract", type=Path, default=support.CANONICAL_R225_PATH)
    parser.add_argument("--r236-contract", type=Path, default=support.CANONICAL_R236_PATH)
    parser.add_argument("--submission-id-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    _common_arguments(materialize)
    materialize.add_argument("--episode-metadata", type=Path, required=True)
    materialize.add_argument("--replay", type=Path, required=True)
    materialize.add_argument("--seat-0-logs", type=Path, required=True)
    materialize.add_argument("--seat-1-logs", type=Path, required=True)
    download = commands.add_parser("download")
    _common_arguments(download)
    download.add_argument("--episode-id", type=int, required=True)
    return parser


def main() -> int:
    args = _arguments().parse_args()
    shared = {
        "archive_path": args.archive,
        "manifest_path": args.manifest,
        "manifest_member": args.manifest_member,
        "binding_path": args.binding,
        "r225_contract": args.r225_contract,
        "r236_contract": args.r236_contract,
        "submission_id_receipt_path": args.submission_id_receipt,
        "output_path": args.output,
    }
    try:
        if args.command == "materialize":
            result = materialize_evidence(
                **shared,
                episode_metadata_path=args.episode_metadata,
                replay_path=args.replay,
                seat_0_log_path=args.seat_0_logs,
                seat_1_log_path=args.seat_1_logs,
            )
        else:
            # Do not even import/authenticate a remote client until the exact
            # package, immutable binding, consumed authority, and resolved ID
            # have all passed locally.  ``download_evidence_once`` repeats
            # this validation immediately before its four singular calls.
            _context_and_submission_id(
                archive_path=args.archive,
                manifest_path=args.manifest,
                manifest_member=args.manifest_member,
                binding_path=args.binding,
                r225_contract=args.r225_contract,
                r236_contract=args.r236_contract,
                submission_id_receipt_path=args.submission_id_receipt,
            )
            _integer(args.episode_id, label="selected episode ID")
            # Importing/authenticating the client is intentionally delayed
            # until an operator explicitly chooses the download subcommand
            # and that offline provenance preflight has passed.
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            result = download_evidence_once(api, **shared, episode_id=args.episode_id)
    except (OSError, R235EvidenceError, support.R235SupportError, ValueError) as exc:
        print(f"R235 evidence BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result.get(key) for key in ("path", "sha256", "status")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
