#!/usr/bin/env python3
"""Atomically promote one checksum-bound current-deck guide corpus on Inzi.

The source is Elmo's already-finalized latest-20 guide directory, exposed
through Inzi's read-only/user-mounted SMB path. Files land in a staging
directory under a bandwidth limit. Only after every receipt and checksum
validates does the script atomically replace the specialist's ordinary corpus
directory. The previous directory remains as a rollback artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


READY_SCHEMA = "poke_bot.current_deck_guide_corpus_ready/v1"
POINTER_SCHEMA = "poke_bot.pinned_expert_corpus/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate(
    root: Path,
    *,
    specialist_id: str,
    verify_shards: bool = True,
) -> dict[str, Any]:
    ready_path = root / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    pointer_path = root / "PROTECTED_EXPERT_CORPUS.json"
    ready = read_json(ready_path)
    pointer = read_json(pointer_path)
    manifest_path = root / str(pointer.get("manifest") or "")
    manifest = read_json(manifest_path)
    totals = dict(manifest.get("totals") or {})
    coverage = dict(totals.get("target_coverage") or {})
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("status") != "ready"
        or ready.get("specialist_id") != specialist_id
        or pointer.get("schema") != POINTER_SCHEMA
        or pointer.get("protected") is not True
        or (pointer.get("selection") or {}).get("value") != specialist_id
        or sha256(manifest_path) != pointer.get("manifest_sha256")
        or sha256(manifest_path) != ready.get("manifest_sha256")
        or sha256(pointer_path) != ready.get("protected_pointer_sha256")
        or int(totals.get("decisions_kept") or 0)
        != int(ready.get("decisions") or 0)
        or int(coverage.get("guide_rows") or 0)
        != int(ready.get("guide_rows") or 0)
        or int(ready.get("guide_rows") or 0) <= 0
        or int(ready.get("days") or 0) != 20
        or len(ready.get("daily_shards") or ()) != 20
    ):
        raise RuntimeError("current-deck guide corpus identity is invalid")
    if verify_shards:
        for row in ready["daily_shards"]:
            date = str(row.get("date") or "")
            shard = root / f"{specialist_id}-{date}.features"
            if (
                not shard.is_file()
                or sha256(shard) != str(row.get("sha256") or "")
            ):
                raise RuntimeError(f"guide feature checksum failed: {date}")
    return {
        "specialist_id": specialist_id,
        "guide_version": ready["guide_version"],
        "dates": list(ready["dates"]),
        "decisions": int(ready["decisions"]),
        "guide_rows": int(ready["guide_rows"]),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "protected_pointer": str(pointer_path),
        "protected_pointer_sha256": sha256(pointer_path),
        "ready_receipt": str(ready_path),
        "ready_receipt_sha256": sha256(ready_path),
    }


def identity_signature(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "specialist_id",
            "guide_version",
            "dates",
            "decisions",
            "guide_rows",
            "manifest_sha256",
            "protected_pointer_sha256",
            "ready_receipt_sha256",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--bwlimit-kib", type=int, default=4_000)
    args = parser.parse_args()
    if args.bwlimit_kib <= 0:
        raise ValueError("--bwlimit-kib must be positive")

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    # Elmo's immutable finalizer has already checksummed every daily shard.
    # Re-reading all source shards across the constrained LAN before copying
    # would double network traffic. Recheck the small source identity here,
    # then verify every landed shard locally before promotion.
    source_identity = validate(
        source,
        specialist_id=args.specialist_id,
        verify_shards=False,
    )
    if destination.is_dir():
        try:
            current_identity = validate(
                destination,
                specialist_id=args.specialist_id,
                verify_shards=False,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            current_identity = None
        if (
            current_identity is not None
            and identity_signature(current_identity)
            == identity_signature(source_identity)
        ):
            atomic_json(
                args.state,
                {
                    "schema": (
                        "poke_bot.current_deck_guide_corpus_promotion/v1"
                    ),
                    "status": "ready",
                    "specialist_id": args.specialist_id,
                    "source": str(source),
                    "destination": str(destination),
                    "identity": current_identity,
                    "already_promoted": True,
                    "completed_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "active_training_modified": False,
                },
            )
            return 0
    staging = destination.with_name(
        f".{destination.name}.guide-{source_identity['manifest_sha256'][-12:]}.partial"
    )
    backup = destination.with_name(
        f".{destination.name}.unguided-{source_identity['manifest_sha256'][-12:]}"
    )
    # The manifest digest is part of the staging name, so a surviving partial
    # directory is safe to resume and belongs to this exact source identity.
    # Keeping it makes systemd retries continue the transfer rather than
    # discarding already copied gigabytes.
    staging.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    atomic_json(
        args.state,
        {
            "schema": "poke_bot.current_deck_guide_corpus_promotion/v1",
            "status": "syncing",
            "specialist_id": args.specialist_id,
            "source": str(source),
            "destination": str(destination),
            "staging": str(staging),
            "bandwidth_limit_kib_per_second": args.bwlimit_kib,
            "source_identity": source_identity,
            "started_at_utc": started,
            "active_training_modified": False,
        },
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            # Full landed-shard SHA-256 validation below is authoritative.
            # Plain --append also works with macOS' older rsync and safely
            # resumes this manifest-digest-named partial directory.
            "--append",
            f"--bwlimit={args.bwlimit_kib}",
            f"{source}/",
            f"{staging}/",
        ],
        check=True,
    )
    staged_identity = validate(staging, specialist_id=args.specialist_id)
    if identity_signature(staged_identity) != identity_signature(
        source_identity
    ):
        raise RuntimeError("staged guide corpus identity differs from source")

    if backup.exists():
        raise RuntimeError(f"rollback path already exists: {backup}")
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    promoted_identity = validate(
        destination, specialist_id=args.specialist_id
    )
    atomic_json(
        args.state,
        {
            "schema": "poke_bot.current_deck_guide_corpus_promotion/v1",
            "status": "ready",
            "specialist_id": args.specialist_id,
            "source": str(source),
            "destination": str(destination),
            "rollback_directory": str(backup),
            "bandwidth_limit_kib_per_second": args.bwlimit_kib,
            "identity": promoted_identity,
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "active_training_modified": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
