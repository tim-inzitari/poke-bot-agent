#!/usr/bin/env python3
"""Atomically import one checksum-bound pre-staged specialist corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _validate(
    root: Path,
    *,
    specialist_id: str,
    guide_version: str,
    minimum_records: int,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    pointer_path = root / "PROTECTED_EXPERT_CORPUS.json"
    ready_path = root / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    manifest = _read(manifest_path)
    pointer = _read(pointer_path)
    ready = _read(ready_path)
    totals = dict(manifest.get("totals") or {})
    coverage = dict(totals.get("target_coverage") or {})
    if (
        manifest.get("format") != "pokebot-bootstrap-feature-manifest"
        or pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
        or (pointer.get("selection") or {}).get("value") != specialist_id
        or pointer.get("manifest_sha256") != _sha256(manifest_path)
        or ready.get("schema") != "poke_bot.current_deck_guide_corpus_ready/v1"
        or ready.get("status") != "ready"
        or ready.get("specialist_id") != specialist_id
        or ready.get("guide_version") != guide_version
        or ready.get("manifest_sha256") != _sha256(manifest_path)
        or ready.get("protected_pointer_sha256") != _sha256(pointer_path)
        or int(ready.get("decisions") or 0)
        != int(totals.get("decisions_kept") or 0)
        or int(ready.get("records") or 0)
        != int(totals.get("records_kept") or 0)
        or int(ready.get("records") or 0) < int(minimum_records)
        or int(ready.get("guide_rows") or 0)
        != int(coverage.get("guide_rows") or 0)
        or int(ready.get("decisions") or 0) <= 0
        or int(ready.get("guide_rows") or 0) <= 0
    ):
        raise RuntimeError("pre-staged specialist corpus identity failed")
    for row in manifest.get("shards") or ():
        path = root / str(row.get("path") or "")
        if not path.is_file() or row.get("sha256") != _sha256(path):
            raise RuntimeError(f"pre-staged shard checksum failed: {path}")
    return {
        "specialist_id": specialist_id,
        "guide_version": guide_version,
        "records": int(ready["records"]),
        "decisions": int(ready["decisions"]),
        "guide_rows": int(ready["guide_rows"]),
        "manifest_sha256": _sha256(manifest_path),
        "protected_pointer_sha256": _sha256(pointer_path),
        "ready_receipt_sha256": _sha256(ready_path),
    }


def _unavailable_placeholder(
    destination: Path, *, specialist_id: str
) -> dict[str, Any] | None:
    if not destination.is_dir():
        return None
    entries = list(destination.iterdir())
    if len(entries) != 1 or entries[0].name != "UNAVAILABLE_EXPERT_CORPUS.json":
        return None
    payload = _read(entries[0])
    if (
        payload.get("schema") != "poke_bot.specialist_expert_corpus_unavailable/v1"
        or payload.get("archetype") != specialist_id
        or payload.get("reason")
        != "no acting-seat records in the pinned source corpus"
    ):
        return None
    return {
        "path": str(entries[0]),
        "sha256": _sha256(entries[0]),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable import receipt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--guide-version", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bwlimit-kib", type=int, default=8_000)
    parser.add_argument("--minimum-records", type=int, default=0)
    args = parser.parse_args()
    if args.bwlimit_kib <= 0 or args.minimum_records < 0:
        raise ValueError("bandwidth and minimum-record contracts are invalid")

    destination = args.destination.resolve()
    placeholder = _unavailable_placeholder(
        destination,
        specialist_id=args.specialist_id,
    )
    if destination.is_dir() and placeholder is None:
        identity = _validate(
            destination,
            specialist_id=args.specialist_id,
            guide_version=args.guide_version,
            minimum_records=int(args.minimum_records),
        )
        replacement = {
            "replaced_unavailable_placeholder": False,
            "unavailable_placeholder_archive": None,
            "unavailable_placeholder_sha256": None,
        }
    else:
        staging = destination.with_name(
            f".{destination.name}.importing.{os.getpid()}"
        )
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    f"--bwlimit={int(args.bwlimit_kib)}",
                    f"{args.host}:{args.remote_root.rstrip('/')}/",
                    str(staging) + "/",
                ],
                check=True,
            )
            identity = _validate(
                staging,
                specialist_id=args.specialist_id,
                guide_version=args.guide_version,
                minimum_records=int(args.minimum_records),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            placeholder_archive = None
            if placeholder is not None:
                placeholder_archive = destination.with_name(
                    f".{destination.name}.unavailable-"
                    f"{str(placeholder['sha256']).removeprefix('sha256:')[:12]}"
                )
                if placeholder_archive.exists():
                    raise RuntimeError(
                        "unavailable placeholder archive already exists: "
                        f"{placeholder_archive}"
                    )
                os.replace(destination, placeholder_archive)
            try:
                os.replace(staging, destination)
            except BaseException:
                if (
                    placeholder_archive is not None
                    and placeholder_archive.exists()
                    and not destination.exists()
                ):
                    os.replace(placeholder_archive, destination)
                raise
            replacement = {
                "replaced_unavailable_placeholder": placeholder is not None,
                "unavailable_placeholder_archive": (
                    str(placeholder_archive)
                    if placeholder_archive is not None
                    else None
                ),
                "unavailable_placeholder_sha256": (
                    placeholder["sha256"] if placeholder is not None else None
                ),
            }
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    receipt_path = args.receipt.resolve()
    existing_receipt = _read(receipt_path) if receipt_path.is_file() else {}
    receipt = {
        "schema": "poke_bot.prestaged_specialist_corpus_import/v1",
        "status": "ready",
        "created_at_utc": existing_receipt.get("created_at_utc")
        or datetime.now(timezone.utc).isoformat(),
        "source_host": args.host,
        "source_root": args.remote_root,
        "destination": str(destination),
        **identity,
        **replacement,
        "active_training_modified": False,
    }
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
