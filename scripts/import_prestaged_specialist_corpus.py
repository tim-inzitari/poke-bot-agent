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
    root: Path, *, specialist_id: str, guide_version: str
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
        "decisions": int(ready["decisions"]),
        "guide_rows": int(ready["guide_rows"]),
        "manifest_sha256": _sha256(manifest_path),
        "protected_pointer_sha256": _sha256(pointer_path),
        "ready_receipt_sha256": _sha256(ready_path),
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
    args = parser.parse_args()
    if args.bwlimit_kib <= 0:
        raise ValueError("bandwidth limit must be positive")

    destination = args.destination.resolve()
    if destination.is_dir():
        identity = _validate(
            destination,
            specialist_id=args.specialist_id,
            guide_version=args.guide_version,
        )
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
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
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
        "active_training_modified": False,
    }
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
