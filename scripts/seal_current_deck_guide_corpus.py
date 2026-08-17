#!/usr/bin/env python3
"""Seal a protected expert corpus as a checksum-bound current-deck guide."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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


def _atomic(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable guide receipt differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--guide-version", required=True)
    parser.add_argument("--expected-date", action="append", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    pointer_path = root / "PROTECTED_EXPERT_CORPUS.json"
    manifest = _read(manifest_path)
    pointer = _read(pointer_path)
    totals = dict(manifest.get("totals") or {})
    coverage = dict(totals.get("target_coverage") or {})
    expected_dates = tuple(dict.fromkeys(str(value) for value in args.expected_date))
    manifest_dates = tuple(
        dict.fromkeys(
            str(value)
            for row in (manifest.get("shards") or ())
            for value in (row.get("source_dates") or ())
        )
    )
    guide_rows = int(coverage.get("guide_rows") or 0)
    decisions = int(totals.get("decisions_kept") or 0)
    if (
        manifest.get("format") != "pokebot-bootstrap-feature-manifest"
        or pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
        or (pointer.get("selection") or {}).get("value") != args.specialist_id
        or pointer.get("manifest_sha256") != _sha256(manifest_path)
        or set(manifest_dates) != set(expected_dates)
        or guide_rows <= 0
        or decisions <= 0
    ):
        raise RuntimeError("protected current-deck guide corpus identity failed")
    shards = []
    for row in manifest.get("shards") or ():
        path = root / str(row.get("path") or "")
        if not path.is_file() or row.get("sha256") != _sha256(path):
            raise RuntimeError(f"guide shard checksum changed: {path}")
        shards.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "source_dates": list(row.get("source_dates") or ()),
                "records": int((row.get("stats") or {}).get("records_kept") or 0),
                "decisions": int(
                    (row.get("stats") or {}).get("decisions_kept") or 0
                ),
                "guide_rows": int(
                    (
                        (row.get("stats") or {}).get("target_coverage") or {}
                    ).get("guide_rows")
                    or 0
                ),
            }
        )
    if not shards:
        raise RuntimeError("protected current-deck guide corpus has no shards")
    receipt_path = root / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    existing_receipt = _read(receipt_path) if receipt_path.is_file() else {}
    receipt = {
        "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
        "status": "ready",
        "created_at_utc": existing_receipt.get("created_at_utc")
        or datetime.now(timezone.utc).isoformat(),
        "specialist_id": args.specialist_id,
        "guide_version": args.guide_version,
        "date_start": min(expected_dates),
        "date_end": max(expected_dates),
        "dates": list(expected_dates),
        "days": len(expected_dates),
        "records": int(totals.get("records_kept") or 0),
        "decisions": decisions,
        "guide_rows": guide_rows,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "protected_pointer": str(pointer_path),
        "protected_pointer_sha256": _sha256(pointer_path),
        "pointer_manifest_sha256": pointer.get("manifest_sha256"),
        "daily_shards": shards,
        "active_training_modified": False,
    }
    _atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
