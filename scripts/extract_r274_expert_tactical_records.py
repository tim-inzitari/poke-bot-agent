#!/usr/bin/env python3
"""Extract a bounded checksum-backed Alakazam record set for r274 shadow labels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from poke_bot.own_deck_rollout_store import (
    _iter_archive_native_records,
    _load_classifier,
    load_source_window,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--versioned-receipt-sha256", required=True)
    parser.add_argument("--classifier-mix", type=Path, required=True)
    parser.add_argument("--classifier-representatives", type=Path, required=True)
    parser.add_argument("--card-csv", type=Path, required=True)
    parser.add_argument("--records", type=int, default=512)
    parser.add_argument(
        "--source-day",
        action="append",
        default=[],
        help=(
            "Restrict extraction to one or more exact source days. This is used "
            "to keep tactical training labels source-disjoint from heldout days."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.records < 64:
        raise SystemExit("record bound must be at least 64")
    window = load_source_window(
        args.source_manifest,
        expected_manifest_sha256=args.source_manifest_sha256,
        expected_versioned_receipt_sha256=args.versioned_receipt_sha256,
        verify_archives=True,
    )
    classifier = _load_classifier(
        args.classifier_mix,
        args.classifier_representatives,
        args.card_csv,
    )
    selected: list[dict] = []
    source_days: dict[str, int] = {}
    requested_days = tuple(dict.fromkeys(str(day) for day in args.source_day))
    available_days = {str(archive.day) for archive in window.archives}
    missing_days = sorted(set(requested_days) - available_days)
    if missing_days:
        raise SystemExit(f"requested source days are absent: {missing_days}")
    archives = tuple(
        archive
        for archive in reversed(window.archives)
        if not requested_days or str(archive.day) in requested_days
    )
    for archive in archives:
        for record in _iter_archive_native_records(archive, classifier):
            selected.append(record)
            source_days[archive.day] = source_days.get(archive.day, 0) + 1
            if len(selected) >= args.records:
                break
        if len(selected) >= args.records:
            break
    if len(selected) != args.records:
        raise SystemExit(f"only extracted {len(selected)}/{args.records} records")
    output = args.output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"immutable output already exists: {output}")
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            for record in selected:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema": "poke_bot.r274_expert_tactical_record_extract/v1",
        "mode": "shadow_training_input_only",
        "evaluation_or_kaggle_replay": False,
        "source_manifest": {
            "path": str(window.manifest_path),
            "sha256": window.manifest_sha256,
        },
        "source_days": dict(sorted(source_days.items())),
        "requested_source_days": list(requested_days),
        "records": len(selected),
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
