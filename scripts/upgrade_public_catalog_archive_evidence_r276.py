#!/usr/bin/env python3
"""Create an immutable archive-attested derivative of one legacy catalog.

The legacy Format-6 Teal catalog predates the ``source_archives`` field now
required by the causal-router fitter.  This tool preserves every existing
identity and label byte-for-byte at the JSON-value level and adds only exact
per-day replay-archive identities and an explicit provenance marker.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


CATALOG_SCHEMA = "poke_bot.public_deck_archetype_catalog/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def attach_archive_evidence(catalog: dict, archive_root: Path) -> dict:
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise RuntimeError("legacy public catalog schema changed")
    window = dict(catalog.get("source_window") or {})
    try:
        start = date.fromisoformat(str(window["start"]))
        end = date.fromisoformat(str(window["end"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("legacy public catalog window is invalid") from exc
    days = (end - start).days + 1
    if days <= 0 or int(window.get("days") or 0) != days:
        raise RuntimeError("legacy public catalog day count changed")
    observed = dict(catalog.get("observed_by_day") or {})
    expected_dates = [
        (start + timedelta(days=offset)).isoformat() for offset in range(days)
    ]
    if sorted(observed) != expected_dates:
        raise RuntimeError("legacy public catalog observed-day coverage changed")
    if catalog.get("source_archives"):
        raise RuntimeError("legacy public catalog already has archive evidence")

    evidence: list[dict[str, object]] = []
    for day in expected_dates:
        archive = archive_root / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if archive.is_symlink() or not archive.is_file():
            raise RuntimeError(f"immutable replay archive missing: {archive}")
        with zipfile.ZipFile(archive, "r") as stream:
            members = [
                row
                for row in stream.infolist()
                if not row.is_dir() and row.filename.endswith(".json")
            ]
        if not members:
            raise RuntimeError(f"immutable replay archive has no JSON members: {archive}")
        evidence.append(
            {
                "date": day,
                "archive": archive.name,
                "archive_sha256": _sha256(archive),
                "manifest_rows": len(members),
                "json_replays": len(members),
            }
        )

    result = json.loads(json.dumps(catalog))
    result["source_archives"] = evidence
    result["archive_evidence_upgrade"] = {
        "revision": 276,
        "operation": "add_exact_daily_archive_identities_only",
        "legacy_catalog_sha256": None,
        "labels_actions_outcomes_or_deck_rules_changed": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("output catalog must be create-only")
    if args.catalog.is_symlink() or not args.catalog.is_file():
        raise RuntimeError("legacy catalog must be a regular non-symlink file")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise RuntimeError("legacy catalog must contain a JSON object")
    result = attach_archive_evidence(catalog, args.archive_root)
    result["archive_evidence_upgrade"]["legacy_catalog_sha256"] = _sha256(
        args.catalog
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(result))
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": _sha256(args.output),
                "source_sha256": _sha256(args.catalog),
                "archive_days": len(result["source_archives"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
