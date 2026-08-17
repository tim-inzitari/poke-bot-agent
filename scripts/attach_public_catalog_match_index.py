#!/usr/bin/env python3
"""Attach a checksum-bound public match index to an older exact deck catalog."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = "poke_bot.public_deck_archetype_catalog/v1"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _dates(window: dict) -> list[str]:
    first = date.fromisoformat(str(window.get("start") or ""))
    last = date.fromisoformat(str(window.get("end") or ""))
    if last < first:
        raise RuntimeError("catalog source window is invalid")
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def attach_index(
    *,
    base_catalog_path: Path,
    index_catalog_path: Path,
    target_id: str,
) -> dict:
    base = json.loads(base_catalog_path.read_text(encoding="utf-8"))
    index_catalog = json.loads(index_catalog_path.read_text(encoding="utf-8"))
    auxiliary = dict(
        (index_catalog.get("auxiliary_source_match_indexes") or {}).get(
            target_id
        )
        or {}
    )
    dates = _dates(dict(base.get("source_window") or {}))
    fingerprints = sorted(str(value) for value in base.get("deck_fingerprints") or ())
    raw_facts = auxiliary.get("source_match_facts")
    archive_evidence = {
        str(row.get("date") or ""): dict(row)
        for row in (index_catalog.get("source_archives") or ())
        if isinstance(row, dict)
    }
    if (
        base.get("schema") != SCHEMA
        or base.get("specialist_id") != target_id
        or int((base.get("source_window") or {}).get("days") or 0) != len(dates)
        or sorted(base.get("observed_by_day") or {}) != dates
        or not fingerprints
        or index_catalog.get("schema") != SCHEMA
        or index_catalog.get("specialist_id") == target_id
        or sorted(auxiliary.get("deck_fingerprints") or ()) != fingerprints
        or not isinstance(raw_facts, list)
        or _digest(raw_facts) != auxiliary.get("source_match_facts_sha256")
        or any(day not in archive_evidence for day in dates)
    ):
        raise RuntimeError("public catalog match-index attachment identity changed")

    allowed_dates = set(dates)
    facts = [
        row
        for row in raw_facts
        if isinstance(row, list)
        and len(row) == 4
        and str(row[0]) in allowed_dates
    ]
    by_day = {day: 0 for day in dates}
    for row in facts:
        day, episode_id, seat, fingerprint = row
        if (
            not str(episode_id)
            or int(seat) not in (0, 1)
            or str(fingerprint) not in fingerprints
        ):
            raise RuntimeError("public catalog match-index row changed")
        by_day[str(day)] += 1
    if (
        by_day != {
            day: int(count)
            for day, count in (base.get("observed_by_day") or {}).items()
        }
        or len(facts) != int(base.get("source_match_rows") or -1)
    ):
        raise RuntimeError("public catalog match-index counts do not reproduce base")

    revised = dict(base)
    revised["source_match_facts"] = facts
    revised["source_match_facts_sha256"] = _digest(facts)
    revised["source_archives"] = [archive_evidence[day] for day in dates]
    revised["source_archives_sha256"] = _digest(revised["source_archives"])
    revised["match_index_attachment"] = {
        "schema": "poke_bot.public_deck_match_index_attachment/v1",
        "source_catalog": str(index_catalog_path),
        "source_catalog_sha256": _sha256_file(index_catalog_path),
        "target_id": target_id,
        "rows": len(facts),
    }
    return revised


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--index-catalog", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = attach_index(
        base_catalog_path=args.base_catalog.resolve(),
        index_catalog_path=args.index_catalog.resolve(),
        target_id=str(args.target_id).strip(),
    )
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    _atomic_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
