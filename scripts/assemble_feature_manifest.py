#!/usr/bin/env python
"""Verify compact feature shards and atomically assemble a training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.feature_shards import (
    COMPACT_MODE,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _verified_digest(
    shard: Path,
    sidecar: Path,
    metadata: dict[str, Any],
    verified_dir: Path | None,
) -> str:
    """Hash once on arrival and safely reuse that result during final assembly."""
    expected = str(metadata.get("sha256") or "")
    stat = shard.stat()
    sidecar_digest = _sha256(sidecar)
    cache_path = (
        verified_dir / f"{sidecar.name}.verified.json"
        if verified_dir is not None
        else None
    )
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if (
            cached.get("path") == shard.name
            and cached.get("sha256") == expected
            and cached.get("sidecar_sha256") == sidecar_digest
            and int(cached.get("bytes", -1)) == stat.st_size
            and int(cached.get("ctime_ns", -1)) == stat.st_ctime_ns
        ):
            return expected

    digest = _sha256(shard)
    if digest != expected:
        raise SystemExit(f"digest mismatch: {shard}")
    if cache_path is not None:
        _atomic_json(
            cache_path,
            {
                "path": shard.name,
                "sha256": digest,
                "sidecar_sha256": sidecar_digest,
                "bytes": stat.st_size,
                "ctime_ns": stat.st_ctime_ns,
            },
        )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-date", action="append", required=True)
    parser.add_argument(
        "--only-date",
        action="append",
        default=[],
        help="Verify/assemble only these dates (used for per-shard pipelining).",
    )
    parser.add_argument(
        "--verified-dir",
        type=Path,
        default=None,
        help="Cache post-transfer shard digests for the final manifest pass.",
    )
    parser.add_argument("--min-free-gib", type=float, default=25.0)
    args = parser.parse_args()

    staging = args.staging_dir.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging).free
    minimum = int(args.min_free_gib * (1024**3))
    if free < minimum:
        raise SystemExit(
            f"free-space guard failed: {free / 1024**3:.1f} GiB < "
            f"{args.min_free_gib:.1f} GiB"
        )

    expected_dates = list(args.expected_date)
    if len(expected_dates) != len(set(expected_dates)):
        raise SystemExit("duplicate --expected-date values")
    only_dates = set(args.only_date)
    if not only_dates.issubset(set(expected_dates)):
        raise SystemExit("--only-date must be included in --expected-date")
    verified_dir = args.verified_dir.resolve() if args.verified_dir else None
    if verified_dir is not None:
        verified_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    actual_dates: list[str] = []
    for sidecar in sorted(staging.glob("*.features.json")):
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("format") != SHARD_FORMAT:
            raise SystemExit(f"invalid shard format in {sidecar}")
        if int(metadata.get("format_version", -1)) != SHARD_FORMAT_VERSION:
            raise SystemExit(f"invalid shard version in {sidecar}")
        if metadata.get("compact_mode") != COMPACT_MODE:
            raise SystemExit(f"invalid compact mode in {sidecar}")
        shard = staging / str(metadata.get("path") or "")
        if not shard.is_file():
            raise SystemExit(f"missing shard for {sidecar}: {shard}")
        dates = [str(value) for value in metadata.get("source_dates") or []]
        if only_dates and not only_dates.intersection(dates):
            continue
        if only_dates and not set(dates).issubset(only_dates):
            raise SystemExit(f"selected shard contains unexpected dates: {sidecar}")
        digest = _verified_digest(shard, sidecar, metadata, verified_dir)
        overlap = set(actual_dates).intersection(dates)
        if overlap:
            raise SystemExit(f"overlapping shard dates: {sorted(overlap)}")
        actual_dates.extend(dates)
        stats = dict(metadata.get("stats") or {})
        if int(stats.get("records_kept", 0)) <= 0:
            raise SystemExit(f"empty feature shard: {shard}")
        total = int(stats.get("records_total", 0))
        kept = int(stats.get("records_kept", 0))
        if total <= 0 or kept / total < 0.98:
            raise SystemExit(
                f"usable-record gate failed for {shard}: kept={kept} total={total}"
            )
        rows.append(
            {
                "path": shard.name,
                "sha256": digest,
                "bytes": shard.stat().st_size,
                "source_dates": dates,
                "stats": stats,
            }
        )

    if sorted(actual_dates) != sorted(expected_dates):
        raise SystemExit(
            f"date coverage mismatch: expected={sorted(expected_dates)} "
            f"actual={sorted(actual_dates)}"
        )
    if not rows:
        raise SystemExit("no feature shard sidecars found")
    rows.sort(key=lambda row: min(row["source_dates"]))
    payload = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "date_start": min(actual_dates),
        "date_end": max(actual_dates),
        "dates": sorted(actual_dates),
        "shards": rows,
        "totals": {
            "bytes": sum(int(row["bytes"]) for row in rows),
            "records_total": sum(
                int(row["stats"].get("records_total", 0)) for row in rows
            ),
            "records_kept": sum(
                int(row["stats"].get("records_kept", 0)) for row in rows
            ),
            "decisions_kept": sum(
                int(row["stats"].get("decisions_kept", 0)) for row in rows
            ),
        },
    }
    out = args.out.resolve()
    _atomic_json(out, payload)
    print(json.dumps(payload["totals"], sort_keys=True), flush=True)
    print(f"manifest={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
