#!/usr/bin/env python3
"""Refresh and validate the latest ladder window for an expert boundary.

The downloader is resumable and publishes an atomic status/plan.  A daily zip
is considered ready only after its member count matches Kaggle's index and all
members are JSON files.  Downstream featurizers must consume only ``ready``
rows from the completed plan.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.episodes_index import (  # noqa: E402
    DailyDatasetEntry,
    download_daily_dataset,
)

SCHEMA = "poke_bot.expert_boundary_refresh/v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _entries(path: Path, days: int) -> list[DailyDatasetEntry]:
    rows: list[DailyDatasetEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = str(row.get("date") or "").strip()
            slug = str(row.get("daily_dataset_slug") or "").strip()
            count = int(row.get("episode_count") or 0)
            if not day or not slug or count <= 0 or not slug.endswith(day):
                raise ValueError(f"invalid episodes-index row: {row!r}")
            rows.append(
                DailyDatasetEntry(
                    date=day,
                    slug=slug,
                    url=str(row.get("daily_dataset_url") or ""),
                    episode_count=count,
                    total_bytes=int(row.get("total_bytes") or 0),
                    top_avg_score=float(row.get("top_avg_score") or 0),
                    median_avg_score=float(row.get("median_avg_score") or 0),
                )
            )
    rows.sort(key=lambda value: value.date)
    if len(rows) < days or len({row.date for row in rows}) != len(rows):
        raise ValueError("episodes index is short or contains duplicate dates")
    return rows[-days:]


def _validate_zip(path: Path, expected: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1024 * 1024:
        raise ValueError(f"daily archive is missing/truncated: {path}")
    with zipfile.ZipFile(path) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.endswith(".json")
        ]
        if len(members) != expected:
            raise ValueError(
                f"daily archive member count mismatch: {path} "
                f"expected={expected} actual={len(members)}"
            )
        if any(info.file_size <= 0 for info in members):
            raise ValueError(f"daily archive has an empty episode: {path}")
    return {"bytes": path.stat().st_size, "episodes": len(members)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--kaggle-bin", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    args = parser.parse_args()

    if args.days <= 0 or args.download_workers <= 0:
        raise SystemExit("--days and --download-workers must be positive")
    entries = _entries(args.index.resolve(), int(args.days))
    raw_dir = args.raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    status = args.status.resolve()
    status.parent.mkdir(parents=True, exist_ok=True)
    lock_path = status.with_suffix(status.suffix + ".lock")
    started = time.time()

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = {
            "schema": SCHEMA,
            "state": "downloading",
            "started_at": started,
            "window": {"start": entries[0].date, "end": entries[-1].date,
                       "days": len(entries)},
            "expected_episodes": sum(row.episode_count for row in entries),
            "ready": [],
        }
        _atomic_json(status, base)

        def fetch(entry: DailyDatasetEntry) -> dict[str, Any]:
            free = shutil.disk_usage(raw_dir).free
            if free < int(args.min_free_gib * 1024**3):
                raise RuntimeError(
                    f"free-space guard: {free / 1024**3:.1f} GiB "
                    f"< {args.min_free_gib:.1f} GiB"
                )
            path = download_daily_dataset(
                entry,
                root=raw_dir,
                kaggle_bin=args.kaggle_bin.resolve(),
                unzip=False,
            )
            checked = _validate_zip(path, entry.episode_count)
            return {
                "date": entry.date,
                "slug": entry.slug,
                "path": str(path),
                **checked,
                "state": "ready",
            }

        ready: list[dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=int(args.download_workers)) as pool:
                futures = {pool.submit(fetch, entry): entry for entry in entries}
                for future in as_completed(futures):
                    row = future.result()
                    ready.append(row)
                    ready.sort(key=lambda value: value["date"])
                    _atomic_json(
                        status,
                        {**base, "updated_at": time.time(), "ready": ready},
                    )
                    print(json.dumps(row, sort_keys=True), flush=True)
            finished = time.time()
            payload = {
                **base,
                "state": "ready_for_featurization",
                "updated_at": finished,
                "completed_at": finished,
                "elapsed_seconds": finished - started,
                "ready": ready,
                "matchup_head_policy": {
                    "required": True,
                    "routing": "causal_public_evidence_only",
                    "isolation": "one_route_one_adapter_base_frozen",
                },
            }
            _atomic_json(args.plan.resolve(), payload)
            _atomic_json(status, payload)
            return 0
        except BaseException as exc:
            _atomic_json(
                status,
                {**base, "state": "failed", "updated_at": time.time(),
                 "ready": ready, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
