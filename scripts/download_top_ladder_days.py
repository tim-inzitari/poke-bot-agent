#!/usr/bin/env python
"""Download a bounded date range of official daily ladder episode archives."""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.episodes_index import (
    EPISODES_RAW_DIR,
    download_daily_dataset,
    ensure_episodes_index,
    load_daily_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def _validate_archive(path: Path, expected: int) -> int:
    with zipfile.ZipFile(path, "r") as archive:
        count = sum(
            name.endswith(".json") and not name.endswith("/")
            for name in archive.namelist()
        )
    if count != expected:
        raise RuntimeError(
            f"archive {path.name} has {count} JSON episodes; expected {expected}"
        )
    return count


def main() -> int:
    args = _parse_args()
    if args.start_date > args.end_date:
        raise SystemExit("--start-date must not be after --end-date")
    rows = [
        row
        for row in load_daily_manifest(ensure_episodes_index())
        if args.start_date <= row.date <= args.end_date
    ]
    if not rows or rows[0].date != args.start_date or rows[-1].date != args.end_date:
        raise SystemExit(
            f"manifest does not cover {args.start_date}..{args.end_date}"
        )

    EPISODES_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        print(
            f"[download {index}/{len(rows)}] {row.date} "
            f"episodes={row.episode_count}",
            flush=True,
        )
        last_error: BaseException | None = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                path = Path(
                    download_daily_dataset(
                        row, root=EPISODES_RAW_DIR, unzip=False
                    )
                )
                count = _validate_archive(path, row.episode_count)
                print(
                    f"[ready] {row.date} path={path} episodes={count} "
                    f"bytes={path.stat().st_size}",
                    flush=True,
                )
                break
            except BaseException as exc:  # noqa: BLE001 - retry/report boundary
                last_error = exc
                print(
                    f"[retry] {row.date} attempt={attempt} error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt < max(1, args.retries):
                    time.sleep(min(30, 5 * attempt))
        else:
            raise RuntimeError(f"failed to prepare {row.date}") from last_error
    print(f"[complete] prepared {len(rows)} daily archives", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
