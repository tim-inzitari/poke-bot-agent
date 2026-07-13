#!/usr/bin/env python3
"""Drop bulky observation blobs from rollout JSONL; training uses precomputed features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DROP_KEYS = ("observation", "next_observation")


def strip_file(path: Path, *, dry_run: bool = False) -> tuple[int, int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        return 0, 0, 0

    before_bytes = path.stat().st_size
    rows = 0
    changed = 0
    tmp = path.with_suffix(path.suffix + ".strip.tmp")

    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if any(key in row for key in DROP_KEYS):
                row = {key: value for key, value in row.items() if key not in DROP_KEYS}
                changed += 1
            dst.write(json.dumps(row, separators=(",", ":")) + "\n")

    after_bytes = tmp.stat().st_size
    if dry_run:
        tmp.unlink(missing_ok=True)
        return before_bytes, after_bytes, changed

    tmp.replace(path)
    return before_bytes, after_bytes, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_before = 0
    total_after = 0
    for path in args.paths:
        before, after, changed = strip_file(path.resolve(), dry_run=args.dry_run)
        if before == 0:
            continue
        total_before += before
        total_after += after
        label = "would strip" if args.dry_run else "stripped"
        print(
            f"{label} {path}: {before / 1e9:.2f} GiB -> {after / 1e9:.2f} GiB "
            f"({changed:,} rows with observations removed)"
        )
    if total_before:
        saved = total_before - total_after
        print(f"total saved: {saved / 1e9:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
