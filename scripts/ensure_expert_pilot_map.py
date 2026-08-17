#!/usr/bin/env python3
"""Reuse a verified pilot map or run the archive extraction once.

The Crustle handoff may receive its map from the checksum-pinned Elmo archive
host.  Do not force the trainer to possess every multi-gigabyte raw archive
when that immutable map is already present and bound to the exact targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--pilot-map", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    target_digest = digest(args.targets.resolve())
    if args.pilot_map.is_file():
        value = json.loads(args.pilot_map.read_text(encoding="utf-8"))
        if (
            value.get("schema") == "poke_bot.expert_pilot_map/v1"
            and value.get("targets_sha256") == target_digest
            and int(value.get("unverifiable_rows", -1)) == 0
            and len(value.get("source_archives") or []) == 31
        ):
            print(json.dumps({"status": "reused", "targets_sha256": target_digest}))
            return 0
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/materialize_expert_pilot_importance.py"),
        "extract",
        "--targets", str(args.targets.resolve()),
        "--archive-dir", str(args.archive_dir.resolve()),
        "--output", str(args.pilot_map.resolve()),
        "--workers", str(args.workers),
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
