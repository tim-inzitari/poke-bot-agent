#!/usr/bin/env python3
"""Apply a staged trainer unit at the next committed iteration boundary.

The trainer immediately begins the next collection after committing its
ledger, so this helper polls quickly, stops that just-started wave, quarantines
only its uncommitted partial shard, and resumes from the same committed model.
It lives outside the deployed trainer source tree to preserve the immutable
source fingerprint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--service", required=True)
    parser.add_argument("--target-next-iteration", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()

    loop_path = args.run_dir / "loop_state.json"
    target = int(args.target_next_iteration)
    atomic_json(
        args.status,
        {
            "state": "armed",
            "target_next_iteration": target,
            "service": args.service,
            "updated_at": time.time(),
        },
    )
    while True:
        try:
            loop = json.loads(loop_path.read_text())
            next_iteration = int(loop.get("next_iteration", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            next_iteration = -1
        if next_iteration >= target:
            break
        time.sleep(max(0.05, float(args.poll_seconds)))

    atomic_json(
        args.status,
        {
            "state": "boundary_seen",
            "target_next_iteration": target,
            "observed_next_iteration": next_iteration,
            "updated_at": time.time(),
        },
    )
    subprocess.run(
        ["systemctl", "--user", "stop", args.service],
        check=True,
        timeout=60,
    )

    shard = args.run_dir / "shards" / f"iter_{target:05d}.jsonl"
    quarantined: str | None = None
    if shard.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        partial = shard.with_name(f"{shard.name}.pre_adaptive_dispatch.{stamp}.partial")
        os.replace(shard, partial)
        quarantined = str(partial)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=30)
    subprocess.run(
        ["systemctl", "--user", "start", args.service],
        check=True,
        timeout=60,
    )
    atomic_json(
        args.status,
        {
            "state": "applied",
            "target_next_iteration": target,
            "observed_next_iteration": next_iteration,
            "quarantined_partial_shard": quarantined,
            "updated_at": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
