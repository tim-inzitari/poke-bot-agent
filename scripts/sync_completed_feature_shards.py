#!/usr/bin/env python3
"""Continuously return atomically published feature shards over the LAN."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _complete_local_shards(directory: Path) -> int:
    complete = 0
    for sidecar in directory.glob("*.features.json"):
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shard = directory / str(metadata.get("path") or "")
        if shard.is_file() and shard.stat().st_size > 0:
            complete += 1
    return complete


def _returned_remote_shards(directory: Path, remote: dict) -> int:
    """Count only shards named by this remote assignment, not unrelated local days."""
    complete = 0
    for row in remote.get("completed") or []:
        if not isinstance(row, dict):
            continue
        name = Path(str(row.get("output") or "")).name
        if not name.endswith(".features"):
            continue
        shard = directory / name
        sidecar = directory / f"{name}.json"
        if shard.is_file() and shard.stat().st_size > 0 and sidecar.is_file():
            complete += 1
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--remote-status", required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=43200.0)
    args = parser.parse_args()
    if args.expected_shards <= 0:
        raise SystemExit("--expected-shards must be positive")
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(args.timeout_seconds)

    while True:
        sync_result = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--include=*.features",
                "--include=*.features.json",
                "--exclude=*",
                f"{args.host}:{args.source_dir.rstrip('/')}/",
                f"{args.dest_dir}/",
            ],
            check=False,
        )
        status_result = subprocess.run(
            ["ssh", args.host, "cat", args.remote_status],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            remote = json.loads(status_result.stdout) if status_result.returncode == 0 else {}
        except json.JSONDecodeError:
            remote = {}
        if remote.get("state") == "failed":
            raise SystemExit(f"remote feature job failed: {remote.get('error')}")
        local = _returned_remote_shards(args.dest_dir, remote)
        print(
            json.dumps(
                {
                    "state": "complete" if local >= args.expected_shards else "syncing",
                    "local_shards": local,
                    "expected_shards": args.expected_shards,
                    "remote_state": remote.get("state", "unknown"),
                    "rsync_returncode": sync_result.returncode,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            sync_result.returncode == 0
            and local >= args.expected_shards
            and remote.get("state") == "complete"
        ):
            return 0
        if time.monotonic() >= deadline:
            raise SystemExit("timed out returning feature shards")
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
