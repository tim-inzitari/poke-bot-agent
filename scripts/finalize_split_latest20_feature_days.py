#!/usr/bin/env python3
"""Join split daily-feature workers before releasing the corpus finalizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import time
from typing import Any


SCHEMA = "poke_bot.expert_missing_daily_features/v1"
SHARD_FORMAT = "pokebot-bootstrap-feature-shard"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.aggregate.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(root: Path, days: list[str]) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    for day in days:
        status_path = root / "status" / f"{day}.json"
        if not status_path.is_file():
            return None
        status = read_json(status_path)
        if status.get("state") == "failed":
            raise RuntimeError(f"daily feature job failed: {day}")
        completed = status.get("completed")
        if status.get("state") != "complete":
            return None
        if (
            not isinstance(completed, list)
            or len(completed) != 1
            or completed[0].get("date") != day
        ):
            raise RuntimeError(f"daily completion receipt is malformed: {day}")
        rows.append(dict(completed[0]))
    return rows


def validate_completed_shards(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    required_dataset_schema: int,
    required_expanded_target_schema: str,
    required_expanded_target_digest: str,
) -> None:
    """Reject stale worker outputs before they can release the finalizer."""
    for row in rows:
        output = Path(str(row.get("output") or ""))
        shard = root / output.name
        if not shard.is_file():
            raise RuntimeError(f"daily feature shard is missing: {shard}")
        with shard.open("rb") as stream:
            header = pickle.load(stream)
        expanded = header.get("expanded_strategic_targets") if isinstance(header, dict) else None
        if (
            not isinstance(header, dict)
            or header.get("format") != SHARD_FORMAT
            or int(header.get("dataset_schema", -1)) != required_dataset_schema
            or not isinstance(expanded, dict)
            or expanded.get("schema") != required_expanded_target_schema
            or expanded.get("digest") != required_expanded_target_digest
        ):
            raise RuntimeError(
                "daily feature shard has stale or incompatible strategic schema: "
                f"{shard}"
            )


def container_state(name: str) -> tuple[str, int]:
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "docker",
            "inspect",
            name,
            "--format",
            "{{.State.Status}} {{.State.ExitCode}}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect managed feature container {name}")
    state, code = result.stdout.strip().split()
    return state, int(code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--day", action="append", required=True)
    parser.add_argument("--worker-container", action="append", required=True)
    parser.add_argument("--finalizer-container", required=True)
    parser.add_argument("--required-dataset-schema", type=int, required=True)
    parser.add_argument("--required-expanded-target-schema", required=True)
    parser.add_argument("--required-expanded-target-digest", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600)
    args = parser.parse_args()

    days = list(args.day)
    if len(days) != len(set(days)):
        raise RuntimeError("aggregate day list contains duplicates")
    deadline = time.monotonic() + args.timeout_seconds
    rows: list[dict[str, Any]] | None = None
    while rows is None:
        rows = collect(args.root, days)
        all_workers_exited = True
        for name in args.worker_container:
            state, code = container_state(name)
            if state in {"exited", "dead"} and code != 0:
                raise RuntimeError(
                    f"managed feature container failed: {name} exit={code}"
                )
            if state != "exited" or code != 0:
                all_workers_exited = False
        if rows is not None and all_workers_exited:
            break
        rows = None
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for split daily features")
        time.sleep(max(1.0, args.poll_seconds))

    validate_completed_shards(
        args.root,
        rows,
        required_dataset_schema=args.required_dataset_schema,
        required_expanded_target_schema=args.required_expanded_target_schema,
        required_expanded_target_digest=args.required_expanded_target_digest,
    )

    payload = {
        "schema": SCHEMA,
        "status": "ready",
        "days": days,
        "completed": rows,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "corpus_location": "elmo_only",
        "split_worker_barrier": True,
        "worker_containers": list(args.worker_container),
    }
    ready = args.root / "MISSING_DAYS_READY.json"
    atomic_json(ready, payload)
    status_path = args.root / "status" / "split-worker-aggregate.json"
    atomic_json(
        status_path,
        {
            "schema": "poke_bot.split_daily_feature_barrier/v1",
            "status": "ready_finalizer_released",
            "days": days,
            "ready_receipt": str(ready),
            "finalizer_container": args.finalizer_container,
            "completed_at": payload["completed_at"],
        },
    )
    subprocess.run(
        ["sudo", "-n", "docker", "start", args.finalizer_container],
        check=True,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
