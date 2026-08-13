#!/usr/bin/env python3
"""Restart only the managed r274 trainer after iter-1 collection seals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


SCHEMA = "poke_bot.alakazam_r274_iter1_terminal_fence_activation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _show(unit: str, *properties: str) -> dict[str, str]:
    command = ["systemctl", "--user", "show", unit]
    for name in properties:
        command.extend(("-p", name))
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def _write_create_only(path: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable activation receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--collection-receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--drop-in", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        return 0
    while not args.collection_receipt.is_file():
        time.sleep(args.poll_seconds)
    collection = json.loads(args.collection_receipt.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (
        collection.get("schema") != "poke_bot.completed_collection/v1"
        or int(collection.get("iteration", -1)) != 1
        or int(collection.get("source_games", -1)) != 8196
        or int(contract.get("latest_owner_clarification_revision", -1)) != 304
        or not bool(contract.get("iteration_1_terminal_submission_override"))
    ):
        raise RuntimeError("revision-304 sealed collection or contract is invalid")
    before = _show(args.unit, "MainPID", "ActiveState", "NRestarts")
    if before.get("ActiveState") != "active" or int(before.get("MainPID", "0")) <= 0:
        raise RuntimeError("managed r274 trainer is not active at sealed boundary")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "restart", args.unit], check=True)
    deadline = time.monotonic() + 60.0
    after: dict[str, str] = {}
    while time.monotonic() < deadline:
        after = _show(args.unit, "MainPID", "ActiveState", "NRestarts")
        if (
            after.get("ActiveState") == "active"
            and int(after.get("MainPID", "0")) > 0
            and after.get("MainPID") != before.get("MainPID")
        ):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("managed r274 trainer did not activate the r304 fence")
    _write_create_only(
        args.output,
        {
            "schema": SCHEMA,
            "status": "activated",
            "owner_revision": 304,
            "unit": args.unit,
            "collection_receipt": {
                "path": str(args.collection_receipt),
                "sha256": _sha256(args.collection_receipt),
            },
            "contract": {"path": str(args.contract), "sha256": _sha256(args.contract)},
            "drop_in": {"path": str(args.drop_in), "sha256": _sha256(args.drop_in)},
            "before": before,
            "after": after,
            "activated_at_utc": datetime.now(timezone.utc).isoformat(),
            "interactive_session_touched": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
