#!/usr/bin/env python3
"""Restore heldout evidence paired with the checksum-exact rollback parent."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


SCHEMA = "poke_bot.marnie_family_rollback_heldout_evidence_repair/v1"


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def immutable_json(path: Path, payload: dict[str, Any]) -> None:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != body:
            raise RuntimeError(f"immutable output changed: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_inactive(service: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", service],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    state = result.stdout.strip() or "unknown"
    if state in {"active", "activating", "reloading"}:
        raise RuntimeError(f"managed trainer is still {state}")
    return state


def run(args: argparse.Namespace) -> dict[str, Any]:
    service_state = require_inactive(args.service)
    loop_path = args.loop_state.resolve()
    commit_path = args.rollback_parent_commit.resolve()
    rollback_path = args.rollback_receipt.resolve()
    migration_path = args.design_migration.resolve()
    loop = read_json(loop_path)
    commit = read_json(commit_path)
    rollback = read_json(rollback_path)
    migration = read_json(migration_path)

    if (
        rollback.get("status") != "rolled_back_at_clean_boundary"
        or int(loop.get("last_completed_iteration", -1)) != 10
        or int(loop.get("next_iteration", -1)) != 11
        or int(commit.get("last_completed_iteration", -1)) != 7
    ):
        raise RuntimeError("rollback boundary is not the exact iteration-10 to 11 pause")
    if migration.get("current_fingerprint") != loop.get("design_fingerprint"):
        raise RuntimeError("family rollback design migration is not committed")

    heldout = dict(loop.get("heldout_champion") or {})
    commit_heldout = dict(commit.get("heldout_champion") or {})
    evidence = dict(commit.get("heldout_champion_evidence") or {})
    if (
        heldout != commit_heldout
        or heldout.get("digest") != rollback.get("restored_checkpoint", {}).get("sha256")
        or evidence.get("checkpoint_digest") != heldout.get("digest")
        or int(evidence.get("games", 0)) <= 0
    ):
        raise RuntimeError("iteration-7 heldout evidence does not bind the rollback parent")

    run_dir = loop_path.parent
    if any(run_dir.rglob("*00011*")):
        raise RuntimeError("iteration 11 artifacts already exist")
    before_path = args.before_snapshot.resolve()
    immutable_json(before_path, loop)
    repaired = copy.deepcopy(loop)
    repaired["heldout_champion_evidence"] = evidence
    repaired.setdefault("family_design_rollbacks", []).append(
        {
            "heldout_evidence_repair": str(args.receipt.resolve()),
            "rollback_parent_commit": str(commit_path),
            "rollback_parent_commit_sha256": sha256(commit_path),
            "checkpoint_digest": heldout["digest"],
            "boundary_next_iteration": 11,
        }
    )
    atomic_json(loop_path, repaired)

    receipt = {
        "schema": SCHEMA,
        "status": "repaired_ready_for_managed_resume",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_revision": 150,
        "service": args.service,
        "service_state_during_repair": service_state,
        "loop_state": {"path": str(loop_path), "sha256": sha256(loop_path)},
        "before_snapshot": {
            "path": str(before_path),
            "sha256": sha256(before_path),
        },
        "rollback_receipt": {
            "path": str(rollback_path),
            "sha256": sha256(rollback_path),
        },
        "rollback_parent_commit": {
            "path": str(commit_path),
            "sha256": sha256(commit_path),
        },
        "design_migration": {
            "path": str(migration_path),
            "sha256": sha256(migration_path),
        },
        "proof": {
            "heldout_checkpoint_digest": heldout["digest"],
            "heldout_evidence_digest": evidence["checkpoint_digest"],
            "heldout_evidence_games": int(evidence["games"]),
            "next_collection_started_before_repair": False,
            "guide_or_model_authority_changed": False,
        },
    }
    immutable_json(args.receipt.resolve(), receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--rollback-parent-commit", type=Path, required=True)
    parser.add_argument("--rollback-receipt", type=Path, required=True)
    parser.add_argument("--design-migration", type=Path, required=True)
    parser.add_argument("--before-snapshot", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
