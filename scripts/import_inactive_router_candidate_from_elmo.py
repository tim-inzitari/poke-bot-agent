#!/usr/bin/env python3
"""Import and audit an inactive public matchup router from Elmo.

This deliberately stops at an immutable boundary-only promotion receipt.  It
does not activate the router, modify the runtime selector, or touch training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

READY_SCHEMA = "poke_bot.public_matchup_decision_tree_receipt/v1"
TREE_SCHEMA = "poke_bot.public_matchup_decision_tree/v1"


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_object_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {source}")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    return _read_object_bytes(path.read_bytes(), source=str(path))


def _remote_bytes(host: str, path: str) -> bytes:
    completed = subprocess.run(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"sudo -n cat -- {shlex.quote(path)}",
        ],
        check=False,
        stdout=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(
            f"remote read failed rc={completed.returncode}: {host}:{path}"
        )
    return completed.stdout


def _install_immutable(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def import_candidate(args: argparse.Namespace) -> dict[str, Any]:
    remote_root = args.remote_root.rstrip("/")
    ready_remote = f"{remote_root}/PUBLIC_MATCHUP_TREE_READY.json"
    tree_remote = f"{remote_root}/public-matchup-tree.json"
    ready_bytes = _remote_bytes(args.host, ready_remote)
    tree_bytes = _remote_bytes(args.host, tree_remote)
    ready = _read_object_bytes(ready_bytes, source=ready_remote)
    tree = _read_object_bytes(tree_bytes, source=tree_remote)
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("runtime_enabled") is not False
        or ready.get("artifact_sha256") != _sha256_bytes(tree_bytes)
        or tree.get("schema") != TREE_SCHEMA
        or tree.get("runtime_enabled") is not False
    ):
        raise RuntimeError("remote inactive-router receipt identity failed")

    tree_output = args.tree_output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    roster_path = args.roster.expanduser().resolve()
    roster = _read_object(roster_path)
    targets = tuple(str(value) for value in roster.get("expert_ids") or ())
    active_targets = tuple(
        str(value) for value in roster.get("active_expert_ids") or ()
    )
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or not targets
        or targets != active_targets
        or tuple(str(value) for value in tree.get("targets") or ()) != targets
        or int(roster.get("required_specialist_count") or 0) != len(targets)
    ):
        raise RuntimeError("canonical router roster identity failed")

    from scripts.audit_public_matchup_tree_candidate import audit

    _install_immutable(tree_output, tree_bytes)
    candidate_audit = audit(
        tree_output,
        minimum_precision=args.minimum_precision,
        minimum_weighted_support=args.minimum_weighted_support,
        expected_targets=targets,
        target_registry=roster_path,
    )
    if (
        int(candidate_audit.get("accepted_count") or 0) != len(targets)
        or candidate_audit.get("rejected_specialists")
    ):
        raise RuntimeError("not every canonical route passed inactive audit")
    _install_immutable(audit_output, _json_bytes(candidate_audit))

    repository = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-u",
        str(repository / "scripts/register_router_only_promotion.py"),
        "--tree",
        str(tree_output),
        "--audit",
        str(audit_output),
        "--roster",
        str(roster_path),
        "--corpus-root",
        str(args.corpus_root.expanduser().resolve()),
        "--receipt",
        str(args.promotion_receipt.expanduser().resolve()),
    ]
    if args.parent_receipt is not None:
        command.extend(
            (
                "--parent-receipt",
                str(args.parent_receipt.expanduser().resolve()),
            )
        )
    for specialist_id in args.ready_archetype:
        command.extend(("--ready-archetype", specialist_id))
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(
            "router-only promotion registration failed "
            f"rc={completed.returncode}"
        )
    promotion = _read_object(args.promotion_receipt.expanduser().resolve())
    return {
        "status": "ready",
        "remote_ready_receipt": ready_remote,
        "remote_ready_receipt_sha256": _sha256_bytes(ready_bytes),
        "candidate_tree": str(tree_output),
        "candidate_tree_sha256": _sha256(tree_output),
        "candidate_audit": str(audit_output),
        "candidate_audit_sha256": _sha256(audit_output),
        "accepted_count": promotion.get("accepted_count"),
        "accepted_specialist_ids": promotion.get(
            "accepted_specialist_ids"
        ),
        "promotion_receipt": str(
            args.promotion_receipt.expanduser().resolve()
        ),
        "promotion_receipt_sha256": _sha256(
            args.promotion_receipt.expanduser().resolve()
        ),
        "runtime_enabled": False,
        "live_trainer_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="elmo")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--tree-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--promotion-receipt", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path)
    parser.add_argument("--ready-archetype", action="append", default=[])
    parser.add_argument("--minimum-precision", type=float, default=0.93)
    parser.add_argument("--minimum-weighted-support", type=int, default=10_000)
    args = parser.parse_args()
    print(
        json.dumps(import_candidate(args), indent=2, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
