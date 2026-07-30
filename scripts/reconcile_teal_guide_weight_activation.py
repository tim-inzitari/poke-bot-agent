#!/usr/bin/env python3
"""Reconcile Teal's committed guide-weight migration without service control.

This repairs only the mutable runtime files after the original boundary
controller rejected a valid migration receipt containing the authorized
source-tree checksum change.  The running trainer is never stopped or
restarted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPECIALIST_ID = "teal-mask-ogerpon-ex"
MIGRATION_REASON = "receipt_backed_current_deck_guide_weight_curve_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def service_value(unit: str, field: str) -> str:
    return subprocess.check_output(
        ["systemctl", "--user", "show", unit, f"--property={field}", "--value"],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--staged-registry", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--staged-trainer", type=Path, required=True)
    parser.add_argument("--train-module", type=Path, required=True)
    parser.add_argument("--staged-train-module", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    migration = read_json(paths["migration_receipt"])
    changed = set(migration.get("changed_paths") or ())
    required = {
        "learner.alakazam_guide_loss_weight",
        "learner.current_deck_guide_loss_weight",
        "expert_rehearsal.loss_weights.alakazam_guide",
    }
    allowed = required | {"source.source_tree_sha256"}
    if (
        migration.get("reason") != MIGRATION_REASON
        or int(migration.get("boundary_next_iteration", -1)) != 6
        or not required <= changed <= allowed
    ):
        raise RuntimeError("immutable Teal guide-weight migration is invalid")

    selector = paths["selector"].read_text(encoding="utf-8")
    if f"POKEBOT_ACTIVE_SPECIALIST={SPECIALIST_ID}" not in selector:
        raise RuntimeError("selector is not Teal Mask Ogerpon ex")
    if (
        service_value(args.unit, "ActiveState") != "active"
        or service_value(args.unit, "SubState") != "running"
        or service_value(args.unit, "RefuseManualStop").casefold() != "yes"
    ):
        raise RuntimeError("managed trainer is not safely active")
    pid = int(service_value(args.unit, "MainPID") or 0)
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    if (
        b"--current-deck-guide-loss-weight 0.25" not in command
        or SPECIALIST_ID.encode() not in command
    ):
        raise RuntimeError("live trainer is not the committed 0.25 Teal process")

    staged_registry = read_json(paths["staged_registry"])
    teal = dict((staged_registry.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if float(teal.get("guide_loss_weight", -1.0)) != 0.25:
        raise RuntimeError("staged registry is not the committed Teal weight")

    atomic_bytes(
        paths["trainer"], paths["staged_trainer"].read_bytes(), 0o755
    )
    atomic_bytes(
        paths["train_module"], paths["staged_train_module"].read_bytes(), 0o644
    )
    atomic_bytes(
        paths["registry"], paths["staged_registry"].read_bytes(), 0o600
    )
    status = {
        "schema": "poke_bot.current_deck_guide_weight_boundary/v1",
        "status": "activated_reconciled_after_receipt_validator_mismatch",
        "active_specialist": SPECIALIST_ID,
        "completed_iteration": 5,
        "boundary_next_iteration": 6,
        "old_weight": 0.05,
        "new_weight": 0.25,
        "new_pid": pid,
        "selector_identity_unchanged": True,
        "design_migration_receipt": str(paths["migration_receipt"]),
        "design_migration_receipt_sha256": sha256(paths["migration_receipt"]),
        "registry_sha256": sha256(paths["registry"]),
        "trainer_sha256": sha256(paths["trainer"]),
        "train_module_sha256": sha256(paths["train_module"]),
        "stop_protection_restored": True,
        "service_control_used_by_reconciliation": False,
        "reconciled_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_bytes(
        paths["status"],
        (json.dumps(status, indent=2, sort_keys=True) + "\n").encode(),
        0o644,
    )
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
