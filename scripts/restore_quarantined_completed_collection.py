#!/usr/bin/env python3
"""Restore one receipt-verified completed collection through managed lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import train_pure_rl
from scripts.apply_decision_fusion_warmup_managed_boundary import (
    _atomic_text,
    _install_stop_override,
    _read,
    _remove_stop_override,
    _run,
    _service_value,
)


SCHEMA = "poke_bot.managed_completed_collection_restore/v1"


def _publish(path: Path, status: str, **values: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            {
                "schema": SCHEMA,
                "status": status,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **values,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _verified_plan(attempt: Path, run_dir: Path) -> list[dict[str, Any]]:
    plan = _read(attempt / "plan.json")
    failure = _read(attempt / "failure.json")
    rows = list(plan.get("artifacts") or [])
    failure_rows = list(failure.get("artifacts") or [])
    if not (
        int(plan.get("iteration", -1)) == 15
        and int(failure.get("iteration", -1)) == 15
        and str(failure.get("quarantine_completed_at_utc") or "")
        and failure_rows == rows
        and rows
    ):
        raise RuntimeError("quarantine attempt is not one completed transaction")
    required = {
        "shards/iter_00015.jsonl",
        "collection_receipts/iter_00015.json",
    }
    if {str(row.get("relative_path") or "") for row in rows} != required:
        raise RuntimeError("quarantine attempt does not contain the exact pair")
    for row in rows:
        source = Path(str(row["source"])).resolve()
        destination = Path(str(row["destination"])).resolve()
        if (
            source != (run_dir / str(row["relative_path"])).resolve()
            or not destination.is_file()
            or int(destination.stat().st_size) != int(row["size"])
            or train_pure_rl._sha256_file(destination) != str(row["digest"])
        ):
            raise RuntimeError(f"quarantine artifact changed: {destination}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--stop-override", type=Path, required=True)
    parser.add_argument("--maintenance-lock", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.attempt = args.attempt.expanduser().resolve()
    args.stop_override = args.stop_override.expanduser().resolve()
    args.maintenance_lock = args.maintenance_lock.expanduser().resolve()
    args.status = args.status.expanduser().resolve()

    rows = _verified_plan(args.attempt, args.run_dir)
    state = _read(args.run_dir / "loop_state.json")
    if not (
        int(state.get("last_completed_iteration", -1)) == 14
        and int(state.get("next_iteration", -1)) == 15
    ):
        raise RuntimeError("restore is not at the declared iteration-15 boundary")
    _publish(args.status, "validated", attempt=str(args.attempt))
    stop_authority = False
    lock_installed = False
    try:
        _atomic_text(
            args.maintenance_lock,
            json.dumps(
                {
                    "schema": "poke_bot.managed_training_maintenance/v1",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "expires_at_epoch": time.time() + 900.0,
                    "owner_pid": os.getpid(),
                    "training_service": args.unit,
                    "authority": SCHEMA,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        lock_installed = True
        _install_stop_override(args.stop_override)
        stop_authority = True
        if _service_value(args.unit, "RefuseManualStop") != "no":
            raise RuntimeError("managed restore stop authority was not installed")
        _run(["systemctl", "--user", "stop", args.unit], timeout=90)
        if _service_value(args.unit, "ActiveState") not in {"inactive", "failed"}:
            raise RuntimeError("trainer did not stop for collection restore")

        state = _read(args.run_dir / "loop_state.json")
        quarantined = train_pure_rl._recover_interrupted_iteration(
            args.run_dir,
            state,
            preserve_completed_collection=False,
        )
        if quarantined is None:
            raise RuntimeError("partial retry was not quarantined")
        for row in rows:
            source = Path(str(row["source"])).resolve()
            destination = Path(str(row["destination"])).resolve()
            if source.exists() or not destination.is_file():
                raise RuntimeError("restore source/destination state is ambiguous")
            destination.replace(source)
            if train_pure_rl._sha256_file(source) != str(row["digest"]):
                raise RuntimeError(f"restored artifact failed checksum: {source}")

        manifest = _read(args.run_dir / "manifest.json")
        completed, _contract = (
            train_pure_rl._verified_completed_collection_across_design_chain(
                args.run_dir, state, manifest
            )
        )
        if completed is None or int(completed.get("requested_games", -1)) != 8192:
            raise RuntimeError("restored completed collection did not verify")

        _remove_stop_override(args.stop_override)
        stop_authority = False
        _run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        _run(["systemctl", "--user", "start", args.unit], timeout=90)
        pid = int(_service_value(args.unit, "MainPID") or 0)
        if pid <= 0 or _service_value(args.unit, "ActiveState") != "active":
            raise RuntimeError("trainer did not restart after collection restore")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if (
                int(_service_value(args.unit, "MainPID") or 0) != pid
                or _service_value(args.unit, "ActiveState") != "active"
            ):
                raise RuntimeError("restored trainer failed stability check")
            time.sleep(1.0)
        if not (args.run_dir / "collection_receipts/iter_00015.json").is_file():
            raise RuntimeError("trainer discarded the restored receipt")
        args.maintenance_lock.unlink(missing_ok=True)
        lock_installed = False
        _publish(
            args.status,
            "complete",
            main_pid=pid,
            restored_attempt=str(args.attempt),
            quarantined_partial=str(quarantined),
            requested_games=8192,
        )
        return 0
    except BaseException as exc:
        if stop_authority:
            _remove_stop_override(args.stop_override)
        _run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        if _service_value(args.unit, "ActiveState") not in {"active", "activating"}:
            _run(["systemctl", "--user", "start", "--no-block", args.unit], check=False)
        if lock_installed:
            args.maintenance_lock.unlink(missing_ok=True)
        _publish(
            args.status,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
