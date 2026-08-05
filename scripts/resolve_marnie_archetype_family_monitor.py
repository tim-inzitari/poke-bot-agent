#!/usr/bin/env python3
"""Resolve the status-78 Marnie family monitor before managed resume.

A passing monitor preserves the activated family runtime. A failing monitor
applies the checksum-bound rollback. This program never starts or stops the
trainer; systemd owns the subsequent managed resume.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import sha256  # noqa: E402
from poke_bot.archetype_family_study import (  # noqa: E402
    validate_post_activation_monitor,
)
from scripts.apply_marnie_archetype_family_rollback import (  # noqa: E402
    apply as apply_rollback,
)


SCHEMA = "poke_bot.marnie_family_post_activation_resolution/v1"
GUIDE_SHADOW_SCHEMA = "poke_bot.marnie_family_guide_shadow_runtime/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _trainer_state(service: str) -> str:
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


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != body:
            raise RuntimeError("immutable monitor-resolution receipt changed")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _guide_shadow_runtime(
    args: argparse.Namespace, *, rollback_required: bool
) -> dict[str, Any]:
    authority_path = args.guide_shadow_runtime_receipt.expanduser().resolve()
    authority = _read(authority_path)
    if (
        authority.get("schema") != GUIDE_SHADOW_SCHEMA
        or authority.get("status") != "active_next_start_overlay"
        or int(authority.get("owner_revision", -1)) != 142
    ):
        raise RuntimeError("family monitor lacks guide-shadow runtime authority")
    key = "guide_retired_registry" if rollback_required else "merged_registry"
    target_row = dict(authority.get(key) or {})
    target = Path(str(target_row.get("path") or "")).resolve()
    if not target.is_file() or target_row.get("sha256") != sha256(target):
        raise RuntimeError("guide-shadow monitor target changed")
    registry = _read(target)
    row = dict(
        (registry.get("specialists") or {}).get("marnie-s-grimmsnarl-ex") or {}
    )
    if (
        float(row.get("guide_loss_weight", -1.0)) != 0.0
        or row.get("guide_retired") is not True
        or row.get("guide_target_generation_required") is not False
        or row.get("guide_conditioned_losses_enabled") is not False
        or row.get("guide_action_influence") is not False
    ):
        raise RuntimeError("monitor resume target restores guide authority")
    python = args.python_executable.expanduser().resolve()
    launcher = args.launcher.expanduser().resolve()
    body = (
        "[Service]\n"
        f"ExecStartPre={python} -u {launcher} --registry {target} --check\n"
        "ExecStart=\n"
        f"ExecStart={python} -u {launcher} --registry {target}\n"
    ).encode()
    overlay = args.guide_shadow_overlay.expanduser().resolve()
    if not overlay.is_file() or overlay.read_bytes() != body:
        _atomic_bytes(overlay, body)
    return {
        "authority": {"path": str(authority_path), "sha256": sha256(authority_path)},
        "effective_registry": {"path": str(target), "sha256": sha256(target)},
        "overlay": {"path": str(overlay), "sha256": sha256(overlay)},
        "guide_weight": 0.0,
        "guide_blocking": False,
        "family_rollback_applied": rollback_required,
    }


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.resolution_receipt.expanduser().resolve()
    if receipt_path.is_file():
        existing = _read(receipt_path)
        if existing.get("schema") != SCHEMA or existing.get("resume_authorized") is not True:
            raise RuntimeError("existing family-monitor resolution is invalid")
        return existing

    service_state = _trainer_state(args.service)
    monitor_path = args.monitor.expanduser().resolve()
    monitor = validate_post_activation_monitor(_read(monitor_path))
    migration_path = args.migration.expanduser().resolve()
    if (
        (monitor.get("migration") or {}).get("path") != str(migration_path)
        or (monitor.get("migration") or {}).get("sha256") != sha256(migration_path)
    ):
        raise RuntimeError("monitor resolution binds another migration")

    rollback_receipt: dict[str, Any] | None = None
    if monitor.get("rollback_required") is True:
        if not args.rollback_request.is_file():
            raise RuntimeError("monitor requires rollback but request is absent")
        rollback_receipt = apply_rollback(args)
        if rollback_receipt.get("status") != "rolled_back_at_clean_boundary":
            raise RuntimeError("required family rollback did not complete")

    guide_shadow_runtime = _guide_shadow_runtime(
        args, rollback_required=rollback_receipt is not None
    )

    payload = {
        "schema": SCHEMA,
        "status": (
            "rollback_applied_resume_parent"
            if rollback_receipt is not None
            else "monitor_passed_resume_family_design"
        ),
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitor": {"path": str(monitor_path), "sha256": sha256(monitor_path)},
        "migration": {
            "path": str(migration_path),
            "sha256": sha256(migration_path),
        },
        "rollback_required": bool(monitor.get("rollback_required")),
        "rollback_receipt": (
            {
                "path": str(args.receipt.expanduser().resolve()),
                "sha256": sha256(args.receipt.expanduser().resolve()),
            }
            if rollback_receipt is not None
            else None
        ),
        "guide_shadow_runtime": guide_shadow_runtime,
        "managed_trainer_state_during_resolution": service_state,
        "next_collection_started": False,
        "resume_authorized": True,
        "restart_authority": "declared_service_manager_only",
    }
    _write_immutable(receipt_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--rollback-request", type=Path, required=True)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--rollback-registry", type=Path, required=True)
    parser.add_argument("--environment-drop-in", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--resolution-receipt", type=Path, required=True)
    parser.add_argument("--guide-shadow-runtime-receipt", type=Path, required=True)
    parser.add_argument("--guide-shadow-overlay", type=Path, required=True)
    return parser


def main() -> int:
    result = resolve(_parser().parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
