#!/usr/bin/env python3
"""Bind the family rollback registry to Marnie's retired guide contract.

The family rollback registry carries the one clean-boundary migration reason
that authorizes removal of the failed family sampler.  The guide-shadow
overlay must therefore launch an immutable derivative of that registry, not
the older guide-retired parent registry which lacks the rollback reason.
"""

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


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
SCHEMA = "poke_bot.marnie_family_rollback_guide_shadow_repair/v2"
ROLLBACK_REASON = "receipt_backed_marnie_family_rollback_r133"
GUIDE_FIELDS = {
    "guide_loss_weight",
    "guide_retired",
    "guide_retirement_revision",
    "guide_target_generation_required",
    "guide_conditioned_losses_enabled",
    "guide_action_influence",
    "guide_historical_artifacts",
    "guide_shadow_only",
    "guide_shadow_blocking",
    "guide_shadow_runtime_authority",
}


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


def atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def specialist_row(registry: dict[str, Any]) -> dict[str, Any]:
    row = dict((registry.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if not row:
        raise RuntimeError("Marnie specialist row is absent")
    return row


def migration_reason(registry: dict[str, Any]) -> str:
    args = list(registry.get("common_trainer_args") or [])
    try:
        index = args.index("--boundary-design-migration-reason") + 1
    except ValueError as exc:
        raise RuntimeError("rollback registry lacks a migration reason") from exc
    if index >= len(args):
        raise RuntimeError("rollback registry migration reason is empty")
    return str(args[index])


def merge(
    rollback_registry: dict[str, Any],
    guide_retired_registry: dict[str, Any],
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    if migration_reason(rollback_registry) != ROLLBACK_REASON:
        raise RuntimeError("rollback registry lacks the exact family rollback reason")
    rollback_row = specialist_row(rollback_registry)
    retired_row = specialist_row(guide_retired_registry)
    if (
        float(retired_row.get("guide_loss_weight", -1.0)) != 0.0
        or retired_row.get("guide_retired") is not True
        or retired_row.get("guide_target_generation_required") is not False
        or retired_row.get("guide_conditioned_losses_enabled") is not False
        or retired_row.get("guide_action_influence") is not False
    ):
        raise RuntimeError("retired guide registry restores guide authority")

    merged = copy.deepcopy(rollback_registry)
    merged["runtime_root"] = str(runtime_root)
    merged_row = dict(rollback_row)
    for field in GUIDE_FIELDS:
        if field in retired_row:
            merged_row[field] = copy.deepcopy(retired_row[field])
    merged_row.update(
        {
            "guide_loss_weight": 0.0,
            "guide_retired": True,
            "guide_retirement_revision": 140,
            "guide_target_generation_required": False,
            "guide_conditioned_losses_enabled": False,
            "guide_action_influence": False,
            "guide_historical_artifacts": "optional_offline_shadow_only",
            "guide_shadow_only": True,
            "guide_shadow_blocking": False,
            "guide_shadow_runtime_authority": False,
        }
    )
    merged["specialists"][SPECIALIST_ID] = merged_row
    merged["marnie_guide_retirement"] = {
        "owner_revision": 141,
        "status": "shadow_only_non_authoritative",
        "weight": 0.0,
        "blocking": False,
    }
    merged["marnie_family_rollback_guide_shadow_repair"] = {
        "schema": SCHEMA,
        "owner_revision": 149,
        "family_rollback_preserved": True,
        "postupload_runtime_rebound": True,
        "guide_shadow_only": True,
        "guide_weight": 0.0,
        "guide_blocking": False,
    }
    return merged


def validate_non_guide_identity(
    rollback_registry: dict[str, Any],
    merged_registry: dict[str, Any],
    *,
    runtime_root: Path,
) -> None:
    before = copy.deepcopy(rollback_registry)
    after = copy.deepcopy(merged_registry)
    before_row = specialist_row(before)
    after_row = specialist_row(after)
    for field in GUIDE_FIELDS:
        before_row.pop(field, None)
        after_row.pop(field, None)
    before["specialists"][SPECIALIST_ID] = before_row
    after["specialists"][SPECIALIST_ID] = after_row
    for key in (
        "marnie_guide_retirement",
        "marnie_family_rollback_guide_shadow_repair",
    ):
        before.pop(key, None)
        after.pop(key, None)
    if after.get("runtime_root") != str(runtime_root):
        raise RuntimeError("post-upload runtime root was not rebound")
    before["runtime_root"] = str(runtime_root)
    if before != after:
        raise RuntimeError("non-guide rollback registry fields changed")
    if migration_reason(after) != ROLLBACK_REASON:
        raise RuntimeError("family rollback reason changed")


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
    rollback_receipt = read_json(args.rollback_receipt.resolve())
    resolution = read_json(args.resolution.resolve())
    if (
        rollback_receipt.get("status") != "rolled_back_at_clean_boundary"
        or resolution.get("status") != "rollback_applied_resume_parent"
        or resolution.get("resume_authorized") is not True
    ):
        raise RuntimeError("family rollback is not fully resolved")
    rollback_path = Path(str(rollback_receipt.get("rollback_registry", ""))).resolve()
    if (
        not rollback_path.is_file()
        or rollback_receipt.get("rollback_registry_sha256") != sha256(rollback_path)
    ):
        raise RuntimeError("rollback registry identity changed")

    retired_path = args.guide_retired_registry.resolve()
    output_path = args.output_registry.resolve()
    rollback = read_json(rollback_path)
    retired = read_json(retired_path)
    launcher = args.launcher.resolve()
    runtime_root = launcher.parents[1]
    if launcher.parent.name != "scripts" or runtime_root.name != "final-format-marnie-postupload-r136":
        raise RuntimeError("repair launcher is not the canonical post-upload runtime")
    merged = merge(rollback, retired, runtime_root=runtime_root)
    validate_non_guide_identity(rollback, merged, runtime_root=runtime_root)
    immutable_json(output_path, merged)

    for path in (args.python_executable, args.launcher, output_path):
        if not path.resolve().is_absolute() or " " in str(path.resolve()):
            raise RuntimeError("invalid managed runtime path")
    body = (
        "[Service]\n"
        f"ExecStartPre={args.python_executable.resolve()} -u "
        f"{args.launcher.resolve()} --registry {output_path} --check\n"
        "ExecStart=\n"
        f"ExecStart={args.python_executable.resolve()} -u "
        f"{args.launcher.resolve()} --registry {output_path}\n"
    ).encode()
    atomic_bytes(args.overlay.resolve(), body)

    receipt = {
        "schema": SCHEMA,
        "status": "repaired_ready_for_managed_resume",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_revision": 149,
        "service": args.service,
        "service_state_during_repair": service_state,
        "rollback_receipt": {
            "path": str(args.rollback_receipt.resolve()),
            "sha256": sha256(args.rollback_receipt.resolve()),
        },
        "resolution": {
            "path": str(args.resolution.resolve()),
            "sha256": sha256(args.resolution.resolve()),
        },
        "rollback_registry": {
            "path": str(rollback_path),
            "sha256": sha256(rollback_path),
        },
        "guide_retired_registry": {
            "path": str(retired_path),
            "sha256": sha256(retired_path),
        },
        "effective_registry": {
            "path": str(output_path),
            "sha256": sha256(output_path),
        },
        "overlay": {
            "path": str(args.overlay.resolve()),
            "sha256": sha256(args.overlay.resolve()),
        },
        "proof": {
            "family_rollback_reason_preserved": True,
            "all_non_guide_registry_fields_preserved": True,
            "only_operational_runtime_root_rebound": True,
            "guide_weight": 0.0,
            "guide_shadow_only": True,
            "guide_action_influence": False,
            "guide_blocking": False,
            "next_collection_started_before_repair": False,
        },
    }
    immutable_json(args.receipt.resolve(), receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--rollback-receipt", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--guide-retired-registry", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
