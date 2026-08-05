#!/usr/bin/env python3
"""Apply a requested Marnie family rollback at an exact managed boundary."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import (  # noqa: E402
    PAUSE_SCHEMA,
    ROLLBACK_RECEIPT_SCHEMA,
    sha256,
    validate_rollback_request,
)
from poke_bot.archetype_family_study import validate_post_activation_monitor  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_immutable(path: Path, value: Any) -> Path:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable rollback artifact changed: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _service_state(service: str) -> str:
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


def apply(args: argparse.Namespace) -> dict[str, Any]:
    monitor = _read(args.monitor)
    validate_post_activation_monitor(monitor)
    if monitor.get("rollback_required") is not True:
        return {"status": "monitor_passed_no_rollback"}
    if not args.rollback_request.is_file():
        raise RuntimeError("required rollback request is absent")
    request = _read(args.rollback_request)
    validate_rollback_request(
        request,
        migration_path=args.migration.resolve(),
        monitor_path=args.monitor.resolve(),
    )
    migration = _read(args.migration)
    service_state = _service_state(args.service)
    pause_path = Path(str((request.get("pause") or {}).get("path", ""))).resolve()
    if sha256(pause_path) != (request.get("pause") or {}).get("sha256"):
        raise RuntimeError("rollback pause digest changed")
    pause = _read(pause_path)
    if (
        pause.get("schema") != PAUSE_SCHEMA
        or pause.get("next_collection_started") is not False
        or int(pause.get("restart_prevent_status", -1)) != 78
    ):
        raise RuntimeError("rollback is not at its clean boundary")
    for key in ("restore_registry", "restore_selector", "restore_checkpoint"):
        row = request.get(key) or {}
        path = Path(str(row.get("path", ""))).resolve()
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise RuntimeError(f"sealed rollback identity changed: {key}")
    if args.receipt.is_file():
        receipt = _read(args.receipt)
        if (
            receipt.get("schema") != ROLLBACK_RECEIPT_SCHEMA
            or receipt.get("status") != "rolled_back_at_clean_boundary"
            or receipt.get("request_sha256") != sha256(args.rollback_request)
        ):
            raise RuntimeError("existing rollback receipt is invalid")
        return receipt
    loop = _read(args.loop_state)
    current_learner = dict(loop.get("learner") or {})
    restore_digest = str(request["restore_checkpoint"]["sha256"])
    if current_learner.get("digest") not in {
        pause.get("learner_sha256"),
        restore_digest,
    }:
        raise RuntimeError("rollback learner moved after the monitor pause")
    evidence_dir = args.receipt.resolve().parent / "rollback-evidence"
    before_loop_path = evidence_dir / "pre-rollback-loop-state.json"
    if before_loop_path.is_file():
        original_loop = _read(before_loop_path)
        if (original_loop.get("learner") or {}).get("digest") != pause.get(
            "learner_sha256"
        ):
            raise RuntimeError("pre-rollback loop evidence changed")
    else:
        if current_learner.get("digest") != pause.get("learner_sha256"):
            raise RuntimeError("partial rollback lacks its original loop evidence")
        original_loop = loop
        _write_immutable(before_loop_path, original_loop)
    activation_drop_in_path = args.environment_drop_in.resolve()
    if not activation_drop_in_path.is_file():
        raise RuntimeError("active family systemd drop-in is absent")
    activation_drop_in_snapshot = evidence_dir / "activated-family.conf"
    if activation_drop_in_snapshot.is_file():
        if sha256(activation_drop_in_snapshot) != migration.get(
            "environment_drop_in_sha256"
        ):
            raise RuntimeError("activated drop-in evidence changed")
    else:
        if sha256(activation_drop_in_path) != migration.get(
            "environment_drop_in_sha256"
        ):
            raise RuntimeError("active family drop-in changed before rollback")
        activation_drop_in_snapshot.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            activation_drop_in_snapshot,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(activation_drop_in_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    before_registry = _read(
        Path(str(request["restore_registry"]["path"])).resolve()
    )
    rollback_registry = copy.deepcopy(before_registry)
    trainer_args = list(rollback_registry.get("common_trainer_args") or [])
    try:
        reason_index = trainer_args.index("--boundary-design-migration-reason") + 1
    except ValueError as exc:
        raise RuntimeError("rollback registry lacks a migration reason") from exc
    if reason_index >= len(trainer_args):
        raise RuntimeError("rollback registry migration reason is empty")
    trainer_args[reason_index] = "receipt_backed_marnie_family_rollback_r133"
    rollback_registry["common_trainer_args"] = trainer_args
    rollback_registry_path = _write_immutable(
        args.rollback_registry.resolve(), rollback_registry
    )
    rollback_loop = copy.deepcopy(original_loop)
    restored_identity = {
        "path": str(Path(str(request["restore_checkpoint"]["path"])).resolve()),
        "digest": str(request["restore_checkpoint"]["sha256"]),
    }
    rollback_loop["learner"] = dict(restored_identity)
    rollback_loop["champion"] = dict(restored_identity)
    rollback_loop["heldout_champion"] = dict(restored_identity)
    rollback_loop.setdefault("family_design_rollbacks", []).append(
        {
            "request": str(args.rollback_request.resolve()),
            "request_sha256": sha256(args.rollback_request),
            "monitor": str(args.monitor.resolve()),
            "monitor_sha256": sha256(args.monitor),
            "restored_learner": dict(restored_identity),
            "restored_champion": dict(restored_identity),
            "restored_heldout_champion": dict(restored_identity),
            "boundary_next_iteration": int(loop.get("next_iteration", -1)),
        }
    )
    rollback_loop_bytes = (
        json.dumps(rollback_loop, indent=2, sort_keys=True) + "\n"
    ).encode()
    drop_in_bytes = (
        "[Service]\n"
        "UnsetEnvironment=POKEBOT_ARCHETYPE_FAMILY_MANIFEST "
        "POKEBOT_ARCHETYPE_LOSS_CONTRACT POKEBOT_ARCHETYPE_LOSS_VECTOR\n"
        "ExecStart=\n"
        f"ExecStart={args.python_executable.resolve()} -u "
        f"{args.launcher.resolve()} --registry {rollback_registry_path}\n"
    ).encode()
    # Training is stopped and cannot be restarted by this program. A crash
    # between replacements is recoverable and the receipt is written only
    # after both exact rollback identities are installed.
    _atomic_bytes(args.loop_state.resolve(), rollback_loop_bytes)
    _atomic_bytes(activation_drop_in_path, drop_in_bytes)
    receipt = {
        "schema": ROLLBACK_RECEIPT_SCHEMA,
        "status": "rolled_back_at_clean_boundary",
        "request": str(args.rollback_request.resolve()),
        "request_sha256": sha256(args.rollback_request),
        "monitor": str(args.monitor.resolve()),
        "monitor_sha256": sha256(args.monitor),
        "pause": str(pause_path),
        "pause_sha256": sha256(pause_path),
        "service_state_during_rollback": service_state,
        "pre_rollback_loop_state": str(before_loop_path),
        "pre_rollback_loop_state_sha256": sha256(before_loop_path),
        "restored_checkpoint": dict(request["restore_checkpoint"]),
        "restored_authority_fields": [
            "learner", "champion", "heldout_champion"
        ],
        "rollback_registry": str(rollback_registry_path),
        "rollback_registry_sha256": sha256(rollback_registry_path),
        "environment_drop_in": str(activation_drop_in_path),
        "environment_drop_in_sha256": sha256(activation_drop_in_path),
        "activated_drop_in_evidence": str(activation_drop_in_snapshot),
        "activated_drop_in_evidence_sha256": sha256(activation_drop_in_snapshot),
        "next_collection_started": False,
        "restart_authority": "declared_service_manager_only",
    }
    _write_immutable(args.receipt.resolve(), receipt)
    return receipt


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
    return parser


def main() -> int:
    result = apply(_parser().parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
