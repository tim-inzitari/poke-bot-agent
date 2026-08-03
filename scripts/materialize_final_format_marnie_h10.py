#!/usr/bin/env python3
"""Materialize Marnie's exact H10/Fusion-v3 child at its core boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.pure_rl.model_registry import verify_frozen_model  # noqa: E402
from scripts.validate_final_format_marnie_h10 import validate as validate_h10  # noqa: E402


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
BOUNDARY_SCHEMA = "poke_bot.post_alakazam_core_refresh_boundary/v1"
READY_SCHEMA = "poke_bot.final_format_marnie_h10_materialization/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"Marnie H10 materialization changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def _validate_authority(
    *, prestage_path: Path, boundary_path: Path, latest_core_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    prestage = _read(prestage_path)
    boundary = _read(boundary_path)
    pointer = _read(latest_core_path)
    if (
        prestage.get("schema") != "poke_bot.final_format_marnie_refresh_prestage/v1"
        or prestage.get("status") != "authorized_preparation_started"
        or prestage.get("specialist_id") != SPECIALIST_ID
        or int(prestage.get("boundary_iteration") or -1) != 20
        or prestage.get("final_capacity_profile") != "H10-I/v1"
        or prestage.get("final_decision_fusion_schema")
        != "poke_bot.causal_decision_fusion/v3"
        or prestage.get("training_authority") is not False
        or prestage.get("selector_authority") is not False
    ):
        raise RuntimeError("Marnie pre-stage is not authorized at iteration 20")
    if (
        boundary.get("schema") != BOUNDARY_SCHEMA
        or boundary.get("status") != "selected_core_ready_for_marnie_h10"
        or boundary.get("predecessor_refresh") != "alakazam"
        or boundary.get("specialist_id") != SPECIALIST_ID
        or boundary.get("normal_core_refresh_attempted") is not True
        or boundary.get("rejected_candidate_blocks_production") is not False
        or boundary.get("training_authority") is not False
        or boundary.get("selector_authority") is not False
        or boundary.get("latest_core_pointer") != str(latest_core_path)
        or boundary.get("latest_core_pointer_sha256") != _sha256(latest_core_path)
        or boundary.get("selected_core_checkpoint_sha256")
        != pointer.get("checkpoint_digest")
    ):
        raise RuntimeError("post-Alakazam core boundary is not authoritative")
    ready_path = Path(str(pointer.get("ready") or "")).expanduser().resolve()
    family_path = Path(str(pointer.get("family") or "")).expanduser().resolve()
    if (
        pointer.get("schema") != "poke_bot.latest_cumulative_core_pointer/v1"
        or not ready_path.is_file()
        or checkpoint.checkpoint_digest(ready_path) != pointer.get("ready_digest")
    ):
        raise RuntimeError("latest accepted cumulative-core pointer is invalid")
    ready = _read(ready_path)
    frozen = verify_frozen_model(family_path)
    parent = Path(str(frozen.get("model_path") or "")).resolve()
    if (
        frozen.get("checkpoint_digest") != pointer.get("checkpoint_digest")
        or ready.get("checkpoint_digest") != pointer.get("checkpoint_digest")
        or checkpoint.checkpoint_digest(parent) != pointer.get("checkpoint_digest")
    ):
        raise RuntimeError("selected cumulative-core checkpoint identity disagrees")
    return prestage, boundary, pointer, parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prestage", type=Path, required=True)
    parser.add_argument("--core-boundary", type=Path, required=True)
    parser.add_argument("--latest-core", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--migration-script", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--role-route-receipt", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    prestage_path = args.prestage.expanduser().resolve()
    boundary_path = args.core_boundary.expanduser().resolve()
    latest_core_path = args.latest_core.expanduser().resolve()
    _prestage, boundary, pointer, parent = _validate_authority(
        prestage_path=prestage_path,
        boundary_path=boundary_path,
        latest_core_path=latest_core_path,
    )
    child = args.child.expanduser().resolve()
    migration = args.migration_receipt.expanduser().resolve()
    roles = args.role_route_receipt.expanduser().resolve()
    validation_path = args.validation_receipt.expanduser().resolve()
    outputs = (child, migration, roles)
    if any(path.exists() for path in outputs) and not all(path.is_file() for path in outputs):
        raise RuntimeError("partial Marnie H10 materialization exists")
    if not all(path.is_file() for path in outputs):
        child.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python.expanduser().resolve()),
            "-u",
            str(args.migration_script.expanduser().resolve()),
            "--parent",
            str(parent),
            "--expected-parent-sha256",
            str(pointer["checkpoint_digest"]),
            "--specialist-id",
            SPECIALIST_ID,
            "--model-id",
            "final-format-marnie-r104-h10-i-fusion-v3",
            "--migration-schema",
            "poke_bot.final_format_marnie_h10_migration/v1",
            "--child",
            str(child),
            "--migration-receipt",
            str(migration),
            "--role-route-receipt",
            str(roles),
            "--directional-fusion-v3",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise RuntimeError(f"Marnie H10 migration failed: rc={result.returncode}")
    validation = validate_h10(
        child_path=child,
        migration_path=migration,
        roles_path=roles,
    )
    _write_once(validation_path, validation)
    receipt = {
        "schema": READY_SCHEMA,
        "status": "h10_child_validated_ready_for_exact_25_epoch_bootstrap",
        "specialist_id": SPECIALIST_ID,
        "prestage": str(prestage_path),
        "prestage_sha256": _sha256(prestage_path),
        "core_boundary": str(boundary_path),
        "core_boundary_sha256": _sha256(boundary_path),
        "attempted_core_generation": boundary.get("attempted_core_generation"),
        "selected_core_generation": pointer.get("version"),
        "selected_core_checkpoint_sha256": pointer.get("checkpoint_digest"),
        "checkpoint": str(child),
        "checkpoint_sha256": _sha256(child),
        "migration_receipt": str(migration),
        "migration_receipt_sha256": _sha256(migration),
        "role_route_receipt": str(roles),
        "role_route_receipt_sha256": _sha256(roles),
        "validation_receipt": str(validation_path),
        "validation_receipt_sha256": _sha256(validation_path),
        "capacity_profile": "H10-I/v1",
        "architecture": validation["architecture"],
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "training_before_h10_migration_allowed": False,
        "training_authority": False,
        "selector_authority": False,
        "next_boundary": "exact_25_epoch_h10_bootstrap",
    }
    _write_once(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
