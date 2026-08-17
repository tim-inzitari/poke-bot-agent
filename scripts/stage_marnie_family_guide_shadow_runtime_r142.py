#!/usr/bin/env python3
"""Merge Marnie's active family registry with the zero-authority guide contract.

The family activation was materialized before the guide was retired, so its
immutable registry still carries the historical 0.05 guide weight.  This
script creates a new immutable derivative that preserves the family sampler,
typed loss vector, latent policy, Fusion-v3 inventory, and every non-guide
field while replacing only the guide authority fields with the checksum-bound
retired/shadow-only contract.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
SCHEMA = "poke_bot.marnie_family_guide_shadow_runtime/v1"
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


def _row(registry: dict[str, Any]) -> dict[str, Any]:
    row = dict((registry.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if not row:
        raise RuntimeError("Marnie specialist row is absent")
    return row


def merge_registries(
    family_registry: dict[str, Any], guide_retired_registry: dict[str, Any]
) -> dict[str, Any]:
    family_row = _row(family_registry)
    retired_row = _row(guide_retired_registry)
    if (
        float(family_row.get("guide_loss_weight", -1.0)) != 0.05
        or float(retired_row.get("guide_loss_weight", -1.0)) != 0.0
        or retired_row.get("guide_retired") is not True
        or int(retired_row.get("guide_retirement_revision", -1)) != 140
        or retired_row.get("guide_target_generation_required") is not False
        or retired_row.get("guide_conditioned_losses_enabled") is not False
        or retired_row.get("guide_action_influence") is not False
    ):
        raise RuntimeError("family/guide parent contracts are not mergeable")

    merged = copy.deepcopy(family_registry)
    merged_row = dict(family_row)
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
    merged["marnie_family_guide_shadow_runtime"] = {
        "schema": SCHEMA,
        "owner_revision": 142,
        "family_system_preserved": True,
        "guide_shadow_only": True,
        "guide_weight": 0.0,
        "guide_blocking": False,
    }
    return merged


def validate_non_guide_identity(
    family_registry: dict[str, Any], merged_registry: dict[str, Any]
) -> None:
    before = copy.deepcopy(family_registry)
    after = copy.deepcopy(merged_registry)
    before_row = _row(before)
    after_row = _row(after)
    for field in GUIDE_FIELDS:
        before_row.pop(field, None)
        after_row.pop(field, None)
    before["specialists"][SPECIALIST_ID] = before_row
    after["specialists"][SPECIALIST_ID] = after_row
    after.pop("marnie_guide_retirement", None)
    after.pop("marnie_family_guide_shadow_runtime", None)
    before.pop("marnie_guide_retirement", None)
    before.pop("marnie_family_guide_shadow_runtime", None)
    if before != after:
        raise RuntimeError("family/non-guide registry fields changed during merge")


def drop_in_bytes(*, python_executable: Path, launcher: Path, registry: Path) -> bytes:
    for path in (python_executable, launcher, registry):
        if not path.is_absolute() or "\n" in str(path) or " " in str(path):
            raise RuntimeError("invalid managed runtime path")
    return (
        "[Service]\n"
        f"ExecStartPre={python_executable} -u {launcher} --registry {registry} --check\n"
        "ExecStart=\n"
        f"ExecStart={python_executable} -u {launcher} --registry {registry}\n"
    ).encode()


def stage(args: argparse.Namespace) -> dict[str, Any]:
    family_path = args.family_registry.expanduser().resolve()
    retired_path = args.guide_retired_registry.expanduser().resolve()
    output_path = args.output_registry.expanduser().resolve()
    drop_in_path = args.environment_drop_in.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    family = read_json(family_path)
    retired = read_json(retired_path)
    merged = merge_registries(family, retired)
    validate_non_guide_identity(family, merged)
    immutable_json(output_path, merged)
    body = drop_in_bytes(
        python_executable=args.python_executable.expanduser().resolve(),
        launcher=args.launcher.expanduser().resolve(),
        registry=output_path,
    )
    if not drop_in_path.is_file() or drop_in_path.read_bytes() != body:
        atomic_bytes(drop_in_path, body)
    receipt = {
        "schema": SCHEMA,
        "status": "active_next_start_overlay",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_revision": 142,
        "family_registry": {"path": str(family_path), "sha256": sha256(family_path)},
        "guide_retired_registry": {
            "path": str(retired_path),
            "sha256": sha256(retired_path),
        },
        "merged_registry": {"path": str(output_path), "sha256": sha256(output_path)},
        "environment_drop_in": {
            "path": str(drop_in_path),
            "sha256": sha256(drop_in_path),
            "rollback_controller_owned": True,
        },
        "proof": {
            "family_and_typed_loss_system_preserved": True,
            "latent_policy_and_fusion_preserved": True,
            "all_non_guide_registry_fields_unchanged": True,
            "guide_weight": 0.0,
            "guide_shadow_only": True,
            "guide_runtime_authority": False,
            "guide_blocking_authority": False,
            "active_bootstrap_interrupted": False,
        },
    }
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        for key in ("schema", "owner_revision", "family_registry", "guide_retired_registry", "merged_registry"):
            if existing.get(key) != receipt.get(key):
                raise RuntimeError("existing family-guide-shadow receipt changed")
        receipt = existing
    else:
        immutable_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-registry", type=Path, required=True)
    parser.add_argument("--guide-retired-registry", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--environment-drop-in", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    print(json.dumps(stage(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
