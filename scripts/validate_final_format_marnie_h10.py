#!/usr/bin/env python3
"""Validate a Marnie H10/Fusion-v3 child before any training authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.model import (  # noqa: E402
    DECISION_FUSION_V3_ROUTE_SCHEMA,
    DECISION_FUSION_V3_SCHEMA,
)
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"Marnie H10 validation receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def validate(
    *,
    child_path: Path,
    migration_path: Path,
    roles_path: Path,
) -> dict[str, Any]:
    child_path = child_path.expanduser().resolve()
    migration_path = migration_path.expanduser().resolve()
    roles_path = roles_path.expanduser().resolve()
    migration = _read(migration_path)
    roles = _read(roles_path)
    child_sha256 = _sha256(child_path)
    roles_sha256 = _sha256(roles_path)
    payload = checkpoint.load_checkpoint(child_path, map_location="cpu")
    model = load_model_from_checkpoint(child_path, device=torch.device("cpu"))
    cfg = model.cfg
    inventory = model.decision_fusion_inventory()
    routes = dict(inventory.get("dedicated_routes") or {})
    required_heads = list(inventory.get("required_heads") or [])

    if (
        migration.get("schema")
        != "poke_bot.final_format_marnie_h10_migration/v1"
        or migration.get("status")
        != "issued_step_zero_passed_training_pending"
        or migration.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or migration.get("capacity_profile") != "H10-I/v1"
        or migration.get("child_checkpoint_sha256") != child_sha256
        or migration.get("learned_head_role_map_sha256") != roles_sha256
        or migration.get("learned_route_inventory_sha256") != roles_sha256
        or migration.get("decision_fusion_schema") != DECISION_FUSION_V3_SCHEMA
        or migration.get("route_schema") != DECISION_FUSION_V3_ROUTE_SCHEMA
        or migration.get("typed_output_centered_routes") is not True
        or migration.get("positive_bounded_route_reliability") is not True
        or migration.get("route_reliability_bounds") != [0.25, 4.0]
        or float(migration.get("action_type_reliability_cap") or -1.0) != 0.25
        or int(migration.get("learned_head_count") or 0) != 19
        or int(migration.get("learned_route_count") or 0) != 19
        or dict(migration.get("step_zero_parity") or {}).get("passed") is not True
        or migration.get("runtime_authority") != "none"
        or migration.get("selector_eligible") is not False
        or migration.get("kaggle_eligible") is not False
    ):
        raise RuntimeError("Marnie H10 migration receipt is incomplete")
    if (
        roles.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or int(roles.get("learned_head_count") or 0) != 19
        or int(roles.get("learned_route_count") or 0) != 19
        or roles.get("all_learned_heads_influence_actions") is not True
        or roles.get("decision_fusion_schema") != DECISION_FUSION_V3_SCHEMA
        or roles.get("route_schema") != DECISION_FUSION_V3_ROUTE_SCHEMA
        or roles.get("typed_output_centered_routes") is not True
        or roles.get("positive_bounded_route_reliability") is not True
        or roles.get("route_reliability_bounds") != [0.25, 4.0]
        or float(roles.get("action_type_reliability_cap") or -1.0) != 0.25
    ):
        raise RuntimeError("Marnie H10 role/route inventory is incomplete")
    if (
        payload.get("archetype_id") != "marnie-s-grimmsnarl-ex"
        or (cfg.spatial_layers, cfg.temporal_layers, cfg.option_decoder_layers)
        != (7, 3, 7)
        or cfg.ff_dim != 2496
        or cfg.h10_head_residual_width != 512
        or cfg.decision_fusion_typed_output_centered_routes_enabled is not True
        or cfg.decision_fusion_action_type_reliability_cap != 0.25
        or inventory.get("schema") != DECISION_FUSION_V3_SCHEMA
        or len(required_heads) != 19
        or routes.get("schema") != DECISION_FUSION_V3_ROUTE_SCHEMA
        or routes.get("route_names") != required_heads
        or int(routes.get("route_count") or 0) != 19
        or routes.get("positive_bounded_reliability") is not True
        or routes.get("reliability_bounds") != [0.25, 4.0]
        or float(routes.get("action_type_reliability_cap") or -1.0) != 0.25
    ):
        raise RuntimeError("Marnie checkpoint is not exact H10/Fusion-v3")

    return {
        "schema": "poke_bot.final_format_marnie_h10_validation/v1",
        "status": "validated_training_pending",
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "checkpoint": str(child_path),
        "checkpoint_sha256": child_sha256,
        "migration_receipt": str(migration_path),
        "migration_receipt_sha256": _sha256(migration_path),
        "role_route_receipt": str(roles_path),
        "role_route_receipt_sha256": roles_sha256,
        "capacity_profile": "H10-I/v1",
        "architecture": {
            "spatial_layers": 7,
            "temporal_layers": 3,
            "option_decoder_layers": 7,
            "feed_forward_width": 2496,
            "strategic_head_residual_width": 512,
        },
        "decision_fusion_schema": DECISION_FUSION_V3_SCHEMA,
        "route_schema": DECISION_FUSION_V3_ROUTE_SCHEMA,
        "learned_head_count": 19,
        "learned_route_count": 19,
        "training_authority": False,
        "selector_authority": False,
        "next_boundary": "exact_25_epoch_h10_specialist_bootstrap",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--role-route-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = validate(
        child_path=args.child,
        migration_path=args.migration_receipt,
        roles_path=args.role_route_receipt,
    )
    if args.receipt is not None:
        _write_once(args.receipt.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
