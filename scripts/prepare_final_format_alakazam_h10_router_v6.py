#!/usr/bin/env python3
"""Add the current dormant Router Format 6 bank to the immutable H10 child."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from poke_bot import checkpoint
from poke_bot.matchup_adapters_v6 import (
    LEGACY_V5_PREFIX_LENGTH,
    SLOT_CAPACITY,
    load_slot_registry,
    migrate_v5_checkpoint_payload,
    registry_digest,
)
from poke_bot.train import load_model_from_checkpoint
from scripts.migrate_final_format_alakazam_h10 import _role_inventory


SCHEMA = "poke_bot.final_format_alakazam_h10_router_v6_migration/v1"
HANDOFF_SCHEMA = "poke_bot.final_format_alakazam_h10_router_v6_handoff/v1"
ADAPTER_PREFIX = "matchup_adapter_bank."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _non_adapter_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(payload.get("model_state_dict") or {}).items()
        if not key.startswith(ADAPTER_PREFIX)
    }


def _tensor_maps_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def prepare(
    *,
    source: Path,
    expected_source_sha256: str,
    source_handoff: Path,
    expected_source_handoff_sha256: str,
    registry_path: Path,
    output: Path,
    role_route_receipt: Path,
    migration_receipt: Path,
    handoff_receipt: Path,
) -> dict[str, Any]:
    source = source.resolve()
    source_handoff = source_handoff.resolve()
    registry_path = registry_path.resolve()
    output = output.resolve()
    if output == source:
        raise RuntimeError("Router Format 6 derivative may not replace its source")
    if _sha256(source) != expected_source_sha256:
        raise RuntimeError("H10 source checkpoint digest changed")
    if _sha256(source_handoff) != expected_source_handoff_sha256:
        raise RuntimeError("H10 source handoff digest changed")
    source_handoff_payload = json.loads(source_handoff.read_text(encoding="utf-8"))
    if (
        source_handoff_payload.get("schema")
        != "poke_bot.final_format_alakazam_h10_handoff/v1"
        or source_handoff_payload.get("status") != "h10_migrated_training_pending"
        or source_handoff_payload.get("child_checkpoint_sha256")
        != expected_source_sha256
        or source_handoff_payload.get("selector_eligible") is not False
        or source_handoff_payload.get("kaggle_eligible") is not False
    ):
        raise RuntimeError("H10 source handoff is not the isolated training parent")

    registry = load_slot_registry(registry_path)
    source_payload = checkpoint.load_checkpoint(source, map_location="cpu")
    migrated = migrate_v5_checkpoint_payload(source_payload, registry=registry)
    migrated_config = dict(migrated.get("model_config") or {})
    if migrated_config.get("matchup_adapters_enabled") is not False:
        raise RuntimeError("Router Format 6 migration unexpectedly activated adapters")

    source_state = dict(source_payload.get("model_state_dict") or {})
    target_state = dict(migrated.get("model_state_dict") or {})
    if not _tensor_maps_equal(
        _non_adapter_state(source_payload),
        _non_adapter_state(migrated),
    ):
        raise RuntimeError("Router Format 6 migration changed a non-adapter tensor")
    retained = 0
    appended_zero = 0
    for key, value in target_state.items():
        if not key.startswith(ADAPTER_PREFIX):
            continue
        route = int(key.removeprefix(ADAPTER_PREFIX).split(".")[1])
        if route < LEGACY_V5_PREFIX_LENGTH:
            if not torch.equal(value, source_state[key]):
                raise RuntimeError("Router Format 6 changed a retained adapter tensor")
            retained += 1
        else:
            if int(value.count_nonzero().item()) != 0:
                raise RuntimeError("Router Format 6 appended a nonzero dormant slot")
            appended_zero += 1

    with tempfile.TemporaryDirectory(prefix="alakazam-h10-v6-") as raw_temp:
        temporary_checkpoint = Path(raw_temp) / "candidate.pt"
        torch.save(migrated, temporary_checkpoint)
        model = load_model_from_checkpoint(
            temporary_checkpoint,
            device=torch.device("cpu"),
        )
        role_inventory = _role_inventory(model)
    role_inventory.update(
        {
            "status": "issued_step_zero_router_v6_training_pending",
            "adapter_format": "poke-bot-matchup-adapter-bank-v6",
            "adapter_slot_capacity": SLOT_CAPACITY,
            "active_logical_matchup_routes": len(registry["active_expert_ids"]),
            "matchup_adapters_runtime_enabled": False,
        }
    )
    _write_json_exclusive(role_route_receipt, role_inventory)
    role_sha = _sha256(role_route_receipt)

    extra = dict(migrated.get("extra") or {})
    extra["final_format_role_route_inventory_sha256"] = role_sha
    extra["h10_router_v6_migration"] = {
        "schema": SCHEMA,
        "source_checkpoint_sha256": expected_source_sha256,
        "source_handoff_sha256": expected_source_handoff_sha256,
        "adapter_format": "poke-bot-matchup-adapter-bank-v6",
        "slot_registry_digest": registry_digest(registry),
        "slot_capacity": SLOT_CAPACITY,
        "active_logical_routes": len(registry["active_expert_ids"]),
        "retained_v5_slots": LEGACY_V5_PREFIX_LENGTH,
        "appended_slots_exact_zero": True,
        "runtime_enabled": False,
        "step_zero_policy_behavior_preserved": True,
    }
    migrated["extra"] = extra
    checkpoint.immutable_torch_save(migrated, output)
    output_sha = _sha256(output)
    reloaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    learned_parameters = sum(parameter.numel() for parameter in reloaded.parameters())
    if len(reloaded.decision_fusion.required_heads) != 19:
        raise RuntimeError("Router Format 6 derivative lost the H10 head inventory")

    migration = {
        "schema": SCHEMA,
        "status": "passed_isolated_training_derivative_ready",
        "specialist_id": "alakazam",
        "capacity_profile": "H10-I/v1",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": expected_source_sha256,
        "source_handoff": str(source_handoff),
        "source_handoff_sha256": expected_source_handoff_sha256,
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": output_sha,
        "adapter_format": "poke-bot-matchup-adapter-bank-v6",
        "slot_registry": str(registry_path),
        "slot_registry_digest": registry_digest(registry),
        "slot_capacity": SLOT_CAPACITY,
        "active_logical_routes": len(registry["active_expert_ids"]),
        "retained_v5_adapter_tensors_bit_exact": retained,
        "appended_adapter_tensors_exact_zero": appended_zero,
        "non_adapter_model_tensors_bit_exact": len(_non_adapter_state(migrated)),
        "matchup_adapters_runtime_enabled": False,
        "step_zero_policy_behavior_preserved": True,
        "learned_parameters": learned_parameters,
        "learned_head_count": 19,
        "learned_route_count": 19,
        "guide_runtime_route_count": 0,
        "role_route_receipt": str(role_route_receipt),
        "role_route_receipt_sha256": role_sha,
        "runtime_authority": "isolated_h10_training_only",
        "production_selector_write_authority": False,
        "selector_eligible": False,
        "kaggle_eligible": False,
    }
    _write_json_exclusive(migration_receipt, migration)
    migration_sha = _sha256(migration_receipt)
    handoff = {
        "schema": HANDOFF_SCHEMA,
        "status": "h10_router_v6_migrated_training_pending",
        "specialist_id": "alakazam",
        "capacity_profile": "H10-I/v1",
        "source_h10_handoff_sha256": expected_source_handoff_sha256,
        "router_v6_migration_receipt_sha256": migration_sha,
        "role_route_receipt_sha256": role_sha,
        "child_checkpoint_sha256": output_sha,
        "learned_parameters": learned_parameters,
        "learned_head_count": 19,
        "learned_route_count": 19,
        "guide_runtime_route_count": 0,
        "adapter_format": "poke-bot-matchup-adapter-bank-v6",
        "adapter_slot_capacity": SLOT_CAPACITY,
        "active_logical_matchup_routes": len(registry["active_expert_ids"]),
        "matchup_adapters_runtime_enabled": False,
        "training_seat_split": {"first": 0.5, "second": 0.5},
        "package_preference": "first_if_allowed",
        "second_focused_arm_allowed": False,
        "runtime_authority": "isolated_h10_training_only",
        "production_selector_write_authority": False,
        "selector_eligible": False,
        "kaggle_eligible": False,
        "next_boundary": "isolated_h10_training_preflight",
    }
    _write_json_exclusive(handoff_receipt, handoff)
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-handoff", type=Path, required=True)
    parser.add_argument("--expected-source-handoff-sha256", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role-route-receipt", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--handoff-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        source=args.source,
        expected_source_sha256=args.expected_source_sha256,
        source_handoff=args.source_handoff,
        expected_source_handoff_sha256=args.expected_source_handoff_sha256,
        registry_path=args.registry,
        output=args.output,
        role_route_receipt=args.role_route_receipt,
        migration_receipt=args.migration_receipt,
        handoff_receipt=args.handoff_receipt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
