#!/usr/bin/env python3
"""Freeze/register terminal Crustle H10 and release the capacity barrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from poke_bot import checkpoint
from poke_bot.pure_rl.model_registry import verify_frozen_model
from scripts.register_final_format_marnie_h10_rl import _validate_checkpoint


SPECIALIST_ID = "crustle"
HANDLER_SCHEMA = "poke_bot.passed_gate_handler/v1"
READY_SCHEMA = "poke_bot.specialist_expert_bootstrap_ready/v1"
REGISTRY_SCHEMA = "poke_bot.post_fleet_refresh_registry/v1"
COMPLETION_SCHEMA = "poke_bot.post_fleet_specialist_refresh_completion/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    body = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable Crustle receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def validate_static(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "bootstrap_ready": args.bootstrap_ready.resolve(),
        "runtime_registry": args.runtime_registry.resolve(),
        "protocol": args.protocol.resolve(),
        "specialist_state": args.specialist_state.resolve(),
        "next_unit": args.next_unit.resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Crustle completion inputs missing: " + ",".join(missing))
    return {name: sha256(path) for name, path in paths.items()}


def validated_population_inputs(
    *,
    handler: Mapping[str, Any],
    checkpoint_digest: str,
    runtime_registry_path: Path,
) -> dict[str, str]:
    """Bind the exact Crustle package and causal training-side assets.

    The population phase materializes the newly completed H10 Crustle from its
    final-format submission bundle.  The historical public Crustle agent is
    deliberately not used as a population member or selected-history entry.
    """

    registry_path = runtime_registry_path.resolve()
    registry = read(registry_path)
    runtime_row = dict(
        (registry.get("specialists") or {}).get(SPECIALIST_ID) or {}
    )
    expert = Path(str(runtime_row.get("expert_manifest") or "")).resolve()
    tree = Path(str(runtime_row.get("matchup_runtime_tree") or "")).resolve()
    bundle = dict(handler.get("submission_bundle") or {})
    queued = dict((handler.get("queued_submissions") or [{}])[0])
    bundle_path = Path(str(queued.get("file") or "")).resolve()
    bundle_digest = str(queued.get("file_sha256") or "")
    if (
        registry.get("schema") != "poke_bot.specialist_runtime_registry/v1"
        or runtime_row.get("status") != "ready"
        or not expert.is_file()
        or sha256(expert).removeprefix("sha256:")
        != str(runtime_row.get("expert_manifest_sha256") or "")
        or not tree.is_file()
        or sha256(tree).removeprefix("sha256:")
        != str(runtime_row.get("matchup_runtime_tree_sha256") or "")
        or bundle.get("specialist_id") != SPECIALIST_ID
        or bundle.get("turn_order_preference") != "first_if_allowed"
        or str((bundle.get("contents") or {}).get("model_sha256") or "")
        != checkpoint_digest
        or str(bundle.get("sha256") or "") != bundle_digest
        or queued.get("checkpoint_checksum") != checkpoint_digest
        or not bundle_path.is_file()
        or sha256(bundle_path) != bundle_digest
    ):
        raise RuntimeError("Crustle population runtime/package identity is invalid")
    return {
        "expert_manifest": str(expert),
        "expert_manifest_sha256": sha256(expert),
        "matchup_runtime_tree": str(tree),
        "matchup_runtime_tree_sha256": sha256(tree),
        "submission_bundle": str(bundle_path),
        "submission_bundle_sha256": bundle_digest,
        "runtime_registry": str(registry_path),
        "runtime_registry_sha256": sha256(registry_path),
    }


def disposition(gate: Mapping[str, Any]) -> dict[str, Any]:
    authority = str(gate.get("completion_authority") or "")
    if authority == "measured_both_gates_pass":
        return {
            "status": "passed_frozen_registered",
            "completion_authority": "measured_gate_pass",
            "measured_gate_pass": True,
            "failed_gate_results_preserved": False,
        }
    if authority == "explicit_owner_ceiling_acceptance":
        return {
            "status": "ceiling_accepted_frozen_registered",
            "completion_authority": authority,
            "measured_gate_pass": False,
            "failed_gate_results_preserved": True,
        }
    raise RuntimeError("Crustle completion authority is absent")


def complete(args: argparse.Namespace) -> dict[str, Any]:
    static = validate_static(args)
    handler = read(args.handler_state.resolve())
    gate = dict(handler.get("gate") or {})
    frozen = dict(handler.get("frozen_model") or {})
    queued = list(handler.get("queued_submissions") or [])
    model = Path(str(frozen.get("model_path") or "")).resolve()
    digest = str(frozen.get("checkpoint_digest") or "")
    family = model.parent
    manifest = family / "manifest.json"
    if (
        handler.get("schema") != HANDLER_SCHEMA
        or handler.get("submission_mode") != "queue_and_continue"
        or len(queued) != 1
        or int(gate.get("iteration", -1)) != 15
        or gate.get("checkpoint_digest") != digest
        or not model.is_file()
        or checkpoint.checkpoint_digest(model) != digest
        or not manifest.is_file()
    ):
        raise RuntimeError("Crustle terminal handler identity is invalid")
    verified = verify_frozen_model(family)
    if verified.get("checkpoint_digest") != digest:
        raise RuntimeError("Crustle frozen model verification failed")
    _validate_checkpoint(model, digest)
    outcome = disposition(gate)

    ready = read(args.bootstrap_ready.resolve())
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("acting_seat_archetype") != SPECIALIST_ID
        or int(ready.get("epochs_completed") or 0) != 25
        or ready.get("status") != "ready"
    ):
        raise RuntimeError("Crustle bootstrap lineage is invalid")
    runtime = read(args.runtime_registry.resolve())
    runtime_row = dict((runtime.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if (
        runtime.get("schema") != "poke_bot.specialist_runtime_registry/v1"
        or runtime_row.get("status") != "ready"
        or int(runtime.get("iteration_ceiling", -1)) != 15
        or int(runtime_row.get("iteration_ceiling", 15)) != 15
        or dict(runtime_row.get("decision_fusion") or {}).get("runtime_enabled")
        is not True
        or len(dict(runtime_row.get("decision_fusion") or {}).get("required_heads") or [])
        != 19
    ):
        raise RuntimeError("Crustle H10 runtime contract is invalid")
    population = validated_population_inputs(
        handler=handler,
        checkpoint_digest=digest,
        runtime_registry_path=args.runtime_registry,
    )

    registry_path = args.refresh_registry.resolve()
    registry = read(registry_path)
    rows = [dict(row) for row in registry.get("refreshes") or []]
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or [row.get("specialist_id") for row in rows]
        not in (
            ["alakazam", "marnie-s-grimmsnarl-ex"],
            ["alakazam", "marnie-s-grimmsnarl-ex", SPECIALIST_ID],
        )
    ):
        raise RuntimeError("Crustle must follow the two completed H10 refreshes")
    row = {
        "specialist_id": SPECIALIST_ID,
        "refresh_model_version": "final-format-crustle-r113-h10-v1",
        "checkpoint": str(model),
        "checkpoint_checksum": digest,
        "frozen_manifest": str(manifest),
        "frozen_manifest_sha256": sha256(manifest),
        "bootstrap_checkpoint_checksum": str(ready.get("checkpoint_digest") or ""),
        "gate_id": str(gate.get("gate_id") or ""),
        "gate_iteration": 15,
        "completion_authority": outcome["completion_authority"],
        "measured_gate_pass": outcome["measured_gate_pass"],
        "package_preference": "first_if_allowed",
        "capacity_profile": "H10-I/v1",
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "public_baseline_substituted": False,
        **population,
    }
    if len(rows) == 3 and rows[2] != row:
        raise RuntimeError("a different Crustle completion is already registered")
    registry["refreshes"] = rows if len(rows) == 3 else [*rows, row]
    registry["ordered_refresh_ids"] = [
        "alakazam",
        "marnie-s-grimmsnarl-ex",
        SPECIALIST_ID,
    ]
    registry["historical_frozen_registry_modified"] = False
    atomic_json(registry_path, registry)

    completion = {
        "schema": COMPLETION_SCHEMA,
        **outcome,
        "specialist_id": SPECIALIST_ID,
        "refresh_model_version": row["refresh_model_version"],
        "refresh_checkpoint_checksum": digest,
        "bootstrap_checkpoint_checksum": row["bootstrap_checkpoint_checksum"],
        "frozen": True,
        "registered": True,
        "gate_receipt_sha256": str(gate.get("commit_file_sha256") or ""),
        "freeze_receipt_sha256": sha256(manifest),
        "registry_sha256": sha256(registry_path),
        "training_contract": {
            "canonical_source": "config/rl_protocol.yaml#/specialist_training",
            "sha256": static["protocol"],
            "bootstrap_epochs": 25,
            "games_per_iteration": 8192,
            "rl_epochs_per_iteration": 5,
            "expert_rehearsal_epochs_per_iteration": 5,
            "terminal_iteration": 15,
            "collect_iteration_16": False,
        },
        "capacity_profile": "H10-I/v1",
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "public_baseline_substituted": False,
    }
    if not str(completion["gate_receipt_sha256"]).startswith("sha256:"):
        raise RuntimeError("Crustle gate commit digest is absent")
    write_once(args.completion_receipt.resolve(), completion)

    state_path = args.specialist_state.resolve()
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    phase = dict(state.get("post_fleet_refresh") or {})
    prior = list(phase.get("completed_refresh_specialist_ids") or [])
    if prior not in (
        ["alakazam", "marnie-s-grimmsnarl-ex"],
        ["alakazam", "marnie-s-grimmsnarl-ex", SPECIALIST_ID],
    ):
        raise RuntimeError("post-fleet state is not at Crustle completion")
    receipt_row = {
        "specialist_id": SPECIALIST_ID,
        "receipt": str(args.completion_receipt.resolve()),
        "receipt_sha256": sha256(args.completion_receipt.resolve()),
    }
    receipts = list(phase.get("completed_refresh_receipts") or [])
    if len(prior) == 2:
        phase["completed_refresh_specialist_ids"] = [*prior, SPECIALIST_ID]
        phase["completed_refresh_receipts"] = [*receipts, receipt_row]
    elif not receipts or receipts[-1] != receipt_row:
        raise RuntimeError("Crustle completion projection changed")
    phase["pending_specialist_ids"] = []
    phase["active_refresh_specialist_id"] = None
    phase["next_refresh_specialist_id"] = None
    phase["status"] = "refresh_sequence_complete_population_handoff_pending"
    versions = dict(phase.get("refresh_model_versions") or {})
    versions[SPECIALIST_ID] = row["refresh_model_version"]
    phase["refresh_model_versions"] = versions
    state["post_fleet_refresh"] = phase
    atomic_yaml(state_path, state)

    started = subprocess.run(
        ["systemctl", "--user", "--no-block", "start", args.next_service],
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode:
        raise RuntimeError("could not start capacity boundary: " + started.stderr.strip())
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--handler-state", type=Path, required=True)
    parser.add_argument("--bootstrap-ready", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--specialist-state", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--next-service", required=True)
    parser.add_argument("--next-unit", type=Path, required=True)
    args = parser.parse_args()
    static = validate_static(args)
    if args.check:
        print("POST_MARNIE_CRUSTLE_COMPLETION_OK " + json.dumps(static, sort_keys=True))
        return 0
    print(json.dumps(complete(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
