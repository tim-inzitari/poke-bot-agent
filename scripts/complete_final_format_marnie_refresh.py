#!/usr/bin/env python3
"""Register completed Marnie H10 and start the post-Marnie Crustle chain."""

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


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
HANDLER_SCHEMA = "poke_bot.passed_gate_handler/v1"
COMPLETION_SCHEMA = "poke_bot.post_fleet_specialist_refresh_completion/v1"
REGISTRY_SCHEMA = "poke_bot.post_fleet_refresh_registry/v1"
REGISTRATION_SCHEMA = "poke_bot.post_fleet_refresh_registration/v1"
BOOTSTRAP_SCHEMA = "poke_bot.final_format_marnie_h10_bootstrap_ready/v1"
CORE_BOUNDARY_SCHEMA = "poke_bot.post_alakazam_core_refresh_boundary/v1"
RUNTIME_REGISTRATION_SCHEMA = (
    "poke_bot.final_format_marnie_h10_runtime_registration/v1"
)
ORIGINAL_MARNIE = (
    "sha256:52a5207e4c98dce80b49b6403cbb17f14d6fc4d2ac5b625532020a1a25f233ac"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _validated_population_inputs(
    *,
    handler: Mapping[str, Any],
    runtime_registration: Mapping[str, Any],
    checkpoint_digest: str,
) -> dict[str, str]:
    registry_path = Path(
        str(runtime_registration.get("runtime_registry") or "")
    ).resolve()
    registry = _read(registry_path)
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
        or _sha256(registry_path)
        != runtime_registration.get("runtime_registry_sha256")
        or runtime_row.get("status") != "ready"
        or not expert.is_file()
        or _sha256(expert).removeprefix("sha256:")
        != str(runtime_row.get("expert_manifest_sha256") or "")
        or not tree.is_file()
        or _sha256(tree).removeprefix("sha256:")
        != str(runtime_row.get("matchup_runtime_tree_sha256") or "")
        or bundle.get("specialist_id") != SPECIALIST_ID
        or bundle.get("turn_order_preference") != "first_if_allowed"
        or str((bundle.get("contents") or {}).get("model_sha256") or "")
        != checkpoint_digest
        or str(bundle.get("sha256") or "") != bundle_digest
        or queued.get("checkpoint_checksum") != checkpoint_digest
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle_digest
    ):
        raise RuntimeError("Marnie population runtime/package identity is invalid")
    return {
        "expert_manifest": str(expert),
        "expert_manifest_sha256": _sha256(expert),
        "matchup_runtime_tree": str(tree),
        "matchup_runtime_tree_sha256": _sha256(tree),
        "submission_bundle": str(bundle_path),
        "submission_bundle_sha256": bundle_digest,
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    body = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"refusing to replace immutable receipt: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    body = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    body = yaml.safe_dump(dict(value), sort_keys=False)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _disposition(gate: Mapping[str, Any]) -> dict[str, Any]:
    authority = str(gate.get("completion_authority") or "")
    if authority == "measured_both_gates_pass":
        return {
            "status": "passed_frozen_registered",
            "completion_authority": "measured_gate_pass",
            "current_gate_pass": True,
            "measured_gate_pass": True,
            "failed_gate_results_preserved": False,
        }
    if authority == "explicit_owner_ceiling_acceptance":
        return {
            "status": "ceiling_accepted_frozen_registered",
            "completion_authority": authority,
            "current_gate_pass": False,
            "measured_gate_pass": False,
            "failed_gate_results_preserved": True,
        }
    raise RuntimeError("Marnie completion authority is absent or invalid")


def _validate_static(args: argparse.Namespace) -> dict[str, str]:
    always = {
        "original_checkpoint": args.original_checkpoint.expanduser().resolve(),
        "protocol": args.protocol.expanduser().resolve(),
        "specialist_state": args.specialist_state.expanduser().resolve(),
        "next_unit": args.next_unit.expanduser().resolve(),
    }
    missing = [name for name, path in always.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Marnie completion path is missing: " + ",".join(missing))
    if _sha256(always["original_checkpoint"]) != ORIGINAL_MARNIE:
        raise RuntimeError("immutable historical Marnie checkpoint changed")
    dynamic = {
        "handler_state": args.handler_state.expanduser().resolve(),
        "bootstrap_ready": args.bootstrap_ready.expanduser().resolve(),
        "core_boundary": args.core_boundary.expanduser().resolve(),
        "runtime_registration": args.runtime_registration.expanduser().resolve(),
    }
    if not args.check:
        absent = [name for name, path in dynamic.items() if not path.is_file()]
        if absent:
            raise RuntimeError(
                "Marnie completion boundary is incomplete: " + ",".join(absent)
            )
    return {
        name: _sha256(path)
        for name, path in {**always, **dynamic}.items()
        if path.is_file()
    }


def _handler_is_terminal_and_queued(handler: dict[str, Any]) -> bool:
    phase = str(handler.get("phase") or "")
    phase_is_terminal = phase == "submissions_queued" or (
        phase == "complete_handoff_started"
        and handler.get("handoff_started") is True
    )
    return bool(
        handler.get("schema") == HANDLER_SCHEMA
        and phase_is_terminal
        and handler.get("submission_mode") == "queue_and_continue"
        and len(handler.get("queued_submissions") or []) == 1
    )


def complete(args: argparse.Namespace) -> dict[str, Any]:
    static = _validate_static(args)
    handler = _read(args.handler_state.expanduser().resolve())
    if not _handler_is_terminal_and_queued(handler):
        raise RuntimeError("Marnie gate handler is not terminal and queued")
    gate = dict(handler.get("gate") or {})
    frozen_row = dict(handler.get("frozen_model") or {})
    terminal_iteration = int(gate.get("iteration", -1))
    final_checkpoint = Path(str(frozen_row.get("model_path") or "")).resolve()
    final_digest = str(frozen_row.get("checkpoint_digest") or "")
    final_family = final_checkpoint.parent
    manifest = final_family / "manifest.json"
    disposition = _disposition(gate)
    verified = verify_frozen_model(final_family)
    if (
        terminal_iteration != 20
        or gate.get("checkpoint_digest") != final_digest
        or verified.get("checkpoint_digest") != final_digest
        or Path(str(verified.get("model_path") or "")).resolve() != final_checkpoint
        or checkpoint.checkpoint_digest(final_checkpoint) != final_digest
        or final_digest == ORIGINAL_MARNIE
        or not manifest.is_file()
    ):
        raise RuntimeError("Marnie final frozen identity is invalid")
    _validate_checkpoint(final_checkpoint, final_digest)

    bootstrap = _read(args.bootstrap_ready.expanduser().resolve())
    boundary = _read(args.core_boundary.expanduser().resolve())
    runtime_registration = _read(args.runtime_registration.expanduser().resolve())
    bootstrap_checkpoint = Path(str(bootstrap.get("checkpoint") or "")).resolve()
    if (
        bootstrap.get("schema") != BOOTSTRAP_SCHEMA
        or bootstrap.get("status") != "ready_for_managed_rl_registration"
        or bootstrap.get("specialist_id") != SPECIALIST_ID
        or int(bootstrap.get("epochs_completed") or 0) != 25
        or bootstrap.get("capacity_profile") != "H10-I/v1"
        or bootstrap.get("decision_fusion_schema")
        != "poke_bot.causal_decision_fusion/v3"
        or int(bootstrap.get("learned_head_count") or 0) != 19
        or int(bootstrap.get("learned_route_count") or 0) != 19
        or boundary.get("schema") != CORE_BOUNDARY_SCHEMA
        or boundary.get("status") != "selected_core_ready_for_marnie_h10"
        or boundary.get("selected_core_checkpoint_sha256")
        != bootstrap.get("core_checkpoint_sha256")
        or runtime_registration.get("schema") != RUNTIME_REGISTRATION_SCHEMA
        or runtime_registration.get("status") != "registered_ready_for_managed_rl"
        or runtime_registration.get("checkpoint_sha256")
        != bootstrap.get("checkpoint_sha256")
        or Path(str(runtime_registration.get("checkpoint") or "")).resolve()
        != bootstrap_checkpoint
    ):
        raise RuntimeError("Marnie H10 bootstrap/runtime lineage is invalid")
    population = _validated_population_inputs(
        handler=handler,
        runtime_registration=runtime_registration,
        checkpoint_digest=final_digest,
    )

    registry_path = args.refresh_registry.expanduser().resolve()
    registry = _read(registry_path)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("post-fleet refresh registry schema changed")
    entries = [dict(row) for row in registry.get("refreshes") or []]
    entry_ids = [existing.get("specialist_id") for existing in entries]
    if entry_ids not in (["alakazam"], ["alakazam", SPECIALIST_ID]):
        raise RuntimeError("Alakazam refresh must be registered before Marnie")
    row = {
        "specialist_id": SPECIALIST_ID,
        "refresh_model_version": "final-format-marnie-r104-h10-v1",
        "checkpoint": str(final_checkpoint),
        "checkpoint_checksum": final_digest,
        "frozen_manifest": str(manifest),
        "frozen_manifest_sha256": _sha256(manifest),
        "original_checkpoint_checksum": ORIGINAL_MARNIE,
        "bootstrap_checkpoint_checksum": str(bootstrap["checkpoint_sha256"]),
        "resolved_core_checkpoint_checksum": str(
            bootstrap["core_checkpoint_sha256"]
        ),
        "gate_id": str(gate.get("gate_id") or ""),
        "gate_iteration": terminal_iteration,
        "completion_authority": disposition["completion_authority"],
        "measured_gate_pass": disposition["measured_gate_pass"],
        "package_preference": "first_if_allowed",
        "historical_specialist_row_replaced": False,
        "capacity_profile": "H10-I/v1",
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "learned_head_count": 19,
        "learned_route_count": 19,
        **population,
    }
    if entry_ids == ["alakazam", SPECIALIST_ID] and entries[1] != row:
        raise RuntimeError("a different Marnie refresh is already registered")
    registry["refreshes"] = entries if len(entries) == 2 else [*entries, row]
    registry["ordered_refresh_ids"] = ["alakazam", SPECIALIST_ID, "crustle"]
    registry["historical_frozen_registry_modified"] = False
    _atomic_json(registry_path, registry)
    registry_digest = _sha256(registry_path)

    registration = {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered_separate_refresh_derivative",
        "specialist_id": SPECIALIST_ID,
        "refresh_model_version": row["refresh_model_version"],
        "checkpoint_checksum": final_digest,
        "registry": str(registry_path),
        "registry_sha256": registry_digest,
        "historical_frozen_registry_modified": False,
        "original_checkpoint_checksum": ORIGINAL_MARNIE,
        "completion_authority": disposition["completion_authority"],
        "measured_gate_pass": disposition["measured_gate_pass"],
        "capacity_profile": "H10-I/v1",
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
    }
    registration_path = args.registration_receipt.expanduser().resolve()
    _write_once(registration_path, registration)
    completion = {
        "schema": COMPLETION_SCHEMA,
        **disposition,
        "specialist_id": SPECIALIST_ID,
        "refresh_model_version": row["refresh_model_version"],
        "refresh_checkpoint_checksum": final_digest,
        "original_checkpoint_checksum": ORIGINAL_MARNIE,
        "bootstrap_checkpoint_checksum": str(bootstrap["checkpoint_sha256"]),
        "frozen": True,
        "registered": True,
        "gate_receipt_sha256": str(gate.get("commit_file_sha256") or ""),
        "freeze_receipt_sha256": _sha256(manifest),
        "registration_receipt_sha256": _sha256(registration_path),
        "resolved_core": {
            "status": "checksum_accepted",
            "checkpoint_checksum": str(bootstrap["core_checkpoint_sha256"]),
            "boundary_receipt_sha256": static["core_boundary"],
        },
        "training_contract": {
            "canonical_source": "config/rl_protocol.yaml#/specialist_training",
            "sha256": static["protocol"],
            "bootstrap_epochs": 25,
            "games_per_iteration": 8192,
            "rl_epochs_per_iteration": 5,
            "expert_rehearsal_epochs_per_iteration": 5,
            "terminal_iteration": 20,
            "collect_iteration_21": False,
        },
        "capacity_profile": "H10-I/v1",
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "package_preference": "first_if_allowed",
        "historical_marnie_rewritten": False,
    }
    gate_receipt = str(completion["gate_receipt_sha256"])
    if not gate_receipt.startswith("sha256:"):
        raise RuntimeError("Marnie terminal gate receipt digest is absent")
    completion_path = args.completion_receipt.expanduser().resolve()
    _write_once(completion_path, completion)

    state_path = args.specialist_state.expanduser().resolve()
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    phase = dict(state.get("post_fleet_refresh") or {})
    completed_ids = list(phase.get("completed_refresh_specialist_ids") or [])
    if completed_ids not in (["alakazam"], ["alakazam", SPECIALIST_ID]):
        raise RuntimeError("post-fleet refresh sequence is not at Marnie")
    prior_receipts = list(phase.get("completed_refresh_receipts") or [])
    if not prior_receipts or prior_receipts[0].get("specialist_id") != "alakazam":
        raise RuntimeError("Alakazam completion receipt projection is invalid")
    marnie_receipt = {
        "specialist_id": SPECIALIST_ID,
        "receipt": str(completion_path),
        "receipt_sha256": _sha256(completion_path),
    }
    if completed_ids == ["alakazam", SPECIALIST_ID]:
        if len(prior_receipts) != 2 or prior_receipts[1] != marnie_receipt:
            raise RuntimeError("Marnie completion receipt projection changed")
    else:
        phase["completed_refresh_specialist_ids"] = ["alakazam", SPECIALIST_ID]
        phase["completed_refresh_receipts"] = [*prior_receipts, marnie_receipt]
    phase["pending_specialist_ids"] = ["crustle"]
    phase["active_refresh_specialist_id"] = None
    phase["next_refresh_specialist_id"] = "crustle"
    phase["status"] = "marnie_complete_crustle_h10_bootstrap_starting"
    versions = dict(phase.get("refresh_model_versions") or {})
    versions[SPECIALIST_ID] = row["refresh_model_version"]
    phase["refresh_model_versions"] = versions
    state["post_fleet_refresh"] = phase
    _atomic_yaml(state_path, state)

    started = subprocess.run(
        ["systemctl", "--user", "--no-block", "start", args.next_service],
        check=False,
        capture_output=True,
        text=True,
    )
    if started.returncode:
        raise RuntimeError(
            "could not start post-Marnie Crustle bootstrap: "
            + started.stdout.strip()
            + started.stderr.strip()
        )
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--handler-state", type=Path, required=True)
    parser.add_argument("--bootstrap-ready", type=Path, required=True)
    parser.add_argument("--core-boundary", type=Path, required=True)
    parser.add_argument("--runtime-registration", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--specialist-state", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--registration-receipt", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--next-service", required=True)
    parser.add_argument("--next-unit", type=Path, required=True)
    args = parser.parse_args()
    static = _validate_static(args)
    if args.check:
        print("FINAL_FORMAT_MARNIE_COMPLETION_OK " + json.dumps(static, sort_keys=True))
        return 0
    print(json.dumps(complete(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
