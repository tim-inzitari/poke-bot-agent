#!/usr/bin/env python3
"""Stage a versioned Marnie runtime registry for a newer expert corpus.

This command never changes a selector, systemd unit, active registry, or
training process.  It validates the checksum-staged latest-20 corpus and emits
immutable candidate registry/registration/stage receipts for a later clean
rehearsal-boundary activation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
SYNC_SCHEMA = "poke_bot.latest20_specialist_sync/v1"
REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"
REGISTRATION_SCHEMA = (
    "poke_bot.final_format_marnie_h10_runtime_registration/v1"
)
POINTER_SCHEMA = "poke_bot.pinned_expert_corpus/v1"
TARGET_SCHEMA = "poke_bot.expanded_strategic_targets/v2"
TARGET_DIGEST = (
    "sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f"
)
MIGRATION_REASON = "receipt_backed_latest20_rehearsal_boundary_r109"
EXPECTED_DATES = [
    "2026-07-14",
    "2026-07-15",
    "2026-07-16",
    "2026-07-17",
    "2026-07-18",
    "2026-07-19",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
    "2026-07-23",
    "2026-07-24",
    "2026-07-25",
    "2026-07-26",
    "2026-07-27",
    "2026-07-28",
    "2026-07-29",
    "2026-07-30",
    "2026-07-31",
    "2026-08-01",
    "2026-08-02",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sync_receipt_ready(path: Path) -> bool:
    """Return true only after the mutable transfer receipt reaches ready."""
    try:
        return read_json(path).get("status") == "ready"
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return False


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    body = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"refusing to replace immutable artifact: {path}")
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


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    sync_path = args.sync_receipt.expanduser().resolve()
    source_registry_path = args.source_registry.expanduser().resolve()
    source_registration_path = args.source_registration.expanduser().resolve()
    output_registry_path = args.output_registry.expanduser().resolve()
    output_registration_path = args.output_registration.expanduser().resolve()
    stage_receipt_path = args.stage_receipt.expanduser().resolve()

    sync = read_json(sync_path)
    destination = Path(str(sync.get("destination") or "")).resolve()
    current_pointer = Path(str(sync.get("current_pointer") or "")).resolve()
    if (
        sync.get("schema") != SYNC_SCHEMA
        or sync.get("status") != "ready"
        or list(sync.get("dates") or ()) != EXPECTED_DATES
        or int(sync.get("specialist_count") or 0) != 18
        or sync.get("expanded_target_schema") != TARGET_SCHEMA
        or sync.get("expanded_target_digest") != TARGET_DIGEST
        or int(sync.get("copied_bytes") or -1)
        != int(sync.get("source_bytes") or -2)
        or int(sync.get("source_bytes") or 0) <= 0
        or not destination.is_dir()
        or not current_pointer.is_dir()
        or current_pointer != destination
    ):
        raise RuntimeError("latest-20 strategic sync receipt is not activation-ready")

    pointer_path = destination / SPECIALIST_ID / "PROTECTED_EXPERT_CORPUS.json"
    pointer = read_json(pointer_path)
    manifest_path = pointer_path.parent / str(pointer.get("manifest") or "")
    manifest = read_json(manifest_path)
    expanded = manifest.get("expanded_strategic_targets")
    totals = dict(manifest.get("totals") or {})
    decisions = int(totals.get("decisions_kept") or 0)
    if (
        pointer.get("schema") != POINTER_SCHEMA
        or pointer.get("protected") is not True
        or not manifest_path.is_file()
        or sha256(manifest_path) != pointer.get("manifest_sha256")
        or not isinstance(expanded, dict)
        or expanded.get("schema") != TARGET_SCHEMA
        or expanded.get("digest") != TARGET_DIGEST
        or int(expanded.get("decisions") or -1) != decisions
        or decisions < int(args.minimum_decisions)
    ):
        raise RuntimeError("Marnie latest-20 protected corpus identity is invalid")

    source_registry = read_json(source_registry_path)
    source_registration = read_json(source_registration_path)
    source_rows = dict(source_registry.get("specialists") or {})
    source_row = dict(source_rows.get(SPECIALIST_ID) or {})
    if (
        source_registry.get("schema") != REGISTRY_SCHEMA
        or source_registration.get("schema") != REGISTRATION_SCHEMA
        or source_registration.get("status")
        != "registered_ready_for_managed_rl"
        or Path(str(source_registration.get("runtime_registry") or "")).resolve()
        != source_registry_path
        or source_registration.get("runtime_registry_sha256")
        != sha256(source_registry_path)
        or source_row.get("status") != "ready"
    ):
        raise RuntimeError("source Marnie runtime registration is not authoritative")

    candidate_registry = json.loads(json.dumps(source_registry))
    common_args = candidate_registry.get("common_trainer_args")
    if not isinstance(common_args, list):
        raise RuntimeError("source registry has no common trainer arguments")
    reason_flag = "--boundary-design-migration-reason"
    if common_args.count(reason_flag) != 1:
        raise RuntimeError("source registry has ambiguous migration authority")
    reason_index = common_args.index(reason_flag) + 1
    if reason_index >= len(common_args) or not str(common_args[reason_index]):
        raise RuntimeError("source registry migration reason is missing")
    common_args[reason_index] = MIGRATION_REASON
    candidate_row = candidate_registry["specialists"][SPECIALIST_ID]
    old_manifest = str(candidate_row.get("expert_manifest") or "")
    old_manifest_digest = str(candidate_row.get("expert_manifest_sha256") or "")
    candidate_row["expert_manifest"] = str(pointer_path)
    candidate_row["expert_manifest_sha256"] = sha256(pointer_path).removeprefix(
        "sha256:"
    )
    candidate_registry["expert_corpus_activation"] = {
        "status": "staged_for_rehearsal_boundary",
        "window_start": EXPECTED_DATES[0],
        "window_end": EXPECTED_DATES[-1],
        "sync_receipt": str(sync_path),
        "sync_receipt_sha256": sha256(sync_path),
        "protected_pointer": str(pointer_path),
        "protected_pointer_sha256": sha256(pointer_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "active_before_boundary": False,
        "boundary_design_migration_reason": MIGRATION_REASON,
    }
    write_once(output_registry_path, candidate_registry)

    candidate_registration = json.loads(json.dumps(source_registration))
    candidate_registration["runtime_registry"] = str(output_registry_path)
    candidate_registration["runtime_registry_sha256"] = sha256(
        output_registry_path
    )
    candidate_registration["expert_corpus_activation"] = {
        "status": "staged_for_rehearsal_boundary",
        "sync_receipt": str(sync_path),
        "sync_receipt_sha256": sha256(sync_path),
        "protected_pointer": str(pointer_path),
        "protected_pointer_sha256": sha256(pointer_path),
        "manifest_sha256": sha256(manifest_path),
        "active_before_boundary": False,
        "boundary_design_migration_reason": MIGRATION_REASON,
    }
    write_once(output_registration_path, candidate_registration)

    if stage_receipt_path.exists():
        existing = read_json(stage_receipt_path)
        if (
            existing.get("schema")
            != "poke_bot.marnie_latest20_runtime_migration_stage/v1"
            or existing.get("sync_receipt_sha256") != sha256(sync_path)
            or existing.get("candidate_registry_sha256")
            != sha256(output_registry_path)
            or existing.get("candidate_registration_sha256")
            != sha256(output_registration_path)
            or existing.get("candidate_expert_pointer_sha256")
            != sha256(pointer_path)
        ):
            raise RuntimeError(
                f"existing migration stage receipt changed: {stage_receipt_path}"
            )
        return existing

    receipt = {
        "schema": "poke_bot.marnie_latest20_runtime_migration_stage/v1",
        "status": "ready_for_clean_rehearsal_boundary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specialist_id": SPECIALIST_ID,
        "window_start": EXPECTED_DATES[0],
        "window_end": EXPECTED_DATES[-1],
        "sync_receipt": str(sync_path),
        "sync_receipt_sha256": sha256(sync_path),
        "old_registry": str(source_registry_path),
        "old_registry_sha256": sha256(source_registry_path),
        "candidate_registry": str(output_registry_path),
        "candidate_registry_sha256": sha256(output_registry_path),
        "candidate_registration": str(output_registration_path),
        "candidate_registration_sha256": sha256(output_registration_path),
        "old_expert_manifest": old_manifest,
        "old_expert_manifest_sha256": old_manifest_digest,
        "candidate_expert_pointer": str(pointer_path),
        "candidate_expert_pointer_sha256": sha256(pointer_path),
        "candidate_expert_manifest": str(manifest_path),
        "candidate_expert_manifest_sha256": sha256(manifest_path),
        "decisions": decisions,
        "selector_modified": False,
        "managed_units_modified": False,
        "active_registry_modified": False,
        "training_interrupted": False,
        "activation_boundary": "first_available_future_five_iteration_hard_pause",
        "boundary_design_migration_reason": MIGRATION_REASON,
    }
    write_once(stage_receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-receipt", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--source-registration", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--output-registration", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--minimum-decisions", type=int, default=100_000)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.wait_seconds > 0:
        deadline = time.monotonic() + args.wait_seconds
        sync_receipt = args.sync_receipt.expanduser()
        while not sync_receipt_ready(sync_receipt):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for ready strategic sync receipt"
                )
            time.sleep(max(1.0, args.poll_seconds))
    print(json.dumps(prepare(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
