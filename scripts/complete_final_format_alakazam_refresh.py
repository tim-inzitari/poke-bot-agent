#!/usr/bin/env python3
"""Register a passed Alakazam H10 refresh without rewriting its ancestor."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


HANDLER_SCHEMA = "poke_bot.passed_gate_handler/v1"
COMPLETION_SCHEMA = "poke_bot.post_fleet_specialist_refresh_completion/v1"
REGISTRY_SCHEMA = "poke_bot.post_fleet_refresh_registry/v1"
REGISTRATION_SCHEMA = "poke_bot.post_fleet_refresh_registration/v1"
SEAT_SCHEMA = "poke_bot.alakazam_refresh_seat_split/v1"
ORIGINAL_ALAKAZAM = (
    "sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f32534f99af74aece2ea"
)
ACCEPTED_CORE = (
    "sha256:7d9b60e68f4c51bb931298ae3941e5b7bddf1370566b23d18acadd33e8357056"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"refusing to replace different immutable receipt: {path}")
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


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    body = yaml.safe_dump(dict(payload), sort_keys=False)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _validate_stage(path: Path, *, iteration: int, stage: str) -> str:
    payload = _read_json(path)
    first = int(payload.get("first_games", -1))
    second = int(payload.get("second_games", -1))
    if (
        payload.get("schema") != SEAT_SCHEMA
        or payload.get("status") != "issued_passed"
        or payload.get("specialist_id") != "alakazam"
        or payload.get("research_derivative") != "final_format_alakazam_h10_i"
        or int(payload.get("iteration", -1)) != iteration
        or payload.get("stage") != stage
        or first <= 0
        or first != second
        or int(payload.get("total_games", -1)) != first + second
        or payload.get("exact_even_split") is not True
        or payload.get("second_focus_1_to_7_used") is not False
        or payload.get("package_preference") != "first_if_allowed"
        or not str(payload.get("deterministic_assignment_manifest_sha256") or "").startswith("sha256:")
    ):
        raise RuntimeError(f"invalid exact-seat receipt: {path}")
    return _sha256(path)


def _validate_seat_receipts(run_dir: Path, terminal_iteration: int) -> dict[str, Any]:
    receipt_root = run_dir / "seat_split_receipts"
    rl: list[dict[str, Any]] = []
    rehearsal: list[dict[str, Any]] = []
    for iteration in range(terminal_iteration + 1):
        index_path = receipt_root / f"iter_{iteration:05d}.index.json"
        index = _read_json(index_path)
        if (
            index.get("schema") != "poke_bot.alakazam_refresh_seat_split_index/v1"
            or int(index.get("iteration", -1)) != iteration
            or index.get("policy") != "exact_50_50_first_second_training"
            or index.get("second_seat_priority") is not False
            or index.get("passed") is not True
        ):
            raise RuntimeError(f"invalid exact-seat index: {index_path}")
        stage_digests = {
            stage: _validate_stage(
                receipt_root / f"iter_{iteration:05d}.{stage}.json",
                iteration=iteration,
                stage=stage,
            )
            for stage in ("assigned", "actual", "consumed")
        }
        rl.append(
            {
                "iteration": iteration,
                "index": str(index_path),
                "index_sha256": _sha256(index_path),
                "stage_sha256": stage_digests,
            }
        )
    for before_iteration in range(5, terminal_iteration + 1, 5):
        index_path = receipt_root / (
            f"rehearsal_before_iter_{before_iteration:05d}.index.json"
        )
        index = _read_json(index_path)
        if (
            index.get("schema")
            != "poke_bot.alakazam_refresh_rehearsal_seat_split_index/v1"
            or int(index.get("before_iteration", -1)) != before_iteration
            or index.get("policy") != "exact_50_50_first_second_training"
            or index.get("second_seat_priority") is not False
            or index.get("passed") is not True
        ):
            raise RuntimeError(f"invalid rehearsal exact-seat index: {index_path}")
        stage_digests = {
            stage: _validate_stage(
                receipt_root
                / f"rehearsal_before_iter_{before_iteration:05d}.{stage}.json",
                iteration=before_iteration,
                stage=stage,
            )
            for stage in ("assigned", "actual", "consumed")
        }
        rehearsal.append(
            {
                "before_iteration": before_iteration,
                "index": str(index_path),
                "index_sha256": _sha256(index_path),
                "stage_sha256": stage_digests,
            }
        )
    return {"rl": rl, "rehearsal": rehearsal}


def _validate_static(args: argparse.Namespace) -> dict[str, str]:
    required = {
        "handler_state": args.handler_state,
        "original_checkpoint": args.original_checkpoint,
        "accepted_core": args.accepted_core,
        "accepted_core_ready": args.accepted_core_ready,
        "protocol": args.protocol,
        "state": args.specialist_state,
        "next_unit": args.next_unit,
    }
    missing = [name for name, path in required.items() if not path.expanduser().is_file()]
    # The handler state is intentionally absent before training. Its configured
    # parent directory still has to exist for the exact terminal command.
    if "handler_state" in missing and args.check:
        missing.remove("handler_state")
        args.handler_state.expanduser().parent.mkdir(parents=True, exist_ok=True)
    if missing:
        raise RuntimeError("completion path is missing: " + ",".join(missing))
    if _sha256(args.original_checkpoint.expanduser()) != ORIGINAL_ALAKAZAM:
        raise RuntimeError("immutable historical Alakazam checkpoint changed")
    if _sha256(args.accepted_core.expanduser()) != ACCEPTED_CORE:
        raise RuntimeError("accepted core checkpoint changed")
    return {
        name: _sha256(path.expanduser())
        for name, path in required.items()
        if path.expanduser().is_file()
    }


def complete(args: argparse.Namespace) -> dict[str, Any]:
    static = _validate_static(args)
    handler = _read_json(args.handler_state.expanduser())
    if (
        handler.get("schema") != HANDLER_SCHEMA
        or handler.get("phase") != "submissions_queued"
        or handler.get("submission_mode") != "queue_and_continue"
        or len(handler.get("queued_submissions") or []) != 1
    ):
        raise RuntimeError("Alakazam refresh handler is not terminal and queued")
    gate = dict(handler.get("gate") or {})
    frozen = dict(handler.get("frozen_model") or {})
    terminal_iteration = int(gate.get("iteration", -1))
    checkpoint_digest = str(frozen.get("checkpoint_digest") or "")
    checkpoint_path = Path(str(frozen.get("model_path") or "")).expanduser().resolve()
    frozen_manifest = checkpoint_path.parent / "manifest.json"
    if (
        terminal_iteration < 5
        or gate.get("checkpoint_digest") != checkpoint_digest
        or not checkpoint_path.is_file()
        or _sha256(checkpoint_path) != checkpoint_digest
        or checkpoint_digest == ORIGINAL_ALAKAZAM
        or not frozen_manifest.is_file()
        or _read_json(frozen_manifest).get("checkpoint_digest") != checkpoint_digest
    ):
        raise RuntimeError("Alakazam refresh frozen identity is invalid")
    seats = _validate_seat_receipts(args.run_dir.expanduser(), terminal_iteration)

    registry_path = args.refresh_registry.expanduser().resolve()
    existing = _read_json(registry_path) if registry_path.is_file() else {}
    if existing and existing.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("post-fleet refresh registry schema changed")
    entries = [dict(row) for row in (existing.get("refreshes") or [])]
    row = {
        "specialist_id": "alakazam",
        "refresh_model_version": "final-format-alakazam-r79-h10-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_checksum": checkpoint_digest,
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": _sha256(frozen_manifest),
        "original_checkpoint_checksum": ORIGINAL_ALAKAZAM,
        "gate_id": str(gate.get("gate_id") or ""),
        "gate_iteration": terminal_iteration,
        "package_preference": "first_if_allowed",
        "training_seat_split": {"first": 0.5, "second": 0.5},
        "historical_specialist_row_replaced": False,
    }
    matching = [entry for entry in entries if entry.get("specialist_id") == "alakazam"]
    if matching and matching != [row]:
        raise RuntimeError("a different Alakazam refresh is already registered")
    if not matching:
        entries.append(row)
    registry = {
        "schema": REGISTRY_SCHEMA,
        "historical_frozen_registry_modified": False,
        "ordered_refresh_ids": ["alakazam", "marnie-s-grimmsnarl-ex"],
        "refreshes": entries,
    }
    registry_body = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_registry = registry_path.with_name(
        f".{registry_path.name}.partial.{os.getpid()}"
    )
    with temporary_registry.open("x", encoding="utf-8") as stream:
        stream.write(registry_body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_registry, registry_path)
    registry_digest = _sha256(registry_path)

    registration = {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered_separate_refresh_derivative",
        "specialist_id": "alakazam",
        "refresh_model_version": row["refresh_model_version"],
        "checkpoint_checksum": checkpoint_digest,
        "registry": str(registry_path),
        "registry_sha256": registry_digest,
        "historical_frozen_registry_modified": False,
        "original_checkpoint_checksum": ORIGINAL_ALAKAZAM,
    }
    _atomic_json(args.registration_receipt.expanduser(), registration)
    registration_digest = _sha256(args.registration_receipt.expanduser())
    gate_digest = str(gate.get("commit_file_sha256") or "")
    if not gate_digest.startswith("sha256:"):
        raise RuntimeError("terminal gate receipt digest is absent")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "passed_frozen_registered",
        "specialist_id": "alakazam",
        "refresh_model_version": row["refresh_model_version"],
        "refresh_checkpoint_checksum": checkpoint_digest,
        "original_checkpoint_checksum": ORIGINAL_ALAKAZAM,
        "current_gate_pass": True,
        "frozen": True,
        "registered": True,
        "gate_receipt_sha256": gate_digest,
        "freeze_receipt_sha256": _sha256(frozen_manifest),
        "registration_receipt_sha256": registration_digest,
        "resolved_core": {
            "status": "checksum_accepted",
            "checkpoint_checksum": ACCEPTED_CORE,
            "ready_receipt_sha256": static["accepted_core_ready"],
        },
        "training_contract": {
            "canonical_source": "config/rl_protocol.yaml#/specialist_training",
            "sha256": static["protocol"],
        },
        "training_seat_receipts": seats,
        "package_preference": "first_if_allowed",
        "second_focused_arm_used": False,
        "historical_alakazam_rewritten": False,
    }
    _atomic_json(args.completion_receipt.expanduser(), completion)
    completion_digest = _sha256(args.completion_receipt.expanduser())

    state_path = args.specialist_state.expanduser().resolve()
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    phase = dict(state.get("post_fleet_refresh") or {})
    expected_completed = list(phase.get("completed_refresh_specialist_ids") or [])
    if expected_completed not in ([], ["alakazam"]):
        raise RuntimeError("post-fleet refresh sequence is not at Alakazam")
    receipt_row = {
        "specialist_id": "alakazam",
        "receipt": str(args.completion_receipt.expanduser().resolve()),
        "receipt_sha256": completion_digest,
    }
    if expected_completed == []:
        phase["completed_refresh_specialist_ids"] = ["alakazam"]
        phase["completed_refresh_receipts"] = [receipt_row]
        phase["pending_specialist_ids"] = ["marnie-s-grimmsnarl-ex"]
        phase["active_refresh_specialist_id"] = "marnie-s-grimmsnarl-ex"
        phase["next_refresh_specialist_id"] = "marnie-s-grimmsnarl-ex"
        phase["status"] = "marnie_refresh_handoff_pending"
        versions = dict(phase.get("refresh_model_versions") or {})
        versions["alakazam"] = row["refresh_model_version"]
        phase["refresh_model_versions"] = versions
        state["post_fleet_refresh"] = phase
        _atomic_yaml(state_path, state)
    elif list(phase.get("completed_refresh_receipts") or []) != [receipt_row]:
        raise RuntimeError("Alakazam completion receipt projection changed")

    started = subprocess.run(
        ["systemctl", "--user", "start", args.next_service],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if started.returncode != 0:
        raise RuntimeError(
            f"could not start Marnie refresh handoff: {started.stdout.strip()}"
        )
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--handler-state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--accepted-core", type=Path, required=True)
    parser.add_argument("--accepted-core-ready", type=Path, required=True)
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
        print(
            "FINAL_FORMAT_ALAKAZAM_COMPLETION_OK "
            + json.dumps(static, sort_keys=True),
            flush=True,
        )
        return 0
    print(json.dumps(complete(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
