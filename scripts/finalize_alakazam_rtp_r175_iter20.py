#!/usr/bin/env python3
"""Freeze and register the exact terminal Alakazam RTP r175 iter20 learner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "poke_bot.alakazam_rtp_r175_terminal_completion/v1"
REGISTRATION_SCHEMA = "poke_bot.alakazam_rtp_r175_terminal_registration/v1"
REGISTRY_SCHEMA = "poke_bot.alakazam_rtp_r175_terminal_registry/v1"
BOUNDARY_SCHEMA = "poke_bot.committed_iteration_pause/v1"
COLLECTION_SCHEMA = "poke_bot.completed_collection/v1"
ITERATION = 20
NEXT_ITERATION = 21
FAMILY = "final-format-alakazam-rtp-r175-iter20-v1"
H10_MARNIE_ID = "specialist-marnie-final-format-h10-f20efb20f5c3"
H10_MARNIE_ARCHETYPE = "marnie-s-grimmsnarl-ex"
H10_MARNIE_CONTENT_DIGEST = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
H10_MARNIE_CHECKPOINT_DIGEST = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
H10_MARNIE_FLOOR = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
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


def _canonical_digest(value: dict[str, Any]) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _service_inactive(unit: str) -> bool:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "MainPID",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    return bool(
        result.returncode == 0
        and values.get("ActiveState") in {"inactive", "failed"}
        and int(values.get("MainPID") or 0) == 0
    )


def _completion_disposition(measured: bool) -> dict[str, Any]:
    return {
        "status": (
            "passed_frozen_registered"
            if measured
            else "ceiling_accepted_frozen_registered"
        ),
        "completion_authority": (
            "measured_gate_pass" if measured else "explicit_owner_ceiling_acceptance"
        ),
        "current_gate_pass": measured,
        "measured_gate_pass": measured,
        "failed_gate_results_preserved": not measured,
    }


def _validate_collection_plan_and_h10_rows(
    collection: dict[str, Any], stats: dict[str, Any]
) -> dict[str, Any]:
    plan_path = Path(
        str(stats.get("strong_public_practice_plan") or "")
    ).expanduser().resolve()
    if not plan_path.is_file():
        raise RuntimeError("iter20 strong-public plan is missing")
    plan = _read(plan_path)
    minimums = dict(plan.get("minimum_games_by_opponent") or {})
    per_opponent = dict(plan.get("per_opponent") or {})
    h10 = dict(per_opponent.get(H10_MARNIE_ID) or {})
    h10_games = int(h10.get("games", -1))
    h10_seat0 = int(h10.get("seat0", -1))
    h10_seat1 = int(h10.get("seat1", -1))
    if (
        plan.get("schema") != "poke_bot.strong_public_practice_plan/v1"
        or int(plan.get("iteration", -1)) != ITERATION
        or int(plan.get("games", -1)) != 4586
        or int(minimums.get(H10_MARNIE_ID, -1)) != H10_MARNIE_FLOOR
        or h10.get("archetype_id") != H10_MARNIE_ARCHETYPE
        or int(h10.get("minimum_games", -1)) != H10_MARNIE_FLOOR
        or h10_games < H10_MARNIE_FLOOR
        or h10_seat0 + h10_seat1 != h10_games
        or abs(h10_seat0 - h10_seat1) > 1
    ):
        raise RuntimeError("iter20 H10 Marnie S++ plan floor is invalid")

    record_receipt = dict(stats.get("strong_public_practice_record_receipt") or {})
    if (
        record_receipt.get("passed") is not True
        or int(record_receipt.get("expected_results", -1)) != 4586
        or int(record_receipt.get("successful_results", -1)) != 4586
        or int(record_receipt.get("canonical_records_written", -1)) != 4586
    ):
        raise RuntimeError("iter20 strong-public record receipt is invalid")

    shard = dict(collection.get("shard") or {})
    shard_path = Path(str(shard.get("path") or "")).expanduser().resolve()
    expected_shard_digest = str(shard.get("sha256") or "")
    if not shard_path.is_file() or shard_path.stat().st_size != int(
        shard.get("size", -1)
    ):
        raise RuntimeError("iter20 collection shard identity is invalid")
    digest = hashlib.sha256()
    h10_rows = 0
    with shard_path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if H10_MARNIE_ID.encode("utf-8") not in raw:
                continue
            row = json.loads(raw)
            provenance = dict(row.get("target_provenance") or {})
            matchup = dict(provenance.get("matchup_runtime_audit") or {})
            if (
                row.get("source") != "pure_rl"
                or provenance.get("collect") != "strong_public_practice"
                or provenance.get("opponent_training_group")
                != "strong_public_practice"
                or provenance.get("opponent_id") != H10_MARNIE_ID
                or provenance.get("opponent_archetype_id")
                != H10_MARNIE_ARCHETYPE
                or provenance.get("opponent_content_digest")
                != H10_MARNIE_CONTENT_DIGEST
                or int(provenance.get("strong_public_practice_floor_games", -1))
                != H10_MARNIE_FLOOR
                or provenance.get("target_source") != "recursive_turn_planner"
                or provenance.get("trusted") is not True
                or matchup.get("runtime_enabled") is not True
            ):
                raise RuntimeError("iter20 H10 Marnie retained provenance is invalid")
            h10_rows += 1
    actual_shard_digest = "sha256:" + digest.hexdigest()
    if actual_shard_digest != expected_shard_digest or h10_rows != h10_games:
        raise RuntimeError("iter20 shard checksum/H10 Marnie count is invalid")
    return {
        "plan_path": plan_path,
        "plan_sha256": sha256(plan_path),
        "shard_path": shard_path,
        "shard_sha256": actual_shard_digest,
        "h10_marnie_games": h10_games,
    }


def _validate_r192_contract(path: Path) -> str:
    path = path.expanduser().resolve()
    contract = _read(path)
    opponent = dict(contract.get("opponent") or {})
    collection = dict(contract.get("collection_contract") or {})
    transport = dict(contract.get("transport") or {})
    if (
        contract.get("schema")
        != "poke_bot.alakazam_marnie_splusplus_opponent_r192/v1"
        or int(contract.get("owner_decision_revision", -1)) != 192
        or not str(contract.get("status") or "").startswith("activated_")
        or opponent.get("opponent_id") != H10_MARNIE_ID
        or opponent.get("archetype_id") != H10_MARNIE_ARCHETYPE
        or opponent.get("checkpoint_sha256") != H10_MARNIE_CHECKPOINT_DIGEST
        or opponent.get("content_digest") != H10_MARNIE_CONTENT_DIGEST
        or opponent.get("tier") != "S++"
        or float(opponent.get("weight", -1.0)) != 4.0
        or int(opponent.get("floor_games_per_set", -1)) != H10_MARNIE_FLOOR
        or int(collection.get("games_per_iteration", -1)) != 8196
        or int(collection.get("self_play_mirrors", -1)) != 1024
        or int(collection.get("public_mix_games", -1)) != 7172
        or int(collection.get("strong_public_practice_games", -1)) != 4586
        or transport.get("activation_training_group")
        != "strong_public_practice"
        or transport.get("dispatch_mode") != "singleton_remote_play"
        or transport.get("r182_default_deny_unchanged") is not True
    ):
        raise RuntimeError("canonical r192 H10 Marnie contract is invalid")
    return sha256(path)


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run_dir.expanduser().resolve()
    boundary_path = args.boundary.expanduser().resolve()
    commit_path = run / "commits" / "iter_00020.json"
    next_commit = run / "commits" / "iter_00021.json"
    loop_path = run / "loop_state.json"
    checkpoint = run / "checkpoints" / "iter_00020.pt"
    collection_path = run / "collection_receipts" / "iter_00020.json"
    eval_path = run / "eval" / "iter_00020.json"
    required = (boundary_path, commit_path, loop_path, checkpoint, collection_path, eval_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("terminal artifact missing: " + ",".join(missing))
    if next_commit.exists():
        raise RuntimeError("iteration 21 was committed; terminal boundary violated")
    if not _service_inactive(args.unit):
        raise RuntimeError("trainer is not inactive at the terminal boundary")
    r192_sha256 = _validate_r192_contract(args.r192)

    boundary = _read(boundary_path)
    commit = _read(commit_path)
    loop = _read(loop_path)
    collection = _read(collection_path)
    evaluation = _read(eval_path)
    learner = dict(commit.get("learner") or {})
    digest = str(learner.get("digest") or "")
    boundary_ok = str(boundary.get("status") or "").startswith("paused")
    if (
        boundary.get("schema") != BOUNDARY_SCHEMA
        or not boundary_ok
        or int(boundary.get("completed_iteration", -1)) != ITERATION
        or int(boundary.get("next_iteration", -1)) != NEXT_ITERATION
        or boundary.get("recovery_required") is not False
        or boundary.get("uncommitted_next_iteration_started") is not False
        or int(commit.get("last_completed_iteration", -1)) != ITERATION
        or int(commit.get("next_iteration", -1)) != NEXT_ITERATION
        or loop != commit
        or Path(str(learner.get("path") or "")).resolve() != checkpoint
        or sha256(checkpoint) != digest
        or boundary.get("checkpoint_digest") != digest
        or boundary.get("commit_digest") != sha256(commit_path)
    ):
        raise RuntimeError("terminal boundary/learner identity is invalid")

    stats = dict(collection.get("stats") or {})
    seats = dict(stats.get("training_seat_split") or {})
    retained = dict(seats.get("retained_source_games") or {})
    matchup = dict(stats.get("matchup_runtime") or {})
    remote = dict(stats.get("remote_execution_proof") or {})
    if (
        collection.get("schema") != COLLECTION_SCHEMA
        or int(collection.get("iteration", -1)) != ITERATION
        or int(stats.get("retained_source_games", -1)) != 8196
        or int(stats.get("retained_self_play_source_games", -1)) != 1024
        or int(stats.get("retained_public_mix_source_games", -1)) != 7172
        or float(stats.get("usable_game_fraction", -1.0)) != 1.0
        or seats.get("passed") is not True
        or retained.get("exact_50_50") is not True
        or int(retained.get("seat0", -1)) != 4098
        or int(retained.get("seat1", -1)) != 4098
        or matchup.get("all_runtime_enabled") is not True
        or matchup.get("contract_clean") is not True
        or remote.get("strict") is not True
        or remote.get("missing_endpoints") != []
        or int(remote.get("local_fallback_count", -1)) != 0
    ):
        raise RuntimeError("iter20 exact collection contract is invalid")
    h10_proof = _validate_collection_plan_and_h10_rows(collection, stats)

    candidate = dict(evaluation.get("candidate") or {})
    gate = dict(evaluation.get("active_gate_result") or {})
    audit = dict(gate.get("audit") or {})
    if (
        int(evaluation.get("iteration", -1)) != ITERATION
        or candidate.get("digest") != digest
        or Path(str(candidate.get("path") or "")).resolve() != checkpoint
        or int(gate.get("iteration", -1)) != ITERATION
        or gate.get("checkpoint_digest") != digest
        or audit.get("passed") is not True
        or audit.get("checkpoint_digest") != digest
    ):
        raise RuntimeError("iter20 measured-gate evidence is invalid")
    disposition = _completion_disposition(gate.get("passed") is True)
    return {
        "run": run,
        "boundary_path": boundary_path,
        "commit_path": commit_path,
        "checkpoint": checkpoint,
        "collection_path": collection_path,
        "eval_path": eval_path,
        "digest": digest,
        "gate": gate,
        "disposition": disposition,
        "r192_sha256": r192_sha256,
        **h10_proof,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    from poke_bot.pure_rl.model_registry import freeze_model, verify_frozen_model

    proof = _validate(args)
    sources = {
        "boundary_receipt": str(proof["boundary_path"]),
        "boundary_receipt_sha256": sha256(proof["boundary_path"]),
        "commit": str(proof["commit_path"]),
        "commit_sha256": sha256(proof["commit_path"]),
        "collection_receipt": str(proof["collection_path"]),
        "collection_receipt_sha256": sha256(proof["collection_path"]),
        "evaluation": str(proof["eval_path"]),
        "evaluation_sha256": sha256(proof["eval_path"]),
        "collection_plan": str(proof["plan_path"]),
        "collection_plan_sha256": proof["plan_sha256"],
        "collection_shard": str(proof["shard_path"]),
        "collection_shard_sha256": proof["shard_sha256"],
        "h10_marnie_retained_games": proof["h10_marnie_games"],
        "goal_sha256": sha256(args.goal.expanduser().resolve()),
        "r192_sha256": proof["r192_sha256"],
        "r193_sha256": sha256(args.r193.expanduser().resolve()),
    }
    frozen = freeze_model(
        registry_root=args.registry_root.expanduser().resolve(),
        family=FAMILY,
        display_name="Final-format Alakazam RTP r175 exact iteration 20",
        checkpoint=proof["checkpoint"],
        expected_digest=proof["digest"],
        provenance={
            "specialist_id": "alakazam",
            "run_name": "final_format_alakazam_rtp_r175_i_v6_8k",
            "completed_iteration": ITERATION,
            "next_iteration_collected": False,
            **sources,
        },
        evidence={"active_gate_result": proof["gate"]},
    )
    family_dir = args.registry_root.expanduser().resolve() / FAMILY
    verify_frozen_model(family_dir)
    row = {
        "specialist_id": "alakazam",
        "model_version": FAMILY,
        "checkpoint": str(family_dir / "model.pt"),
        "checkpoint_digest": proof["digest"],
        "manifest": str(family_dir / "manifest.json"),
        "manifest_sha256": sha256(family_dir / "manifest.json"),
        "completed_iteration": ITERATION,
        **proof["disposition"],
        **sources,
    }
    registry = {
        "schema": REGISTRY_SCHEMA,
        "immutable": True,
        "entries": [row],
    }
    _atomic_json(args.terminal_registry.expanduser().resolve(), registry)
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered_exact_terminal_derivative",
        "specialist_id": "alakazam",
        "model_version": FAMILY,
        "checkpoint_digest": proof["digest"],
        "registry": str(args.terminal_registry.expanduser().resolve()),
        "registry_sha256": sha256(args.terminal_registry.expanduser().resolve()),
        "manifest_sha256": row["manifest_sha256"],
        **proof["disposition"],
    }
    _atomic_json(args.registration.expanduser().resolve(), registration)
    completion = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specialist_id": "alakazam",
        "model_version": FAMILY,
        "completed_iteration": ITERATION,
        "next_iteration_collected": False,
        "frozen": True,
        "registered": True,
        "checkpoint_digest": proof["digest"],
        "freeze_manifest_sha256": row["manifest_sha256"],
        "registration_receipt_sha256": sha256(args.registration.expanduser().resolve()),
        "terminal_registry_sha256": sha256(args.terminal_registry.expanduser().resolve()),
        "gate_evidence_sha256": sources["evaluation_sha256"],
        "source_identity_digest": _canonical_digest(sources),
        **proof["disposition"],
        **sources,
    }
    _atomic_json(args.completion.expanduser().resolve(), completion)
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--terminal-registry", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--r192", type=Path, required=True)
    parser.add_argument("--r193", type=Path, required=True)
    args = parser.parse_args()
    result = _validate(args) if args.check else finalize(args)
    print(json.dumps(result if not args.check else {"status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
