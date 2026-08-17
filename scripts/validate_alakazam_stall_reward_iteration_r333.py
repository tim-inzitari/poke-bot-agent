#!/usr/bin/env python3
"""Validate the first r333 no-progress-shaped collection boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_FIELDS = (
    "stall_games",
    "stall_rows_by_seat",
    "stagnant_turn_distribution",
    "training_return_override_counts",
    "ordinary_completed_result_counts",
    "disabled_path_parity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    body += b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)


def validate(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    activation_path: Path,
    collection_path: Path,
    expected_iteration: int = 6,
) -> dict[str, Any]:
    contract_sha256 = sha256_file(contract_path)
    if contract_sha256 != expected_contract_sha256:
        raise RuntimeError("goal contract identity drifted")
    contract = load_json(contract_path)
    gate = dict(contract.get("revision_31_stall_reward_and_submission_mirror_gate") or {})
    training_contract = dict(gate.get("training_only_stall_reward") or {})
    if (
        contract.get("goal_revision") != 31
        or contract.get("root_handoff_revision") != 333
        or tuple(training_contract.get("required_receipt_fields") or ())
        != (
            "activation_boundary_iteration",
            "source_module_sha256",
            "maximum_complete_turns_without_win_progress",
            *EXPECTED_FIELDS,
        )
    ):
        raise RuntimeError("revision-31 receipt contract is stale or malformed")

    activation = load_json(activation_path)
    if (
        activation.get("schema")
        != "poke_bot.alakazam_stall_reward_boundary_activation/v1"
        or activation.get("status")
        != "passed_clean_iteration_5_boundary_before_iteration_6"
        or activation.get("goal_contract_sha256") != contract_sha256
        or activation.get("activation_boundary_iteration") != 5
        or activation.get("first_eligible_collection_iteration") != 6
        or activation.get("maximum_complete_turns_without_win_progress") != 64
    ):
        raise RuntimeError("stall-reward activation receipt is not authoritative")
    source_module_sha256 = str(activation.get("source_module_sha256") or "")
    deployment_root = Path(str(activation.get("deployment_root") or ""))
    source_module = deployment_root / "poke_bot/pure_rl/no_progress.py"
    if not source_module_sha256.startswith("sha256:") or (
        sha256_file(source_module) != source_module_sha256
    ):
        raise RuntimeError("activated no-progress source identity drifted")
    boundary_commit = Path(str(activation.get("boundary_commit_path") or ""))
    if sha256_file(boundary_commit) != activation.get("boundary_commit_sha256"):
        raise RuntimeError("iteration-5 boundary commit identity drifted")

    if expected_iteration < int(activation["first_eligible_collection_iteration"]):
        raise RuntimeError("collection predates the active stall-reward boundary")

    collection = load_json(collection_path)
    stats = dict(collection.get("stats") or {})
    stall = dict(stats.get("no_progress_stall_receipt") or {})
    if (
        collection.get("schema") != "poke_bot.completed_collection/v1"
        or collection.get("iteration") != expected_iteration
        or int(collection.get("requested_games", -1)) != 8196
        or int(collection.get("source_games", -1)) != 8196
        or any(field not in stall for field in EXPECTED_FIELDS)
        or stall.get("maximum_complete_turns_without_win_progress") != 64
        or stall.get("disabled_path_parity") is not False
    ):
        raise RuntimeError(
            f"iteration-{expected_iteration} collection lacks the active stall contract"
        )

    checkpoint_path = Path(str(collection.get("checkpoint") or ""))
    checkpoint_digest = str(collection.get("checkpoint_digest") or "")
    if (
        not checkpoint_digest.startswith("sha256:")
        or sha256_file(checkpoint_path) != checkpoint_digest
        or (
            expected_iteration == activation["first_eligible_collection_iteration"]
            and checkpoint_digest != activation.get("learner_checkpoint_sha256")
        )
    ):
        raise RuntimeError("collection checkpoint identity drifted")

    stall_games = int(stall.get("stall_games", -1))
    stall_rows_by_seat = dict(stall.get("stall_rows_by_seat") or {})
    stagnant_turns = dict(stall.get("stagnant_turn_distribution") or {})
    override_counts = dict(stall.get("training_return_override_counts") or {})
    ordinary_results = dict(stall.get("ordinary_completed_result_counts") or {})
    if stall_games < 0 or set(stall_rows_by_seat) - {"0", "1"}:
        raise RuntimeError("invalid stall game or seat counters")
    if any(int(value) < 0 for value in stall_rows_by_seat.values()):
        raise RuntimeError("negative stall seat counter")
    if sum(int(value) for value in stagnant_turns.values()) != stall_games:
        raise RuntimeError("stagnant-turn distribution disagrees with stall games")
    if any(int(turn) < 64 for turn in stagnant_turns):
        raise RuntimeError("stall row terminated below the 64-turn boundary")
    overridden_rows = sum(int(value) for value in stall_rows_by_seat.values())
    if override_counts != {"-1.0": overridden_rows}:
        raise RuntimeError("training-return override counters disagree by seat")
    if sum(int(value) for value in ordinary_results.values()) + stall_games != 8196:
        raise RuntimeError("ordinary and stalled source-game counts do not partition collection")

    shard = dict(collection.get("shard") or {})
    shard_path = Path(str(shard.get("path") or ""))
    shard_stat = shard_path.stat()
    if (
        int(shard.get("size", -1)) != shard_stat.st_size
        or int(shard.get("mtime_ns", -1)) != shard_stat.st_mtime_ns
        or sha256_file(shard_path) != shard.get("sha256")
    ):
        raise RuntimeError(f"sealed iteration-{expected_iteration} shard identity drifted")

    return {
        "schema": "poke_bot.alakazam_stall_reward_iteration_validation/v1",
        "status": f"passed_iteration_{expected_iteration}_stall_reward_collection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_goal_revision": 31,
        "root_owner_revision": 333,
        "goal_contract_sha256": contract_sha256,
        "activation_receipt_path": str(activation_path.resolve()),
        "activation_receipt_sha256": sha256_file(activation_path),
        "collection_receipt_path": str(collection_path.resolve()),
        "collection_receipt_sha256": sha256_file(collection_path),
        "collection_shard_path": str(shard_path.resolve()),
        "collection_shard_sha256": str(shard["sha256"]),
        "activation_boundary_iteration": 5,
        "collection_iteration": expected_iteration,
        "collection_checkpoint_path": str(checkpoint_path.resolve()),
        "collection_checkpoint_sha256": checkpoint_digest,
        "source_module_sha256": source_module_sha256,
        "maximum_complete_turns_without_win_progress": 64,
        **{field: stall[field] for field in EXPECTED_FIELDS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--collection-receipt", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        contract_path=args.contract,
        expected_contract_sha256=args.contract_sha256,
        activation_path=args.activation_receipt,
        collection_path=args.collection_receipt,
        expected_iteration=args.expected_iteration,
    )
    write_json_create_only(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
