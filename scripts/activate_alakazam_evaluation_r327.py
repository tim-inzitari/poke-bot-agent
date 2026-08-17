#!/usr/bin/env python3
"""Activate derivative revision-25 evaluation at a committed iteration boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from poke_bot.alakazam_rule_derivative_evaluation_r327 import (
    CONTRACT_SHA256,
    FORMAL_GAMES_PER_OPPONENT,
    FORMAL_GAMES_PER_SEAT,
    GOAL_SHA256,
    MIGRATION_SCHEMA,
    sha256_file,
    validate_revision25_activation_receipt,
    validate_revision25_formal_gate_contract,
)


def run(argv: list[str]) -> str:
    result = subprocess.run(argv, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def service_property(unit: str, name: str) -> str:
    return run(["systemctl", "--user", "show", unit, f"-p{name}", "--value"])


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def build_activation_receipt(
    *,
    base_contract: Path,
    formal_contract: Path,
    boundary_commit: Path,
    run_name: str,
    elmo_endpoint: str,
) -> dict[str, object]:
    base = _load_json(base_contract)
    formal = _load_json(formal_contract)
    validate_revision25_formal_gate_contract(base, formal)
    commit = _load_json(boundary_commit)
    first_iteration = int(commit.get("next_iteration", -1))
    boundary_iteration = first_iteration - 1
    if boundary_iteration < 0:
        raise ValueError("boundary commit has no completed iteration")
    receipt: dict[str, object] = {
        "schema": MIGRATION_SCHEMA,
        "status": "authorized_at_clean_boundary",
        "owner_goal_revision": 25,
        "root_owner_revision": 327,
        "run_name": run_name,
        "first_formal_holdout_iteration": first_iteration,
        "goal_sha256": GOAL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "base_gate_contract_sha256": sha256_file(base_contract),
        "formal_gate_contract_sha256": sha256_file(formal_contract),
        "boundary_commit_iteration": boundary_iteration,
        "boundary_commit_sha256": sha256_file(boundary_commit),
        "formal_games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
        "candidate_first_games_per_opponent": FORMAL_GAMES_PER_SEAT,
        "candidate_second_games_per_opponent": FORMAL_GAMES_PER_SEAT,
        "separate_research_control_games": 0,
        "training_collection_mix_changed": False,
        "elmo_remote_endpoint_preserved": elmo_endpoint,
    }
    return validate_revision25_activation_receipt(
        receipt,
        base_contract_path=base_contract,
        formal_contract_path=formal_contract,
        boundary_commit_path=boundary_commit,
    )


def write_json_create_only(path: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--expected-main-pid", type=int, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-gate-contract", type=Path, required=True)
    parser.add_argument("--formal-gate-contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--elmo-endpoint", default="elmo:8765")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()

    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError("activation receipt is create-only")
    while not args.commit.is_file():
        if int(service_property(args.training_unit, "MainPID") or "0") != int(
            args.expected_main_pid
        ):
            raise RuntimeError("training PID changed before boundary commit")
        time.sleep(args.poll_seconds)

    pre_pid = int(service_property(args.training_unit, "MainPID") or "0")
    if pre_pid != int(args.expected_main_pid):
        raise RuntimeError("training PID changed at boundary")
    if service_property(args.training_unit, "ActiveState") != "active":
        raise RuntimeError("training service is not active at boundary")
    unit_text = run(["systemctl", "--user", "cat", args.training_unit])
    required_fragments = (
        f"--remote-worker-endpoints {args.elmo_endpoint}",
        f"--formal-holdout-contract {args.formal_gate_contract}",
        f"--r327-evaluation-boundary-receipt {args.receipt}",
        "--heldout-games 900",
        "--disable-research-control-measurement",
        "--boundary-design-migration-reason owner_r327_future_holdout_50_no_research",
    )
    missing = [fragment for fragment in required_fragments if fragment not in unit_text]
    if missing:
        raise RuntimeError(f"next-start unit lacks revision-25 fields: {missing!r}")

    receipt = build_activation_receipt(
        base_contract=args.base_gate_contract,
        formal_contract=args.formal_gate_contract,
        boundary_commit=args.commit,
        run_name=args.run_name,
        elmo_endpoint=args.elmo_endpoint,
    )
    write_json_create_only(args.receipt, receipt)
    run(["systemctl", "--user", "restart", args.training_unit])
    deadline = time.time() + 120.0
    post_pid = 0
    while time.time() < deadline:
        post_pid = int(service_property(args.training_unit, "MainPID") or "0")
        if (
            post_pid > 0
            and post_pid != pre_pid
            and service_property(args.training_unit, "ActiveState") == "active"
        ):
            break
        time.sleep(1.0)
    if post_pid < 1 or post_pid == pre_pid:
        raise RuntimeError("boundary restart did not produce a new active learner PID")
    time.sleep(5.0)
    if (
        service_property(args.training_unit, "ActiveState") != "active"
        or int(service_property(args.training_unit, "MainPID") or "0") != post_pid
    ):
        raise RuntimeError("restarted learner did not remain stable")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
