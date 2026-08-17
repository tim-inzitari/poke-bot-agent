#!/usr/bin/env python3
"""Run the checksum-bound 15-member own-model population round robin."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from scripts.materialize_frozen_specialist_gate import _sync_one_remote
from scripts.materialize_population_opponents import (
    build_opponent_registry,
    materialize_current_version,
)
from scripts.population_round_robin_state import (
    MEMBER_COUNT,
    REHEARSAL_EPOCHS,
    RL_EPOCHS,
    STATE_SCHEMA,
    initialize_state,
    record_completed_member_cycle,
)


CONTRACT_SCHEMA = "poke_bot.population_round_robin_controller_contract/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _path(contract: dict[str, Any], key: str) -> Path:
    raw = str((contract.get("paths") or {}).get(key) or "")
    if not raw:
        raise RuntimeError(f"population path is missing: {key}")
    return Path(raw).expanduser().resolve()


def validate_contract(contract: dict[str, Any]) -> None:
    schedule = dict(contract.get("schedule") or {})
    policy = dict(contract.get("opponent_policy") or {})
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or int(schedule.get("members") or 0) != MEMBER_COUNT
        or int(schedule.get("rl_iterations_per_member_cycle") or 0)
        != RL_EPOCHS
        or int(
            schedule.get("expert_rehearsal_epochs_per_member_cycle") or 0
        )
        != REHEARSAL_EPOCHS
        or int(schedule.get("games_per_rl_iteration") or 0) != 8192
        or int(schedule.get("expert_rehearsal_every") or 0) != 5
        or policy.get("own_models_only") is not True
        or policy.get("include_current_versions") is not True
        or policy.get("include_selected_historical_versions") is not True
        or policy.get("external_agents_training_eligible") is not False
        or policy.get("official_agents_role") != "research_only"
        or policy.get("premium_agents_role") != "research_only"
    ):
        raise RuntimeError("population controller contract changed")
    for key in (
        "readiness",
        "state",
        "opponent_registry",
        "baseline_root",
        "baseline_manifest",
        "output_root",
        "python",
        "runtime_root",
        "active_gate_contract",
        "research_control_registry",
        "frozen_specialist_registry",
    ):
        _path(contract, key)


def initialize(contract: dict[str, Any]) -> dict[str, Any]:
    validate_contract(contract)
    state_path = _path(contract, "state")
    if state_path.is_file():
        state = _read(state_path)
        if state.get("schema") != STATE_SCHEMA:
            raise RuntimeError("existing population state schema changed")
    else:
        state = initialize_state(_read(_path(contract, "readiness")))
        _atomic(state_path, state)
    build_opponent_registry(
        state=state,
        baseline_root=_path(contract, "baseline_root"),
        output=_path(contract, "opponent_registry"),
    )
    return state


def _member_command(
    contract: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[str], Path]:
    schedule = dict(contract["schedule"])
    training = dict(contract["training"])
    index = int(state["active_member_index"])
    member = dict(state["members"][index])
    specialist_id = str(member["specialist_id"])
    cycle = int(state["population_cycle"])
    run_name = f"population_{specialist_id}_cycle_{cycle:04d}"
    run_dir = _path(contract, "output_root") / run_name
    python = str(_path(contract, "python"))
    runtime_root = _path(contract, "runtime_root")
    command = [
        python,
        "-u",
        str(runtime_root / "scripts" / "launch_pure_rl.py"),
        "--run-name",
        run_name,
        "--mode",
        "specialist",
        "--python",
        python,
        "--preflight-profile",
        "none",
        "--allow-single-gpu",
        "--multi-env-per-worker",
        "4",
        "--stall-minutes",
        "20",
        "--log",
        str(
            Path("/home/pokebot/poke-bot-agent/outputs/logs")
            / f"{run_name}.log"
        ),
        "--",
        "--population-own-models-only",
        "--population-opponent-registry",
        str(_path(contract, "opponent_registry")),
        "--specialist-archetype",
        specialist_id,
        "--initial-learner-checkpoint",
        str(member["current"]["checkpoint"]),
        "--active-gate-contract",
        str(_path(contract, "active_gate_contract")),
        "--research-control-registry",
        str(_path(contract, "research_control_registry")),
        "--frozen-specialist-registry",
        str(_path(contract, "frozen_specialist_registry")),
        "--official-collect-frac",
        "0",
        "--research-control-games-per-iter",
        str(training["research_control_games_per_iteration"]),
        "--iterations",
        str(schedule["rl_iterations_per_member_cycle"]),
        "--games-per-iter",
        str(schedule["games_per_rl_iteration"]),
        "--train-epochs",
        str(schedule["train_epochs_per_rl_iteration"]),
        "--train-games-per-batch",
        str(training["train_games_per_batch"]),
        "--train-max-decisions-per-batch",
        str(training["train_max_decisions_per_batch"]),
        "--measurement-decks",
        specialist_id,
        "--heldout-games",
        str(training["heldout_games"]),
        "--heldout-remotes",
        "--promotion-games",
        "80",
        "--game-timeout-s",
        "600",
        "--min-usable-game-frac",
        "0.98",
        "--remote-worker-endpoints",
        str(training["remote_worker_endpoints"]),
        "--expert-rehearsal-every",
        str(schedule["expert_rehearsal_every"]),
        "--expert-manifest",
        str(member["expert_manifest"]),
        "--expert-rehearsal-epochs",
        str(schedule["expert_rehearsal_epochs_per_member_cycle"]),
        "--expert-rehearsal-batch-size",
        str(training["expert_rehearsal_batch_size"]),
        "--expert-min-decisions",
        str(training["minimum_expert_decisions"]),
        "--artifact-history-iterations",
        str(training["artifact_history_iterations"]),
        "--min-free-disk-gb",
        str(training["minimum_free_disk_gb"]),
        "--continue-after-gate",
        "--minimum-terminal-iteration",
        "-1",
        "--resume",
        "never",
    ]
    return command, run_dir


def run_one(contract: dict[str, Any]) -> dict[str, Any]:
    state = initialize(contract)
    command, run_dir = _member_command(contract, state)
    boundary_path = (
        run_dir / "population" / "cycle_after_iter_00005.json"
    )
    if boundary_path.is_file():
        boundary = _read(boundary_path)
    else:
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"population member trainer failed rc={completed.returncode}"
            )
        boundary = _read(boundary_path)
    index = int(state["active_member_index"])
    member = dict(state["members"][index])
    rehearsed = dict(boundary["rehearsed"])
    materialized = materialize_current_version(
        specialist_id=str(member["specialist_id"]),
        population_cycle=int(state["population_cycle"]),
        checkpoint=Path(str(rehearsed["path"])),
        checkpoint_digest=str(rehearsed["digest"]),
        source_package=Path(str(member["current"]["baseline_package"])),
        baseline_root=_path(contract, "baseline_root"),
        baseline_manifest=_path(contract, "baseline_manifest"),
    )
    package = Path(str(materialized["baseline_package"]))
    fleet = [
        _sync_one_remote(
            host=str(row["host"]),
            remote_root=str(row["baseline_root"]),
            package=package,
            manifest=_path(contract, "baseline_manifest"),
            container=(
                str(row["container"])
                if row.get("container") is not None
                else None
            ),
            group="population",
        )
        for row in (contract.get("fleet_sync") or [])
    ]
    # Rotation becomes visible only after every simulator can load the exact
    # current-version package. If sync fails, the immutable member boundary is
    # reused on retry and no five-iteration training block is repeated.
    state = record_completed_member_cycle(state, boundary, materialized)
    _atomic(_path(contract, "state"), state)
    build_opponent_registry(
        state=state,
        baseline_root=_path(contract, "baseline_root"),
        output=_path(contract, "opponent_registry"),
    )
    return {
        "completed_specialist_id": str(member["specialist_id"]),
        "population_cycle": int(state["population_cycle"]),
        "next_specialist_id": str(state["active_specialist_id"]),
        "checkpoint_digest": str(rehearsed["digest"]),
        "fleet": fleet,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    contract_path = args.contract.expanduser().resolve()
    contract = _read(contract_path)
    lock_path = _path(contract, "state").with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        state = initialize(contract)
        if args.check:
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        while True:
            print(json.dumps(run_one(contract), sort_keys=True), flush=True)
            if args.once:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
