#!/usr/bin/env python3
"""Authorize the own-model population phase after all required refreshes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from poke_bot.baselines_runtime import baseline_content_digest
from poke_bot.pure_rl.model_registry import sha256
from scripts.materialize_frozen_specialist_gate import materialize_from_contract
from scripts.run_specialist_cycle_handoff import (
    _active_specialist,
    _path,
    _read,
    _required_specialist_ids,
    _source,
    _validated_post_fleet_refresh_progress,
)


SCHEMA = "poke_bot.specialist_cycle_handoff_contract/v1"
RECEIPT_SCHEMA = "poke_bot.population_round_robin_ready/v1"


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _population_roster(
    *,
    state_path: Path,
    frozen_registry_path: Path,
    runtime_registry_path: Path,
    baseline_root: Path,
) -> list[dict[str, Any]]:
    required_ids = _required_specialist_ids(state_path)
    frozen = _read(frozen_registry_path)
    runtime = _read(runtime_registry_path)
    frozen_rows = [
        dict(row) for row in (frozen.get("specialists") or [])
    ]
    by_id = {
        str(row.get("specialist_id") or ""): row for row in frozen_rows
    }
    runtime_rows = dict(runtime.get("specialists") or {})
    if (
        frozen.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or set(by_id) != required_ids
        or len(frozen_rows) != 22
        or set(runtime_rows) != required_ids
    ):
        raise RuntimeError("complete 22-specialist population identity failed")
    roster: list[dict[str, Any]] = []
    for specialist_id in sorted(required_ids):
        frozen_row = by_id[specialist_id]
        runtime_row = dict(runtime_rows[specialist_id])
        package = (
            baseline_root
            / str(frozen_row["baseline_group"])
            / str(frozen_row["baseline_dir"])
        )
        checkpoint = package / "model.pt"
        expert = Path(str(runtime_row.get("expert_manifest") or "")).resolve()
        tree = Path(str(runtime_row.get("matchup_runtime_tree") or "")).resolve()
        if (
            frozen_row.get("frozen") is not True
            or frozen_row.get("public_mix_eligible") is not True
            or runtime_row.get("status") != "ready"
            or not checkpoint.is_file()
            or sha256(checkpoint) != frozen_row.get("checkpoint_digest")
            or baseline_content_digest(package)
            != frozen_row.get("content_digest")
            or not expert.is_file()
            or sha256(expert).removeprefix("sha256:")
            != str(runtime_row.get("expert_manifest_sha256") or "")
            or not tree.is_file()
            or sha256(tree).removeprefix("sha256:")
            != str(runtime_row.get("matchup_runtime_tree_sha256") or "")
        ):
            raise RuntimeError(
                f"population member identity failed: {specialist_id}"
            )
        roster.append(
            {
                "specialist_id": specialist_id,
                "opponent_id": frozen_row["opponent_id"],
                "baseline_group": frozen_row["baseline_group"],
                "baseline_dir": frozen_row["baseline_dir"],
                "baseline_package": str(package),
                "checkpoint": str(checkpoint),
                "checkpoint_digest": frozen_row["checkpoint_digest"],
                "content_digest": frozen_row["content_digest"],
                "expert_manifest": str(expert),
                "expert_manifest_digest": sha256(expert),
                "matchup_runtime_tree": str(tree),
                "matchup_runtime_tree_digest": sha256(tree),
                "trainable_in_population": True,
                "external_agent": False,
            }
        )
    return roster


def prepare(contract_path: Path, *, launch: bool = True) -> dict[str, Any]:
    contract = _read(contract_path.expanduser().resolve())
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("specialist cycle contract schema changed")
    runtime = dict(contract["runtime"])
    lock = _path(runtime, "lock").with_name("population-round-robin-v1.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        state_path = _path(dict(contract["selection"]), "state")
        (
            required_refresh_order,
            completed_refresh_ids,
        ) = _validated_post_fleet_refresh_progress(
            state_path=state_path,
            cycle_contract=contract,
        )
        if completed_refresh_ids != required_refresh_order:
            raise RuntimeError(
                "post-fleet specialist refresh phase is incomplete"
            )
        active_id = _active_specialist(runtime)
        source_contract, source_evidence = _source(
            contract=contract,
            active_id=active_id,
        )
        gate = dict(contract["gate_materialization"])
        state_root = _path(runtime, "state_root")
        final_contract = {
            "source_specialist": source_contract,
            "next_specialist": {
                "gate_contract": gate["base_gate_contract"],
                "frozen_specialist_registry": gate[
                    "base_frozen_specialist_registry"
                ],
            },
            "gate_materialization": {
                **gate,
                "archetype_label": active_id.replace("-", " ").title(),
                "receipt": str(
                    state_root
                    / f"{active_id}-final-population-materialization-v1.json"
                ),
            },
        }
        materialized = materialize_from_contract(
            final_contract,
            source_evidence,
        )
        roster = _population_roster(
            state_path=state_path,
            frozen_registry_path=_path(runtime, "frozen_specialist_registry"),
            runtime_registry_path=_path(runtime, "runtime_registry"),
            baseline_root=Path(str(gate["baseline_root"])).resolve(),
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "ready",
            "member_count": len(roster),
            "members": roster,
            "post_fleet_refresh_order": required_refresh_order,
            "post_fleet_refresh_specialist_ids": completed_refresh_ids,
            "final_specialist_materialization": materialized,
            "training_opponent_scope": "own_models_only",
            "external_agents_training_eligible": False,
            "official_agents_role": "research_only",
            "premium_agents_role": "research_only",
            "rl_epochs_per_cycle": 5,
            "expert_rehearsal_epochs_per_cycle": 5,
            "population_training_service": runtime[
                "population_training_service"
            ],
        }
        receipt["identity_sha256"] = "sha256:" + sha256(
            Path(str(materialized["frozen_specialist_registry"]))
        ).removeprefix("sha256:")
        output = state_root / "population-round-robin-ready-v1.json"
        if output.is_file():
            existing = _read(output)
            if existing != receipt:
                raise RuntimeError("population readiness identity changed")
        else:
            _atomic(output, receipt)
        if launch:
            service = str(runtime["population_training_service"])
            completed = subprocess.run(
                ["/usr/bin/systemctl", "--user", "start", service],
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"population training service failed to start: {service}"
                )
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = prepare(args.contract, launch=not args.check)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
