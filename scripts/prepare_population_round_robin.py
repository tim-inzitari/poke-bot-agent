#!/usr/bin/env python3
"""Authorize the own-model population phase after all required refreshes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any
import yaml

from poke_bot.baselines_runtime import baseline_content_digest
from poke_bot.pure_rl.model_registry import sha256
from scripts.materialize_population_opponents import materialize_refresh_bundle
from scripts.run_specialist_cycle_handoff import (
    _path,
    _read,
    _validated_post_fleet_refresh_progress,
)


SCHEMA = "poke_bot.specialist_cycle_handoff_contract/v1"
RECEIPT_SCHEMA = "poke_bot.population_round_robin_ready/v1"


def _readiness_identity(value: dict[str, Any]) -> str:
    """Bind the complete readiness payload before adding its own digest."""

    if "identity_sha256" in value:
        raise RuntimeError("population readiness identity is already present")
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    baseline_manifest: Path,
    refresh_registry_path: Path,
) -> list[dict[str, Any]]:
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    passed_ids = {
        str(row.get("id") or "")
        for row in (state.get("specialists") or [])
        if isinstance(row, dict)
        and row.get("status") == "passed_frozen"
        and row.get("frozen") is True
    }
    frozen = _read(frozen_registry_path)
    runtime = _read(runtime_registry_path)
    refresh_registry = _read(refresh_registry_path)
    frozen_rows = [
        dict(row) for row in (frozen.get("specialists") or [])
    ]
    by_id = {
        str(row.get("specialist_id") or ""): row for row in frozen_rows
    }
    runtime_rows = dict(runtime.get("specialists") or {})
    refresh_rows = {
        str(row.get("specialist_id") or ""): dict(row)
        for row in (refresh_registry.get("refreshes") or [])
    }
    historical_ids = set(by_id)
    refresh_ids = set(refresh_rows)
    population_ids = historical_ids | refresh_ids
    if (
        frozen.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or historical_ids != passed_ids
        or len(frozen_rows) != 14
        or refresh_registry.get("schema")
        != "poke_bot.post_fleet_refresh_registry/v1"
        or list(refresh_registry.get("ordered_refresh_ids") or [])
        != ["alakazam", "marnie-s-grimmsnarl-ex", "crustle"]
        or set(refresh_rows)
        != {"alakazam", "marnie-s-grimmsnarl-ex", "crustle"}
        or population_ids != passed_ids | {"crustle"}
        or len(population_ids) != 15
        or "crustle" in historical_ids
    ):
        raise RuntimeError("complete 15-specialist population identity failed")
    roster: list[dict[str, Any]] = []
    for specialist_id in sorted(population_ids):
        frozen_row = by_id.get(specialist_id)
        historical = None
        if frozen_row is not None:
            package = (
                baseline_root
                / str(frozen_row["baseline_group"])
                / str(frozen_row["baseline_dir"])
            )
            checkpoint = package / "model.pt"
            historical = {
                "opponent_id": frozen_row["opponent_id"],
                "baseline_group": frozen_row["baseline_group"],
                "baseline_dir": frozen_row["baseline_dir"],
                "baseline_package": str(package),
                "checkpoint": str(checkpoint),
                "checkpoint_digest": frozen_row["checkpoint_digest"],
                "content_digest": frozen_row["content_digest"],
            }
        refresh = refresh_rows.get(specialist_id)
        runtime_row = dict(runtime_rows.get(specialist_id) or {})
        if refresh is not None:
            current = materialize_refresh_bundle(
                specialist_id=specialist_id,
                checkpoint_digest=str(refresh["checkpoint_checksum"]),
                bundle=Path(str(refresh["submission_bundle"])),
                bundle_digest=str(refresh["submission_bundle_sha256"]),
                baseline_root=baseline_root,
                baseline_manifest=baseline_manifest,
            )
            expert = Path(str(refresh.get("expert_manifest") or "")).resolve()
            tree = Path(str(refresh.get("matchup_runtime_tree") or "")).resolve()
            expert_digest = str(refresh.get("expert_manifest_sha256") or "")
            tree_digest = str(refresh.get("matchup_runtime_tree_sha256") or "")
        else:
            if historical is None:
                raise RuntimeError(
                    f"population member has neither refresh nor history: {specialist_id}"
                )
            current = historical
            expert = Path(str(runtime_row.get("expert_manifest") or "")).resolve()
            tree = Path(str(runtime_row.get("matchup_runtime_tree") or "")).resolve()
            expert_digest = "sha256:" + str(
                runtime_row.get("expert_manifest_sha256") or ""
            )
            tree_digest = "sha256:" + str(
                runtime_row.get("matchup_runtime_tree_sha256") or ""
            )
        historical_is_valid = historical is None or (
            frozen_row is not None
            and frozen_row.get("frozen") is True
            and frozen_row.get("public_mix_eligible") is True
            and checkpoint.is_file()
            and sha256(checkpoint) == frozen_row.get("checkpoint_digest")
            and baseline_content_digest(package) == frozen_row.get("content_digest")
        )
        if (
            not historical_is_valid
            or not expert.is_file()
            or sha256(expert) != expert_digest
            or not tree.is_file()
            or sha256(tree) != tree_digest
        ):
            raise RuntimeError(
                f"population member identity failed: {specialist_id}"
            )
        roster.append(
            {
                "specialist_id": specialist_id,
                **{
                    key: current[key]
                    for key in (
                        "opponent_id",
                        "baseline_group",
                        "baseline_dir",
                        "baseline_package",
                        "checkpoint",
                        "checkpoint_digest",
                        "content_digest",
                    )
                },
                "current_role": (
                    "current_post_fleet_refresh"
                    if refresh is not None
                    else "immutable_baseline_history"
                ),
                # Crustle has no historical own-model specialist.  Its public
                # agent remains research/holdout-only and cannot enter the
                # trainable population through selected history.
                "selected_history": [] if historical is None else [historical],
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
        gate = dict(contract["gate_materialization"])
        state_root = _path(runtime, "state_root")
        capacity_path = state_root / "post_refresh_sequence_complete_for_capacity_v2.json"
        capacity = _read(capacity_path)
        if (
            capacity.get("schema")
            != "poke_bot.post_refresh_sequence_complete_for_capacity/v2"
            or capacity.get("status")
            != "post_refresh_sequence_complete_for_capacity_v2"
            or capacity.get("ordered_refresh_ids") != required_refresh_order
        ):
            raise RuntimeError("post-refresh capacity release receipt is invalid")
        roster = _population_roster(
            state_path=state_path,
            frozen_registry_path=_path(runtime, "frozen_specialist_registry"),
            runtime_registry_path=_path(runtime, "runtime_registry"),
            baseline_root=Path(str(gate["baseline_root"])).resolve(),
            baseline_manifest=Path(str(gate["baseline_manifest"])).resolve(),
            refresh_registry_path=state_root / "post-fleet-refresh-registry-v1.json",
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "ready",
            "member_count": len(roster),
            "members": roster,
            "post_fleet_refresh_order": required_refresh_order,
            "post_fleet_refresh_specialist_ids": completed_refresh_ids,
            "post_refresh_capacity_receipt": str(capacity_path),
            "post_refresh_capacity_receipt_sha256": sha256(capacity_path),
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
        receipt["identity_sha256"] = _readiness_identity(receipt)
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
