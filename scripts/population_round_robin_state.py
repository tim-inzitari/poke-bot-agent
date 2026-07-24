#!/usr/bin/env python3
"""Checksum-bound state machine for the 22-member population phase.

This module does not start population training.  It provides the authoritative
rotation and checkpoint-history rules consumed by the population controller:
one active member at a time, exactly five RL iterations, exactly five expert
rehearsal epochs, current versions plus immutable selected history, and no
external training opponents.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


READY_SCHEMA = "poke_bot.population_round_robin_ready/v1"
STATE_SCHEMA = "poke_bot.population_round_robin_state/v1"
BOUNDARY_SCHEMA = "poke_bot.population_member_cycle_boundary/v1"
MEMBER_COUNT = 22
RL_EPOCHS = 5
REHEARSAL_EPOCHS = 5


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_readiness(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    members = [
        dict(row)
        for row in (receipt.get("members") or [])
        if isinstance(row, dict)
    ]
    ids = [str(row.get("specialist_id") or "") for row in members]
    if (
        receipt.get("schema") != READY_SCHEMA
        or receipt.get("status") != "ready"
        or int(receipt.get("member_count") or 0) != MEMBER_COUNT
        or len(members) != MEMBER_COUNT
        or len(set(ids)) != MEMBER_COUNT
        or "" in ids
        or receipt.get("training_opponent_scope") != "own_models_only"
        or receipt.get("external_agents_training_eligible") is not False
        or int(receipt.get("rl_epochs_per_cycle") or 0) != RL_EPOCHS
        or int(receipt.get("expert_rehearsal_epochs_per_cycle") or 0)
        != REHEARSAL_EPOCHS
    ):
        raise RuntimeError("population readiness contract is incomplete")
    for row in members:
        if (
            row.get("external_agent") is not False
            or row.get("trainable_in_population") is not True
            or not str(row.get("checkpoint_digest") or "").startswith(
                "sha256:"
            )
            or not str(row.get("content_digest") or "").startswith("sha256:")
            or not str(row.get("checkpoint") or "")
            or not str(row.get("expert_manifest") or "")
            or not str(row.get("opponent_id") or "")
            or not str(row.get("baseline_group") or "")
            or not str(row.get("baseline_dir") or "")
            or not str(row.get("baseline_package") or "")
        ):
            raise RuntimeError(
                "population member lacks checksum-bound own-model identity"
            )
    return members


def initialize_state(receipt: dict[str, Any]) -> dict[str, Any]:
    members = validate_readiness(receipt)
    rows: list[dict[str, Any]] = []
    for row in members:
        baseline = {
            "role": "immutable_baseline_history",
            "checkpoint": str(row["checkpoint"]),
            "checkpoint_digest": str(row["checkpoint_digest"]),
            "content_digest": str(row["content_digest"]),
            "opponent_id": str(row["opponent_id"]),
            "baseline_group": str(row["baseline_group"]),
            "baseline_dir": str(row["baseline_dir"]),
            "baseline_package": str(row["baseline_package"]),
            "population_cycle": -1,
        }
        rows.append(
            {
                "specialist_id": str(row["specialist_id"]),
                "expert_manifest": str(row["expert_manifest"]),
                "expert_manifest_digest": str(
                    row["expert_manifest_digest"]
                ),
                "current": copy.deepcopy(baseline),
                "selected_history": [copy.deepcopy(baseline)],
                "rl_epochs_completed": 0,
                "rehearsal_epochs_completed": 0,
                "cycles_completed": 0,
            }
        )
    state = {
        "schema": STATE_SCHEMA,
        "status": "ready",
        "member_count": MEMBER_COUNT,
        "population_cycle": 0,
        "active_member_index": 0,
        "active_specialist_id": rows[0]["specialist_id"],
        "members": rows,
        "external_agents_training_eligible": False,
        "official_agents_role": "research_only",
        "premium_agents_role": "research_only",
        "rl_epochs_per_member_cycle": RL_EPOCHS,
        "expert_rehearsal_epochs_per_member_cycle": REHEARSAL_EPOCHS,
        "readiness_identity": _canonical_digest(receipt),
    }
    state["identity"] = _canonical_digest(
        {key: value for key, value in state.items() if key != "identity"}
    )
    return state


def eligible_own_opponents(
    state: dict[str, Any],
    *,
    active_specialist_id: str,
) -> list[dict[str, Any]]:
    """Return current and selected historical own models, without duplicates."""

    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("external_agents_training_eligible") is not False
        or int(state.get("member_count") or 0) != MEMBER_COUNT
    ):
        raise RuntimeError("population state contract changed")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for member in state["members"]:
        specialist_id = str(member["specialist_id"])
        candidates = [dict(member["current"])] + [
            dict(row) for row in member["selected_history"]
        ]
        for candidate in candidates:
            key = (
                specialist_id,
                str(candidate["checkpoint_digest"]),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    **candidate,
                    "specialist_id": specialist_id,
                    "is_active_member": specialist_id
                    == active_specialist_id,
                    "external_agent": False,
                }
            )
    if {row["specialist_id"] for row in output} != {
        str(row["specialist_id"]) for row in state["members"]
    }:
        raise RuntimeError("population opponent field lost a member")
    return output


def record_completed_member_cycle(
    state: dict[str, Any],
    boundary: dict[str, Any],
    materialized_opponent: dict[str, Any],
) -> dict[str, Any]:
    """Advance one member only after its exact 5-RL/5-rehearsal boundary."""

    result = copy.deepcopy(state)
    index = int(result.get("active_member_index") or 0)
    members = result.get("members") or []
    if (
        result.get("schema") != STATE_SCHEMA
        or len(members) != MEMBER_COUNT
        or not 0 <= index < MEMBER_COUNT
    ):
        raise RuntimeError("population state is invalid")
    member = members[index]
    specialist_id = str(member["specialist_id"])
    parent = dict(boundary.get("parent") or {})
    rehearsed = dict(boundary.get("rehearsed") or {})
    if (
        boundary.get("schema") != BOUNDARY_SCHEMA
        or str(boundary.get("specialist_id") or "") != specialist_id
        or int(boundary.get("rl_iterations_completed") or 0) != RL_EPOCHS
        or int(boundary.get("expert_rehearsal_epochs_completed") or 0)
        != REHEARSAL_EPOCHS
        or boundary.get("external_agents_training_eligible") is not False
        or str(parent.get("digest") or "")
        != str(member["current"]["checkpoint_digest"])
        or not str(rehearsed.get("digest") or "").startswith("sha256:")
        or not str(rehearsed.get("path") or "")
        or str(materialized_opponent.get("checkpoint_digest") or "")
        != str(rehearsed.get("digest") or "")
        or not str(materialized_opponent.get("content_digest") or "").startswith(
            "sha256:"
        )
        or not str(materialized_opponent.get("opponent_id") or "")
        or not str(materialized_opponent.get("baseline_package") or "")
    ):
        raise RuntimeError("population member boundary is not exact")
    member["current"] = {
        "role": "current_population_version",
        "checkpoint": str(rehearsed["path"]),
        "checkpoint_digest": str(rehearsed["digest"]),
        "content_digest": str(materialized_opponent["content_digest"]),
        "opponent_id": str(materialized_opponent["opponent_id"]),
        "baseline_group": str(
            materialized_opponent.get("baseline_group") or "population"
        ),
        "baseline_dir": str(materialized_opponent["baseline_dir"]),
        "baseline_package": str(materialized_opponent["baseline_package"]),
        "population_cycle": int(result["population_cycle"]),
    }
    member["rl_epochs_completed"] += RL_EPOCHS
    member["rehearsal_epochs_completed"] += REHEARSAL_EPOCHS
    member["cycles_completed"] += 1
    next_index = index + 1
    if next_index == MEMBER_COUNT:
        next_index = 0
        result["population_cycle"] += 1
    result["active_member_index"] = next_index
    result["active_specialist_id"] = str(
        members[next_index]["specialist_id"]
    )
    result["status"] = "training"
    result["identity"] = _canonical_digest(
        {key: value for key, value in result.items() if key != "identity"}
    )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.readiness.read_text(encoding="utf-8"))
    print(json.dumps(initialize_state(receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
