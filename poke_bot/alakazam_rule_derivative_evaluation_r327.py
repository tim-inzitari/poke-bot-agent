"""Revision-25 derivative formal-evaluation migration helpers.

This module changes only future formal evaluation volume and suppresses the
separate research-control measurement wave.  It deliberately leaves the
training collection mix, roster, weights, pass criteria, and model state
untouched.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FORMAL_GAMES_PER_OPPONENT = 50
FORMAL_GAMES_PER_SEAT = 25
MIGRATION_SCHEMA = (
    "poke_bot.alakazam_rule_derivative_future_evaluation_migration_r327/v1"
)
GOAL_SHA256 = (
    "sha256:624d7b0d88b19948eabc504bc1e9b9030a0d161816b11ad7559fe5534712e5b8"
)
CONTRACT_SHA256 = (
    "sha256:e97227a42a0b1675220b5faec4c8fed1776b3eebe9b2f2d5544bd0f73bb6a59d"
)


class Revision25EvaluationError(ValueError):
    """Raised when the formal-evaluation migration is not exact."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Revision25EvaluationError(f"{label} must be an object")
    return copy.deepcopy(dict(value))


def _roster_identity(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    roster = gate.get("roster")
    if not isinstance(roster, list) or not roster:
        raise Revision25EvaluationError("formal gate roster is empty")
    return [copy.deepcopy(dict(row)) for row in roster]


def build_revision25_formal_gate_contract(
    base_contract: Mapping[str, Any],
    *,
    exact_result_pointer: str,
) -> dict[str, Any]:
    """Derive the exact 50-game formal contract from the sealed active gate."""

    derived = _mapping(base_contract, label="base gate contract")
    gate = _mapping(derived.get("next_gate"), label="base next_gate")
    roster = _roster_identity(gate)
    evaluation = _mapping(gate.get("evaluation"), label="base evaluation")
    if not str(exact_result_pointer).startswith("/"):
        raise Revision25EvaluationError("exact result pointer must be absolute")

    prior_id = str(gate.get("id") or "")
    if not prior_id:
        raise Revision25EvaluationError("base gate has no identity")
    next_id = prior_id + "+derivative-r327-g50"
    evaluation.update(
        {
            "games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
            "games_total": len(roster) * FORMAL_GAMES_PER_OPPONENT,
            "minimum_games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
            "seat0_games_per_opponent": FORMAL_GAMES_PER_SEAT,
            "seat1_games_per_opponent": FORMAL_GAMES_PER_SEAT,
        }
    )
    gate["id"] = next_id
    gate["label"] = str(gate.get("label") or prior_id) + " · 50/deck"
    gate["evaluation"] = evaluation
    gate["exact_result_pointer"] = str(exact_result_pointer)
    gate["revision_25_future_evaluation"] = {
        "owner_goal_revision": 25,
        "root_owner_revision": 327,
        "source_gate_id": prior_id,
        "games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
        "candidate_first_games_per_opponent": FORMAL_GAMES_PER_SEAT,
        "candidate_second_games_per_opponent": FORMAL_GAMES_PER_SEAT,
        "separate_research_control_games": 0,
        "training_collection_mix_changed": False,
    }
    derived["active_gate_id"] = next_id
    derived["next_gate"] = gate
    derived["derived_from_gate_id"] = prior_id
    derived["revision_25_future_evaluation"] = copy.deepcopy(
        gate["revision_25_future_evaluation"]
    )
    return validate_revision25_formal_gate_contract(base_contract, derived)


def validate_revision25_formal_gate_contract(
    base_contract: Mapping[str, Any],
    formal_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the derivative changes only count/identity/result-pointer fields."""

    base = _mapping(base_contract, label="base gate contract")
    formal = _mapping(formal_contract, label="formal gate contract")
    base_gate = _mapping(base.get("next_gate"), label="base next_gate")
    formal_gate = _mapping(formal.get("next_gate"), label="formal next_gate")
    if base.get("schema") != "poke_bot.competition_gate_program/v1":
        raise Revision25EvaluationError("base gate schema is unsupported")
    if formal.get("schema") != base.get("schema"):
        raise Revision25EvaluationError("formal gate schema changed")
    if _roster_identity(base_gate) != _roster_identity(formal_gate):
        raise Revision25EvaluationError("formal gate roster or weights changed")
    for field in (
        "pass_criteria",
        "kaggle_rating_simulation",
        "research_measurements",
        "excluded_aliases",
        "threshold_transition",
    ):
        if formal_gate.get(field) != base_gate.get(field):
            raise Revision25EvaluationError(f"formal gate changed {field}")

    evaluation = _mapping(formal_gate.get("evaluation"), label="formal evaluation")
    roster_size = len(_roster_identity(formal_gate))
    if (
        int(evaluation.get("games_per_opponent", -1))
        != FORMAL_GAMES_PER_OPPONENT
        or int(evaluation.get("minimum_games_per_opponent", -1))
        != FORMAL_GAMES_PER_OPPONENT
        or int(evaluation.get("seat0_games_per_opponent", -1))
        != FORMAL_GAMES_PER_SEAT
        or int(evaluation.get("seat1_games_per_opponent", -1))
        != FORMAL_GAMES_PER_SEAT
        or int(evaluation.get("games_total", -1))
        != roster_size * FORMAL_GAMES_PER_OPPONENT
        or evaluation.get("all_matchups_must_complete") is not True
        or evaluation.get("partial_results_gate_eligible") is not False
        or evaluation.get("sequential_early_stop") is not False
    ):
        raise Revision25EvaluationError("formal evaluation is not exact 50/deck 25/25")
    metadata = _mapping(
        formal_gate.get("revision_25_future_evaluation"),
        label="revision-25 metadata",
    )
    if (
        metadata.get("owner_goal_revision") != 25
        or metadata.get("root_owner_revision") != 327
        or metadata.get("source_gate_id") != base_gate.get("id")
        or metadata.get("separate_research_control_games") != 0
        or metadata.get("training_collection_mix_changed") is not False
        or formal.get("active_gate_id") != formal_gate.get("id")
        or formal.get("derived_from_gate_id") != base_gate.get("id")
    ):
        raise Revision25EvaluationError("revision-25 metadata is invalid")

    # Compare after removing the explicitly permitted fields. This avoids a
    # recursive builder call while still making every other byte semantic.
    base_compare = copy.deepcopy(base)
    formal_compare = copy.deepcopy(formal)
    for row in (base_compare, formal_compare):
        row.pop("active_gate_id", None)
        row.pop("derived_from_gate_id", None)
        row.pop("revision_25_future_evaluation", None)
        gate = dict(row.get("next_gate") or {})
        gate.pop("id", None)
        gate.pop("label", None)
        gate.pop("evaluation", None)
        gate.pop("exact_result_pointer", None)
        gate.pop("revision_25_future_evaluation", None)
        row["next_gate"] = gate
    if base_compare != formal_compare:
        raise Revision25EvaluationError("formal gate changed a non-evaluation field")
    return formal


def validate_revision25_activation_receipt(
    receipt: Mapping[str, Any],
    *,
    base_contract_path: Path | str,
    formal_contract_path: Path | str,
    boundary_commit_path: Path | str,
) -> dict[str, Any]:
    """Validate the immutable receipt consumed by the resumed trainer."""

    row = _mapping(receipt, label="revision-25 activation receipt")
    required = {
        "schema",
        "status",
        "owner_goal_revision",
        "root_owner_revision",
        "run_name",
        "first_formal_holdout_iteration",
        "goal_sha256",
        "contract_sha256",
        "base_gate_contract_sha256",
        "formal_gate_contract_sha256",
        "boundary_commit_iteration",
        "boundary_commit_sha256",
        "formal_games_per_opponent",
        "candidate_first_games_per_opponent",
        "candidate_second_games_per_opponent",
        "separate_research_control_games",
        "training_collection_mix_changed",
        "elmo_remote_endpoint_preserved",
    }
    if set(row) != required:
        raise Revision25EvaluationError("activation receipt fields are not exact")
    if (
        row.get("schema") != MIGRATION_SCHEMA
        or row.get("status") != "authorized_at_clean_boundary"
        or row.get("owner_goal_revision") != 25
        or row.get("root_owner_revision") != 327
        or row.get("run_name") != "alakazam_rule_derivative_g5_r12"
        or int(row.get("first_formal_holdout_iteration", -1)) < 1
        or row.get("goal_sha256") != GOAL_SHA256
        or row.get("contract_sha256") != CONTRACT_SHA256
        or row.get("base_gate_contract_sha256") != sha256_file(base_contract_path)
        or row.get("formal_gate_contract_sha256") != sha256_file(formal_contract_path)
        or int(row.get("boundary_commit_iteration", -1)) < 0
        or int(row.get("first_formal_holdout_iteration", -1))
        != int(row.get("boundary_commit_iteration", -2)) + 1
        or row.get("boundary_commit_sha256") != sha256_file(boundary_commit_path)
        or row.get("formal_games_per_opponent") != FORMAL_GAMES_PER_OPPONENT
        or row.get("candidate_first_games_per_opponent") != FORMAL_GAMES_PER_SEAT
        or row.get("candidate_second_games_per_opponent") != FORMAL_GAMES_PER_SEAT
        or row.get("separate_research_control_games") != 0
        or row.get("training_collection_mix_changed") is not False
        or row.get("elmo_remote_endpoint_preserved") != "192.168.1.143:8765"
    ):
        raise Revision25EvaluationError("activation receipt is invalid")
    base = json.loads(Path(base_contract_path).read_text(encoding="utf-8"))
    formal = json.loads(Path(formal_contract_path).read_text(encoding="utf-8"))
    validate_revision25_formal_gate_contract(base, formal)
    return row


__all__ = [
    "CONTRACT_SHA256",
    "FORMAL_GAMES_PER_OPPONENT",
    "FORMAL_GAMES_PER_SEAT",
    "GOAL_SHA256",
    "MIGRATION_SCHEMA",
    "Revision25EvaluationError",
    "build_revision25_formal_gate_contract",
    "sha256_file",
    "validate_revision25_activation_receipt",
    "validate_revision25_formal_gate_contract",
]
