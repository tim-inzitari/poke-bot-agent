"""Research-only Slowking archetype lineage and capability contracts.

Every observed Slowking acting seat is learning-eligible. Exact deck lists are
retained for conditioning, capability masking, drift analysis, and evaluation;
they are never an eligibility gate. This module creates metadata only and has
no training, checkpoint, selector, serving, or submission authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .strategic_schedule import EXPANDED_HEAD_IDS

SCHEMA = "poke_bot.slowking_archetype_learning/v1"
AGGREGATE_SCHEMA = "poke_bot.slowking_multi_day_replay_distillation/v1"
ARCHETYPE_ID = 86
ARCHETYPE_NAME = "Slowking"
SPECIALIZED_HEADS = ("setup_board_outcome", "combo_state")
TEACHER_MODULE = "poke_bot.slowking_reverse_engineered_policy"
TEACHER_ROLE = "offline_sparse_feature_confidence_mask_and_regression_baseline"
TEACHER_AUDIT_SCHEMA = "poke_bot.slowking_reverse_engineered_policy_audit/v1"


class SlowkingArchetypeLearningError(ValueError):
    """The research-only archetype contract or its evidence is inconsistent."""


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def validate_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(payload))
    if data.get("schema") != SCHEMA:
        raise SlowkingArchetypeLearningError("wrong learning-contract schema")
    if data.get("status") != "research_only_no_training_or_runtime_authority":
        raise SlowkingArchetypeLearningError("contract must remain research-only")
    archetype = data.get("archetype") or {}
    if archetype.get("id") != ARCHETYPE_ID or archetype.get("name") != ARCHETYPE_NAME:
        raise SlowkingArchetypeLearningError("wrong Slowking archetype identity")
    if archetype.get("exact_list_required_for_learning") is not False:
        raise SlowkingArchetypeLearningError("exact-list learning gate is forbidden")
    sampling = data.get("sampling") or {}
    if sampling.get("exact_list_filter") is not False:
        raise SlowkingArchetypeLearningError("exact-list sampling filter is forbidden")
    if sampling.get("win_only_filter") is not False:
        raise SlowkingArchetypeLearningError("win-only filtering is forbidden")
    if sampling.get("preserve_losses_and_draws") is not True:
        raise SlowkingArchetypeLearningError("losses and draws must be preserved")
    targets = data.get("learning_targets") or {}
    if tuple(targets.get("expanded_strategic_heads") or ()) != EXPANDED_HEAD_IDS:
        raise SlowkingArchetypeLearningError("expanded strategic-head inventory changed")
    if tuple(targets.get("slowking_option_conditioned_heads") or ()) != SPECIALIZED_HEADS:
        raise SlowkingArchetypeLearningError("Slowking specialized-head inventory changed")
    if targets.get("missing_target_behavior") != "mask_not_zero":
        raise SlowkingArchetypeLearningError("missing targets must be masked")
    if targets.get("unavailable_card_action_behavior") != "legal_action_mask":
        raise SlowkingArchetypeLearningError("unavailable card actions must be legally masked")
    authority = data.get("authority") or {}
    forbidden = (
        "may_start_training",
        "may_register_checkpoint",
        "may_change_selector",
        "may_serve",
        "may_submit",
    )
    if any(authority.get(name) is not False for name in forbidden):
        raise SlowkingArchetypeLearningError("contract grants forbidden authority")
    dates: list[str] = []
    split = data.get("split") or {}
    for key in ("train_dates", "validation_dates", "test_dates"):
        values = split.get(key)
        if not isinstance(values, list):
            raise SlowkingArchetypeLearningError(f"missing split dates: {key}")
        dates.extend(str(value) for value in values)
    if len(dates) != len(set(dates)):
        raise SlowkingArchetypeLearningError("calendar-day splits overlap")
    if split.get("frame_level_random_split_allowed") is not False:
        raise SlowkingArchetypeLearningError("frame-level random split is forbidden")
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        raise SlowkingArchetypeLearningError("capability contract is missing")
    infrastructure = data.get("infrastructure") or {}
    if infrastructure.get("reverse_engineered_teacher") != TEACHER_MODULE:
        raise SlowkingArchetypeLearningError("wrong reverse-engineered teacher module")
    if infrastructure.get("reverse_engineered_teacher_role") != TEACHER_ROLE:
        raise SlowkingArchetypeLearningError("wrong reverse-engineered teacher role")
    if infrastructure.get("reverse_engineered_teacher_serving_logit_authority") is not False:
        raise SlowkingArchetypeLearningError("reverse-engineered teacher has serving authority")
    return data


def load_contract(path: Path) -> dict[str, Any]:
    return validate_contract(json.loads(path.read_text()))


def load_aggregate(contract: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    source = contract.get("source") or {}
    path = root / str(source.get("aggregate_receipt") or "")
    if not path.is_file():
        raise SlowkingArchetypeLearningError("aggregate replay receipt is missing")
    if _sha256(path) != source.get("aggregate_receipt_sha256"):
        raise SlowkingArchetypeLearningError("aggregate replay receipt checksum mismatch")
    aggregate = json.loads(path.read_text())
    validate_aggregate(contract, aggregate)
    return aggregate


def load_reverse_teacher_audit(
    contract: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    """Load the checksum-pinned sparse teacher feature index."""
    infrastructure = contract.get("infrastructure") or {}
    path = root / str(infrastructure.get("reverse_engineered_teacher_audit") or "")
    if not path.is_file():
        raise SlowkingArchetypeLearningError("reverse-engineered teacher audit is missing")
    if _sha256(path) != infrastructure.get("reverse_engineered_teacher_audit_sha256"):
        raise SlowkingArchetypeLearningError("reverse-engineered teacher audit checksum mismatch")
    audit = json.loads(path.read_text())
    if audit.get("schema") != TEACHER_AUDIT_SCHEMA:
        raise SlowkingArchetypeLearningError("wrong reverse-engineered teacher audit schema")
    if audit.get("status") != "research_only_complete":
        raise SlowkingArchetypeLearningError("reverse-engineered teacher audit is not complete")
    policy = audit.get("policy") or {}
    if policy.get("runtime_authority") != "none" or policy.get("future_or_result_inputs") != []:
        raise SlowkingArchetypeLearningError("reverse-engineered teacher is not causal research-only")
    source = audit.get("source") or {}
    if int(source.get("requested_games", -1)) != int(contract["source"]["games"]):
        raise SlowkingArchetypeLearningError("reverse-engineered teacher game count changed")
    decisions = audit.get("covered_decisions")
    if not isinstance(decisions, list) or len(decisions) != int((audit.get("overall") or {}).get("covered", -1)):
        raise SlowkingArchetypeLearningError("reverse-engineered teacher decisions are incomplete")
    return audit


def validate_aggregate(
    contract: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> None:
    if aggregate.get("schema") != AGGREGATE_SCHEMA:
        raise SlowkingArchetypeLearningError("wrong aggregate receipt schema")
    if aggregate.get("status") != "research_only_no_training_or_runtime_authority":
        raise SlowkingArchetypeLearningError("aggregate receipt is not research-only")
    source = contract.get("source") or {}
    overall = aggregate.get("overall") or {}
    for name in ("games", "wins", "losses", "draws"):
        if int(overall.get(name, -1)) != int(source.get(name, -2)):
            raise SlowkingArchetypeLearningError(f"aggregate {name} changed")
    teams = aggregate.get("by_team") or []
    lineages = aggregate.get("by_deck_lineage") or []
    if len(teams) != int(source.get("teams", -1)):
        raise SlowkingArchetypeLearningError("aggregate team count changed")
    if len(lineages) != int(source.get("exact_lists", -1)):
        raise SlowkingArchetypeLearningError("aggregate exact-list count changed")
    if sum(int(row.get("games", 0)) for row in lineages) != int(overall["games"]):
        raise SlowkingArchetypeLearningError("lineage games do not cover archetype")
    identities = {
        (str(game.get("date")), str(game.get("episode_id")), int(game.get("seat", -1)))
        for game in aggregate.get("games") or []
    }
    if len(identities) != int(overall["games"]):
        raise SlowkingArchetypeLearningError("duplicate or missing acting-seat identity")


def _split_by_date(contract: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    split = contract["split"]
    for name, key in (
        ("train", "train_dates"),
        ("validation", "validation_dates"),
        ("test", "test_dates"),
    ):
        for date in split[key]:
            result[str(date)] = name
    return result


def _capabilities(
    contract: Mapping[str, Any], card_counts: Mapping[str, int]
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, raw in contract["capabilities"].items():
        capability = dict(raw)
        if capability.get("always") is True:
            result[name] = True
            continue
        required = [str(value) for value in capability.get("requires_cards") or ()]
        minimum = {
            str(card): int(count)
            for card, count in dict(capability.get("minimum_card_counts") or {}).items()
        }
        result[name] = all(card_counts.get(card, 0) > 0 for card in required) and all(
            card_counts.get(card, 0) >= count for card, count in minimum.items()
        )
    return result


def build_game_rows(
    contract: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Create archetype-wide learning metadata for all observed acting seats."""

    validate_contract(contract)
    validate_aggregate(contract, aggregate)
    split_by_date = _split_by_date(contract)
    decks: dict[str, dict[str, Any]] = {}
    for lineage in aggregate["by_deck_lineage"]:
        fingerprint = str(lineage["fingerprint"])
        card_counts = {
            str(row["card_name"]): int(row["count"])
            for row in lineage["cards"]
        }
        decks[fingerprint] = {
            "card_counts": card_counts,
            "capabilities": _capabilities(contract, card_counts),
        }
    rows: list[dict[str, Any]] = []
    for game in aggregate["games"]:
        date = str(game["date"])
        split = split_by_date.get(date)
        if split is None:
            raise SlowkingArchetypeLearningError(f"unassigned replay date: {date}")
        fingerprint = str(game["deck_fingerprint"])
        deck = decks.get(fingerprint)
        if deck is None:
            raise SlowkingArchetypeLearningError("game references unknown deck lineage")
        rows.append(
            {
                "date": date,
                "episode_id": str(game["episode_id"]),
                "seat": int(game["seat"]),
                "team_name": str(game["team_name"]),
                "archetype_id": ARCHETYPE_ID,
                "archetype_name": ARCHETYPE_NAME,
                "deck_fingerprint": fingerprint,
                "deck_card_counts": dict(deck["card_counts"]),
                "split": split,
                "result": str(game["result"]),
                "turn_order": str(game["turn_order"]),
                "decision_frames": int(game["decision_frames"]),
                "sample_weight": float(contract["sampling"]["base_game_weight"]),
                "policy_learning_eligible": True,
                "exact_list_learning_gate": False,
                "capabilities": dict(deck["capabilities"]),
                "learning_heads": {
                    "core": list(contract["learning_targets"]["core"]),
                    "expanded_strategic": list(EXPANDED_HEAD_IDS),
                    "slowking_option_conditioned": list(SPECIALIZED_HEADS),
                },
            }
        )
    expected = contract["split"]["expected_games"]
    actual = {
        name: sum(row["split"] == name for row in rows)
        for name in ("train", "validation", "test")
    }
    if actual != {name: int(expected[name]) for name in actual}:
        raise SlowkingArchetypeLearningError(
            f"calendar-day split counts changed: expected={expected} actual={actual}"
        )
    if len(rows) != int(contract["sampling"]["policy_learning_eligible_games"]):
        raise SlowkingArchetypeLearningError("not every Slowking game is learning-eligible")
    return rows


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "policy_learning_eligible": sum(
            row.get("policy_learning_eligible") is True for row in rows
        ),
        "by_split": {
            split: sum(row.get("split") == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "teams": sorted({str(row["team_name"]) for row in rows}),
        "exact_lists": sorted({str(row["deck_fingerprint"]) for row in rows}),
    }


__all__ = [
    "AGGREGATE_SCHEMA",
    "ARCHETYPE_ID",
    "ARCHETYPE_NAME",
    "SCHEMA",
    "SPECIALIZED_HEADS",
    "SlowkingArchetypeLearningError",
    "build_game_rows",
    "load_aggregate",
    "load_contract",
    "load_reverse_teacher_audit",
    "summarize_rows",
    "validate_aggregate",
    "validate_contract",
]
