"""Derive the immutable r241 public-baseline roster from preserved evidence.

The r241 source snapshot deliberately does not make a local ``baselines``
checkout authoritative.  This module instead reads the sealed r175/r192
provenance objects, verifies all four immutable source digests, and emits the
small receipt consumed by :mod:`poke_bot.r241_baseline_payload_snapshot`.

It is intentionally data-only: it never imports a baseline, starts a worker,
or touches a selector.  ``write_receipt`` is create-only so an operator may
materialize the already-derived result at a later receipt-backed boundary.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import r241_baseline_payload_snapshot as baseline_payload
from .r241_direct_policy_runtime import (
    R241_H10_CONTENT_SHA256,
    R241_H10_DIR_NAME,
    R241_H10_MODEL_SHA256,
    R241_H10_OPPONENT_ID,
)


SCHEMA = baseline_payload.CANONICAL_BASELINE_ROSTER_SCHEMA
CANDIDATE_ID = baseline_payload.CANDIDATE_ID
REVISION = baseline_payload.REVISION

# These are the immutable source identities used to reconstruct the peak-r195
# public library.  They are deliberately not inferred from a mutable checkout.
R175_RUN_MANIFEST_SHA256 = (
    "sha256:c65c3d191b6b9996d03f7ca4512ef5835baba6189e2114fe46d935cba0f62e40"
)
R192_H10_CONTRACT_SHA256 = (
    "sha256:2c63f2c7a579b002d85cc7e4569c81eff3cf64554775c6a93a979e5fd33f3138"
)
R175_ITER20_PLAN_SHA256 = (
    "sha256:d64f382c38789f4ccd07339f9e38488c616999da2c06d5d19580974710e17f6a"
)
R175_ITER20_RECEIPT_SHA256 = (
    "sha256:859091c7b0bf998d73717d85411f10fc381dc089c1bef9a6521227181c0949e1"
)

DEFAULT_PROVENANCE_SHA256S: dict[str, str] = {
    "r175_run_manifest": R175_RUN_MANIFEST_SHA256,
    "r192_h10_contract": R192_H10_CONTRACT_SHA256,
    "r175_iter20_strong_public_practice_plan": R175_ITER20_PLAN_SHA256,
    "r175_iter20_collection_receipt": R175_ITER20_RECEIPT_SHA256,
}

_EXPECTED_PROVENANCE_KEYS = frozenset(DEFAULT_PROVENANCE_SHA256S)
_EXPECTED_GROUP_GAMES = {
    "self_play": 1024,
    "strong_public_practice": 4586,
    "diverse_public": 2586,
}
_SELF_PLAY_GAMES = 1024
_STRONG_PUBLIC_GAMES = 4586
_DIVERSE_PUBLIC_GAMES = 2586
_PUBLIC_MIX_GAMES = _STRONG_PUBLIC_GAMES + _DIVERSE_PUBLIC_GAMES
_TOTAL_GAMES = _SELF_PLAY_GAMES + _PUBLIC_MIX_GAMES
_RESEARCH_CONTROL_MEASUREMENT_GAMES = 1000
_H10_MINIMUM_GAMES = 1024


class R241CanonicalBaselineRosterError(baseline_payload.R241BaselinePayloadError):
    """The immutable r175/r192 provenance cannot prove the r241 roster."""


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise R241CanonicalBaselineRosterError(f"{label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise R241CanonicalBaselineRosterError(f"{label} must be a list")
    return value


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241CanonicalBaselineRosterError(f"{label} must be an exact integer")
    return value


def _required_bool(value: object, *, label: str, expected: bool) -> None:
    if value is not expected:
        raise R241CanonicalBaselineRosterError(f"{label} must be {expected!r}")


def _required_int(value: object, *, label: str, expected: int) -> None:
    if _exact_int(value, label=label) != expected:
        raise R241CanonicalBaselineRosterError(f"{label} must be exactly {expected}")


def _required_text(value: object, *, label: str, expected: str) -> None:
    if str(value or "") != expected:
        raise R241CanonicalBaselineRosterError(f"{label} drifted")


def _read_json_verified(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, object], str]:
    source = baseline_payload.regular_file(path, label=label)
    expected = baseline_payload.valid_sha256(expected_sha256, label=f"{label} expected SHA")
    actual = baseline_payload.sha256_file(source)
    if actual != expected:
        raise R241CanonicalBaselineRosterError(
            f"{label} checksum drifted: expected={expected} actual={actual}"
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241CanonicalBaselineRosterError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R241CanonicalBaselineRosterError(f"{label} must contain a JSON object")
    return source, value, actual


def _source_identity(path: Path, sha256: str) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": baseline_payload.valid_sha256(sha256, label="provenance source SHA"),
        "size_bytes": int(path.stat().st_size),
    }


def _source_row(raw: object, *, label: str) -> dict[str, str]:
    """Normalize one r175 opponent row into a mounted-baseline placement."""

    row = _mapping(raw, label=label)
    opponent_id = str(row.get("id") or "").strip()
    content = _mapping(row.get("content"), label=f"{label}.content")
    content_path = str(content.get("path") or "").strip()
    content_digest = baseline_payload.valid_sha256(
        content.get("digest"), label=f"{label}.content.digest"
    )
    if not opponent_id or not content_path:
        raise R241CanonicalBaselineRosterError(f"{label} has no package identity")
    path = PurePosixPath(content_path)
    positions = [index for index, part in enumerate(path.parts) if part == "baselines"]
    if len(positions) != 1:
        raise R241CanonicalBaselineRosterError(
            f"{label}.content.path must contain one baselines root: {content_path!r}"
        )
    tail = path.parts[positions[0] + 1 :]
    if len(tail) != 2:
        raise R241CanonicalBaselineRosterError(
            f"{label}.content.path must identify exactly group/dir: {content_path!r}"
        )
    try:
        return baseline_payload.normalized_roster(
            [
                {
                    "id": opponent_id,
                    "group": tail[0],
                    "dir": tail[1],
                    "content_digest": content_digest,
                }
            ]
        )[0]
    except baseline_payload.R241BaselinePayloadError as exc:
        raise R241CanonicalBaselineRosterError(f"{label} has unsafe package placement") from exc


def _source_rows(value: object, *, label: str) -> list[dict[str, str]]:
    rows = [_source_row(item, label=f"{label}[{index}]") for index, item in enumerate(_list(value, label=label))]
    try:
        return baseline_payload.normalized_roster(rows)
    except baseline_payload.R241BaselinePayloadError as exc:
        raise R241CanonicalBaselineRosterError(f"{label} has duplicate package identity") from exc


def _same_rows(left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]]) -> bool:
    return baseline_payload.normalized_roster(left) == baseline_payload.normalized_roster(right)


def _assert_disjoint_lanes(**lanes: Sequence[Mapping[str, object]]) -> None:
    seen: dict[str, str] = {}
    for lane_name, lane_rows in lanes.items():
        for row in baseline_payload.normalized_roster(lane_rows):
            previous = seen.setdefault(row["id"], lane_name)
            if previous != lane_name:
                raise R241CanonicalBaselineRosterError(
                    f"r175 lane rows overlap: {row['id']!r} belongs to {previous} and {lane_name}"
                )


def _contract_reference(value: object, *, label: str) -> dict[str, object]:
    reference = _mapping(value, label=label)
    if str(reference.get("kind") or "") != "file":
        raise R241CanonicalBaselineRosterError(f"{label}.kind must be file")
    path = str(reference.get("path") or "").strip()
    if not path:
        raise R241CanonicalBaselineRosterError(f"{label}.path is missing")
    size_bytes = _exact_int(reference.get("size"), label=f"{label}.size")
    if size_bytes < 0:
        raise R241CanonicalBaselineRosterError(f"{label}.size must be non-negative")
    return {
        "path": path,
        "sha256": baseline_payload.valid_sha256(reference.get("digest"), label=f"{label}.digest"),
        "size_bytes": size_bytes,
    }


def _verify_r192_h10(value: Mapping[str, object]) -> tuple[dict[str, str], str]:
    _required_text(
        value.get("schema"),
        label="r192 H10 contract schema",
        expected="poke_bot.alakazam_marnie_splusplus_opponent_r192/v1",
    )
    _required_int(
        value.get("owner_decision_revision"), label="r192 owner decision revision", expected=192
    )
    opponent = _mapping(value.get("opponent"), label="r192 opponent")
    _required_text(opponent.get("opponent_id"), label="r192 H10 opponent id", expected=R241_H10_OPPONENT_ID)
    _required_text(
        opponent.get("checkpoint_sha256"),
        label="r192 H10 checkpoint",
        expected=R241_H10_MODEL_SHA256,
    )
    _required_text(
        opponent.get("content_digest"),
        label="r192 H10 content digest",
        expected=R241_H10_CONTENT_SHA256,
    )
    _required_text(opponent.get("tier"), label="r192 H10 tier", expected="S++")
    if isinstance(opponent.get("weight"), bool) or float(opponent.get("weight") or 0.0) != 4.0:
        raise R241CanonicalBaselineRosterError("r192 H10 weight must be exactly 4.0")
    _required_int(
        opponent.get("floor_games_per_set"),
        label="r192 H10 floor games",
        expected=_H10_MINIMUM_GAMES,
    )
    _required_bool(
        opponent.get("distinct_additional_specialist_row"),
        label="r192 H10 distinct additional row",
        expected=True,
    )
    _required_bool(
        opponent.get("duplicate_alias_row_allowed"),
        label="r192 H10 duplicate alias row",
        expected=False,
    )
    historical = _mapping(value.get("historical_marnie"), label="r192 historical Marnie")
    historical_id = str(historical.get("opponent_id") or "").strip()
    if not historical_id or historical_id == R241_H10_OPPONENT_ID:
        raise R241CanonicalBaselineRosterError("r192 historical Marnie identity is invalid")
    _required_bool(
        historical.get("must_remain_distinct"),
        label="r192 historical Marnie distinction",
        expected=True,
    )
    _required_bool(
        historical.get("may_be_collapsed_or_substituted"),
        label="r192 historical Marnie substitution",
        expected=False,
    )
    return (
        {
            "id": R241_H10_OPPONENT_ID,
            "group": "specialists",
            "dir": R241_H10_DIR_NAME,
            "content_digest": R241_H10_CONTENT_SHA256,
        },
        historical_id,
    )


def _verify_iter20_plan(
    value: Mapping[str, object], *, strong_ids: Sequence[str]
) -> dict[str, object]:
    _required_text(
        value.get("schema"),
        label="r175 iter20 plan schema",
        expected="poke_bot.strong_public_practice_plan/v1",
    )
    _required_int(value.get("iteration"), label="r175 iter20 plan iteration", expected=20)
    _required_int(value.get("games"), label="r175 iter20 strong games", expected=_STRONG_PUBLIC_GAMES)
    _required_bool(
        value.get("training_eligible"), label="r175 iter20 strong training eligibility", expected=True
    )
    group_games = _mapping(
        value.get("group_games_per_iteration"), label="r175 iter20 group games"
    )
    if dict(group_games) != _EXPECTED_GROUP_GAMES:
        raise R241CanonicalBaselineRosterError("r175 iter20 group game allocation drifted")
    minimums = _mapping(
        value.get("minimum_games_by_opponent"), label="r175 iter20 minimums"
    )
    per_opponent = _mapping(value.get("per_opponent"), label="r175 iter20 per-opponent plan")
    expected_ids = set(str(item) for item in strong_ids)
    if set(minimums) != expected_ids or set(per_opponent) != expected_ids:
        raise R241CanonicalBaselineRosterError("r175 iter20 strong roster membership drifted")
    planned_games = 0
    for opponent_id in sorted(expected_ids):
        minimum = _exact_int(minimums.get(opponent_id), label=f"r175 iter20 minimum {opponent_id}")
        row = _mapping(per_opponent.get(opponent_id), label=f"r175 iter20 row {opponent_id}")
        games = _exact_int(row.get("games"), label=f"r175 iter20 games {opponent_id}")
        row_minimum = _exact_int(
            row.get("minimum_games"), label=f"r175 iter20 row minimum {opponent_id}"
        )
        if minimum < 0 or games < minimum or row_minimum != minimum:
            raise R241CanonicalBaselineRosterError(
                f"r175 iter20 invalid minimum/game count for {opponent_id}"
            )
        planned_games += games
    if planned_games != _STRONG_PUBLIC_GAMES:
        raise R241CanonicalBaselineRosterError("r175 iter20 per-opponent total drifted")
    if (
        _exact_int(minimums.get(R241_H10_OPPONENT_ID), label="r175 iter20 H10 minimum")
        != _H10_MINIMUM_GAMES
        or _exact_int(
            _mapping(
                per_opponent.get(R241_H10_OPPONENT_ID), label="r175 iter20 H10 row"
            ).get("games"),
            label="r175 iter20 H10 games",
        )
        < _H10_MINIMUM_GAMES
    ):
        raise R241CanonicalBaselineRosterError("r175 iter20 H10 floor is not preserved")
    research = _mapping(value.get("research_controls"), label="r175 iter20 research controls")
    _required_int(
        research.get("additive_measurement_games"),
        label="r175 iter20 research measurement games",
        expected=_RESEARCH_CONTROL_MEASUREMENT_GAMES,
    )
    _required_bool(
        research.get("training_eligible"), label="r175 iter20 research training eligibility", expected=False
    )
    _required_bool(
        research.get("replay_eligible"), label="r175 iter20 research replay eligibility", expected=False
    )
    if str(research.get("stage") or "") != "measure:research_controls":
        raise R241CanonicalBaselineRosterError("r175 iter20 research stage drifted")
    return {
        "iteration": 20,
        "strong_public_practice_games": _STRONG_PUBLIC_GAMES,
        "group_games_per_iteration": dict(_EXPECTED_GROUP_GAMES),
        "h10_minimum_games": _H10_MINIMUM_GAMES,
        "h10_planned_games": _exact_int(
            _mapping(per_opponent[R241_H10_OPPONENT_ID], label="r175 iter20 H10 row").get(
                "games"
            ),
            label="r175 iter20 H10 games",
        ),
        "research_controls": {
            "additive_measurement_games": _RESEARCH_CONTROL_MEASUREMENT_GAMES,
            "training_eligible": False,
            "replay_eligible": False,
        },
    }


def _verify_iter20_receipt(value: Mapping[str, object]) -> dict[str, object]:
    _required_text(
        value.get("schema"),
        label="r175 iter20 collection receipt schema",
        expected="poke_bot.completed_collection/v1",
    )
    _required_int(value.get("iteration"), label="r175 iter20 receipt iteration", expected=20)
    _required_int(value.get("requested_games"), label="r175 iter20 requested games", expected=_TOTAL_GAMES)
    stats = _mapping(value.get("stats"), label="r175 iter20 collection stats")
    for field, expected in {
        "n_baseline_jobs": _PUBLIC_MIX_GAMES,
        "n_public_mix_jobs": _PUBLIC_MIX_GAMES,
        "n_self_play_jobs": _SELF_PLAY_GAMES,
        "n_research_control_jobs": 0,
        "self_play": _SELF_PLAY_GAMES,
        "retained_public_mix_source_games": _PUBLIC_MIX_GAMES,
        "retained_self_play_source_games": _SELF_PLAY_GAMES,
        "retained_source_games": _TOTAL_GAMES,
        "ok": _TOTAL_GAMES,
    }.items():
        _required_int(stats.get(field), label=f"r175 iter20 stats.{field}", expected=expected)
    strong_records = _mapping(
        stats.get("strong_public_practice_record_receipt"),
        label="r175 iter20 strong-public record receipt",
    )
    _required_text(
        strong_records.get("schema"),
        label="r175 iter20 strong-public record schema",
        expected="poke_bot.strong_public_practice_record_receipt/v1",
    )
    _required_bool(
        strong_records.get("passed"), label="r175 iter20 strong-public record pass", expected=True
    )
    for field in ("expected_results", "seen_results", "successful_results", "canonical_records_written"):
        _required_int(
            strong_records.get(field),
            label=f"r175 iter20 strong-public record {field}",
            expected=_STRONG_PUBLIC_GAMES,
        )
    _required_int(
        strong_records.get("failed_results"),
        label="r175 iter20 strong-public failed results",
        expected=0,
    )
    strong_plan_reference = str(stats.get("strong_public_practice_plan") or "").strip()
    if not strong_plan_reference.endswith("/collection_plans/iter_00020.json"):
        raise R241CanonicalBaselineRosterError("r175 iter20 receipt does not bind its strong plan")
    seat_split = _mapping(stats.get("training_seat_split"), label="r175 iter20 training seat split")
    _required_text(
        seat_split.get("schema"),
        label="r175 iter20 seat split schema",
        expected="poke_bot.alakazam_refresh_seat_split_summary/v1",
    )
    _required_text(
        seat_split.get("policy"),
        label="r175 iter20 seat split policy",
        expected="exact_50_50_first_second_training",
    )
    _required_bool(seat_split.get("passed"), label="r175 iter20 seat split pass", expected=True)
    for stage_name in ("assigned_source_games", "retained_source_games"):
        stage = _mapping(seat_split.get(stage_name), label=f"r175 iter20 seat {stage_name}")
        _required_bool(
            stage.get("exact_50_50"), label=f"r175 iter20 seat {stage_name} exactness", expected=True
        )
        for field, expected in (("seat0", 4098), ("seat1", 4098), ("total", _TOTAL_GAMES)):
            _required_int(
                stage.get(field),
                label=f"r175 iter20 seat {stage_name}.{field}",
                expected=expected,
            )
    return {
        "iteration": 20,
        "requested_games": _TOTAL_GAMES,
        "self_play_games": _SELF_PLAY_GAMES,
        "public_mix_games": _PUBLIC_MIX_GAMES,
        "research_control_jobs": 0,
        "strong_public_practice_plan_reference": strong_plan_reference,
        "strong_public_practice_records": _STRONG_PUBLIC_GAMES,
        "training_seats": {"first": 4098, "second": 4098},
    }


def _lane(rows: Sequence[Mapping[str, object]], *, source_field: str) -> dict[str, object]:
    members = baseline_payload.normalized_roster(rows)
    return {
        "source_field": source_field,
        "count": len(members),
        "ids": [row["id"] for row in members],
        "members": members,
    }


def _provenance_expectations(
    values: Mapping[str, object] | None,
) -> dict[str, str]:
    source = DEFAULT_PROVENANCE_SHA256S if values is None else values
    if set(source) != _EXPECTED_PROVENANCE_KEYS:
        raise R241CanonicalBaselineRosterError(
            "provenance SHA map must contain exactly the four immutable source names"
        )
    return {
        name: baseline_payload.valid_sha256(value, label=f"expected {name}")
        for name, value in source.items()
    }


def build_receipt(
    *,
    r175_manifest: Path | str,
    r192_h10_contract: Path | str,
    r175_iter20_plan: Path | str,
    r175_iter20_receipt: Path | str,
    owner_contract_sha256: str,
    expected_provenance_sha256s: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one r241 canonical roster receipt without writing it.

    ``expected_provenance_sha256s`` exists for isolated tests only; normal
    callers use the four immutable defaults above.  It is intentionally not
    exposed by the command-line tool.
    """

    expected = _provenance_expectations(expected_provenance_sha256s)
    owner_sha = baseline_payload.valid_sha256(
        owner_contract_sha256, label="r241 canonical roster owner contract"
    )
    r175_path, r175, r175_sha = _read_json_verified(
        r175_manifest,
        label="immutable r175 run manifest",
        expected_sha256=expected["r175_run_manifest"],
    )
    r192_path, r192, r192_sha = _read_json_verified(
        r192_h10_contract,
        label="immutable r192 H10 contract",
        expected_sha256=expected["r192_h10_contract"],
    )
    plan_path, plan, plan_sha = _read_json_verified(
        r175_iter20_plan,
        label="immutable r175 iter20 strong-public plan",
        expected_sha256=expected["r175_iter20_strong_public_practice_plan"],
    )
    collection_path, collection, collection_sha = _read_json_verified(
        r175_iter20_receipt,
        label="immutable r175 iter20 collection receipt",
        expected_sha256=expected["r175_iter20_collection_receipt"],
    )

    design = _mapping(r175.get("design_contract"), label="r175 design contract")
    opponents = _mapping(design.get("opponents"), label="r175 design opponents")
    collect = _source_rows(opponents.get("collect"), label="r175 opponents.collect")
    historical_gate = _source_rows(opponents.get("heldout"), label="r175 opponents.heldout")
    official_target = _source_rows(
        opponents.get("official_target_training"), label="r175 opponents.official_target_training"
    )
    research_controls = _source_rows(
        opponents.get("research_controls"), label="r175 opponents.research_controls"
    )
    if len(collect) != 19 or len(historical_gate) != 17 or len(research_controls) != 4:
        raise R241CanonicalBaselineRosterError(
            "r175 canonical lane cardinality must be collect=19, heldout=17, research=4"
        )
    if not _same_rows(historical_gate, official_target):
        raise R241CanonicalBaselineRosterError(
            "r175 official target training no longer equals its historical active gate"
        )
    h10_row, historical_marnie_id = _verify_r192_h10(r192)
    if historical_marnie_id not in {row["id"] for row in historical_gate}:
        raise R241CanonicalBaselineRosterError(
            "r192 historical Marnie row is absent from the r175 historical active gate"
        )
    if R241_H10_OPPONENT_ID in {
        row["id"] for row in [*collect, *historical_gate, *research_controls]
    }:
        raise R241CanonicalBaselineRosterError(
            "r192 H10 Marnie must be an additional row, not an r175 lane alias"
        )
    _assert_disjoint_lanes(
        diverse_public=collect,
        historical_active_gate=historical_gate,
        r192_h10_additional=[h10_row],
        research_controls=research_controls,
    )
    strong_public = baseline_payload.normalized_roster([*historical_gate, h10_row])
    strong_ids = [row["id"] for row in strong_public]
    plan_summary = _verify_iter20_plan(plan, strong_ids=strong_ids)
    collection_summary = _verify_iter20_receipt(collection)

    gates = _mapping(design.get("gates"), label="r175 design gates")
    collection_design = _mapping(design.get("collection"), label="r175 design collection")
    research_design = _mapping(
        collection_design.get("research_control_phase"), label="r175 research-control design"
    )
    public_contract_sources = {
        "active_gate_contract": _contract_reference(
            r175.get("active_gate_contract"), label="r175 active gate contract"
        ),
        "frozen_specialist_registry": _contract_reference(
            gates.get("frozen_specialist_registry"), label="r175 frozen specialist registry"
        ),
        "research_control_registry": _contract_reference(
            research_design.get("registry"), label="r175 research-control registry"
        ),
    }
    public_contract_sha256s = {
        name: str(reference["sha256"]) for name, reference in public_contract_sources.items()
    }

    roster = baseline_payload.normalized_roster(
        [*collect, *strong_public, *research_controls]
    )
    if len(roster) != 41:
        raise R241CanonicalBaselineRosterError("canonical baseline union must contain exactly 41 rows")
    training_public_union = baseline_payload.normalized_roster([*collect, *strong_public])
    if len(training_public_union) != 37:
        raise R241CanonicalBaselineRosterError("training public union must contain exactly 37 rows")

    minimal_manifest = baseline_payload.minimal_manifest_bytes(roster)
    minimal_manifest_sha256 = baseline_payload.sha256_bytes(minimal_manifest)
    lane_membership = {
        "r175_diverse_public": _lane(
            collect, source_field="design_contract.opponents.collect"
        ),
        "r175_historical_active_gate": _lane(
            historical_gate, source_field="design_contract.opponents.heldout"
        ),
        "r192_h10_additional_strong_specialist": {
            **_lane([h10_row], source_field="r192.opponent"),
            "not_in_historical_active_gate": True,
            "historical_marnie_row_id": historical_marnie_id,
        },
        "r175_iter20_strong_public_practice": _lane(
            strong_public,
            source_field="r175 opponents.heldout plus r192 additional H10 Marnie",
        ),
        "r175_research_controls": {
            **_lane(research_controls, source_field="design_contract.opponents.research_controls"),
            "training_eligible": False,
            "replay_eligible": False,
        },
        "training_public_union": _lane(
            training_public_union,
            source_field="r175 diverse public plus r175/r192 strong public practice",
        ),
        "canonical_baseline_union": _lane(
            roster,
            source_field="r175 diverse public plus r175/r192 strong public plus research controls",
        ),
    }

    return {
        "schema": SCHEMA,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "owner_contract_sha256": owner_sha,
        "provenance": {
            "r175_run_manifest": _source_identity(r175_path, r175_sha),
            "r192_h10_contract": _source_identity(r192_path, r192_sha),
            "r175_iter20_strong_public_practice_plan": _source_identity(plan_path, plan_sha),
            "r175_iter20_collection_receipt": _source_identity(collection_path, collection_sha),
            "r175_public_contract_sources": public_contract_sources,
        },
        # These exact keys are consumed by the existing immutable payload
        # stager and launch validators.
        "public_contract_sha256s": public_contract_sha256s,
        "lane_membership": lane_membership,
        "counts": {
            "r175_diverse_public_rows": len(collect),
            "r175_historical_active_gate_rows": len(historical_gate),
            "r192_h10_additional_rows": 1,
            "r175_iter20_strong_public_practice_rows": len(strong_public),
            "r175_research_control_rows": len(research_controls),
            "training_public_union_rows": len(training_public_union),
            "canonical_baseline_union_rows": len(roster),
            "iter20_self_play_games": _SELF_PLAY_GAMES,
            "iter20_diverse_public_games": _DIVERSE_PUBLIC_GAMES,
            "iter20_strong_public_practice_games": _STRONG_PUBLIC_GAMES,
            "iter20_public_mix_games": _PUBLIC_MIX_GAMES,
            "iter20_total_games": _TOTAL_GAMES,
            "iter20_research_measurement_games": _RESEARCH_CONTROL_MEASUREMENT_GAMES,
            "iter20_research_training_eligible": False,
            "iter20_research_replay_eligible": False,
        },
        "iter20_plan_summary": plan_summary,
        "iter20_collection_summary": collection_summary,
        "baseline_roster": roster,
        "baseline_roster_sha256": baseline_payload.roster_digest(roster),
        "baseline_manifest_sha256": minimal_manifest_sha256,
        "minimal_baseline_manifest": {
            "path": "manifest.json",
            "encoding": "utf-8",
            "canonical_json_utf8": minimal_manifest.decode("utf-8"),
            "size_bytes": len(minimal_manifest),
            "sha256": minimal_manifest_sha256,
            "agent_count": len(roster),
        },
    }


def write_receipt(path: Path | str, receipt: Mapping[str, object]) -> Path:
    """Create one immutable receipt, accepting only byte-identical replays."""

    raw = Path(path).expanduser()
    if not raw.name or raw.name in {".", ".."}:
        raise R241CanonicalBaselineRosterError("canonical roster receipt output has no file name")
    parent = raw.parent
    if parent.is_symlink() or not parent.is_dir():
        raise R241CanonicalBaselineRosterError(
            f"canonical roster receipt parent must be a real directory: {parent}"
        )
    target = parent.resolve() / raw.name
    encoded = baseline_payload.canonical_json(dict(receipt))
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
            raise R241CanonicalBaselineRosterError(
                f"canonical roster receipt already exists with different bytes: {target}"
            )
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
                raise R241CanonicalBaselineRosterError(
                    f"canonical roster receipt already exists with different bytes: {target}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return target
