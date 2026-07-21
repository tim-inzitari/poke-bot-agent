"""Exact strong-public Alakazam gate contract and aggregation.

The eight public opponents are the active gate.  The original four baselines
are a separately audited, zero-weight research control and can only contribute
the explicit accepted-anchor non-regression guardrail.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable


GATE_PROGRAM_SCHEMA = "poke_bot.competition_gate_program/v1"
GATE_RESULT_SCHEMA = "poke_bot.public_agent_gate_result/v1"
LEGACY_ORIGINAL_FOUR = frozenset(
    {"iono", "dragapult-ex", "mega-abomasnow-ex", "mega-lucario-ex"}
)


def load_active_gate_contract(path: Path) -> dict[str, Any]:
    """Load the selected gate and reject stale/ambiguous launch contracts.

    Counts are derived from the active contract so a deliberate future gate
    change does not require another trainer edit.  The one forbidden rollback
    is promoting the legacy original-four research set back into the active
    gate; that historical regression must fail closed.
    """
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema") != GATE_PROGRAM_SCHEMA:
        raise ValueError("strong-public gate program has the wrong schema")
    active_id = str(contract.get("active_gate_id") or "")
    gate = contract.get("next_gate")
    if not active_id or not isinstance(gate, dict) or gate.get("id") != active_id:
        raise ValueError("active gate identity does not match next_gate")
    evaluation = gate.get("evaluation")
    roster = gate.get("roster")
    research = gate.get("research_measurements")
    if not isinstance(evaluation, dict) or not isinstance(roster, list):
        raise ValueError("active gate evaluation or roster is missing")
    if not isinstance(research, list):
        raise ValueError("active gate research controls are missing")
    roster_ids = [str(row.get("opponent_id") or "") for row in roster]
    digests = [str(row.get("content_digest") or "") for row in roster]
    games_per_opponent = int(evaluation.get("games_per_opponent") or 0)
    games_total = int(evaluation.get("games_total") or 0)
    seat0 = int(evaluation.get("seat0_games_per_opponent") or 0)
    seat1 = int(evaluation.get("seat1_games_per_opponent") or 0)
    if (
        not roster
        or len(set(roster_ids)) != len(roster)
        or not all(roster_ids)
        or len(set(digests)) != len(roster)
        or not all(digests)
        or games_per_opponent <= 0
        or games_per_opponent % 2
        or seat0 != games_per_opponent // 2
        or seat1 != games_per_opponent // 2
        or games_total != len(roster) * games_per_opponent
        or evaluation.get("mode") != "greedy"
        or evaluation.get("all_matchups_must_complete") is not True
        or evaluation.get("partial_results_gate_eligible") is not False
        or evaluation.get("sequential_early_stop") is not False
    ):
        raise ValueError("active gate is not an exact complete both-seat contract")
    if set(roster_ids) == LEGACY_ORIGINAL_FOUR:
        raise ValueError("legacy original-four research controls cannot be the active gate")
    if any(float(row.get("weight") or 0.0) <= 0.0 for row in roster):
        raise ValueError("every active-gate opponent must have positive weight")

    research_ids = [str(row.get("opponent_id") or "") for row in research]
    if (
        len(set(research_ids)) != len(research_ids)
        or any(not opponent_id for opponent_id in research_ids)
        or any(
            int(row.get("games") or 0) <= 0
            or int(row.get("games") or 0) % 2
            or int(row.get("seat0_games") or 0)
            != int(row.get("games") or 0) // 2
            or int(row.get("seat1_games") or 0)
            != int(row.get("games") or 0) // 2
            or float(row.get("gate_weight") or 0.0) != 0.0
            or row.get("included_in_gate_pass") is not False
            for row in research
        )
    ):
        raise ValueError("research controls must be exact, both-seat, and zero-weight")

    semantics = contract.get("active_gate_semantics") or {}
    if semantics and (
        int(semantics.get("gate_roster_size") or 0) != len(roster)
        or int(semantics.get("games_per_opponent") or 0) != games_per_opponent
        or int(semantics.get("gate_games_total") or 0) != games_total
    ):
        raise ValueError("active_gate_semantics disagrees with the selected gate")
    return contract


def verify_roster_content(
    gate: dict[str, Any], installed_digests: dict[str, str]
) -> None:
    for row in gate.get("roster") or []:
        opponent_id = str(row.get("opponent_id") or "")
        expected = str(row.get("content_digest") or "")
        actual = str(installed_digests.get(opponent_id) or "")
        if not actual or actual != expected:
            raise ValueError(
                f"strong-public package digest mismatch for {opponent_id}: "
                f"expected={expected!r} actual={actual!r}"
            )


def _score(row: dict[str, Any]) -> float:
    seat = int(row["our_seat"])
    winner = int(row["winner"])
    if winner == 2:
        return 0.5
    return 1.0 if winner == seat else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    position = min(1.0, max(0.0, q)) * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _matchup_rows(
    rows: Iterable[dict[str, Any]], ordered_ids: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[float, float]]]]:
    valid_rows = [
        row
        for row in rows
        if not row.get("invalid") and not row.get("baseline_failed")
    ]
    by_job = {int(row["job_index"]): row for row in valid_rows}
    matchups: list[dict[str, Any]] = []
    clusters: dict[str, list[tuple[float, float]]] = {key: [] for key in ordered_ids}
    for opponent_id in ordered_ids:
        selected = [row for row in valid_rows if row.get("opponent_id") == opponent_id]
        scores = [_score(row) for row in selected]
        matchups.append(
            {
                "opponent_id": opponent_id,
                "games": len(selected),
                "wins": sum(score == 1.0 for score in scores),
                "draws": sum(score == 0.5 for score in scores),
                "losses": sum(score == 0.0 for score in scores),
                "wr": statistics.fmean(scores) if scores else 0.0,
                "seat0": sum(int(row.get("our_seat", -1)) == 0 for row in selected),
                "seat1": sum(int(row.get("our_seat", -1)) == 1 for row in selected),
            }
        )
    for pair_index in range(max(by_job, default=-1) // 2 + 1):
        first = by_job.get(2 * pair_index)
        second = by_job.get(2 * pair_index + 1)
        if first is None or second is None:
            continue
        opponent_id = str(first.get("opponent_id") or "")
        if (
            opponent_id not in clusters
            or second.get("opponent_id") != opponent_id
            or {int(first.get("our_seat", -1)), int(second.get("our_seat", -1))}
            != {0, 1}
        ):
            continue
        seat_scores = {int(first["our_seat"]): _score(first), int(second["our_seat"]): _score(second)}
        clusters[opponent_id].append((seat_scores[0], seat_scores[1]))
    return matchups, clusters


def _weighted_cluster_interval(
    clusters: dict[str, list[tuple[float, float]]],
    weights: dict[str, float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    ordered = tuple(weights)
    total_weight = sum(weights.values())

    def weighted_rate(sampled: dict[str, list[tuple[float, float]]]) -> float:
        return sum(
            weights[opponent_id]
            * statistics.fmean(
                score
                for pair in sampled[opponent_id]
                for score in pair
            )
            for opponent_id in ordered
        ) / total_weight

    center = weighted_rate(clusters)
    rng = random.Random(int(seed))
    samples: list[float] = []
    for _ in range(int(resamples)):
        drawn = {
            opponent_id: [
                pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))
            ]
            for opponent_id, pairs in clusters.items()
        }
        samples.append(weighted_rate(drawn))
    samples.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    return center, _quantile(samples, alpha), _quantile(samples, 1.0 - alpha)


def build_active_gate_result(
    *,
    contract: dict[str, Any],
    checkpoint: str,
    checkpoint_digest: str,
    iteration: int,
    gate_rows: list[dict[str, Any]],
    gate_audit: dict[str, Any],
    gate_seed: int,
    bootstrap_resamples: int = 4000,
) -> dict[str, Any]:
    """Aggregate only the active gate; research controls are never implicit.

    This is the production-loop entry point.  It deliberately has no argument
    for the historical original-four sweep, so a normal RL iteration cannot
    silently spend another 1,000 games on or gate against that research set.
    """
    active = contract["next_gate"]
    evaluation = active["evaluation"]
    criteria = active["pass_criteria"]
    roster = active["roster"]
    ordered_ids = tuple(str(row["opponent_id"]) for row in roster)
    weights = {str(row["opponent_id"]): float(row["weight"]) for row in roster}
    matchups, clusters = _matchup_rows(gate_rows, ordered_ids)
    expected_pairs = int(evaluation["games_per_opponent"]) // 2
    cluster_exact = all(
        len(clusters[opponent_id]) == expected_pairs
        for opponent_id in ordered_ids
    )
    confidence = float(evaluation.get("confidence_level") or 0.9)
    if cluster_exact:
        weighted_wr, lower, upper = _weighted_cluster_interval(
            clusters,
            weights,
            confidence=confidence,
            resamples=bootstrap_resamples,
            seed=gate_seed ^ 0xA11A,
        )
    else:
        weighted_wr = lower = upper = 0.0
    by_id = {str(row["opponent_id"]): row for row in matchups}
    s_ids = [str(row["opponent_id"]) for row in roster if row.get("tier") == "S"]
    s_mean = (
        statistics.fmean(float(by_id[key]["wr"]) for key in s_ids)
        if s_ids
        else weighted_wr
    )
    minimum_wr = min((float(row["wr"]) for row in matchups), default=0.0)
    gate_audit_ok = bool(
        gate_audit.get("passed") is True
        and gate_audit.get("exact_distribution") is True
        and gate_audit.get("exact_weights") is True
        and cluster_exact
        and int(gate_audit.get("valid_games") or 0)
        == int(evaluation["games_total"])
        and set((gate_audit.get("per_opponent") or {})) == set(ordered_ids)
    )
    checks = {
        "audit": gate_audit_ok,
        "skill_weighted_win_rate": weighted_wr
        >= float(criteria["skill_weighted_win_rate"]),
        "skill_weighted_confidence_lower": lower
        >= float(criteria["skill_weighted_confidence_lower"]),
    }
    if "s_tier_mean_floor" in criteria:
        checks["s_tier_mean_floor"] = s_mean >= float(
            criteria["s_tier_mean_floor"]
        )
    if "individual_opponent_floor" in criteria:
        checks["individual_opponent_floor"] = minimum_wr >= float(
            criteria["individual_opponent_floor"]
        )
    seed_manifest = {
        "gate_seed": int(gate_seed),
        "gate_games": int(evaluation["games_total"]),
        "mapping": (
            "seed_base + job_index; adjacent job pair is same opponent "
            "with seats 0/1"
        ),
    }
    seed_digest = "sha256:" + hashlib.sha256(
        json.dumps(seed_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": GATE_RESULT_SCHEMA,
        "gate_id": active["id"],
        "iteration": int(iteration),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": str(checkpoint_digest),
        "games": int(evaluation["games_total"]),
        "skill_weighted_wr": weighted_wr,
        "confidence_lower": lower,
        "confidence_upper": upper,
        "confidence_level": confidence,
        "confidence_method": (
            "opponent-stratified matched-seat cluster nonparametric bootstrap"
        ),
        "bootstrap_resamples": int(bootstrap_resamples),
        "s_tier_mean": s_mean,
        "minimum_opponent_wr": minimum_wr,
        "passed": all(checks.values()),
        "checks": checks,
        "matchups": matchups,
        "audit": {
            **dict(gate_audit),
            "passed": gate_audit_ok,
            "both_seats": cluster_exact,
            "greedy": gate_audit.get("greedy_required") is True,
            "fixed_seed_manifest": seed_manifest,
            "fixed_seed_manifest_digest": seed_digest,
        },
    }


def build_strong_public_gate_result(
    *,
    contract: dict[str, Any],
    checkpoint: str,
    checkpoint_digest: str,
    iteration: int,
    gate_rows: list[dict[str, Any]],
    gate_audit: dict[str, Any],
    research_rows: list[dict[str, Any]],
    research_audit: dict[str, Any],
    gate_seed: int,
    research_seed: int,
    bootstrap_resamples: int = 4000,
) -> dict[str, Any]:
    active = contract["next_gate"]
    evaluation = active["evaluation"]
    criteria = active["pass_criteria"]
    roster = active["roster"]
    research_contract = active["research_measurements"]
    ordered_ids = tuple(str(row["opponent_id"]) for row in roster)
    research_ids = tuple(str(row["opponent_id"]) for row in research_contract)
    weights = {str(row["opponent_id"]): float(row["weight"]) for row in roster}
    matchups, clusters = _matchup_rows(gate_rows, ordered_ids)
    research_matchups, research_clusters = _matchup_rows(research_rows, research_ids)
    expected_pairs = int(evaluation["games_per_opponent"]) // 2
    cluster_exact = all(len(clusters[opponent_id]) == expected_pairs for opponent_id in ordered_ids)
    confidence = float(evaluation.get("confidence_level") or 0.9)
    if cluster_exact:
        weighted_wr, lower, upper = _weighted_cluster_interval(
            clusters,
            weights,
            confidence=confidence,
            resamples=bootstrap_resamples,
            seed=gate_seed ^ 0xA11A,
        )
    else:
        weighted_wr = lower = upper = 0.0
    by_id = {str(row["opponent_id"]): row for row in matchups}
    s_ids = [str(row["opponent_id"]) for row in roster if row.get("tier") == "S"]
    s_mean = statistics.fmean(float(by_id[key]["wr"]) for key in s_ids)
    minimum_wr = min(float(row["wr"]) for row in matchups)
    research_games = sum(int(row["games"]) for row in research_matchups)
    research_score = sum(
        float(row["wr"]) * int(row["games"]) for row in research_matchups
    )
    research_wr = research_score / research_games if research_games else 0.0
    gate_audit_ok = bool(
        gate_audit.get("passed") is True
        and gate_audit.get("exact_distribution") is True
        and gate_audit.get("exact_weights") is True
        and cluster_exact
        and int(gate_audit.get("valid_games") or 0)
        == int(evaluation["games_total"])
    )
    research_audit_ok = bool(
        research_audit.get("passed") is True
        and research_audit.get("exact_distribution") is True
        and research_audit.get("exact_weights") is True
        and all(len(research_clusters[key]) == 125 for key in research_ids)
        and research_games == 1000
    )
    checks = {
        "audit": gate_audit_ok,
        "skill_weighted_win_rate": weighted_wr
        >= float(criteria["skill_weighted_win_rate"]),
        "skill_weighted_confidence_lower": lower
        >= float(criteria["skill_weighted_confidence_lower"]),
        "s_tier_mean_floor": s_mean >= float(criteria["s_tier_mean_floor"]),
        "individual_opponent_floor": minimum_wr
        >= float(criteria["individual_opponent_floor"]),
    }
    research_checks = {
        "research_control_audit": research_audit_ok,
        "accepted_official_holdout_non_regression": research_wr
        >= float(criteria.get("accepted_official_holdout_non_regression", 0.0)),
    }
    seed_manifest = {
        "gate_seed": int(gate_seed),
        "gate_games": int(evaluation["games_total"]),
        "research_seed": int(research_seed),
        "research_games": research_games,
        "mapping": "seed_base + job_index; adjacent job pair is same opponent with seats 0/1",
    }
    seed_digest = "sha256:" + hashlib.sha256(
        json.dumps(seed_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": GATE_RESULT_SCHEMA,
        "gate_id": active["id"],
        "iteration": int(iteration),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": str(checkpoint_digest),
        "games": int(evaluation["games_total"]),
        "skill_weighted_wr": weighted_wr,
        "confidence_lower": lower,
        "confidence_upper": upper,
        "confidence_level": confidence,
        "confidence_method": "opponent-stratified matched-seat cluster nonparametric bootstrap",
        "bootstrap_resamples": int(bootstrap_resamples),
        "s_tier_mean": s_mean,
        "minimum_opponent_wr": minimum_wr,
        "passed": all(checks.values()),
        "checks": checks,
        "research_checks": research_checks,
        "matchups": matchups,
        "audit": {
            **dict(gate_audit),
            "passed": gate_audit_ok,
            "both_seats": cluster_exact,
            "greedy": gate_audit.get("greedy_required") is True,
            "fixed_seed_manifest": seed_manifest,
            "fixed_seed_manifest_digest": seed_digest,
        },
        "research_controls": {
            "games": research_games,
            "pooled_wr": research_wr,
            "gate_weight": 0.0,
            "included_in_skill_weighted_wr": False,
            "matchups": research_matchups,
            "audit": {**dict(research_audit), "passed": research_audit_ok},
        },
    }
