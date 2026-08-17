"""Exact strong-public specialist gate contract and aggregation.

The established eight public opponents plus every frozen completed specialist
are the active gate. Frozen specialists at S+ and above count in the S-tier
safety mean and premium individual-floor check. The original four baselines
are a separately audited, zero-weight research control.
"""

from __future__ import annotations

import copy
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
S_TIER_TIERS = frozenset({"S", "S+", "S++"})
PREMIUM_TIERS = frozenset({"S+", "S++"})
R192_SPLUSPLUS_SEMANTICS_KEY = "exact_additional_splusplus_specialist"
R192_MARNIE_H10_OPPONENT_ID = (
    "specialist-marnie-final-format-h10-f20efb20f5c3"
)
R192_MARNIE_H10_CHECKPOINT_DIGEST = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
R192_MARNIE_H10_CONTENT_DIGEST = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
R192_HISTORICAL_MARNIE_OPPONENT_ID = (
    "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98"
)
R192_MARNIE_SPLUSPLUS_SEMANTICS = {
    "opponent_id": R192_MARNIE_H10_OPPONENT_ID,
    "checkpoint_digest": R192_MARNIE_H10_CHECKPOINT_DIGEST,
    "content_digest": R192_MARNIE_H10_CONTENT_DIGEST,
    "tier": "S++",
    "weight": 4.0,
    "strong_public_practice_floor_games": 1024,
}


def materialize_fallback_gate_contract(
    contract: dict[str, Any],
    *,
    completed_iteration: int,
    prior_gate_passed: bool,
) -> dict[str, Any] | None:
    """Build the owner-authorized post-iteration fallback contract.

    The primary gate remains authoritative through the named completed
    iteration. The fallback is a distinct gate identity and changes only the
    configured confidence-lower threshold. Returning ``None`` means the
    primary gate still applies or already passed.
    """

    fallback = contract.get("fallback_transition")
    if not isinstance(fallback, dict):
        return None
    activate_after = int(fallback.get("activate_after_completed_iteration", -1))
    if int(completed_iteration) < activate_after or bool(prior_gate_passed):
        return None
    primary = contract.get("next_gate")
    if not isinstance(primary, dict):
        raise ValueError("fallback transition has no primary gate")
    primary_id = str(primary.get("id") or "")
    fallback_id = str(fallback.get("id") or "")
    if (
        not fallback_id
        or fallback_id == primary_id
        or str(fallback.get("prior_gate_id") or "") != primary_id
        or fallback.get("only_if_prior_gate_unpassed") is not True
    ):
        raise ValueError("fallback gate identity/condition is invalid")
    criteria = primary.get("pass_criteria")
    if not isinstance(criteria, dict):
        raise ValueError("primary gate criteria are missing")
    prior_lower = float(criteria.get("skill_weighted_confidence_lower", -1.0))
    expected_prior = float(fallback.get("prior_confidence_lower", -2.0))
    next_lower = float(fallback.get("skill_weighted_confidence_lower", -1.0))
    if (
        abs(prior_lower - expected_prior) > 1e-12
        or not 0.0 <= next_lower <= prior_lower
    ):
        raise ValueError("fallback confidence threshold is inconsistent")

    derived = copy.deepcopy(contract)
    gate = copy.deepcopy(primary)
    gate["id"] = fallback_id
    gate["label"] = str(fallback.get("label") or fallback_id)
    gate["status"] = "queued"
    gate["pass_criteria"]["skill_weighted_confidence_lower"] = next_lower
    gate["activation"] = {
        "schema": "poke_bot.iteration_gate_fallback_activation/v1",
        "prior_gate_id": primary_id,
        "activate_after_completed_iteration": activate_after,
        "observed_completed_iteration": int(completed_iteration),
        "prior_gate_passed": False,
        "only_changed_criterion": "skill_weighted_confidence_lower",
        "prior_confidence_lower": prior_lower,
        "active_confidence_lower": next_lower,
    }
    derived["active_gate_id"] = fallback_id
    derived["next_gate"] = gate
    derived["derived_from_gate_id"] = primary_id
    derived["activated_fallback_transition"] = copy.deepcopy(fallback)
    derived.pop("fallback_transition", None)
    return derived


def _validate_r192_marnie_splusplus_roster(
    *,
    roster: list[dict[str, Any]],
    semantics: dict[str, Any],
) -> None:
    """Authorize S++ only for the checksum-bound additional r192 Marnie row."""

    splusplus_rows = [row for row in roster if row.get("tier") == "S++"]
    binding = semantics.get(R192_SPLUSPLUS_SEMANTICS_KEY)
    if binding is None and not splusplus_rows:
        return
    if (
        binding != R192_MARNIE_SPLUSPLUS_SEMANTICS
        or type(
            dict(binding).get("strong_public_practice_floor_games")
        ) is not int
        or len(splusplus_rows) != 1
    ):
        raise ValueError("invalid exact additional S++ specialist semantics")

    h10 = splusplus_rows[0]
    historical = next(
        (
            row
            for row in roster
            if row.get("opponent_id") == R192_HISTORICAL_MARNIE_OPPONENT_ID
        ),
        None,
    )
    if (
        h10.get("opponent_id") != R192_MARNIE_H10_OPPONENT_ID
        or h10.get("content_digest") != R192_MARNIE_H10_CONTENT_DIGEST
        or h10.get("frozen_checkpoint_digest")
        != R192_MARNIE_H10_CHECKPOINT_DIGEST
        or h10.get("frozen_specialist") is not True
        or (h10.get("tier"), float(h10.get("weight") or 0.0)) != ("S++", 4.0)
        or h10.get("strong_public_practice_floor_games") != 1024
        or type(h10.get("strong_public_practice_floor_games")) is not int
        or not isinstance(historical, dict)
        or historical.get("frozen_specialist") is not True
        or not str(historical.get("content_digest") or "")
        or historical.get("content_digest") == R192_MARNIE_H10_CONTENT_DIGEST
        or any(
            (row.get("tier"), float(row.get("weight") or 0.0)) != ("S+", 2.0)
            for row in roster
            if row.get("frozen_specialist") is True
            and row.get("opponent_id") != R192_MARNIE_H10_OPPONENT_ID
        )
    ):
        raise ValueError("invalid exact additional S++ specialist roster")


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

    rating_simulation = gate.get("kaggle_rating_simulation")
    if rating_simulation is not None:
        if not isinstance(rating_simulation, dict):
            raise ValueError("kaggle_rating_simulation must be an object")
        anchors = [
            float(row["kaggle_rating_anchor"])
            for row in roster
            if row.get("kaggle_rating_anchor") is not None
        ]
        if (
            rating_simulation.get("separate_from_premium_strength_gate") is not True
            or rating_simulation.get("training_eligible") is not False
            or rating_simulation.get("replay_eligible") is not False
            or int(rating_simulation.get("minimum_anchor_count") or 0) < 1
            or len(anchors)
            < int(rating_simulation.get("minimum_anchor_count") or 0)
            or not 0.5 < float(rating_simulation.get("confidence_level") or 0.0) < 1.0
            or float(rating_simulation.get("projected_rating_lower_bound") or 0.0)
            <= 0.0
            or any(not 0.0 < rating < 5000.0 for rating in anchors)
        ):
            raise ValueError("invalid independent Kaggle rating simulation contract")

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
    if not isinstance(semantics, dict):
        raise ValueError("active_gate_semantics must be an object")
    _validate_r192_marnie_splusplus_roster(roster=roster, semantics=semantics)
    if semantics and (
        int(semantics.get("gate_roster_size") or 0) != len(roster)
        or int(semantics.get("games_per_opponent") or 0) != games_per_opponent
        or int(semantics.get("gate_games_total") or 0) != games_total
    ):
        raise ValueError("active_gate_semantics disagrees with the selected gate")
    fallback = contract.get("fallback_transition")
    if fallback is not None:
        if not isinstance(fallback, dict):
            raise ValueError("fallback_transition must be an object")
        materialized = materialize_fallback_gate_contract(
            contract,
            completed_iteration=int(
                fallback.get("activate_after_completed_iteration", -1)
            ),
            prior_gate_passed=False,
        )
        if materialized is None:
            raise ValueError("fallback_transition cannot materialize at its boundary")
        # Recursively validate the derived gate after removing the staging
        # instruction, preventing an accidental infinite materialization loop.
        temporary_gate = materialized["next_gate"]
        if (
            temporary_gate["evaluation"] != gate["evaluation"]
            or temporary_gate["roster"] != roster
            or temporary_gate["research_measurements"] != research
        ):
            raise ValueError("fallback gate changed evaluation membership")
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


def _bradley_terry_rating(
    samples: dict[str, list[tuple[float, float]]],
    anchors: dict[str, float],
) -> float:
    """Fit one candidate rating against fixed Kaggle-rated opponents."""

    def residual(candidate_rating: float) -> float:
        value = 0.0
        for opponent_id, pairs in samples.items():
            observed = sum(score for pair in pairs for score in pair)
            games = 2 * len(pairs)
            expected = games / (
                1.0
                + 10.0
                ** ((float(anchors[opponent_id]) - candidate_rating) / 400.0)
            )
            value += observed - expected
        return value

    low, high = -1000.0, 3000.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _kaggle_rating_simulation(
    clusters: dict[str, list[tuple[float, float]]],
    roster: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Project rating from actual balanced-seat games, independently of WR."""

    anchors = {
        str(row["opponent_id"]): float(row["kaggle_rating_anchor"])
        for row in roster
        if row.get("kaggle_rating_anchor") is not None
    }
    minimum_anchors = int(contract["minimum_anchor_count"])
    confidence = float(contract.get("confidence_level", 0.90))
    resamples = int(contract.get("bootstrap_resamples", 4000))
    minimum_rating = float(contract["projected_rating_lower_bound"])
    exact = (
        len(anchors) >= minimum_anchors
        and all(opponent_id in clusters and clusters[opponent_id] for opponent_id in anchors)
    )
    if not exact:
        return {
            "schema": "poke_bot.kaggle_rating_simulation/v1",
            "passed": False,
            "anchor_count": len(anchors),
            "minimum_anchor_count": minimum_anchors,
            "projected_rating": None,
            "confidence_lower": None,
            "projected_rating_lower_bound": minimum_rating,
            "reason": "insufficient_exact_rating_anchors",
        }
    selected = {opponent_id: clusters[opponent_id] for opponent_id in anchors}
    center = _bradley_terry_rating(selected, anchors)
    rng = random.Random(int(seed) ^ 0xA1A2A3)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = {
            opponent_id: [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
            for opponent_id, pairs in selected.items()
        }
        estimates.append(_bradley_terry_rating(sampled, anchors))
    estimates.sort()
    alpha = 1.0 - confidence
    lower = _quantile(estimates, alpha)
    return {
        "schema": "poke_bot.kaggle_rating_simulation/v1",
        "passed": lower >= minimum_rating,
        "training_eligible": False,
        "replay_eligible": False,
        "estimator": "multiplayer_bradley_terry_elo_mle",
        "actual_simulated_games": sum(2 * len(pairs) for pairs in selected.values()),
        "balanced_seat_pairs": sum(len(pairs) for pairs in selected.values()),
        "anchor_count": len(anchors),
        "minimum_anchor_count": minimum_anchors,
        "anchors": anchors,
        "projected_rating": center,
        "confidence_level": confidence,
        "confidence_lower": lower,
        "bootstrap_resamples": resamples,
        "projected_rating_lower_bound": minimum_rating,
        "separate_from_skill_weighted_win_rate": True,
    }


def _s_plus_floor_check(
    roster: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    criteria: dict[str, Any],
) -> tuple[bool, list[str], float, int]:
    floor = float(criteria.get("s_plus_individual_floor", 0.0))
    allowance = int(criteria.get("s_plus_below_floor_allowance", 0))
    if not 0.0 <= floor <= 1.0 or allowance < 0:
        raise ValueError("invalid S+ matchup floor contract")
    # Keep the established result-field names for compatibility, while treating
    # S++ as the same premium floor class as S+.
    below = sorted(
        str(row["opponent_id"])
        for row in roster
        if row.get("tier") in PREMIUM_TIERS
        and float(by_id[str(row["opponent_id"])]["wr"]) < floor
    )
    return len(below) <= allowance, below, floor, allowance


def _s_tier_mean(
    roster: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    fallback: float,
) -> float:
    """Return the tier-S-or-higher weighted matchup mean.

    Gate weights already determine the primary strength metric.  Preserve that
    meaning for the S-tier safety metric so an explicit S++/4.0 opponent has
    its declared premium influence rather than being silently flattened.
    """

    s_rows = [row for row in roster if row.get("tier") in S_TIER_TIERS]
    if not s_rows:
        return fallback
    total_weight = sum(float(row["weight"]) for row in s_rows)
    if total_weight <= 0.0:
        raise ValueError("S-tier opponents must have positive total weight")
    return sum(
        float(by_id[str(row["opponent_id"])]["wr"]) * float(row["weight"])
        for row in s_rows
    ) / total_weight


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
    s_mean = _s_tier_mean(roster, by_id, fallback=weighted_wr)
    minimum_wr = min((float(row["wr"]) for row in matchups), default=0.0)
    s_plus_ok, s_plus_below, s_plus_floor, s_plus_allowance = (
        _s_plus_floor_check(roster, by_id, criteria)
    )
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
    if "s_plus_individual_floor" in criteria:
        checks["s_plus_matchup_floor_allowance"] = s_plus_ok
    rating_simulation = None
    if isinstance(active.get("kaggle_rating_simulation"), dict):
        rating_simulation = _kaggle_rating_simulation(
            clusters,
            roster,
            dict(active["kaggle_rating_simulation"]),
            seed=gate_seed,
        )
        checks["kaggle_rating_simulation"] = bool(rating_simulation["passed"])
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
    result = {
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
        "s_plus_individual_floor": s_plus_floor,
        "s_plus_below_floor_allowance": s_plus_allowance,
        "s_plus_below_floor_count": len(s_plus_below),
        "s_plus_below_floor_opponent_ids": s_plus_below,
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
    if rating_simulation is not None:
        result["kaggle_rating_simulation"] = rating_simulation
    return result


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
    s_mean = _s_tier_mean(roster, by_id, fallback=weighted_wr)
    minimum_wr = min(float(row["wr"]) for row in matchups)
    s_plus_ok, s_plus_below, s_plus_floor, s_plus_allowance = (
        _s_plus_floor_check(roster, by_id, criteria)
    )
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
        "s_plus_matchup_floor_allowance": s_plus_ok,
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
        "s_plus_individual_floor": s_plus_floor,
        "s_plus_below_floor_allowance": s_plus_allowance,
        "s_plus_below_floor_count": len(s_plus_below),
        "s_plus_below_floor_opponent_ids": s_plus_below,
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
