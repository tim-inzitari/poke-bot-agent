"""Held-out official public-bot WR aggregation (forfeit-safe)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Optional


OFFICIAL_BASELINE_IDS = (
    "iono",
    "dragapult-ex",
    "mega-abomasnow-ex",
    "mega-lucario-ex",
)


@dataclass
class HeldOutGateResult:
    win_rate: float
    games: int
    wins: float
    losses: float
    draws: float
    forfeits_excluded: int
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0
    seat_games: dict[str, int] = field(default_factory=dict)
    per_opponent: dict[str, dict[str, float]] = field(default_factory=dict)
    passed: bool = False
    reason: str = ""


def _wilson_interval(
    wins: float,
    games: int,
    *,
    confidence_z: float = 1.96,
) -> tuple[float, float]:
    """Return a Wilson interval for wins where a draw counts as half a win."""
    n = int(games)
    if n <= 0:
        return 0.0, 0.0
    wr = float(wins) / n
    z = max(0.0, float(confidence_z))
    denom = 1.0 + z * z / n
    center = (wr + z * z / (2.0 * n)) / denom
    radius = (
        z
        * math.sqrt(max(0.0, wr * (1.0 - wr) / n + z * z / (4.0 * n * n)))
        / denom
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def heldout_goal_rank(
    evidence: dict[str, Any],
    *,
    official_ids: Optional[Iterable[str]] = None,
    per_opponent_floor: float = 0.50,
    confidence_z: float = 1.96,
) -> tuple[float, ...]:
    """Rank exact held-out evidence by progress toward beating every baseline.

    The weakest official matchup is intentionally primary.  Pooled win rate
    alone can hide a catastrophic matchup and is therefore only a late
    tie-breaker.  Missing per-opponent evidence ranks below audited evidence.
    """
    ordered = tuple(official_ids or OFFICIAL_BASELINE_IDS)
    per = evidence.get("per_opponent")
    if not isinstance(per, dict) or any(oid not in per for oid in ordered):
        return (
            0.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            float(evidence.get("confidence_lower", -1.0)),
            float(evidence.get("win_rate", -1.0)),
            float(evidence.get("games", -1)),
        )

    wrs: list[float] = []
    lowers: list[float] = []
    for oid in ordered:
        bucket = per.get(oid)
        if not isinstance(bucket, dict):
            return (0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        try:
            games = int(bucket.get("games", 0))
            wins = float(bucket.get("wins", 0.0))
        except (TypeError, ValueError):
            return (0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        if games <= 0:
            return (0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        wr = wins / games
        lower, _ = _wilson_interval(wins, games, confidence_z=confidence_z)
        wrs.append(wr)
        lowers.append(lower)

    floor = max(float(per_opponent_floor), 1e-12)
    clipped_progress = sum(min(wr / floor, 1.0) for wr in wrs) / len(wrs)
    all_clear = float(all(wr >= floor for wr in wrs))
    return (
        1.0,
        all_clear,
        min(lowers),
        min(wrs),
        clipped_progress,
        sum(lowers) / len(lowers),
        float(evidence.get("confidence_lower", -1.0)),
        float(evidence.get("win_rate", -1.0)),
        float(evidence.get("games", -1)),
    )


def active_gate_goal_rank(
    evidence: dict[str, Any],
    *,
    active_gate: dict[str, Any],
) -> tuple[float, ...]:
    """Rank exact evidence by progress toward the active gate contract.

    Generic held-out ranking intentionally prioritizes the weakest matchup.
    The competition gate has a different, explicit objective: clear the
    weighted point estimate and its confidence lower bound while preserving
    the S-tier and individual-opponent floors.  Ranking checkpoints by the
    normalized bottleneck across those four requirements keeps protected-best
    selection aligned with the gate that can actually terminate training.
    """
    invalid = (0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    roster = active_gate.get("roster")
    criteria = active_gate.get("pass_criteria")
    per = evidence.get("per_opponent")
    if not isinstance(roster, list) or not roster:
        return invalid
    if not isinstance(criteria, dict) or not isinstance(per, dict):
        return invalid

    rates: dict[str, float] = {}
    weights: dict[str, float] = {}
    s_tier_ids: list[str] = []
    for entry in roster:
        if not isinstance(entry, dict):
            return invalid
        opponent_id = str(entry.get("opponent_id") or "").strip()
        bucket = per.get(opponent_id)
        if not opponent_id or not isinstance(bucket, dict):
            return invalid
        try:
            games = int(bucket.get("games", 0))
            wins = float(bucket.get("wins", 0.0))
            rate = float(bucket.get("win_rate", wins / games if games else -1.0))
            weight = float(entry.get("weight", 1.0))
        except (TypeError, ValueError, ZeroDivisionError):
            return invalid
        if games <= 0 or not 0.0 <= rate <= 1.0 or weight <= 0.0:
            return invalid
        rates[opponent_id] = rate
        weights[opponent_id] = weight
        if str(entry.get("tier") or "").strip().upper() == "S":
            s_tier_ids.append(opponent_id)
    if not s_tier_ids:
        return invalid

    total_weight = sum(weights.values())
    weighted_wr = sum(rates[oid] * weights[oid] for oid in rates) / total_weight
    confidence_lower = float(evidence.get("confidence_lower", -1.0))
    s_tier_mean = sum(rates[oid] for oid in s_tier_ids) / len(s_tier_ids)
    minimum_opponent_wr = min(rates.values())
    targets = (
        float(criteria.get("skill_weighted_win_rate", 0.0)),
        float(criteria.get("skill_weighted_confidence_lower", 0.0)),
        float(criteria.get("s_tier_mean_floor", 0.0)),
        float(criteria.get("individual_opponent_floor", 0.0)),
    )
    values = (
        weighted_wr,
        confidence_lower,
        s_tier_mean,
        minimum_opponent_wr,
    )

    normalized: list[float] = []
    for value, target in zip(values, targets):
        normalized.append(1.0 if target <= 0.0 else min(value / target, 1.0))
    audit = evidence.get("audit")
    audit_ok = not isinstance(audit, dict) or bool(audit.get("passed"))
    return (
        1.0 if audit_ok else 0.0,
        min(normalized),
        sum(normalized) / len(normalized),
        confidence_lower,
        weighted_wr,
        s_tier_mean,
        minimum_opponent_wr,
        float(evidence.get("games", -1)),
    )


def heldout_exploration_decision(
    candidate: dict[str, Any],
    anchor: dict[str, Any],
    *,
    official_ids: Optional[Iterable[str]] = None,
    per_opponent_floor: float = 0.50,
    max_per_opponent_regression: float = 0.02,
    minimum_progress_gain: float = 0.01,
) -> dict[str, Any]:
    """Decide whether a non-champion candidate is safe to keep learning from.

    The immutable held-out champion is still selected by :func:`heldout_goal_rank`.
    This narrower decision only prevents the *learner* from snapping back to an
    old checkpoint because of a few games of sampling noise.  A candidate must
    have complete equal-or-better sample coverage, remain within the explicit
    point-WR margin against every official opponent, and improve average clipped
    progress toward the per-opponent floor.  Comparing every candidate to the
    protected anchor (rather than to the previous exploratory learner) prevents
    cumulative regression across iterations.
    """
    ordered = tuple(official_ids or OFFICIAL_BASELINE_IDS)
    margin = max(0.0, float(max_per_opponent_regression))
    required_gain = max(0.0, float(minimum_progress_gain))
    floor = max(float(per_opponent_floor), 1e-12)

    def _rates(evidence: dict[str, Any]) -> tuple[dict[str, float], dict[str, int]] | None:
        per = evidence.get("per_opponent")
        if not isinstance(per, dict) or any(oid not in per for oid in ordered):
            return None
        rates: dict[str, float] = {}
        games_by_opponent: dict[str, int] = {}
        for oid in ordered:
            bucket = per.get(oid)
            if not isinstance(bucket, dict):
                return None
            try:
                games = int(bucket.get("games", 0))
                wins = float(bucket.get("wins", 0.0))
            except (TypeError, ValueError):
                return None
            if games <= 0 or not 0.0 <= wins <= float(games):
                return None
            games_by_opponent[oid] = games
            rates[oid] = wins / games
        return rates, games_by_opponent

    candidate_values = _rates(candidate)
    anchor_values = _rates(anchor)
    if candidate_values is None or anchor_values is None:
        return {
            "eligible": False,
            "reason": "missing_or_invalid_exact_per_opponent_evidence",
        }
    candidate_rates, candidate_games = candidate_values
    anchor_rates, anchor_games = anchor_values
    insufficient = [
        oid
        for oid in ordered
        if candidate_games[oid] < anchor_games[oid]
    ]
    regressions = {
        oid: candidate_rates[oid] - anchor_rates[oid]
        for oid in ordered
    }
    candidate_progress = sum(
        min(candidate_rates[oid] / floor, 1.0) for oid in ordered
    ) / len(ordered)
    anchor_progress = sum(
        min(anchor_rates[oid] / floor, 1.0) for oid in ordered
    ) / len(ordered)
    progress_gain = candidate_progress - anchor_progress
    violating = [
        oid for oid in ordered if regressions[oid] < -margin - 1e-12
    ]
    eligible = bool(
        not insufficient
        and not violating
        and progress_gain > required_gain + 1e-12
    )
    if insufficient:
        reason = "insufficient_candidate_coverage"
    elif violating:
        reason = "per_opponent_regression_exceeds_margin"
    elif progress_gain <= required_gain + 1e-12:
        reason = "insufficient_clipped_progress_gain"
    else:
        reason = "pareto_noninferior_progress"
    return {
        "eligible": eligible,
        "reason": reason,
        "max_per_opponent_regression": margin,
        "minimum_progress_gain": required_gain,
        "candidate_clipped_progress": candidate_progress,
        "anchor_clipped_progress": anchor_progress,
        "clipped_progress_gain": progress_gain,
        "per_opponent_delta": regressions,
        "insufficient_coverage": insufficient,
        "violating_opponents": violating,
    }


def aggregate_heldout_wr(
    game_rows: Iterable[dict[str, Any]],
    *,
    target_wr: float = 0.70,
    min_games: int = 200,
    official_ids: Optional[Iterable[str]] = None,
    min_games_per_opponent: Optional[int] = None,
    per_opponent_floor: float = 0.50,
    confidence_z: float = 1.96,
) -> HeldOutGateResult:
    """Aggregate a formal held-out gate with coverage and uncertainty checks.

    Each row: ``{opponent_id, winner, our_seat, baseline_failed?}``
    ``winner``: 0/1 seat index, or 2 for draw.

    Promotion/curriculum decisions use the Wilson lower bound, not a noisy
    point estimate. Every opponent must have enough games, clear a modest
    point-WR floor, and be represented from both seats.
    """
    allowed_ordered = tuple(official_ids or OFFICIAL_BASELINE_IDS)
    allowed = set(allowed_ordered)
    per: dict[str, dict[str, float]] = {
        oid: {
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
            "games": 0.0,
            "seat0_games": 0.0,
            "seat1_games": 0.0,
            "win_rate": 0.0,
        }
        for oid in allowed_ordered
    }
    wins = losses = draws = 0.0
    games = 0
    forfeits = 0
    seat_games = {"seat0": 0, "seat1": 0}
    for row in game_rows:
        opp = str(row.get("opponent_id") or "")
        if opp not in allowed:
            continue
        if row.get("baseline_failed") or row.get("invalid"):
            forfeits += 1
            continue
        our_seat = int(row.get("our_seat") or 0)
        winner = int(row.get("winner"))
        bucket = per[opp]
        bucket["games"] += 1.0
        bucket[f"seat{our_seat}_games"] += 1.0
        seat_games[f"seat{our_seat}"] += 1
        games += 1
        if winner == 2:
            draws += 1.0
            bucket["draws"] += 1.0
            wins += 0.5
            bucket["wins"] += 0.5
        elif winner == our_seat:
            wins += 1.0
            bucket["wins"] += 1.0
        else:
            losses += 1.0
            bucket["losses"] += 1.0
    wr = (wins / games) if games else 0.0
    lower, upper = _wilson_interval(
        wins,
        games,
        confidence_z=confidence_z,
    )

    per_min = (
        int(min_games_per_opponent)
        if min_games_per_opponent is not None
        else max(1, int(min_games) // max(1, len(allowed_ordered)))
    )
    coverage_ok = True
    per_floor_ok = True
    seat_balance_ok = abs(seat_games["seat0"] - seat_games["seat1"]) <= 1
    for bucket in per.values():
        n = int(bucket["games"])
        bucket["win_rate"] = float(bucket["wins"] / n) if n else 0.0
        coverage_ok = coverage_ok and n >= per_min
        per_floor_ok = per_floor_ok and bucket["win_rate"] >= float(
            per_opponent_floor
        )
        seat_balance_ok = seat_balance_ok and (
            abs(int(bucket["seat0_games"]) - int(bucket["seat1_games"])) <= 1
        )

    enough = games >= int(min_games)
    confidence_ok = lower >= float(target_wr)
    passed = bool(
        enough and coverage_ok and per_floor_ok and seat_balance_ok and confidence_ok
    )
    if passed:
        reason = "ok"
    elif not enough:
        reason = "insufficient_games"
    elif not coverage_ok:
        reason = "insufficient_per_opponent_games"
    elif not seat_balance_ok:
        reason = "seat_imbalance"
    elif not per_floor_ok:
        reason = "per_opponent_floor"
    else:
        reason = "confidence_lower_below_target"
    return HeldOutGateResult(
        win_rate=wr,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        forfeits_excluded=forfeits,
        confidence_lower=lower,
        confidence_upper=upper,
        seat_games=seat_games,
        per_opponent=per,
        passed=passed,
        reason=reason,
    )
