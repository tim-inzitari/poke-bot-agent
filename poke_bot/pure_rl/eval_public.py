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
    z = max(0.0, float(confidence_z))
    if games:
        denom = 1.0 + z * z / games
        center = (wr + z * z / (2.0 * games)) / denom
        radius = (
            z
            * math.sqrt(
                max(0.0, wr * (1.0 - wr) / games + z * z / (4.0 * games * games))
            )
            / denom
        )
        lower = max(0.0, center - radius)
        upper = min(1.0, center + radius)
    else:
        lower = upper = 0.0

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
