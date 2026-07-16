"""Held-out official public-bot WR aggregation (forfeit-safe)."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    per_opponent: dict[str, dict[str, float]] = field(default_factory=dict)
    passed: bool = False
    reason: str = ""


def aggregate_heldout_wr(
    game_rows: Iterable[dict[str, Any]],
    *,
    target_wr: float = 0.70,
    min_games: int = 200,
    official_ids: Optional[Iterable[str]] = None,
) -> HeldOutGateResult:
    """Aggregate seat-balanced formal games; drop baseline_failed forfeits.

    Each row: ``{opponent_id, winner, our_seat, baseline_failed?}``
    ``winner``: 0/1 seat index, or 2 for draw.
    """
    allowed = set(official_ids or OFFICIAL_BASELINE_IDS)
    per: dict[str, dict[str, float]] = {
        oid: {"wins": 0.0, "losses": 0.0, "draws": 0.0, "games": 0.0}
        for oid in allowed
    }
    wins = losses = draws = 0.0
    games = 0
    forfeits = 0
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
    passed = games >= int(min_games) and wr >= float(target_wr)
    reason = "ok" if passed else (
        "insufficient_games" if games < int(min_games) else "wr_below_target"
    )
    return HeldOutGateResult(
        win_rate=wr,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        forfeits_excluded=forfeits,
        per_opponent=per,
        passed=passed,
        reason=reason,
    )
