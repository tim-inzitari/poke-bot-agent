"""Wilson score intervals and seat-swap eval helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return ``(center, lower, upper)`` Wilson score interval for a binomial WR.

    ``wins`` may be fractional (draws as 0.5).
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    phat = float(wins) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(max(0.0, (phat * (1 - phat) + z2 / (4 * n)) / n))
    return centre, max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_lower(wins: float, n: int, z: float = 1.96) -> float:
    return wilson_interval(wins, n, z)[1]


@dataclass
class MatchupStats:
    opponent_id: str
    wins: float = 0.0
    losses: float = 0.0
    draws: float = 0.0
    games: int = 0
    mirror_wins: float = 0.0
    mirror_games: int = 0
    non_mirror_wins: float = 0.0
    non_mirror_games: int = 0
    seat0_wins: float = 0.0
    seat0_games: int = 0
    seat1_wins: float = 0.0
    seat1_games: int = 0
    mcts_on_wins: float = 0.0
    mcts_on_games: int = 0
    greedy_wins: float = 0.0
    greedy_games: int = 0

    def record(
        self,
        *,
        our_seat: int,
        winner: int,
        is_mirror: bool,
        mcts_on: bool,
    ) -> None:
        self.games += 1
        if winner == 2:
            score = 0.5
            self.draws += 1
        elif winner == our_seat:
            score = 1.0
            self.wins += 1
        else:
            score = 0.0
            self.losses += 1

        if our_seat == 0:
            self.seat0_games += 1
            self.seat0_wins += score
        else:
            self.seat1_games += 1
            self.seat1_wins += score

        if is_mirror:
            self.mirror_games += 1
            self.mirror_wins += score
        else:
            self.non_mirror_games += 1
            self.non_mirror_wins += score

        if mcts_on:
            self.mcts_on_games += 1
            self.mcts_on_wins += score
        else:
            self.greedy_games += 1
            self.greedy_wins += score

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def wilson_lo(self) -> float:
        score = self.wins + 0.5 * self.draws
        return wilson_lower(score, self.games)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "wr": self.win_rate,
            "wilson_lo": self.wilson_lo,
            "mirror": {
                "games": self.mirror_games,
                "wr": (self.mirror_wins / self.mirror_games) if self.mirror_games else None,
            },
            "non_mirror": {
                "games": self.non_mirror_games,
                "wr": (self.non_mirror_wins / self.non_mirror_games)
                if self.non_mirror_games
                else None,
            },
            "seat0_wr": (self.seat0_wins / self.seat0_games) if self.seat0_games else None,
            "seat1_wr": (self.seat1_wins / self.seat1_games) if self.seat1_games else None,
            "mcts_on_wr": (self.mcts_on_wins / self.mcts_on_games) if self.mcts_on_games else None,
            "greedy_wr": (self.greedy_wins / self.greedy_games) if self.greedy_games else None,
        }


@dataclass
class FieldReport:
    matchups: dict[str, MatchupStats] = field(default_factory=dict)
    gate_threshold: float = 0.55

    def get(self, opponent_id: str) -> MatchupStats:
        if opponent_id not in self.matchups:
            self.matchups[opponent_id] = MatchupStats(opponent_id=opponent_id)
        return self.matchups[opponent_id]

    def merge_game(
        self,
        opponent_id: str,
        *,
        our_seat: int,
        winner: int,
        is_mirror: bool,
        mcts_on: bool,
    ) -> None:
        self.get(opponent_id).record(
            our_seat=our_seat, winner=winner, is_mirror=is_mirror, mcts_on=mcts_on
        )

    def opponents_passing(self) -> list[str]:
        return [
            oid
            for oid, st in self.matchups.items()
            if st.games > 0 and st.wilson_lo >= self.gate_threshold
        ]

    def opponents_failing(self) -> list[str]:
        return [
            oid
            for oid, st in self.matchups.items()
            if st.games > 0 and st.wilson_lo < self.gate_threshold
        ]

    def summary(self) -> dict[str, Any]:
        rows = [st.to_dict() for st in sorted(self.matchups.values(), key=lambda s: s.opponent_id)]
        n_pass = len(self.opponents_passing())
        n_eval = sum(1 for st in self.matchups.values() if st.games > 0)
        return {
            "n_evaluated": n_eval,
            "n_passing_wilson": n_pass,
            "gate_threshold": self.gate_threshold,
            "all_pass": n_eval > 0 and n_pass == n_eval,
            "matchups": rows,
        }
