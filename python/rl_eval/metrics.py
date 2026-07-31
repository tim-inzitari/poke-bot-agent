"""Draw-aware, seat-stratified evaluation metrics (domain-agnostic)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
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


def stratified_bootstrap_interval(
    strata: dict[str, list[float]],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    groups = [list(v) for _, v in sorted(strata.items()) if v]
    observed = [x for group in groups for x in group]
    if not observed:
        return 0.0, 0.0, 0.0
    center = sum(observed) / len(observed)
    if len(observed) == 1 or resamples <= 0:
        return center, center, center
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(int(resamples)):
        sample = [
            group[rng.randrange(len(group))]
            for group in groups
            for _ in range(len(group))
        ]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    lo_i = max(0, min(len(means) - 1, int((alpha / 2.0) * len(means))))
    hi_i = max(
        0,
        min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)) - 1),
    )
    return center, means[lo_i], means[hi_i]


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
    formal_scores: list[tuple[int, float]] = field(default_factory=list)

    def record(
        self,
        *,
        our_seat: int,
        winner: int,
        is_mirror: bool,
        mcts_on: bool = True,
        draw_code: int = 2,
    ) -> None:
        """Record one game. ``winner == draw_code`` is a draw; else seat id."""
        if winner == draw_code:
            score = 0.5
        elif winner == our_seat:
            score = 1.0
        else:
            score = 0.0
        if not mcts_on:
            return
        self.games += 1
        if winner == draw_code:
            self.draws += 1
        elif winner == our_seat:
            self.wins += 1
        else:
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
        self.formal_scores.append((int(our_seat), score))

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def draw_aware_lower(self) -> float:
        strata: dict[str, list[float]] = {}
        for seat, score in self.formal_scores:
            strata.setdefault(f"seat{seat}", []).append(score)
        return stratified_bootstrap_interval(strata, seed=17)[1]

    def to_dict(self) -> dict[str, Any]:
        interval = stratified_bootstrap_interval(
            {
                f"seat{seat}": [
                    score for row_seat, score in self.formal_scores if row_seat == seat
                ]
                for seat in (0, 1)
            },
            seed=17,
        )
        return {
            "opponent_id": self.opponent_id,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "wr": self.win_rate,
            "draw_aware_score_interval": {
                "center": interval[0],
                "lower": interval[1],
                "upper": interval[2],
            },
            "seat0_wr": (self.seat0_wins / self.seat0_games) if self.seat0_games else None,
            "seat1_wr": (self.seat1_wins / self.seat1_games) if self.seat1_games else None,
        }


@dataclass
class FieldReport:
    matchups: dict[str, MatchupStats] = field(default_factory=dict)
    gate_threshold: float = 0.55
    min_games_per_opponent: int = 1

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
        mcts_on: bool = True,
        draw_code: int = 2,
    ) -> None:
        self.get(opponent_id).record(
            our_seat=our_seat,
            winner=winner,
            is_mirror=is_mirror,
            mcts_on=mcts_on,
            draw_code=draw_code,
        )

    def opponents_passing(self) -> list[str]:
        return [
            oid
            for oid, st in self.matchups.items()
            if st.games >= self.min_games_per_opponent
            and st.draw_aware_lower >= self.gate_threshold
        ]
