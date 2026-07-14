"""Draw-aware, seat-stratified evaluation metrics."""

from __future__ import annotations

import math
import random
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


def stratified_bootstrap_interval(
    strata: dict[str, list[float]],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile interval for mean W/D/L score, resampling within strata.

    Draws remain observed score ``0.5`` rather than being treated as fractional
    Bernoulli successes. Seat and opponent strata preserve the scheduled field
    composition without claiming paired randomness when the engine is not
    seedable.
    """
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
    mcts_on_wins: float = 0.0
    mcts_on_games: int = 0
    greedy_wins: float = 0.0
    greedy_games: int = 0
    paired_scores: dict[str, list[float]] = field(default_factory=dict)
    formal_scores: list[tuple[int, float]] = field(default_factory=list)

    def record(
        self,
        *,
        our_seat: int,
        winner: int,
        is_mirror: bool,
        mcts_on: bool,
        pair_id: Optional[str] = None,
    ) -> None:
        if winner == 2:
            score = 0.5
        elif winner == our_seat:
            score = 1.0
        else:
            score = 0.0

        # Greedy is an ablation only. It must never enter formal MCTS Wilson,
        # seat, mirror, or pooled game counts.
        if not mcts_on:
            self.greedy_games += 1
            self.greedy_wins += score
            return

        self.games += 1
        if winner == 2:
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

        self.mcts_on_games += 1
        self.mcts_on_wins += score
        self.formal_scores.append((int(our_seat), score))
        if pair_id is not None:
            self.paired_scores.setdefault(str(pair_id), []).append(score)

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def wilson_lo(self) -> float:
        score = self.wins + 0.5 * self.draws
        return wilson_lower(score, self.games)

    @property
    def draw_aware_lower(self) -> float:
        strata: dict[str, list[float]] = {}
        for seat, score in self.formal_scores:
            strata.setdefault(f"seat{seat}", []).append(score)
        return stratified_bootstrap_interval(strata, seed=17)[1]

    def to_dict(self) -> dict[str, Any]:
        complete_pairs = [v for v in self.paired_scores.values() if len(v) == 2]
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
                "method": "seat_stratified_nonparametric_bootstrap",
                "center": interval[0],
                "lower": interval[1],
                "upper": interval[2],
            },
            "legacy_fractional_wilson_diagnostic": {
                "lower": self.wilson_lo,
                "formal_gate": False,
            },
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
            "greedy_games": self.greedy_games,
            "paired": {
                "complete_pairs": len(complete_pairs),
                "incomplete_pairs": sum(
                    1 for v in self.paired_scores.values() if len(v) != 2
                ),
                "mean_pair_score": (
                    sum(sum(v) / 2.0 for v in complete_pairs) / len(complete_pairs)
                    if complete_pairs
                    else None
                ),
            },
        }


@dataclass
class FieldReport:
    matchups: dict[str, MatchupStats] = field(default_factory=dict)
    gate_threshold: float = 0.55
    expected_opponents: Optional[set[str]] = None
    min_games_per_opponent: int = 1
    invalid_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.expected_opponents is not None:
            self.expected_opponents = set(self.expected_opponents)
            for opponent_id in sorted(self.expected_opponents):
                self.get(opponent_id)

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
        pair_id: Optional[str] = None,
    ) -> None:
        if (
            self.expected_opponents is not None
            and opponent_id not in self.expected_opponents
        ):
            self.mark_invalid(f"unexpected opponent result: {opponent_id}")
        self.get(opponent_id).record(
            our_seat=our_seat,
            winner=winner,
            is_mirror=is_mirror,
            mcts_on=mcts_on,
            pair_id=pair_id,
        )

    def mark_invalid(self, reason: str) -> None:
        if reason not in self.invalid_reasons:
            self.invalid_reasons.append(reason)

    def opponents_passing(self) -> list[str]:
        return [
            oid
            for oid, st in self.matchups.items()
            if st.games >= self.min_games_per_opponent
            and st.draw_aware_lower >= self.gate_threshold
        ]

    def opponents_failing(self) -> list[str]:
        return [
            oid
            for oid, st in self.matchups.items()
            if st.games > 0 and st.draw_aware_lower < self.gate_threshold
        ]

    def summary(self) -> dict[str, Any]:
        rows = [st.to_dict() for st in sorted(self.matchups.values(), key=lambda s: s.opponent_id)]
        n_pass = len(self.opponents_passing())
        n_eval = sum(1 for st in self.matchups.values() if st.games > 0)
        expected = (
            set(self.expected_opponents)
            if self.expected_opponents is not None
            else set(self.matchups)
        )
        missing = sorted(oid for oid in expected if self.get(oid).games == 0)
        insufficient = sorted(
            oid
            for oid in expected
            if 0 < self.get(oid).games < self.min_games_per_opponent
        )
        formal = [self.get(oid) for oid in sorted(expected)]
        pooled_games = sum(st.games for st in formal)
        pooled_score = sum(st.wins + 0.5 * st.draws for st in formal)
        pooled_center, pooled_lo, pooled_hi = stratified_bootstrap_interval(
            {
                f"{st.opponent_id}:seat{seat}": [
                    score
                    for row_seat, score in st.formal_scores
                    if row_seat == seat
                ]
                for st in formal
                for seat in (0, 1)
            },
            seed=23,
        )
        greedy_games = sum(st.greedy_games for st in formal)
        greedy_score = sum(st.greedy_wins for st in formal)
        complete_pairs = [
            scores
            for st in formal
            for scores in st.paired_scores.values()
            if len(scores) == 2
        ]
        field_valid = (
            bool(expected)
            and not self.invalid_reasons
            and not missing
            and not insufficient
        )
        all_pass = field_valid and set(self.opponents_passing()) == expected
        return {
            "valid": field_valid,
            "invalid_reasons": list(self.invalid_reasons),
            "n_expected": len(expected),
            "n_evaluated": n_eval,
            "n_passing_draw_aware": n_pass,
            "n_passing_wilson": n_pass,
            "gate_threshold": self.gate_threshold,
            "min_games_per_opponent": self.min_games_per_opponent,
            "missing_opponents": missing,
            "insufficient_opponents": insufficient,
            "all_pass": all_pass,
            "pooled_formal": {
                "games": pooled_games,
                "score": pooled_score,
                "wr": pooled_score / pooled_games if pooled_games else None,
                "uncertainty_method": "opponent_and_seat_stratified_nonparametric_bootstrap",
                "interval_center": pooled_center,
                "interval_lower": pooled_lo,
                "interval_upper": pooled_hi,
                "macro_opponent_wr": (
                    sum(st.win_rate for st in formal) / len(formal)
                    if formal
                    else None
                ),
            },
            # Compatibility alias. These values are draw-aware bootstrap
            # estimates, not Wilson/binomial intervals.
            "pooled_mcts": {
                "games": pooled_games,
                "score": pooled_score,
                "wr": pooled_score / pooled_games if pooled_games else None,
                "interval_center": pooled_center,
                "interval_lower": pooled_lo,
                "interval_upper": pooled_hi,
                "formal_gate": "draw_aware_bootstrap",
            },
            "paired_mcts": {
                "complete_pairs": len(complete_pairs),
                "mean_pair_score": (
                    sum(sum(v) / 2.0 for v in complete_pairs) / len(complete_pairs)
                    if complete_pairs
                    else None
                ),
            },
            "greedy_ablation": {
                "games": greedy_games,
                "score": greedy_score,
                "wr": greedy_score / greedy_games if greedy_games else None,
            },
            "matchups": rows,
        }
