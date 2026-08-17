"""Decision-budget collection and game-first replay contracts."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, replace
from typing import Optional, Sequence, TypeVar

from .dataset import GameSequence

T = TypeVar("T")

# Native competition-engine pilot, 2026-07-14:
# 52 balanced games covering all 26 required baseline opponents at 128
# completed simulations/decision.
PILOT_DERIVED_TARGET_SEARCH_DECISIONS = 3_664
PILOT_DERIVATION = {
    "run": "blackwell_full_roster_pilot_20260714T2015Z",
    "games": 52,
    "opponents": 26,
    "balanced_seats": True,
    "completed_sims_per_decision": 128,
    "mean_decisions_per_game": 8.807692307692308,
    "median_decisions_per_game": 6.0,
    "min_decisions_per_game": 2,
    "max_decisions_per_game": 41,
    "method": "mean_decisions_per_game_x_26_opponents_x_16_games",
    "target_search_decisions": PILOT_DERIVED_TARGET_SEARCH_DECISIONS,
}


@dataclass(frozen=True)
class SearchCollectionContract:
    opponents: int
    nominal_games_per_opponent: int = 16
    min_games_per_opponent: int = 12
    max_games_per_opponent: int = 24
    target_search_decisions: int = 0

    def __post_init__(self) -> None:
        if self.opponents < 1:
            raise ValueError("collection requires at least one opponent")
        values = (
            self.min_games_per_opponent,
            self.nominal_games_per_opponent,
            self.max_games_per_opponent,
        )
        if any(value < 2 or value % 2 for value in values):
            raise ValueError("game safeguards must be even and >=2")
        if not (
            self.min_games_per_opponent
            <= self.nominal_games_per_opponent
            <= self.max_games_per_opponent
        ):
            raise ValueError("expected min <= nominal <= max games")
        if self.target_search_decisions < 0:
            raise ValueError("target search decisions must be non-negative")

    def stop_reason(
        self,
        *,
        games_per_opponent: int,
        trusted_decisions: int,
        coverage_complete: bool,
    ) -> Optional[str]:
        if games_per_opponent < self.min_games_per_opponent:
            return None
        if not coverage_complete:
            return None
        if (
            self.target_search_decisions > 0
            and trusted_decisions >= self.target_search_decisions
        ):
            return "target_search_decisions"
        if (
            self.target_search_decisions == 0
            and games_per_opponent >= self.nominal_games_per_opponent
        ):
            return "nominal_games"
        if games_per_opponent >= self.max_games_per_opponent:
            return "max_games"
        return None

    @property
    def min_games(self) -> int:
        return self.opponents * self.min_games_per_opponent

    @property
    def nominal_games(self) -> int:
        return self.opponents * self.nominal_games_per_opponent

    @property
    def max_games(self) -> int:
        return self.opponents * self.max_games_per_opponent


@dataclass(frozen=True)
class IterationSampleAccounting:
    fresh_games: int
    trusted_search_decisions: int
    promotion_games: int

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            "collection": {
                "fresh_games": int(self.fresh_games),
                "trusted_search_decisions": int(self.trusted_search_decisions),
            },
            "promotion": {
                "games": int(self.promotion_games),
                "included_in_training": 0,
            },
        }


def derive_target_search_decisions(
    decisions_per_game: Sequence[int | float],
    *,
    opponents: int = 26,
    nominal_games_per_opponent: int = 16,
) -> dict[str, float | int | str]:
    """Derive the nominal decision target from a measured native pilot.

    Total decisions are additive, so the arithmetic mean is the unbiased
    estimator for the nominal aggregate budget. The median is retained in the
    report as a skew diagnostic; no hidden fallback exists.
    """
    samples = [float(value) for value in decisions_per_game if float(value) > 0]
    if len(samples) < 8:
        raise ValueError("decision target requires at least eight valid pilot games")
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    target = int(round(mean * opponents * nominal_games_per_opponent))
    return {
        "method": "native_pilot_mean_decisions_per_game_x_nominal_games",
        "pilot_games": len(samples),
        "mean_decisions_per_game": mean,
        "median_decisions_per_game": median,
        "min_decisions_per_game": min(samples),
        "max_decisions_per_game": max(samples),
        "opponents": int(opponents),
        "nominal_games_per_opponent": int(nominal_games_per_opponent),
        "target_search_decisions": target,
    }


def history_games_for_fraction(fresh_games: int, replay_fraction: float) -> int:
    if fresh_games < 0:
        raise ValueError("fresh games must be non-negative")
    if not 0.5 <= replay_fraction <= 0.75:
        raise ValueError("replay fraction must be in [0.50, 0.75]")
    if fresh_games == 0:
        return 0
    return int(round(fresh_games * replay_fraction / (1.0 - replay_fraction)))


def game_first_sample(
    games: Sequence[T],
    *,
    count: int,
    rng: random.Random,
) -> list[T]:
    """Uniformly sample games, never decisions, without replacement."""
    rows = list(games)
    rng.shuffle(rows)
    return rows[: max(0, min(int(count), len(rows)))]


def cap_game_decisions(
    sequence: GameSequence,
    *,
    max_decisions: int,
    rng: random.Random,
) -> GameSequence:
    """Select one chronological window so each sampled game has bounded weight."""
    cap = int(max_decisions)
    if cap <= 0 or len(sequence.decisions) <= cap:
        return sequence
    start = rng.randint(0, len(sequence.decisions) - cap)
    end = start + cap
    policy_targets = (
        sequence.policy_targets[start:end]
        if sequence.policy_targets is not None
        else None
    )
    factorized = (
        sequence.factorized_policy_targets[start:end]
        if sequence.factorized_policy_targets is not None
        else None
    )
    provenance = {
        **sequence.target_provenance,
        "game_first_decision_window": {
            "original_decisions": len(sequence.decisions),
            "start": start,
            "count": cap,
        },
    }
    return replace(
        sequence,
        decisions=list(sequence.decisions[start:end]),
        policy_targets=policy_targets,
        factorized_policy_targets=factorized,
        target_provenance=provenance,
    )


def balanced_seat_coverage(
    rows: Sequence[tuple[str, int]],
    *,
    opponents: Sequence[str],
    min_games_per_opponent: int,
) -> tuple[bool, dict[str, dict[int, int]]]:
    if min_games_per_opponent % 2:
        raise ValueError("balanced minimum must be even")
    counts = {
        opponent: {0: 0, 1: 0}
        for opponent in opponents
    }
    for opponent, seat in rows:
        if opponent in counts and seat in (0, 1):
            counts[opponent][int(seat)] += 1
    required = min_games_per_opponent // 2
    complete = all(
        seat_counts[0] >= required and seat_counts[1] >= required
        for seat_counts in counts.values()
    )
    return complete, counts

