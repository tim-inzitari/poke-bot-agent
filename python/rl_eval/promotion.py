"""Immutable checkpoint identities and draw-aware candidate promotion gates."""

from __future__ import annotations

import hashlib
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .metrics import stratified_bootstrap_interval


@dataclass(frozen=True)
class CheckpointIdentity:
    path: str
    digest: str

    @classmethod
    def from_path(cls, path: str | Path, algorithm: str = "sha256") -> "CheckpointIdentity":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
        h = hashlib.new(algorithm)
        with resolved.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return cls(path=str(resolved), digest=f"{algorithm}:{h.hexdigest()}")

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionGateConfig:
    min_games: int = 40
    min_complete_pairs: int = 20
    threshold: float = 0.5
    confidence: float = 0.90
    bootstrap_resamples: int = 4000

    def validate(self) -> None:
        if self.min_games < 2 or self.min_games % 2:
            raise ValueError("promotion min_games must be even and >= 2")
        if self.min_complete_pairs < 1:
            raise ValueError("promotion minimum games per seat must be >= 1")
        if self.min_complete_pairs * 2 > self.min_games:
            raise ValueError("promotion per-seat minimum exceeds game minimum")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("promotion threshold must be in [0, 1]")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("promotion confidence must be in (0, 1)")
        if self.bootstrap_resamples < 100:
            raise ValueError("promotion bootstrap_resamples must be >= 100")


def next_iteration(
    loop_state: dict[str, Any], checkpoint_iteration: Optional[int] = None
) -> int:
    completed: list[int] = []
    explicit = loop_state.get("last_completed_iteration")
    if explicit is not None:
        completed.append(int(explicit))
    for row in loop_state.get("history") or []:
        if row.get("iteration") is not None and row.get("completed", True):
            completed.append(int(row["iteration"]))
    if checkpoint_iteration is not None and int(checkpoint_iteration) >= 0:
        completed.append(int(checkpoint_iteration))
    return max(completed, default=-1) + 1


def evaluate_candidate_gate(
    results: Iterable[dict[str, Any]],
    config: PromotionGateConfig,
) -> dict[str, Any]:
    config.validate()
    rows = list(results)
    failures = [
        str(r.get("error") or "invalid promotion game")
        for r in rows
        if not r.get("valid", True)
    ]
    strata: dict[str, list[float]] = {"seat0": [], "seat1": []}
    clusters: dict[str, list[tuple[int, float]]] = {}
    wins = draws = losses = 0
    for row in rows:
        if not row.get("valid", True):
            continue
        candidate_seat = int(row["candidate_seat"])
        winner = int(row["winner"])
        if winner == 2:
            score = 0.5
            draws += 1
        elif winner == candidate_seat:
            score = 1.0
            wins += 1
        else:
            score = 0.0
            losses += 1
        strata[f"seat{candidate_seat}"].append(score)
        if row.get("cluster_id") is not None:
            clusters.setdefault(str(row["cluster_id"]), []).append(
                (candidate_seat, score)
            )

    valid_games = wins + draws + losses
    score_total = wins + 0.5 * draws
    center, lower, upper = stratified_bootstrap_interval(
        strata,
        confidence=config.confidence,
        resamples=config.bootstrap_resamples,
        seed=101,
    )
    uncertainty_method = "candidate_seat_stratified_nonparametric_bootstrap"
    complete_clusters = [
        values
        for values in clusters.values()
        if {seat for seat, _score in values} == {0, 1}
    ]
    if len(complete_clusters) >= 2:
        rng = random.Random(101)
        samples: list[float] = []
        for _ in range(config.bootstrap_resamples):
            drawn = [
                complete_clusters[rng.randrange(len(complete_clusters))]
                for _ in complete_clusters
            ]
            by_seat = {
                seat: [
                    score
                    for cluster in drawn
                    for row_seat, score in cluster
                    if row_seat == seat
                ]
                for seat in (0, 1)
            }
            samples.append(
                statistics.fmean(statistics.fmean(by_seat[seat]) for seat in (0, 1))
            )
        samples.sort()
        alpha = (1.0 - config.confidence) / 2.0

        def _quantile(q: float) -> float:
            position = q * (len(samples) - 1)
            lower_index = int(position)
            upper_index = min(lower_index + 1, len(samples) - 1)
            weight = position - lower_index
            return (
                samples[lower_index] * (1.0 - weight) + samples[upper_index] * weight
            )

        center = statistics.fmean(statistics.fmean(strata[key]) for key in ("seat0", "seat1"))
        lower = _quantile(alpha)
        upper = _quantile(1.0 - alpha)
        uncertainty_method = "balanced_seat_cluster_nonparametric_bootstrap"
    sample_ok = valid_games >= config.min_games and all(
        len(values) >= config.min_complete_pairs for values in strata.values()
    )
    passed = not failures and sample_ok and lower > config.threshold
    return {
        "passed": passed,
        "valid": not failures and sample_ok,
        "failures": failures,
        "games": valid_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score_total,
        "wr": score_total / valid_games if valid_games else None,
        "uncertainty_method": uncertainty_method,
        "interval_center": center,
        "interval_lower": lower,
        "interval_upper": upper,
        "games_per_seat": {key: len(values) for key, values in strata.items()},
        "cluster_count": len(complete_clusters),
        "config": asdict(config),
    }
