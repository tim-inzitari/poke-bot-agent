"""AlphaZero visit-count policy targets from MCTS trees.

The improved policy used for training is the normalised visit distribution at
the search root (optionally temperature-sharpened). Leaf values and raw network
priors are recorded for diagnostics / value blending, but the primary policy
target is visit counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class SearchTarget:
    """One decision's MCTS-improved training target."""

    #: Option-index combos scored at the root (same order as policy vectors).
    action_combos: list[list[int]]
    #: Visit counts per child (0 if never expanded).
    visits: list[int]
    #: Normalised visit policy (sums to 1 over children with visits, else uniform).
    policy: list[float]
    #: Root mean value from the acting seat's perspective in [-1, 1].
    value: float
    #: Network prior at root expansion (optional).
    prior: Optional[list[float]] = None
    #: Diagnostics blob (sims run, time used, …).
    diagnostics: dict = field(default_factory=dict)

    @property
    def total_visits(self) -> int:
        return int(sum(self.visits))


def visit_count_policy(
    visits: Sequence[int],
    *,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> list[float]:
    """Convert visit counts to a probability distribution.

    ``temperature`` < 1 sharpens toward the argmax; ``temperature`` → 0 is
    handled as a hard one-hot on the most-visited child.
    """
    n = len(visits)
    if n == 0:
        return []
    total = float(sum(visits))
    if total <= 0:
        return [1.0 / n] * n

    if temperature <= 0:
        best = max(range(n), key=lambda i: visits[i])
        out = [0.0] * n
        out[best] = 1.0
        return out

    if abs(temperature - 1.0) < 1e-9:
        return [v / total for v in visits]

    # Softmax over (visit^(1/T)).
    scaled = [max(v, 0) ** (1.0 / temperature) for v in visits]
    s = sum(scaled) + eps
    return [x / s for x in scaled]


def build_search_target(
    action_combos: list[list[int]],
    visits: Sequence[int],
    value: float,
    *,
    prior: Optional[Sequence[float]] = None,
    temperature: float = 1.0,
    diagnostics: Optional[dict] = None,
) -> SearchTarget:
    """Build a :class:`SearchTarget` from root visit statistics."""
    if len(visits) != len(action_combos):
        raise ValueError(
            f"visits ({len(visits)}) must match action_combos ({len(action_combos)})"
        )
    policy = visit_count_policy(visits, temperature=temperature)
    return SearchTarget(
        action_combos=[list(c) for c in action_combos],
        visits=[int(v) for v in visits],
        policy=policy,
        value=float(value),
        prior=list(prior) if prior is not None else None,
        diagnostics=dict(diagnostics or {}),
    )


def select_by_visits(
    action_combos: list[list[int]],
    visits: Sequence[int],
) -> list[int]:
    """Greedy action = most-visited child (ties → lowest index)."""
    if not action_combos:
        return []
    best = max(range(len(action_combos)), key=lambda i: (visits[i], -i))
    return list(action_combos[best])
