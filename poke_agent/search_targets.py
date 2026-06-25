from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from poke_agent.features import stable_hash_index


@dataclass
class SearchDiagnostics:
    """Training/inference metadata from one beam search decision."""

    search_value: float = 0.0
    search_policy_kl: float = 0.0
    search_transition_class: int | None = None
    search_policy_target: dict[str, float] = field(default_factory=dict)
    ranked_actions: list[list[int]] = field(default_factory=list)

    def to_row_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "search_value": float(self.search_value),
            "search_policy_kl": float(self.search_policy_kl),
        }
        if self.search_transition_class is not None:
            fields["search_transition_class"] = int(self.search_transition_class)
        if self.search_policy_target:
            fields["search_policy_target"] = dict(self.search_policy_target)
        return fields


def action_key(action: list[int]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def action_class(action: list[int], policy_dim: int) -> int:
    return stable_hash_index(action_key(action), policy_dim)


def policy_distribution_over_actions(
    logits: np.ndarray,
    actions: list[list[int]],
    *,
    policy_dim: int,
    temperature: float = 1.0,
) -> dict[str, float]:
    if not actions:
        return {}
    scores = np.array(
        [float(logits[action_class(action, policy_dim)]) for action in actions],
        dtype=np.float64,
    )
    scores = scores / max(temperature, 1e-6)
    scores -= scores.max()
    probs = np.exp(scores)
    total = probs.sum()
    if total <= 0:
        uniform = 1.0 / len(actions)
        return {action_key(action): uniform for action in actions}
    probs /= total
    return {action_key(actions[index]): float(probs[index]) for index in range(len(actions))}


def search_distribution_from_ranked(
    ranked: list[tuple[float, list[int]]],
    *,
    policy_dim: int,
    temperature: float = 0.5,
) -> dict[str, float]:
    del policy_dim  # action keys are stored directly; class mapping happens at train time.
    if not ranked:
        return {}
    values = np.array([float(score) for score, _ in ranked], dtype=np.float64)
    values = values / max(temperature, 1e-6)
    values -= values.max()
    probs = np.exp(values)
    total = probs.sum()
    if total <= 0:
        uniform = 1.0 / len(ranked)
        return {action_key(action): uniform for _, action in ranked}
    probs /= total
    return {action_key(action): float(probs[index]) for index, (_, action) in enumerate(ranked)}


def kl_divergence(target: dict[str, float], approximate: dict[str, float]) -> float:
    keys = set(target) | set(approximate)
    if not keys:
        return 0.0
    kl = 0.0
    for key in keys:
        p = float(target.get(key, 0.0))
        q = float(approximate.get(key, 1e-8))
        if p > 0:
            kl += p * math.log(max(p, 1e-8) / max(q, 1e-8))
    return float(max(0.0, kl))


def build_search_diagnostics(
    *,
    logits: np.ndarray,
    actions: list[list[int]],
    ranked: list[tuple[float, list[int]]],
    best_action: list[int],
    best_value: float,
    policy_dim: int,
) -> SearchDiagnostics:
    search_target = search_distribution_from_ranked(ranked, policy_dim=policy_dim)
    policy_target = policy_distribution_over_actions(logits, actions, policy_dim=policy_dim)
    kl = kl_divergence(search_target, policy_target)
    return SearchDiagnostics(
        search_value=float(best_value),
        search_policy_kl=kl,
        search_transition_class=action_class(best_action, policy_dim),
        search_policy_target=search_target,
        ranked_actions=[action for _, action in ranked],
    )
