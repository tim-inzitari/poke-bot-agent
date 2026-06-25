from __future__ import annotations

import numpy as np

from poke_agent.search_targets import (
    build_search_diagnostics,
    kl_divergence,
    search_distribution_from_ranked,
)


def test_search_distribution_from_ranked_is_normalized():
    ranked = [(1.0, [0]), (0.5, [1]), (0.0, [2])]
    dist = search_distribution_from_ranked(ranked, policy_dim=64)
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert len(dist) == 3


def test_build_search_diagnostics_prefers_best_action():
    logits = np.zeros(64, dtype=np.float32)
    actions = [[0], [1]]
    ranked = [(0.9, [1]), (0.1, [0])]
    diag = build_search_diagnostics(
        logits=logits,
        actions=actions,
        ranked=ranked,
        best_action=[1],
        best_value=0.9,
        policy_dim=64,
    )
    assert diag.search_value == 0.9
    assert diag.search_transition_class is not None
    assert diag.search_policy_kl >= 0.0


def test_kl_divergence_is_non_negative():
    target = {"a": 0.7, "b": 0.3}
    approx = {"a": 0.5, "b": 0.5}
    assert kl_divergence(target, approx) >= 0.0
