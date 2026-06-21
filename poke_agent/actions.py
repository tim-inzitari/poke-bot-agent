from __future__ import annotations

import itertools


def legal_actions(option_count: int, min_count: int, max_count: int) -> list[list[int]]:
    """Enumerate legal CABT action index lists (combinations / permutations)."""
    actions: list[list[int]] = []
    for count in range(min_count, max_count + 1):
        for combo in itertools.combinations(range(option_count), count):
            if count <= 1:
                actions.append(list(combo))
            else:
                actions.extend(list(perm) for perm in itertools.permutations(combo))
    return actions
