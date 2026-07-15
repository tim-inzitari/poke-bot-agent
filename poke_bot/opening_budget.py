"""Opening-turn / clarity search-budget helpers for belief MCTS.

Keeps the trusted simulation floor intact while trimming optional wall-clock
and ramped sims on early, book-like turns, and enabling visit-gap STOP once
the floor is met (Baier/Winands early-stop; DS-MCTS-style certainty).
"""

from __future__ import annotations

from typing import Optional, Sequence

from . import config
from .heuristics_registry import prior_margin


def opening_budget_enabled() -> bool:
    return bool(getattr(config.SEARCH, "opening_budget", True))


def observation_turn(obs_dict: dict) -> Optional[int]:
    """Return current turn from a deployment observation, if present."""
    current = obs_dict.get("current") or {}
    turn = current.get("turn")
    if turn is None:
        return None
    try:
        return int(turn)
    except (TypeError, ValueError):
        return None


def is_opening_turn(turn: Optional[int]) -> bool:
    if turn is None:
        return False
    if not opening_budget_enabled():
        return False
    return int(turn) <= int(config.SEARCH.opening_turn_max)


def scale_opening_budgets(
    *,
    turn: Optional[int],
    requested_sims: int,
    move_time_s: float,
    min_trusted_sims: int,
) -> tuple[int, float, bool]:
    """Return (sims_cap, move_time, opening_applied).

    Sims never drop below ``min_trusted_sims``. Opening only shrinks when the
    requested budget is above the floor.
    """
    opening = is_opening_turn(turn)
    sims = max(int(min_trusted_sims), int(requested_sims))
    move_t = max(0.05, float(move_time_s))
    if not opening:
        return sims, move_t, False
    sims_mult = float(config.SEARCH.opening_sims_mult)
    time_mult = float(config.SEARCH.opening_move_time_mult)
    # Shrink only the optional portion above the trust floor.
    extra = max(0, sims - int(min_trusted_sims))
    sims = int(min_trusted_sims) + int(round(extra * sims_mult))
    sims = max(int(min_trusted_sims), sims)
    move_t = max(0.05, move_t * time_mult)
    return sims, move_t, True


def clarity_caps_to_floor(
    priors: Sequence[float],
    *,
    min_trusted_sims: int,
    current_plan: int,
) -> tuple[int, bool]:
    """If top−2nd prior margin is large, cap sims at the trust floor."""
    if not opening_budget_enabled():
        return int(current_plan), False
    margin = prior_margin(priors)
    thresh = float(config.SEARCH.clarity_prior_margin)
    if margin >= thresh:
        capped = max(int(min_trusted_sims), min(int(current_plan), int(min_trusted_sims)))
        # Cap at floor even midgame when prior is extremely sharp (book-like).
        return capped, True
    return int(current_plan), False


def visit_stop_triggered(
    visits: Sequence[int],
    *,
    sims_run: int,
    sims_plan: int,
    min_trusted_sims: int,
    elapsed_s: float,
    move_budget_s: float,
) -> bool:
    """True when 2nd-best cannot catch 1st given expected remaining sims."""
    if not opening_budget_enabled():
        return False
    if not bool(config.SEARCH.clarity_visit_stop):
        return False
    if int(sims_run) < int(min_trusted_sims):
        return False
    if len(visits) < 2:
        return True if len(visits) == 1 and sims_run >= min_trusted_sims else False
    ordered = sorted((int(v) for v in visits), reverse=True)
    gap = ordered[0] - ordered[1]
    remaining_quota = max(0, int(sims_plan) - int(sims_run))
    time_left = max(0.0, float(move_budget_s) - float(elapsed_s))
    rate = float(sims_run) / max(float(elapsed_s), 1e-9)
    expected_by_time = int(rate * time_left) if time_left > 0 else 0
    expected_remaining = min(remaining_quota, expected_by_time)
    p = max(0.0, min(1.0, float(config.SEARCH.clarity_visit_stop_p)))
    return gap > expected_remaining * p
