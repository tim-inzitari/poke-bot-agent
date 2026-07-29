"""Specialist-agnostic registry for training-only current-deck guides.

Guide scorers are sparse teachers over the existing legal-action policy
logits. They are never imported by serving and never choose an action.
"""

from __future__ import annotations

import os
from types import ModuleType
from typing import Iterable, Optional, Sequence

from . import (
    alakazam_heuristics,
    archaludon_ex_heuristics,
    garchomp_heuristics,
    grimmsnarl_heuristics,
    hammer_heuristics,
    rockets_mewtwo_heuristics,
    team_rockets_spidops_heuristics,
    teal_mask_ogerpon_heuristics,
    thwackey_heuristics,
)


_GUIDES: dict[str, ModuleType] = {
    "alakazam": alakazam_heuristics,
    "archaludon-ex": archaludon_ex_heuristics,
    "garchomp": garchomp_heuristics,
    "hammer-pult": hammer_heuristics,
    "marnie-s-grimmsnarl-ex": grimmsnarl_heuristics,
    "rockets-mewtwo": rockets_mewtwo_heuristics,
    "team-rockets-spidops": team_rockets_spidops_heuristics,
    "teal-mask-ogerpon-ex": teal_mask_ogerpon_heuristics,
    "thwackey": thwackey_heuristics,
}


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def selected_id() -> Optional[str]:
    """Return the explicitly selected guide, preserving Alakazam compatibility."""
    selected = os.environ.get("POKEBOT_CURRENT_DECK_GUIDE", "").strip().lower()
    if selected:
        return selected
    if alakazam_heuristics.enabled():
        return "alakazam"
    return None


def enabled() -> bool:
    selected = selected_id()
    if selected is None:
        return False
    if selected not in _GUIDES:
        raise RuntimeError(f"unknown current-deck guide: {selected!r}")
    # An explicit selector is itself fail-closed authorization. The additional
    # boolean is useful to pre-stage a selector while target generation is off.
    return _truthy("POKEBOT_CURRENT_DECK_GUIDE_TARGETS") or (
        selected == "alakazam" and alakazam_heuristics.enabled()
    )


def guide_version() -> Optional[str]:
    selected = selected_id()
    module = _GUIDES.get(selected or "")
    return None if module is None else str(module.GUIDE_VERSION)


def supported_ids() -> tuple[str, ...]:
    return tuple(sorted(_GUIDES))


def guide_scores(
    obs: dict,
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
) -> Optional[list[float]]:
    """Dispatch a sparse score request to the selected specialist guide."""
    if not enabled():
        return None
    selected = selected_id()
    assert selected is not None
    module = _GUIDES[selected]
    if selected == "alakazam" and alakazam_heuristics.enabled():
        # Preserve the original callable contract for old feature workers and
        # tests which monkeypatch the validated Alakazam teacher directly.
        return module.guide_scores(obs, action_combos, deck=deck)
    return module.guide_scores(obs, action_combos, deck=deck, force_enabled=True)


__all__ = [
    "enabled",
    "guide_scores",
    "guide_version",
    "selected_id",
    "supported_ids",
]
