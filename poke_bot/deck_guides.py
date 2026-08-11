"""Specialist-agnostic registry for training-only current-deck guides.

Guide scorers are sparse teachers over the existing legal-action policy
logits. They are never imported by serving and never choose an action.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from types import ModuleType

from . import (
    alakazam_heuristics,
    alakazam_new_list_heuristics,
    archaludon_ex_heuristics,
    garchomp_heuristics,
    grimmsnarl_heuristics,
    hammer_heuristics,
    rockets_mewtwo_heuristics,
    slowking_heuristics,
    teal_mask_ogerpon_heuristics,
    team_rockets_spidops_heuristics,
    thwackey_heuristics,
)

_GUIDES: dict[str, ModuleType] = {
    "alakazam": alakazam_heuristics,
    "archaludon-ex": archaludon_ex_heuristics,
    "garchomp": garchomp_heuristics,
    "hammer-pult": hammer_heuristics,
    "marnie-s-grimmsnarl-ex": grimmsnarl_heuristics,
    "rockets-mewtwo": rockets_mewtwo_heuristics,
    "slowking": slowking_heuristics,
    "team-rockets-spidops": team_rockets_spidops_heuristics,
    "teal-mask-ogerpon-ex": teal_mask_ogerpon_heuristics,
    "thwackey": thwackey_heuristics,
}

_ALAKAZAM_GUIDE_VERSIONS: dict[str, ModuleType] = {
    str(alakazam_heuristics.GUIDE_VERSION): alakazam_heuristics,
    str(alakazam_new_list_heuristics.GUIDE_VERSION): (
        alakazam_new_list_heuristics
    ),
}


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def selected_id() -> str | None:
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
    module = _selected_module(selected)
    # An explicit selector is itself fail-closed authorization. The additional
    # boolean is useful to pre-stage a selector while target generation is off.
    return _truthy("POKEBOT_CURRENT_DECK_GUIDE_TARGETS") or (
        selected == "alakazam" and module.enabled()
    )


def _selected_module(selected: str | None = None) -> ModuleType:
    selected = selected if selected is not None else selected_id()
    if selected is None or selected not in _GUIDES:
        raise RuntimeError(f"unknown current-deck guide: {selected!r}")
    if selected != "alakazam":
        return _GUIDES[selected]
    requested = os.environ.get(
        "POKEBOT_CURRENT_DECK_GUIDE_VERSION",
        str(alakazam_heuristics.GUIDE_VERSION),
    ).strip()
    module = _ALAKAZAM_GUIDE_VERSIONS.get(requested)
    if module is None:
        raise RuntimeError(f"unknown Alakazam guide version: {requested!r}")
    return module


def guide_version() -> str | None:
    selected = selected_id()
    return None if selected is None else str(_selected_module(selected).GUIDE_VERSION)


def supported_ids() -> tuple[str, ...]:
    return tuple(sorted(_GUIDES))


def guide_scores(
    obs: dict,
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
) -> list[float] | None:
    """Dispatch a sparse score request to the selected specialist guide."""
    if not enabled():
        return None
    selected = selected_id()
    assert selected is not None
    module = _selected_module(selected)
    if module is alakazam_heuristics and alakazam_heuristics.enabled():
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
