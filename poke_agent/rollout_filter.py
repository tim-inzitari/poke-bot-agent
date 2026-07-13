from __future__ import annotations

from typing import Any

from poke_agent.archetypes import HEURISTIC_ARCHETYPE_SLUG_PATTERNS
from poke_agent.deck_pool import deck_matches_archetype_patterns


def episode_involves_archetype(
    episode_rows: list[dict[str, Any]],
    archetype: str,
    *,
    registry: Any | None = None,
) -> bool:
    """True when either seat's deck slug belongs to the heuristic archetype family."""
    if not episode_rows:
        return False
    patterns = HEURISTIC_ARCHETYPE_SLUG_PATTERNS.get(archetype)
    if not patterns:
        return True

    def seat_matches(slug: str, cards: Any) -> bool:
        if deck_matches_archetype_patterns(slug, patterns):
            return True
        if any(f"_{pattern}" in slug or f"-{pattern}" in slug for pattern in patterns):
            return True
        if cards and registry is not None:
            classified, _ = registry.classify_deck(cards)
            return deck_matches_archetype_patterns(classified, patterns)
        return False

    if registry is None:
        from poke_agent.archetypes import load_archetype_registry
        from poke_agent.paths import resolve_root

        registry = load_archetype_registry(resolve_root())

    for row in episode_rows:
        deck0 = str(row.get("deck0") or "")
        deck1 = str(row.get("deck1") or "")
        if seat_matches(deck0, row.get("deck0_cards")) or seat_matches(deck1, row.get("deck1_cards")):
            return True
    return False
