"""Archetype registry and deck classification.

Critical rule (from the plan): **Hammer Dragapult is NOT pure Dragapult.** The
four Dragapult variants and Hammer-Pult are treated as *separate* archetypes for
pools, filters, eval matrices, and checkpoints.

``hammer-pult`` is defined by a **card signature** (4x Crushing Hammer + Munkidori
+ Budew + Unfair Stamp + the Dragapult line), not by deck slug. A
``dragapult-dudunsparce`` list that matches the Hammer signature classifies as
``hammer-pult``.

Two classifiers are exposed:
  - :func:`classify_deck` — authoritative, uses the actual 60-card id multiset.
  - :func:`classify_slug` — filename/slug heuristic (cannot detect hammer-pult).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Signature card ids (verified against cards/EN_Card_Data.csv)
# ---------------------------------------------------------------------------

CRUSHING_HAMMER = 1120
UNFAIR_STAMP = 1080
BUDEW = 235
MUNKIDORI = {112, 139}  # Munkidori / Munkidori ex

DRAGAPULT_LINE = {119, 120, 121}          # Dreepy / Drakloak / Dragapult ex
DUDUNSPARCE_LINE = {65, 66, 305, 306}     # Dunsparce / Dudunsparce (+ ex variants)
DUSKNOIR_LINE = {131, 132, 133}           # Duskull / Dusclops / Dusknoir
BLAZIKEN_LINE = {324, 325, 326, 410, 411, 412}  # Torchic / Combusken / Blaziken(+ex)

#: Minimum Crushing Hammers for the Hammer-Pult signature (list runs 3-4).
CRUSHING_HAMMER_MIN = 3

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Archetype:
    """A registered archetype/deck family."""

    id: str
    name: str
    description: str
    #: All substrings that must appear in a slug for the slug heuristic to match.
    slug_all: tuple[str, ...] = ()
    #: If True, classification is card-signature driven (not slug driven).
    signature_only: bool = False


# Ordered registry. Order matters for slug matching (more specific first).
ARCHETYPES: dict[str, Archetype] = {}


def register(arch: Archetype) -> Archetype:
    """Register (or replace) an archetype in the global registry."""
    ARCHETYPES[arch.id] = arch
    return arch


register(Archetype(
    id="hammer-pult",
    name="Hammer Pult",
    description=(
        "Dragapult disruption build: 4x Crushing Hammer + Munkidori + Budew + "
        "Unfair Stamp. Separate Phase 6+ archetype (not v1 primary). "
        "Signature-defined, not slug."
    ),
    signature_only=True,
))
register(Archetype(
    id="dragapult-dusknoir",
    name="Dragapult Dusknoir",
    description="Dragapult with the Duskull/Dusclops/Dusknoir damage-move tech line.",
    slug_all=("dragapult", "dusknoir"),
))
register(Archetype(
    id="dragapult-blaziken",
    name="Dragapult Blaziken",
    description="Dragapult with the Torchic/Combusken/Blaziken tech line.",
    slug_all=("dragapult", "blaziken"),
))
register(Archetype(
    id="dragapult-dudunsparce",
    name="Dragapult Dudunsparce",
    description="Dragapult with the Dunsparce/Dudunsparce consistency line.",
    slug_all=("dragapult", "dudunsparce"),
))
register(Archetype(
    id="dragapult",
    name="Dragapult",
    description=(
        "Generic/pure Dragapult ex without Hammer-Pult signature or a "
        "distinguishing tech line. Primary v1 submission / bootstrap target."
    ),
    slug_all=("dragapult",),
))


def _counts(card_ids: Iterable[int]) -> Counter:
    return Counter(int(c) for c in card_ids)


def has_line(counts: Counter, line: set[int]) -> bool:
    return any(counts.get(cid, 0) > 0 for cid in line)


def is_hammer_signature(card_ids: Iterable[int]) -> bool:
    """True if the 60-card list matches the Hammer-Pult card signature."""
    c = _counts(card_ids)
    return (
        c.get(CRUSHING_HAMMER, 0) >= CRUSHING_HAMMER_MIN
        and has_line(c, MUNKIDORI)
        and c.get(BUDEW, 0) > 0
        and c.get(UNFAIR_STAMP, 0) > 0
        and has_line(c, DRAGAPULT_LINE)
    )


def classify_deck(card_ids: Iterable[int]) -> str:
    """Classify a deck from its card-id multiset. Authoritative.

    Precedence: Hammer signature first (steals matching Dudunsparce lists), then
    Dragapult tech variants (Dusknoir > Blaziken > Dudunsparce), then generic
    Dragapult, else ``"unknown"``.
    """
    card_ids = list(card_ids)
    if is_hammer_signature(card_ids):
        return "hammer-pult"

    c = _counts(card_ids)
    if has_line(c, DRAGAPULT_LINE):
        if has_line(c, DUSKNOIR_LINE):
            return "dragapult-dusknoir"
        if has_line(c, BLAZIKEN_LINE):
            return "dragapult-blaziken"
        if has_line(c, DUDUNSPARCE_LINE):
            return "dragapult-dudunsparce"
        return "dragapult"
    return UNKNOWN


def classify_slug(slug: str) -> str:
    """Heuristic classification from a deck slug/filename.

    Cannot return ``hammer-pult`` (that requires the card signature). Falls back
    to ``"unknown"`` when no registered slug pattern matches.
    """
    s = slug.lower()
    for arch in ARCHETYPES.values():
        if arch.signature_only or not arch.slug_all:
            continue
        if all(token in s for token in arch.slug_all):
            return arch.id
    return UNKNOWN


def get_archetype(archetype_id: str) -> Optional[Archetype]:
    """Return the registered :class:`Archetype`, or None if unknown."""
    return ARCHETYPES.get(archetype_id)


def archetype_ids() -> list[str]:
    """All registered archetype ids (registration order)."""
    return list(ARCHETYPES.keys())
