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
MARNIE_GRIMMSNARL_LINE = {646, 647, 648}
TEAM_ROCKETS_TAROUNTULA = 400
TEAM_ROCKETS_SPIDOPS = 401
TEAM_ROCKETS_MEWTWO_EX = 431
THWACKEY_GROOKEY = 89
THWACKEY = 90
THWACKEY_DIPPLIN = 93
FESTIVAL_GROUNDS = 1245
TEAL_MASK_OGERPON_EX = 96
RAGING_BOLT_EX = 63
WELLSPRING_MASK_OGERPON_EX = 108
LILLIES_CLEFAIRY_EX = 272
MEGA_KANGASKHAN_EX = 756
MEOWTH_EX = 1071
ENERGY_SWITCH = 1116
AREA_ZERO_UNDERDEPTHS = 1250
SLOP_BOX_KANGASKHAN_COUNT = 3

# Exact top-ladder modal representative.  Full multiset equality is deliberate:
# Archaludon shares generic Metal engines with other decks, so a loose marker
# signature would not be collision-safe.
ARCHALUDON_EX_MODAL_REPRESENTATIVE: tuple[int, ...] = (
    190, 190, 190, 190,
    169, 169, 169, 169,
    57, 57,
    414,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    1159,
    1097, 1097, 1097, 1097,
    1121, 1121, 1121, 1121,
    1122, 1122, 1122, 1122,
    1152, 1152, 1152, 1152,
    1182, 1182, 1182, 1182,
    1192, 1192, 1192, 1192,
    1227, 1227, 1227, 1227,
    1213, 1213,
    1197,
    1244, 1244, 1244, 1244,
)
_ARCHALUDON_EX_MODAL_MULTISET = tuple(
    sorted(ARCHALUDON_EX_MODAL_REPRESENTATIVE)
)

# Exact current Crustle specialist representative. Full multiset equality is
# deliberate: Mega Kangaskhan ex is shared with Slop Box and generic marker
# signatures would steal unrelated decks.
CRUSTLE_NAIC_2026_REPRESENTATIVE: tuple[int, ...] = (
    756, 756, 756, 756,
    344, 344, 344,
    345, 345, 345,
    858,
    1227, 1227, 1227, 1227,
    1219, 1219, 1219, 1219,
    1182, 1182, 1182, 1182,
    1204, 1204,
    1225, 1225,
    1197,
    1120, 1120, 1120, 1120,
    1147, 1147, 1147, 1147,
    1122, 1122, 1122,
    1086, 1086,
    1121,
    1123,
    1112,
    1159,
    1245,
    1257,
    1267,
    11, 11, 11, 11,
    14, 14, 14, 14,
    18, 18, 18, 18,
)
_CRUSTLE_NAIC_2026_MULTISET = tuple(
    sorted(CRUSTLE_NAIC_2026_REPRESENTATIVE)
)

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

# Preserve the row order used by every checkpoint created before the complete
# ladder roster was registered.  Warm-start expansion copies these five rows
# by name and moves the old final ``unknown`` row to the new final row.
LEGACY_AUX_ARCHETYPE_IDS: tuple[str, ...] = tuple(ARCHETYPES)

# Deck-agnostic core roster pinned by data/training_mixes/top_ladder.v1.json.
# The three Dragapult-family entries above are already present; these entries
# make every remaining ladder ID a real supervised auxiliary class instead of
# silently masking 14/17 decks as unknown.
_CORE_LADDER_GENERIC: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("alakazam", "Alakazam", ("alakazam",)),
    ("crustle", "Crustle", ("crustle",)),
    (
        "marnie-s-grimmsnarl-ex",
        "Marnie's Grimmsnarl ex",
        ("marnie", "grimmsnarl"),
    ),
    ("garchomp", "Garchomp", ("garchomp",)),
    ("cornerstone-ogerpon", "Cornerstone Ogerpon", ("cornerstone", "ogerpon")),
    ("rockets-mewtwo", "Rocket's Mewtwo", ("rocket", "mewtwo")),
    ("starmie", "Starmie", ("starmie",)),
    ("archaludon-ex", "Archaludon ex", ("archaludon",)),
    ("lopunny", "Lopunny", ("lopunny",)),
    ("lucario", "Lucario", ("lucario",)),
    ("gardevoir", "Gardevoir", ("gardevoir",)),
    ("ns-zoroark", "N's Zoroark", ("zoroark",)),
    ("raging-bolt", "Raging Bolt", ("raging", "bolt")),
    ("festival-lead", "Festival Lead", ("festival", "lead")),
)

for _id, _name, _slug_all in _CORE_LADDER_GENERIC:
    register(
        Archetype(
            id=_id,
            name=_name,
            description="Top-ladder deck family used by the deck-agnostic core.",
            slug_all=_slug_all,
        )
    )

# Exact auxiliary-head row order used by the deck-agnostic checkpoints made
# immediately before the post-snapshot ladder families below were registered.
# Keep this immutable: expansion maps rows by name and only appends capacity.
PINNED_CORE_AUX_ARCHETYPE_IDS: tuple[str, ...] = tuple(ARCHETYPES)

# Exact auxiliary-head row order embedded in the accepted cumulative-v4
# deck-agnostic core (sha256:07f035f8e8093900c47409edab6f20e72cb23f14466737cae213cbee58101ea9).
# This lineage added three ladder families and reordered the bank independently
# of the registry declaration order. Keep it immutable so warm starts copy
# semantic rows by name and move only the old final ``unknown`` row.
CUMULATIVE_V4_AUX_ARCHETYPE_IDS: tuple[str, ...] = (
    "crustle",
    "marnie-s-grimmsnarl-ex",
    "garchomp",
    "cornerstone-ogerpon",
    "rockets-mewtwo",
    "starmie",
    "hammer-pult",
    "alakazam",
    "lucario",
    "archaludon-ex",
    "dragapult-dudunsparce",
    "dragapult",
    "dudunsparce",
    "hops-trevenant",
    "walrein",
    "dragapult-dusknoir",
    "dragapult-blaziken",
    "lopunny",
    "gardevoir",
    "ns-zoroark",
    "raging-bolt",
    "festival-lead",
)

# Additive post-snapshot families discovered by later ladder windows.  These
# append after every pre-existing auxiliary class so checkpoint row indices
# stay stable.  They are not inserted into CORE_LADDER_ARCHETYPE_IDS until a
# separately checksummed ladder-mix artifact promotes them.
_ADDITIVE_LADDER_GENERIC: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("dudunsparce", "Dudunsparce", ("dudunsparce",)),
    ("hops-trevenant", "Hop's Trevenant", ("hop", "trevenant")),
    ("walrein", "Walrein", ("walrein",)),
    ("thwackey", "Thwackey", ("thwackey",)),
    (
        "team-rockets-spidops",
        "Team Rocket's Spidops",
        ("team", "rocket", "spidops"),
    ),
)

for _id, _name, _slug_all in _ADDITIVE_LADDER_GENERIC:
    register(
        Archetype(
            id=_id,
            name=_name,
            description="Additive ladder family discovered after the pinned core snapshot.",
            slug_all=_slug_all,
        )
    )

register(Archetype(
    id="teal-mask-ogerpon-ex",
    name="Teal Mask Ogerpon ex",
    description=(
        "Nonlinear Slop Box toolbox: the public archetype-151 Raging Bolt "
        "Ogerpon family centered on Mega Kangaskhan ex, Meowth ex, Teal Mask "
        "acceleration, Energy Switch, Crispin, Glass Trumpet, and Area Zero. "
        "The stable logical program ID remains Teal Mask Ogerpon ex."
    ),
    signature_only=True,
))

CORE_LADDER_ARCHETYPE_IDS: tuple[str, ...] = (
    "alakazam",
    "crustle",
    "marnie-s-grimmsnarl-ex",
    "garchomp",
    "cornerstone-ogerpon",
    "rockets-mewtwo",
    "starmie",
    "hammer-pult",
    "archaludon-ex",
    "lopunny",
    "lucario",
    "dragapult-dudunsparce",
    "dragapult",
    "gardevoir",
    "ns-zoroark",
    "raging-bolt",
    "festival-lead",
)


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


def is_teal_mask_ogerpon_box_signature(card_ids: Iterable[int]) -> bool:
    """True for Slop Box; legacy function name retained for compatibility."""

    c = _counts(card_ids)
    return (
        c.get(RAGING_BOLT_EX, 0) >= 2
        and c.get(TEAL_MASK_OGERPON_EX, 0) >= 3
        and c.get(WELLSPRING_MASK_OGERPON_EX, 0) >= 1
        and c.get(LILLIES_CLEFAIRY_EX, 0) >= 1
        and c.get(MEGA_KANGASKHAN_EX, 0) == SLOP_BOX_KANGASKHAN_COUNT
        and c.get(MEOWTH_EX, 0) >= 3
        and c.get(ENERGY_SWITCH, 0) >= 4
        and c.get(AREA_ZERO_UNDERDEPTHS, 0) >= 4
    )


def is_archaludon_ex_modal_representative(
    card_ids: Iterable[int],
) -> bool:
    """Match only the pinned exact 60-card Archaludon ex modal multiset."""

    return tuple(sorted(card_ids)) == _ARCHALUDON_EX_MODAL_MULTISET


def is_crustle_naic_2026_representative(card_ids: Iterable[int]) -> bool:
    """Match only the checksum-bound current Crustle representative."""

    return tuple(sorted(card_ids)) == _CRUSTLE_NAIC_2026_MULTISET


def classify_deck(card_ids: Iterable[int]) -> str:
    """Classify a deck from its card-id multiset. Authoritative.

    Precedence: Hammer signature first (steals matching Dudunsparce lists), then
    Dragapult tech variants (Dusknoir > Blaziken > Dudunsparce), then generic
    Dragapult, else ``"unknown"``.
    """
    card_ids = list(card_ids)
    if is_archaludon_ex_modal_representative(card_ids):
        return "archaludon-ex"
    if is_crustle_naic_2026_representative(card_ids):
        return "crustle"
    if is_hammer_signature(card_ids):
        return "hammer-pult"
    if is_teal_mask_ogerpon_box_signature(card_ids):
        return "teal-mask-ogerpon-ex"

    c = _counts(card_ids)
    if has_line(c, DRAGAPULT_LINE):
        if has_line(c, DUSKNOIR_LINE):
            return "dragapult-dusknoir"
        if has_line(c, BLAZIKEN_LINE):
            return "dragapult-blaziken"
        if has_line(c, DUDUNSPARCE_LINE):
            return "dragapult-dudunsparce"
        return "dragapult"
    # Main-attacker signatures for additive post-snapshot families. Check each
    # distinctive attacker before Dudunsparce because several lists use the
    # Dunsparce line as a support engine.
    if c.get(648, 0) > 0 and has_line(c, MARNIE_GRIMMSNARL_LINE):
        return "marnie-s-grimmsnarl-ex"
    # The standalone Team Rocket's Spidops family uses the same 4-4
    # Tarountula/Spidops engine as Rocket's Mewtwo, so the line alone is not
    # distinctive.  Require a real multi-copy Spidops attacker line and at
    # most one Mewtwo ex.  The pinned Rocket's Mewtwo representative has two
    # Mewtwo ex and therefore remains in its established family.
    if (
        c.get(TEAM_ROCKETS_TAROUNTULA, 0) >= 2
        and c.get(TEAM_ROCKETS_SPIDOPS, 0) >= 2
        and c.get(TEAM_ROCKETS_MEWTWO_EX, 0) <= 1
    ):
        return "team-rockets-spidops"
    if c.get(879, 0) > 0:
        return "hops-trevenant"
    if c.get(943, 0) > 0:
        return "walrein"
    # The canonical specialist name is ``thwackey`` even though historical
    # source lists used the physical ``festival-lead`` label. Require the full
    # multi-copy Boom Boom Groove/Festival Lead engine so ordinary grass decks
    # cannot be captured by a single support card.
    if (
        c.get(THWACKEY_GROOKEY, 0) >= 3
        and c.get(THWACKEY, 0) >= 3
        and c.get(THWACKEY_DIPPLIN, 0) >= 2
        and c.get(FESTIVAL_GROUNDS, 0) >= 2
    ):
        return "thwackey"
    if c.get(306, 0) > 0 or c.get(997, 0) > 0:
        return "dudunsparce"
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
