"""Deck loading and the per-archetype deck pool.

Deck CSVs in this repo are a flat list of card ids — **one card id per line,
repeated per copy** (60 lines total), matching the competition ``deck.csv``
contract read by ``sample_submission/main.py``. This reader additionally accepts
a ``cardId,quantity`` two-column form for robustness. Either way it yields the
flat 60-int list that :func:`cg_env.battle_start` expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from . import archetypes, paths

DECK_SIZE = 60


def read_deck(path: Union[str, Path]) -> list[int]:
    """Read a deck CSV into a flat list of card ids.

    Supports both the flat one-id-per-line form and a ``cardId,quantity`` form.
    Raises ``ValueError`` if the result is not exactly 60 cards.
    """
    path = Path(path)
    flat: list[int] = []
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) == 1:
                flat.append(int(parts[0]))
            elif len(parts) >= 2:
                card_id, qty = int(parts[0]), int(parts[1])
                flat.extend([card_id] * qty)
            # ignore malformed lines silently
    if len(flat) != DECK_SIZE:
        raise ValueError(
            f"Deck {path} resolved to {len(flat)} cards (expected {DECK_SIZE})."
        )
    return flat


@dataclass(frozen=True)
class DeckEntry:
    """A named deck registered in the pool."""

    name: str
    path: Path
    #: Declared archetype id (may be refined by signature classification).
    archetype_id: str

    def load(self) -> list[int]:
        return read_deck(self.path)

    def classify(self) -> str:
        """Signature-based archetype id for this deck's actual card list."""
        return archetypes.classify_deck(self.load())


class DeckPool:
    """Registry of decks keyed by name, grouped by archetype.

    Archetype membership is resolved by *card signature* (not the declared slug),
    so a Hammer-signature Dudunsparce list lands under ``hammer-pult``.
    """

    def __init__(self) -> None:
        self._decks: dict[str, DeckEntry] = {}

    def register(self, name: str, path: Union[str, Path], archetype_id: Optional[str] = None) -> DeckEntry:
        path = Path(path)
        if archetype_id is None:
            archetype_id = archetypes.classify_deck(read_deck(path))
        entry = DeckEntry(name=name, path=path, archetype_id=archetype_id)
        self._decks[name] = entry
        return entry

    def get(self, name: str) -> DeckEntry:
        return self._decks[name]

    def load(self, name: str) -> list[int]:
        return self._decks[name].load()

    def names(self) -> list[str]:
        return list(self._decks.keys())

    def by_archetype(self, archetype_id: str) -> list[DeckEntry]:
        return [e for e in self._decks.values() if e.archetype_id == archetype_id]

    def __len__(self) -> int:
        return len(self._decks)

    def __contains__(self, name: str) -> bool:
        return name in self._decks


def dragapult_deck() -> list[int]:
    """Load the locked pure-Dragapult v1 submission deck (flat 60-int list)."""
    return read_deck(paths.DRAGAPULT_DECK)


def hammer_pult_deck() -> list[int]:
    """Load the Hammer-Pult list (Campinas 2026 4th; hammer signature)."""
    return read_deck(paths.HAMMER_PULT_DECK)


#: Strong dunsparce-line list (SE Lima 2026 2nd; classifies dragapult-dudunsparce).
DUDUNSPARCE_DECK: Path = (
    paths.DECKS_DIR
    / "competitive"
    / "high_performing"
    / "2026-05_se-lima-2026_2nd_dragapult.csv"
)

#: Deck used per primary archetype (data-driven selection writes the env var).
PRIMARY_DECK_BY_ARCHETYPE: dict[str, Path] = {
    "dragapult": paths.DRAGAPULT_DECK,
    "hammer-pult": paths.HAMMER_PULT_DECK,
    "dragapult-dudunsparce": DUDUNSPARCE_DECK,
}


def primary_archetype() -> str:
    """Selected primary archetype (``POKEBOT_PRIMARY_ARCHETYPE``, default dragapult)."""
    import os

    return os.environ.get("POKEBOT_PRIMARY_ARCHETYPE", "dragapult")


def primary_deck() -> list[int]:
    """Primary submission deck for the selected primary archetype.

    Prefers ``submission/deck.csv`` when it matches the selected archetype
    (the collector/pipeline copies the right list there); otherwise falls back
    to the registered per-archetype deck path.
    """
    arch = primary_archetype()
    if paths.SUBMISSION_DECK.is_file():
        try:
            deck = read_deck(paths.SUBMISSION_DECK)
            if archetypes.classify_deck(deck) == arch:
                return deck
        except ValueError:
            pass
    path = PRIMARY_DECK_BY_ARCHETYPE.get(arch, paths.DRAGAPULT_DECK)
    return read_deck(path)


def default_pool() -> DeckPool:
    """Build the default pool with pure Dragapult + Hammer-Pult registered."""
    pool = DeckPool()
    if paths.DRAGAPULT_DECK.is_file():
        pool.register("dragapult", paths.DRAGAPULT_DECK, archetype_id="dragapult")
    if paths.SUBMISSION_DECK.is_file():
        # Prefer live submission/deck.csv when present (should match DRAGAPULT_DECK).
        try:
            pool.register(
                "submission",
                paths.SUBMISSION_DECK,
                archetype_id=archetypes.classify_deck(read_deck(paths.SUBMISSION_DECK)),
            )
        except ValueError:
            pass
    if paths.HAMMER_PULT_DECK.is_file():
        pool.register("hammer-pult", paths.HAMMER_PULT_DECK, archetype_id="hammer-pult")
    return pool
