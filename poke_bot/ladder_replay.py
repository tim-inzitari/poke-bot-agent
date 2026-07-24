"""Deck-family labels for hot-starting from official top-ladder replays.

The classifier is pinned to the checksummed ladder-mix and representative
artifacts.  Exact modal lists are recognized first, followed by the artifact's
card signatures.  Families that the source report labeled by played ace name
derive one unambiguous non-support ``ex`` signature from their representative
list and the competition card database.

This module deliberately does not guess a family for an unrecognized deck.
Unknown seats remain useful as opponents, but are never behavior-cloned as one
of our active ladder policies.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from . import archetypes
from .ladder_deck_mix import (
    LadderDeckMix,
    LadderDeckRepresentatives,
    load_ladder_deck_mix,
    load_ladder_deck_representatives,
)
from .replay_import import extract_setup_decks


# Utility/support Pokemon ex must not define an archetype's main attacker.
SUPPORT_EX_IDS = frozenset({140, 1071, 184, 754, 272})


def canonical_deck_fingerprint(card_ids: Sequence[int]) -> tuple[int, ...]:
    """Stable multiset identity for one submitted 60-card list."""
    return tuple(sorted(int(card_id) for card_id in card_ids))


def _numeric_hp(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _load_card_rows(path: Optional[Path]) -> dict[int, dict[str, str]]:
    if path is None or not Path(path).is_file():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                card_id = int(row["Card ID"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[card_id] = dict(row)
    return rows


@dataclass(frozen=True)
class LadderReplayLabel:
    deck_id: str
    method: str


class LadderReplayClassifier:
    """Fail-closed classifier bound to one immutable top-ladder snapshot."""

    def __init__(
        self,
        mix: LadderDeckMix,
        representatives: LadderDeckRepresentatives,
        *,
        card_csv: Optional[Path] = None,
        additive_registered_ids: Sequence[str] = (),
    ) -> None:
        bound = representatives.bind(mix)
        self.mix = mix
        self.representatives = representatives
        self.active_ids = tuple(entry.bucket.deck_id for entry in bound)
        self._active = frozenset(self.active_ids)
        additive = tuple(dict.fromkeys(str(value) for value in additive_registered_ids))
        unknown_additive = sorted(set(additive) - set(archetypes.archetype_ids()))
        if unknown_additive:
            raise ValueError(
                f"unregistered additive ladder archetypes: {unknown_additive}"
            )
        self.additive_registered_ids = additive
        self._additive_registered = frozenset(additive)

        exact: dict[tuple[int, ...], str] = {}
        for entry in bound:
            fingerprint = canonical_deck_fingerprint(entry.card_ids)
            previous = exact.get(fingerprint)
            if previous is not None and previous != entry.bucket.deck_id:
                raise ValueError(
                    "ladder representatives share a deck multiset: "
                    f"{previous!r} and {entry.bucket.deck_id!r}"
                )
            exact[fingerprint] = entry.bucket.deck_id
        self._exact = exact

        # More-specific signatures sort first.  This matters for Dragapult:
        # Hammer/Dudunsparce must win before the generic Dragapult line.
        self._signature_rows = tuple(
            sorted(
                (entry for entry in mix.decks if entry.signature_groups),
                key=lambda entry: (-len(entry.signature_groups), entry.source_rank),
            )
        )

        card_rows = _load_card_rows(card_csv)
        derived: dict[str, tuple[int, ...]] = {}
        for entry in mix.decks:
            if entry.signature_groups:
                continue
            rep = representatives.decks[entry.deck_id]
            counts = Counter(int(card_id) for card_id in rep["card_ids"])
            candidates: list[tuple[float, int, int]] = []
            for card_id, count in counts.items():
                if card_id in SUPPORT_EX_IDS:
                    continue
                row = card_rows.get(card_id) or {}
                name = str(row.get("Card Name", "") or "")
                hp = _numeric_hp(row.get("HP"))
                if hp > 0.0 and re.search(r"\bex\b", name, re.IGNORECASE):
                    candidates.append((hp, count, card_id))
            if candidates:
                # Main attacker is normally the highest-HP non-support ex.
                best_hp = max(value[0] for value in candidates)
                ace_ids = sorted(
                    card_id
                    for hp, _count, card_id in candidates
                    if hp == best_hp
                )
                derived[entry.deck_id] = tuple(ace_ids)
        self._derived_ace_ids = derived

    @classmethod
    def from_paths(
        cls,
        mix_path: str | Path,
        representatives_path: str | Path,
        *,
        card_csv: Optional[str | Path] = None,
        additive_registered_ids: Sequence[str] = (),
    ) -> "LadderReplayClassifier":
        return cls(
            load_ladder_deck_mix(mix_path),
            load_ladder_deck_representatives(representatives_path),
            card_csv=Path(card_csv) if card_csv is not None else None,
            additive_registered_ids=additive_registered_ids,
        )

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "mix_artifact_sha256": self.mix.artifact_sha256,
            "representatives_artifact_sha256": (
                self.representatives.artifact_sha256
            ),
            "active_deck_ids": list(self.active_ids),
            "additive_registered_ids": list(self.additive_registered_ids),
            "derived_ace_ids": {
                key: list(value)
                for key, value in sorted(self._derived_ace_ids.items())
            },
        }

    def classify_deck(self, card_ids: Optional[Sequence[int]]) -> LadderReplayLabel:
        if card_ids is None or len(card_ids) != 60:
            return LadderReplayLabel(archetypes.UNKNOWN, "invalid_or_missing_deck")
        cards = [int(card_id) for card_id in card_ids]
        exact = self._exact.get(canonical_deck_fingerprint(cards))
        if exact is not None:
            return LadderReplayLabel(exact, "representative_exact")

        registered = archetypes.classify_deck(cards)
        if registered in self._active or registered in self._additive_registered:
            return LadderReplayLabel(registered, "registered_signature")

        present = set(cards)
        for entry in self._signature_rows:
            if all(present.intersection(group) for group in entry.signature_groups):
                return LadderReplayLabel(entry.deck_id, "artifact_signature")

        for deck_id, ace_ids in self._derived_ace_ids.items():
            if present.intersection(ace_ids):
                return LadderReplayLabel(deck_id, "derived_primary_ace")
        return LadderReplayLabel(archetypes.UNKNOWN, "unrecognized")

    def classify_episode(
        self, payload: dict[str, Any]
    ) -> tuple[list[Optional[list[int]]], list[LadderReplayLabel]]:
        decks = extract_setup_decks(payload)
        return decks, [self.classify_deck(deck) for deck in decks]
