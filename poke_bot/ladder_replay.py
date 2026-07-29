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
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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


def canonical_deck_sha256(card_ids: Sequence[int]) -> str:
    """Content identity used by checksum-bound public archetype catalogs."""
    encoded = json.dumps(
        canonical_deck_fingerprint(card_ids),
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


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
        logical_aliases: Mapping[str, str] | None = None,
        authoritative_deck_catalogs: Sequence[str | Path] = (),
        authoritative_only_ids: Sequence[str] = (),
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
        aliases = {
            str(source).strip().casefold(): str(target).strip().casefold()
            for source, target in dict(logical_aliases or {}).items()
        }
        known = set(archetypes.archetype_ids())
        invalid_aliases = sorted(
            (source, target)
            for source, target in aliases.items()
            if not source or not target or source not in known or target not in known
        )
        if invalid_aliases:
            raise ValueError(f"invalid logical ladder aliases: {invalid_aliases}")
        self.logical_aliases = dict(sorted(aliases.items()))
        authoritative_only = tuple(
            dict.fromkeys(
                str(value).strip().casefold()
                for value in authoritative_only_ids
                if str(value).strip()
            )
        )
        unknown_authoritative_only = sorted(set(authoritative_only) - known)
        if unknown_authoritative_only:
            raise ValueError(
                "unregistered authoritative-only ladder archetypes: "
                f"{unknown_authoritative_only}"
            )
        authoritative: dict[str, str] = {}
        catalog_contracts: list[dict[str, Any]] = []
        for raw_path in authoritative_deck_catalogs:
            path = Path(raw_path).expanduser().resolve()
            payload = json.loads(path.read_text(encoding="utf-8"))
            specialist_id = str(payload.get("specialist_id") or "").casefold()
            fingerprints = tuple(
                str(value) for value in payload.get("deck_fingerprints") or ()
            )
            source_archetype = dict(payload.get("source_archetype") or {})
            source_deck_rows = payload.get("source_deck_rows")
            source_window = dict(payload.get("source_window") or {})
            observed_by_day = dict(payload.get("observed_by_day") or {})
            minimum_games = int(payload.get("minimum_acting_seat_games") or 0)
            observed_games = int(payload.get("observed_acting_seat_games") or 0)
            try:
                start = date.fromisoformat(str(source_window.get("start") or ""))
                end = date.fromisoformat(str(source_window.get("end") or ""))
            except ValueError as exc:
                raise ValueError(
                    f"invalid authoritative public deck catalog: {path}"
                ) from exc
            expected_dates = [
                (start + timedelta(days=index)).isoformat()
                for index in range((end - start).days + 1)
            ]
            if (
                payload.get("schema")
                != "poke_bot.public_deck_archetype_catalog/v1"
                or specialist_id not in archetypes.archetype_ids()
                or not str(payload.get("source") or "").startswith("https://")
                or minimum_games <= 0
                or observed_games < minimum_games
                or int(source_window.get("days") or 0) != len(expected_dates)
                or end < start
                or sorted(observed_by_day) != expected_dates
                or any(int(value) < 0 for value in observed_by_day.values())
                or sum(int(value) for value in observed_by_day.values())
                != observed_games
                or not fingerprints
                or len(set(fingerprints)) != len(fingerprints)
                or any(
                    not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                    for value in fingerprints
                )
            ):
                raise ValueError(
                    f"invalid authoritative public deck catalog: {path}"
                )
            source_row_fingerprints: set[str] = set()
            source_rows_bound = bool(
                isinstance(source_archetype.get("id"), int)
                and not isinstance(source_archetype.get("id"), bool)
                and int(source_archetype["id"]) > 0
                and bool(str(source_archetype.get("name") or "").strip())
                and isinstance(source_deck_rows, list)
                and bool(source_deck_rows)
            )
            if source_rows_bound:
                for row in source_deck_rows:
                    cards = (
                        row.get("card_ids")
                        if isinstance(row, dict)
                        else None
                    )
                    if (
                        not isinstance(cards, list)
                        or len(cards) != 60
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            for value in cards
                        )
                        or row.get("archetype_id")
                        != source_archetype["id"]
                    ):
                        source_rows_bound = False
                        break
                    source_row_fingerprints.add(
                        canonical_deck_sha256(cards)
                    )
                source_rows_bound = bool(
                    source_rows_bound
                    and source_row_fingerprints == set(fingerprints)
                )
            for fingerprint in fingerprints:
                previous = authoritative.get(fingerprint)
                if previous is not None and previous != specialist_id:
                    raise ValueError(
                        "authoritative public deck catalogs disagree for "
                        f"{fingerprint}"
                    )
                authoritative[fingerprint] = specialist_id
            catalog_contracts.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "specialist_id": specialist_id,
                    "source": payload["source"],
                    "source_archetype": source_archetype,
                    "source_deck_rows_bound_to_fingerprints": (
                        source_rows_bound
                    ),
                    "source_window": source_window,
                    "minimum_acting_seat_games": minimum_games,
                    "observed_acting_seat_games": observed_games,
                    "deck_fingerprint_count": len(fingerprints),
                }
            )
        catalog_specialist_ids = {
            str(row["specialist_id"]) for row in catalog_contracts
        }
        missing_authoritative_catalogs = sorted(
            set(authoritative_only) - catalog_specialist_ids
        )
        if missing_authoritative_catalogs:
            raise ValueError(
                "authoritative-only ladder archetypes lack a public deck "
                f"catalog: {missing_authoritative_catalogs}"
            )
        unbound_authoritative_catalogs = sorted(
            specialist_id
            for specialist_id in authoritative_only
            if not any(
                row["specialist_id"] == specialist_id
                and row["source_deck_rows_bound_to_fingerprints"] is True
                for row in catalog_contracts
            )
        )
        if unbound_authoritative_catalogs:
            raise ValueError(
                "authoritative-only ladder archetypes lack exact public "
                "source-archetype deck-row binding: "
                f"{unbound_authoritative_catalogs}"
            )
        self._authoritative_deck_labels = authoritative
        self._authoritative_catalog_contracts = tuple(catalog_contracts)
        self.authoritative_only_ids = authoritative_only
        self._authoritative_only = frozenset(authoritative_only)

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
        logical_aliases: Mapping[str, str] | None = None,
        authoritative_deck_catalogs: Sequence[str | Path] = (),
        authoritative_only_ids: Sequence[str] = (),
    ) -> "LadderReplayClassifier":
        return cls(
            load_ladder_deck_mix(mix_path),
            load_ladder_deck_representatives(representatives_path),
            card_csv=Path(card_csv) if card_csv is not None else None,
            additive_registered_ids=additive_registered_ids,
            logical_aliases=logical_aliases,
            authoritative_deck_catalogs=authoritative_deck_catalogs,
            authoritative_only_ids=authoritative_only_ids,
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
            "logical_aliases": self.logical_aliases,
            "authoritative_deck_catalogs": list(
                self._authoritative_catalog_contracts
            ),
            "authoritative_only_ids": list(self.authoritative_only_ids),
            "derived_ace_ids": {
                key: list(value)
                for key, value in sorted(self._derived_ace_ids.items())
            },
        }

    def classify_deck(self, card_ids: Optional[Sequence[int]]) -> LadderReplayLabel:
        if card_ids is None or len(card_ids) != 60:
            return LadderReplayLabel(archetypes.UNKNOWN, "invalid_or_missing_deck")
        cards = [int(card_id) for card_id in card_ids]
        authoritative = self._authoritative_deck_labels.get(
            canonical_deck_sha256(cards)
        )
        if authoritative is not None:
            return self._logical_label(
                authoritative,
                "authoritative_public_deck_identity",
            )
        exact = self._exact.get(canonical_deck_fingerprint(cards))
        if exact is not None:
            return self._fallback_label(exact, "representative_exact")

        registered = archetypes.classify_deck(cards)
        if registered in self._active or registered in self._additive_registered:
            return self._fallback_label(registered, "registered_signature")

        present = set(cards)
        for entry in self._signature_rows:
            if all(present.intersection(group) for group in entry.signature_groups):
                return self._fallback_label(
                    entry.deck_id,
                    "artifact_signature",
                )

        for deck_id, ace_ids in self._derived_ace_ids.items():
            if present.intersection(ace_ids):
                return self._fallback_label(deck_id, "derived_primary_ace")
        return LadderReplayLabel(archetypes.UNKNOWN, "unrecognized")

    def _fallback_label(self, deck_id: str, method: str) -> LadderReplayLabel:
        label = self._logical_label(deck_id, method)
        if label.deck_id in self._authoritative_only:
            return LadderReplayLabel(
                archetypes.UNKNOWN,
                "authoritative_public_deck_identity_required",
            )
        return label

    def _logical_label(self, deck_id: str, method: str) -> LadderReplayLabel:
        logical = self.logical_aliases.get(str(deck_id).casefold(), str(deck_id))
        suffix = "+logical_alias" if logical != deck_id else ""
        return LadderReplayLabel(logical, method + suffix)

    def classify_episode(
        self, payload: dict[str, Any]
    ) -> tuple[list[Optional[list[int]]], list[LadderReplayLabel]]:
        decks = extract_setup_decks(payload)
        return decks, [self.classify_deck(deck) for deck in decks]
