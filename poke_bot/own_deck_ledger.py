"""Causal, match-scoped accounting for an acting player's own deck.

This module is deliberately a dormant side-store.  It does not select actions,
does not inspect a simulator/search state, and is not wired into the active
r241 runtime.  A caller supplies only the actor-visible observation at a real
decision boundary.  In particular, it never reads opponent zones, hidden deck
order, hidden prize arrays, or ``transition_after`` labels.

The important distinction is between exact accounting and uncertainty.  The
starting deck and currently visible own cards are exact.  Face-down prizes make
the identity of cards in the draw pile uncertain until a real prompt exposes
them.  Consequently every card gets conservative lower/upper availability
bounds; a ``select.deck`` prompt can tighten those bounds without predicting
anything beyond the current observation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OWN_DECK_LEDGER_SCHEMA = "poke_bot.own_deck_ledger/v1"
OWN_DECK_LEDGER_SCHEMA_VERSION = 1

# ``features_for_card`` / ``option_features`` are intentionally plain floats so
# the dormant model adapter can consume them without importing torch here.
OPTION_FEATURE_DIM = 8
OPTION_FEATURE_NAMES = (
    "lower",
    "upper",
    "expected",
    "probability_at_least_one",
    "exact",
    "select_count",
    "looking_count",
    "exposed",
)
SCALAR_VECTOR_NAMES = (
    "deck_count_over_60",
    "prize_count_over_6",
    "unknown_prize_slots_over_6",
    "unknown_non_deck_slots_over_60",
    "visible_own_cards_over_60",
    "select_deck_cards_over_60",
    "looking_cards_over_60",
    "full_deck_exposed",
    "integrity_ok",
    "fail_closed",
)


class OwnDeckLedgerError(ValueError):
    """The immutable ledger contract cannot represent supplied input."""


def _exact_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        return result if value == result else None
    except (OverflowError, TypeError, ValueError):
        return None


def _field(value: Any, *names: str) -> Any:
    """Read a raw JSON field or a ``cg.api`` attribute without importing cg."""

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        try:
            return getattr(value, name)
        except (AttributeError, TypeError):
            pass
    return None


def _as_rows(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _enum_token(value: Any) -> str:
    if hasattr(value, "name"):
        value = value.name
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _card_id(value: Any) -> int | None:
    """Resolve a card identity without recursively walking arbitrary payloads."""

    direct = _exact_int(value)
    if direct is not None:
        return direct if direct >= 0 else None
    raw = _field(value, "id")
    ident = _exact_int(raw)
    if ident is not None:
        return ident if ident >= 0 else None
    # A few JSON surfaces wrap a physical card in one well-known field.  Do not
    # recurse arbitrary mappings: that would double-count card metadata.
    for name in ("card", "pokemon", "energy", "source"):
        nested = _field(value, name)
        if nested is not None:
            ident = _card_id(nested)
            if ident is not None:
                return ident
    return None


def _card_serial(value: Any) -> int | None:
    serial = _exact_int(_field(value, "serial"))
    return serial if serial is not None and serial >= 0 else None


def _pairs(counter: Counter[int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(card_id), int(count)) for card_id, count in counter.items() if count > 0))


def _counter(pairs: Sequence[Sequence[int]]) -> Counter[int]:
    result: Counter[int] = Counter()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise OwnDeckLedgerError("card-count pair must contain exactly two entries")
        card_id = _exact_int(pair[0])
        count = _exact_int(pair[1])
        if card_id is None or card_id < 0 or count is None or count <= 0:
            raise OwnDeckLedgerError("card-count pair is invalid")
        result[card_id] += count
    return result


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hypergeom_probability_at_least_one(total: int, copies: int, draws: int) -> float | None:
    if total < 0 or copies < 0 or draws < 0 or copies > total or draws > total:
        return None
    if copies == 0 or draws == 0:
        return 0.0
    if total - copies < draws:
        return 1.0
    # Product form avoids enormous integer combinations and is stable at <= 60.
    no_hit = 1.0
    for index in range(draws):
        no_hit *= float(total - copies - index) / float(total - index)
    return max(0.0, min(1.0, 1.0 - no_hit))


@dataclass(frozen=True)
class CardAvailability:
    """Conservative current-draw-pile availability for one original card id."""

    card_id: int
    lower: int
    upper: int
    expected: float | None
    probability_at_least_one: float | None
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "lower": self.lower,
            "upper": self.upper,
            "expected": self.expected,
            "probability_at_least_one": self.probability_at_least_one,
            "exact": self.exact,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CardAvailability:
        card_id = _exact_int(value.get("card_id"))
        lower = _exact_int(value.get("lower"))
        upper = _exact_int(value.get("upper"))
        if card_id is None or card_id < 0 or lower is None or upper is None or lower < 0 or upper < lower:
            raise OwnDeckLedgerError("invalid card availability")
        expected_raw = value.get("expected")
        expected = None if expected_raw is None else float(expected_raw)
        probability_raw = value.get("probability_at_least_one")
        probability = None if probability_raw is None else float(probability_raw)
        if expected is not None and (not math.isfinite(expected) or not lower <= expected <= upper):
            raise OwnDeckLedgerError("availability expected value is outside bounds")
        if probability is not None and (not math.isfinite(probability) or not 0.0 <= probability <= 1.0):
            raise OwnDeckLedgerError("availability probability is invalid")
        if not isinstance(value.get("exact"), bool):
            raise OwnDeckLedgerError("availability exact flag is invalid")
        return cls(card_id, lower, upper, expected, probability, value["exact"])


@dataclass(frozen=True)
class PromptCard:
    """One prompt-local card occurrence; index preserves menu multiplicity."""

    card_id: int
    serial: int | None
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {"card_id": self.card_id, "serial": self.serial, "index": self.index}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromptCard:
        card_id = _exact_int(value.get("card_id"))
        index = _exact_int(value.get("index"))
        serial = value.get("serial")
        serial_i = None if serial is None else _exact_int(serial)
        if card_id is None or card_id < 0 or index is None or index < 0 or (serial is not None and (serial_i is None or serial_i < 0)):
            raise OwnDeckLedgerError("invalid prompt card")
        return cls(card_id, serial_i, index)


@dataclass(frozen=True)
class OwnDeckLedgerSnapshot:
    """Immutable, JSON-serializable match ledger observation.

    ``card_availability`` covers every distinct id in the starting deck.  It is
    deliberately a tuple rather than a mapping so snapshot equality, deep-copy,
    and fingerprinting remain deterministic across processes.
    """

    schema: str
    version: int
    deck_fingerprint: str
    observation_fingerprint: str
    fingerprint: str
    revision: int
    actor: int | None
    starting_counts: tuple[tuple[int, int], ...]
    visible_zone_counts: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    known_prize_slots: tuple[tuple[int, int], ...]
    deck_count: int | None
    prize_count: int | None
    unknown_prize_slots: int | None
    unknown_non_deck_slots: int | None
    unaccounted_non_deck_slots: int | None
    select_deck_entries: tuple[PromptCard, ...]
    looking_entries: tuple[PromptCard, ...]
    select_deck_counts: tuple[tuple[int, int], ...]
    looking_counts: tuple[tuple[int, int], ...]
    prompt_exposure_scope: str
    card_availability: tuple[CardAvailability, ...]
    integrity_ok: bool
    fail_closed: bool
    integrity_flags: tuple[str, ...]

    @property
    def availability_by_card(self) -> dict[int, CardAvailability]:
        return {row.card_id: row for row in self.card_availability}

    @property
    def select_deck_counter(self) -> Counter[int]:
        return _counter(self.select_deck_counts)

    @property
    def looking_counter(self) -> Counter[int]:
        return _counter(self.looking_counts)

    @property
    def scalar_vector(self) -> tuple[float, ...]:
        total = max(sum(count for _card, count in self.starting_counts), 1)
        deck = -1.0 if self.deck_count is None else float(self.deck_count) / float(total)
        prize = -1.0 if self.prize_count is None else float(self.prize_count) / 6.0
        unknown_prizes = -1.0 if self.unknown_prize_slots is None else float(self.unknown_prize_slots) / 6.0
        unknown_outside = -1.0 if self.unknown_non_deck_slots is None else float(self.unknown_non_deck_slots) / float(total)
        visible = sum(sum(count for _card, count in rows) for _name, rows in self.visible_zone_counts)
        select_total = sum(count for _card, count in self.select_deck_counts)
        looking_total = sum(count for _card, count in self.looking_counts)
        return (
            deck,
            prize,
            unknown_prizes,
            unknown_outside,
            float(visible) / float(total),
            float(select_total) / float(total),
            float(looking_total) / float(total),
            1.0 if self.prompt_exposure_scope == "select_deck_full_by_count" else 0.0,
            1.0 if self.integrity_ok else 0.0,
            1.0 if self.fail_closed else 0.0,
        )

    def features_for_card(self, card_id: Any) -> tuple[float, ...]:
        """Return the fixed input-only availability row for a visible card id."""

        ident = _exact_int(card_id)
        if ident is None or ident < 0:
            return (0.0,) * OPTION_FEATURE_DIM
        row = self.availability_by_card.get(ident)
        if row is None:
            return (0.0,) * OPTION_FEATURE_DIM
        select_count = self.select_deck_counter.get(ident, 0)
        looking_count = self.looking_counter.get(ident, 0)
        exposed = bool(select_count or looking_count)
        return (
            float(row.lower),
            float(row.upper),
            0.0 if row.expected is None else float(row.expected),
            0.0 if row.probability_at_least_one is None else float(row.probability_at_least_one),
            1.0 if row.exact else 0.0,
            float(select_count),
            float(looking_count),
            1.0 if exposed else 0.0,
        )

    def option_features(
        self,
        observation: Any,
        action_combos: Sequence[Sequence[int]],
    ) -> tuple[tuple[float, ...], ...]:
        """Return one ledger row for each legal/factorized candidate.

        A factorized candidate differs from its prefix by its final selected
        option, so the final card-bearing option is used.  Full ordered actions
        use the final card-bearing option for the same reason.  STOP/non-card
        candidates are exact zero rows.  Resolution reads only current own
        zones plus ``select.deck``/``current.looking`` from ``observation``.
        """

        options = _as_rows(_field(_field(observation, "select"), "option", "options")) or []
        rows: list[tuple[float, ...]] = []
        for combo in action_combos:
            card: int | None = None
            for raw_index in reversed(tuple(combo)):
                index = _exact_int(raw_index)
                if index is None or not 0 <= index < len(options):
                    card = None
                    break
                card = _option_card_id(observation, options[index])
                if card is not None:
                    break
            rows.append(self.features_for_card(card) if card is not None else (0.0,) * OPTION_FEATURE_DIM)
        return tuple(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "deck_fingerprint": self.deck_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "fingerprint": self.fingerprint,
            "revision": self.revision,
            "actor": self.actor,
            "starting_counts": [list(pair) for pair in self.starting_counts],
            "visible_zone_counts": {
                name: [list(pair) for pair in pairs]
                for name, pairs in self.visible_zone_counts
            },
            "known_prize_slots": [list(pair) for pair in self.known_prize_slots],
            "deck_count": self.deck_count,
            "prize_count": self.prize_count,
            "unknown_prize_slots": self.unknown_prize_slots,
            "unknown_non_deck_slots": self.unknown_non_deck_slots,
            "unaccounted_non_deck_slots": self.unaccounted_non_deck_slots,
            "select_deck_entries": [row.to_dict() for row in self.select_deck_entries],
            "looking_entries": [row.to_dict() for row in self.looking_entries],
            "select_deck_counts": [list(pair) for pair in self.select_deck_counts],
            "looking_counts": [list(pair) for pair in self.looking_counts],
            "prompt_exposure_scope": self.prompt_exposure_scope,
            "card_availability": [row.to_dict() for row in self.card_availability],
            "integrity_ok": self.integrity_ok,
            "fail_closed": self.fail_closed,
            "integrity_flags": list(self.integrity_flags),
            "scalar_vector": list(self.scalar_vector),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OwnDeckLedgerSnapshot:
        if value.get("schema") != OWN_DECK_LEDGER_SCHEMA or value.get("version") != OWN_DECK_LEDGER_SCHEMA_VERSION:
            raise OwnDeckLedgerError("own-deck ledger snapshot schema mismatch")
        zones_raw = value.get("visible_zone_counts")
        if not isinstance(zones_raw, Mapping):
            raise OwnDeckLedgerError("visible_zone_counts must be a mapping")
        zones = tuple(sorted((str(name), _pairs(_counter(rows))) for name, rows in zones_raw.items()))
        availability_raw = value.get("card_availability")
        if not isinstance(availability_raw, list):
            raise OwnDeckLedgerError("card_availability must be a list")
        cards = tuple(CardAvailability.from_dict(row) for row in availability_raw if isinstance(row, Mapping))
        if len(cards) != len(availability_raw):
            raise OwnDeckLedgerError("card_availability entry is invalid")
        select_raw = value.get("select_deck_entries") or []
        looking_raw = value.get("looking_entries") or []
        if not isinstance(select_raw, list) or not isinstance(looking_raw, list):
            raise OwnDeckLedgerError("prompt entries must be lists")
        flags = value.get("integrity_flags")
        if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
            raise OwnDeckLedgerError("integrity flags are invalid")
        revision = _exact_int(value.get("revision"))
        actor_raw = value.get("actor")
        actor = None if actor_raw is None else _exact_int(actor_raw)
        if revision is None or revision < 0 or actor not in (None, 0, 1):
            raise OwnDeckLedgerError("snapshot revision or actor is invalid")
        integer_fields: dict[str, int | None] = {}
        for name in ("deck_count", "prize_count", "unknown_prize_slots", "unknown_non_deck_slots", "unaccounted_non_deck_slots"):
            raw = value.get(name)
            integer_fields[name] = None if raw is None else _exact_int(raw)
            if integer_fields[name] is not None and integer_fields[name] < 0:
                raise OwnDeckLedgerError(f"{name} is invalid")
        if not isinstance(value.get("integrity_ok"), bool) or not isinstance(value.get("fail_closed"), bool):
            raise OwnDeckLedgerError("snapshot integrity flags are invalid")
        snapshot = cls(
            schema=OWN_DECK_LEDGER_SCHEMA,
            version=OWN_DECK_LEDGER_SCHEMA_VERSION,
            deck_fingerprint=str(value.get("deck_fingerprint")),
            observation_fingerprint=str(value.get("observation_fingerprint")),
            fingerprint=str(value.get("fingerprint")),
            revision=revision,
            actor=actor,
            starting_counts=_pairs(_counter(value.get("starting_counts") or [])),
            visible_zone_counts=zones,
            known_prize_slots=tuple(sorted((int(pair[0]), int(pair[1])) for pair in (value.get("known_prize_slots") or []))),
            deck_count=integer_fields["deck_count"],
            prize_count=integer_fields["prize_count"],
            unknown_prize_slots=integer_fields["unknown_prize_slots"],
            unknown_non_deck_slots=integer_fields["unknown_non_deck_slots"],
            unaccounted_non_deck_slots=integer_fields["unaccounted_non_deck_slots"],
            select_deck_entries=tuple(PromptCard.from_dict(row) for row in select_raw if isinstance(row, Mapping)),
            looking_entries=tuple(PromptCard.from_dict(row) for row in looking_raw if isinstance(row, Mapping)),
            select_deck_counts=_pairs(_counter(value.get("select_deck_counts") or [])),
            looking_counts=_pairs(_counter(value.get("looking_counts") or [])),
            prompt_exposure_scope=str(value.get("prompt_exposure_scope")),
            card_availability=cards,
            integrity_ok=value["integrity_ok"],
            fail_closed=value["fail_closed"],
            integrity_flags=tuple(flags),
        )
        # Fingerprints are integrity fields, not decoration.  Validate the
        # canonical payload after rebuilding it from JSON primitives.
        canonical = snapshot._fingerprint_payload()
        if snapshot.fingerprint != _digest(canonical):
            raise OwnDeckLedgerError("snapshot fingerprint mismatch")
        return snapshot

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "deck_fingerprint": self.deck_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "revision": self.revision,
            "actor": self.actor,
            "starting_counts": self.starting_counts,
            "visible_zone_counts": self.visible_zone_counts,
            "known_prize_slots": self.known_prize_slots,
            "deck_count": self.deck_count,
            "prize_count": self.prize_count,
            "unknown_prize_slots": self.unknown_prize_slots,
            "unknown_non_deck_slots": self.unknown_non_deck_slots,
            "unaccounted_non_deck_slots": self.unaccounted_non_deck_slots,
            "select_deck_entries": tuple((row.card_id, row.serial, row.index) for row in self.select_deck_entries),
            "looking_entries": tuple((row.card_id, row.serial, row.index) for row in self.looking_entries),
            "select_deck_counts": self.select_deck_counts,
            "looking_counts": self.looking_counts,
            "prompt_exposure_scope": self.prompt_exposure_scope,
            "card_availability": tuple((row.card_id, row.lower, row.upper, row.expected, row.probability_at_least_one, row.exact) for row in self.card_availability),
            "integrity_ok": self.integrity_ok,
            "fail_closed": self.fail_closed,
            "integrity_flags": self.integrity_flags,
        }


def _zone_rows(value: Any) -> list[Any]:
    rows = _as_rows(value)
    if rows is not None:
        return rows
    return [] if value is None else [value]


def _is_actor_owned_card(value: Any, actor: int | None, *, require_owner: bool = False) -> bool:
    """Whether a global-zone card is safely attributable to the acting seat.

    Player-local zones do not need this check. ``current.looking`` is normally
    scoped to the acting player, but production payloads may annotate cards
    with ``playerIndex``. Stadium is global, so it needs an explicit matching
    owner before it can enter own-deck accounting.
    """

    if actor not in (0, 1):
        return False
    raw_owner = _field(value, "playerIndex", "player_index")
    if raw_owner is None:
        return not require_owner
    return _exact_int(raw_owner) == actor


def _actor_owned_rows(
    value: Any,
    actor: int | None,
    *,
    require_owner: bool = False,
) -> list[Any]:
    return [
        row
        for row in _zone_rows(value)
        if _is_actor_owned_card(row, actor, require_owner=require_owner)
    ]


def _extract_zone(
    value: Any,
    *,
    seen_serials: dict[int, int],
    flags: list[str],
    label: str,
) -> tuple[Counter[int], int]:
    """Extract physical root/tool/energy cards, deduping documented serials."""

    counts: Counter[int] = Counter()
    unknown = 0

    def add(item: Any) -> None:
        nonlocal unknown
        if item is None:
            unknown += 1
            return
        ident = _card_id(item)
        if ident is None:
            unknown += 1
            flags.append(f"unresolved_visible_card:{label}")
            return
        serial = _card_serial(item)
        if serial is None:
            # Counts remain useful bounds, but without a stable identity we
            # cannot prove an alias has not appeared in a nested/menu surface.
            flags.append(f"unkeyed_visible_card:{label}")
            counts[ident] += 1
        else:
            prior = seen_serials.get(serial)
            if prior is not None:
                if prior != ident:
                    flags.append("serial_card_identity_conflict")
                return
            seen_serials[serial] = ident
            counts[ident] += 1
        # ``energies`` may be a scalar counter in cg payloads; it is not a
        # physical-card zone. ``preEvolution`` can be a singleton mapping.
        for attachment_name in ("tools", "toolCards", "energyCards", "preEvolution"):
            for attachment in _zone_rows(_field(item, attachment_name)):
                add(attachment)

    for row in _zone_rows(value):
        add(row)
    return counts, unknown


def _count_field(player: Any, names: Sequence[str]) -> int | None:
    for name in names:
        value = _exact_int(_field(player, name))
        if value is not None and value >= 0:
            return value
    return None


def _area_token(value: Any) -> str:
    number = _exact_int(value)
    if number is not None:
        return {
            1: "deck",
            2: "hand",
            3: "discard",
            4: "active",
            5: "bench",
            6: "prize",
            7: "stadium",
            12: "looking",
        }.get(number, "unknown")
    token = _enum_token(value)
    return {
        "deck": "deck",
        "hand": "hand",
        "discard": "discard",
        "active": "active",
        "bench": "bench",
        "prize": "prize",
        "stadium": "stadium",
        "looking": "looking",
    }.get(token, "unknown")


def _option_card_id(observation: Any, option: Any) -> int | None:
    """Resolve one option from only the actor's currently visible zones."""

    current = _field(observation, "current")
    select = _field(observation, "select")
    actor = _exact_int(_field(current, "yourIndex", "your_index"))
    if actor not in (0, 1):
        return None
    option_actor = _field(option, "playerIndex", "player_index")
    if option_actor is not None:
        declared = _exact_int(option_actor)
        if declared != actor:
            return None
    option_type = _enum_token(_field(option, "type"))
    area = "hand" if option_type == "play" else _area_token(_field(option, "area"))
    index = _exact_int(_field(option, "index"))
    if index is None or index < 0:
        return None
    players = _as_rows(_field(current, "players")) or []
    player = players[actor] if len(players) == 2 else None
    if area == "deck":
        zone = _zone_rows(_field(select, "deck"))
    elif area == "looking":
        zone = _zone_rows(_field(current, "looking"))
    elif area == "stadium":
        zone = _zone_rows(_field(current, "stadium"))
    elif player is not None:
        zone = _zone_rows(_field(player, area))
    else:
        zone = []
    if not 0 <= index < len(zone):
        return None
    host = zone[index]
    if area in {"deck", "looking"} and not _is_actor_owned_card(host, actor):
        return None
    if area == "stadium" and not _is_actor_owned_card(
        host, actor, require_owner=True
    ):
        return None
    if _area_token(_field(option, "area")) == "unknown" and option_type != "play":
        return None
    tool_index = _exact_int(_field(option, "toolIndex", "tool_index"))
    if tool_index is not None:
        tools = _zone_rows(_field(host, "tools", "toolCards"))
        if 0 <= tool_index < len(tools):
            return _card_id(tools[tool_index])
    energy_index = _exact_int(_field(option, "energyIndex", "energy_index"))
    if energy_index is not None:
        energies = _zone_rows(_field(host, "energyCards", "energies"))
        if 0 <= energy_index < len(energies):
            return _card_id(energies[energy_index])
    return _card_id(host)


class OwnDeckLedger:
    """Pure per-match own-deck state, intentionally dormant until staged wiring.

    The class is safe to construct in a collector, evaluator, or model-serving
    process.  ``observe`` is idempotent for the same actor-visible projection;
    ``fork`` is a standard deepcopy for speculative *local* use, but this core
    itself never invokes search or action execution.
    """

    def __init__(self, starting_deck: Iterable[int] | Mapping[int, int]) -> None:
        cards: list[int] = []
        if isinstance(starting_deck, Mapping):
            iterable: Iterable[Any] = (
                card_id
                for card_id, count in starting_deck.items()
                for _ in range(_validated_starting_count(card_id, count))
            )
        else:
            iterable = starting_deck
        for raw in iterable:
            card_id = _exact_int(raw)
            if card_id is None or card_id < 0:
                raise OwnDeckLedgerError(f"invalid starting deck card id: {raw!r}")
            cards.append(card_id)
        if not cards:
            raise OwnDeckLedgerError("starting deck must not be empty")
        self._starting = Counter(cards)
        self._deck_fingerprint = _digest({"starting_counts": _pairs(self._starting)})
        self.reset()

    @property
    def starting_counter(self) -> Counter[int]:
        return Counter(self._starting)

    @property
    def deck_fingerprint(self) -> str:
        return self._deck_fingerprint

    @property
    def snapshot(self) -> OwnDeckLedgerSnapshot | None:
        return self._snapshot

    def reset(self) -> None:
        """Forget only match state; the exact starting multiset remains fixed."""

        self._revision = 0
        self._snapshot: OwnDeckLedgerSnapshot | None = None
        self._last_observation_fingerprint: str | None = None
        self._known_prize_slots: dict[int, int] = {}
        self._last_prize_count: int | None = None

    def fork(self) -> OwnDeckLedger:
        return copy.deepcopy(self)

    def observe(self, observation: Any) -> OwnDeckLedgerSnapshot:
        """Consume one current actor-visible observation and return a snapshot."""

        parsed = self._parse(observation)
        observation_fingerprint = _digest(parsed["projection"])
        if self._snapshot is not None and observation_fingerprint == self._last_observation_fingerprint:
            return self._snapshot
        self._revision += 1
        snapshot = self._build_snapshot(parsed, observation_fingerprint)
        self._snapshot = snapshot
        self._last_observation_fingerprint = observation_fingerprint
        return snapshot

    def _parse(self, observation: Any) -> dict[str, Any]:
        flags: list[str] = []
        current = _field(observation, "current")
        select = _field(observation, "select")
        actor = _exact_int(_field(current, "yourIndex", "your_index"))
        players = _as_rows(_field(current, "players")) or []
        player = None
        if current is None:
            flags.append("missing_current")
        if actor not in (0, 1):
            flags.append("invalid_actor")
        elif len(players) != 2 or players[actor] is None:
            flags.append("missing_own_player")
        else:
            player = players[actor]

        # The projection deliberately excludes opponent player data, logs,
        # visualization/search payloads, hidden deck order, and transition_after.
        raw_zones: dict[str, Any] = {}
        if player is not None:
            for name in ("hand", "active", "bench", "discard", "prize"):
                raw_zones[name] = _field(player, name)
        select_deck = _field(select, "deck")
        looking = _field(current, "looking")
        stadium = _field(current, "stadium")
        projection = {
            "actor": actor,
            "zones": {name: _projection_zone(value) for name, value in raw_zones.items()},
            "deck_count": _count_field(player, ("deckCount", "deck_count")) if player is not None else None,
            "hand_count": _count_field(player, ("handCount", "hand_count")) if player is not None else None,
            "prize_count": _prize_count(player) if player is not None else None,
            # Global zones are projected only after actor ownership filtering;
            # opponent evidence must not perturb this ledger's fingerprint.
            "stadium": _projection_zone(stadium, actor=actor, require_owner=True),
            "select_deck": _projection_zone(select_deck, actor=actor),
            "looking": _projection_zone(looking, actor=actor),
            "options": _projection_options(_field(select, "option", "options")),
        }
        return {
            "flags": flags,
            "current": current,
            "select": select,
            "actor": actor if actor in (0, 1) else None,
            "player": player,
            "raw_zones": raw_zones,
            "select_deck": select_deck,
            "looking": looking,
            "stadium": stadium,
            "projection": projection,
        }

    def _build_snapshot(self, parsed: Mapping[str, Any], observation_fingerprint: str) -> OwnDeckLedgerSnapshot:
        flags = list(parsed["flags"])
        actor = parsed["actor"]
        player = parsed["player"]
        seen_serials: dict[int, int] = {}
        zone_counts: dict[str, Counter[int]] = {}
        zone_unknown: dict[str, int] = {}
        if player is not None:
            for name in ("hand", "active", "bench", "discard"):
                counts, unknown = _extract_zone(parsed["raw_zones"].get(name), seen_serials=seen_serials, flags=flags, label=name)
                # Own hand may be intentionally masked in a malformed surface;
                # handCount gives a conservative unknown-outside slot count.
                if name == "hand" and parsed["raw_zones"].get(name) is None:
                    hand_count = _count_field(player, ("handCount", "hand_count"))
                    if hand_count is not None:
                        unknown = max(unknown, hand_count)
                zone_counts[name] = counts
                zone_unknown[name] = unknown

        prize_count = _prize_count(player) if player is not None else None
        direct_prize_rows = _zone_rows(parsed["raw_zones"].get("prize")) if player is not None else []
        if prize_count is not None and self._last_prize_count is not None and prize_count != self._last_prize_count:
            # Slot identity is not safe across a shrinking/reordered prize list.
            self._known_prize_slots.clear()
        direct_prize: dict[int, int] = {}
        # Build current direct evidence before mutating remembered slots.  A
        # stable list length alone does not prove slot identity if a later
        # observation reveals a card at a previously different/unknown slot.
        for slot, card in enumerate(direct_prize_rows):
            ident = _card_id(card)
            if ident is not None:
                direct_prize[slot] = ident
        if direct_prize and self._known_prize_slots and any(
            self._known_prize_slots.get(slot) != ident
            for slot, ident in direct_prize.items()
        ):
            # Current direct evidence is authoritative.  Invalidate rather
            # than guessing whether an unseen old slot survived a reordering.
            self._known_prize_slots.clear()
            flags.append("prize_history_invalidated")

        direct_prize_is_new_physical: dict[int, bool] = {}
        for slot, card in enumerate(direct_prize_rows):
            ident = _card_id(card)
            if ident is None:
                continue
            serial = _card_serial(card)
            is_new_physical = True
            if serial is None:
                flags.append("unkeyed_visible_card:prize")
            else:
                prior_serial_id = seen_serials.get(serial)
                if prior_serial_id is not None:
                    is_new_physical = False
                    if prior_serial_id != ident:
                        flags.append("serial_card_identity_conflict")
                else:
                    seen_serials[serial] = ident
            self._known_prize_slots[slot] = ident
            direct_prize_is_new_physical[slot] = is_new_physical
        if prize_count is not None:
            self._known_prize_slots = {
                slot: ident for slot, ident in self._known_prize_slots.items() if 0 <= slot < prize_count
            }
            self._last_prize_count = prize_count
        elif player is not None:
            flags.append("missing_prize_count")

        prize_counter: Counter[int] = Counter()
        known_prize_slots: dict[int, int] = {}
        if prize_count is not None:
            for slot in range(prize_count):
                direct = direct_prize.get(slot)
                remembered = self._known_prize_slots.get(slot)
                ident = direct if direct is not None else remembered
                if ident is not None:
                    # A current prize object can be an alias of another
                    # currently visible physical card. Count that serial only
                    # once; a masked remembered prize still reserves one card.
                    if direct is None or direct_prize_is_new_physical.get(slot, True):
                        prize_counter[ident] += 1
                    known_prize_slots[slot] = ident
        zone_counts["prize"] = prize_counter
        unknown_prize_slots = None if prize_count is None else max(0, prize_count - len(known_prize_slots))

        # Stadium is global.  Only a card whose explicit playerIndex names the
        # actor is safe own-deck evidence; silently treating an unowned global
        # card as ours would leak opponent state into deterministic accounting.
        raw_stadium = _zone_rows(parsed.get("stadium"))
        owned_stadium = _actor_owned_rows(parsed.get("stadium"), actor, require_owner=True)
        # A foreign stadium is opponent state and must be a complete no-op for
        # this ledger. An ownerless global stadium is ambiguous rather than
        # foreign, so surface that conservative omission for auditability.
        if any(_field(card, "playerIndex", "player_index") is None for card in raw_stadium):
            flags.append("unowned_stadium_ignored")
        stadium_counts, stadium_unknown = _extract_zone(
            owned_stadium,
            seen_serials=seen_serials,
            flags=flags,
            label="stadium",
        )
        zone_counts["stadium"] = stadium_counts
        zone_unknown["stadium"] = stadium_unknown

        # `looking` is a real transit zone, not a deck menu.  Its physical
        # cards are outside the current draw pile and therefore subtract from
        # the residual exactly once.  A select.deck alias with the same serial
        # stays menu-local below and does not subtract a second time.
        owned_looking = _actor_owned_rows(parsed.get("looking"), actor)
        looking_visible_counts, looking_visible_unknown = _extract_zone(
            owned_looking,
            seen_serials=seen_serials,
            flags=flags,
            label="looking",
        )
        zone_counts["looking"] = looking_visible_counts
        zone_unknown["looking"] = looking_visible_unknown

        select_entries, select_unknown = _prompt_entries(
            parsed["select_deck"], flags, "select_deck", actor=actor
        )
        looking_entries, looking_prompt_unknown = _prompt_entries(
            parsed["looking"], flags, "looking", actor=actor
        )
        select_counts = Counter(row.card_id for row in select_entries)
        looking_counts = Counter(row.card_id for row in looking_entries)

        deck_count = _count_field(player, ("deckCount", "deck_count")) if player is not None else None
        if deck_count is None:
            flags.append("missing_deck_count")

        visible = Counter()
        for counts in zone_counts.values():
            visible.update(counts)
        for card_id, count in visible.items():
            if card_id not in self._starting:
                flags.append("visible_card_not_in_starting_deck")
            elif count > self._starting[card_id]:
                flags.append("visible_card_count_exceeds_starting")
        residual = Counter()
        for card_id, count in self._starting.items():
            residual[card_id] = max(0, count - visible.get(card_id, 0))

        select_evidence, select_aliases_visible = _select_deck_evidence(
            select_entries,
            visible_serials=seen_serials,
            flags=flags,
        )
        for card_id in select_counts:
            if card_id not in self._starting:
                flags.append("select_card_not_in_starting_deck")
        for card_id, count in select_evidence.items():
            if count > residual.get(card_id, 0):
                flags.append("select_card_count_exceeds_possible")
        # Looking is already in `visible` and hence absent from `residual`.
        # Check only its source identity here; duplicate menu aliases must not
        # be mistaken for extra physical cards.
        for card_id in looking_counts:
            if card_id not in self._starting:
                flags.append("looking_card_not_in_starting_deck")
        if select_unknown:
            flags.append("unresolved_select_deck_card")
        if looking_prompt_unknown:
            flags.append("unresolved_looking_card")

        total_residual = sum(residual.values())
        unknown_non_deck_slots: int | None = None
        unaccounted_non_deck_slots: int | None = None
        unknown_visible_non_prize = sum(
            zone_unknown.get(name, 0)
            for name in ("hand", "active", "bench", "discard", "stadium", "looking")
        )
        if deck_count is not None:
            inferred = total_residual - deck_count
            if inferred < 0:
                flags.append("deck_count_exceeds_residual")
            else:
                unknown_non_deck_slots = inferred
                required_known_unknown = unknown_visible_non_prize + int(unknown_prize_slots or 0)
                if inferred < required_known_unknown:
                    flags.append("unknown_slots_exceed_conservation")
                else:
                    unaccounted_non_deck_slots = inferred - required_known_unknown

        full_select = (
            deck_count is not None
            and not select_unknown
            and not select_aliases_visible
            and bool(select_entries)
            and sum(select_evidence.values()) == deck_count
        )
        if full_select:
            prompt_scope = "select_deck_full_by_count"
        elif select_entries and select_aliases_visible:
            prompt_scope = "select_deck_candidates_aliasing_visible"
        elif select_entries:
            prompt_scope = "select_deck_candidates"
        elif looking_entries:
            prompt_scope = "looking_candidates_unknown_origin"
        else:
            prompt_scope = "none"

        hard_flags = {
            "missing_current",
            "invalid_actor",
            "missing_own_player",
            "missing_deck_count",
            "missing_prize_count",
            "visible_card_not_in_starting_deck",
            "visible_card_count_exceeds_starting",
            "deck_count_exceeds_residual",
            "unknown_slots_exceed_conservation",
            "prize_slot_identity_conflict",
            "serial_card_identity_conflict",
            "select_card_not_in_starting_deck",
            "select_card_count_exceeds_possible",
            "looking_card_not_in_starting_deck",
            "unresolved_select_deck_card",
            "unresolved_looking_card",
        }
        # A row without a physical serial can remain in the serialized prompt,
        # but it cannot safely be merged with other current zones.  Keep the
        # raw row for auditability while fail-closing model-facing bounds.
        fail_closed = any(
            flag in hard_flags
            or flag.startswith(("unkeyed_", "unresolved_visible_card:"))
            for flag in flags
        )
        integrity_ok = not fail_closed

        availability: list[CardAvailability] = []
        for card_id in sorted(self._starting):
            remaining = residual.get(card_id, 0)
            if full_select and integrity_ok:
                exact_count = select_evidence.get(card_id, 0)
                availability.append(CardAvailability(card_id, exact_count, exact_count, float(exact_count), 1.0 if exact_count else 0.0, True))
                continue
            if deck_count is None or unknown_non_deck_slots is None or fail_closed:
                availability.append(CardAvailability(card_id, 0, remaining, None, None, False))
                continue
            lower = max(0, remaining - unknown_non_deck_slots)
            upper = min(remaining, deck_count)
            known_select = select_evidence.get(card_id, 0)
            if known_select > upper:
                availability.append(CardAvailability(card_id, 0, remaining, None, None, False))
                continue
            lower = max(lower, known_select)
            known_total = sum(select_evidence.values())
            remaining_total = total_residual - known_total
            remaining_deck = deck_count - known_total
            remaining_card = remaining - known_select
            if remaining_total < 0 or remaining_deck < 0 or remaining_card < 0:
                expected = probability = None
            else:
                expected = float(known_select) + (
                    float(remaining_card * remaining_deck) / float(remaining_total)
                    if remaining_total
                    else 0.0
                )
                probability = 1.0 if known_select else _hypergeom_probability_at_least_one(remaining_total, remaining_card, remaining_deck)
            availability.append(CardAvailability(card_id, lower, upper, expected, probability, lower == upper))

        snapshot_without_fingerprint = OwnDeckLedgerSnapshot(
            schema=OWN_DECK_LEDGER_SCHEMA,
            version=OWN_DECK_LEDGER_SCHEMA_VERSION,
            deck_fingerprint=self._deck_fingerprint,
            observation_fingerprint=observation_fingerprint,
            fingerprint="",
            revision=self._revision,
            actor=actor,
            starting_counts=_pairs(self._starting),
            visible_zone_counts=tuple(sorted((name, _pairs(counts)) for name, counts in zone_counts.items())),
            known_prize_slots=tuple(sorted(known_prize_slots.items())),
            deck_count=deck_count,
            prize_count=prize_count,
            unknown_prize_slots=unknown_prize_slots,
            unknown_non_deck_slots=unknown_non_deck_slots,
            unaccounted_non_deck_slots=unaccounted_non_deck_slots,
            select_deck_entries=tuple(select_entries),
            looking_entries=tuple(looking_entries),
            select_deck_counts=_pairs(select_counts),
            looking_counts=_pairs(looking_counts),
            prompt_exposure_scope=prompt_scope,
            card_availability=tuple(availability),
            integrity_ok=integrity_ok,
            fail_closed=fail_closed,
            integrity_flags=tuple(sorted(set(flags))),
        )
        fingerprint = _digest(snapshot_without_fingerprint._fingerprint_payload())
        return OwnDeckLedgerSnapshot(
            **{
                **snapshot_without_fingerprint.__dict__,
                "fingerprint": fingerprint,
            }
        )


def _prize_count(player: Any) -> int | None:
    prize = _field(player, "prize")
    rows = _as_rows(prize)
    if rows is not None:
        return len(rows)
    return _count_field(player, ("prizeCount", "prize_count", "remainingPrizes"))


def _validated_starting_count(card_id: Any, count: Any) -> int:
    ident = _exact_int(card_id)
    parsed = _exact_int(count)
    if ident is None or ident < 0 or parsed is None or parsed <= 0:
        raise OwnDeckLedgerError("starting deck mapping contains an invalid card/count")
    return parsed


def _prompt_entries(
    value: Any,
    flags: list[str],
    label: str,
    *,
    actor: int | None = None,
) -> tuple[list[PromptCard], int]:
    entries: list[PromptCard] = []
    unknown = 0
    for index, card in enumerate(_zone_rows(value)):
        if actor is not None and not _is_actor_owned_card(card, actor):
            continue
        ident = _card_id(card)
        if ident is None:
            unknown += 1
            continue
        serial = _card_serial(card)
        if serial is None:
            flags.append(f"unkeyed_{label}_card")
        entries.append(PromptCard(ident, serial, index))
    return entries, unknown


def _select_deck_evidence(
    entries: Sequence[PromptCard],
    *,
    visible_serials: Mapping[int, int],
    flags: list[str],
) -> tuple[Counter[int], bool]:
    """Return unique deck-membership evidence without counting transit aliases.

    ``select.deck`` is a prompt-local serialization of current deck candidates,
    not an additional physical zone.  The same card may also be exposed in
    ``current.looking``.  In that case the physical card was already removed
    from the draw-pile residual by the looking zone, so it cannot simultaneously
    tighten a deck bound.  Menu multiplicity remains available through
    ``select_deck_entries``/``select_deck_counts``; this helper is only for
    conservative accounting.
    """

    evidence: Counter[int] = Counter()
    menu_serials: dict[int, int] = {}
    aliases_visible = False
    for entry in entries:
        if entry.serial is None:
            # The prompt row is retained but lacks a physical identity, so it
            # cannot prove an additional deck copy.
            continue
        visible_id = visible_serials.get(entry.serial)
        if visible_id is not None:
            if visible_id != entry.card_id:
                flags.append("serial_card_identity_conflict")
            else:
                aliases_visible = True
            continue
        prior = menu_serials.get(entry.serial)
        if prior is not None:
            if prior != entry.card_id:
                flags.append("serial_card_identity_conflict")
            # Repeated rendering of one physical menu card is not a second
            # membership proof.
            aliases_visible = True
            continue
        menu_serials[entry.serial] = entry.card_id
        evidence[entry.card_id] += 1
    return evidence, aliases_visible


def _projection_zone(
    value: Any,
    *,
    actor: int | None = None,
    require_owner: bool = False,
) -> list[dict[str, Any]]:
    return [
        _projection_card(row)
        for row in _zone_rows(value)
        if actor is None or _is_actor_owned_card(row, actor, require_owner=require_owner)
    ]


def _projection_card(value: Any) -> dict[str, Any]:
    """Canonicalize only physical card fields that affect residual accounting."""

    result: dict[str, Any] = {"id": _card_id(value), "serial": _card_serial(value)}
    for name in ("tools", "toolCards", "energyCards", "preEvolution"):
        nested = _field(value, name)
        if nested is not None:
            result[name] = [_projection_card(row) for row in _zone_rows(nested)]
    return result


def _projection_options(value: Any) -> list[dict[str, Any]]:
    options = _as_rows(value) or []
    rows: list[dict[str, Any]] = []
    for option in options:
        rows.append(
            {
                "type": _enum_token(_field(option, "type")),
                "area": _area_token(_field(option, "area")),
                "index": _exact_int(_field(option, "index")),
                "player": _exact_int(_field(option, "playerIndex", "player_index")),
                "tool": _exact_int(_field(option, "toolIndex", "tool_index")),
                "energy": _exact_int(_field(option, "energyIndex", "energy_index")),
            }
        )
    return rows


__all__ = [
    "OPTION_FEATURE_DIM",
    "OPTION_FEATURE_NAMES",
    "OWN_DECK_LEDGER_SCHEMA",
    "OWN_DECK_LEDGER_SCHEMA_VERSION",
    "SCALAR_VECTOR_NAMES",
    "CardAvailability",
    "OwnDeckLedger",
    "OwnDeckLedgerError",
    "OwnDeckLedgerSnapshot",
    "PromptCard",
]
