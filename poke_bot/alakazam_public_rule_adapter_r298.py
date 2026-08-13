"""Versioned public-rule representation for the isolated Alakazam r298 study.

The existing r195/r241/r274 model remains immutable.  This module is a
parallel *representation* and optional zero-gated metadata residual; nothing
imports it from the live policy path.  It has four deliberately narrow jobs:

* project a raw simulator observation into the acting player's information set;
* describe the public state and complete legal-option semantics without using
  raw candidate ordinals or global physical-card serials;
* make public structured card/attack metadata available to a future derivative
  through an exactly-off residual; and
* provide deterministic hashes useful to the r298 collision census.

The competition simulator remains the rules authority.  In particular this
module does not infer attack legality, parse card text, manufacture successor
states, or select an action.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# The semantic encoder is deliberately usable on the CPU-only audit host.  The
# optional residual needs torch, but importing the public-information boundary
# must not make Phase A/Phase 5 tests depend on a training runtime.
try:  # pragma: no cover - both branches are exercised on different hosts.
    import torch
    import torch.nn as nn
    from torch import Tensor
except ModuleNotFoundError:  # lightweight Elmo/audit environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

PUBLIC_RULE_ADAPTER_SCHEMA = "poke_bot.alakazam_public_rule_adapter/v1"
PUBLIC_RULE_REPRESENTATION_SCHEMA = "poke_bot.alakazam_public_rule_representation/v1"
PUBLIC_RULE_CONFIG_SCHEMA = "poke_bot.alakazam_public_rule_adapter_config/v1"
PUBLIC_CATALOG_SCHEMA = "poke_bot.alakazam_public_catalog_r298/v1"
PUBLIC_STRUCTURED_VECTORS_SCHEMA = "poke_bot.alakazam_structured_rule_vectors_r298/v1"
PUBLIC_CATALOG_RECEIPT_SCHEMA = "poke_bot.alakazam_public_catalog_r298_receipt/v1"
PUBLIC_RULE_ADAPTER_REVISION = 298
R298_CANONICAL_GOAL_REVISION = 5
R298_ROOT_OWNER_REVISION = 303
R298_PREDECESSOR_GOAL_REVISION = 4

# The revision-5 gateway/contract are the live authority for this isolated
# derivative.  The structured catalog was sealed under revision 4 and is
# immutable predecessor evidence only: revision 5 may reuse its exact bytes,
# but a consumer still needs the separate revision-5 schema-freeze receipt
# before it can claim r5 readiness.  Keep the two identities separate rather
# than silently rebinding the historical artifact.
R298_CANONICAL_GOAL_SHA256 = (
    "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6"
)
R298_CANONICAL_CONTRACT_SHA256 = (
    "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891"
)
R298_PREDECESSOR_GOAL_SHA256 = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)
R298_PREDECESSOR_CONTRACT_SHA256 = (
    "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
)
R298_CANONICAL_LIBCG_SHA256 = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)
R298_CANONICAL_LIBCG_SIZE_BYTES = 1_342_400
R298_CANONICAL_LIBCG_TYPED_SOURCE_SHA256 = (
    "sha256:d75ff752808ead08f3ae20f7f2f8a034c9e6163109188a46d3b877bf1910ae2d"
)
R298_PUBLIC_CATALOG_CSV_SHA256 = (
    "sha256:408bc978661c8b0628e5f17b27693dc8da9c732472168f5574999be4774031c1"
)
R298_PUBLIC_CATALOG_ENGINE_CARDS_SHA256 = (
    "sha256:5e92642577d5e61324e4d4095b883fb7926e4d7e822967b48778ecd1996998f4"
)
R298_PUBLIC_CATALOG_ENGINE_ATTACKS_SHA256 = (
    "sha256:97834eb17429fbacfedcf563d43753b129103c0a1d1941d1ee9c9af36392be3f"
)
R298_PUBLIC_CATALOG_FILE_SHA256 = (
    "sha256:4d1c35124cdeeddcaca34a7d0ab3f2fc94e4257fe4578a03c8608ac561d00df6"
)
R298_PUBLIC_CATALOG_VECTORS_FILE_SHA256 = (
    "sha256:2e1a817dac1d17d0131056ead9669b76803b41f5864d4e9cd82d7f34a9180e09"
)
R298_PUBLIC_CATALOG_RECEIPT_FILE_SHA256 = (
    "sha256:9ca7534281dd08d1804babe223fa0779a84a1f5a7ee6f609497d595dca10b4d1"
)
R298_PUBLIC_CATALOG_STRUCTURED_VECTORS_SHA256 = (
    "sha256:6602af831128364a7a33ea3e691a73e4cac7e39ad037855ada7b13c80f87ebb1"
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "policy_layers"
    / "alakazam-public-rule-adapter-r298.json"
)

# These are representational safety limits, not inferred game limits.  A
# changed simulator shape fails closed rather than folding distinct values into
# one feature bucket.
MAX_NUMBER_VALUE = 64
MAX_SELECTION_COUNT = 60
MAX_OPTION_COUNT = 60
MAX_SLOT_INDEX = 63
MAX_HP_VALUE = 4096
MAX_BENCH_MAXIMUM = 16
MAX_TURN_ACTION_COUNT = 255
MAX_CARD_ID = 16384
MAX_ATTACK_ID = 16384

# The public-information boundary is deliberately an allowlist, rather than a
# growing denylist.  Raw visual/search dumps have acquired extra root and
# PlayerState fields over time; copying an unrecognised field into a public
# fingerprint would make a hidden implementation payload observable to a
# future consumer even if the current representation happens to ignore it.
# These sets list only the r236 API fields used by this isolated adapter.
_PUBLIC_CURRENT_SCALAR_FIELDS = (
    "turn",
    "turnActionCount",
    "yourIndex",
    "firstPlayer",
    "supporterPlayed",
    "stadiumPlayed",
    "energyAttached",
    "retreated",
)
_PUBLIC_PLAYER_SCALAR_FIELDS = (
    "benchMax",
    "deckCount",
    "handCount",
    "prizeCount",
    "prize_count",
    "remainingPrizes",
    "asleep",
    "burned",
    "confused",
    "paralyzed",
    "poisoned",
)
_PUBLIC_CARD_SCALAR_FIELDS = (
    "id",
    "cardId",
    "playerIndex",
    "hp",
    "currentHp",
    "current_hp",
    "maxHp",
    "max_hp",
    "appearThisTurn",
    "appear_this_turn",
    "energyType",
    "energy_type",
    "pokemonType",
    "type",
)
_PUBLIC_CARD_ENERGY_FIELDS = (
    "typedEnergyUnits",
    "typed_energy_units",
    "energyTypes",
    "energy_types",
    "providesEnergy",
    "provides_energy",
)
_PUBLIC_OPTION_FIELDS = (
    "type",
    "number",
    "count",
    "attackId",
    "cardId",
    "area",
    "index",
    "playerIndex",
    "toolIndex",
    "tool_index",
    "energyIndex",
    "energy_index",
    "inPlayArea",
    "inPlayIndex",
    "inPlayPlayerIndex",
    "specialConditionType",
    "special_condition_type",
    "sourcePlayerIndex",
    "sourceArea",
    "sourceIndex",
    "sourceCardId",
    "targetPlayerIndex",
    "targetArea",
    "targetIndex",
    "targetCardId",
    "toolCardId",
    "tool_card_id",
    "energyCardId",
    "energy_card_id",
    "skillId",
    "skill_id",
    "simulatorDiscriminator",
    "simulator_discriminator",
    "semanticDiscriminator",
    "semantic_discriminator",
    "legalActionDiscriminator",
    "legal_action_discriminator",
)
_PUBLIC_SELECT_SCALAR_FIELDS = (
    "type",
    "context",
    "minCount",
    "maxCount",
    "remainDamageCounter",
    "remainingDamageCounter",
    "remainEnergyCost",
    "remainingEnergyCost",
    "optionOrderSemantic",
)
_PUBLIC_VISIBLE_MODIFIER_FIELDS = (
    "exact_yield",
    "exactYield",
    "yield",
    "delta",
    "prize_delta",
    "prizeDelta",
    "reduction",
    "prize_reduction",
    "prizeReduction",
)

# These zones are visible multisets rather than physical board coordinates.
# Their display order may be rewritten while the same legal card choices
# remain available, so a raw menu/list index is not a stable policy semantic.
# The simulator's candidate-row order is retained by the caller for execution
# alignment; this representation hashes only the visible card
# identity/multiplicity and any explicit simulator discriminator.  Active and
# Bench deliberately remain outside this set: their slots are board positions.
_ORDERLESS_MENU_AREAS = frozenset({"deck", "looking", "hand", "discard"})

_AREA_NAMES = {
    1: "deck",
    2: "hand",
    3: "discard",
    4: "active",
    5: "bench",
    6: "prize",
    7: "stadium",
    8: "energy",
    9: "tool",
    10: "pre_evolution",
    11: "player",
    12: "looking",
}
_AREA_ALIASES = {
    "preevolution": "pre_evolution",
    "pre_evolution": "pre_evolution",
    "preevolutions": "pre_evolution",
    "toolcard": "tool",
    "energycard": "energy",
}
_OPTION_NAMES = {
    0: "number",
    1: "yes",
    2: "no",
    3: "card",
    4: "tool_card",
    5: "energy_card",
    6: "energy",
    7: "play",
    8: "attach",
    9: "evolve",
    10: "ability",
    11: "discard",
    12: "retreat",
    13: "attack",
    14: "end",
    15: "skill",
    16: "special_condition",
}
_OPTION_ALIASES = {
    "toolcard": "tool_card",
    "energycard": "energy_card",
    "specialcondition": "special_condition",
}
_CONTEXT_NAMES = {0: "main", 41: "is_first", 46: "coin_head"}
_CONTEXT_ALIASES = {
    "isfirst": "is_first",
    "coinhead": "coin_head",
}


class PublicRuleAdapterError(ValueError):
    """The public-rule schema cannot exactly represent an input."""


@dataclass(frozen=True)
class PublicCatalogPins:
    """Immutable identities required for a trusted r298 catalog artifact.

    The pins are intentionally file-level as well as semantic-level: an
    arbitrary mapping that happens to contain similar structured values cannot
    become mechanics authority.  Tests can pass a separately constructed pin
    set to the loader, but the normal policy/materializer boundary always uses
    :data:`DEFAULT_PUBLIC_CATALOG_PINS`.
    """

    goal_sha256: str
    contract_sha256: str
    libcg_sha256: str
    libcg_size_bytes: int
    csv_sha256: str
    engine_cards_sha256: str
    engine_attacks_sha256: str
    catalog_file_sha256: str
    vectors_file_sha256: str
    receipt_file_sha256: str
    structured_vectors_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_sha256": self.goal_sha256,
            "contract_sha256": self.contract_sha256,
            "libcg_sha256": self.libcg_sha256,
            "libcg_size_bytes": self.libcg_size_bytes,
            "csv_sha256": self.csv_sha256,
            "engine_cards_sha256": self.engine_cards_sha256,
            "engine_attacks_sha256": self.engine_attacks_sha256,
            "catalog_file_sha256": self.catalog_file_sha256,
            "vectors_file_sha256": self.vectors_file_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "structured_vectors_sha256": self.structured_vectors_sha256,
        }


# This pin set intentionally verifies the immutable revision-4 artifact.  It
# is not an r5 activation/eligibility receipt and must never be relabelled as
# one merely because the structured catalog bytes are reused unchanged.
DEFAULT_PUBLIC_CATALOG_PINS = PublicCatalogPins(
    goal_sha256=R298_PREDECESSOR_GOAL_SHA256,
    contract_sha256=R298_PREDECESSOR_CONTRACT_SHA256,
    libcg_sha256=R298_CANONICAL_LIBCG_SHA256,
    libcg_size_bytes=R298_CANONICAL_LIBCG_SIZE_BYTES,
    csv_sha256=R298_PUBLIC_CATALOG_CSV_SHA256,
    engine_cards_sha256=R298_PUBLIC_CATALOG_ENGINE_CARDS_SHA256,
    engine_attacks_sha256=R298_PUBLIC_CATALOG_ENGINE_ATTACKS_SHA256,
    catalog_file_sha256=R298_PUBLIC_CATALOG_FILE_SHA256,
    vectors_file_sha256=R298_PUBLIC_CATALOG_VECTORS_FILE_SHA256,
    receipt_file_sha256=R298_PUBLIC_CATALOG_RECEIPT_FILE_SHA256,
    structured_vectors_sha256=R298_PUBLIC_CATALOG_STRUCTURED_VECTORS_SHA256,
)


@dataclass(frozen=True)
class PublicCatalogProvenance:
    """The auditable availability state for one structured catalog.

    ``eligible`` is intentionally stricter than "the JSON parsed": all
    receipt, binary, catalog, and vector identities must agree before any
    mechanics-dependent r298 feature can be available.
    """

    eligible: bool
    reason: Optional[str] = None
    catalog_file_sha256: Optional[str] = None
    receipt_file_sha256: Optional[str] = None
    vectors_file_sha256: Optional[str] = None
    catalog_semantic_sha256: Optional[str] = None
    structured_vectors_sha256: Optional[str] = None
    engine_cards_sha256: Optional[str] = None
    engine_attacks_sha256: Optional[str] = None
    test_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "available": self.eligible,
            "reason": self.reason,
            "catalog_file_sha256": self.catalog_file_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "vectors_file_sha256": self.vectors_file_sha256,
            "catalog_semantic_sha256": self.catalog_semantic_sha256,
            "structured_vectors_sha256": self.structured_vectors_sha256,
            "engine_cards_sha256": self.engine_cards_sha256,
            "engine_attacks_sha256": self.engine_attacks_sha256,
            "test_only": self.test_only,
        }


@dataclass(frozen=True)
class SealedPublicCatalog:
    """Receipt-bound structured cards/attacks with no text mechanics path."""

    cards: tuple[Mapping[str, Any], ...]
    attacks: tuple[Mapping[str, Any], ...]
    provenance: PublicCatalogProvenance
    pins: PublicCatalogPins
    catalog_path: Path
    receipt_path: Path
    vectors_path: Path

    @property
    def card_vocab(self) -> int:
        """ID-indexed vector vocabulary implied by the sealed card records."""

        return max(
            (_card_id(row, field="sealed catalog.cardId", optional=False) or 0)
            for row in self.cards
        ) + 1

    @property
    def attack_vocab(self) -> int:
        """ID-indexed vector vocabulary implied by the sealed attack records."""

        return max(
            (_attack_id(row.get("attackId"), field="sealed catalog.attackId", optional=False) or 0)
            for row in self.attacks
        ) + 1


@dataclass(frozen=True)
class SealedPublicCatalogVectors:
    """Validated structured-only vector tensors for the additive residual."""

    card_features: Any
    attack_features: Any
    provenance: PublicCatalogProvenance
    vector_schema: str


def _require_torch() -> None:
    if torch is None or nn is None:
        raise PublicRuleAdapterError(
            "the optional public-rule metadata residual requires torch; "
            "the public semantic encoder itself does not"
        )


def _canonical_json(value: Any) -> str:
    """Canonical JSON for semantic equality and content addressing."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _norm_token(value: Any) -> str:
    if hasattr(value, "name"):
        value = getattr(value, "name")
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _exact_int(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    optional: bool = False,
) -> Optional[int]:
    if value is None:
        if optional:
            return None
        raise PublicRuleAdapterError(f"{field} is required")
    if isinstance(value, bool):
        raise PublicRuleAdapterError(f"{field} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PublicRuleAdapterError(f"{field} is not an exact integer: {value!r}") from exc
    try:
        exact = bool(value == result)
    except Exception:  # pragma: no cover - foreign scalar defensive path
        exact = False
    if not exact:
        raise PublicRuleAdapterError(f"{field} is not an exact integer: {value!r}")
    if minimum is not None and result < minimum:
        raise PublicRuleAdapterError(f"{field}={result} is below {minimum}")
    if maximum is not None and result > maximum:
        raise PublicRuleAdapterError(f"{field}={result} exceeds {maximum}")
    return result


def _optional_bool(value: Any, *, field: str) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    # Native visual rows have historically represented booleans as 0/1.  They
    # are accepted only when exact, never through truthiness coercion.
    numeric = _exact_int(value, field=field, minimum=0, maximum=1)
    return bool(numeric)


def _enum_name(
    value: Any,
    *,
    field: str,
    numeric_names: Mapping[int, str],
    aliases: Mapping[str, str] = {},
    optional: bool = True,
) -> Optional[str]:
    if value is None:
        if optional:
            return None
        raise PublicRuleAdapterError(f"{field} is required")
    if isinstance(value, bool):
        raise PublicRuleAdapterError(f"{field} cannot be bool")
    if isinstance(value, str) or hasattr(value, "name"):
        token = _norm_token(value)
        if not token:
            raise PublicRuleAdapterError(f"{field} is empty")
        for number, name in numeric_names.items():
            if _norm_token(name) == token:
                return name
        if token in aliases:
            return aliases[token]
        # Unknown enum names are not silently projected to an OOV row.  They
        # remain explicit stable semantic atoms for the collision census.
        return f"{field}:{token}"
    number = _exact_int(value, field=field)
    return numeric_names.get(number, f"{field}:{number}")


def _area(value: Any, *, field: str, optional: bool = True) -> Optional[str]:
    return _enum_name(
        value,
        field=field,
        numeric_names=_AREA_NAMES,
        aliases=_AREA_ALIASES,
        optional=optional,
    )


def _is_unrevealed_prize_area(area: Optional[str]) -> bool:
    """Whether an area has no card-identity ABI in the r298 public surface."""

    # There is deliberately no inferred "revealed Prize" exception.  Until a
    # separately versioned, provenance-bound public ABI exists, every Prize
    # card identity is treated as hidden even for the acting player.
    return area == "prize"


def _reference_area(value: Mapping[str, Any], *, field: str) -> Optional[str]:
    """Read a card-reference area when the raw ABI supplies one.

    This is intentionally narrow and only recognizes the same field aliases
    used for option source/target bindings.  It lets sanitizer and semantic
    helpers suppress an explicitly Prize-scoped card ID without guessing about
    a context card that has no location information at all.
    """

    return _area(
        _first_present(value, ("area", "sourceArea", "targetArea", "inPlayArea")),
        field=field,
    )


def _option_type(value: Any) -> str:
    result = _enum_name(
        value,
        field="option_type",
        numeric_names=_OPTION_NAMES,
        aliases=_OPTION_ALIASES,
        optional=False,
    )
    assert result is not None
    return result


def _context(value: Any) -> Optional[str]:
    return _enum_name(
        value,
        field="select_context",
        numeric_names=_CONTEXT_NAMES,
        aliases=_CONTEXT_ALIASES,
        optional=True,
    )


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicRuleAdapterError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str, optional: bool = True) -> list[Any]:
    if value is None and optional:
        return []
    if not isinstance(value, (list, tuple)):
        raise PublicRuleAdapterError(f"{field} must be a list")
    return list(value)


def _card_id(value: Any, *, field: str, optional: bool = True) -> Optional[int]:
    if value is None:
        if optional:
            return None
        raise PublicRuleAdapterError(f"{field} is required")
    raw = value
    if isinstance(value, Mapping):
        if "id" in value:
            raw = value.get("id")
        elif "cardId" in value:
            raw = value.get("cardId")
        else:
            if optional:
                return None
            raise PublicRuleAdapterError(f"{field} has no card id")
    return _exact_int(raw, field=field, minimum=0, maximum=MAX_CARD_ID)


def _attack_id(value: Any, *, field: str, optional: bool = True) -> Optional[int]:
    return _exact_int(
        value,
        field=field,
        minimum=0,
        maximum=MAX_ATTACK_ID,
        optional=optional,
    )


def _serial(value: Any, *, field: str) -> Optional[int]:
    return _exact_int(value, field=field, minimum=0, optional=True)


def _safe_json_value(value: Any, *, field: str) -> Any:
    """Normalize structured simulator metadata without using card text.

    This intentionally supports only JSON-shaped structured mechanics.  Card
    names and text are never treated as a rules parser or converted into a
    synthetic text hash here.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PublicRuleAdapterError(f"{field} contains a non-finite float")
        return value
    if hasattr(value, "name"):
        return _norm_token(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_json_value(item, field=f"{field}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in {"text", "name", "description", "effectExplanation"}
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_json_value(item, field=f"{field}[]")
            for item in value
        ]
    raise PublicRuleAdapterError(
        f"{field} contains unsupported structured value {type(value).__name__}"
    )


def _structured_mechanics(record: Mapping[str, Any]) -> Any:
    for field in (
        "structuredMechanics",
        "structured_mechanics",
        "publicMechanics",
        "public_mechanics",
        "mechanics",
        "effectIds",
        "effects",
    ):
        if field in record and record.get(field) is not None:
            return _safe_json_value(record[field], field=f"metadata.{field}")
    return None


def _public_scalar(value: Any, *, field: str) -> Any:
    """Copy one scalar accepted by the public r236 surface.

    Sanitization is not a best-effort serializer.  A novel object under an
    otherwise familiar field fails closed instead of being stringified into a
    stable-looking policy fingerprint.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PublicRuleAdapterError(f"{field} contains a non-finite float")
        return float(value)
    if hasattr(value, "name"):
        name = getattr(value, "name")
        if isinstance(name, str) and name:
            return name
    raise PublicRuleAdapterError(
        f"{field} is not a public scalar: {type(value).__name__}"
    )


def _public_scalar_sequence(value: Any, *, field: str) -> Any:
    """Keep only scalar/list or scalar-map data for typed public fields."""

    if not isinstance(value, (Mapping, list, tuple)):
        return _public_scalar(value, field=field)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_key, (str, int)):
                raise PublicRuleAdapterError(f"{field} has a non-public mapping key")
            result[str(raw_key)] = _public_scalar(raw_value, field=f"{field}.{raw_key}")
        return result
    return [
        _public_scalar(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]


def _public_card(value: Any, *, field: str) -> Optional[dict[str, Any]]:
    """Project one visible card/Pokémon without names, text, or private state."""

    if value is None:
        return None
    row = _mapping(value, field=field)
    if _is_unrevealed_prize_area(_reference_area(row, field=f"{field}.area")):
        # A raw Prize card can occur in a context/effect payload as well as in
        # PlayerState.  r298 has no explicit revealed-Prize ABI, so preserve
        # only the fact that this is an unavailable identity—not its card ID,
        # serial, HP, attachments, or other identity-derived properties.
        return {"unrevealed_prize_identity": True}
    result: dict[str, Any] = {}
    for name in _PUBLIC_CARD_SCALAR_FIELDS:
        if name in row:
            result[name] = _public_scalar(row[name], field=f"{field}.{name}")
    for name in _PUBLIC_CARD_ENERGY_FIELDS:
        if name in row:
            result[name] = _public_scalar_sequence(row[name], field=f"{field}.{name}")
    # These are the only nested physical-card locations that can contribute to
    # a stable within-observation source binding.  A raw serial can be used as
    # an internal join key below, but is never emitted by the representation.
    for name in ("energyCards", "tools", "toolCards", "preEvolution", "preEvolutions", "pre_evolution"):
        if name not in row:
            continue
        values = _rows(row[name], field=f"{field}.{name}")
        result[name] = [
            _public_card(item, field=f"{field}.{name}[{index}]")
            for index, item in enumerate(values)
        ]
    return result


def _public_card_rows(value: Any, *, field: str) -> list[Optional[dict[str, Any]]]:
    rows = _rows(value, field=field)
    return [
        _public_card(item, field=f"{field}[{index}]")
        for index, item in enumerate(rows)
    ]


def _public_orderless_menu_cards(
    value: Any,
    *,
    field: str,
) -> list[Optional[dict[str, Any]]]:
    """Project an exposed menu as a deterministic card multiset.

    This does not alter the simulator option sequence used to execute a
    choice.  It removes only an incidental presentation order from the public
    evidence/fingerprint surface.
    """

    rows = _public_card_rows(value, field=field)
    return sorted(rows, key=_canonical_json)


def _public_visible_modifier(value: Any, *, field: str) -> Any:
    """Keep only explicitly structured, visible Prize-modifier facts."""

    if value is None or not isinstance(value, (Mapping, list, tuple)):
        return _public_scalar(value, field=field)
    if isinstance(value, (list, tuple)):
        return [
            _public_visible_modifier(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    # A global modifier map is keyed by seat; a row modifier may identify its
    # seat explicitly.  Preserve neither opaque descriptions nor card text.
    keys = set(value)
    if keys and all(isinstance(key, (str, int)) and str(key) in {"0", "1"} for key in keys):
        return {
            str(key): _public_visible_modifier(item, field=f"{field}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    result: dict[str, Any] = {}
    for name in (*_PUBLIC_VISIBLE_MODIFIER_FIELDS, "playerIndex", "player_index", "targetPlayerIndex"):
        if name in value:
            result[name] = _public_scalar(value[name], field=f"{field}.{name}")
    return result


def _public_once_per_turn_flags(value: Any, *, field: str) -> dict[str, bool]:
    row = _mapping(value, field=field)
    result: dict[str, bool] = {}
    for raw_name, raw_value in sorted(row.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_name, str) or not raw_name:
            raise PublicRuleAdapterError(f"{field} has a non-public flag name")
        parsed = _optional_bool(raw_value, field=f"{field}.{raw_name}")
        if parsed is None:
            raise PublicRuleAdapterError(f"{field}.{raw_name} cannot be null")
        result[raw_name] = parsed
    return result


def _public_player(value: Any, *, acting: bool, field: str) -> dict[str, Any]:
    row = _mapping(value, field=field)
    result: dict[str, Any] = {}
    for name in _PUBLIC_PLAYER_SCALAR_FIELDS:
        if name in row:
            result[name] = _public_scalar(row[name], field=f"{field}.{name}")
    for name in ("active", "bench", "discard"):
        if name in row:
            if name in _ORDERLESS_MENU_AREAS:
                result[name] = _public_orderless_menu_cards(
                    row[name], field=f"{field}.{name}"
                )
            else:
                result[name] = _public_card_rows(row[name], field=f"{field}.{name}")
    if acting and "hand" in row:
        result["hand"] = _public_orderless_menu_cards(
            row["hand"], field=f"{field}.hand"
        )
    else:
        # The actor may observe the opponent hand cardinality, never its IDs.
        # Do not derive a count from an accidental private list: a well-formed
        # r236 observation supplies handCount directly.
        result["hand"] = None
    if "prize" in row:
        # The r298 path does not have an explicit revealed-prize ABI yet.  A
        # non-null raw object in this zone is therefore treated conservatively
        # as potentially unrevealed; only the public cardinality survives.
        result["prize"] = [None] * len(_rows(row["prize"], field=f"{field}.prize"))
    for name in (
        "visiblePrizeModifier",
        "visible_prize_modifier",
        "prizeModifier",
        "prize_modifier",
        "prizeReduction",
        "prize_reduction",
    ):
        if name in row:
            result[name] = _public_visible_modifier(row[name], field=f"{field}.{name}")
    for name in ("oncePerTurnActionFlags", "once_per_turn_action_flags"):
        if name in row:
            result[name] = _public_once_per_turn_flags(row[name], field=f"{field}.{name}")
    return result


def _public_option(
    value: Any,
    *,
    field: str,
    actor: Optional[int] = None,
    locators: Optional[Mapping[int, tuple[dict[str, Any], ...]]] = None,
) -> dict[str, Any]:
    """Project a legal option without admitting unproven card identities.

    The raw option list can carry an ID even when its source is not a public
    card/menu.  That is insufficient provenance: it may be an unrevealed
    Prize/deck identity or another private implementation payload.  Callers
    that provide the raw public locator table retain an ID only when the same
    binding resolves to a visible source/target; the standalone fallback is
    deliberately more conservative and removes all card/skill identities.
    """

    row = _mapping(value, field=field)
    result: dict[str, Any] = {}
    source_area = _area(
        _first_present(row, ("area", "sourceArea")),
        field=f"{field}.sourceArea",
    )
    target_area = _area(
        _first_present(row, ("targetArea", "inPlayArea")),
        field=f"{field}.targetArea",
    )
    source_binding: Optional[dict[str, Any]] = None
    target_binding: Optional[dict[str, Any]] = None
    attachment_bindings: dict[str, Optional[dict[str, Any]]] = {}
    if actor is not None and locators is not None:
        source_binding = _binding(row, role="source", actor=actor, locators=locators)
        target_binding = _binding(row, role="target", actor=actor, locators=locators)
        attachment_bindings = {
            "tool": _attachment_binding(row, kind="tool", actor=actor, locators=locators),
            "energy": _attachment_binding(row, kind="energy", actor=actor, locators=locators),
        }

    source_identity_public = (
        source_binding is not None
        and source_binding.get("card_id") is not None
        and source_binding.get("physical_source_status")
        not in {"unavailable_unrevealed_prize", "unavailable_unproven_public_identity"}
    )
    target_identity_public = (
        target_binding is not None
        and target_binding.get("card_id") is not None
        and target_binding.get("physical_source_status")
        not in {"unavailable_unrevealed_prize", "unavailable_unproven_public_identity"}
    )
    for name in _PUBLIC_OPTION_FIELDS:
        if name not in row:
            continue
        # A Prize slot may be a legal action location, but its unrevealed
        # identity is not actor-visible policy evidence.  Do not retain a raw
        # card/skill ID merely because it rode alongside an otherwise public
        # area/index option payload.  There is no r298 revealed-Prize ABI.
        if (
            _is_unrevealed_prize_area(source_area)
            and name in {
                "cardId",
                "sourceCardId",
                "skillId",
                "skill_id",
                "toolCardId",
                "tool_card_id",
                "energyCardId",
                "energy_card_id",
                "toolIndex",
                "tool_index",
                "energyIndex",
                "energy_index",
            }
        ) or (
            _is_unrevealed_prize_area(target_area)
            and name in {"cardId", "targetCardId"}
        ):
            continue
        if name in {"cardId", "sourceCardId", "skillId", "skill_id"} and not source_identity_public:
            continue
        if name in {"targetCardId"} and not target_identity_public:
            continue
        if name in {"toolCardId", "tool_card_id"} and not (
            attachment_bindings.get("tool") is not None
            and attachment_bindings["tool"].get("card_id") is not None
            and attachment_bindings["tool"].get("physical_source_status")
            not in {"unavailable_unrevealed_prize", "unavailable_unproven_public_identity"}
        ):
            continue
        if name in {"energyCardId", "energy_card_id"} and not (
            attachment_bindings.get("energy") is not None
            and attachment_bindings["energy"].get("card_id") is not None
            and attachment_bindings["energy"].get("physical_source_status")
            not in {"unavailable_unrevealed_prize", "unavailable_unproven_public_identity"}
        ):
            continue
        if (
            name in {"index", "sourceIndex"}
            and source_area in _ORDERLESS_MENU_AREAS
        ) or (
            name in {"targetIndex", "inPlayIndex"}
            and target_area in _ORDERLESS_MENU_AREAS
        ):
            continue
        if name.endswith("Discriminator") or name.endswith("_discriminator"):
            # This is an explicitly simulator-named legal-option identity.  It
            # is considered only after a semantic collision, never as an
            # option ordinal or a direct global serial embedding.
            result[name] = _safe_json_value(row[name], field=f"{field}.{name}")
        else:
            result[name] = _public_scalar(row[name], field=f"{field}.{name}")
    return result


def _public_effect(
    value: Any,
    *,
    field: str,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
) -> Any:
    if value is None or not isinstance(value, Mapping):
        return _public_scalar(value, field=field)
    # In the r236 API a selection effect can itself be a visible Card payload.
    # Preserve its typed public card surface, but never its global serial.
    # An explicit effectId wins over this shape test because it is a distinct
    # simulator effect identity rather than a card reference.
    if "effectId" not in value and any(
        name in value
        for name in (
            "serial",
            "hp",
            "currentHp",
            "maxHp",
            "energyCards",
            "tools",
            "preEvolution",
        )
    ):
        return _public_card_reference_projection(
            value, actor=actor, locators=locators, field=field
        )
    result: dict[str, Any] = {}
    prize_scoped = _is_unrevealed_prize_area(
        _reference_area(value, field=f"{field}.area")
    )
    # A named effect ID is a public simulator semantic, whereas a bare ``id``
    # is ambiguous between an effect and a card reference.  In a Prize-scoped
    # payload the conservative ABI therefore admits only explicit effectId and
    # never a card-derived identity.
    allowed_identity_fields = ("effectId",) if prize_scoped else ("effectId", "id")
    for name in allowed_identity_fields:
        if name in value:
            result[name] = _public_scalar(value[name], field=f"{field}.{name}")
    card_binding = _public_card_reference_binding(
        value, actor=actor, locators=locators, field=field
    )
    if not prize_scoped and "cardId" in value and card_binding.get("card_id") is not None:
        result["cardId"] = _public_scalar(
            card_binding["card_id"], field=f"{field}.cardId"
        )
    for name in ("source", "contextCard"):
        if name in value:
            result[name] = _public_card_reference_projection(
                value[name],
                actor=actor,
                locators=locators,
                field=f"{field}.{name}",
            )
    return result


def sanitize_public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the acting player's explicit r236 public information set.

    This is a *projection*, not a copied observation with a few fields
    deleted.  It consequently removes search tokens, logs/history, raw
    snapshots, future transitions, terminal outcome/reason, private aliases,
    actor deck order, opponent hand IDs, and Prize identities before a hash or
    feature builder can see them.  Terminal facts are available exclusively
    through :func:`extract_public_terminal_target`.
    """

    root = _mapping(observation, field="observation")
    current = _mapping(root.get("current"), field="observation.current")
    players = _rows(current.get("players"), field="current.players", optional=False)
    if len(players) != 2 or not all(isinstance(player, Mapping) for player in players):
        raise PublicRuleAdapterError("current.players must contain exactly two player objects")
    actor = _exact_int(
        current.get("yourIndex"),
        field="current.yourIndex",
        minimum=0,
        maximum=1,
    )
    assert actor is not None
    select = _mapping(root.get("select"), field="observation.select")
    # The locator table is an internal raw-serial join only.  Passing it into
    # the option projection lets this public boundary prove that a raw card ID
    # is backed by a visible board/menu row before retaining it; the table and
    # its serial keys never appear in the sanitized result.
    locators = _locators_for_public_cards(root, actor=actor)

    sanitized_current: dict[str, Any] = {}
    for name in _PUBLIC_CURRENT_SCALAR_FIELDS:
        if name in current:
            sanitized_current[name] = _public_scalar(current[name], field=f"current.{name}")
    sanitized_current["players"] = [
        _public_player(player, acting=index == actor, field=f"current.players[{index}]")
        for index, player in enumerate(players)
    ]
    if "stadium" in current:
        sanitized_current["stadium"] = _public_card_rows(
            current["stadium"], field="current.stadium"
        )
    if "looking" in current:
        sanitized_current["looking"] = _public_orderless_menu_cards(
            current["looking"], field="current.looking"
        )
    for name in (
        "visiblePrizeModifiers",
        "visible_prize_modifiers",
        "prizeModifiers",
    ):
        if name in current:
            sanitized_current[name] = _public_visible_modifier(current[name], field=f"current.{name}")
    for name in ("oncePerTurnActionFlags", "once_per_turn_action_flags"):
        if name in current:
            sanitized_current[name] = _public_once_per_turn_flags(current[name], field=f"current.{name}")

    sanitized_select: dict[str, Any] = {}
    for name in _PUBLIC_SELECT_SCALAR_FIELDS:
        if name in select:
            sanitized_select[name] = _public_scalar(select[name], field=f"select.{name}")
    options = _rows(select.get("option"), field="select.option", optional=False)
    sanitized_select["option"] = [
        _public_option(
            option,
            field=f"select.option[{index}]",
            actor=actor,
            locators=locators,
        )
        for index, option in enumerate(options)
    ]
    if "deck" in select:
        # This is a currently exposed selection menu, not an actor deck-order
        # payload.  Its raw display order is not public policy semantics.
        sanitized_select["deck"] = _public_orderless_menu_cards(
            select["deck"], field="select.deck"
        )
    if "contextCard" in select:
        sanitized_select["contextCard"] = _public_card_reference_projection(
            select["contextCard"],
            actor=actor,
            locators=locators,
            field="select.contextCard",
        )
    if "effect" in select:
        sanitized_select["effect"] = _public_effect(
            select["effect"],
            field="select.effect",
            actor=actor,
            locators=locators,
        )
    return {"current": sanitized_current, "select": sanitized_select}


def extract_public_terminal_target(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return public terminal facts for a *target-only* payload.

    The r300 contract explicitly keeps terminal result/reason and every future
    transition out of policy features.  This helper is therefore intentionally
    separate from :func:`build_public_rule_representation`; callers building
    a policy/vector input cannot receive terminal fields by accident.
    """

    # First validate the same public structural boundary used by the policy
    # representation.  Terminal fields themselves are then read only into this
    # separately typed target sidecar, never returned by the sanitizer.
    sanitize_public_observation(observation)
    raw_current = _mapping(
        _mapping(observation, field="observation").get("current"),
        field="observation.current",
    )
    return {
        "schema": "poke_bot.alakazam_public_terminal_target/v1",
        "target_only": True,
        "result": _exact_int(
            raw_current.get("result"),
            field="current.result",
            minimum=-1,
            maximum=2,
            optional=True,
        ),
        "reason": _enum_name(
            raw_current.get("resultReason", raw_current.get("reason")),
            field="terminal_reason",
            numeric_names={},
            optional=True,
        ),
    }


def _owner_relation(value: Any, *, actor: int, field: str) -> str:
    if value is None:
        return "unspecified"
    index = _exact_int(value, field=field, minimum=0, maximum=1)
    assert index is not None
    return "acting" if index == actor else "opponent"


def _owner_index(relation: str, *, actor: int) -> Optional[int]:
    if relation == "acting":
        return actor
    if relation == "opponent":
        return 1 - actor
    return None


def _maybe_card_rows(value: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for row in _rows(value, field="card rows"):
        if row is None:
            continue
        if not isinstance(row, Mapping):
            continue
        result.append(row)
    return result


def _locators_for_public_cards(
    observation: Mapping[str, Any], *, actor: int
) -> Mapping[int, tuple[dict[str, Any], ...]]:
    """Map a visible raw serial to a normalized local physical locator.

    The raw serial is an internal join key only.  It is never emitted into the
    representation, hash, or residual.  A locator is instead expressed in the
    public source/area/slot namespace that gives a SKILL its physical meaning.
    """

    # This function deliberately consumes the raw observation only as a
    # transient join table.  Its numeric serial keys never leave this function:
    # every successful binding is emitted as owner/area/slot/card_id.  Keeping
    # it separate from ``sanitize_public_observation`` lets public hashes stay
    # invariant under a globally consistent serial renumbering.
    current = _mapping(observation.get("current"), field="observation.current")
    players = _rows(current.get("players"), field="current.players", optional=False)
    select = _mapping(observation.get("select"), field="observation.select")
    by_serial: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def add_card(
        card: Any,
        *,
        owner: str,
        area: str,
        slot: Optional[int],
        attachment_role: Optional[str] = None,
        attachment_slot: Optional[int] = None,
    ) -> None:
        if not isinstance(card, Mapping):
            return
        raw_serial = _serial(card.get("serial"), field="card.serial")
        if raw_serial is None:
            return
        locator: dict[str, Any] = {
            "owner": owner,
            "area": area,
            "slot": slot,
            "card_id": _card_id(card, field="visible card id"),
        }
        if attachment_role is not None:
            locator["attachment_role"] = attachment_role
            locator["attachment_slot"] = attachment_slot
        by_serial[raw_serial].append(locator)

        for field, role in (
            ("tools", "tool"),
            ("toolCards", "tool"),
            ("energyCards", "energy"),
            ("preEvolution", "pre_evolution"),
            ("preEvolutions", "pre_evolution"),
        ):
            nested = card.get(field)
            if not isinstance(nested, (list, tuple)):
                continue
            for index, child in enumerate(nested):
                add_card(
                    child,
                    owner=owner,
                    area=area,
                    slot=slot,
                    attachment_role=role,
                    attachment_slot=index,
                )

    for seat, player in enumerate(players):
        if not isinstance(player, Mapping):
            continue
        owner = "acting" if seat == actor else "opponent"
        for zone in ("active", "bench", "discard", "hand", "prize"):
            # Prize card identity is hidden for both seats.  The raw serial
            # must not become a transient join key here because that would
            # restore an unrevealed card ID through a later SKILL/source
            # binding.  r298 has no revealed-Prize ABI.
            if zone == "prize" or (owner == "opponent" and zone == "hand"):
                continue
            rows = player.get(zone)
            if not isinstance(rows, (list, tuple)):
                continue
            for index, card in enumerate(rows):
                add_card(
                    card,
                    owner=owner,
                    area=zone,
                    slot=None if zone in _ORDERLESS_MENU_AREAS else index,
                )

    for zone, rows in (("stadium", current.get("stadium")), ("looking", current.get("looking"))):
        if isinstance(rows, (list, tuple)):
            for index, card in enumerate(rows):
                owner = "unspecified"
                if isinstance(card, Mapping) and card.get("playerIndex") is not None:
                    owner = _owner_relation(card.get("playerIndex"), actor=actor, field=f"{zone}.playerIndex")
                add_card(
                    card,
                    owner=owner,
                    area=zone,
                    # Looking is an orderless exposed menu.  Stadium still
                    # has a physical board position if the API exposes more
                    # than one row in the future.
                    slot=None if zone in _ORDERLESS_MENU_AREAS else index,
                )
    deck_rows = select.get("deck")
    if isinstance(deck_rows, (list, tuple)):
        for index, card in enumerate(deck_rows):
            add_card(card, owner="acting", area="deck", slot=None)

    normalized: dict[int, tuple[dict[str, Any], ...]] = {}
    for raw_serial, locators in by_serial.items():
        unique = {
            _canonical_json(locator): locator
            for locator in locators
        }
        normalized[raw_serial] = tuple(
            unique[key] for key in sorted(unique)
        )
    return normalized


def _lookup_visible_card(
    locators: Mapping[int, tuple[dict[str, Any], ...]],
    *,
    owner: str,
    area: Optional[str],
    slot: Optional[int],
    attachment_role: Optional[str] = None,
    attachment_slot: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    if area is None or slot is None:
        return None
    candidates: list[dict[str, Any]] = []
    for rows in locators.values():
        for locator in rows:
            if (
                locator.get("owner") == owner
                and locator.get("area") == area
                and locator.get("slot") == slot
                and locator.get("attachment_role") == attachment_role
                and locator.get("attachment_slot") == attachment_slot
            ):
                candidates.append(locator)
    unique = {_canonical_json(candidate): candidate for candidate in candidates}
    if len(unique) == 1:
        return next(iter(unique.values()))
    return None


def _lookup_visible_card_by_id(
    locators: Mapping[int, tuple[dict[str, Any], ...]],
    *,
    owner: str,
    area: Optional[str],
    card_id: int,
) -> Optional[dict[str, Any]]:
    """Resolve one direct ID only when a public zone/menu proves it visible.

    This is intentionally not a global ID lookup.  The area is required and
    the locator table already excludes opponent hands and every Prize row.
    For orderless visible menus, duplicate copies collapse to the same local
    multiset locator; for board zones, different physical slots remain
    ambiguous unless the raw option gave an explicit slot or visible serial.
    """

    if area is None or _is_unrevealed_prize_area(area):
        return None
    candidates: list[dict[str, Any]] = []
    for rows in locators.values():
        for locator in rows:
            if locator.get("area") != area or locator.get("card_id") != card_id:
                continue
            # An omitted player index may resolve only if the complete public
            # locator is otherwise unique; it is never a wildcard emitted into
            # the semantic representation.
            if owner != "unspecified" and locator.get("owner") != owner:
                continue
            if locator.get("attachment_role") is not None:
                continue
            candidates.append(locator)
    unique = {_canonical_json(candidate): candidate for candidate in candidates}
    if len(unique) == 1:
        return dict(next(iter(unique.values())))
    return None


def _public_card_reference_binding(
    value: Any,
    *,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
    field: str,
) -> dict[str, Any]:
    """Resolve a context/effect card only when a visible locator proves it.

    Context and effect payloads are not themselves a public-zone ABI.  Their
    raw ``id``/``cardId`` is therefore admissible only if a visible serial or
    explicit public area/slot maps it back to the acting information set.
    """

    if not isinstance(value, Mapping):
        return {
            "card_id": None,
            "physical_source": None,
            "physical_source_status": "unavailable_unproven_public_identity",
        }
    row = dict(value)
    if "cardId" not in row and "id" in row:
        row["cardId"] = row["id"]
    area = _reference_area(row, field=f"{field}.area")
    if area is not None and "area" not in row:
        row["area"] = area
    slot = _first_present(row, ("index", "sourceIndex", "targetIndex", "inPlayIndex"))
    if slot is not None and "index" not in row:
        row["index"] = slot
    owner = _first_present(
        row,
        ("playerIndex", "sourcePlayerIndex", "targetPlayerIndex", "inPlayPlayerIndex"),
    )
    if owner is not None and "playerIndex" not in row:
        row["playerIndex"] = owner
    return _binding(row, role="source", actor=actor, locators=locators) or {
        "card_id": None,
        "physical_source": None,
        "physical_source_status": "unavailable_unproven_public_identity",
    }


def _public_card_reference_projection(
    value: Any,
    *,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
    field: str,
) -> Any:
    """Sanitize an effect/context card without trusting an unscoped ID."""

    if value is None:
        return None
    binding = _public_card_reference_binding(
        value, actor=actor, locators=locators, field=field
    )
    if binding.get("card_id") is None:
        return {"identity_status": str(binding["physical_source_status"])}
    return _public_card(value, field=field)


def _serial_binding(
    value: Any,
    *,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
    field: str,
) -> tuple[Optional[dict[str, Any]], str]:
    raw_serial = _serial(value, field=field)
    if raw_serial is None:
        return None, "not_supplied"
    rows = locators.get(raw_serial, ())
    if len(rows) == 1:
        return dict(rows[0]), "bound_visible_physical_source"
    if not rows:
        return None, "unresolved_public_serial"
    return None, "ambiguous_public_serial"


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _binding(
    option: Mapping[str, Any],
    *,
    role: str,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
) -> Optional[dict[str, Any]]:
    if role == "source":
        owner_raw = _first_present(option, ("playerIndex", "sourcePlayerIndex"))
        area_raw = _first_present(option, ("area", "sourceArea"))
        slot_raw = _first_present(option, ("index", "sourceIndex"))
        card_raw = _first_present(option, ("cardId", "sourceCardId"))
        serial_raw = _first_present(option, ("sourceSerial", "cardSerial", "serial"))
    elif role == "target":
        owner_raw = _first_present(option, ("targetPlayerIndex", "inPlayPlayerIndex"))
        area_raw = _first_present(option, ("targetArea", "inPlayArea"))
        slot_raw = _first_present(option, ("targetIndex", "inPlayIndex"))
        card_raw = _first_present(option, ("targetCardId", "inPlayCardId"))
        serial_raw = _first_present(option, ("targetSerial", "inPlaySerial"))
    else:
        raise AssertionError(f"unknown binding role {role}")

    supplied = any(value is not None for value in (owner_raw, area_raw, slot_raw, card_raw, serial_raw))
    if not supplied:
        return None
    owner = _owner_relation(owner_raw, actor=actor, field=f"{role}.playerIndex")
    area = _area(area_raw, field=f"{role}.area")
    slot = _exact_int(
        slot_raw,
        field=f"{role}.slot",
        minimum=0,
        maximum=MAX_SLOT_INDEX,
        optional=True,
    )
    # A deck/looking/hand/discard index is a rendering/list position, not a
    # physical board locator.  Do not make it a policy token.  If it has to
    # distinguish an otherwise identical legal action, only an explicit
    # simulator semantic discriminator may do that later in the collision
    # pass.
    if area in _ORDERLESS_MENU_AREAS:
        slot = None
    prize_identity_hidden = _is_unrevealed_prize_area(area)
    # A Prize slot itself is a permissible action location.  Its raw card ID
    # and serial are not.  Do not even parse them: malformed/private payloads
    # must not affect an otherwise public policy representation.
    direct_card = None if prize_identity_hidden else _card_id(card_raw, field=f"{role}.cardId")
    if prize_identity_hidden:
        physical = None
        serial_status = "unavailable_unrevealed_prize"
        resolved = None
    else:
        physical, serial_status = _serial_binding(
            serial_raw,
            locators=locators,
            field=f"{role}.serial",
        )
        resolved = _lookup_visible_card(
            locators,
            owner=owner,
            area=area,
            slot=slot,
        )
    if physical is not None and resolved is not None and physical != resolved:
        raise PublicRuleAdapterError(f"{role} serial binding conflicts with area/slot binding")
    locator = physical if physical is not None else resolved
    if locator is None and direct_card is not None:
        locator = _lookup_visible_card_by_id(
            locators,
            owner=owner,
            area=area,
            card_id=direct_card,
        )
    if direct_card is not None and locator is not None and locator.get("card_id") not in (None, direct_card):
        raise PublicRuleAdapterError(f"{role} card id conflicts with visible physical source")
    card_id = direct_card
    if card_id is None and locator is not None:
        card_id = _card_id(locator.get("card_id"), field=f"{role}.visibleCardId")
    if card_id is not None and locator is None:
        # A bare ID is not public information.  It may represent an
        # unrevealed Prize/deck card or a future private ABI field, so the
        # semantic layer records an explicit unavailable binding instead of
        # embedding it or inventing a global-ID feature.
        card_id = None
        serial_status = "unavailable_unproven_public_identity"
    # A SKILL option may give only a visible serial.  Replace absent raw
    # source fields with its normalized public locator, never the serial.
    if locator is not None:
        if owner == "unspecified":
            owner = str(locator["owner"])
        if area is None:
            area = str(locator["area"])
        if slot is None and locator.get("slot") is not None and area not in _ORDERLESS_MENU_AREAS:
            slot = int(locator["slot"])
    return {
        "owner": owner,
        "area": area,
        "slot": slot,
        "card_id": card_id,
        "physical_source": locator,
        "physical_source_status": serial_status,
    }


def _attachment_binding(
    option: Mapping[str, Any],
    *,
    kind: str,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
) -> Optional[dict[str, Any]]:
    if kind == "tool":
        slot_raw = _first_present(option, ("toolIndex", "tool_index"))
        card_raw = _first_present(option, ("toolCardId", "tool_card_id"))
        serial_raw = _first_present(option, ("toolSerial", "tool_serial"))
    elif kind == "energy":
        slot_raw = _first_present(option, ("energyIndex", "energy_index"))
        card_raw = _first_present(option, ("energyCardId", "energy_card_id"))
        serial_raw = _first_present(option, ("energySerial", "energy_serial"))
    else:
        raise AssertionError(f"unknown attachment kind {kind}")
    if slot_raw is None and card_raw is None and serial_raw is None:
        return None
    source = _binding(option, role="source", actor=actor, locators=locators)
    source_is_unrevealed_prize = (
        source is not None
        and _is_unrevealed_prize_area(source.get("area"))
    )
    if source_is_unrevealed_prize:
        # An attachment identity on an unrevealed Prize is private for the
        # same reason as the parent card.  Retain only a typed unavailable
        # marker; even the attachment's list slot can fingerprint hidden
        # composition/order.
        return {
            "kind": kind,
            "slot": None,
            "card_id": None,
            "parent_source": source,
            "physical_source": None,
            "physical_source_status": "unavailable_unrevealed_prize",
        }
    slot = _exact_int(
        slot_raw,
        field=f"{kind}.attachment_slot",
        minimum=0,
        maximum=MAX_SLOT_INDEX,
        optional=True,
    )
    direct_card = _card_id(card_raw, field=f"{kind}.attachment_card_id")
    physical, status = _serial_binding(
        serial_raw,
        locators=locators,
        field=f"{kind}.attachment_serial",
    )
    if physical is None and source is not None:
        physical = _lookup_visible_card(
            locators,
            owner=str(source["owner"]),
            area=source.get("area"),
            slot=source.get("slot"),
            attachment_role=kind,
            attachment_slot=slot,
        )
        if physical is not None:
            status = "bound_visible_attachment"
    if direct_card is not None and physical is not None and physical.get("card_id") not in (None, direct_card):
        raise PublicRuleAdapterError(f"{kind} attachment card conflicts with visible source")
    if direct_card is not None and physical is None:
        # Attachment IDs need the same visible-locator proof as parent card
        # IDs.  A bare tool/energy ID can otherwise reintroduce a private card
        # identity after the source binding was correctly masked.
        direct_card = None
        status = "unavailable_unproven_public_identity"
    return {
        "kind": kind,
        "slot": slot,
        "card_id": direct_card if direct_card is not None else (None if physical is None else physical.get("card_id")),
        "parent_source": source,
        "physical_source": physical,
        "physical_source_status": status,
    }


def _catalog_payload_rows(
    catalog: Mapping[str, Any],
    *,
    field: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Validate the non-text card/attack records shared by loader and tests."""

    raw_cards = _rows(catalog.get("cards"), field=f"{field}.cards", optional=False)
    raw_attacks = _rows(catalog.get("attacks"), field=f"{field}.attacks", optional=False)
    cards = tuple(_mapping(row, field=f"{field}.cards[]") for row in raw_cards)
    attacks = tuple(_mapping(row, field=f"{field}.attacks[]") for row in raw_attacks)
    card_ids: set[int] = set()
    attack_ids: set[int] = set()
    for row in cards:
        card_id = _card_id(row, field=f"{field}.cards.cardId", optional=False)
        assert card_id is not None
        if card_id in card_ids:
            raise PublicRuleAdapterError(f"duplicate metadata card id {card_id}")
        card_ids.add(card_id)
    for row in attacks:
        attack_id = _attack_id(
            row.get("attackId", row.get("id")),
            field=f"{field}.attacks.attackId",
            optional=False,
        )
        assert attack_id is not None
        if attack_id in attack_ids:
            raise PublicRuleAdapterError(f"duplicate metadata attack id {attack_id}")
        attack_ids.add(attack_id)
    return cards, attacks


def _forbidden_text_key(value: Any) -> bool:
    """Reject prose/name/hash routes from new structured mechanics artifacts."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            token = _norm_token(raw_key)
            if token in {
                "name",
                "text",
                "description",
                "effectexplanation",
                "textderived",
                "texthash",
                "namehash",
            }:
                return True
            if _forbidden_text_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_forbidden_text_key(item) for item in value)
    return False


def _receipt_payload_digest(receipt: Mapping[str, Any]) -> str:
    """Digest the receipt exactly as the create-only sealer did before adding it."""

    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    return _sha256(payload)


def _unavailable_catalog_provenance(
    reason: str,
    *,
    catalog_file_sha256: Optional[str] = None,
    receipt_file_sha256: Optional[str] = None,
    vectors_file_sha256: Optional[str] = None,
) -> PublicCatalogProvenance:
    return PublicCatalogProvenance(
        eligible=False,
        reason=reason,
        catalog_file_sha256=catalog_file_sha256,
        receipt_file_sha256=receipt_file_sha256,
        vectors_file_sha256=vectors_file_sha256,
    )


def validate_public_catalog_provenance(
    catalog: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    pins: PublicCatalogPins = DEFAULT_PUBLIC_CATALOG_PINS,
    catalog_file_sha256: Optional[str] = None,
    receipt_file_sha256: Optional[str] = None,
    vectors_file_sha256: Optional[str] = None,
) -> PublicCatalogProvenance:
    """Validate a sealed r298 structured catalog without importing torch.

    This is intentionally a *status* API: malformed or mismatched evidence
    returns an unavailable provenance record rather than accidentally making a
    caller fall back to guide prose or an old text-hashed metadata catalog.
    :func:`load_sealed_public_catalog` turns that status into a fail-closed
    exception for materialization paths.
    """

    try:
        if not isinstance(catalog, Mapping) or not isinstance(receipt, Mapping):
            return _unavailable_catalog_provenance("catalog_or_receipt_not_mapping")
        if catalog_file_sha256 != pins.catalog_file_sha256:
            return _unavailable_catalog_provenance(
                "catalog_file_sha256_mismatch",
                catalog_file_sha256=catalog_file_sha256,
                receipt_file_sha256=receipt_file_sha256,
                vectors_file_sha256=vectors_file_sha256,
            )
        if receipt_file_sha256 != pins.receipt_file_sha256:
            return _unavailable_catalog_provenance(
                "receipt_file_sha256_mismatch",
                catalog_file_sha256=catalog_file_sha256,
                receipt_file_sha256=receipt_file_sha256,
                vectors_file_sha256=vectors_file_sha256,
            )
        if vectors_file_sha256 != pins.vectors_file_sha256:
            return _unavailable_catalog_provenance(
                "vectors_file_sha256_mismatch",
                catalog_file_sha256=catalog_file_sha256,
                receipt_file_sha256=receipt_file_sha256,
                vectors_file_sha256=vectors_file_sha256,
            )
        if catalog.get("schema") != PUBLIC_CATALOG_SCHEMA or catalog.get("revision") != PUBLIC_RULE_ADAPTER_REVISION:
            return _unavailable_catalog_provenance("catalog_schema_or_revision_mismatch")
        if catalog.get("status") != "sealed_public_simulator_catalog":
            return _unavailable_catalog_provenance("catalog_not_sealed")
        if receipt.get("schema") != PUBLIC_CATALOG_RECEIPT_SCHEMA or receipt.get("revision") != PUBLIC_RULE_ADAPTER_REVISION:
            return _unavailable_catalog_provenance("receipt_schema_or_revision_mismatch")
        if receipt.get("status") != "passed_elmo_only_nonproduction":
            return _unavailable_catalog_provenance("receipt_not_passed_elmo_only")
        if receipt.get("receipt_payload_sha256") != _receipt_payload_digest(receipt):
            return _unavailable_catalog_provenance("receipt_payload_sha256_mismatch")

        source = _mapping(catalog.get("source"), field="catalog.source")
        for key, expected in (
            ("goal_sha256", pins.goal_sha256),
            ("contract_sha256", pins.contract_sha256),
            ("libcg_sha256", pins.libcg_sha256),
            ("csv_sha256", pins.csv_sha256),
        ):
            if source.get(key) != expected or receipt.get(key) != expected:
                return _unavailable_catalog_provenance(f"pinned_{key}_mismatch")
        if source.get("libcg_size_bytes") != pins.libcg_size_bytes or receipt.get("libcg_size_bytes") != pins.libcg_size_bytes:
            return _unavailable_catalog_provenance("pinned_libcg_size_mismatch")
        if receipt.get("catalog_schema") != PUBLIC_CATALOG_SCHEMA:
            return _unavailable_catalog_provenance("receipt_catalog_schema_mismatch")
        if receipt.get("catalog_file_sha256") != pins.catalog_file_sha256:
            return _unavailable_catalog_provenance("receipt_catalog_file_sha256_mismatch")
        if receipt.get("fixed_vectors_file_sha256") != pins.vectors_file_sha256:
            return _unavailable_catalog_provenance("receipt_vectors_file_sha256_mismatch")

        computed_catalog_semantic = _sha256(catalog)
        if receipt.get("catalog_semantic_sha256") != computed_catalog_semantic:
            return _unavailable_catalog_provenance("catalog_semantic_sha256_mismatch")
        catalog_provenance = _mapping(catalog.get("provenance"), field="catalog.provenance")
        receipt_provenance = _mapping(receipt.get("metadata_provenance"), field="receipt.metadata_provenance")
        if _canonical_json(catalog_provenance) != _canonical_json(receipt_provenance):
            return _unavailable_catalog_provenance("catalog_receipt_metadata_provenance_mismatch")
        for key, expected in (
            ("engine_cards_sha256", pins.engine_cards_sha256),
            ("engine_attacks_sha256", pins.engine_attacks_sha256),
            ("csv_sha256", pins.csv_sha256),
            ("structured_rule_vectors_sha256", pins.structured_vectors_sha256),
        ):
            if catalog_provenance.get(key) != expected:
                return _unavailable_catalog_provenance(f"metadata_{key}_mismatch")
        if receipt_provenance.get("engine_cards_sha256") != pins.engine_cards_sha256 or receipt_provenance.get("engine_attacks_sha256") != pins.engine_attacks_sha256:
            return _unavailable_catalog_provenance("receipt_engine_digest_mismatch")
        if receipt.get("facts", {}).get("structured_rule_vectors_sha256") != pins.structured_vectors_sha256:
            return _unavailable_catalog_provenance("receipt_structured_vector_semantic_mismatch")
        facts = _mapping(receipt.get("facts"), field="receipt.facts")
        if (
            facts.get("mechanics_from_pinned_engine") is not True
            or facts.get("csv_used_as_mechanics_authority") is not False
            or facts.get("text_hash_used_as_rules_parser") is not False
            or facts.get("text_or_name_hash_in_new_residual_vectors") is not False
            or facts.get("card2vec_preserved_unchanged") is not True
        ):
            return _unavailable_catalog_provenance("receipt_fact_boundary_mismatch")
        authority = _mapping(catalog.get("authority"), field="catalog.authority")
        if (
            authority.get("mechanics") != "pinned_official_libcg_r236"
            or authority.get("csv_role") != "provenance_checked_identity_join_only"
            or authority.get("card_text_rules_parser") is not False
            or authority.get("new_residual_vector_source") != "structured_engine_fields_only"
            or authority.get("card2vec_preserved_unchanged") is not True
            or authority.get("policy_or_runtime_authority") is not False
        ):
            return _unavailable_catalog_provenance("catalog_authority_boundary_mismatch")
        # The final rev4 receipt must state both the experiment role and the
        # immutable execution hostname explicitly.  Never infer either from a
        # default, the current machine, or a storage path.
        if receipt.get("execution_host_role") != "elmo":
            return _unavailable_catalog_provenance("receipt_execution_host_role_not_elmo")
        if receipt.get("execution_hostname") != "truenas":
            return _unavailable_catalog_provenance("receipt_execution_hostname_not_truenas")
        _catalog_payload_rows(catalog, field="catalog")
        if _forbidden_text_key(catalog.get("cards")) or _forbidden_text_key(catalog.get("attacks")):
            return _unavailable_catalog_provenance("catalog_contains_text_or_name_mechanics")
        return PublicCatalogProvenance(
            eligible=True,
            catalog_file_sha256=catalog_file_sha256,
            receipt_file_sha256=receipt_file_sha256,
            vectors_file_sha256=vectors_file_sha256,
            catalog_semantic_sha256=computed_catalog_semantic,
            structured_vectors_sha256=pins.structured_vectors_sha256,
            engine_cards_sha256=pins.engine_cards_sha256,
            engine_attacks_sha256=pins.engine_attacks_sha256,
        )
    except (PublicRuleAdapterError, TypeError, ValueError, KeyError) as exc:
        return _unavailable_catalog_provenance(f"invalid_catalog_provenance:{type(exc).__name__}")


def load_sealed_public_catalog(
    catalog_path: Path | str,
    receipt_path: Path | str,
    *,
    vectors_path: Path | str | None = None,
    pins: PublicCatalogPins = DEFAULT_PUBLIC_CATALOG_PINS,
) -> SealedPublicCatalog:
    """Load only an exact receipt-bound, public structured r298 catalog.

    Raw mappings, ``MetadataCatalog``, CSVs, and guide IDs are intentionally
    not accepted here.  The vectors file is hashed even if the caller only
    needs metadata, so a catalog cannot be separated from the residual surface
    sealed alongside it.
    """

    catalog_file = Path(catalog_path)
    receipt_file = Path(receipt_path)
    vector_file = Path(vectors_path) if vectors_path is not None else catalog_file.with_name("fixed-vectors.pt")
    try:
        catalog_hash = _file_sha256(catalog_file)
        receipt_hash = _file_sha256(receipt_file)
        vector_hash = _file_sha256(vector_file)
        catalog = _mapping(json.loads(catalog_file.read_text(encoding="utf-8")), field="catalog")
        receipt = _mapping(json.loads(receipt_file.read_text(encoding="utf-8")), field="receipt")
    except (OSError, json.JSONDecodeError, PublicRuleAdapterError) as exc:
        raise PublicRuleAdapterError("cannot read sealed public catalog artifact") from exc
    status = validate_public_catalog_provenance(
        catalog,
        receipt,
        pins=pins,
        catalog_file_sha256=catalog_hash,
        receipt_file_sha256=receipt_hash,
        vectors_file_sha256=vector_hash,
    )
    if not status.eligible:
        raise PublicRuleAdapterError(
            f"sealed public catalog is unavailable: {status.reason}"
        )
    cards, attacks = _catalog_payload_rows(catalog, field="catalog")
    return SealedPublicCatalog(
        cards=cards,
        attacks=attacks,
        provenance=status,
        pins=pins,
        catalog_path=catalog_file.resolve(),
        receipt_path=receipt_file.resolve(),
        vectors_path=vector_file.resolve(),
    )


def load_sealed_public_catalog_vectors(
    catalog: SealedPublicCatalog,
) -> SealedPublicCatalogVectors:
    """Load verified 64/20 structured vectors for the new additive residual.

    Existing Card2Vec tensors are never read, copied, or modified.  This
    loader verifies that the sealed vector file contains only the fixed
    structured dimensions produced for r298 and rejects legacy text-hashed
    ``card_metadata`` vectors.
    """

    _require_torch()
    if not isinstance(catalog, SealedPublicCatalog):
        raise PublicRuleAdapterError("structured residual requires SealedPublicCatalog")
    # Reopen through the strict loader: a hand-constructed dataclass cannot
    # masquerade as a receipt-bound materializer input.
    verified = load_sealed_public_catalog(
        catalog.catalog_path,
        catalog.receipt_path,
        vectors_path=catalog.vectors_path,
        pins=catalog.pins,
    )
    try:
        try:
            payload = torch.load(verified.vectors_path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch releases without weights_only
            payload = torch.load(verified.vectors_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - corrupt trusted bytes guard
        raise PublicRuleAdapterError("cannot load sealed structured vector artifact") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != PUBLIC_STRUCTURED_VECTORS_SCHEMA:
        raise PublicRuleAdapterError("sealed structured vector schema mismatch")
    provenance = _mapping(payload.get("provenance"), field="vectors.provenance")
    if (
        provenance.get("structured_rule_vectors_sha256") != verified.pins.structured_vectors_sha256
        or provenance.get("text_or_name_hash_columns_included") is not False
        or provenance.get("card2vec_tensor_mutation") is not False
    ):
        raise PublicRuleAdapterError("sealed structured vector provenance mismatch")
    card_features = payload.get("card_features")
    attack_features = payload.get("attack_features")
    if not isinstance(card_features, torch.Tensor) or not isinstance(attack_features, torch.Tensor):
        raise PublicRuleAdapterError("sealed structured vectors must contain tensors")
    if (
        card_features.ndim != 2
        or attack_features.ndim != 2
        or int(card_features.size(1)) != 64
        or int(attack_features.size(1)) != 20
        or not torch.isfinite(card_features).all()
        or not torch.isfinite(attack_features).all()
    ):
        raise PublicRuleAdapterError("sealed structured vector dimensions or values are invalid")
    max_card = max((_card_id(row, field="catalog.cardId", optional=False) or 0) for row in verified.cards)
    max_attack = max((_attack_id(row.get("attackId"), field="catalog.attackId", optional=False) or 0) for row in verified.attacks)
    if int(card_features.size(0)) <= max_card or int(attack_features.size(0)) <= max_attack:
        raise PublicRuleAdapterError("sealed structured vector vocab does not cover catalog ids")
    digest = hashlib.sha256()
    for tensor in (card_features, attack_features):
        value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    vector_semantic = "sha256:" + digest.hexdigest()
    if vector_semantic != verified.pins.structured_vectors_sha256:
        raise PublicRuleAdapterError("sealed structured vector semantic digest mismatch")
    return SealedPublicCatalogVectors(
        card_features=card_features.detach().clone(),
        attack_features=attack_features.detach().clone(),
        provenance=verified.provenance,
        vector_schema=PUBLIC_STRUCTURED_VECTORS_SCHEMA,
    )


def is_public_catalog_eligible(catalog: Any) -> bool:
    """Return true only for the exact default-pinned sealed catalog artifact."""

    if not isinstance(catalog, SealedPublicCatalog):
        return False
    if catalog.pins != DEFAULT_PUBLIC_CATALOG_PINS or not catalog.provenance.eligible:
        return False
    try:
        verified = load_sealed_public_catalog(
            catalog.catalog_path,
            catalog.receipt_path,
            vectors_path=catalog.vectors_path,
            pins=DEFAULT_PUBLIC_CATALOG_PINS,
        )
    except PublicRuleAdapterError:
        return False
    return verified.provenance.eligible


def _metadata_maps(
    catalog: Any,
    *,
    allow_test_catalog: bool = False,
) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]], dict[str, Any]]:
    """Expose catalog records only through the sealed/test-only boundary."""

    if catalog is None:
        return {}, {}, {
            "available": False,
            "eligible": False,
            "reason": "metadata_catalog_not_supplied",
        }
    if isinstance(catalog, SealedPublicCatalog):
        if not is_public_catalog_eligible(catalog):
            return {}, {}, {
                "available": False,
                "eligible": False,
                "reason": "sealed_catalog_provenance_unavailable",
            }
        cards = catalog.cards
        attacks = catalog.attacks
        provenance = catalog.provenance.to_dict()
    elif isinstance(catalog, Mapping) and allow_test_catalog:
        # Explicit fixture escape hatch.  It cannot authorize materialization,
        # residual construction, or a nonzero policy feature; the provenance
        # stamp makes accidental fixture leakage auditable.
        cards, attacks = _catalog_payload_rows(catalog, field="test_catalog")
        provenance = {
            "available": True,
            "eligible": False,
            "test_only": True,
            "reason": "unsealed_test_catalog_fixture",
        }
    elif isinstance(catalog, Mapping):
        raise PublicRuleAdapterError(
            "raw metadata mapping is forbidden; pass a sealed public catalog or allow_test_catalog=True"
        )
    else:
        raise PublicRuleAdapterError(
            "legacy MetadataCatalog/text-hashed metadata is forbidden for r298"
        )
    card_rows: dict[int, Mapping[str, Any]] = {}
    attack_rows: dict[int, Mapping[str, Any]] = {}
    for row in cards:
        card_id = _card_id(row, field="metadata.cardId", optional=False)
        assert card_id is not None
        card_rows[card_id] = row
    for row in attacks:
        attack_id = _attack_id(row.get("attackId", row.get("id")), field="metadata.attackId", optional=False)
        assert attack_id is not None
        attack_rows[attack_id] = row
    return card_rows, attack_rows, provenance


def _energy_type_units(
    energy: Mapping[str, Any],
    *,
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> tuple[Counter[str], int]:
    """Return typed units and the count that has no public typed declaration."""

    candidates = _first_present(
        energy,
        (
            "typedEnergyUnits",
            "typed_energy_units",
            "energyTypes",
            "energy_types",
            "providesEnergy",
            "provides_energy",
            "energyType",
            "energy_type",
            "types",
            "type",
        ),
    )
    if candidates is None:
        card_id = _card_id(energy, field="energy card id")
        record = None if card_id is None else metadata_cards.get(card_id)
        if record is not None:
            candidates = _first_present(record, ("energyType", "energy_type", "pokemonType", "type"))
    if candidates is None:
        return Counter(), 1
    units: Counter[str] = Counter()
    if isinstance(candidates, Mapping):
        for raw_type, raw_count in candidates.items():
            count = _exact_int(raw_count, field="typed energy unit count", minimum=0, maximum=MAX_SELECTION_COUNT)
            assert count is not None
            units[f"energy_type:{_norm_token(raw_type)}"] += count
    elif isinstance(candidates, (list, tuple)):
        for raw_type in candidates:
            units[f"energy_type:{_norm_token(raw_type)}"] += 1
    else:
        units[f"energy_type:{_norm_token(candidates)}"] += 1
    return units, 0


def _pre_evolution_ids(card: Mapping[str, Any]) -> list[int]:
    raw = _first_present(card, ("preEvolution", "preEvolutions", "pre_evolution"))
    if raw is None:
        return []
    rows = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    result: list[int] = []
    for index, item in enumerate(rows):
        ident = _card_id(item, field=f"preEvolution[{index}]")
        if ident is not None:
            result.append(ident)
    return result


def _card_state(
    card: Any,
    *,
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    if card is None:
        return None
    mapping = _mapping(card, field="visible card")
    card_id = _card_id(mapping, field="visible card id")
    record = None if card_id is None else metadata_cards.get(card_id)
    current_hp_raw = _first_present(mapping, ("currentHp", "current_hp", "hp"))
    # Recorded libcg successor states can retain negative overkill HP before
    # terminal cleanup.  Public game semantics expose that as zero HP.
    if type(current_hp_raw) is int and current_hp_raw < 0:
        current_hp_raw = 0
    current_hp = _exact_int(
        current_hp_raw,
        field="visible card current hp",
        minimum=0,
        maximum=MAX_HP_VALUE,
        optional=True,
    )
    max_hp_raw = _first_present(mapping, ("maxHp", "max_hp"))
    if max_hp_raw is None and record is not None:
        max_hp_raw = record.get("hp")
    max_hp = _exact_int(
        max_hp_raw,
        field="visible card max hp",
        minimum=0,
        maximum=MAX_HP_VALUE,
        optional=True,
    )
    energy_ids: list[int] = []
    typed_units: Counter[str] = Counter()
    unknown_energy = 0
    for index, energy in enumerate(_rows(mapping.get("energyCards"), field="energyCards")):
        if not isinstance(energy, Mapping):
            raise PublicRuleAdapterError(f"energyCards[{index}] is not a card object")
        ident = _card_id(energy, field=f"energyCards[{index}].id")
        if ident is not None:
            energy_ids.append(ident)
        units, unknown = _energy_type_units(energy, metadata_cards=metadata_cards)
        typed_units.update(units)
        unknown_energy += unknown
    tool_ids = [
        card_id
        for index, tool in enumerate(_rows(_first_present(mapping, ("tools", "toolCards")), field="tools"))
        if (card_id := _card_id(tool, field=f"tools[{index}].id")) is not None
    ]
    return {
        "card_id": card_id,
        "current_hp": current_hp,
        "max_hp": max_hp,
        "pre_evolution_card_ids": _pre_evolution_ids(mapping),
        "pre_evolution_count": len(_pre_evolution_ids(mapping)),
        "appear_this_turn": _optional_bool(
            _first_present(mapping, ("appearThisTurn", "appear_this_turn")),
            field="appearThisTurn",
        ),
        "attached_energy_card_ids": sorted(energy_ids),
        "typed_energy_units": [list(item) for item in sorted(typed_units.items())],
        "unknown_typed_energy_card_count": unknown_energy,
        "attached_tool_card_ids": sorted(tool_ids),
    }


def _zone_cards(
    rows: Any,
    *,
    metadata_cards: Mapping[int, Mapping[str, Any]],
    preserve_slots: bool,
) -> list[Any]:
    result: list[Any] = []
    for index, card in enumerate(_rows(rows, field="zone")):
        state = _card_state(card, metadata_cards=metadata_cards) if card is not None else None
        if preserve_slots:
            result.append({"slot": index, "card": state})
        elif state is not None:
            result.append(state)
    if not preserve_slots:
        result.sort(key=_canonical_json)
    return result


def _count_from_player(player: Mapping[str, Any], *, field: str, fallback_zone: str) -> Optional[int]:
    raw = player.get(field)
    if raw is not None:
        return _exact_int(raw, field=field, minimum=0, maximum=MAX_SELECTION_COUNT)
    zone = player.get(fallback_zone)
    if isinstance(zone, (list, tuple)):
        return len(zone)
    return None


def _player_state(
    player: Mapping[str, Any],
    *,
    relation: str,
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    bench_max = _exact_int(
        player.get("benchMax"),
        field=f"{relation}.benchMax",
        minimum=0,
        maximum=MAX_BENCH_MAXIMUM,
        optional=True,
    )
    active = _zone_cards(
        player.get("active"), metadata_cards=metadata_cards, preserve_slots=True
    )
    bench = _zone_cards(
        player.get("bench"), metadata_cards=metadata_cards, preserve_slots=True
    )
    # Discard and acting hand are public / actor-visible multisets.  Their
    # arbitrary display order is not made into an ordinal token.
    discard = _zone_cards(
        player.get("discard"), metadata_cards=metadata_cards, preserve_slots=False
    )
    hand = (
        _zone_cards(player.get("hand"), metadata_cards=metadata_cards, preserve_slots=False)
        if relation == "acting"
        else None
    )
    return {
        "relation": relation,
        "effective_bench_maximum": bench_max,
        "active": active,
        "bench": bench,
        "discard": discard,
        "hand": hand,
        "hand_count": _count_from_player(player, field="handCount", fallback_zone="hand"),
        "deck_count": _count_from_player(player, field="deckCount", fallback_zone="deck"),
        "prize_count": _count_from_player(player, field="prizeCount", fallback_zone="prize"),
        "special_conditions": {
            name: _optional_bool(player.get(name), field=f"{relation}.{name}")
            for name in ("asleep", "burned", "confused", "paralyzed", "poisoned")
        },
    }


def _menu_signature(
    rows: Any,
    *,
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cards = _zone_cards(rows, metadata_cards=metadata_cards, preserve_slots=False)
    counts: Counter[str] = Counter(_canonical_json(card) for card in cards)
    return [
        {"card": json.loads(key), "count": count}
        for key, count in sorted(counts.items())
    ]


def _card_reference(
    value: Any,
    *,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
    field: str,
) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    return _public_card_reference_binding(
        value, actor=actor, locators=locators, field=field
    )


def _effect_reference(
    value: Any,
    *,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return {"effect_id": _exact_int(value, field="effect id", minimum=0, maximum=MAX_CARD_ID)}
    card_payload = "effectId" not in value and any(
        name in value
        for name in (
            "serial",
            "hp",
            "currentHp",
            "maxHp",
            "energyCards",
            "tools",
            "preEvolution",
        )
    )
    effect_id = _exact_int(
        _first_present(value, ("effectId",) if card_payload else ("effectId", "id")),
        field="effect.id",
        minimum=0,
        maximum=MAX_CARD_ID,
        optional=True,
    )
    source_value = value.get("source", value.get("contextCard"))
    if source_value is None and card_payload:
        source_value = value
    source = _card_reference(
        source_value,
        actor=actor,
        locators=locators,
        field="effect.source",
    )
    direct_card = _card_reference(
        value,
        actor=actor,
        locators=locators,
        field="effect.card",
    ) if (card_payload or "cardId" in value) else None
    card_id = None
    if direct_card is not None and direct_card.get("card_id") is not None:
        card_id = _card_id(direct_card.get("card_id"), field="effect.cardId")
    elif (
        source is not None
        and source.get("card_id") is not None
        and source.get("physical_source_status")
        not in {"unavailable_unrevealed_prize", "unavailable_unproven_public_identity"}
    ):
        # An explicit public source can supply the card identity when the
        # effect payload itself omits it.  A bare effect.cardId never can.
        card_id = _card_id(source.get("card_id"), field="effect.sourceCardId")
    return {"effect_id": effect_id, "card_id": card_id, "source": source}


def _selection_state(
    select: Mapping[str, Any],
    *,
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    options = _rows(select.get("option"), field="select.option", optional=False)
    if len(options) > MAX_OPTION_COUNT:
        raise PublicRuleAdapterError(
            f"select.option length {len(options)} exceeds {MAX_OPTION_COUNT}"
        )
    min_count = _exact_int(
        select.get("minCount"),
        field="select.minCount",
        minimum=0,
        maximum=MAX_SELECTION_COUNT,
    )
    max_count = _exact_int(
        select.get("maxCount"),
        field="select.maxCount",
        minimum=0,
        maximum=MAX_SELECTION_COUNT,
    )
    assert min_count is not None and max_count is not None
    if min_count > max_count or max_count > len(options):
        raise PublicRuleAdapterError(
            "selection bounds are invalid for the complete legal option list"
        )
    selection_type = _enum_name(
        select.get("type"),
        field="select_type",
        numeric_names={},
        optional=True,
    )
    return {
        "context": _context(select.get("context")),
        "selection_type": selection_type,
        "context_card": _card_reference(
            select.get("contextCard"), actor=actor, locators=locators, field="contextCard"
        ),
        "effect_source": _effect_reference(select.get("effect"), actor=actor, locators=locators),
        "min_count": min_count,
        "max_count": max_count,
        "remain_damage_counter": _exact_int(
            _first_present(select, ("remainDamageCounter", "remainingDamageCounter")),
            field="select.remainDamageCounter",
            minimum=0,
            maximum=MAX_SELECTION_COUNT,
            optional=True,
        ),
        "remain_energy_cost": _exact_int(
            _first_present(select, ("remainEnergyCost", "remainingEnergyCost")),
            field="select.remainEnergyCost",
            minimum=0,
            maximum=MAX_SELECTION_COUNT,
            optional=True,
        ),
        "looking_menu": _menu_signature(
            select.get("deck"), metadata_cards=metadata_cards
        ),
        "option_count": len(options),
    }


def _stable_simulator_discriminator(option: Mapping[str, Any]) -> Optional[str]:
    """Return an opaque stable simulator discriminator, never a raw ordinal.

    Only explicitly named simulator identities are eligible.  In particular we
    do not look at ``index``, the option-list position, ``id``, or ``serial``.
    The digest is an opaque canonical identity rather than an embedding of a
    global simulator number.
    """

    value = _first_present(
        option,
        (
            "simulatorDiscriminator",
            "simulator_discriminator",
            "semanticDiscriminator",
            "semantic_discriminator",
            "legalActionDiscriminator",
            "legal_action_discriminator",
        ),
    )
    if value is None:
        return None
    return _sha256({"stable_simulator_discriminator": _safe_json_value(value, field="simulator discriminator")})


def _option_semantics(
    option: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    actor: int,
    locators: Mapping[int, tuple[dict[str, Any], ...]],
) -> dict[str, Any]:
    option_type = _option_type(option.get("type"))
    number = None
    if option_type == "number":
        number = _exact_int(
            option.get("number"),
            field="option.number",
            minimum=0,
            maximum=MAX_NUMBER_VALUE,
        )
    count = _exact_int(
        option.get("count"),
        field="option.count",
        minimum=0,
        maximum=MAX_SELECTION_COUNT,
        optional=True,
    )
    attack_id = _attack_id(option.get("attackId"), field="option.attackId")
    source = _binding(option, role="source", actor=actor, locators=locators)
    target = _binding(option, role="target", actor=actor, locators=locators)
    attachments = [
        attachment
        for attachment in (
            _attachment_binding(option, kind="tool", actor=actor, locators=locators),
            _attachment_binding(option, kind="energy", actor=actor, locators=locators),
        )
        if attachment is not None
    ]
    source_is_unrevealed_prize = (
        source is not None
        and _is_unrevealed_prize_area(source.get("area"))
    )
    source_identity_public = (
        source is not None
        and source.get("card_id") is not None
        and source.get("physical_source_status")
        not in {"unavailable_unrevealed_prize", "unavailable_unproven_public_identity"}
    )
    # Do not read ``option.cardId`` directly.  It becomes a semantic feature
    # only through the already-proven public source binding, which prevents a
    # bare Skill/Card payload from injecting a hidden identity.
    direct_card = (
        _card_id(source.get("card_id"), field="option.sourceCardId")
        if source_identity_public and not source_is_unrevealed_prize
        else None
    )
    skill_identity = None
    if option_type == "skill":
        skill_id = (
            _exact_int(
                _first_present(option, ("skillId", "skill_id")),
                field="option.skillId",
                minimum=0,
                maximum=MAX_SELECTION_COUNT,
                optional=True,
            )
            if source_identity_public
            else None
        )
        skill_identity = {
            "card_id": direct_card,
            "skill_id": skill_id,
            "physical_source": None if source is None else source.get("physical_source"),
            "physical_source_status": None if source is None else source.get("physical_source_status"),
        }
    special_condition = _enum_name(
        _first_present(option, ("specialConditionType", "special_condition_type")),
        field="special_condition_type",
        numeric_names={},
        optional=True,
    )
    return {
        "option_type": option_type,
        "number": number,
        "count": count,
        "attack_id": attack_id,
        "card_id": direct_card,
        "special_condition": special_condition,
        "source": source,
        "target": target,
        "attachments": attachments,
        "skill_identity": skill_identity,
        # This field is intentionally populated only after identical semantic
        # rows are grouped below; see build_public_rule_representation().
        "stable_simulator_discriminator": None,
    }


def _referenced_ids(value: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    card_ids: set[int] = set()
    attack_ids: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if key == "card_id" and isinstance(nested, int) and nested > 0:
                    card_ids.add(nested)
                elif key == "attack_id" and isinstance(nested, int) and nested > 0:
                    attack_ids.add(nested)
                else:
                    visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(sorted(card_ids)), tuple(sorted(attack_ids))


def _metadata_attack_summary(
    attack_id: int,
    *,
    attacks: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    record = attacks.get(attack_id)
    if record is None:
        return {"attack_id": attack_id, "available": False}
    raw_cost = _first_present(record, ("energies", "energyCost", "energy_cost"))
    if raw_cost is None:
        cost: list[Any] = []
    elif isinstance(raw_cost, (list, tuple)):
        cost = [_safe_json_value(value, field="attack.cost") for value in raw_cost]
    else:
        cost = [_safe_json_value(raw_cost, field="attack.cost")]
    return {
        "attack_id": attack_id,
        "available": True,
        "cost": cost,
        "damage": _exact_int(record.get("damage"), field="metadata.attack.damage", minimum=0, maximum=MAX_HP_VALUE, optional=True),
        "public_mechanics": _structured_mechanics(record),
    }


def _metadata_card_summary(
    card_id: int,
    *,
    cards: Mapping[int, Mapping[str, Any]],
    attacks: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    record = cards.get(card_id)
    if record is None:
        return {"card_id": card_id, "available": False}
    stage = "unknown"
    for name, canonical in (("basic", "basic"), ("stage1", "stage1"), ("stage2", "stage2")):
        if bool(record.get(name, False)):
            stage = canonical
            break
    raw_attacks = _rows(record.get("attacks"), field="metadata.card.attacks")
    attack_ids = [
        attack_id
        for index, value in enumerate(raw_attacks)
        if (attack_id := _attack_id(value, field=f"metadata.card.attacks[{index}]")) is not None
    ]
    skill_rows = _rows(record.get("skills"), field="metadata.card.skills")
    skills: list[dict[str, Any]] = []
    for index, skill in enumerate(skill_rows):
        mapping = _mapping(skill, field=f"metadata.card.skills[{index}]")
        skills.append(
            {
                "card_id": card_id,
                # Index is stable under a metadata-catalog identity; it is a
                # skill identity, not a candidate ordinal.
                "skill_index": index,
                "skill_id": _exact_int(
                    _first_present(mapping, ("skillId", "id")),
                    field="metadata.skillId",
                    minimum=0,
                    maximum=MAX_SELECTION_COUNT,
                    optional=True,
                ),
                "public_mechanics": _structured_mechanics(mapping),
            }
        )
    # This is the visible card-class default only.  A simulator-exposed exact
    # yield or a visible Prize modifier belongs to the selected-transition
    # target compiler, never a text-derived policy feature.  The ordering is
    # deliberate: catalogs that expose both flags are valid and Mega ex wins.
    mega_ex = bool(record.get("megaEx", record.get("mega_ex", False)))
    ordinary_ex = bool(record.get("ex", False))
    default_prize_yield = 3 if mega_ex else 2 if ordinary_ex else 1
    return {
        "card_id": card_id,
        "available": True,
        "card_type": _safe_json_value(_first_present(record, ("cardType", "card_type")), field="metadata.cardType"),
        "pokemon_type": _safe_json_value(_first_present(record, ("pokemonType", "energyType", "type")), field="metadata.pokemonType"),
        "stage": stage,
        "evolves_from": _safe_json_value(_first_present(record, ("evolvesFrom", "evolves_from")), field="metadata.evolvesFrom"),
        "hp": _exact_int(record.get("hp"), field="metadata.card.hp", minimum=0, maximum=MAX_HP_VALUE, optional=True),
        "retreat": _exact_int(_first_present(record, ("retreatCost", "retreat")), field="metadata.card.retreat", minimum=0, maximum=MAX_SELECTION_COUNT, optional=True),
        "ex": ordinary_ex,
        "mega_ex": mega_ex,
        "default_prize_yield": default_prize_yield,
        "prize_class": (
            "mega_ex" if mega_ex else "ordinary_ex" if ordinary_ex else "ordinary_pokemon"
        ),
        "tera": bool(record.get("tera", False)),
        "ace_spec": bool(record.get("aceSpec", record.get("ace_spec", False))),
        "weakness": _safe_json_value(record.get("weakness"), field="metadata.card.weakness"),
        "resistance": _safe_json_value(record.get("resistance"), field="metadata.card.resistance"),
        "attacks": [
            _metadata_attack_summary(attack_id, attacks=attacks)
            for attack_id in attack_ids
        ],
        "skills": skills,
        "public_mechanics": _structured_mechanics(record),
    }


def _metadata_summary(
    state: Mapping[str, Any],
    selection: Mapping[str, Any],
    options: Sequence[Mapping[str, Any]],
    *,
    cards: Mapping[int, Mapping[str, Any]],
    attacks: Mapping[int, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    card_ids, attack_ids = _referenced_ids({"state": state, "selection": selection, "options": list(options)})
    if not cards and not attacks:
        return {
            "available": False,
            "reason": str(provenance.get("reason", "metadata_catalog_not_supplied")),
            "referenced_card_ids": list(card_ids),
            "referenced_attack_ids": list(attack_ids),
        }
    return {
        "available": True,
        "provenance": dict(provenance),
        "referenced_card_ids": list(card_ids),
        "referenced_attack_ids": list(attack_ids),
        "cards": [
            _metadata_card_summary(card_id, cards=cards, attacks=attacks)
            for card_id in card_ids
        ],
        "attacks": [
            _metadata_attack_summary(attack_id, attacks=attacks)
            for attack_id in attack_ids
        ],
    }


@dataclass(frozen=True)
class PublicRuleOption:
    """One option's order-free public semantic key and metadata references."""

    semantic: Mapping[str, Any]
    semantic_key_sha256: str
    referenced_card_ids: tuple[int, ...]
    referenced_attack_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic": copy.deepcopy(dict(self.semantic)),
            "semantic_key_sha256": self.semantic_key_sha256,
            "referenced_card_ids": list(self.referenced_card_ids),
            "referenced_attack_ids": list(self.referenced_attack_ids),
        }


@dataclass(frozen=True)
class PublicRuleRepresentation:
    """Public, simulator-shaped representation for a single legal stage."""

    schema: str
    revision: int
    public_observation_hash: str
    semantic_token_hash: str
    canonical_option_multiset_hash: str
    state: Mapping[str, Any]
    selection: Mapping[str, Any]
    options: tuple[PublicRuleOption, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "public_observation_hash": self.public_observation_hash,
            "semantic_token_hash": self.semantic_token_hash,
            "canonical_option_multiset_hash": self.canonical_option_multiset_hash,
            "state": copy.deepcopy(dict(self.state)),
            "selection": copy.deepcopy(dict(self.selection)),
            # This sequence aligns with the supplied legal options.  The row
            # contents intentionally carry no list ordinal; multiset hash is
            # the order-independent identity used for collision work.
            "options": [option.to_dict() for option in self.options],
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


def semantic_option_key(
    observation: Mapping[str, Any],
    option: Mapping[str, Any],
    *,
    metadata_catalog: Any = None,
    allow_test_catalog: bool = False,
) -> dict[str, Any]:
    """Return one order-free public semantic option key.

    This convenience boundary is useful for Phase A and metamorphic tests.
    It never receives a candidate ordinal, does not emit a raw serial, and
    intentionally does *not* attach a simulator discriminator: that step is
    valid only after comparison with colliding semantic rows in the complete
    legal list, which :func:`build_public_rule_representation` performs.
    """

    raw_observation = _mapping(observation, field="observation")
    public = sanitize_public_observation(raw_observation)
    current = _mapping(public.get("current"), field="observation.current")
    actor = _exact_int(
        current.get("yourIndex"),
        field="current.yourIndex",
        minimum=0,
        maximum=1,
    )
    assert actor is not None
    raw_select = _mapping(raw_observation.get("select"), field="observation.select")
    metadata_cards, _metadata_attacks, _metadata_provenance = _metadata_maps(
        metadata_catalog, allow_test_catalog=allow_test_catalog
    )
    locators = _locators_for_public_cards(raw_observation, actor=actor)
    selection = _selection_state(
        raw_select,
        actor=actor,
        locators=locators,
        metadata_cards=metadata_cards,
    )
    base = _option_semantics(
        _mapping(option, field="option"),
        selection=selection,
        actor=actor,
        locators=locators,
    )
    return {"selection": selection, "option": base}


def build_public_rule_representation(
    observation: Mapping[str, Any],
    *,
    metadata_catalog: Any = None,
    allow_test_catalog: bool = False,
) -> PublicRuleRepresentation:
    """Build an exact, order-free semantic representation of a legal stage.

    The returned ``options`` preserve the caller's order solely to align later
    model outputs to the simulator legal list.  No option semantic key contains
    that ordinal.  Reordering legal options therefore only reorders rows unless
    the simulator supplies a stable discriminator that represents a real
    semantic distinction.
    """

    raw_observation = _mapping(observation, field="observation")
    public = sanitize_public_observation(raw_observation)
    current = _mapping(public.get("current"), field="observation.current")
    actor = _exact_int(current.get("yourIndex"), field="current.yourIndex", minimum=0, maximum=1)
    assert actor is not None
    players = _rows(current.get("players"), field="current.players", optional=False)
    raw_select = _mapping(raw_observation.get("select"), field="observation.select")
    metadata_cards, metadata_attacks, metadata_provenance = _metadata_maps(
        metadata_catalog, allow_test_catalog=allow_test_catalog
    )
    locators = _locators_for_public_cards(raw_observation, actor=actor)

    state = {
        "turn": _exact_int(current.get("turn"), field="current.turn", minimum=0, optional=True),
        "turn_action_count": _exact_int(
            current.get("turnActionCount"),
            field="current.turnActionCount",
            minimum=0,
            maximum=MAX_TURN_ACTION_COUNT,
            optional=True,
        ),
        "supporter_played": _optional_bool(current.get("supporterPlayed"), field="current.supporterPlayed"),
        "stadium_played": _optional_bool(current.get("stadiumPlayed"), field="current.stadiumPlayed"),
        "energy_attached": _optional_bool(current.get("energyAttached"), field="current.energyAttached"),
        "retreated": _optional_bool(current.get("retreated"), field="current.retreated"),
        "players": {
            "acting": _player_state(_mapping(players[actor], field="acting player"), relation="acting", metadata_cards=metadata_cards),
            "opponent": _player_state(_mapping(players[1 - actor], field="opponent player"), relation="opponent", metadata_cards=metadata_cards),
        },
        "stadium": _menu_signature(current.get("stadium"), metadata_cards=metadata_cards),
        "looking": _menu_signature(current.get("looking"), metadata_cards=metadata_cards),
    }
    selection = _selection_state(
        raw_select,
        actor=actor,
        locators=locators,
        metadata_cards=metadata_cards,
    )

    raw_options = _rows(raw_select.get("option"), field="select.option", optional=False)
    base_rows: list[dict[str, Any]] = []
    discriminators: list[Optional[str]] = []
    for raw_option in raw_options:
        option = _mapping(raw_option, field="select.option[]")
        base_rows.append(
            _option_semantics(
                option,
                selection=selection,
                actor=actor,
                locators=locators,
            )
        )
        discriminators.append(_stable_simulator_discriminator(option))

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, base in enumerate(base_rows):
        grouped[_canonical_json({"selection": selection, "option": base})].append(index)
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        explicit = [discriminators[index] for index in indexes]
        distinct = set(explicit)
        # Duplicate semantically identical options intentionally remain
        # equivalent.  The collision census is responsible for proving their
        # simulator successors are also equivalent.  If the simulator supplies
        # distinct semantic identities, use them—never their candidate ordinal.
        if len(distinct) > 1:
            if None in distinct:
                raise PublicRuleAdapterError(
                    "semantic option collision has only a partial stable simulator discriminator"
                )
            for index in indexes:
                base_rows[index]["stable_simulator_discriminator"] = discriminators[index]

    public_hash = _sha256({"state": state, "selection": selection})
    options: list[PublicRuleOption] = []
    semantic_hashes: list[str] = []
    for base in base_rows:
        semantic = {"selection": selection, "option": base}
        semantic_hash = _sha256(semantic)
        cards, attacks = _referenced_ids(semantic)
        options.append(
            PublicRuleOption(
                semantic=semantic,
                semantic_key_sha256=semantic_hash,
                referenced_card_ids=cards,
                referenced_attack_ids=attacks,
            )
        )
        semantic_hashes.append(semantic_hash)
    option_multiset_hash = _sha256(sorted(semantic_hashes))
    metadata = _metadata_summary(
        state,
        selection,
        [option.semantic for option in options],
        cards=metadata_cards,
        attacks=metadata_attacks,
        provenance=metadata_provenance,
    )
    token_hash = _sha256(
        {
            "schema": PUBLIC_RULE_REPRESENTATION_SCHEMA,
            "state": state,
            "selection": selection,
            "option_multiset": sorted(semantic_hashes),
            "metadata": metadata,
        }
    )
    return PublicRuleRepresentation(
        schema=PUBLIC_RULE_REPRESENTATION_SCHEMA,
        revision=PUBLIC_RULE_ADAPTER_REVISION,
        public_observation_hash=public_hash,
        semantic_token_hash=token_hash,
        canonical_option_multiset_hash=option_multiset_hash,
        state=state,
        selection=selection,
        options=tuple(options),
        metadata=metadata,
    )


def public_rule_observation_fingerprint(observation: Mapping[str, Any]) -> str:
    """Return the serial- and option-order-invariant public stage identity.

    This is the correct r298 fingerprint for collision census, target-chain,
    and restoration binding.  It consumes raw observations solely to resolve a
    visible serial into the local owner/area/slot binding; neither the raw
    serial nor a candidate ordinal is hashed.  Terminal and future payloads
    are excluded by the public representation boundary.
    """

    representation = build_public_rule_representation(observation)
    return _sha256(
        {
            "schema": PUBLIC_RULE_REPRESENTATION_SCHEMA,
            "revision": PUBLIC_RULE_ADAPTER_REVISION,
            "public_state": representation.public_observation_hash,
            "selection": representation.selection,
            "canonical_option_multiset": representation.canonical_option_multiset_hash,
        }
    )


_ResidualBase = object if nn is None else nn.Module


class PublicRuleMetadataResidual(_ResidualBase):
    """Unwired, exactly-off structured residual after frozen Card2Vec.

    This class deliberately never instantiates or consumes legacy
    ``MetadataCatalog`` vectors: those carry the frozen name/text-hash tail
    used by the existing Card2Vec system.  Instead it projects the receipt
    sealed 64/20 structured tensor surface into the *already-built* Card2Vec
    option hidden vector.  Consequently Card2Vec inputs, parameters, buffers,
    and runtime behavior remain bit-identical; this is only an additive path
    after that frozen representation.
    """

    def __init__(
        self,
        catalog: SealedPublicCatalog,
        *,
        d_model: int,
        vectors: Optional[SealedPublicCatalogVectors] = None,
    ) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(catalog, SealedPublicCatalog) or not is_public_catalog_eligible(catalog):
            raise PublicRuleAdapterError(
                "PublicRuleMetadataResidual requires the default-pinned SealedPublicCatalog"
            )
        if int(d_model) <= 0:
            raise PublicRuleAdapterError("d_model must be positive")
        structured_vectors = vectors or load_sealed_public_catalog_vectors(catalog)
        if not isinstance(structured_vectors, SealedPublicCatalogVectors):
            raise PublicRuleAdapterError("structured residual vectors have an invalid type")
        if structured_vectors.provenance != catalog.provenance:
            raise PublicRuleAdapterError("structured residual vector/catalog provenance mismatch")
        if structured_vectors.vector_schema != PUBLIC_STRUCTURED_VECTORS_SCHEMA:
            raise PublicRuleAdapterError("structured residual vector schema mismatch")
        card_features = structured_vectors.card_features
        attack_features = structured_vectors.attack_features
        if (
            not isinstance(card_features, torch.Tensor)
            or not isinstance(attack_features, torch.Tensor)
            or card_features.ndim != 2
            or attack_features.ndim != 2
            or int(card_features.size(1)) != 64
            or int(attack_features.size(1)) != 20
        ):
            raise PublicRuleAdapterError("structured residual requires 64/20 sealed vector tensors")
        self.card_vocab = int(catalog.card_vocab)
        self.attack_vocab = int(catalog.attack_vocab)
        self.d_model = int(d_model)
        self.register_buffer("structured_card_metadata", card_features.detach().clone(), persistent=True)
        self.register_buffer("structured_attack_metadata", attack_features.detach().clone(), persistent=True)
        self.card_projection = nn.Linear(64, self.d_model, bias=False)
        self.attack_projection = nn.Linear(20, self.d_model, bias=False)
        self.card_gate = nn.Parameter(torch.zeros(()))
        self.attack_gate = nn.Parameter(torch.zeros(()))
        self.catalog_provenance = catalog.provenance

    @property
    def metadata_provenance(self) -> dict[str, Any]:
        """Compatibility/audit view; no legacy metadata surface is exposed."""

        return self.catalog_provenance.to_dict()

    def option_metadata_features(
        self, representation: PublicRuleRepresentation
    ) -> tuple[Tensor, Tensor]:
        """Mean-pool only card/attack identities named by each semantic row."""

        card_rows: list[Tensor] = []
        attack_rows: list[Tensor] = []
        for option in representation.options:
            card_ids = list(option.referenced_card_ids)
            attack_ids = list(option.referenced_attack_ids)
            if any(card_id <= 0 or card_id >= self.card_vocab for card_id in card_ids):
                raise PublicRuleAdapterError("option card id is outside sealed structured catalog")
            if any(attack_id <= 0 or attack_id >= self.attack_vocab for attack_id in attack_ids):
                raise PublicRuleAdapterError("option attack id is outside sealed structured catalog")
            if card_ids:
                card_rows.append(self.structured_card_metadata[torch.tensor(card_ids, device=self.structured_card_metadata.device)].mean(dim=0))
            else:
                card_rows.append(torch.zeros(64, dtype=self.structured_card_metadata.dtype, device=self.structured_card_metadata.device))
            if attack_ids:
                attack_rows.append(self.structured_attack_metadata[torch.tensor(attack_ids, device=self.structured_attack_metadata.device)].mean(dim=0))
            else:
                attack_rows.append(torch.zeros(20, dtype=self.structured_attack_metadata.dtype, device=self.structured_attack_metadata.device))
        if not card_rows:
            return (
                torch.zeros((0, 64), dtype=self.structured_card_metadata.dtype, device=self.structured_card_metadata.device),
                torch.zeros((0, 20), dtype=self.structured_attack_metadata.dtype, device=self.structured_attack_metadata.device),
            )
        return torch.stack(card_rows), torch.stack(attack_rows)

    def augment_option_hidden(
        self,
        base: Tensor,
        representation: PublicRuleRepresentation,
    ) -> Tensor:
        """Apply the two zero-initialized metadata residuals to option vectors."""

        # Exact-off is a *bypass*, rather than algebraically adding ``0 *
        # delta``.  Besides avoiding needless work, this preserves object and
        # byte identity even if a malformed/uninitialized residual contains
        # NaN, signed zero, or an implementation-specific payload.
        card_gate = float(self.card_gate.detach().cpu())
        attack_gate = float(self.attack_gate.detach().cpu())
        if card_gate == 0.0 and attack_gate == 0.0:
            return base
        if not math.isfinite(card_gate) or not math.isfinite(attack_gate):
            raise PublicRuleAdapterError("public-rule metadata gates must be finite")
        if base.ndim != 2 or base.size(0) != len(representation.options) or base.size(1) != self.d_model:
            raise PublicRuleAdapterError(
                "option hidden shape must be [legal_option_count, d_model] "
                f"(got {tuple(base.shape)}, expected [{len(representation.options)}, {self.d_model}])"
            )
        card_features, attack_features = self.option_metadata_features(representation)
        # A disabled *individual* branch is also a true bypass.  In
        # particular, an uninitialized/poisoned card projection cannot leak a
        # NaN through ``0 * card_delta`` while only the attack branch is
        # enabled.  Conversely, every nonzero branch checks its actual delta
        # before adding it to the otherwise frozen Card2Vec hidden state.
        result = base
        if card_gate != 0.0:
            card_delta = self.card_projection(card_features).to(
                dtype=base.dtype, device=base.device
            )
            if not torch.isfinite(card_delta).all():
                raise PublicRuleAdapterError("public-rule card residual must be finite")
            result = result + self.card_gate.to(dtype=base.dtype) * card_delta
        if attack_gate != 0.0:
            attack_delta = self.attack_projection(attack_features).to(
                dtype=base.dtype, device=base.device
            )
            if not torch.isfinite(attack_delta).all():
                raise PublicRuleAdapterError("public-rule attack residual must be finite")
            result = result + self.attack_gate.to(dtype=base.dtype) * attack_delta
        return result

    def checkpoint_contract(self) -> dict[str, Any]:
        return {
            "schema": PUBLIC_RULE_ADAPTER_SCHEMA,
            "revision": PUBLIC_RULE_ADAPTER_REVISION,
            "module": type(self).__name__,
            "integration": "additive_after_frozen_card2vec_option_hidden_structured_residual",
            "runtime_wired": False,
            "legacy_exact_when_gates_zero": True,
            "card2vec_inputs_or_tensors_mutated": False,
            "structured_only": True,
            "text_or_name_hash_mechanics_authority": False,
            "card_gate": float(self.card_gate.detach().cpu()),
            "attack_gate": float(self.attack_gate.detach().cpu()),
            "metadata_provenance": self.metadata_provenance,
        }


def apply_zero_gated_logit_residual(
    base_logits: Tensor,
    residual: Tensor,
    *,
    gate: float | Tensor = 0.0,
) -> Tensor:
    """Generic exact-off final-logit helper for an unwired future derivative."""

    _require_torch()
    gate_tensor = torch.as_tensor(gate, dtype=base_logits.dtype, device=base_logits.device)
    if gate_tensor.numel() != 1 or not torch.isfinite(gate_tensor).all():
        raise PublicRuleAdapterError("public-rule gate must be one finite scalar")
    # The disabled path must not inspect or materialize residual bytes.  A
    # future caller may keep an uninitialized/nonfinite candidate residual
    # while the feature is off; it has no authority and cannot perturb the
    # exact parent logits.
    if float(gate_tensor.detach().cpu()) == 0.0:
        return base_logits
    if base_logits.shape != residual.shape:
        raise PublicRuleAdapterError(
            f"logit residual shape mismatch: base={tuple(base_logits.shape)}, residual={tuple(residual.shape)}"
        )
    if not torch.isfinite(residual).all():
        raise PublicRuleAdapterError("public-rule logit residual must be finite")
    return base_logits + gate_tensor.reshape(()) * residual.to(dtype=base_logits.dtype)


def load_public_rule_adapter_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the r298 zero-gated, non-runtime-wired configuration strictly."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicRuleAdapterError(f"cannot load public-rule adapter config {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PUBLIC_RULE_CONFIG_SCHEMA:
        raise PublicRuleAdapterError("public-rule adapter config schema mismatch")
    if payload.get("revision") != PUBLIC_RULE_ADAPTER_REVISION:
        raise PublicRuleAdapterError("public-rule adapter config revision mismatch")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise PublicRuleAdapterError("public-rule adapter config has no runtime block")
    if runtime.get("enabled_default") is not False or runtime.get("runtime_wired") is not False:
        raise PublicRuleAdapterError("r298 adapter must remain default-off and unwired")
    gates = runtime.get("zero_gates")
    expected_gate_names = {
        "card_metadata_residual",
        "attack_metadata_residual",
        "option_logit_residual",
    }
    if (
        not isinstance(gates, Mapping)
        or set(gates) != expected_gate_names
        or any(float(value) != 0.0 for value in gates.values())
    ):
        raise PublicRuleAdapterError("r298 adapter config must bind every gate to exact zero")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        raise PublicRuleAdapterError("public-rule adapter config has no authority binding")
    project_root = Path(__file__).resolve().parents[1]
    for path_key, digest_key in (
        ("canonical_goal", "canonical_goal_sha256"),
        ("canonical_contract", "canonical_contract_sha256"),
    ):
        relative = authority.get(path_key)
        expected_digest = authority.get(digest_key)
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise PublicRuleAdapterError(f"public-rule config omits {path_key} binding")
        bound_path = (project_root / relative).resolve()
        if not bound_path.is_file() or _file_sha256(bound_path) != expected_digest:
            raise PublicRuleAdapterError(
                f"public-rule config has stale or missing {path_key} binding"
            )
    if (
        authority.get("canonical_goal_sha256") != R298_CANONICAL_GOAL_SHA256
        or authority.get("canonical_contract_sha256") != R298_CANONICAL_CONTRACT_SHA256
        or authority.get("root_handoff_revision") != R298_ROOT_OWNER_REVISION
    ):
        raise PublicRuleAdapterError("public-rule config has foreign r298 authority pins")
    migration = authority.get("revision_5_consumer_migration")
    expected_migration = {
        "goal_revision": R298_CANONICAL_GOAL_REVISION,
        "predecessor_goal_revision": R298_PREDECESSOR_GOAL_REVISION,
        "predecessor_gateway_sha256": R298_PREDECESSOR_GOAL_SHA256,
        "predecessor_contract_sha256": R298_PREDECESSOR_CONTRACT_SHA256,
        "revision_4_catalog_is_immutable_predecessor_evidence": True,
        "revision_5_schema_freeze_receipt_required_before_nonzero_or_materializer_eligibility": True,
        "blind_revision_4_hash_substitution_allowed": False,
    }
    if not isinstance(migration, Mapping) or any(
        migration.get(key) != expected for key, expected in expected_migration.items()
    ):
        raise PublicRuleAdapterError("public-rule config has stale r5 consumer migration")
    catalog = payload.get("sealed_public_catalog")
    if not isinstance(catalog, Mapping):
        raise PublicRuleAdapterError("public-rule config has no sealed catalog provenance block")
    expected_catalog_config = {
        "schema": PUBLIC_CATALOG_SCHEMA,
        "receipt_schema": PUBLIC_CATALOG_RECEIPT_SCHEMA,
        "structured_vectors_schema": PUBLIC_STRUCTURED_VECTORS_SCHEMA,
        "catalog_file_sha256": R298_PUBLIC_CATALOG_FILE_SHA256,
        "vectors_file_sha256": R298_PUBLIC_CATALOG_VECTORS_FILE_SHA256,
        "receipt_file_sha256": R298_PUBLIC_CATALOG_RECEIPT_FILE_SHA256,
        "structured_vectors_semantic_sha256": R298_PUBLIC_CATALOG_STRUCTURED_VECTORS_SHA256,
        "engine_cards_sha256": R298_PUBLIC_CATALOG_ENGINE_CARDS_SHA256,
        "engine_attacks_sha256": R298_PUBLIC_CATALOG_ENGINE_ATTACKS_SHA256,
        "libcg_sha256": R298_CANONICAL_LIBCG_SHA256,
        "libcg_size_bytes": R298_CANONICAL_LIBCG_SIZE_BYTES,
        "typed_libcg_receipt_sha256": R298_CANONICAL_LIBCG_TYPED_SOURCE_SHA256,
        "csv_identity_join_sha256": R298_PUBLIC_CATALOG_CSV_SHA256,
        "execution_host_role": "elmo",
        "execution_hostname": "truenas",
        "authority_status": "immutable_revision_4_predecessor_evidence_only",
        "revision_5_schema_freeze_receipt_required_before_policy_or_materializer_eligibility": True,
        "raw_mapping_policy_or_materializer_eligible": False,
        "legacy_text_hashed_metadata_eligible": False,
        "missing_or_mismatched_provenance_behavior": "unavailable_and_exact_zero",
    }
    mismatches = {
        key: {"expected": expected, "actual": catalog.get(key)}
        for key, expected in expected_catalog_config.items()
        if catalog.get(key) != expected
    }
    if mismatches:
        raise PublicRuleAdapterError(
            "public-rule config has stale sealed catalog pins: "
            + _canonical_json(mismatches)
        )
    return payload


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MAX_NUMBER_VALUE",
    "PUBLIC_RULE_ADAPTER_REVISION",
    "R298_CANONICAL_GOAL_REVISION",
    "R298_CANONICAL_GOAL_SHA256",
    "R298_CANONICAL_CONTRACT_SHA256",
    "R298_ROOT_OWNER_REVISION",
    "R298_PREDECESSOR_GOAL_REVISION",
    "R298_PREDECESSOR_GOAL_SHA256",
    "R298_PREDECESSOR_CONTRACT_SHA256",
    "PUBLIC_RULE_ADAPTER_SCHEMA",
    "PUBLIC_CATALOG_RECEIPT_SCHEMA",
    "PUBLIC_CATALOG_SCHEMA",
    "PUBLIC_STRUCTURED_VECTORS_SCHEMA",
    "PUBLIC_RULE_CONFIG_SCHEMA",
    "PUBLIC_RULE_REPRESENTATION_SCHEMA",
    "DEFAULT_PUBLIC_CATALOG_PINS",
    "PublicCatalogPins",
    "PublicCatalogProvenance",
    "PublicRuleAdapterError",
    "PublicRuleMetadataResidual",
    "PublicRuleOption",
    "PublicRuleRepresentation",
    "SealedPublicCatalog",
    "SealedPublicCatalogVectors",
    "apply_zero_gated_logit_residual",
    "build_public_rule_representation",
    "is_public_catalog_eligible",
    "load_sealed_public_catalog",
    "load_sealed_public_catalog_vectors",
    "load_public_rule_adapter_config",
    "public_rule_observation_fingerprint",
    "sanitize_public_observation",
    "semantic_option_key",
    "validate_public_catalog_provenance",
]
