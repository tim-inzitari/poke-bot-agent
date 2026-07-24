"""Fail-closed, deterministic card-mechanics metadata for ``FactorizedCard2Vec``.

The competition engine is the authority for mechanics.  ``EN_Card_Data.csv``
is only a provenance-checked supplemental identity join: its rows have mixed
grain (attacks, abilities and Tera markers), so treating them as one attack per
row silently corrupts the attack table.

This module deliberately does *not* alter the production model.  It provides a
fixed metadata catalog and a zero-gated residual that can be inserted after the
existing learned card/attack identity lookup and before role composition.  A
legacy checkpoint is therefore bit-exact while both gates remain zero.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


CARD_METADATA_SCHEMA = "pokebot.card-metadata.v1"
CARD_STRUCTURED_DIM = 64
ATTACK_STRUCTURED_DIM = 20
DEFAULT_CARD_METADATA_DIM = 128
DEFAULT_ATTACK_METADATA_DIM = 96

# cg.api enum domains.  Keeping these explicit makes unexpected engine schema
# changes fail instead of being folded into a hashed bucket.
CARD_TYPE_COUNT = 7
ENERGY_TYPE_COUNT = 12

_TOKEN_RE = re.compile(r"[\w{}+'’-]+", re.UNICODE)


class MetadataContractError(ValueError):
    """The engine catalog, CSV join, or checkpoint provenance is inconsistent."""


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def _normalized_name(value: Any) -> str:
    return _canonical_text(value).casefold()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MetadataContractError(
            f"{field_name} is not an integer: {value!r}"
        ) from exc


def _canonical_attack(attack: Any) -> dict[str, Any]:
    attack_id = _as_int(_field(attack, "attackId"), field_name="attackId")
    energies = tuple(
        _as_int(value, field_name=f"attack[{attack_id}].energies")
        for value in (_field(attack, "energies", []) or [])
    )
    if any(value < 0 or value >= ENERGY_TYPE_COUNT for value in energies):
        raise MetadataContractError(
            f"attackId={attack_id} contains an unknown energy type: {energies}"
        )
    damage = _as_int(_field(attack, "damage", 0), field_name="damage")
    if damage < 0:
        raise MetadataContractError(f"attackId={attack_id} has negative damage")
    return {
        "attackId": attack_id,
        "name": _canonical_text(_field(attack, "name")),
        "text": _canonical_text(_field(attack, "text")),
        "damage": damage,
        "energies": list(energies),
    }


def _canonical_card(card: Any) -> dict[str, Any]:
    card_id = _as_int(_field(card, "cardId"), field_name="cardId")
    card_type = _as_int(_field(card, "cardType"), field_name="cardType")
    energy_type = _as_int(_field(card, "energyType", 0), field_name="energyType")
    if card_type < 0 or card_type >= CARD_TYPE_COUNT:
        raise MetadataContractError(
            f"cardId={card_id} has unknown cardType={card_type}"
        )
    if energy_type < 0 or energy_type >= ENERGY_TYPE_COUNT:
        raise MetadataContractError(
            f"cardId={card_id} has unknown energyType={energy_type}"
        )

    stage_flags = tuple(
        bool(_field(card, name, False)) for name in ("basic", "stage1", "stage2")
    )
    if sum(stage_flags) > 1:
        raise MetadataContractError(f"cardId={card_id} has contradictory stage flags")
    # Do not require cardType=POKEMON here.  Five Fossil Item cards intentionally
    # expose basic=True and HP=60 in the competition engine.
    if card_type == 0 and sum(stage_flags) != 1:
        raise MetadataContractError(
            f"Pokémon cardId={card_id} does not have exactly one stage flag"
        )

    hp = _as_int(_field(card, "hp", 0), field_name="hp")
    retreat = _as_int(_field(card, "retreatCost", 0), field_name="retreatCost")
    if hp < 0 or retreat < 0:
        raise MetadataContractError(f"cardId={card_id} has negative HP/retreat")

    attack_ids = [
        _as_int(value, field_name=f"card[{card_id}].attacks")
        for value in (_field(card, "attacks", []) or [])
    ]
    if len(attack_ids) != len(set(attack_ids)):
        raise MetadataContractError(f"cardId={card_id} repeats an attack reference")

    skills = []
    for skill in _field(card, "skills", []) or []:
        skills.append(
            {
                "name": _canonical_text(_field(skill, "name")),
                "text": _canonical_text(_field(skill, "text")),
            }
        )
    ex = bool(_field(card, "ex", False))
    mega_ex = bool(_field(card, "megaEx", False))
    if ex and mega_ex:
        raise MetadataContractError(f"cardId={card_id} is both ex and megaEx")

    weakness = _field(card, "weakness")
    resistance = _field(card, "resistance")
    weakness = None if weakness is None else _as_int(weakness, field_name="weakness")
    resistance = (
        None if resistance is None else _as_int(resistance, field_name="resistance")
    )
    for label, value in (("weakness", weakness), ("resistance", resistance)):
        if value is not None and not 0 <= value < ENERGY_TYPE_COUNT:
            raise MetadataContractError(
                f"cardId={card_id} has unknown {label} type={value}"
            )

    return {
        "cardId": card_id,
        "name": _canonical_text(_field(card, "name")),
        "cardType": card_type,
        "retreatCost": retreat,
        "hp": hp,
        "weakness": weakness,
        "resistance": resistance,
        "energyType": energy_type,
        "basic": stage_flags[0],
        "stage1": stage_flags[1],
        "stage2": stage_flags[2],
        "ex": ex,
        "megaEx": mega_ex,
        "tera": bool(_field(card, "tera", False)),
        "aceSpec": bool(_field(card, "aceSpec", False)),
        "evolvesFrom": (
            None
            if _field(card, "evolvesFrom") is None
            else _canonical_text(_field(card, "evolvesFrom"))
        ),
        "skills": skills,
        "attacks": attack_ids,
    }


def _validate_contiguous(records: Sequence[dict[str, Any]], key: str) -> int:
    ids = [int(record[key]) for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted(value for value in set(ids) if ids.count(value) > 1)
        raise MetadataContractError(f"duplicate {key} values: {duplicates[:10]}")
    expected = list(range(1, max(ids, default=0) + 1))
    if sorted(ids) != expected:
        missing = sorted(set(expected).difference(ids))
        raise MetadataContractError(
            f"{key} domain must be contiguous 1..max; missing={missing[:10]}"
        )
    return max(ids, default=0) + 1


def _csv_identity_join(
    path: Path, cards: Sequence[dict[str, Any]]
) -> tuple[dict[int, tuple[dict[str, str], ...]], str]:
    if not path.is_file():
        raise MetadataContractError(f"card CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Card ID", "Card Name", "Move Name", "Effect Explanation"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise MetadataContractError(
                f"card CSV is missing columns: {sorted(missing_columns)}"
            )
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row_number, row in enumerate(reader, start=2):
            try:
                card_id = int(row["Card ID"])
            except (TypeError, ValueError) as exc:
                raise MetadataContractError(
                    f"invalid Card ID at CSV row {row_number}: {row.get('Card ID')!r}"
                ) from exc
            grouped[card_id].append({str(k): str(v or "") for k, v in row.items()})

    engine_ids = {int(card["cardId"]) for card in cards}
    csv_ids = set(grouped)
    if engine_ids != csv_ids:
        raise MetadataContractError(
            "card CSV identity domain differs from engine: "
            f"missing={sorted(engine_ids - csv_ids)[:10]}, "
            f"extra={sorted(csv_ids - engine_ids)[:10]}"
        )

    variant_columns = {"Move Name", "Cost", "Damage", "Effect Explanation"}
    for card in cards:
        card_id = int(card["cardId"])
        rows = grouped[card_id]
        csv_names = {_canonical_text(row["Card Name"]) for row in rows}
        if csv_names != {card["name"]}:
            raise MetadataContractError(
                f"cardId={card_id} name mismatch: engine={card['name']!r}, "
                f"csv={sorted(csv_names)!r}"
            )
        invariant_columns = set(rows[0]).difference(variant_columns)
        for column in invariant_columns:
            values = {_canonical_text(row[column]) for row in rows}
            if len(values) != 1:
                raise MetadataContractError(
                    f"cardId={card_id} has mixed {column!r} values in CSV"
                )
    frozen = {card_id: tuple(rows) for card_id, rows in sorted(grouped.items())}
    return frozen, _sha256_file(path)


def _text_tokens(fields: Iterable[tuple[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for field_name, value in fields:
        words = [word.casefold() for word in _TOKEN_RE.findall(_canonical_text(value))]
        tokens.extend(f"{field_name}:{word}" for word in words)
        tokens.extend(
            f"{field_name}:bigram:{left}_{right}"
            for left, right in zip(words, words[1:])
        )
    return tokens


def _add_signed_hash(vector: Tensor, *, offset: int, tokens: Sequence[str]) -> None:
    width = int(vector.numel()) - int(offset)
    if width <= 0 or not tokens:
        return
    scale = 1.0 / math.sqrt(len(tokens))
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little", signed=False)
        index = offset + (raw % width)
        sign = -1.0 if raw & (1 << 63) else 1.0
        vector[index] += sign * scale


def _build_attack_features(
    records: Sequence[dict[str, Any]], attack_vocab: int, dimension: int
) -> Tensor:
    if dimension < ATTACK_STRUCTURED_DIM + 8:
        raise MetadataContractError(
            f"attack metadata dimension {dimension} is too small"
        )
    result = torch.zeros(attack_vocab, dimension, dtype=torch.float32)
    for attack in records:
        aid = int(attack["attackId"])
        row = result[aid]
        energies = [int(value) for value in attack["energies"]]
        row[0] = 1.0
        row[1] = min(float(attack["damage"]) / 400.0, 4.0)
        row[2] = min(len(energies) / 5.0, 2.0)
        row[3] = float(not energies)
        for energy_type in energies:
            row[4 + energy_type] += 0.25
        row[16] = float(bool(attack["text"]))
        _add_signed_hash(
            row,
            offset=ATTACK_STRUCTURED_DIM,
            tokens=_text_tokens(
                (("attack_name", attack["name"]), ("attack_text", attack["text"]))
            ),
        )
    return result


def _build_card_features(
    records: Sequence[dict[str, Any]],
    attack_by_id: Mapping[int, dict[str, Any]],
    card_vocab: int,
    dimension: int,
) -> Tensor:
    if dimension < CARD_STRUCTURED_DIM + 8:
        raise MetadataContractError(f"card metadata dimension {dimension} is too small")
    result = torch.zeros(card_vocab, dimension, dtype=torch.float32)
    for card in records:
        cid = int(card["cardId"])
        row = result[cid]
        attacks = [attack_by_id[int(aid)] for aid in card["attacks"]]
        costs = [len(attack["energies"]) for attack in attacks]
        damages = [int(attack["damage"]) for attack in attacks]
        row[0] = 1.0
        row[1] = min(float(card["hp"]) / 400.0, 2.0)
        row[2] = min(float(card["retreatCost"]) / 5.0, 2.0)
        row[3] = min(len(attacks) / 4.0, 2.0)
        row[4] = min(len(card["skills"]) / 4.0, 2.0)
        for index, flag in enumerate(
            (
                card["ex"],
                card["megaEx"],
                card["tera"],
                card["aceSpec"],
                card["basic"],
                card["stage1"],
                card["stage2"],
                bool(card["evolvesFrom"]),
                card["weakness"] is not None,
                card["resistance"] is not None,
            ),
            start=5,
        ):
            row[index] = float(flag)
        row[15] = min(sum(damages) / 800.0, 4.0)
        row[16] = min(max(damages, default=0) / 400.0, 4.0)
        row[17] = min(sum(costs) / 10.0, 4.0)
        row[18 + int(card["cardType"])] = 1.0
        row[25 + int(card["energyType"])] = 1.0
        weakness = card["weakness"]
        resistance = card["resistance"]
        if weakness is not None:
            row[37 + int(weakness)] = 1.0
        if resistance is not None:
            row[49 + int(resistance)] = 1.0

        text_fields: list[tuple[str, Any]] = [
            ("card_name", card["name"]),
            ("evolves_from", card["evolvesFrom"]),
        ]
        for skill in card["skills"]:
            text_fields.extend(
                (("skill_name", skill["name"]), ("skill_text", skill["text"]))
            )
        for attack in attacks:
            text_fields.extend(
                (
                    ("attack_name", attack["name"]),
                    ("attack_text", attack["text"]),
                )
            )
        _add_signed_hash(
            row,
            offset=CARD_STRUCTURED_DIM,
            tokens=_text_tokens(text_fields),
        )
    return result


def _tensor_digest(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class MetadataCatalog:
    """Validated engine metadata, fixed features, and immutable provenance."""

    cards: tuple[dict[str, Any], ...]
    attacks: tuple[dict[str, Any], ...]
    card_features: Tensor
    attack_features: Tensor
    evolution_parents: Mapping[int, tuple[int, ...]]
    csv_rows_by_card: Mapping[int, tuple[dict[str, str], ...]]
    provenance: Mapping[str, Any]

    @property
    def card_vocab(self) -> int:
        return int(self.card_features.size(0))

    @property
    def attack_vocab(self) -> int:
        return int(self.attack_features.size(0))

    def assert_provenance(self, expected: Mapping[str, Any]) -> None:
        """Require every supplied checkpoint/source provenance field to match."""
        mismatches = {
            key: {"expected": value, "actual": self.provenance.get(key)}
            for key, value in expected.items()
            if self.provenance.get(key) != value
        }
        if mismatches:
            raise MetadataContractError(
                "metadata provenance mismatch: "
                + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
            )


def build_metadata_catalog(
    cards: Sequence[Any],
    attacks: Sequence[Any],
    *,
    csv_path: Optional[Path | str] = None,
    card_dimension: int = DEFAULT_CARD_METADATA_DIM,
    attack_dimension: int = DEFAULT_ATTACK_METADATA_DIM,
) -> MetadataCatalog:
    """Build and validate a deterministic metadata catalog.

    IDs must be unique and contiguous because the current feature ABI uses raw
    engine IDs as embedding rows.  Any catalog drift therefore requires an
    explicit migration, never an OOV/clamp fallback.
    """
    card_records = tuple(
        sorted((_canonical_card(card) for card in cards), key=lambda x: x["cardId"])
    )
    attack_records = tuple(
        sorted(
            (_canonical_attack(attack) for attack in attacks),
            key=lambda x: x["attackId"],
        )
    )
    card_vocab = _validate_contiguous(card_records, "cardId")
    attack_vocab = _validate_contiguous(attack_records, "attackId")
    attack_by_id = {int(record["attackId"]): record for record in attack_records}
    for card in card_records:
        missing = [int(aid) for aid in card["attacks"] if int(aid) not in attack_by_id]
        if missing:
            raise MetadataContractError(
                f"cardId={card['cardId']} references missing attacks={missing}"
            )

    by_name: dict[str, list[int]] = defaultdict(list)
    for card in card_records:
        by_name[_normalized_name(card["name"])].append(int(card["cardId"]))
    evolution_parents: dict[int, tuple[int, ...]] = {}
    for card in card_records:
        parent_name = card["evolvesFrom"]
        if not parent_name:
            continue
        parents = tuple(by_name.get(_normalized_name(parent_name), ()))
        if not parents:
            raise MetadataContractError(
                f"cardId={card['cardId']} evolves from unknown name={parent_name!r}"
            )
        evolution_parents[int(card["cardId"])] = parents

    csv_rows: Mapping[int, tuple[dict[str, str], ...]] = {}
    csv_digest: Optional[str] = None
    if csv_path is not None:
        csv_rows, csv_digest = _csv_identity_join(Path(csv_path), card_records)

    card_features = _build_card_features(
        card_records,
        attack_by_id,
        card_vocab,
        int(card_dimension),
    )
    attack_features = _build_attack_features(
        attack_records,
        attack_vocab,
        int(attack_dimension),
    )
    provenance = {
        "schema": CARD_METADATA_SCHEMA,
        "card_count": len(card_records),
        "attack_count": len(attack_records),
        "card_vocab": card_vocab,
        "attack_vocab": attack_vocab,
        "card_dimension": int(card_dimension),
        "attack_dimension": int(attack_dimension),
        "engine_cards_sha256": _sha256_bytes(_canonical_json_bytes(card_records)),
        "engine_attacks_sha256": _sha256_bytes(_canonical_json_bytes(attack_records)),
        "csv_sha256": csv_digest,
        "fixed_vectors_sha256": _tensor_digest(card_features, attack_features),
    }
    return MetadataCatalog(
        cards=card_records,
        attacks=attack_records,
        card_features=card_features,
        attack_features=attack_features,
        evolution_parents=evolution_parents,
        csv_rows_by_card=csv_rows,
        provenance=provenance,
    )


class GatedMetadataResidual(nn.Module):
    """Zero-impact metadata residual for the existing learned identity path.

    Integration point inside ``FactorizedCard2Vec._embed_indices``::

        entity = metadata.augment_entity(entity, kind=kind, entity_id=eid)
        composed = compose(cat(entity, role_embedding))

    The fixed catalog is a persistent buffer.  Projections are trainable, but a
    migrated legacy checkpoint is exactly unchanged because both scalar gates
    initialize to zero.  Opening the gates is a separate, measured training
    decision; this class never substitutes metadata for learned identity.
    """

    KIND_NULL = 0
    KIND_CARD = 1
    KIND_ATTACK = 2

    def __init__(self, catalog: MetadataCatalog, *, d_card: int) -> None:
        super().__init__()
        self.card_vocab = catalog.card_vocab
        self.attack_vocab = catalog.attack_vocab
        self.d_card = int(d_card)
        self.register_buffer(
            "card_metadata", catalog.card_features.detach().clone(), persistent=True
        )
        self.register_buffer(
            "attack_metadata", catalog.attack_features.detach().clone(), persistent=True
        )
        self.card_projection = nn.Linear(
            int(catalog.card_features.size(1)), self.d_card, bias=False
        )
        self.attack_projection = nn.Linear(
            int(catalog.attack_features.size(1)), self.d_card, bias=False
        )
        self.card_gate = nn.Parameter(torch.zeros(()))
        self.attack_gate = nn.Parameter(torch.zeros(()))
        self.provenance = dict(catalog.provenance)

    def augment_entity(
        self, base: Tensor, *, kind: Tensor, entity_id: Tensor
    ) -> Tensor:
        if base.shape[:-1] != kind.shape or kind.shape != entity_id.shape:
            raise MetadataContractError(
                "metadata residual shape mismatch: "
                f"base={tuple(base.shape)}, kind={tuple(kind.shape)}, "
                f"entity_id={tuple(entity_id.shape)}"
            )
        card_mask = kind == self.KIND_CARD
        attack_mask = kind == self.KIND_ATTACK
        if bool(card_mask.any()):
            card_ids = entity_id[card_mask]
            if bool(((card_ids <= 0) | (card_ids >= self.card_vocab)).any()):
                raise MetadataContractError(
                    "card entity ID is outside metadata catalog"
                )
        if bool(attack_mask.any()):
            attack_ids = entity_id[attack_mask]
            if bool(((attack_ids <= 0) | (attack_ids >= self.attack_vocab)).any()):
                raise MetadataContractError(
                    "attack entity ID is outside metadata catalog"
                )

        # Safe indices are only used for gather; invalid typed rows were rejected
        # above, while NULL rows intentionally select the all-zero reserved row.
        card_ids = entity_id.masked_fill(~card_mask, 0)
        attack_ids = entity_id.masked_fill(~attack_mask, 0)
        card_fixed = self.card_metadata[card_ids]
        attack_fixed = self.attack_metadata[attack_ids]
        card_delta = self.card_projection(card_fixed) * card_mask.unsqueeze(-1)
        attack_delta = self.attack_projection(attack_fixed) * attack_mask.unsqueeze(-1)
        return (
            base
            + self.card_gate.to(dtype=base.dtype) * card_delta.to(dtype=base.dtype)
            + self.attack_gate.to(dtype=base.dtype) * attack_delta.to(dtype=base.dtype)
        )

    def checkpoint_contract(self) -> dict[str, Any]:
        """Serializable migration gate for checkpoint manifests."""
        return {
            "module": type(self).__name__,
            "integration": "identity_then_metadata_residual_then_role_compose",
            "legacy_exact_when_gates_zero": True,
            "card_gate": float(self.card_gate.detach().cpu()),
            "attack_gate": float(self.attack_gate.detach().cpu()),
            "metadata_provenance": dict(self.provenance),
        }
