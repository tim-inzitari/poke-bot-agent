from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poke_bot.card_metadata import (
    CARD_METADATA_SCHEMA,
    GatedMetadataResidual,
    MetadataContractError,
    build_metadata_catalog,
)


CSV_COLUMNS = [
    "Card ID",
    "Card Name",
    "Expansion",
    "Collection No.",
    "Stage (Pokémon)/Type (Energy and Trainer)",
    "Rule",
    "Category",
    "Previous stage",
    "HP",
    "Type",
    "Weakness",
    "Resistance (Type)",
    "Retreat",
    "Move Name",
    "Cost",
    "Damage",
    "Effect Explanation",
]


def _attack(attack_id: int, name: str, *, damage: int, energies: list[int]):
    return SimpleNamespace(
        attackId=attack_id,
        name=name,
        text=f"Rules for {name}",
        damage=damage,
        energies=energies,
    )


def _card(
    card_id: int,
    name: str,
    *,
    card_type: int,
    basic: bool = False,
    stage1: bool = False,
    evolves_from: str | None = None,
    attacks: list[int] | None = None,
    skills: list[SimpleNamespace] | None = None,
):
    return SimpleNamespace(
        cardId=card_id,
        name=name,
        cardType=card_type,
        retreatCost=1 if card_type == 0 else 0,
        hp=100 if card_type == 0 or basic else 0,
        weakness=2 if card_type == 0 else None,
        resistance=None,
        energyType=1,
        basic=basic,
        stage1=stage1,
        stage2=False,
        ex=False,
        megaEx=False,
        tera=False,
        aceSpec=False,
        evolvesFrom=evolves_from,
        skills=skills or [],
        attacks=attacks or [],
    )


def _fixture_catalog_data():
    attacks = [
        _attack(1, "Tap", damage=20, energies=[0]),
        _attack(2, "Blast", damage=100, energies=[1, 1]),
    ]
    cards = [
        _card(1, "Basic {G} Energy", card_type=5),
        _card(2, "Seed", card_type=0, basic=True, attacks=[1]),
        _card(
            3,
            "Bloom",
            card_type=0,
            stage1=True,
            evolves_from="Seed",
            attacks=[2],
            skills=[SimpleNamespace(name="Grow", text="Once during your turn")],
        ),
    ]
    return cards, attacks


def _write_csv(path: Path, cards, *, wrong_name: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for card in cards:
            row = {column: "" for column in CSV_COLUMNS}
            row.update(
                {
                    "Card ID": str(card.cardId),
                    "Card Name": (
                        "Wrong Name" if wrong_name and card.cardId == 2 else card.name
                    ),
                    "Expansion": "TEST",
                    "Collection No.": str(card.cardId),
                    "Move Name": "n/a",
                }
            )
            writer.writerow(row)


def test_metadata_catalog_is_exact_deterministic_join(tmp_path: Path) -> None:
    cards, attacks = _fixture_catalog_data()
    csv_path = tmp_path / "cards.csv"
    _write_csv(csv_path, cards)
    first = build_metadata_catalog(cards, attacks, csv_path=csv_path)
    second = build_metadata_catalog(cards, attacks, csv_path=csv_path)

    assert first.card_vocab == 4
    assert first.attack_vocab == 3
    assert first.evolution_parents == {3: (2,)}
    assert first.provenance == second.provenance
    assert first.provenance["schema"] == CARD_METADATA_SCHEMA
    assert first.provenance["csv_sha256"].startswith("sha256:")
    assert torch.equal(first.card_features, second.card_features)
    assert torch.equal(first.attack_features, second.attack_features)
    assert torch.count_nonzero(first.card_features[1:].sum(dim=1)) == 3
    assert torch.count_nonzero(first.attack_features[1:].sum(dim=1)) == 2


def test_metadata_catalog_fails_on_join_gap_duplicate_and_oov_ref(
    tmp_path: Path,
) -> None:
    cards, attacks = _fixture_catalog_data()
    csv_path = tmp_path / "cards.csv"
    _write_csv(csv_path, cards, wrong_name=True)
    with pytest.raises(MetadataContractError, match="name mismatch"):
        build_metadata_catalog(cards, attacks, csv_path=csv_path)

    with pytest.raises(MetadataContractError, match="contiguous"):
        build_metadata_catalog([cards[0], cards[2]], attacks)
    with pytest.raises(MetadataContractError, match="duplicate cardId"):
        build_metadata_catalog([cards[0], cards[0], cards[1]], attacks)

    cards[1].attacks = [99]
    with pytest.raises(MetadataContractError, match="missing attacks"):
        build_metadata_catalog(cards, attacks)


def test_zero_gated_metadata_residual_is_legacy_bit_exact(tmp_path: Path) -> None:
    cards, attacks = _fixture_catalog_data()
    csv_path = tmp_path / "cards.csv"
    _write_csv(csv_path, cards)
    catalog = build_metadata_catalog(cards, attacks, csv_path=csv_path)
    module = GatedMetadataResidual(catalog, d_card=8)
    base = torch.randn(4, 8, requires_grad=True)
    kind = torch.tensor([0, 1, 2, 1])
    entity_id = torch.tensor([0, 2, 1, 3])

    unchanged = module.augment_entity(base, kind=kind, entity_id=entity_id)
    assert torch.equal(unchanged, base)
    assert module.checkpoint_contract()["legacy_exact_when_gates_zero"] is True

    with torch.no_grad():
        module.card_gate.fill_(0.25)
        module.attack_gate.fill_(0.25)
        module.card_projection.weight.fill_(0.01)
        module.attack_projection.weight.fill_(0.01)
    changed = module.augment_entity(base, kind=kind, entity_id=entity_id)
    assert torch.equal(changed[0], base[0])  # NULL stays exactly unchanged.
    assert not torch.equal(changed[1:], base[1:])
    changed.sum().backward()
    assert module.card_gate.grad is not None
    assert module.attack_gate.grad is not None


def test_metadata_residual_rejects_typed_oov_ids(tmp_path: Path) -> None:
    cards, attacks = _fixture_catalog_data()
    csv_path = tmp_path / "cards.csv"
    _write_csv(csv_path, cards)
    module = GatedMetadataResidual(
        build_metadata_catalog(cards, attacks, csv_path=csv_path), d_card=4
    )
    with pytest.raises(MetadataContractError, match="card entity ID"):
        module.augment_entity(
            torch.zeros(1, 4), kind=torch.tensor([1]), entity_id=torch.tensor([4])
        )
    with pytest.raises(MetadataContractError, match="attack entity ID"):
        module.augment_entity(
            torch.zeros(1, 4), kind=torch.tensor([2]), entity_id=torch.tensor([3])
        )


def test_provenance_gate_fails_closed(tmp_path: Path) -> None:
    cards, attacks = _fixture_catalog_data()
    csv_path = tmp_path / "cards.csv"
    _write_csv(csv_path, cards)
    catalog = build_metadata_catalog(cards, attacks, csv_path=csv_path)
    catalog.assert_provenance(
        {"engine_cards_sha256": catalog.provenance["engine_cards_sha256"]}
    )
    with pytest.raises(MetadataContractError, match="provenance mismatch"):
        catalog.assert_provenance({"engine_cards_sha256": "sha256:stale"})
