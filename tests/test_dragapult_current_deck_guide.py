from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from poke_bot import deck_guides
from poke_bot import dragapult_heuristics as guide
from poke_bot.deck_pool import dragapult_deck

ROOT = Path(__file__).resolve().parents[1]
DECK_PATH = (
    ROOT
    / "decks"
    / "dragapult-only"
    / "2026-05_regional-campinas-2026_2nd_dragapult.csv"
)
CONTRACT_PATH = ROOT / "config" / "deck_guides" / "dragapult.yaml"
WRITEUP_PATH = ROOT / "docs" / "deck_guides" / "dragapult-expert-brief.txt"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_plain_representative_is_exact_and_excludes_named_variants() -> None:
    deck = dragapult_deck()
    assert len(deck) == 60
    assert guide.is_plain_dragapult_deck(deck)
    assert _sha256(DECK_PATH) == (
        "sha256:ce78f069b7d7172d5edf7e8ea36da1cdf"
        "801fd77412226b94120eafe9566a1e0"
    )
    assert deck.count(guide.CRUSHING_HAMMER) == 0
    assert not any(card in guide.DUNSPARCE_FAMILY for card in deck)
    assert not any(card in guide.DUSKNOIR_FAMILY for card in deck)
    assert not any(card in guide.BLAZIKEN_FAMILY for card in deck)


def test_guide_is_preserved_offline_but_removed_from_current_registry(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "dragapult")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    assert "dragapult" not in deck_guides.supported_ids()
    with pytest.raises(RuntimeError, match="unknown current-deck guide"):
        deck_guides.enabled()
    assert guide.prior_logit_bias({}, [(), (0,), (1,)]) == [0.0, 0.0, 0.0]
    assert guide.guide_scores(
        {},
        [()],
        deck=dragapult_deck(),
        force_enabled=True,
    ) is None


def test_contract_binds_human_guide_identity_and_masking() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["specialist_id"] == "dragapult"
    assert contract["guide_version"] == guide.GUIDE_VERSION
    assert contract["identity_scope"]["no_variant_substitution"] is True
    assert contract["target_safety"]["missing_label_behavior"] == "mask_not_zero"
    assert contract["target_safety"]["runtime_action_override_allowed"] is False
    assert contract["project_representative_binding"]["card_count"] == 60
    assert contract["expert_writeup"]["word_count"] <= 10_000
    assert contract["expert_writeup"]["sha256"] == _sha256(WRITEUP_PATH)
    assert len(WRITEUP_PATH.read_text(encoding="utf-8").split()) == 4231


def test_contract_module_checksum_is_exact() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["teacher_module_sha256"] == _sha256(
        ROOT / "poke_bot" / "dragapult_heuristics.py"
    )
