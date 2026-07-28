from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from poke_bot import deck_guides
from poke_bot import hammer_heuristics as guide


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_PATH = (
    ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"
)
CANONICAL_DECK = json.loads(REPRESENTATIVE_PATH.read_text(encoding="utf-8"))[
    "decks"
]["hammer-pult"]["card_ids"]


def _pokemon(card_id: int, *, energy: list[int] | None = None) -> dict:
    return {
        "id": card_id,
        "energyCards": [{"id": value} for value in (energy or [])],
    }


def _player(
    *,
    active: list[dict] | None = None,
    bench: list[dict] | None = None,
    hand: list[int] | None = None,
) -> dict:
    return {
        "active": list(active or []),
        "bench": list(bench or []),
        "hand": [{"id": value} for value in (hand or [])],
        "discard": [],
        "deckCount": 30,
        "prize": [None] * 6,
    }


def _obs(
    me: dict,
    options: list[dict],
    *,
    opponent: dict | None = None,
    context: int = guide.CTX_MAIN,
) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, opponent or _player()],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": context,
            "option": options,
            "minCount": 1,
            "maxCount": 1,
        },
    }


def _scores(obs: dict) -> list[float] | None:
    return guide.guide_scores(
        obs,
        [[0], [1]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )


def test_signature_accepts_exact_modal_top_ladder_representative() -> None:
    assert len(CANONICAL_DECK) == 60
    assert guide.is_hammer_pult_deck(CANONICAL_DECK)
    assert guide.applies(CANONICAL_DECK)


@pytest.mark.parametrize(
    ("card_id", "minimum"),
    list(guide.CORE_SIGNATURE_MINIMUMS.items()),
)
def test_signature_rejects_each_missing_required_component(
    card_id: int,
    minimum: int,
) -> None:
    deck = list(CANONICAL_DECK)
    while deck.count(card_id) >= minimum:
        deck[deck.index(card_id)] = 999
    assert len(deck) == 60
    assert not guide.is_hammer_pult_deck(deck)


def test_setup_prefers_budew_active_then_dreepy() -> None:
    obs = _obs(
        _player(hand=[guide.BUDEW, guide.DREEPY]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_SETUP_ACTIVE,
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_exact_poffin_prompt_prefers_second_visible_dreepy() -> None:
    obs = _obs(
        _player(active=[_pokemon(guide.BUDEW)], bench=[_pokemon(guide.DREEPY)]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 1},
        ],
        context=guide.CTX_TO_BENCH,
    )
    obs["select"]["effect"] = {"id": guide.BUDDY_BUDDY_POFFIN}
    obs["select"]["deck"] = [
        {"id": guide.DREEPY},
        {"id": guide.BUDEW},
    ]
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


@pytest.mark.parametrize(
    "evolution",
    [guide.DRAKLOAK, guide.DRAGAPULT_EX],
)
def test_main_prefers_visible_dragapult_line_evolution(evolution: int) -> None:
    obs = _obs(
        _player(hand=[evolution]),
        [
            {"type": guide.OPT_EVOLVE, "area": guide.AREA_HAND, "index": 0},
            {"type": 14},
        ],
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_main_prefers_exact_drackloak_recon_directive_activation() -> None:
    obs = _obs(
        _player(active=[_pokemon(guide.DRAKLOAK)]),
        [
            {"type": guide.OPT_ABILITY, "area": guide.AREA_ACTIVE, "index": 0},
            {"type": 14},
        ],
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_hammer_requires_visible_opposing_attached_energy() -> None:
    options = [{"type": guide.OPT_PLAY, "index": 0}, {"type": 14}]
    with_energy = _obs(
        _player(hand=[guide.CRUSHING_HAMMER]),
        options,
        opponent=_player(
            active=[_pokemon(900, energy=[guide.PSYCHIC_ENERGY])]
        ),
    )
    scores = _scores(with_energy)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN

    without_energy = _obs(
        _player(hand=[guide.CRUSHING_HAMMER]),
        options,
        opponent=_player(active=[_pokemon(900)]),
    )
    assert _scores(without_energy) is None


def test_unsupported_or_malformed_stage_masks_completely() -> None:
    obs = _obs(
        _player(),
        [{"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0}],
        context=99,
    )
    obs["select"]["effect"] = {"id": 999}
    obs["select"]["deck"] = [{"id": guide.DREEPY}]
    assert (
        guide.guide_scores(
            obs, [[0]], deck=CANONICAL_DECK, force_enabled=True
        )
        is None
    )
    obs["select"]["context"] = guide.CTX_MAIN
    assert (
        guide.guide_scores(
            obs, [[1]], deck=CANONICAL_DECK, force_enabled=True
        )
        is None
    )


def test_generic_registry_dispatches_hammer_pult(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "hammer-pult")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    assert deck_guides.enabled()
    assert deck_guides.guide_version() == guide.GUIDE_VERSION
    assert "hammer-pult" in deck_guides.supported_ids()


def test_contract_binds_guide_writeup_teacher_and_representative() -> None:
    contract_path = ROOT / "config" / "deck_guides" / "hammer-pult.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    binding = contract["project_representative_binding"]
    ordered_digest = "sha256:" + hashlib.sha256(
        json.dumps(CANONICAL_DECK, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    multiset_digest = "sha256:" + hashlib.sha256(
        json.dumps(sorted(CANONICAL_DECK), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    teacher_path = ROOT / "poke_bot" / "hammer_heuristics.py"
    teacher_digest = "sha256:" + hashlib.sha256(
        teacher_path.read_bytes()
    ).hexdigest()
    writeup_path = ROOT / contract["expert_writeup"]["path"]
    writeup_digest = "sha256:" + hashlib.sha256(
        writeup_path.read_bytes()
    ).hexdigest()

    assert binding["card_count"] == len(CANONICAL_DECK) == 60
    assert binding["cards_sha256"] == ordered_digest
    assert binding["canonical_multiset_sha256"] == multiset_digest
    assert contract["teacher_module_sha256"] == teacher_digest
    assert contract["expert_writeup"]["sha256"] == writeup_digest
    assert contract["expert_writeup"]["word_count"] == len(
        writeup_path.read_text(encoding="utf-8").split()
    )
    assert contract["expert_writeup"]["word_count"] <= 10_000
    assert contract["target_safety"]["future_information_allowed"] is False
    assert (
        contract["target_safety"]["outcome_conditioned_labels_allowed"]
        is False
    )
    assert contract["target_safety"]["runtime_authority"] == "none"
