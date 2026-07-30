from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from poke_bot import archetypes, crustle_heuristics as guide, deck_guides
from poke_bot.ladder_deck_mix import canonical_payload_digest

ROOT = Path(__file__).resolve().parents[1]
DECK_PATH = (
    ROOT
    / "decks"
    / "competitive"
    / "the_rest"
    / "2026-06_naic-2026-new-orleans_55th_crustle.csv"
)
CONTRACT_PATH = (
    ROOT / "config" / "deck_guides" / "crustle-research-proposal.yaml"
)
CANONICAL_CONTRACT_PATH = ROOT / "config" / "deck_guides" / "crustle.yaml"
WRITEUP_PATH = ROOT / "docs" / "deck_guides" / "crustle-expert-brief.txt"
CORPUS_SCRIPT = ROOT / "ops" / "elmo" / "build_crustle_full33_corpus.sh"
LABEL_AUDIT_SCRIPT = ROOT / "ops" / "elmo" / "audit_crustle_full33_labels.sh"
CORPUS_SERVICE = (
    ROOT / "ops" / "elmo" / "pokebot-crustle-full33-corpus-v1.service"
)
LABEL_AUDIT_SERVICE = (
    ROOT
    / "ops"
    / "elmo"
    / "pokebot-crustle-full33-label-audit-v1.service"
)
IMPORT_SERVICE = (
    ROOT / "ops" / "systemd" / "pokebot-crustle-full33-corpus-import.service"
)
IMPORT_TIMER = (
    ROOT / "ops" / "systemd" / "pokebot-crustle-full33-corpus-import.timer"
)
CATALOG_ON_SUCCESS = (
    ROOT
    / "ops"
    / "elmo"
    / "pokebot-crustle-full33-catalog-v1.on-success.conf"
)
CANONICAL_DECK = [
    int(value) for value in DECK_PATH.read_text(encoding="utf-8").splitlines()
]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _player(
    *,
    active: list[int] | None = None,
    bench: list[int] | None = None,
    hand: list[int] | None = None,
    discard: list[int] | None = None,
    deck_count: int = 30,
) -> dict:
    return {
        "active": [{"id": value} for value in (active or [])],
        "bench": [{"id": value} for value in (bench or [])],
        "hand": [{"id": value} for value in (hand or [])],
        "discard": [{"id": value} for value in (discard or [])],
        "deckCount": deck_count,
        "prize": [None] * 6,
    }


def _obs(
    me: dict,
    options: list[dict],
    *,
    context: int,
    effect_id: int | None = None,
    deck_cards: list[int] | None = None,
) -> dict:
    select = {
        "context": context,
        "option": options,
        "minCount": 1,
        "maxCount": 1,
    }
    if effect_id is not None:
        select["effect"] = {"id": effect_id}
    if deck_cards is not None:
        select["deck"] = [{"id": value} for value in deck_cards]
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, _player()],
            "stadium": [],
            "looking": [],
        },
        "select": select,
    }


def _scores(obs: dict, combos: list[list[int]] | None = None):
    return guide.guide_scores(
        obs,
        combos or [[0], [1]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )


def test_exact_research_list_and_single_card_mutation() -> None:
    assert len(CANONICAL_DECK) == 60
    assert guide.is_crustle_deck(CANONICAL_DECK)
    assert guide.applies(reversed(CANONICAL_DECK))
    mutated = list(CANONICAL_DECK)
    mutated[0] = guide.BASIC_GRASS_ENERGY
    assert not guide.is_crustle_deck(mutated)
    assert guide.is_crustle_family_deck(mutated)
    assert guide.applies(mutated)
    assert archetypes.classify_deck(CANONICAL_DECK) == "crustle"
    assert archetypes.classify_deck(mutated) == "unknown"


def test_public_family_signature_covers_reviewed_lists_without_collision() -> None:
    index = ROOT / "decks" / "competitive" / "index.csv"
    import csv

    with index.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    reviewed = 0
    for row in rows:
        relative = (
            ROOT
            / "decks"
            / "competitive"
            / row["tier"]
            / row["filename"]
        )
        if not relative.is_file():
            continue
        cards = [
            int(value)
            for value in relative.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if len(cards) != 60:
            continue
        expected = row["archetype"] == "Crustle"
        assert guide.is_crustle_family_deck(cards) is expected, relative
        reviewed += expected
    assert reviewed >= 28


def test_guide_is_inactive_without_selector_and_serving_bias_is_zero() -> None:
    assert guide.enabled() is False
    assert guide.prior_logit_bias({}, [(), (0,), (1,)], scale=99.0) == [
        0.0,
        0.0,
        0.0,
    ]


def test_crustle_guide_is_preserved_but_not_in_current_registry(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "crustle")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    assert "crustle" not in deck_guides.supported_ids()
    assert deck_guides.selected_id() == "crustle"
    assert deck_guides.guide_version() is None
    with pytest.raises(RuntimeError, match="unknown current-deck guide"):
        deck_guides.enabled()


def test_setup_bench_prefers_second_dwebble() -> None:
    obs = _obs(
        _player(
            active=[guide.MEGA_KANGASKHAN_EX],
            bench=[guide.DWEBBLE],
            hand=[guide.DWEBBLE, guide.MEGA_KANGASKHAN_EX],
        ),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_SETUP_BENCH,
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_ultra_ball_fills_visible_crustle_evolution_gap() -> None:
    obs = _obs(
        _player(active=[guide.DWEBBLE]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 1},
        ],
        context=guide.CTX_TO_HAND,
        effect_id=guide.ULTRA_BALL,
        deck_cards=[guide.CRUSTLE, guide.MEGA_KANGASKHAN_EX],
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_run_errand_uses_only_current_deck_count() -> None:
    options = [{"type": guide.OPT_YES}, {"type": guide.OPT_NO}]
    safe = _obs(
        _player(active=[guide.MEGA_KANGASKHAN_EX], deck_count=4),
        options,
        context=guide.CTX_TO_HAND,
        effect_id=guide.MEGA_KANGASKHAN_EX,
    )
    unsafe = _obs(
        _player(active=[guide.MEGA_KANGASKHAN_EX], deck_count=3),
        options,
        context=guide.CTX_TO_HAND,
        effect_id=guide.MEGA_KANGASKHAN_EX,
    )
    safe_scores = _scores(safe)
    unsafe_scores = _scores(unsafe)
    assert safe_scores is not None and safe_scores[0] > safe_scores[1]
    assert unsafe_scores is not None and unsafe_scores[1] > unsafe_scores[0]


def test_hidden_information_does_not_change_a_supported_label() -> None:
    base = _obs(
        _player(
            active=[guide.MEGA_KANGASKHAN_EX],
            bench=[guide.DWEBBLE],
            hand=[guide.DWEBBLE, guide.MEGA_KANGASKHAN_EX],
        ),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_SETUP_BENCH,
    )
    changed = {
        **base,
        "private": {"future_draws": [guide.CRUSTLE], "eventual_result": "win"},
    }
    changed["current"]["players"][1]["hand"] = [{"id": 9999}]
    changed["current"]["players"][1]["prize"] = [{"id": 8888}]
    assert _scores(base) == _scores(changed)


def test_opening_active_unknown_prompt_and_partial_stage_are_masked() -> None:
    options = [
        {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
        {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
    ]
    opening = _obs(
        _player(hand=[guide.DWEBBLE, guide.MEGA_KANGASKHAN_EX]),
        options,
        context=guide.CTX_SETUP_ACTIVE,
    )
    unknown = _obs(
        _player(active=[guide.DWEBBLE]),
        options,
        context=guide.CTX_TO_HAND,
        effect_id=9999,
    )
    complete = _obs(
        _player(
            active=[guide.MEGA_KANGASKHAN_EX],
            bench=[guide.DWEBBLE],
            hand=[guide.DWEBBLE, guide.MEGA_KANGASKHAN_EX],
        ),
        options,
        context=guide.CTX_SETUP_BENCH,
    )
    assert _scores(opening) is None
    assert _scores(unknown) is None
    assert _scores(complete, combos=[[0]]) is None


def test_contract_binds_artifacts_sources_and_canonical_registry() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["guide_version"] == guide.GUIDE_VERSION
    assert contract["teacher_module_sha256"] == _sha256(
        ROOT / contract["teacher_module_path"]
    )
    assert contract["expert_writeup"]["sha256"] == _sha256(WRITEUP_PATH)
    assert contract["expert_writeup"]["word_count"] == len(
        WRITEUP_PATH.read_text(encoding="utf-8").split()
    )
    assert contract["expert_writeup"]["word_count"] <= 10_000
    binding = contract["research_representative_binding"]
    assert binding["file_sha256"] == _sha256(DECK_PATH)
    multiset_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            sorted(CANONICAL_DECK), separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert binding["canonical_multiset_sha256"] == multiset_digest
    assert binding["card_count"] == len(CANONICAL_DECK) == 60
    assert binding["canonical_representative_registry_entry_exists"] is True
    assert binding["integration_blocker"] is None
    canonical = json.loads(
        (
            ROOT / "data/training_mixes/specialist_representatives.v1.json"
        ).read_text(encoding="utf-8")
    )["decks"]["crustle"]
    assert canonical["card_ids"] == CANONICAL_DECK
    assert (
        binding["canonical_registry_multiset_sha256"]
        == canonical["canonical_multiset_sha256"]
    )
    assert contract["target_safety"]["missing_label_behavior"] == "mask_not_zero"
    assert contract["target_safety"]["hidden_or_future_information_allowed"] is False
    assert len(contract["strategy_sources"]) >= 10


def test_full33_corpus_and_label_audit_are_receipt_gated() -> None:
    corpus = CORPUS_SCRIPT.read_text(encoding="utf-8")
    audit = LABEL_AUDIT_SCRIPT.read_text(encoding="utf-8")
    corpus_service = CORPUS_SERVICE.read_text(encoding="utf-8")
    audit_service = LABEL_AUDIT_SERVICE.read_text(encoding="utf-8")
    import_service = IMPORT_SERVICE.read_text(encoding="utf-8")
    import_timer = IMPORT_TIMER.read_text(encoding="utf-8")
    catalog_drop_in = CATALOG_ON_SUCCESS.read_text(encoding="utf-8")

    assert "--start \"$start_date\"" in corpus
    assert "--end \"$end_date\"" in corpus
    assert '--required-archetype crustle' in corpus
    assert '--current-deck-guide crustle' in corpus
    assert '--authoritative-only-archetype crustle' in corpus
    assert "crustle_card_signature_public_replay_identity" in corpus
    assert "broad_archetype_name_filter_sufficient" in corpus
    assert 'minimum_owner_records="${POKEBOT_CRUSTLE_MINIMUM_RECORDS:-16639}"' in corpus
    assert "CURRENT_DECK_GUIDE_CORPUS_READY.json" in corpus
    assert "actual_records < minimum_records" in corpus
    assert "actual_records > expected_records" in corpus
    assert "excluded_records" in corpus
    assert "guide_rows" in corpus
    assert "CRUSTLE_GUIDE_LABEL_AUDIT_FULL33.json" in audit
    assert "passed_structural_and_observational_validation" in audit
    assert "invalid_target_indices" in audit
    assert "invalid_confidences" in audit
    assert "CRUSTLE_GUIDE_CORPUS_VALIDATED.json" in audit
    assert "poke_bot.crustle_guide_corpus_validation/v1" in audit
    assert (
        "ConditionPathExists=/mnt/Main/main/poke-bot-agent/archive/"
        "crustle-public-family-full33-v1.json"
    ) in corpus_service
    assert (
        "OnSuccess=pokebot-crustle-full33-label-audit-v1.service"
        in corpus_service
    )
    assert "ConditionPathExists=" in audit_service
    assert "OnSuccess=pokebot-crustle-full33-corpus-v1.service" in catalog_drop_in
    assert "CRUSTLE_GUIDE_CORPUS_VALIDATED.json" in import_service
    assert (
        "--finalization-receipt-schema "
        "poke_bot.crustle_guide_corpus_validation/v1"
    ) in import_service
    assert "ConditionPathExists=!" in import_service
    assert "OnUnitActiveSec=5min" in import_timer
    assert "pokebot-crustle-full33-corpus-import.service" in import_timer


def test_canonical_contract_and_representative_are_self_consistent() -> None:
    contract = yaml.safe_load(
        CANONICAL_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    representatives_path = (
        ROOT / "data/training_mixes/specialist_representatives.v1.json"
    )
    representatives = json.loads(
        representatives_path.read_text(encoding="utf-8")
    )
    row = representatives["decks"]["crustle"]
    binding = contract["project_representative_binding"]
    assert representatives["artifact_sha256"] == canonical_payload_digest(
        representatives
    )
    assert contract["schema_version"] == "poke_bot.current_deck_guide/v1"
    assert contract["guide_version"] == guide.GUIDE_VERSION
    assert contract["teacher_module_sha256"] == _sha256(
        ROOT / "poke_bot/crustle_heuristics.py"
    )
    assert binding["canonical_multiset_sha256"] == row[
        "canonical_multiset_sha256"
    ]
    assert binding["cards_sha256"] == row["cards_sha256"]
    assert row["card_ids"] == CANONICAL_DECK
    assert contract["target_safety"]["missing_label_behavior"] == "mask_not_zero"
    assert contract["target_safety"]["runtime_authority"] == "none"
