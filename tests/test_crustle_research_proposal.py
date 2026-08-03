from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from poke_bot import archetypes, deck_guides
from poke_bot import crustle_heuristics as guide
from poke_bot.ladder_deck_mix import canonical_payload_digest

ROOT = Path(__file__).resolve().parents[1]
CARD_DATA_PATH = ROOT / "cards" / "EN_Card_Data.csv"
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


def _engine_rows(card_id: int) -> list[dict[str, str]]:
    with CARD_DATA_PATH.open(newline="", encoding="utf-8-sig") as source:
        return [
            row
            for row in csv.DictReader(source)
            if int(row["Card ID"]) == card_id
        ]


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


def test_poffin_resolves_psyduck_and_prefers_missing_dwebble() -> None:
    obs = _obs(
        _player(active=[guide.MEGA_KANGASKHAN_EX]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 1},
        ],
        context=guide.CTX_TO_BENCH,
        effect_id=guide.BUDDY_BUDDY_POFFIN,
        deck_cards=[guide.DWEBBLE, guide.PSYDUCK],
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_search_counts_the_acting_players_current_hand() -> None:
    obs = _obs(
        _player(
            active=[guide.MEGA_KANGASKHAN_EX],
            hand=[guide.DWEBBLE, guide.DWEBBLE],
        ),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 1},
        ],
        context=guide.CTX_TO_BENCH,
        effect_id=guide.BUDDY_BUDDY_POFFIN,
        deck_cards=[guide.DWEBBLE, guide.PSYDUCK],
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[1] > scores[0] + guide.ABSTENTION_MARGIN


def test_lumiose_search_is_labeled_only_after_exact_activation() -> None:
    obs = _obs(
        _player(active=[guide.MEGA_KANGASKHAN_EX]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 1},
        ],
        context=guide.CTX_TO_BENCH,
        effect_id=guide.LUMIOSE_CITY,
        deck_cards=[guide.DWEBBLE, guide.MEGA_KANGASKHAN_EX],
    )
    scores = _scores(obs)
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN

    wrong_origin = _obs(
        _player(
            active=[guide.MEGA_KANGASKHAN_EX],
            hand=[guide.DWEBBLE, guide.MEGA_KANGASKHAN_EX],
        ),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_TO_BENCH,
        effect_id=guide.LUMIOSE_CITY,
    )
    assert _scores(wrong_origin) is None


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


def test_audit_is_training_only_directional_and_never_a_policy_logit_target() -> None:
    obs = _obs(
        _player(active=[guide.MEGA_KANGASKHAN_EX]),
        [{"type": guide.OPT_YES}, {"type": guide.OPT_NO}],
        context=guide.CTX_TO_HAND,
        effect_id=guide.MEGA_KANGASKHAN_EX,
    )
    audit = guide.guide_audit(
        obs,
        [[0], [1]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )
    assert audit is not None
    assert audit["training_mode"] == "strategic_directional_v2"
    assert audit["guide_preference_index_role"] == (
        "selected_causal_route_pairwise_direction_only"
    )
    assert audit["target_logits"] == "none"
    assert audit["direct_policy_cross_entropy_allowed"] is False
    assert audit["final_policy_logits_are_guide_targets"] is False
    assert audit["runtime_input_allowed"] is False
    assert audit["runtime_action_logit_route_allowed"] is False


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
    assert contract["policy_target"]["training_mode"] == "strategic_directional_v2"
    assert contract["policy_target"]["target_logits"] == "none"
    assert contract["policy_target"]["direct_policy_cross_entropy_allowed"] is False
    assert contract["policy_target"]["runtime_action_logit_route_allowed"] is False
    assert len(contract["strategy_sources"]) == 22


def test_contracts_and_writeup_use_the_same_complete_source_set() -> None:
    research = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    canonical = yaml.safe_load(
        CANONICAL_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    writeup = WRITEUP_PATH.read_text(encoding="utf-8")
    sources = writeup.split("\nSOURCES\n", maxsplit=1)[1]
    writeup_urls = {
        line.strip() for line in sources.splitlines() if line.startswith("http")
    }
    research_urls = {row["url"] for row in research["strategy_sources"]}
    canonical_urls = {row["url"] for row in canonical["strategy_sources"]}
    assert len(writeup_urls) == 22
    assert research_urls == canonical_urls == writeup_urls


def test_expert_writeup_records_exact_turn_and_card_engine_constraints() -> None:
    writeup = WRITEUP_PATH.read_text(encoding="utf-8")
    assert "cannot attack or play an ordinary Supporter" in writeup
    assert "using its Ability ends your turn" in writeup
    assert "Damp stops only self-Knock-Out Abilities" in writeup
    assert "4 Grow Grass Energy" in writeup
    assert "Growing Grass Energy" not in writeup


def test_claimed_card_engine_matches_the_repository_card_records() -> None:
    crustle = _engine_rows(guide.CRUSTLE)
    assert {(row["Move Name"], row["Cost"], row["Damage"]) for row in crustle} == {
        ("[Ability] Mysterious Rock Inn", "n/a", "n/a"),
        ("Superb Scissors", "{G}●●", "120"),
    }
    assert crustle[0]["Effect Explanation"] == (
        "Prevent all damage done to this Pokémon by attacks from your "
        "opponent’s Pokémon {ex}."
    )

    dwebble = _engine_rows(guide.DWEBBLE)[0]
    assert dwebble["HP"] == "70"
    assert dwebble["Move Name"] == "Ascension"
    assert dwebble["Cost"] == "●"

    kangaskhan = _engine_rows(guide.MEGA_KANGASKHAN_EX)
    run_errand = next(row for row in kangaskhan if row["Move Name"].endswith("Run Errand"))
    assert "in the Active Spot" in run_errand["Effect Explanation"]
    assert "Draw 2 cards" in run_errand["Effect Explanation"]

    psyduck = _engine_rows(guide.PSYDUCK)[0]
    assert psyduck["HP"] == "70"
    assert "requires the Pokémon using it to Knock Out itself" in psyduck[
        "Effect Explanation"
    ]

    poffin = _engine_rows(guide.BUDDY_BUDDY_POFFIN)[0]["Effect Explanation"]
    assert "up to 2 Basic Pokémon with 70 HP or less" in poffin
    potion = _engine_rows(guide.SUPER_POTION)[0]["Effect Explanation"]
    assert "Heal 60 damage" in potion and "discard an Energy" in potion
    lumiose = _engine_rows(guide.LUMIOSE_CITY)[0]["Effect Explanation"]
    assert "Basic Pokémon" in lumiose and "their turn ends" in lumiose

    energy_names = {
        _engine_rows(card_id)[0]["Card Name"]
        for card_id in (
            guide.GROW_GRASS_ENERGY,
            guide.MIST_ENERGY,
            guide.SPIKY_ENERGY,
        )
    }
    assert energy_names == {"Grow Grass Energy", "Mist Energy", "Spiky Energy"}


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
    assert Counter(binding["exact_card_counts"]) == Counter(CANONICAL_DECK)
    assert contract["policy_target"]["training_mode"] == "strategic_directional_v2"
    assert contract["policy_target"]["target_logits"] == "none"
    assert contract["policy_target"]["final_policy_logits_are_guide_targets"] is False
    assert contract["policy_target"]["runtime_action_logit_route_allowed"] is False
    assert contract["target_safety"]["missing_label_behavior"] == "mask_not_zero"
    assert contract["target_safety"]["runtime_authority"] == "none"
