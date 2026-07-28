from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from poke_bot import team_rockets_spidops_heuristics as guide


ROOT = Path(__file__).resolve().parents[1]


# Pedro Henrique Pereira Calda's pinned Campinas representative. Keeping the list
# here makes the identity test exercise the project representative rather than
# a synthetic minimum-signature deck.
CANONICAL_DECK: list[int] = (
    [guide.TEAM_ROCKETS_TAROUNTULA] * 4
    + [guide.TEAM_ROCKETS_SPIDOPS] * 4
    + [guide.TEAM_ROCKETS_MIMIKYU] * 2
    + [guide.TEAM_ROCKETS_ARTICUNO] * 2
    + [guide.TEAM_ROCKETS_SNEASEL]
    + [guide.TEAM_ROCKETS_MEWTWO_EX]
    + [guide.LILLIES_CLEFAIRY_EX]
    + [1216] * 4  # Team Rocket's Ariana
    + [1227] * 3  # Lillie's Determination
    + [guide.TEAM_ROCKETS_PROTON] * 2
    + [1217] * 2  # Team Rocket's Archer
    + [1218] * 2  # Team Rocket's Giovanni
    + [1134] * 4  # Team Rocket's Transceiver
    + [1121] * 4  # Ultra Ball
    + [guide.BUDDY_BUDDY_POFFIN] * 2
    + [1097] * 2  # Night Stretcher
    + [1116]  # Energy Switch
    + [1129]  # Sacred Ash
    + [1119]  # Energy Search
    + [1175] * 2  # Brave Bangle
    + [1158]  # Maximum Belt
    + [1257] * 3  # Team Rocket's Factory
    + [guide.BASIC_GRASS_ENERGY] * 6
    + [guide.TEAM_ROCKETS_ENERGY] * 4
    + [guide.BASIC_PSYCHIC_ENERGY]
)


def _player(
    *,
    active: list[int] | None = None,
    bench: list[int] | None = None,
    hand: list[int] | None = None,
    discard: list[int] | None = None,
) -> dict:
    return {
        "active": [{"id": value} for value in (active or [])],
        "bench": [{"id": value} for value in (bench or [])],
        "hand": [{"id": value} for value in (hand or [])],
        "discard": [{"id": value} for value in (discard or [])],
        "deckCount": 30,
        "prize": [None] * 6,
    }


def _obs(
    me: dict,
    options: list[dict],
    *,
    context: int = guide.CTX_MAIN,
    opponent: dict | None = None,
    minimum: int = 1,
    maximum: int = 1,
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
            "minCount": minimum,
            "maxCount": maximum,
        },
    }


def _deck_option(index: int) -> dict:
    return {
        "type": guide.OPT_CARD,
        "area": guide.AREA_DECK,
        "index": index,
    }


def _assert_first_preferred(obs: dict) -> None:
    scores = guide.guide_scores(
        obs,
        [[0], [1]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def _evolution_stage() -> dict:
    me = _player(
        active=[guide.TEAM_ROCKETS_MIMIKYU],
        bench=[guide.TEAM_ROCKETS_TAROUNTULA],
        hand=[guide.TEAM_ROCKETS_SPIDOPS],
    )
    return _obs(
        me,
        [
            {
                "type": guide.OPT_EVOLVE,
                "area": guide.AREA_HAND,
                "index": 0,
                "inPlayArea": guide.AREA_BENCH,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ],
    )


def test_signature_accepts_pinned_dedicated_spidops_identity() -> None:
    assert len(CANONICAL_DECK) == 60
    assert guide.is_team_rockets_spidops_deck(CANONICAL_DECK)

    no_mewtwo = list(CANONICAL_DECK)
    no_mewtwo[no_mewtwo.index(guide.TEAM_ROCKETS_MEWTWO_EX)] = 999
    assert guide.is_team_rockets_spidops_deck(no_mewtwo)


def test_signature_rejects_two_mewtwo_hybrid_and_wrong_deck() -> None:
    two_mewtwo = list(CANONICAL_DECK)
    two_mewtwo[two_mewtwo.index(1216)] = guide.TEAM_ROCKETS_MEWTWO_EX

    assert not guide.is_team_rockets_spidops_deck(two_mewtwo)
    assert not guide.is_team_rockets_spidops_deck([1] * 60)
    assert not guide.is_team_rockets_spidops_deck(CANONICAL_DECK + [999])


def test_signature_accepts_top_ladder_secondary_attacker_variants() -> None:
    variant = list(CANONICAL_DECK)
    for card_id in (
        guide.TEAM_ROCKETS_ARTICUNO,
        guide.TEAM_ROCKETS_ARTICUNO,
        guide.TEAM_ROCKETS_MIMIKYU,
        guide.TEAM_ROCKETS_MIMIKYU,
        guide.TEAM_ROCKETS_SNEASEL,
    ):
        variant[variant.index(card_id)] = 999
    assert len(variant) == 60
    assert guide.is_team_rockets_spidops_deck(variant)


def test_setup_prefers_mimikyu_as_the_exact_zero_retreat_active() -> None:
    obs = _obs(
        _player(
            hand=[
                guide.TEAM_ROCKETS_MIMIKYU,
                guide.TEAM_ROCKETS_TAROUNTULA,
            ]
        ),
        [
            {
                "type": guide.OPT_CARD,
                "area": guide.AREA_HAND,
                "index": 0,
            },
            {
                "type": guide.OPT_CARD,
                "area": guide.AREA_HAND,
                "index": 1,
            },
        ],
        context=guide.CTX_SETUP_ACTIVE,
    )

    _assert_first_preferred(obs)


@pytest.mark.parametrize(
    ("search_effect", "search_context"),
    [
        (guide.BUDDY_BUDDY_POFFIN, guide.CTX_TO_BENCH),
        (guide.TEAM_ROCKETS_PROTON, guide.CTX_TO_HAND),
    ],
)
def test_exact_search_prompt_prefers_redundant_tarountula(
    search_effect: int,
    search_context: int,
) -> None:
    obs = _obs(
        _player(
            active=[guide.TEAM_ROCKETS_MIMIKYU],
            bench=[guide.TEAM_ROCKETS_TAROUNTULA],
        ),
        [_deck_option(0), _deck_option(1)],
        context=search_context,
    )
    obs["select"]["effect"] = {"id": search_effect}
    obs["select"]["deck"] = [
        {"id": guide.TEAM_ROCKETS_TAROUNTULA},
        {"id": guide.TEAM_ROCKETS_ARTICUNO},
    ]

    _assert_first_preferred(obs)


def test_main_stage_prefers_visible_spidops_evolution() -> None:
    _assert_first_preferred(_evolution_stage())


@pytest.mark.parametrize(
    "basic_energy",
    [guide.BASIC_GRASS_ENERGY, guide.BASIC_PSYCHIC_ENERGY],
)
def test_charging_up_requires_visible_basic_energy_in_discard(
    basic_energy: int,
) -> None:
    obs = _obs(
        _player(
            active=[guide.TEAM_ROCKETS_SPIDOPS],
            discard=[basic_energy],
        ),
        [
            {
                "type": guide.OPT_ABILITY,
                "area": guide.AREA_ACTIVE,
                "index": 0,
            },
            {"type": 14},
        ],
    )

    _assert_first_preferred(obs)


def test_charging_up_masks_when_only_team_rockets_energy_is_visible() -> None:
    obs = _obs(
        _player(
            active=[guide.TEAM_ROCKETS_SPIDOPS],
            discard=[guide.TEAM_ROCKETS_ENERGY],
        ),
        [
            {
                "type": guide.OPT_ABILITY,
                "area": guide.AREA_ACTIVE,
                "index": 0,
            },
            {"type": 14},
        ],
    )

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_unsupported_search_prompt_masks_the_complete_stage() -> None:
    obs = _obs(
        _player(),
        [_deck_option(0), _deck_option(1)],
        context=guide.CTX_TO_HAND,
    )
    obs["select"]["effect"] = {"id": 999}
    obs["select"]["deck"] = [
        {"id": guide.TEAM_ROCKETS_TAROUNTULA},
        {"id": guide.TEAM_ROCKETS_ARTICUNO},
    ]

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("effect", "wrong_context"),
    [
        (guide.BUDDY_BUDDY_POFFIN, guide.CTX_TO_HAND),
        (guide.TEAM_ROCKETS_PROTON, guide.CTX_TO_BENCH),
        (guide.TEAM_ROCKETS_PROTON, 99),
    ],
)
def test_search_effect_without_its_exact_context_masks(
    effect: int,
    wrong_context: int,
) -> None:
    obs = _obs(
        _player(),
        [_deck_option(0), _deck_option(1)],
        context=wrong_context,
    )
    obs["select"]["effect"] = {"id": effect}
    obs["select"]["deck"] = [
        {"id": guide.TEAM_ROCKETS_TAROUNTULA},
        {"id": guide.TEAM_ROCKETS_ARTICUNO},
    ]

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_incomplete_factorized_candidate_stage_masks() -> None:
    obs = _evolution_stage()
    obs["select"]["option"].append({"type": 14})

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_complete_later_factorized_search_stage_is_scored() -> None:
    obs = _obs(
        _player(
            active=[guide.TEAM_ROCKETS_MIMIKYU],
            bench=[guide.TEAM_ROCKETS_TAROUNTULA],
        ),
        [_deck_option(0), _deck_option(1), _deck_option(2)],
        context=guide.CTX_TO_HAND,
        minimum=1,
        maximum=3,
    )
    obs["select"]["effect"] = {"id": guide.TEAM_ROCKETS_PROTON}
    obs["select"]["deck"] = [
        {"id": guide.TEAM_ROCKETS_ARTICUNO},
        {"id": guide.TEAM_ROCKETS_TAROUNTULA},
        {"id": guide.TEAM_ROCKETS_SNEASEL},
    ]

    scores = guide.guide_scores(
        obs,
        [[0, 1], [0, 2], [0]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )
    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN
    assert scores[0] > scores[2] + guide.ABSTENTION_MARGIN


def test_main_option_cannot_resolve_an_opponent_spidops() -> None:
    obs = _obs(
        _player(active=[guide.TEAM_ROCKETS_MIMIKYU]),
        [
            {
                "type": guide.OPT_ABILITY,
                "area": guide.AREA_ACTIVE,
                "index": 0,
                "playerIndex": 1,
            },
            {"type": 14},
        ],
        opponent=_player(
            active=[guide.TEAM_ROCKETS_SPIDOPS],
            discard=[guide.BASIC_GRASS_ENERGY],
        ),
    )

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_evolution_must_target_own_visible_tarountula() -> None:
    obs = _evolution_stage()
    obs["select"]["option"][0]["inPlayIndex"] = 99

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minCount", -1),
        ("minCount", 1.0),
        ("maxCount", 99),
        ("maxCount", "1"),
    ],
)
def test_noncanonical_count_bounds_mask(field: str, value: object) -> None:
    obs = _evolution_stage()
    obs["select"][field] = value

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_tied_best_actions_mask_in_the_module() -> None:
    me = _player(
        active=[guide.TEAM_ROCKETS_TAROUNTULA],
        bench=[guide.TEAM_ROCKETS_TAROUNTULA],
        hand=[
            guide.TEAM_ROCKETS_SPIDOPS,
            guide.TEAM_ROCKETS_SPIDOPS,
        ],
    )
    obs = _obs(
        me,
        [
            {
                "type": guide.OPT_EVOLVE,
                "area": guide.AREA_HAND,
                "index": 0,
                "inPlayArea": guide.AREA_ACTIVE,
                "inPlayIndex": 0,
            },
            {
                "type": guide.OPT_EVOLVE,
                "area": guide.AREA_HAND,
                "index": 1,
                "inPlayArea": guide.AREA_BENCH,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ],
    )

    assert (
        guide.guide_scores(
            obs,
            [[0], [1], [2]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


@pytest.mark.parametrize(
    "action_combos",
    [
        [],
        [[2]],
        [[0, 0]],
        [[0], [0]],
    ],
)
def test_malformed_combo_enumerations_mask_the_complete_stage(
    action_combos: list[list[int]],
) -> None:
    assert (
        guide.guide_scores(
            _evolution_stage(),
            action_combos,
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (2, 2),
        (2, 1),
    ],
)
def test_incomplete_or_impossible_count_constraints_mask_the_stage(
    minimum: int,
    maximum: int,
) -> None:
    obs = _evolution_stage()
    obs["select"]["minCount"] = minimum
    obs["select"]["maxCount"] = maximum

    assert (
        guide.guide_scores(
            obs,
            [[0]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_low_margin_tie_masks_instead_of_inventing_a_label() -> None:
    obs = _obs(_player(), [{"type": 14}, {"type": 14}])

    assert (
        guide.guide_scores(
            obs,
            [[0], [1]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_scores_are_invariant_to_opponent_hidden_future_and_outcome_fields() -> None:
    obs = _evolution_stage()
    baseline = guide.guide_scores(
        obs,
        [[0], [1]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )
    assert baseline is not None

    contaminated = deepcopy(obs)
    opponent = contaminated["current"]["players"][1]
    opponent["hand"] = [{"id": 9001}, {"id": 9002}]
    opponent["prize"] = [{"id": 9003}] * 6
    opponent["privateDeckOrder"] = [9004, 9005, 9006]
    opponent["futureDraws"] = [9007, 9008]
    contaminated["future"] = {
        "nextOpponentAction": 17,
        "terminalWinner": 1,
    }
    contaminated["outcome"] = {
        "reward": -1.0,
        "winner": 1,
    }
    contaminated["select"]["counterfactualBestCombo"] = [1]

    changed = guide.guide_scores(
        contaminated,
        [[0], [1]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )
    assert changed == baseline


def test_force_enabled_never_bypasses_deck_identity() -> None:
    assert (
        guide.guide_scores(
            _evolution_stage(),
            [[0], [1]],
            deck=[1] * 60,
            force_enabled=True,
        )
        is None
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_bound_guide_contract_and_brief_are_consistent() -> None:
    contract_path = (
        ROOT / "config" / "deck_guides" / "team-rockets-spidops.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    brief_path = ROOT / contract["expert_writeup"]["path"]
    brief = brief_path.read_text(encoding="utf-8")
    module_path = ROOT / "poke_bot" / "team_rockets_spidops_heuristics.py"

    assert contract["schema_version"] == "poke_bot.current_deck_guide/v1"
    assert contract["specialist_id"] == "team-rockets-spidops"
    assert contract["guide_version"] == guide.GUIDE_VERSION
    assert contract["teacher_module"] == (
        "poke_bot.team_rockets_spidops_heuristics"
    )
    assert contract["teacher_module_sha256"] == _sha256(module_path)
    assert contract["expert_writeup"] == {
        "path": "docs/deck_guides/team-rockets-spidops-expert-brief.txt",
        "sha256": _sha256(brief_path),
        "word_count": len(brief.split()),
        "maximum_words": 10000,
        "audience": "world_champion_subject_matter_experts",
        "guide_identity": "team-rockets-spidops",
        "cites_same_strategy_source_set": True,
        "primary_document_type": "practical_human_deck_pilot_guide",
    }
    assert 0 < len(brief.split()) <= 10_000
    source_urls = {row["url"] for row in contract["strategy_sources"]}
    brief_urls = {
        line.strip()
        for line in brief.splitlines()
        if line.strip().startswith("https://")
    }
    assert source_urls == brief_urls
    assert all(
        row["url"].startswith("https://") and row["reviewed_at_utc"]
        for row in contract["strategy_sources"]
    )
    assert contract["target_safety"]["future_information_allowed"] is False
    assert (
        contract["target_safety"]["outcome_conditioned_labels_allowed"]
        is False
    )
    assert contract["target_safety"]["runtime_authority"] == "none"


def test_contract_binds_the_exact_project_representative() -> None:
    contract = yaml.safe_load(
        (
            ROOT / "config" / "deck_guides" / "team-rockets-spidops.yaml"
        ).read_text(encoding="utf-8")
    )
    representative = json.loads(
        (
            ROOT
            / "data"
            / "training_mixes"
            / "specialist_representatives.v1.json"
        ).read_text(encoding="utf-8")
    )["decks"]["team-rockets-spidops"]
    binding = contract["project_representative_binding"]
    card_ids = representative["card_ids"]

    assert len(card_ids) == binding["card_count"] == len(CANONICAL_DECK) == 60
    assert sorted(card_ids) == sorted(CANONICAL_DECK)
    ordered_digest = "sha256:" + hashlib.sha256(
        json.dumps(card_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    multiset_digest = "sha256:" + hashlib.sha256(
        json.dumps(sorted(card_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert ordered_digest == representative["cards_sha256"]
    assert ordered_digest == binding["cards_sha256"]
    assert multiset_digest == representative["canonical_multiset_sha256"]
    assert multiset_digest == binding["canonical_multiset_sha256"]
