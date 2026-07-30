from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from poke_bot import deck_guides
from poke_bot import slowking_heuristics as guide
from poke_bot.ladder_deck_mix import canonical_payload_digest

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_PATH = (
    ROOT
    / "config"
    / "deck_guides"
    / "slowking-representative.v1.json"
)
CONTRACT_PATH = (
    ROOT / "config" / "deck_guides" / "slowking.yaml"
)
COMBO_COVERAGE_PATH = ROOT / "state" / "slowking_combo_head_coverage_v1.json"
REPRESENTATIVE = json.loads(REPRESENTATIVE_PATH.read_text(encoding="utf-8"))
CANONICAL_DECK = REPRESENTATIVE["deck"]["card_ids"]


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def _scores(obs: dict) -> list[float] | None:
    return guide.guide_scores(
        obs,
        [[0], [1]],
        deck=CANONICAL_DECK,
        force_enabled=True,
    )


def test_exact_specialist_representative_is_checksum_bound() -> None:
    deck = REPRESENTATIVE["deck"]
    ordered = _digest_bytes(
        json.dumps(CANONICAL_DECK, separators=(",", ":")).encode("utf-8")
    )
    multiset = _digest_bytes(
        json.dumps(sorted(CANONICAL_DECK), separators=(",", ":")).encode(
            "utf-8"
        )
    )

    assert REPRESENTATIVE["schema"] == (
        "poke_bot.specialist_deck_representative/v1"
    )
    assert REPRESENTATIVE["status"] == (
        "future_specialist_identity_ready_training_blocked"
    )
    assert REPRESENTATIVE["artifact_sha256"] == canonical_payload_digest(
        REPRESENTATIVE
    )
    assert len(CANONICAL_DECK) == deck["card_count"] == 60
    assert (
        deck["pokemon_count"],
        deck["trainer_count"],
        deck["energy_count"],
    ) == (22, 28, 10)
    assert ordered == deck["cards_sha256"]
    assert multiset == deck["canonical_multiset_sha256"]
    assert ordered == multiset
    assert REPRESENTATIVE["source"]["source_deck_hash"] == "e54ea8e5d444d094"
    assert guide.is_slowking_deck(CANONICAL_DECK)
    assert guide.applies(reversed(CANONICAL_DECK))


def test_exact_deck_predicate_rejects_a_single_card_mutation() -> None:
    mutated = list(CANONICAL_DECK)
    mutated[0] = 1

    assert not guide.is_slowking_deck(mutated)
    assert not guide.applies(mutated)


def test_guide_is_registry_gated_and_runtime_bias_is_neutral(
    monkeypatch,
) -> None:
    monkeypatch.delenv("POKEBOT_CURRENT_DECK_GUIDE", raising=False)
    monkeypatch.delenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", raising=False)
    assert guide.RESEARCH_ONLY is False
    assert guide.enabled() is False
    assert guide.prior_logit_bias({}, [[0], [1]], scale=99.0) == [0.0, 0.0]

    obs = _obs(
        _player(
            active=[guide.SMOOCHUM],
            hand=[guide.SLOWPOKE, guide.MEGA_KANGASKHAN_EX],
        ),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_SETUP_BENCH,
    )
    assert guide.guide_scores(obs, [[0], [1]], deck=CANONICAL_DECK) is None
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "slowking")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    assert guide.enabled() is True
    assert "slowking" in deck_guides.supported_ids()
    assert "dragapult" not in deck_guides.supported_ids()


def test_setup_bench_prefers_the_second_slowpoke_gap() -> None:
    obs = _obs(
        _player(
            active=[guide.SMOOCHUM],
            bench=[guide.SLOWPOKE],
            hand=[guide.SLOWPOKE, guide.MEGA_KANGASKHAN_EX],
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


def test_poke_pad_prefers_visible_slowking_evolution_gap() -> None:
    deck_cards = [guide.SLOWKING, guide.KYUREM]
    obs = _obs(
        _player(active=[guide.SLOWPOKE]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 1},
        ],
        context=guide.CTX_TO_HAND,
        effect_id=guide.POKE_PAD,
        deck_cards=deck_cards,
    )

    scores = _scores(obs)

    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_telepath_search_prefers_second_slowpoke_from_exposed_options() -> None:
    deck_cards = [guide.SLOWPOKE, guide.LATIAS_EX]
    obs = _obs(
        _player(active=[guide.SMOOCHUM], bench=[guide.SLOWPOKE]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DECK, "index": 1},
        ],
        context=guide.CTX_TO_BENCH,
        effect_id=guide.TELEPATH_PSYCHIC_ENERGY,
        deck_cards=deck_cards,
    )

    scores = _scores(obs)

    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_night_stretcher_energy_option_masks_the_complete_stage() -> None:
    me = _player(
        active=[guide.SLOWPOKE],
        discard=[guide.BASIC_PSYCHIC_ENERGY, guide.SLOWKING],
    )
    obs = _obs(
        me,
        [
            {"type": guide.OPT_ENERGY_CARD, "area": guide.AREA_DISCARD, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_DISCARD, "index": 1},
        ],
        context=guide.CTX_TO_HAND,
        effect_id=guide.NIGHT_STRETCHER,
    )

    assert _scores(obs) is None


def test_opening_active_and_top_deck_routes_remain_masked() -> None:
    opening = _obs(
        _player(
            hand=[guide.SMOOCHUM, guide.MEGA_KANGASKHAN_EX],
        ),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_SETUP_ACTIVE,
    )
    academy = _obs(
        _player(hand=[guide.KYUREM, guide.CONKELDURR]),
        [
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 0},
            {"type": guide.OPT_CARD, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_TO_HAND,
        effect_id=guide.ACADEMY_AT_NIGHT,
    )

    assert _scores(opening) is None
    assert _scores(academy) is None


def test_specialist_contract_binds_guide_proposal_teacher_and_sources() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    writeup_path = ROOT / contract["expert_writeup"]["path"]
    proposal_path = ROOT / contract["heuristic_research"]["proposal_path"]
    teacher_path = ROOT / contract["heuristic_research"]["teacher_module_path"]
    writeup = writeup_path.read_text(encoding="utf-8")

    assert contract["schema_version"] == "poke_bot.current_deck_guide/v1"
    assert contract["specialist_id"] == "slowking"
    assert contract["guide_version"] == guide.GUIDE_VERSION
    assert contract["project_representative_binding"]["path"] == str(
        REPRESENTATIVE_PATH.relative_to(ROOT)
    )
    assert contract["expert_writeup"]["sha256"] == _digest_bytes(
        writeup_path.read_bytes()
    )
    assert contract["expert_writeup"]["word_count"] == len(writeup.split())
    assert contract["expert_writeup"]["word_count"] <= 10_000
    assert contract["heuristic_research"]["proposal_sha256"] == _digest_bytes(
        proposal_path.read_bytes()
    )
    assert contract["heuristic_research"]["teacher_module_sha256"] == (
        _digest_bytes(teacher_path.read_bytes())
    )
    expected_source_ids = [f"S{index}" for index in range(1, 35)]
    assert contract["strategy_source_set"]["source_ids"] == expected_source_ids
    assert all(f"[{source_id}]" in writeup for source_id in expected_source_ids)
    assert contract["target_safety"]["missing_label_behavior"] == "mask_not_zero"
    assert contract["target_safety"]["future_information_allowed"] is False
    assert contract["target_safety"]["runtime_authority"] == "none"
    assert (
        contract["heuristic_research"]["policy_target"]["training_mode"]
        == "strategic_curriculum_v1"
    )
    assert (
        contract["heuristic_research"]["policy_target"][
            "direct_policy_cross_entropy_allowed"
        ]
        is False
    )
    assert (
        contract["heuristic_research"]["policy_target"][
            "guide_preference_index_may_affect_loss_or_gradient"
        ]
        is False
    )


def test_slowking_is_required_but_fail_closed_behind_archaludon() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    roster = json.loads(
        (ROOT / "state/matchup_adapter_roster.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = json.loads(
        (ROOT / "ops/frozen_specialist_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    representatives = json.loads(
        (
            ROOT
            / "data"
            / "training_mixes"
            / "specialist_representatives.v1.json"
        ).read_text(encoding="utf-8")
    )
    false_authority_keys = {
        "selector_eligible",
        "runtime_registry_registered",
        "matchup_route_registered",
        "bootstrap_authorized",
        "training_authorized",
        "gate_authorized",
        "freeze_authorized",
        "submission_authorized",
        "service_registered",
    }
    assert all(contract["authority"][key] is False for key in false_authority_keys)
    assert contract["authority"]["required_specialist"] is True
    assert contract["authority"]["completion_eligible"] is True
    assert contract["authority"]["current_deck_guide_registry_registered"] is True
    assert contract["authority"]["prestage_registered"] is True
    assert contract["authority"]["corpus_build_authorized"] is True
    assert contract["authority"]["dashboard_training_queue_registered"] is True
    assert contract["authority"]["runtime_authority"] == "none"
    assert contract["activation_guard"]["activation_authorized"] is False
    assert contract["activation_guard"]["predecessor_specialist"] == (
        "archaludon-ex"
    )
    assert contract["activation_guard"]["strict_order"] == [
        "archaludon-ex",
        "slowking",
    ]
    assert contract["activation_guard"]["owner_decision_revisions"][-1] == 64
    assert contract["activation_guard"]["after_completion"] == (
        "immediate_final_format_alakazam_refresh_then_marnie_refresh_and_"
        "final_submission_preparation"
    )
    assert contract["training_staging"]["status"] == (
        "blocked_behind_archaludon_and_combo_head_receipt"
    )

    state_ids = {row["id"] for row in state["specialists"]}
    assert len(state_ids) == 16
    assert state["target_registry"]["required_target_count"] == 15
    assert "slowking" in state_ids
    assert "dragapult" not in state_ids
    assert state["training_priority"]["ordered_unfinished_ids_after_active"] == [
        "archaludon-ex",
        "slowking",
    ]
    assert "slowking" not in roster["active_expert_ids"]
    assert "slowking" not in roster["expert_ids"]
    assert "slowking" not in roster["specialist_priority"]
    assert all(slot["archetype_id"] != "slowking" for slot in roster["slots"])
    assert "slowking" not in runtime["specialists"]
    assert all(
        row["specialist_id"] != "slowking" for row in frozen["specialists"]
    )
    assert "slowking" in deck_guides.supported_ids()
    assert "slowking" in representatives["decks"]
    assert representatives["decks"]["slowking"]["card_ids"] == CANONICAL_DECK
    slowking = next(row for row in state["specialists"] if row["id"] == "slowking")
    assert slowking["status"] == "blocked"
    assert slowking["selector_eligible"] is False
    assert slowking["training_order"]["predecessor"] == "archaludon-ex"
    assert slowking["training_order"]["after_completion"] == (
        "immediate_final_format_alakazam_refresh_then_marnie_refresh_and_"
        "final_submission_preparation"
    )
    assert slowking["heads"]["combo_head_coverage"]["validation_receipt"] is None
    assert slowking["historical_plain_dragapult_evidence"][
        "corpus_and_audits_preserved"
    ] is True

    crustle = next(row for row in state["specialists"] if row["id"] == "crustle")
    assert crustle["required_specialist"] is False
    assert crustle["selector_eligible"] is False
    assert crustle["completion_eligible"] is False
    assert crustle["submission_authorized"] is False
    assert crustle["matchup_router"] == {
        "stable_matchup_slot": 0,
        "status": "active",
        "lineage": "v5:0",
        "archetype_id": "crustle",
        "source_crosswalk": {
            "source_id": 55,
            "source_name": "Crustle",
        },
        "may_be_deleted_disabled_reindexed_or_reused": False,
    }
    assert crustle["public_practice_gate_opponent"] == {
        "opponent_id": "pilkwang-meta-20260708",
        "archetype_id": "crustle",
        "archetype_label": "Crustle / Great Tusk",
        "source": (
            "pilkwang/pok-mon-tcg-ai-battle-meta-snapshot-08-july"
        ),
        "content_digest": (
            "sha256:"
            "7120bc67415e06c1cf69d64574f1a415"
            "45fd4c2fd084a029d77c5e43a357957f"
        ),
        "inference_only": True,
        "gradient_or_update_authority": False,
        "specialist_training_authority": False,
        "counts_toward_required_fleet": False,
    }


def test_combo_head_coverage_map_is_checksum_bound_and_honestly_blocked() -> None:
    coverage = json.loads(COMBO_COVERAGE_PATH.read_text(encoding="utf-8"))
    required = {
        "top_deck_construction_and_consumption",
        "copied_non_rule_box_attack_legality_and_choice",
        "visible_combo_piece_search_and_recovery",
        "psychic_telepath_boomerang_and_acceleration",
        "slowpoke_slowking_and_next_attacker_bench_continuity",
        "opponent_disruption_and_response",
        "prize_mapping_and_remaining_turn_outcome_timing",
    }

    assert coverage["artifact_sha256"] == canonical_payload_digest(coverage)
    assert {
        row["id"] for row in coverage["coverage_requirements"]
    } == required
    assert coverage["missing_typed_head"]["id"] == "combo_state"
    assert coverage["missing_typed_head"]["implementation_status"] == (
        "not_implemented"
    )
    assert coverage["missing_typed_head"]["outputs"] == 32
    assert coverage["fusion_contract"]["guide_is_only_no_route_exception"] is True
    assert coverage["required_validation"]["receipt"] is None
    assert coverage["training_ready"] is False
    assert coverage["launch_ready"] is False


def test_slowking_parameter_exception_is_scoped_and_bounded() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    budget = contract["specialist_parameter_budget"]

    assert budget["scope"] == "slowking_only"
    assert budget["ordinary_soft_target"] == 2_000_000
    assert budget["soft_target_may_be_exceeded"] is True
    assert budget["initial_fail_closed_hard_ceiling"] == 3_500_000
    assert budget["global_parameter_limit_override"] is False
    assert budget["proposed_added_parameter_total_at_d_model_96"] == (
        24_800 + 2_081
    )
    assert budget["complete_candidate_parameter_count"] is None
    assert budget["exception_receipt_required_if_above_soft_target"] is True
    assert budget["above_hard_ceiling_requires_new_owner_decision"] is True


def test_walrein_is_removed_from_training_but_keeps_protected_matchup_slot() -> None:
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    roster = json.loads(
        (ROOT / "state/matchup_adapter_roster.json").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )["current_owner_overrides"]["slowking_specialist_replacement"]

    assert "walrein" not in {row["id"] for row in state["specialists"]}
    assert "walrein" not in state["training_priority"][
        "ordered_unfinished_ids_after_active"
    ]
    assert "walrein" not in roster["specialist_priority"]
    assert "walrein" in roster["active_expert_ids"]
    assert "walrein" in roster["expert_ids"]
    slot = next(row for row in roster["slots"] if row["slot"] == 13)
    assert slot == {
        "slot": 13,
        "archetype_id": "walrein",
        "status": "active",
        "lineage": "v5:13",
    }
    assert compatibility["walrein"]["selection_eligible"] is False
    assert compatibility["walrein"]["completion_eligible"] is False
    assert compatibility["walrein"]["matchup_route_preserved"] is True
    assert compatibility["walrein"]["stable_matchup_slot"] == 13
