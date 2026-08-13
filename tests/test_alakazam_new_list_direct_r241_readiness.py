from __future__ import annotations

import hashlib
import json
from pathlib import Path

from poke_bot import alakazam_new_list_heuristics as guide
from poke_bot.train import (
    GUIDE_TRAINING_MODE_DIRECTIONAL,
    assert_strategic_curriculum_receipt_contract,
)
from scripts.register_next_specialist_runtime import (
    _validate_strategic_curriculum_bundle,
)
from scripts.reseal_alakazam_new_list_r241_guide import (
    build_readiness_payload,
)

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "state/alakazam-new-list-direct-r241-guide-readiness.json"
ROLE_MAP = ROOT / "state/alakazam-new-list-direct-r241-strategic-head-roles.json"
CURRICULUM = ROOT / "state/alakazam-new-list-direct-r241-strategic-curriculum.json"
VALIDATION = (
    ROOT / "state/alakazam-new-list-direct-r241-strategic-curriculum-validation.json"
)
CONTRACT = ROOT / "config/deck_guides/alakazam-new-list-direct-r241.yaml"
DECK = ROOT / "decks/archetype-samples/alakazam-new-list-direct-r241.csv"
MODULE = ROOT / "poke_bot/alakazam_new_list_heuristics.py"
WRITEUP = ROOT / "docs/deck_guides/alakazam-new-list-direct-r241-expert-brief.md"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _player(*, hand: list[int] | None = None, active_id: int = 741) -> dict:
    return {
        "active": [{"id": active_id, "hp": 60, "maxHp": 60, "energyCards": []}],
        "bench": [{"id": 741, "hp": 60, "maxHp": 60, "energyCards": []}],
        "deckCount": 30,
        "discard": [],
        "prize": [None] * 6,
        "hand": [{"id": card_id} for card_id in (hand or [])],
        "handCount": len(hand or []),
    }


def test_r241_readiness_binds_only_exact_deck_selfplay_guidance() -> None:
    readiness = _json(READINESS)
    cards = [int(value) for value in DECK.read_text(encoding="utf-8").splitlines()]
    canonical = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(sorted(cards), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )

    assert readiness["schema"] == (
        "poke_bot.alakazam_new_list_direct_policy_guide_readiness/v1"
    )
    assert readiness["status"] == "validated"
    assert readiness["scope"] == "exact_deck_direct_policy_ordinary_rl_only"
    assert readiness["guide_contract"] == {
        "path": "config/deck_guides/alakazam-new-list-direct-r241.yaml",
        "sha256": _sha256(CONTRACT),
    }
    assert readiness["expert_writeup"] == {
        "path": "docs/deck_guides/alakazam-new-list-direct-r241-expert-brief.md",
        "sha256": _sha256(WRITEUP),
        "word_count": len(WRITEUP.read_text(encoding="utf-8").split()),
        "maximum_words": 10000,
    }
    assert "It does not stop ordinary attack damage to the Bench." in (
        WRITEUP.read_text(encoding="utf-8").replace("\n", " ")
    )
    assert readiness["teacher"] == {
        **readiness["teacher"],
        "module": "poke_bot/alakazam_new_list_heuristics.py",
        "sha256": _sha256(MODULE),
        "exact_deck_gate_required": True,
        "complete_legal_stage_scoring_required": True,
        "incomplete_or_ambiguous_stage_behavior": "mask_entire_stage",
        "public_state_only": True,
    }
    assert readiness["exact_deck"]["canonical_multiset_sha256"] == canonical
    assert readiness["exact_deck"]["file_sha256"] == _sha256(DECK)
    assert readiness["exact_deck"]["card_count"] == len(cards) == 60

    ordinary = readiness["ordinary_rl"]
    expert = readiness["expert_soft_refresh"]
    assert ordinary["guide_loss_weight"] == 0.05
    assert ordinary["guide_target_generation_enabled"] is True
    assert ordinary["guide_rows"] == readiness["guide_rows"] == 1
    assert ordinary["guide_target_source"] == ("exact_new_deck_direct_policy_self_play")
    assert expert["guide_loss_weight"] == 0.0
    assert expert["guide_target_generation_enabled"] is False
    assert expert["expert_corpus_guide_rows"] == 0
    assert "exact new 60-card multiset" in expert["reason"]
    assert readiness["corpus_readiness"] == {
        "is_current_deck_guide_corpus_ready_receipt": False,
        "expert_corpus_ready_for_r241_guide_targets": False,
        "reason": "the only guide-ready path in this receipt is checksum-bound direct-policy self-play",
    }
    assert (
        readiness["checks"]["historical_r175_or_r79_guide_artifacts_consumed"] is False
    )
    assert readiness["runtime_exclusions"] == {
        "runtime_action_authority": False,
        "runtime_input": False,
        "runtime_action_logit_route": False,
        "mcts": False,
        "recursive_turn_planner": False,
        "guide2vec": False,
        "guide_logit_bias": False,
        "hidden_state_or_future_information": False,
    }


def test_r241_readiness_is_reproducible_from_canonical_sources() -> None:
    payload = build_readiness_payload()
    assert payload == _json(READINESS)
    assert (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode(
        "utf-8"
    ) == READINESS.read_bytes()


def test_r241_curriculum_trio_is_checksum_linked_and_directional_only() -> None:
    roles = _json(ROLE_MAP)
    curriculum = _json(CURRICULUM)
    validation = _json(VALIDATION)

    assert_strategic_curriculum_receipt_contract(
        specialist_id="alakazam",
        curriculum_spec=str(CURRICULUM),
        head_role_map=str(ROLE_MAP),
        validation_receipt=str(VALIDATION),
        expected_training_mode=GUIDE_TRAINING_MODE_DIRECTIONAL,
    )
    strict_bundle = _validate_strategic_curriculum_bundle(
        specialist_id="alakazam",
        guide_contract_sha256=_sha256(CONTRACT).removeprefix("sha256:"),
        curriculum_spec=CURRICULUM,
        curriculum_spec_sha256=_sha256(CURRICULUM).removeprefix("sha256:"),
        head_role_map=ROLE_MAP,
        head_role_map_sha256=_sha256(ROLE_MAP).removeprefix("sha256:"),
        validation_receipt=VALIDATION,
        validation_receipt_sha256=_sha256(VALIDATION).removeprefix("sha256:"),
        training_mode=GUIDE_TRAINING_MODE_DIRECTIONAL,
    )
    assert (
        strict_bundle["required_heads"] == roles["canonical_learned_decision_sources"]
    )
    assert curriculum["head_role_map_sha256"] == _sha256(ROLE_MAP)
    assert validation["curriculum_spec_sha256"] == _sha256(CURRICULUM)
    assert validation["head_role_map_sha256"] == _sha256(ROLE_MAP)
    assert validation["guide_ready_receipt"] == str(READINESS)
    assert validation["guide_ready_receipt_sha256"] == _sha256(READINESS)
    assert validation["measurements"]["guide_labeled_rows"] == 1
    assert validation["r241_exact_deck_guide_execution"] == {
        "guide_readiness_receipt": "state/alakazam-new-list-direct-r241-guide-readiness.json",
        "ordinary_rl": {
            "guide_loss_weight": 0.05,
            "guide_target_source": "exact_new_deck_direct_policy_self_play",
            "guide_target_generation_enabled": True,
        },
        "expert_soft_refresh": {
            "guide_loss_weight": 0.0,
            "guide_target_generation_enabled": False,
            "expert_corpus_guide_rows": 0,
            "reason": "historical expert rows do not carry the exact new 60-card multiset",
        },
        "historical_r175_or_r79_guide_artifacts_consumed": False,
        "runtime_guide_authority": False,
        "mcts_rtp_guide2vec": False,
    }

    pairwise = {
        name
        for name, row in roles["heads"].items()
        if row["guide_pairwise_route_direction_allowed"]
    }
    assert pairwise == {
        "action_q",
        "action_resource",
        "action_utility",
        "setup_board_outcome",
    }
    assert curriculum["guide_pairwise_route_heads"] == sorted(pairwise)
    for row in roles["heads"].values():
        assert row["guide_action_target_allowed"] is False
        assert row["direct_action_selection_authority"] is False
        assert row["route_input"] == "typed_output_centered_option_interaction"
        assert row["zero_safe_final_projection"] is True

    for path in (READINESS, ROLE_MAP, CURRICULUM, VALIDATION):
        text = path.read_text(encoding="utf-8")
        assert "config/deck_guides/alakazam-final-refresh.yaml" not in text
        assert "state/final_format_alakazam_curriculum_r79" not in text
        assert "state/final_format_alakazam_guide_ready_r79.json" not in text


def test_r241_readiness_canary_is_nonflat_only_for_the_exact_deck() -> None:
    me = _player(hand=[guide.BATTLE_CAGE])
    # Battle Cage matters against Froslass only for a public Bench Ability;
    # a bare Abra must not become a generic early-Stadium label.
    me["bench"][0]["id"] = guide.KADABRA
    opponent = _player(active_id=guide.FROSLASS)
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [me, opponent],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": 0,
            "option": [{"type": 7, "index": 0}, {"type": 14}],
            "minCount": 1,
            "maxCount": 1,
        },
    }

    scores = guide.guide_scores(
        observation,
        [[0], [1]],
        deck=guide.EXACT_DECK,
        force_enabled=True,
    )
    assert scores is not None
    assert scores[0] > scores[1]
    assert (
        guide.guide_scores(
            observation,
            [[0], [1]],
            deck=(*guide.EXACT_DECK[:-1], 1),
            force_enabled=True,
        )
        is None
    )

    fez_observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                _player(hand=[guide.FEZANDIPITI_EX]),
                _player(active_id=900),
            ],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": 0,
            "option": [{"type": 7, "index": 0}, {"type": 14}],
            "minCount": 1,
            "maxCount": 1,
        },
    }
    assert (
        guide.guide_scores(
            fez_observation,
            [[0], [1]],
            deck=guide.EXACT_DECK,
            force_enabled=True,
        )
        is None
    )
