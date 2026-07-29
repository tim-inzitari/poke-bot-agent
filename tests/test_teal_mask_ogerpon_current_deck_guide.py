from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from poke_bot import deck_guides
from poke_bot import teal_mask_ogerpon_heuristics as guide


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DECK = json.loads(
    (
        ROOT
        / "data"
        / "training_mixes"
        / "specialist_representatives.v1.json"
    ).read_text(encoding="utf-8")
)["decks"]["teal-mask-ogerpon-ex"]["card_ids"]
CURRENT_PUBLIC_DECK = json.loads(
    (
        ROOT
        / "data"
        / "training_mixes"
        / "teal-mask-ogerpon-ex-public-full32.v1.json"
    ).read_text(encoding="utf-8")
)["source_deck_rows"][0]["card_ids"]
PUBLIC_SIGNATURE_AUDIT = json.loads(
    (
        ROOT
        / "data"
        / "training_mixes"
        / "teal-mask-ogerpon-ex-public-signature-audit.v1.json"
    ).read_text(encoding="utf-8")
)


def _player(
    *,
    active: list[int] | None = None,
    bench: list[int] | None = None,
    hand: list[int] | None = None,
) -> dict:
    return {
        "active": [{"id": value} for value in (active or [])],
        "bench": [{"id": value} for value in (bench or [])],
        "hand": [{"id": value} for value in (hand or [])],
        "discard": [],
        "deckCount": 30,
        "prize": [None] * 6,
    }


def _obs(me: dict, options: list[dict], *, context: int) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, _player()],
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


def test_exact_japan_championships_winner_matches_guide_signature() -> None:
    assert len(CANONICAL_DECK) == 60
    assert guide.is_teal_mask_ogerpon_ex_deck(CANONICAL_DECK)
    assert guide.applies(CANONICAL_DECK)


def test_exact_current_public_list_matches_guide_signature() -> None:
    assert len(CURRENT_PUBLIC_DECK) == 60
    assert CURRENT_PUBLIC_DECK.count(guide.LILLIES_CLEFAIRY_EX) == 1
    assert guide.is_teal_mask_ogerpon_ex_deck(CURRENT_PUBLIC_DECK)
    assert guide.applies(CURRENT_PUBLIC_DECK)


def test_public_mega_kangaskhan_collisions_do_not_match_guide_signature() -> None:
    rows = PUBLIC_SIGNATURE_AUDIT["mega_kangaskhan_collision_rows"]

    assert len(rows) == 2
    assert all(row["card_ids"].count(guide.MEGA_KANGASKHAN_EX) == 4 for row in rows)
    assert all(not guide.applies(row["card_ids"]) for row in rows)


def test_opening_prefers_kangaskhan_active_over_support_basic() -> None:
    obs = _obs(
        _player(hand=[guide.MEGA_KANGASKHAN_EX, guide.PECHARUNT]),
        [
            {"type": 3, "area": guide.AREA_HAND, "index": 0},
            {"type": 3, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_SETUP_ACTIVE,
    )

    scores = _scores(obs)

    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_opening_bench_establishes_teal_mask_before_duplicate_support() -> None:
    obs = _obs(
        _player(
            active=[guide.MEGA_KANGASKHAN_EX],
            bench=[guide.LATIAS_EX],
            hand=[guide.TEAL_MASK_OGERPON_EX, guide.LATIAS_EX],
        ),
        [
            {"type": 3, "area": guide.AREA_HAND, "index": 0},
            {"type": 3, "area": guide.AREA_HAND, "index": 1},
        ],
        context=guide.CTX_SETUP_BENCH,
    )

    scores = _scores(obs)

    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_main_prefers_currently_legal_teal_dance() -> None:
    obs = _obs(
        _player(active=[guide.TEAL_MASK_OGERPON_EX]),
        [
            {
                "type": guide.OPT_ABILITY,
                "area": guide.AREA_ACTIVE,
                "index": 0,
            },
            {"type": 14},
        ],
        context=guide.CTX_MAIN,
    )

    scores = _scores(obs)

    assert scores is not None
    assert scores[0] > scores[1] + guide.ABSTENTION_MARGIN


def test_unsupported_attack_target_stage_masks_completely() -> None:
    obs = _obs(
        _player(active=[guide.TEAL_MASK_OGERPON_EX]),
        [{"type": 3, "area": guide.AREA_ACTIVE, "index": 0}],
        context=99,
    )

    assert (
        guide.guide_scores(
            obs,
            [[0]],
            deck=CANONICAL_DECK,
            force_enabled=True,
        )
        is None
    )


def test_generic_registry_dispatches_teal_mask_ogerpon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "POKEBOT_CURRENT_DECK_GUIDE",
        "teal-mask-ogerpon-ex",
    )
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")

    assert deck_guides.enabled()
    assert deck_guides.guide_version() == guide.GUIDE_VERSION
    assert "teal-mask-ogerpon-ex" in deck_guides.supported_ids()


def test_contract_separates_competitive_taxonomy_from_public_identity() -> None:
    contract = yaml.safe_load(
        (
            ROOT / "config/deck_guides/teal-mask-ogerpon-ex.yaml"
        ).read_text(encoding="utf-8")
    )
    evidence = contract["corpus_binding_evidence"]

    assert contract["guide_version"] == guide.GUIDE_VERSION
    assert contract["deck_signature"]["minimum_card_counts"][272] == 1
    assert contract["deck_signature"]["maximum_card_counts"][756] == 3
    assert contract["physical_source_archetype_id"] == "ogerpon-box"
    assert contract["ptcgreplay_public_archetype_id"] == 151
    assert contract["ptcgreplay_public_archetype_name"] == (
        "Teal Mask Ogerpon ex"
    )
    assert evidence["competitive_taxonomy_identity"] == "ogerpon-box"
    assert evidence["required_public_source_archetype_id"] == 151
    assert evidence["source_indexed_acting_seat_games"] == 1_135
    assert evidence["materialized_acting_seat_games"] == 2_300
    assert evidence["materialized_decisions"] == 156_692
    assert evidence["guide_rows"] == 10_495
    assert evidence["daily_receipts_verified"] == 32
    assert evidence["duplicate_episode_seat_keys"] == 0
    assert evidence["ready_receipt_sha256"] == (
        "sha256:216b60efe50709a7972081ad2c7df60614412a28b09b15458b659eadec8c547c"
    )
    assert evidence["promotion_receipt_sha256"] == (
        "sha256:db8aff2fd5076fd75c4e56384f9bd609dda79836169527a780ec2808722e2674"
    )
    assert evidence["source_window"]["days"] == 32
    assert evidence["public_deck_catalog"].endswith(
        "teal-mask-ogerpon-ex-public-full32.v1.json"
    )
