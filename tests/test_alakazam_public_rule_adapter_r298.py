"""Focused unit coverage for the additive r298 public-rule representation."""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import pytest

from poke_bot.alakazam_public_rule_adapter_r298 import (
    DEFAULT_PUBLIC_CATALOG_PINS,
    MAX_NUMBER_VALUE,
    PublicRuleAdapterError,
    build_public_rule_representation,
    extract_public_terminal_target,
    is_public_catalog_eligible,
    load_public_rule_adapter_config,
    load_sealed_public_catalog,
    public_rule_observation_fingerprint,
    sanitize_public_observation,
    semantic_option_key,
    validate_public_catalog_provenance,
)


def _card(
    card_id: int,
    serial: int,
    *,
    hp: int = 100,
    max_hp: int = 100,
    energy: list[dict] | None = None,
    pre_evolution: list[dict] | None = None,
    appear_this_turn: bool | None = None,
) -> dict:
    row: dict = {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": max_hp,
        "energyCards": list(energy or []),
        "tools": [],
    }
    if pre_evolution is not None:
        row["preEvolution"] = list(pre_evolution)
    if appear_this_turn is not None:
        row["appearThisTurn"] = appear_this_turn
    return row


def _player(
    seat: int,
    *,
    active: list[dict] | None = None,
    bench: list[dict] | None = None,
    hand: list[dict] | None = None,
    discard: list[dict] | None = None,
    prize: list[dict | None] | None = None,
    deck: list[dict] | None = None,
) -> dict:
    return {
        "active": list(active or []),
        "bench": list(bench or []),
        "hand": hand,
        "handCount": 0 if hand is None else len(hand),
        "discard": list(discard or []),
        "prize": list(prize or [None] * 6),
        "deck": deck,
        "deckCount": 20,
        "benchMax": 5,
    }


def _observation(options: list[dict]) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "turnActionCount": 1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "players": [
                _player(
                    0,
                    active=[
                        _card(
                            10,
                            100,
                            hp=80,
                            max_hp=120,
                            energy=[
                                {"id": 20, "serial": 200, "energyType": "Psychic"}
                            ],
                            pre_evolution=[{"id": 9, "serial": 99}],
                            appear_this_turn=False,
                        )
                    ],
                    bench=[_card(77, 1001), _card(77, 1002)],
                    hand=[_card(30, 300)],
                ),
                _player(
                    1,
                    active=[_card(11, 101)],
                    # These intentionally leaked fields must be fully removed
                    # from a policy representation.
                    hand=[_card(901, 9001)],
                    prize=[_card(902, 9002)] + [None] * 5,
                    deck=[_card(903, 9003)],
                ),
            ],
            "looking": [{"id": 31, "serial": 301}],
            "stadium": [],
        },
        "select": {
            "context": "Main",
            "type": "Card",
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 2,
            "remainEnergyCost": 1,
            "option": options,
        },
    }


def test_public_projection_is_hidden_information_invariant() -> None:
    options = [
        {"type": "Skill", "cardId": 77, "serial": 1001},
        {"type": "Number", "number": 4},
    ]
    first = _observation(options)
    second = copy.deepcopy(first)
    opponent = second["current"]["players"][1]
    opponent["hand"] = [_card(1201, 12001), _card(1202, 12002)]
    opponent["deck"] = [_card(1301, 13001), _card(1302, 13002)]
    opponent["prize"] = [_card(1401, 14001)] + [None] * 5

    encoded_first = build_public_rule_representation(first)
    encoded_second = build_public_rule_representation(second)

    assert encoded_first.public_observation_hash == encoded_second.public_observation_hash
    assert encoded_first.semantic_token_hash == encoded_second.semantic_token_hash
    assert encoded_first.to_dict() == encoded_second.to_dict()
    assert "901" not in str(encoded_first.to_dict())
    assert "terminal" not in encoded_first.state


def test_public_projection_strictly_drops_root_and_actor_private_aliases() -> None:
    """Public fingerprints cannot drift with leaked replay/debug payloads."""

    first = _observation([{"type": "Number", "number": 4}])
    second = copy.deepcopy(first)
    second["logs"] = {"private_opponent_deck": [901, 902]}
    second["search_begin_input"] = {"opponent_hand_ids": [903, 904]}
    second["transition_after"] = {"future_prizes": [905]}
    second["privateState"] = {"hidden": True}
    second["current"]["players"][0]["deckOrder"] = [10, 11, 12]
    second["current"]["players"][0]["privateState"] = {"own_prizes": [13]}
    second["current"]["players"][0]["deck"] = [_card(14, 140)]

    from poke_bot.alakazam_public_rule_adapter_r298 import (
        sanitize_public_observation,
    )
    from poke_bot.alakazam_simulator_rule_targets_r298 import (
        public_observation_fingerprint,
    )

    assert sanitize_public_observation(first) == sanitize_public_observation(second)
    assert public_observation_fingerprint(first) == public_observation_fingerprint(second)
    assert build_public_rule_representation(first).to_dict() == (
        build_public_rule_representation(second).to_dict()
    )


def test_public_fingerprint_is_serial_invariant_and_drops_adversarial_aliases() -> None:
    first = _observation([{"type": "Skill", "cardId": 77, "serial": 1001}])
    second = copy.deepcopy(first)
    second.update(
        {
            "searchBeginInput": {"opponent_hand_ids": [901]},
            "history": [{"private": [902]}],
            "future": {"outcome": 903},
            "visualization": {"private": 904},
            "aux": {"private": 905},
        }
    )
    second["current"].update(
        {
            "privateState": {"full_deck": [906]},
            "hiddenState": {"private": 907},
            "deckOrder": [908],
            "transition_after": {"future": 909},
        }
    )
    actor = second["current"]["players"][0]
    opponent = second["current"]["players"][1]
    actor.update({"deck": [_card(910, 911)], "deckOrder": [912], "privateState": {"x": 1}})
    opponent.update(
        {
            "hand": [_card(913, 914)],
            "deck": [_card(915, 916)],
            "prize": [_card(917, 918)] + [None] * 5,
            "deckOrder": [919],
            "privateState": {"x": 2},
        }
    )
    # Serial is only an internal locator join key.  A consistent renumbering
    # leaves the externally visible owner/area/slot binding unchanged.
    second["current"]["players"][0]["bench"][0]["serial"] = 5001
    second["select"]["option"][0]["serial"] = 5001

    assert public_rule_observation_fingerprint(first) == public_rule_observation_fingerprint(second)
    assert build_public_rule_representation(first).to_dict() == build_public_rule_representation(second).to_dict()


def test_exposed_deck_and_looking_menu_order_do_not_enter_semantics() -> None:
    """Visible menu order is execution alignment, not a deck-order feature."""

    first = _observation(
        [
            {
                "type": "Card",
                "playerIndex": 0,
                "area": "Deck",
                "index": 0,
                "cardId": 42,
                "serial": 4201,
            }
        ]
    )
    first["select"]["deck"] = [_card(42, 4201), _card(43, 4301)]
    first["current"]["looking"] = [_card(44, 4401), _card(45, 4501)]
    second = copy.deepcopy(first)
    second["select"]["deck"] = list(reversed(second["select"]["deck"]))
    second["select"]["option"][0]["index"] = 1
    second["current"]["looking"] = list(reversed(second["current"]["looking"]))

    assert sanitize_public_observation(first) == sanitize_public_observation(second)
    encoded_first = build_public_rule_representation(first)
    encoded_second = build_public_rule_representation(second)
    assert encoded_first.to_dict() == encoded_second.to_dict()
    assert public_rule_observation_fingerprint(first) == public_rule_observation_fingerprint(second)
    source = encoded_first.options[0].semantic["option"]["source"]
    assert source["area"] == "deck"
    assert source["slot"] is None
    assert source["physical_source"]["slot"] is None


@pytest.mark.parametrize("area", ["Hand", "Discard"])
def test_hand_and_discard_list_order_do_not_enter_semantics(area: str) -> None:
    """Only board positions—not public multiset display positions—bind rows."""

    first = _observation(
        [
            {
                "type": "Card",
                "playerIndex": 0,
                "area": area,
                "index": 0,
                "cardId": 30,
                "serial": 300,
            }
        ]
    )
    zone = "hand" if area == "Hand" else "discard"
    first["current"]["players"][0][zone] = [_card(30, 300), _card(31, 301)]
    if zone == "hand":
        first["current"]["players"][0]["handCount"] = 2
    else:
        # The base fixture contains hand serial 300; avoid manufacturing an
        # impossible cross-zone duplicate while testing list-order behavior.
        first["current"]["players"][0]["hand"] = []
        first["current"]["players"][0]["handCount"] = 0
    second = copy.deepcopy(first)
    second["current"]["players"][0][zone] = list(
        reversed(second["current"]["players"][0][zone])
    )
    second["select"]["option"][0]["index"] = 1

    assert sanitize_public_observation(first) == sanitize_public_observation(second)
    encoded_first = build_public_rule_representation(first)
    encoded_second = build_public_rule_representation(second)
    assert encoded_first.to_dict() == encoded_second.to_dict()
    assert public_rule_observation_fingerprint(first) == public_rule_observation_fingerprint(second)
    source = encoded_first.options[0].semantic["option"]["source"]
    assert source["area"] == area.lower()
    assert source["slot"] is None
    assert source["physical_source"]["slot"] is None


def test_unrevealed_actor_prize_identity_never_enters_public_semantics() -> None:
    """A legal Prize-slot action keeps only public location semantics."""

    first = _observation(
        [
            {
                "type": "Card",
                "playerIndex": 0,
                "area": "Prize",
                "index": 0,
                "cardId": 901,
                "serial": 9001,
                "toolCardId": 902,
                "toolSerial": 9002,
            }
        ]
    )
    first["current"]["players"][0]["prize"] = [_card(901, 9001)] + [None] * 5
    # A persisted policy observation also carries select.effect, so exercise
    # the direct effect-card leak separately from source/target bindings.
    first["select"]["effect"] = {
        "effectId": 7,
        "sourceArea": "Prize",
        "cardId": 901,
    }
    second = copy.deepcopy(first)
    second["current"]["players"][0]["prize"] = [_card(902, 9002)] + [None] * 5
    second["select"]["option"][0].update(
        {"cardId": 902, "serial": 9002, "toolCardId": 903, "toolSerial": 9003}
    )
    second["select"]["effect"]["cardId"] = 902

    assert sanitize_public_observation(first) == sanitize_public_observation(second)
    assert sanitize_public_observation(first)["select"]["effect"] == {"effectId": 7}
    encoded_first = build_public_rule_representation(first)
    encoded_second = build_public_rule_representation(second)
    assert encoded_first.public_observation_hash == encoded_second.public_observation_hash
    assert encoded_first.semantic_token_hash == encoded_second.semantic_token_hash
    assert encoded_first.options[0].semantic_key_sha256 == encoded_second.options[0].semantic_key_sha256
    assert public_rule_observation_fingerprint(first) == public_rule_observation_fingerprint(second)
    source = encoded_first.options[0].semantic["option"]["source"]
    assert source == {
        "owner": "acting",
        "area": "prize",
        "slot": 0,
        "card_id": None,
        "physical_source": None,
        "physical_source_status": "unavailable_unrevealed_prize",
    }
    attachment = encoded_first.options[0].semantic["option"]["attachments"][0]
    assert attachment["card_id"] is None
    assert attachment["physical_source"] is None
    assert "901" not in str(encoded_first.to_dict())
    assert "902" not in str(encoded_first.to_dict())
    assert "9001" not in str(encoded_first.to_dict())
    assert "9002" not in str(encoded_first.to_dict())


def test_unscoped_skill_identity_is_unavailable_not_a_card_feature() -> None:
    """A bare Skill/cardId is not proof that the card is actor-visible."""

    first = _observation([{"type": "Skill", "cardId": 901, "skillId": 1}])
    second = _observation([{"type": "Skill", "cardId": 902, "skillId": 2}])

    assert sanitize_public_observation(first) == sanitize_public_observation(second)
    encoded_first = build_public_rule_representation(first)
    encoded_second = build_public_rule_representation(second)
    assert encoded_first.to_dict() == encoded_second.to_dict()
    assert public_rule_observation_fingerprint(first) == public_rule_observation_fingerprint(second)
    option = encoded_first.options[0].semantic["option"]
    assert option["card_id"] is None
    assert option["skill_identity"] == {
        "card_id": None,
        "skill_id": None,
        "physical_source": None,
        "physical_source_status": "unavailable_unproven_public_identity",
    }
    assert "card_id': 901" not in str(encoded_first.to_dict())
    assert "card_id': 902" not in str(encoded_second.to_dict())


def test_unproven_deck_card_identity_is_unavailable_without_exposed_menu() -> None:
    """A Deck area string alone cannot certify an otherwise hidden card ID."""

    first = _observation(
        [
            {
                "type": "Card",
                "playerIndex": 0,
                "area": "Deck",
                "index": 0,
                "cardId": 901,
            }
        ]
    )
    second = copy.deepcopy(first)
    second["select"]["option"][0]["cardId"] = 902

    assert sanitize_public_observation(first) == sanitize_public_observation(second)
    encoded_first = build_public_rule_representation(first)
    encoded_second = build_public_rule_representation(second)
    assert encoded_first.to_dict() == encoded_second.to_dict()
    assert public_rule_observation_fingerprint(first) == public_rule_observation_fingerprint(second)
    source = encoded_first.options[0].semantic["option"]["source"]
    assert source == {
        "owner": "acting",
        "area": "deck",
        "slot": None,
        "card_id": None,
        "physical_source": None,
        "physical_source_status": "unavailable_unproven_public_identity",
    }
    assert "card_id': 901" not in str(encoded_first.to_dict())
    assert "card_id': 902" not in str(encoded_second.to_dict())


@pytest.mark.parametrize(
    ("select_field", "first_value", "second_value", "semantic_field"),
    [
        ("contextCard", {"id": 901}, {"id": 902}, "context_card"),
        ("contextCard", {"cardId": 901}, {"cardId": 902}, "context_card"),
        ("effect", {"effectId": 7, "cardId": 901}, {"effectId": 7, "cardId": 902}, "effect_source"),
        (
            "effect",
            {"effectId": 7, "source": {"id": 901}},
            {"effectId": 7, "source": {"id": 902}},
            "effect_source",
        ),
    ],
)
def test_unproven_context_and_effect_identities_are_not_public_features(
    select_field: str,
    first_value: dict,
    second_value: dict,
    semantic_field: str,
) -> None:
    """Bare context/effect IDs require the same visible-locator proof as options."""

    first = _observation([{"type": "Number", "number": 4}])
    first["select"][select_field] = first_value
    second = copy.deepcopy(first)
    second["select"][select_field] = second_value

    assert sanitize_public_observation(first) == sanitize_public_observation(second)
    encoded_first = build_public_rule_representation(first)
    encoded_second = build_public_rule_representation(second)
    assert encoded_first.to_dict() == encoded_second.to_dict()
    assert public_rule_observation_fingerprint(first) == public_rule_observation_fingerprint(second)
    reference = encoded_first.selection[semantic_field]
    assert reference is not None
    if semantic_field == "context_card":
        assert reference["card_id"] is None
        assert reference["physical_source_status"] == "unavailable_unproven_public_identity"
    else:
        assert reference["card_id"] is None
        if reference["source"] is not None:
            assert reference["source"]["card_id"] is None
    assert all(not option.referenced_card_ids for option in encoded_first.options)


def test_number_values_are_exact_and_overflow_fails_closed() -> None:
    observation = _observation(
        [
            {"type": "Number", "number": 4},
            {"type": "Number", "number": 5},
        ]
    )
    encoded = build_public_rule_representation(observation)
    assert encoded.options[0].semantic_key_sha256 != encoded.options[1].semantic_key_sha256
    assert encoded.options[0].semantic["option"]["number"] == 4
    assert encoded.options[1].semantic["option"]["number"] == 5

    overflow = _observation([{"type": "Number", "number": MAX_NUMBER_VALUE + 1}])
    with pytest.raises(PublicRuleAdapterError, match="exceeds"):
        build_public_rule_representation(overflow)


def test_skill_uses_normalized_visible_physical_source_not_raw_serial() -> None:
    observation = _observation(
        [
            {"type": "Skill", "cardId": 77, "serial": 1001},
            {"type": "Skill", "cardId": 77, "serial": 1002},
        ]
    )
    encoded = build_public_rule_representation(observation)
    first = encoded.options[0].semantic["option"]["skill_identity"]
    second = encoded.options[1].semantic["option"]["skill_identity"]

    assert first["physical_source"]["area"] == "bench"
    assert first["physical_source"]["slot"] == 0
    assert second["physical_source"]["area"] == "bench"
    assert second["physical_source"]["slot"] == 1
    assert "1001" not in str(first)
    assert "1002" not in str(second)
    assert encoded.options[0].semantic_key_sha256 != encoded.options[1].semantic_key_sha256


def test_option_permutation_only_permutates_rows() -> None:
    left = _observation(
        [
            {"type": "Number", "number": 4},
            {"type": "Skill", "cardId": 77, "serial": 1001},
            {"type": "Skill", "cardId": 77, "serial": 1002},
        ]
    )
    right = copy.deepcopy(left)
    right["select"]["option"] = list(reversed(right["select"]["option"]))

    first = build_public_rule_representation(left)
    second = build_public_rule_representation(right)
    assert first.canonical_option_multiset_hash == second.canonical_option_multiset_hash
    assert first.semantic_token_hash == second.semantic_token_hash
    assert [row.semantic_key_sha256 for row in second.options] == list(
        reversed([row.semantic_key_sha256 for row in first.options])
    )


def test_stable_discriminator_is_ignored_without_collision_and_used_for_collision() -> None:
    non_collision = _observation(
        [
            {"type": "Number", "number": 4, "simulatorDiscriminator": "ignored"},
            {"type": "Number", "number": 5},
        ]
    )
    encoded_non_collision = build_public_rule_representation(non_collision)
    assert (
        encoded_non_collision.options[0].semantic["option"]["stable_simulator_discriminator"]
        is None
    )

    collision = _observation(
        [
            {"type": "Yes", "simulatorDiscriminator": "left_branch"},
            {"type": "Yes", "simulatorDiscriminator": "right_branch"},
        ]
    )
    encoded_collision = build_public_rule_representation(collision)
    values = [
        option.semantic["option"]["stable_simulator_discriminator"]
        for option in encoded_collision.options
    ]
    assert all(value is not None and value.startswith("sha256:") for value in values)
    assert values[0] != values[1]


def test_selection_bindings_state_and_target_only_terminal_payload() -> None:
    observation = _observation(
        [
            {
                "type": "Attach",
                "playerIndex": 0,
                "area": "Hand",
                "index": 0,
                "inPlayArea": "Bench",
                "inPlayIndex": 1,
                "count": 1,
                "energyIndex": 0,
            }
        ]
    )
    observation["current"]["result"] = 2
    observation["current"]["resultReason"] = "SimultaneousKnockout"
    encoded = build_public_rule_representation(observation)
    option = encoded.options[0].semantic["option"]
    assert option["source"]["owner"] == "acting"
    assert option["source"]["area"] == "hand"
    assert option["target"]["area"] == "bench"
    assert option["target"]["slot"] == 1
    assert option["count"] == 1
    active = encoded.state["players"]["acting"]["active"][0]["card"]
    assert active["current_hp"] == 80
    assert active["max_hp"] == 120
    assert active["pre_evolution_card_ids"] == [9]
    assert active["appear_this_turn"] is False
    assert active["typed_energy_units"] == [["energy_type:psychic", 1]]
    assert encoded.state["players"]["acting"]["effective_bench_maximum"] == 5

    terminal = extract_public_terminal_target(observation)
    assert terminal["target_only"] is True
    assert terminal["result"] == 2
    assert terminal["reason"] == "terminal_reason:simultaneousknockout"


def test_metadata_mapping_requires_explicit_test_only_opt_in() -> None:
    catalog = {
        "cards": [
            {
                "cardId": 77,
                "cardType": 0,
                "hp": 120,
                "retreatCost": 1,
                "stage1": True,
                "evolvesFrom": "Abra",
                "ex": True,
                "megaEx": True,
                "tera": True,
                "aceSpec": False,
                "weakness": 2,
                "resistance": 3,
                "attacks": [44],
                "skills": [{"skillId": 1, "structuredMechanics": {"once": True}}],
            }
        ],
        "attacks": [
            {
                "attackId": 44,
                "energies": [1, 2],
                "damage": 90,
                "structuredMechanics": {"counterPlacement": 3},
            }
        ],
        "provenance": {"catalog": "fixture"},
    }
    with pytest.raises(PublicRuleAdapterError, match="raw metadata mapping is forbidden"):
        build_public_rule_representation(
            _observation([{"type": "Skill", "cardId": 77, "serial": 1001}]),
            metadata_catalog=catalog,
        )
    encoded = build_public_rule_representation(
        _observation([{"type": "Skill", "cardId": 77, "serial": 1001}]),
        metadata_catalog=catalog,
        allow_test_catalog=True,
    )
    row = next(item for item in encoded.metadata["cards"] if item["card_id"] == 77)
    assert row["ex"] is True
    assert row["mega_ex"] is True
    assert row["default_prize_yield"] == 3
    assert row["prize_class"] == "mega_ex"
    assert row["attacks"][0]["damage"] == 90
    assert "text" not in str(row)
    assert encoded.metadata["provenance"]["test_only"] is True
    assert encoded.metadata["provenance"]["eligible"] is False

    config = load_public_rule_adapter_config()
    assert config["runtime"]["enabled_default"] is False
    assert config["runtime"]["runtime_wired"] is False
    assert set(config["runtime"]["zero_gates"].values()) == {0.0}


def test_direct_semantic_key_has_no_ordinal_parameter() -> None:
    observation = _observation([{"type": "Number", "number": 4}])
    direct = semantic_option_key(observation, {"type": "Number", "number": 4})
    encoded = build_public_rule_representation(observation)
    assert direct == encoded.options[0].semantic


def test_catalog_validator_defaults_to_unavailable_without_exact_sealed_provenance() -> None:
    catalog = {
        "schema": "poke_bot.alakazam_public_catalog_r298/v1",
        "revision": 298,
        "status": "sealed_public_simulator_catalog",
        "source": {},
        "provenance": {},
        "authority": {},
        "cards": [],
        "attacks": [],
    }
    receipt = {
        "schema": "poke_bot.alakazam_public_catalog_r298_receipt/v1",
        "revision": 298,
    }
    status = validate_public_catalog_provenance(
        catalog,
        receipt,
        catalog_file_sha256="sha256:0",
        receipt_file_sha256="sha256:0",
        vectors_file_sha256="sha256:0",
    )
    assert status.eligible is False
    assert status.reason == "catalog_file_sha256_mismatch"


def test_final_sealed_catalog_round_trip_when_mounted() -> None:
    artifact_raw = os.environ.get("R298_PUBLIC_CATALOG_ARTIFACT_DIR")
    if not artifact_raw:
        pytest.skip("final r298 public catalog artifact is not mounted")
    artifact = Path(artifact_raw)
    catalog = load_sealed_public_catalog(artifact / "catalog.json", artifact / "receipt.json")
    assert is_public_catalog_eligible(catalog)
    assert len(catalog.cards) == 1267
    assert len(catalog.attacks) == 1556
    assert catalog.provenance.engine_cards_sha256 == DEFAULT_PUBLIC_CATALOG_PINS.engine_cards_sha256
    assert catalog.provenance.engine_attacks_sha256 == DEFAULT_PUBLIC_CATALOG_PINS.engine_attacks_sha256


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="optional residual tests require torch",
)
def test_optional_residual_and_logit_helper_are_exact_object_bypasses_when_off() -> None:
    import torch

    from poke_bot.alakazam_public_rule_adapter_r298 import (
        PublicRuleMetadataResidual,
        apply_zero_gated_logit_residual,
    )
    artifact_raw = os.environ.get("R298_PUBLIC_CATALOG_ARTIFACT_DIR")
    if not artifact_raw:
        pytest.skip("final r298 public catalog artifact is not mounted")
    artifact = Path(artifact_raw)
    residual_module = PublicRuleMetadataResidual(
        load_sealed_public_catalog(artifact / "catalog.json", artifact / "receipt.json"),
        d_model=3,
    )
    representation = build_public_rule_representation(
        _observation([{"type": "Number", "number": 4}])
    )
    base = torch.tensor([[1.0, -0.0, 3.0]], dtype=torch.float32)
    # A poison projection proves the off path does not compute a delta.
    with torch.no_grad():
        residual_module.card_projection.weight.fill_(float("nan"))
        residual_module.attack_projection.weight.fill_(float("nan"))
    # Passing a deliberately malformed representation confirms the off path
    # does not even inspect representation shape/option rows.
    unchanged = residual_module.augment_option_hidden(base, object())
    assert unchanged is base
    assert unchanged.numpy().tobytes() == base.numpy().tobytes()

    # A zero individual branch must not multiply a poisoned projection by
    # zero when another branch is active.  The active branch remains finite
    # and must validate its own output.
    representation = build_public_rule_representation(
        _observation([{"type": "Number", "number": 4}])
    )
    with torch.no_grad():
        residual_module.attack_projection.weight.zero_()
        residual_module.attack_gate.fill_(1.0)
    active_attack = residual_module.augment_option_hidden(base, representation)
    assert torch.isfinite(active_attack).all()
    with torch.no_grad():
        residual_module.attack_projection.weight.fill_(float("nan"))
    with pytest.raises(PublicRuleAdapterError, match="attack residual must be finite"):
        residual_module.augment_option_hidden(base, representation)

    logits = torch.tensor([0.0, -0.0, 1.0], dtype=torch.float32)
    poison = torch.tensor([float("nan"), float("inf"), -float("inf")])
    bypass = apply_zero_gated_logit_residual(logits, poison, gate=0.0)
    assert bypass is logits
    assert bypass.numpy().tobytes() == logits.numpy().tobytes()
    with pytest.raises(PublicRuleAdapterError, match="finite"):
        apply_zero_gated_logit_residual(logits, poison, gate=0.1)
