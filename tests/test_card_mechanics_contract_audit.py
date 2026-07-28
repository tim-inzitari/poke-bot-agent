from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot import cg_env, features
from scripts.audit_card_mechanics_contract import (
    compare_source_trees,
    known_findings,
    representation_matrix,
)


class _AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class _OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class _SpecialConditionType(IntEnum):
    POISON = 0
    BURN = 1
    SLEEP = 2
    PARALYZE = 3
    CONFUSE = 4


class _SelectContext(IntEnum):
    MAIN = 0
    DRAW_COUNT = 38
    SKILL_ORDER = 34
    RECOVER_SPECIAL_CONDITION = 48


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, "_CARD_COUNT", 8)
    monkeypatch.setattr(features, "_ATTACK_COUNT", 8)
    monkeypatch.setattr(features, "_CARD_TABLE", {})
    monkeypatch.setitem(cg_env.__dict__, "AreaType", _AreaType)
    monkeypatch.setitem(cg_env.__dict__, "OptionType", _OptionType)
    monkeypatch.setitem(
        cg_env.__dict__,
        "SpecialConditionType",
        _SpecialConditionType,
    )
    monkeypatch.setitem(cg_env.__dict__, "SelectContext", _SelectContext)


def _card(card_id: int = 3, **extra):
    fields = {
        "id": card_id,
        "hp": 80,
        "maxHp": 100,
        "tools": [],
        "energyCards": [],
        "energies": [],
        "preEvolution": [],
        "appearThisTurn": False,
    }
    fields.update(extra)
    return SimpleNamespace(**fields)


def _player(active, bench=None, **extra):
    fields = {
        "deckCount": 40,
        "discard": [],
        "handCount": 5,
        "hand": [_card(2)],
        "bench": bench or [],
        "active": [active],
        "prize": [None] * 6,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
        "benchMax": 5,
    }
    fields.update(extra)
    return SimpleNamespace(**fields)


def _obs(options, *, context=_SelectContext.MAIN, state_extra=None):
    state_fields = {
        "yourIndex": 0,
        "players": [_player(_card(3)), _player(_card(4))],
        "stadium": [],
        "looking": [],
        "turn": 3,
        "firstPlayer": 0,
        "turnActionCount": 0,
        "supporterPlayed": False,
        "stadiumPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": -1,
    }
    state_fields.update(state_extra or {})
    return SimpleNamespace(
        current=SimpleNamespace(**state_fields),
        select=SimpleNamespace(context=context, option=options, deck=[]),
    )


def _option(option_type, **extra):
    fields = {
        "type": option_type,
        "area": None,
        "index": None,
        "playerIndex": None,
        "inPlayArea": None,
        "inPlayIndex": None,
        "toolIndex": None,
        "energyIndex": None,
        "cardId": None,
    }
    fields.update(extra)
    return SimpleNamespace(**fields)


def _signature(vector: features.SparseVector, word: int):
    start = vector.offset[word]
    end = vector.offset[word + 1] if word + 1 < vector.num_words else len(vector.index)
    values = {}
    for index, value in zip(vector.index[start:end], vector.value[start:end]):
        values[index] = values.get(index, 0.0) + value
    return tuple(sorted(values.items()))


def _sparse_signature(vector: features.SparseVector):
    return tuple(vector.index), tuple(vector.value), tuple(vector.offset), vector.pos


def test_number_four_and_five_are_confirmed_feature_aliases(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    options = [
        _option(_OptionType.NUMBER, number=4),
        _option(_OptionType.NUMBER, number=5),
    ]
    encoded = features.build_option_tokens(
        _obs(options, context=_SelectContext.DRAW_COUNT), [[0], [1]]
    )
    assert _signature(encoded, 0) == _signature(encoded, 1)


def test_skill_serial_and_duplicate_attack_ordinal_are_aliases(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    skills = [
        _option(_OptionType.SKILL, cardId=3, serial=69),
        _option(_OptionType.SKILL, cardId=3, serial=71),
    ]
    encoded_skills = features.build_option_tokens(
        _obs(skills, context=_SelectContext.SKILL_ORDER), [[0], [1]]
    )
    assert _signature(encoded_skills, 0) == _signature(encoded_skills, 1)

    attacks = [
        _option(_OptionType.ATTACK, attackId=2),
        _option(_OptionType.ATTACK, attackId=2),
    ]
    encoded_attacks = features.build_option_tokens(_obs(attacks), [[0], [1]])
    assert _signature(encoded_attacks, 0) == _signature(encoded_attacks, 1)


def test_entity_oov_offsets_fail_closed_before_role_aliases(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    cc = features.card_vocab_size()
    ac = features.attack_vocab_size()

    invalid_card = features.SparseVector()
    invalid_card.word_start()
    with pytest.raises(features.FeatureContractError, match="option card id"):
        features._decoder_card_id(invalid_card, 0, cc + 1, cc)

    attack = _option(_OptionType.ATTACK, attackId=ac + 1)
    with pytest.raises(features.FeatureContractError, match="attackId"):
        features.build_option_tokens(_obs([attack]), [[0]])

    invalid_board_card = features.SparseVector()
    invalid_board_card.word_start()
    with pytest.raises(features.FeatureContractError, match="board card id"):
        features._add_card(invalid_board_card, _card(cc + 1), cc)

    with pytest.raises(features.FeatureContractError, match="negative"):
        invalid_board_card.add(-1, 1.0)


def test_all_official_option_types_emit_or_fail_explicitly(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    active = _card(3, tools=[_card(5)], energyCards=[_card(6)])
    options = [
        _option(_OptionType.NUMBER, number=2),
        _option(_OptionType.YES),
        _option(_OptionType.NO),
        _option(
            _OptionType.CARD,
            area=_AreaType.HAND,
            index=0,
            playerIndex=0,
        ),
        _option(
            _OptionType.TOOL_CARD,
            area=_AreaType.ACTIVE,
            index=0,
            playerIndex=0,
            toolIndex=0,
        ),
        _option(
            _OptionType.ENERGY_CARD,
            area=_AreaType.ACTIVE,
            index=0,
            playerIndex=0,
            energyIndex=0,
        ),
        _option(
            _OptionType.ENERGY,
            area=_AreaType.ACTIVE,
            index=0,
            playerIndex=0,
            energyIndex=0,
            count=2,
        ),
        _option(_OptionType.PLAY, index=0),
        _option(
            _OptionType.ATTACH,
            area=_AreaType.HAND,
            index=0,
            inPlayArea=_AreaType.ACTIVE,
            inPlayIndex=0,
        ),
        _option(
            _OptionType.EVOLVE,
            area=_AreaType.HAND,
            index=0,
            inPlayArea=_AreaType.ACTIVE,
            inPlayIndex=0,
        ),
        _option(_OptionType.ABILITY, area=_AreaType.ACTIVE, index=0),
        _option(_OptionType.DISCARD, area=_AreaType.HAND, index=0),
        _option(_OptionType.RETREAT),
        _option(_OptionType.ATTACK, attackId=2),
        _option(_OptionType.END),
        # Official API uses cardId=0 for special-condition SKILL options.
        _option(_OptionType.SKILL, cardId=0, serial=69),
        _option(_OptionType.SPECIAL_CONDITION, specialConditionType=2),
    ]
    obs = _obs(
        options,
        state_extra={"players": [_player(active), _player(_card(4))]},
    )
    encoded = features.build_option_tokens(
        obs,
        [[index] for index in range(len(options))],
    )
    assert encoded.num_words == len(_OptionType)
    assert all(_signature(encoded, index) for index in range(encoded.num_words))

    with pytest.raises(features.FeatureContractError, match="unsupported option type"):
        features.build_option_tokens(_obs([_option(99)]), [[0]])


def test_official_json_enum_names_match_integer_feature_rows(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    integer = _option(
        _OptionType.CARD,
        area=_AreaType.HAND,
        index=0,
        playerIndex=0,
    )
    serialized = _option(
        "Card",
        area="Hand",
        index=0,
        playerIndex=0,
    )
    integer_encoded = features.build_option_tokens(
        _obs([integer], context=_SelectContext.DRAW_COUNT),
        [[0]],
    )
    serialized_encoded = features.build_option_tokens(
        _obs([serialized], context="DrawCount"),
        [[0]],
    )
    assert _sparse_signature(integer_encoded) == _sparse_signature(
        serialized_encoded
    )

    integer_condition = _option(
        _OptionType.SPECIAL_CONDITION,
        specialConditionType=_SpecialConditionType.POISON,
    )
    serialized_condition = _option(
        "SpecialCondition",
        specialConditionType="Poison",
    )
    assert _sparse_signature(
        features.build_option_tokens(_obs([integer_condition]), [[0]])
    ) == _sparse_signature(
        features.build_option_tokens(_obs([serialized_condition]), [[0]])
    )


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (_option(_OptionType.NUMBER, number=-1), "negative option number"),
        (
            _option(_OptionType.SPECIAL_CONDITION, specialConditionType=5),
            "specialConditionType",
        ),
        (_option(_OptionType.ATTACK, attackId=0), "attackId"),
    ],
)
def test_malformed_typed_options_fail_closed(monkeypatch, option, message) -> None:
    _install_fake_runtime(monkeypatch)
    with pytest.raises(features.FeatureContractError, match=message):
        features.build_option_tokens(_obs([option]), [[0]])


def test_invalid_context_and_action_indices_fail_closed(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    end = _option(_OptionType.END)
    with pytest.raises(features.FeatureContractError, match="select context"):
        features.build_option_tokens(_obs([end], context=49), [[0]])
    with pytest.raises(features.FeatureContractError, match="outside"):
        features.build_option_tokens(_obs([end]), [[-1]])
    with pytest.raises(features.FeatureContractError, match="repeats"):
        features.build_option_tokens(_obs([end]), [[0, 0]])


def test_omitted_state_fields_are_metamorphically_invisible(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    base_active = _card(3)
    changed_active = _card(
        3,
        maxHp=330,
        appearThisTurn=True,
        energies=[1, 2, 3],
        preEvolution=[_card(1), _card(2)],
    )
    base = _obs([], state_extra={"players": [_player(base_active), _player(_card(4))]})
    changed = _obs(
        [],
        state_extra={
            "players": [
                _player(changed_active, benchMax=8),
                _player(_card(4), benchMax=8),
            ],
            "turnActionCount": 19,
            "supporterPlayed": True,
            "stadiumPlayed": True,
            "energyAttached": True,
            "retreated": True,
            "result": 1,
        },
    )
    first = features.build_board_tokens(base, [1] * 60)
    second = features.build_board_tokens(changed, [1] * 60)
    assert _sparse_signature(first) == _sparse_signature(second)


def test_every_legal_composite_binding_tuple_has_a_unique_row(monkeypatch) -> None:
    _install_fake_runtime(monkeypatch)
    rows = set()
    for role in range(features.DECODER_BINDING_ROLE_COUNT):
        for player_index in (0, 1, None):
            for area in range(features.DECODER_BINDING_AREA_COUNT):
                for index_code in range(features.DECODER_BINDING_INDEX_COUNT):
                    vector = features.SparseVector()
                    vector.word_start()
                    features._decoder_binding(
                        vector,
                        role,
                        player_index=player_index,
                        area=area,
                        index=None if index_code == 0 else index_code - 1,
                        your_index=0,
                    )
                    rows.add(vector.index[0])
    assert len(rows) == features.DECODER_BINDING_VOCAB_SIZE
    assert max(rows) < features.decoder_vocab_size()


def test_audit_matrix_keeps_simulator_feature_and_learnability_separate() -> None:
    matrix = {row["mechanic"]: row for row in representation_matrix()}
    assert matrix["special conditions"]["feature"] == "five explicit flags"
    assert matrix["card type / energy type"]["feature"] == "not explicit"
    assert "remainEnergyCost" in matrix["energy/damage selection residuals"][
        "feature"
    ]
    assert "binary multi-hot" in matrix["opponent hidden hand / remainder"][
        "feature"
    ]
    assert "right-censored" in matrix["lethal / prize-take threat"]["learnable"]
    assert matrix["NUMBER options"]["learnable"] == "not identifiable"
    assert "320" not in " ".join(row["mechanic"] for row in matrix.values())


def test_source_audit_surfaces_long_game_guard_divergence(tmp_path: Path) -> None:
    official = tmp_path / "official"
    rebuilt = tmp_path / "rebuilt"
    official.mkdir()
    rebuilt.mkdir()
    (official / "BattleData.h").write_text("if (actionCount >= 3000) draw();")
    (rebuilt / "BattleData.h").write_text("// guard removed")
    (rebuilt / "LICENSE").write_text("competition only")
    source = compare_source_trees(official, rebuilt)
    assert source["comparison"]["changed"] == ["BattleData.h"]
    assert source["rebuilt"]["competition_license_present"] is True
    assert source["rebuilt"]["action_count_3000_guard_present"] is False
    findings = known_findings({}, source)
    finding_ids = {finding["id"] for finding in findings}
    assert any(
        finding["id"] == "ENGINE-LONG-GAME-GUARD-DIVERGENCE" for finding in findings
    )
    assert "FEATURE-ENTITY-OOV-ALIAS" not in finding_ids
    assert "FEATURE-ENERGY-RESIDUAL-OMISSIONS" in finding_ids
    assert "AUX-LETHAL-RIGHT-CENSORING" in finding_ids
    assert "AUX-HIDDEN-MULTIPLICITY-COLLAPSE" in finding_ids
