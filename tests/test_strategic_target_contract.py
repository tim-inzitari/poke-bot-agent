from __future__ import annotations

import copy
import math

import pytest

from poke_bot.strategic_heads import (
    ACTION_UTILITY_NAMES,
    EXPANDED_STRATEGIC_KEY,
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    EXPANDED_STRATEGIC_SCHEMA_VERSION,
    GAME_PHASE_NAMES,
    OPPONENT_RESPONSE_NAMES,
    RESOURCE_FORECAST_NAMES,
    TARGET_SCHEMA_DIGEST,
    StrategicTargetContractError,
    attach_expanded_strategic_labels,
    expanded_strategic_decision_head_mask,
    expanded_strategic_sequence_coverage,
    masked_expanded_strategic_coverage,
    merge_expanded_strategic_coverages,
    public_transition_snapshot,
    validate_expanded_strategic_labels,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


def _card(serial: int, card_id: int | None = None) -> dict:
    return {
        "id": int(card_id if card_id is not None else serial),
        "serial": int(serial),
        "playerIndex": 0,
    }


def _pokemon(
    serial: int,
    hp: int,
    *,
    owner: int,
    energy: tuple[int, ...] = (),
) -> dict:
    return {
        "id": 1000 + serial,
        "serial": serial,
        "playerIndex": owner,
        "hp": hp,
        "energyCards": [
            {
                "id": 2000 + energy_serial,
                "serial": energy_serial,
                "playerIndex": owner,
            }
            for energy_serial in energy
        ],
        "tools": [],
    }


def _player(
    seat: int,
    *,
    active: tuple[int, int] | None,
    bench: tuple[tuple[int, int], ...] = (),
    hand: tuple[int, ...] | None = (),
    hand_count: int | None = None,
    deck_count: int = 40,
    prizes: int = 6,
    discard: tuple[int, ...] = (),
    active_energy: tuple[int, ...] = (),
) -> dict:
    hand_rows = (
        None
        if hand is None
        else [
            {
                **_card(serial),
                "playerIndex": seat,
            }
            for serial in hand
        ]
    )
    active_rows = (
        []
        if active is None
        else [
            _pokemon(
                active[0],
                active[1],
                owner=seat,
                energy=active_energy,
            )
        ]
    )
    return {
        "active": active_rows,
        "bench": [
            _pokemon(serial, hp, owner=seat) for serial, hp in bench
        ],
        "benchMax": 5,
        "hand": hand_rows,
        "handCount": (
            int(hand_count)
            if hand_count is not None
            else len(hand or ())
        ),
        "deckCount": deck_count,
        "prize": [None] * prizes,
        "discard": [
            {
                **_card(serial),
                "playerIndex": seat,
            }
            for serial in discard
        ],
    }


def _options() -> list[dict]:
    return [
        {
            "type": 8,
            "playerIndex": 0,
            "area": 2,
            "index": 0,
            "inPlayArea": 5,
            "inPlayIndex": 0,
        },
        {
            "type": 8,
            "playerIndex": 0,
            "area": 2,
            "index": 1,
            "inPlayArea": 5,
            "inPlayIndex": 1,
        },
    ]


def _obs(
    *,
    turn: int,
    player0: dict,
    player1: dict,
    result: int = -1,
    actor: int = 0,
) -> dict:
    return {
        "current": {
            "yourIndex": actor,
            "turn": turn,
            "result": result,
            "energyAttached": False,
            "retreated": False,
            "players": [player0, player1],
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": _options(),
        },
    }


def _trajectory() -> list[dict]:
    pre0 = _obs(
        turn=2,
        player0=_player(0, active=(10, 100), hand=(1,), prizes=6),
        player1=_player(
            1,
            active=(20, 100),
            hand=None,
            hand_count=5,
            prizes=6,
        ),
    )
    post0 = _obs(
        turn=2,
        player0=_player(
            0,
            active=(10, 100),
            bench=((11, 70),),
            hand=(1, 2),
            prizes=5,
            active_energy=(101,),
        ),
        player1=_player(
            1,
            active=(20, 70),
            hand=None,
            hand_count=5,
            prizes=6,
        ),
        actor=1,
    )
    pre1 = _obs(
        turn=3,
        player0=_player(
            0,
            active=(10, 80),
            bench=((11, 70),),
            hand=(2,),
            prizes=5,
            active_energy=(101,),
        ),
        player1=_player(
            1,
            active=(21, 90),
            hand=None,
            hand_count=5,
            prizes=5,
            active_energy=(201,),
        ),
    )
    post1 = _obs(
        turn=3,
        player0=_player(
            0,
            active=(10, 80),
            bench=((11, 70),),
            hand=(2,),
            prizes=4,
            active_energy=(101,),
        ),
        player1=_player(
            1,
            active=(22, 100),
            hand=None,
            hand_count=5,
            prizes=5,
            discard=(21,),
            active_energy=(201,),
        ),
        actor=1,
    )
    pre2 = _obs(
        turn=5,
        player0=_player(
            0,
            active=(11, 70),
            hand=(2,),
            prizes=4,
            discard=(10,),
        ),
        player1=_player(
            1,
            active=(22, 100),
            hand=None,
            hand_count=5,
            prizes=4,
            discard=(21,),
            active_energy=(201,),
        ),
    )
    post2 = _obs(
        turn=5,
        player0=_player(
            0,
            active=(11, 70),
            hand=(2,),
            prizes=3,
        ),
        player1=_player(
            1,
            active=(22, 60),
            hand=None,
            hand_count=5,
            prizes=4,
            discard=(21,),
            active_energy=(201,),
        ),
        actor=1,
    )
    pre3 = _obs(
        turn=7,
        player0=_player(0, active=(11, 50), hand=(2,), prizes=3),
        player1=_player(
            1,
            active=(22, 60),
            hand=None,
            hand_count=5,
            prizes=3,
            discard=(21,),
            active_energy=(201,),
        ),
    )
    post3 = _obs(
        turn=8,
        player0=_player(0, active=(11, 50), hand=(2,), prizes=0),
        player1=_player(
            1,
            active=None,
            hand=None,
            hand_count=5,
            prizes=3,
            discard=(21, 22),
        ),
        result=0,
        actor=0,
    )
    return [
        {
            "observation": pre,
            "action": [0],
            "env_step": index,
            "aux_labels": {},
            "transition_after": public_transition_snapshot(
                post,
                actor_seat=0,
            ),
        }
        for index, (pre, post) in enumerate(
            (
                (pre0, post0),
                (pre1, post1),
                (pre2, post2),
                (pre3, post3),
            )
        )
    ]


def test_complete_trajectory_builds_strict_targets_and_consumes_post_state() -> None:
    steps = _trajectory()
    original_observations = copy.deepcopy(
        [step["observation"] for step in steps]
    )

    contract = attach_expanded_strategic_labels(
        steps,
        game_value=1.0,
    )

    assert contract["schema"] == EXPANDED_STRATEGIC_SCHEMA
    assert contract["version"] == EXPANDED_STRATEGIC_SCHEMA_VERSION == 2
    assert contract["digest"] == EXPANDED_STRATEGIC_SCHEMA_DIGEST
    assert contract["coverage"]["decisions"] == 4
    assert all("transition_after" not in step for step in steps)
    assert [step["observation"] for step in steps] == original_observations

    first = steps[0]["aux_labels"][EXPANDED_STRATEGIC_KEY]
    validate_expanded_strategic_labels(first)
    assert first["outcome_class"] == 2
    assert first["game_phase"] == GAME_PHASE_NAMES.index("setup")
    assert first["remaining_turns_log1p"] == pytest.approx(math.log1p(6))
    assert first["action_factors"] == [
        {"action_type": False, "target": True, "resource": True}
    ]

    utility = dict(zip(ACTION_UTILITY_NAMES, first["action_utility"]["values"]))
    utility_mask = dict(
        zip(ACTION_UTILITY_NAMES, first["action_utility"]["mask"])
    )
    assert utility_mask == {name: True for name in ACTION_UTILITY_NAMES}
    assert utility == {
        "damage_dealt": 30.0,
        "cards_drawn": 1.0,
        "energy_delta": 1.0,
        "open_bench_delta": -1.0,
        "prize_delta": 1.0,
        "knockout": 0.0,
    }

    response = dict(
        zip(OPPONENT_RESPONSE_NAMES, first["opponent_response"]["values"])
    )
    response_mask = dict(
        zip(OPPONENT_RESPONSE_NAMES, first["opponent_response"]["mask"])
    )
    assert response_mask["attack"] is False
    assert response_mask["end_without_attack"] is False
    assert response["prize"] == 1.0
    assert response["active_change"] == 1.0
    assert response["hand_reduction"] == 1.0
    assert response["board_energy_increase"] == 1.0

    resources = dict(
        zip(RESOURCE_FORECAST_NAMES, first["resource_forecast"]["values"])
    )
    assert resources["hand_size"] == 1.0
    assert resources["deck_size"] == 40.0
    assert resources["attached_energy"] == 1.0
    assert resources["open_bench_slots"] == 4.0
    assert resources["energy_attachment_available"] == 1.0
    assert resources["retreat_available"] == 0.0

    # All three horizons on the first row were constructed from the complete
    # trajectory. The third horizon sees the later KO and both prize deltas.
    horizon3 = first["tactical_outcomes"]["values"][2]
    horizon3_mask = first["tactical_outcomes"]["mask"][2]
    assert horizon3_mask[0] and horizon3_mask[1] and horizon3_mask[2]
    assert horizon3[0] == 1.0
    assert horizon3[1] == 1.0
    assert horizon3[2] == 1.0


def test_action_utility_masks_without_explicit_post_action_snapshot() -> None:
    steps = _trajectory()
    steps[0].pop("transition_after")

    attach_expanded_strategic_labels(steps, game_value=-1.0)

    first = steps[0]["aux_labels"][EXPANDED_STRATEGIC_KEY]
    assert first["action_utility"] == {
        "values": [0.0] * len(ACTION_UTILITY_NAMES),
        "mask": [False] * len(ACTION_UTILITY_NAMES),
    }
    assert (
        first["provenance"]["transition_after"] == "absent_masked"
    )
    # Later same-seat states exist, but they are never used as a contaminated
    # substitute for the missing immediate transition.
    assert any(first["tactical_outcomes"]["mask"][0][:2])


def test_game_phase_uses_current_public_state_and_existing_exact_lethal() -> None:
    ordinary = _trajectory()
    attach_expanded_strategic_labels(ordinary, game_value=1.0)
    assert ordinary[0]["aux_labels"][EXPANDED_STRATEGIC_KEY][
        "game_phase"
    ] == GAME_PHASE_NAMES.index("setup")

    lethal = _trajectory()
    lethal[0]["aux_labels"]["lethal_threat"] = 1.0
    attach_expanded_strategic_labels(lethal, game_value=1.0)
    assert lethal[0]["aux_labels"][EXPANDED_STRATEGIC_KEY][
        "game_phase"
    ] == GAME_PHASE_NAMES.index("closeout")


def test_idempotent_materialization_preserves_targets_and_removes_raw_transition() -> None:
    steps = _trajectory()
    first_contract = attach_expanded_strategic_labels(
        steps,
        game_value=0.0,
    )
    before = copy.deepcopy(
        [step["aux_labels"][EXPANDED_STRATEGIC_KEY] for step in steps]
    )
    steps[0]["transition_after"] = {"not": "reused"}

    second_contract = attach_expanded_strategic_labels(
        steps,
        game_value=0.0,
    )

    assert second_contract == first_contract
    assert "transition_after" not in steps[0]
    assert [
        step["aux_labels"][EXPANDED_STRATEGIC_KEY] for step in steps
    ] == before


def test_strict_validator_rejects_digest_shape_mask_and_partial_trajectory() -> None:
    steps = _trajectory()
    attach_expanded_strategic_labels(steps, game_value=1.0)
    target = copy.deepcopy(steps[0]["aux_labels"][EXPANDED_STRATEGIC_KEY])

    target["digest"] = "sha256:" + "0" * 64
    with pytest.raises(StrategicTargetContractError, match="digest"):
        validate_expanded_strategic_labels(target)

    target = copy.deepcopy(steps[0]["aux_labels"][EXPANDED_STRATEGIC_KEY])
    target["action_utility"]["mask"][0] = "yes"
    with pytest.raises(StrategicTargetContractError, match="must be bool"):
        validate_expanded_strategic_labels(target)

    target = copy.deepcopy(steps[0]["aux_labels"][EXPANDED_STRATEGIC_KEY])
    target["action_utility"]["mask"][0] = False
    target["action_utility"]["values"][0] = 1.0
    with pytest.raises(StrategicTargetContractError, match="zero when masked"):
        validate_expanded_strategic_labels(target)

    del steps[-1]["aux_labels"][EXPANDED_STRATEGIC_KEY]
    with pytest.raises(StrategicTargetContractError, match="partially"):
        attach_expanded_strategic_labels(steps, game_value=1.0)


def test_public_transition_snapshot_excludes_decks_and_opponent_hand_cards() -> None:
    observation = _trajectory()[0]["observation"]
    observation = copy.deepcopy(observation)
    observation["current"]["players"][0]["deck"] = [_card(900)]
    observation["current"]["players"][1]["hand"] = [_card(901)]
    observation["current"]["players"][1]["deck"] = [_card(902)]

    snapshot = public_transition_snapshot(observation, actor_seat=0)

    encoded = repr(snapshot)
    assert "deck" in encoded  # count field is retained
    assert 900 not in snapshot["players"][0].values()
    assert snapshot["players"][1]["hand_serials"] is None
    assert "deck" not in snapshot["players"][1]


def test_canonical_head_coverage_masks_missing_and_right_censored_rows() -> None:
    complete = _trajectory()
    attach_expanded_strategic_labels(complete, game_value=1.0)
    complete_target = complete[0]["aux_labels"][EXPANDED_STRATEGIC_KEY]
    complete_mask = expanded_strategic_decision_head_mask(complete_target)

    assert TARGET_SCHEMA_DIGEST == EXPANDED_STRATEGIC_SCHEMA_DIGEST
    assert tuple(complete_mask) == EXPANDED_HEAD_IDS
    assert complete_mask["action_q"] is True
    assert complete_mask["outcome_distribution"] is True
    assert complete_mask["remaining_turns"] is True

    censored = _trajectory()
    attach_expanded_strategic_labels(
        censored,
        game_value=0.0,
        terminal_complete=False,
    )
    censored_target = censored[0]["aux_labels"][EXPANDED_STRATEGIC_KEY]
    censored_mask = expanded_strategic_decision_head_mask(censored_target)
    assert censored_mask["action_q"] is False
    assert censored_mask["outcome_distribution"] is False
    assert censored_mask["remaining_turns"] is False
    assert censored_mask["game_phase"] is True

    coverage = expanded_strategic_sequence_coverage(
        (complete[0], censored[0], None)
    )
    assert coverage["schema"] == EXPANDED_STRATEGIC_SCHEMA
    assert coverage["digest"] == TARGET_SCHEMA_DIGEST
    assert coverage["decisions"] == 3
    assert coverage["head_coverage"]["action_q"] == {
        "labeled_rows": 1,
        "masked_rows": 2,
        "total_rows": 3,
    }
    assert coverage["head_coverage"]["game_phase"] == {
        "labeled_rows": 2,
        "masked_rows": 1,
        "total_rows": 3,
    }


def test_coverage_merge_is_strict_and_missing_rows_are_masked() -> None:
    left = masked_expanded_strategic_coverage(2)
    right = masked_expanded_strategic_coverage(3)
    merged = merge_expanded_strategic_coverages((left, right))
    assert merged["decisions"] == 5
    assert set(merged["head_coverage"]) == set(EXPANDED_HEAD_IDS)
    assert all(
        row == {
            "labeled_rows": 0,
            "masked_rows": 5,
            "total_rows": 5,
        }
        for row in merged["head_coverage"].values()
    )

    malformed = copy.deepcopy(left)
    malformed["head_coverage"]["action_q"]["masked_rows"] = 1
    with pytest.raises(StrategicTargetContractError, match="inconsistent"):
        merge_expanded_strategic_coverages((malformed,))
