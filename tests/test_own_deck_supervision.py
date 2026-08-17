from __future__ import annotations

import copy

import pytest

from poke_bot.own_deck_supervision import (
    OWN_DECK_SUPERVISION_SCHEMA,
    TERMINAL_CONVERSION_CLASS_SLICE,
    TERMINAL_CONVERSION_CLASSES,
    TERMINAL_CONVERSION_OPPONENT_KNOCKOUT_INDEX,
    TERMINAL_CONVERSION_OUTPUT_DIM,
    TERMINAL_CONVERSION_OUTPUT_LAYOUT,
    TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX,
    VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM,
    VISIBLE_TUTOR_COMPLETION_OUTPUT_LAYOUT,
    VISIBLE_TUTOR_COMPLETION_TERMINAL_CLASS_SLICE,
    VISIBLE_TUTOR_SELECTION_AUXILIARY_OUTPUT_DIM,
    VISIBLE_TUTOR_SELECTION_TARGET_KIND,
    build_own_deck_supervision_targets,
    terminal_conversion_labels,
    terminal_conversion_target_mask,
    terminal_conversion_target_vector,
    visible_tutor_completion_labels,
    visible_tutor_completion_target_mask,
    visible_tutor_completion_target_vector,
)


def _card(card_id: int, serial: int) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": 0}


def _pokemon(card_id: int, serial: int, *, owner: int) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "playerIndex": owner,
        "hp": 100,
        "energyCards": [],
        "tools": [],
    }


def _player(
    *,
    hand: list[dict] | None = None,
    active: list[dict] | None = None,
    bench: list[dict] | None = None,
    discard: list[dict] | None = None,
    prizes: int = 6,
) -> dict:
    return {
        "hand": [] if hand is None else hand,
        "handCount": len(hand or []),
        "active": [] if active is None else active,
        "bench": [] if bench is None else bench,
        "discard": [] if discard is None else discard,
        "prize": [None] * prizes,
        "deckCount": 40,
    }


def _observation(
    *,
    actor: int,
    turn: int,
    result: int = -1,
    player0: dict | None = None,
    player1: dict | None = None,
    select: dict | None = None,
) -> dict:
    return {
        "current": {
            "yourIndex": actor,
            "turn": turn,
            "result": result,
            "players": [
                _player() if player0 is None else player0,
                _player() if player1 is None else player1,
            ],
        },
        "select": {"option": [], "minCount": 1, "maxCount": 1}
        if select is None
        else select,
    }


def _transition(observation: dict, *, next_actor: int) -> dict:
    """Collection-side next-actor proof; current.yourIndex is perspective only."""

    return {"next_actor_seat": next_actor, "observation": observation}


def _tutor_observation(*, target: dict, actor: int = 0, turn: int = 3) -> dict:
    return _observation(
        actor=actor,
        turn=turn,
        player0=_player(),
        player1=_player(active=[_pokemon(900, 90, owner=1)]),
        select={
            "context": "TUTOR",
            "deck": [target, _card(102, 1002)],
            "option": [
                {"type": 3, "area": 1, "index": 0, "playerIndex": actor},
                {"type": 3, "area": 1, "index": 1, "playerIndex": actor},
            ],
            "minCount": 1,
            "maxCount": 1,
        },
    )


@pytest.mark.unit
def test_terminal_conversion_is_immediate_selected_action_only() -> None:
    pre = _observation(
        actor=0,
        turn=7,
        player0=_player(prizes=1),
        player1=_player(active=[_pokemon(900, 90, owner=1)]),
    )
    post = _observation(
        actor=1,
        turn=7,
        result=0,
        player0=_player(prizes=0),
        player1=_player(discard=[_card(900, 90)]),
    )
    step = {"observation": pre, "action": [0], "transition_after": post}
    original = copy.deepcopy(step)

    target = terminal_conversion_labels(step)

    assert TERMINAL_CONVERSION_CLASSES[target["terminal_class"]["value"]] == "own_win"
    assert target["terminal_class"]["mask"] is True
    assert target["prize_closeout"] == {"value": 1.0, "mask": True}
    assert target["opponent_knockout"] == {"value": 1.0, "mask": True}
    assert target["target_only"] is True
    assert "transition_after" not in target
    assert step == original
    assert TERMINAL_CONVERSION_OUTPUT_DIM == 6
    assert len(TERMINAL_CONVERSION_OUTPUT_LAYOUT) == TERMINAL_CONVERSION_OUTPUT_DIM
    assert TERMINAL_CONVERSION_CLASS_SLICE == slice(0, 4)
    assert TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX == 4
    assert TERMINAL_CONVERSION_OPPONENT_KNOCKOUT_INDEX == 5
    assert terminal_conversion_target_vector(target) == (0.0, 1.0, 0.0, 0.0, 1.0, 1.0)
    assert terminal_conversion_target_mask(target) == (True, True, True, True, True, True)


@pytest.mark.unit
def test_terminal_conversion_masks_prefix_missing_transition_and_chance() -> None:
    pre = _observation(actor=0, turn=2)
    post = _observation(actor=1, turn=2, result=0)
    boundary_post = copy.deepcopy(post)
    boundary_post["chance_boundary"] = True
    prefix = {
        "observation": pre,
        "action": [0],
        "transition_after": post,
        "raw_stage_index": 0,
        "raw_stage_count": 2,
    }
    for step in (
        prefix,
        {"observation": pre, "action": [0]},
        {
            "observation": pre,
            "action": [0],
            "transition_after": post,
            "chance_boundary": True,
        },
        {
            "observation": pre,
            "action": [0],
            "transition_after": {
                "chance_boundary": True,
                "observation": post,
            },
        },
        {
            "observation": pre,
            "action": [0],
            "transition_after": {
                "next_actor_seat": 1,
                "observation": boundary_post,
            },
        },
    ):
        target = terminal_conversion_labels(step)
        assert target["terminal_class"]["mask"] is False
        assert target["prize_closeout"]["mask"] is False
        assert target["opponent_knockout"]["mask"] is False
        assert terminal_conversion_target_vector(target) == (0.0,) * 6
        assert terminal_conversion_target_mask(target) == (False,) * 6


@pytest.mark.unit
def test_visible_tutor_tracks_only_selected_visible_card_and_same_actor_followup() -> None:
    target_card = _card(101, 1001)
    tutor_pre = _tutor_observation(target=target_card)
    tutor_post = _observation(
        # Deliberately disagree with the wrapper: this is a replay perspective
        # field, while the outer collection wrapper attests next_actor_seat=0.
        actor=1,
        turn=3,
        player0=_player(hand=[target_card]),
        player1=_player(active=[_pokemon(900, 90, owner=1)]),
    )
    tutor_step = {
        "observation": tutor_pre,
        "action": [0],
        "transition_after": _transition(tutor_post, next_actor=0),
    }
    followup_pre = _observation(
        actor=0,
        turn=3,
        player0=_player(hand=[target_card]),
        player1=_player(active=[_pokemon(900, 90, owner=1)]),
        select={
            "option": [
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0}
            ],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    # The same-actor turn ends after this observed use; no opponent action is
    # included in the target window.
    followup_post = _observation(
        actor=1,
        turn=4,
        player0=_player(),
        player1=_player(active=[_pokemon(900, 90, owner=1)]),
    )
    labels = visible_tutor_completion_labels(
        [
            tutor_step,
            {
                "observation": followup_pre,
                "action": [0],
                "transition_after": _transition(followup_post, next_actor=1),
            },
        ],
        0,
    )

    assert labels["selected_card_ids"] == [101]
    assert labels["selected_card_serials"] == [1001]
    assert labels["selected_card_id"] == {"value": 101, "mask": True}
    assert labels["selected_from_visible_deck"] == {"value": 1.0, "mask": True}
    assert labels["selected_target_observed_after_action"] == {
        "value": 1.0,
        "mask": True,
    }
    assert labels["same_actor_followup"] == {"value": 1.0, "mask": True}
    assert (
        TERMINAL_CONVERSION_CLASSES[labels["same_actor_terminal_class"]["value"]]
        == "nonterminal"
    )
    assert labels["same_actor_terminal_class"]["mask"] is True
    assert VISIBLE_TUTOR_SELECTION_TARGET_KIND == "visible_deck_menu_candidate"
    assert VISIBLE_TUTOR_SELECTION_AUXILIARY_OUTPUT_DIM == 0
    assert VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM == 7
    assert (
        len(VISIBLE_TUTOR_COMPLETION_OUTPUT_LAYOUT)
        == VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM
    )
    assert VISIBLE_TUTOR_COMPLETION_TERMINAL_CLASS_SLICE == slice(3, 7)
    # The selected card ID (101) is policy/option supervision, not a fixed
    # auxiliary output.  This vector contains only completion and outcome.
    assert visible_tutor_completion_target_vector(labels) == (
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert visible_tutor_completion_target_mask(labels) == (True,) * 7


@pytest.mark.unit
def test_visible_tutor_accepts_public_transition_snapshot_for_post_selection_proof() -> None:
    target_card = _card(101, 1001)
    post_snapshot = {
        "schema": "poke_bot.public_transition_snapshot/v1",
        "version": 1,
        "actor_seat": 0,
        "next_actor_seat": 0,
        "turn": 3,
        "result": -1,
        "valid": True,
        "players": [
            {
                "hand_serials": [1001],
                "discard_serials": [],
                "active": [],
                "bench": [],
            },
            {
                "hand_serials": [],
                "discard_serials": [],
                "active": [],
                "bench": [],
            },
        ],
    }
    labels = visible_tutor_completion_labels(
        [
            {
                "observation": _tutor_observation(target=target_card),
                "action": [0],
                "transition_after": post_snapshot,
            }
        ],
        0,
    )

    assert labels["selected_target_observed_after_action"] == {
        "value": 1.0,
        "mask": True,
    }
    # There is no next observed same-actor decision, so completion remains
    # masked rather than inferred from the post-action hand alone.
    assert labels["same_actor_followup"]["mask"] is False


@pytest.mark.unit
def test_visible_tutor_masks_replay_perspective_without_next_actor_provenance() -> None:
    target_card = _card(101, 1001)
    # Kaggle's current.yourIndex can remain the observed perspective even when
    # it does not prove that this actor still owns the turn.
    unwrapped_post = _observation(
        actor=0,
        turn=3,
        player0=_player(hand=[target_card]),
    )
    labels = visible_tutor_completion_labels(
        [
            {
                "observation": _tutor_observation(target=target_card),
                "action": [0],
                "transition_after": unwrapped_post,
            }
        ],
        0,
    )

    assert labels["selected_target_observed_after_action"] == {
        "value": 1.0,
        "mask": True,
    }
    assert labels["same_actor_followup"]["mask"] is False
    assert labels["same_actor_terminal_class"]["mask"] is False
    assert labels["mask_reason"] == "missing_next_actor_provenance"

    status_proven_labels = visible_tutor_completion_labels(
        [
            {
                "observation": _tutor_observation(target=target_card),
                "action": [0],
                "transition_after": {
                    "active_status": {"validated": True, "actor_seat": 0},
                    "observation": unwrapped_post,
                },
            }
        ],
        0,
    )
    assert status_proven_labels["mask_reason"] == "truncated_same_actor_continuation"


@pytest.mark.unit
def test_terminal_knockout_is_stack_aware_and_requires_public_prize_evidence() -> None:
    base = _pokemon(900, 90, owner=1)
    evolved = _pokemon(901, 91, owner=1)
    evolved["preEvolution"] = _pokemon(900, 90, owner=1)
    pre = _observation(
        actor=0,
        turn=5,
        player0=_player(prizes=3),
        player1=_player(active=[base]),
    )
    # This deliberately awkward public rendering used to look like a root-card
    # disappearance into discard.  The recursive evolution stack proves it is
    # still on board and therefore not a KO.
    evolution_post = _observation(
        actor=1,
        turn=5,
        player0=_player(prizes=3),
        player1=_player(
            active=[evolved],
            discard=[_card(900, 90)],
        ),
    )
    evolution_target = terminal_conversion_labels(
        {
            "observation": pre,
            "action": [0],
            "transition_after": evolution_post,
        }
    )
    assert evolution_target["opponent_knockout"] == {"value": 0.0, "mask": True}

    # A disappeared board stack in discard is still not a factual KO unless a
    # public prize decrease (or future explicit public KO/log proof) attests it.
    unproven_post = _observation(
        actor=1,
        turn=5,
        player0=_player(prizes=3),
        player1=_player(discard=[_card(900, 90)]),
    )
    unproven_target = terminal_conversion_labels(
        {
            "observation": pre,
            "action": [0],
            "transition_after": unproven_post,
        }
    )
    assert unproven_target["opponent_knockout"]["mask"] is False


@pytest.mark.unit
def test_post_tutor_scan_recurses_tools_energy_and_pre_evolution() -> None:
    tool = _card(101, 1001)
    energy = _card(102, 1002)
    pre_evolution = _card(103, 1003)
    host = _pokemon(700, 7000, owner=0)
    host["tools"] = [tool]
    host["energyCards"] = [energy]
    host["preEvolution"] = pre_evolution
    tutor_pre = _observation(
        actor=0,
        turn=3,
        player0=_player(),
        player1=_player(),
        select={
            "deck": [tool, energy, pre_evolution],
            "option": [
                {"type": 3, "area": 1, "index": 0, "playerIndex": 0},
                {"type": 3, "area": 1, "index": 1, "playerIndex": 0},
                {"type": 3, "area": 1, "index": 2, "playerIndex": 0},
            ],
            "minCount": 1,
            "maxCount": 3,
        },
    )
    post = _observation(
        actor=0,
        turn=3,
        player0=_player(active=[host]),
    )
    labels = visible_tutor_completion_labels(
        [
            {
                "observation": tutor_pre,
                "action": [0, 1, 2],
                "transition_after": _transition(post, next_actor=0),
            }
        ],
        0,
    )

    assert labels["selected_target_observed_after_action"] == {
        "value": 1.0,
        "mask": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attachment_name", "attachment_index_name"),
    (("tools", "toolIndex"), ("energyCards", "energyIndex")),
)
def test_tutor_followup_resolves_selected_tool_or_energy_source(
    attachment_name: str,
    attachment_index_name: str,
) -> None:
    target_card = _card(101, 1001)
    host = _pokemon(700, 7000, owner=0)
    host[attachment_name] = [target_card]
    tutor_post = _observation(
        actor=0,
        turn=3,
        player0=_player(active=[host]),
    )
    followup_pre = _observation(
        actor=0,
        turn=3,
        player0=_player(active=[host]),
        select={
            "option": [
                {
                    "type": 7,
                    "area": 4,
                    "index": 0,
                    "playerIndex": 0,
                    attachment_index_name: 0,
                }
            ],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    followup_post = _observation(actor=1, turn=4, player0=_player())
    labels = visible_tutor_completion_labels(
        [
            {
                "observation": _tutor_observation(target=target_card),
                "action": [0],
                "transition_after": _transition(tutor_post, next_actor=0),
            },
            {
                "observation": followup_pre,
                "action": [0],
                "transition_after": _transition(followup_post, next_actor=1),
            },
        ],
        0,
    )

    assert labels["selected_target_observed_after_action"] == {
        "value": 1.0,
        "mask": True,
    }
    assert labels["same_actor_followup"] == {"value": 1.0, "mask": True}


@pytest.mark.unit
def test_visible_tutor_masks_same_actor_continuation_after_opponent_or_chance_boundary() -> None:
    target_card = _card(101, 1001)
    pre = _tutor_observation(target=target_card)
    opponent_next = _observation(
        actor=1,
        turn=3,
        player0=_player(hand=[target_card]),
        player1=_player(),
    )
    opponent_step = {
        "observation": pre,
        "action": [0],
        "transition_after": _transition(opponent_next, next_actor=1),
    }
    opponent_labels = visible_tutor_completion_labels([opponent_step], 0)
    assert opponent_labels["selected_target_observed_after_action"]["mask"] is True
    assert opponent_labels["same_actor_followup"]["mask"] is False
    assert opponent_labels["same_actor_terminal_class"]["mask"] is False
    assert opponent_labels["mask_reason"] == "actor_boundary"

    chance_step = {
        "observation": pre,
        "action": [0],
        "transition_after": _transition(opponent_next, next_actor=1),
        "chance_boundary": True,
    }
    chance_labels = visible_tutor_completion_labels([chance_step], 0)
    assert chance_labels["selected_from_visible_deck"]["mask"] is False
    assert chance_labels["same_actor_followup"]["mask"] is False
    assert chance_labels["mask_reason"] == "non_immediate_or_boundary"


@pytest.mark.unit
def test_visible_tutor_rejects_boundary_on_transition_wrapper_or_post() -> None:
    target_card = _card(101, 1001)
    pre = _tutor_observation(target=target_card)
    post = _observation(
        actor=0,
        turn=3,
        player0=_player(hand=[target_card]),
    )
    boundary_post = copy.deepcopy(post)
    boundary_post["chance_boundary"] = True

    for transition in (
        {"chance_boundary": True, "observation": post},
        {"next_actor_seat": 0, "observation": boundary_post},
    ):
        labels = visible_tutor_completion_labels(
            [
                {
                    "observation": pre,
                    "action": [0],
                    "transition_after": transition,
                }
            ],
            0,
        )

        assert labels["selected_card_ids"] == []
        assert labels["selected_from_visible_deck"]["mask"] is False
        assert labels["selected_target_observed_after_action"]["mask"] is False
        assert labels["same_actor_followup"]["mask"] is False
        assert labels["same_actor_terminal_class"]["mask"] is False
        assert labels["mask_reason"] == "non_immediate_or_boundary"


@pytest.mark.unit
def test_full_builder_is_target_only_and_preserves_alignment() -> None:
    target_card = _card(101, 1001)
    pre = _tutor_observation(target=target_card)
    post = _observation(actor=1, turn=3, result=0, player0=_player(prizes=0))
    steps = [
        {
            "observation": pre,
            "action": [0],
            "transition_after": post,
        },
        {"observation": _observation(actor=0, turn=4), "action": [0]},
    ]
    original = copy.deepcopy(steps)

    targets = build_own_deck_supervision_targets(steps)

    assert len(targets) == len(steps)
    assert all(row["schema"] == OWN_DECK_SUPERVISION_SCHEMA for row in targets)
    assert targets[0]["terminal_conversion"]["target_only"] is True
    assert targets[1]["terminal_conversion"]["terminal_class"]["mask"] is False
    assert steps == original
