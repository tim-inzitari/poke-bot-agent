from __future__ import annotations

from dataclasses import replace
import time

import pytest

from poke_bot.tactical_sequence_planner import (
    ExactTerminalWinGoal,
    PublicFactGoal,
    RankedAction,
    TacticalSearchConfig,
    TacticalSearchState,
    TacticalSequenceError,
    TacticalSequencePlanner,
    TacticalTransition,
    VisibleTutorTargetGoal,
)
from poke_bot.alakazam_tactical_sequence import (
    compile_alakazam_public_tactical_facts,
    rank_alakazam_sme_candidates,
)
from poke_bot.tactical_sequence_process_backend import OwnedProcessTacticalBackend


def _state(
    name: str,
    *,
    actor: int = 0,
    turn: int = 7,
    actions=((0,),),
    ordered_count: int | None = None,
    winner: int | None = None,
    chance: bool = False,
    information: bool = False,
    facts=None,
    visible_tutor_cards=(),
    previous_action=None,
):
    return TacticalSearchState(
        observation_fingerprint=f"obs-{name}",
        semantic_fingerprint=f"semantic-{name}",
        actor=actor,
        turn_id=turn,
        legal_actions=tuple(actions),
        ordered_action_count=(len(actions) if ordered_count is None else ordered_count),
        terminal_winner=winner,
        explicit_chance_boundary=chance,
        information_boundary=information,
        public_facts={} if facts is None else facts,
        visible_tutor_cards=tuple(visible_tutor_cards),
        previous_action_token=previous_action,
    )


class _GraphBackend:
    isolation_mode = "deterministic_test_fixture"

    def __init__(self, edges):
        self.edges = edges
        self.calls = []

    def advance(self, state, action, *, deadline_monotonic):
        self.calls.append((state.semantic_fingerprint, action, deadline_monotonic))
        next_state = self.edges[(state.semantic_fingerprint, action)]
        return TacticalTransition(
            next_state=replace(next_state, previous_action_token=action),
            action_token=action,
        )


class _OwnedFixtureWorker:
    def advance(self, state, action):
        next_state = _state("owned-win", actions=(), winner=0, previous_action=action)
        return TacticalTransition(next_state=next_state, action_token=action)

    def close(self):
        return None


class _HangingOwnedFixtureWorker:
    def advance(self, state, action):
        time.sleep(30.0)
        raise AssertionError("unreachable")

    def close(self):
        return None


def _owned_fixture_factory():
    return _OwnedFixtureWorker()


def _hanging_owned_fixture_factory():
    return _HangingOwnedFixtureWorker()


def _ranker(order):
    def rank(state):
        rows = order[state.semantic_fingerprint]
        return [
            RankedAction(action=action, probability=probability, sme_priority=sme)
            for action, probability, sme in rows
        ]

    return rank


def _config(**overrides):
    return TacticalSearchConfig(
        allow_deterministic_test_fixture=True,
        **overrides,
    )


def test_one_policy_discrepancy_finds_exact_terminal_win_but_stays_shadow_only():
    root = _state("root", actions=((0,), (1,)))
    dead = _state("dead", actions=(), winner=1)
    win = _state("win", actions=(), winner=0)
    backend = _GraphBackend(
        {
            (root.semantic_fingerprint, (0,)): dead,
            (root.semantic_fingerprint, (1,)): win,
        }
    )
    planner = TacticalSequencePlanner(
        backend=backend,
        rank_actions=_ranker(
            {root.semantic_fingerprint: [((0,), 0.8, 0.0), ((1,), 0.2, 5.0)]}
        ),
        config=_config(max_discrepancies=1),
    )

    result = planner.search(
        root=root,
        direct_action=(0,),
        goal=ExactTerminalWinGoal(root_actor=0),
    )

    assert result.status == "proven_exact_terminal_win_shadow"
    assert result.proposed_action == (1,)
    assert result.dispatch_authorized is False
    assert result.receipt["action_changed"] is True
    assert result.receipt["proof"]["terminal_winner"] == 0
    assert result.receipt["tactical_outcome_head_is_proof"] is False


def test_public_sme_goal_is_shadow_only_even_when_reached():
    root = _state("root", actions=((0,),))
    ready = _state(
        "ready",
        actions=((0,),),
        facts={"attacker_ready": True, "replacement_line_live": True},
    )
    planner = TacticalSequencePlanner(
        backend=_GraphBackend({(root.semantic_fingerprint, (0,)): ready}),
        rank_actions=_ranker({root.semantic_fingerprint: [((0,), 1.0, 4.0)]}),
        config=_config(),
    )
    result = planner.search(
        root=root,
        direct_action=(0,),
        goal=PublicFactGoal(
            goal_id="sme_attacker_and_replacement",
            required_facts={"attacker_ready": True, "replacement_line_live": True},
        ),
    )
    assert result.status == "public_goal_reached_shadow"
    assert result.proposed_action == (0,)
    assert result.dispatch_authorized is False
    assert result.receipt["proof"] is None


@pytest.mark.parametrize(
    ("next_state", "reason"),
    [
        (_state("chance", chance=True), "explicit_chance_pre_random"),
        (_state("opponent", actor=1), "actor_change"),
        (_state("next-turn", turn=8), "turn_change"),
        (_state("hidden-tutor", information=True), "information_reobservation"),
        (
            _state("wide", actions=((0,),), ordered_count=65),
            "deterministic_internal_fanout_over_64",
        ),
    ],
)
def test_boundaries_never_satisfy_nonterminal_goal(next_state, reason):
    root = _state("root", actions=((0,),))
    next_state = replace(next_state, public_facts={"target": True})
    planner = TacticalSequencePlanner(
        backend=_GraphBackend({(root.semantic_fingerprint, (0,)): next_state}),
        rank_actions=_ranker({root.semantic_fingerprint: [((0,), 1.0, 0.0)]}),
        config=_config(),
    )
    result = planner.search(
        root=root,
        direct_action=(0,),
        goal=PublicFactGoal(goal_id="target", required_facts={"target": True}),
    )
    assert result.proposed_action is None
    assert result.receipt["boundary_counts"][reason] == 1


def test_visible_tutor_target_requires_real_observed_nonboundary_prompt():
    goal = VisibleTutorTargetGoal(target_card_ids=(743, 744))
    hidden = _state(
        "hidden",
        information=True,
        visible_tutor_cards=(743,),
    )
    visible = _state("visible", visible_tutor_cards=(743, 999))
    unobserved = replace(visible, public_facts_are_observed=False)
    assert goal.satisfied(hidden) is False
    assert goal.satisfied(unobserved) is False
    assert goal.satisfied(visible) is True


def test_missing_simulated_previous_action_token_fails_closed():
    root = _state("root", actions=((0,),))
    next_state = _state("next", actions=((0,),), previous_action=None)

    class BrokenHistoryBackend(_GraphBackend):
        def advance(self, state, action, *, deadline_monotonic):
            return TacticalTransition(next_state=next_state, action_token=action)

    planner = TacticalSequencePlanner(
        backend=BrokenHistoryBackend({}),
        rank_actions=_ranker({root.semantic_fingerprint: [((0,), 1.0, 0.0)]}),
        config=_config(),
    )
    result = planner.search(
        root=root,
        direct_action=(0,),
        goal=ExactTerminalWinGoal(root_actor=0),
    )
    assert result.status == "backend_fault"
    assert "previous-action history" in result.receipt["failure"]
    assert result.proposed_action is None


def test_in_process_native_backend_is_rejected():
    class UnsafeBackend:
        isolation_mode = "in_process_native"

    with pytest.raises(TacticalSequenceError, match="owned bounded child"):
        TacticalSequencePlanner(
            backend=UnsafeBackend(),
            rank_actions=lambda _state: (),
        )


def test_test_fixture_requires_explicit_opt_in():
    with pytest.raises(TacticalSequenceError, match="test-fixture opt-in"):
        TacticalSequencePlanner(
            backend=_GraphBackend({}),
            rank_actions=lambda _state: (),
        )


def test_tactical_head_hint_can_order_telemetry_but_cannot_create_proof():
    root = _state("root", actions=((0,),))
    nonterminal = _state("nonterminal", actions=())
    planner = TacticalSequencePlanner(
        backend=_GraphBackend({(root.semantic_fingerprint, (0,)): nonterminal}),
        rank_actions=lambda _state: [
            RankedAction((0,), 1.0, tactical_head_hint=999.0)
        ],
        config=_config(),
    )
    result = planner.search(
        root=root,
        direct_action=(0,),
        goal=ExactTerminalWinGoal(root_actor=0),
    )
    assert result.proposed_action is None
    assert result.receipt["proof"] is None
    assert result.receipt["tactical_outcome_head_is_proof"] is False


def test_backend_action_token_mismatch_fails_closed():
    root = _state("root", actions=((0,),))
    next_state = _state("next", actions=((0,),), previous_action=(1,))

    class WrongTokenBackend(_GraphBackend):
        def advance(self, state, action, *, deadline_monotonic):
            return TacticalTransition(next_state=next_state, action_token=(1,))

    planner = TacticalSequencePlanner(
        backend=WrongTokenBackend({}),
        rank_actions=_ranker({root.semantic_fingerprint: [((0,), 1.0, 0.0)]}),
        config=_config(),
    )
    result = planner.search(
        root=root,
        direct_action=(0,),
        goal=ExactTerminalWinGoal(root_actor=0),
    )
    assert result.status == "backend_fault"
    assert "action token mismatch" in result.receipt["failure"]


def test_revision_257_configuration_cannot_enable_dispatch():
    with pytest.raises(TacticalSequenceError, match="shadow-only"):
        TacticalSearchConfig(shadow_only=False)


def test_alakazam_public_fact_compiler_recognizes_powerful_hand_closeout():
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [{"id": 743}],
                    "bench": [{"id": 743}],
                    "hand": [{"id": index} for index in range(6)],
                    "deckCount": 12,
                    "prize": [{}],
                },
                {
                    "active": [{"id": 999, "hp": 120, "energyCards": []}],
                    "bench": [],
                    "handCount": 3,
                    "deckCount": 20,
                    "prize": [{}, {}],
                },
            ],
        },
        "select": {
            "option": [{"type": 13, "attackId": 1072}],
            "deck": [{"id": 1079}, {"id": 743}],
        },
    }
    facts = compile_alakazam_public_tactical_facts(observation)
    assert facts.powerful_hand_required_hand_cards == 6
    assert facts.powerful_hand_current_lethal is True
    assert facts.close_game_search_candidate is True
    assert facts.replacement_line_live is True
    assert facts.visible_tutor_card_ids == (1079, 743)
    assert facts.to_public_facts()["action_authority"] is False


def test_alakazam_sme_ranker_preserves_r195_principal_and_orders_deviations():
    state = _state("root", actions=((0,), (1,), (2,)))
    state = replace(state, raw_observation={"current": {}, "select": {}})
    policy = [
        RankedAction((0,), 0.7),
        RankedAction((1,), 0.2),
        RankedAction((2,), 0.1),
    ]

    def scores(_obs, actions, **_kwargs):
        assert actions == [(0,), (1,), (2,)]
        return [0.0, 1.0, 5.0]

    ranked = rank_alakazam_sme_candidates(
        state,
        policy,
        deck=(650, 651, 743),
        score_actions=scores,
    )
    assert [row.action for row in ranked] == [(0,), (2,), (1,)]
    assert ranked[0].probability == 0.7
    assert ranked[1].sme_priority == 5.0


def test_owned_process_backend_returns_typed_transition_and_closes_cleanly():
    root = _state("root", actions=((0,),))
    with OwnedProcessTacticalBackend(_owned_fixture_factory) as backend:
        transition = backend.advance(
            root,
            (0,),
            deadline_monotonic=time.monotonic() + 2.0,
        )
        assert transition.next_state.terminal_winner == 0
        assert transition.action_token == (0,)
        pid = backend.child_pid
    receipt = backend.receipt
    assert pid is not None
    assert receipt.requests == 1
    assert receipt.completed == 1
    assert receipt.closed is True
    assert receipt.bounded_reap_succeeded is True


def test_owned_process_backend_reaps_only_its_timed_out_child():
    root = _state("root", actions=((0,),))
    backend = OwnedProcessTacticalBackend(
        _hanging_owned_fixture_factory,
        cleanup_seconds=0.5,
    )
    with pytest.raises(TacticalSequenceError, match="deadline exceeded"):
        backend.advance(
            root,
            (0,),
            deadline_monotonic=time.monotonic() + 0.05,
        )
    receipt = backend.receipt
    assert receipt.timeouts == 1
    assert receipt.bounded_reap_attempted is True
    assert receipt.bounded_reap_succeeded is True
    assert receipt.closed is True


def test_root_over_64_is_a_hard_boundary_without_backend_call():
    root = _state("wide-root", actions=((0,),), ordered_count=65)
    backend = _GraphBackend({})
    planner = TacticalSequencePlanner(
        backend=backend,
        rank_actions=lambda _state: (_ for _ in ()).throw(
            AssertionError("ranker must not run past the cardinality boundary")
        ),
        config=_config(),
    )
    result = planner.search(
        root=root,
        direct_action=(0,),
        goal=ExactTerminalWinGoal(root_actor=0),
    )
    assert result.proposed_action is None
    assert result.receipt["boundary_counts"] == {
        "deterministic_internal_fanout_over_64": 1
    }
    assert backend.calls == []


def test_planner_can_use_owned_process_backend_for_shadow_terminal_proof():
    root = _state("root", actions=((0,),))
    with OwnedProcessTacticalBackend(_owned_fixture_factory) as backend:
        planner = TacticalSequencePlanner(
            backend=backend,
            rank_actions=lambda _state: [RankedAction((0,), 1.0)],
            config=TacticalSearchConfig(wall_seconds=2.0),
        )
        result = planner.search(
            root=root,
            direct_action=(0,),
            goal=ExactTerminalWinGoal(root_actor=0),
        )
    assert result.status == "proven_exact_terminal_win_shadow"
    assert result.receipt["backend_isolation_mode"] == "owned_bounded_child"
    assert result.dispatch_authorized is False
