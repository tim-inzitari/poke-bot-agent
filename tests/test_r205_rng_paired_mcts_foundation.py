"""Focused tests for the isolated r205 simulator-backed search foundation.

The fake engine is intentionally tiny, but every action edge is expanded by a
real call to its clone + step methods.  This prevents a prebuilt phase-1 tree
or a BattleStart replay from accidentally satisfying the actual-search claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

from experiments.r205_rng_paired_mcts.foundation import (
    BoundaryTransition,
    DeterministicTransition,
    ExactChanceOutcome,
    ExactChanceTransition,
    MidgameCloneCapability,
    OneTurnSearchConfig,
    R205FoundationError,
    SealedStartMaterial,
    SearchView,
    SimulatorBackedOneTurnMCTS,
    TerminalTransition,
    build_seat_swapped_bo1000_schedule,
    complete_action_fingerprint,
    restore_scheduled_game_root,
    validate_bo1000_schedule,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _State:
    name: str


def _view(
    name: str,
    *,
    turn: tuple[int, int],
    actions: tuple[tuple[int, ...], ...],
    direct: tuple[int, ...],
    seat: int = 0,
) -> SearchView:
    return SearchView(
        state_id=name,
        turn_key=turn,
        acting_seat=seat,
        legal_actions=actions,
        direct_action=direct,
        observation_sha256=_digest(f"obs:{name}"),
        legal_actions_sha256=complete_action_fingerprint(actions),
        option_encoding_sha256=_digest(f"options:{name}"),
    )


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Branch:
    def __init__(self, owner: _Forker, state: _State) -> None:
        self._owner = owner
        self._state = state
        self._closed = False

    def step(self, action: tuple[int, ...]):
        assert not self._closed
        self._owner.step_calls.append((self._state.name, action))
        if self._owner.on_step is not None:
            self._owner.on_step()
        return self._owner.transitions[(self._state.name, action)]

    def close(self) -> None:
        self._closed = True
        self._owner.close_calls += 1


class _Forker:
    def __init__(
        self,
        views: dict[str, SearchView],
        transitions: dict[tuple[str, tuple[int, ...]], object],
        *,
        capability: MidgameCloneCapability | None = None,
        on_step=None,
    ) -> None:
        self.views = views
        self.transitions = transitions
        self.capability = capability or _native_capability()
        self.on_step = on_step
        self.clone_calls: list[str] = []
        self.step_calls: list[tuple[str, tuple[int, ...]]] = []
        self.close_calls = 0

    def clone(self, state: _State) -> _Branch:
        self.clone_calls.append(state.name)
        return _Branch(self, state)


def _native_capability() -> MidgameCloneCapability:
    return MidgameCloneCapability(
        abi_name="poke_bot.r205.test_native_midgame_clone",
        abi_version=1,
        source_kind="native_midgame_clone",
        status="passed",
        arbitrary_midgame_policy_visible_decision=True,
        full_state_game_rng_config_counters=True,
        exact_future_legality=True,
        information_set_safe=True,
        independent_clone=True,
        engine_artifact_sha256=_digest("test-engine"),
    )


def _deterministic(next_state: _State, view: SearchView) -> DeterministicTransition:
    return DeterministicTransition(
        next_state=next_state,
        successor_certificate_sha256=_digest(f"successor:{next_state.name}"),
        future_legality_sha256=view.legal_actions_sha256,
    )


def test_real_clone_and_step_calls_expand_multiple_same_turn_actions() -> None:
    root = _State("root")
    deep = _State("deep")
    views = {
        "root": _view(
            "root",
            turn=(0, 7),
            actions=((0,), (1,)),
            direct=(1,),
        ),
        "deep": _view(
            "deep",
            turn=(0, 7),
            actions=((2,), (3,)),
            direct=(2,),
        ),
    }
    transitions = {
        ("root", (0,)): _deterministic(deep, views["deep"]),
        ("root", (1,)): TerminalTransition(Fraction(3)),
        ("deep", (2,)): TerminalTransition(Fraction(5)),
        ("deep", (3,)): TerminalTransition(Fraction(2)),
    }
    forker = _Forker(views, transitions)
    planner = SimulatorBackedOneTurnMCTS(
        forker=forker,
        view_of=lambda state: views[state.name],
        leaf_value=lambda _state, _view: Fraction(0),
    )

    result = planner.plan(root)

    assert result.executed_action == (1,)  # shadow-only foundation remains direct
    assert result.shadow_recommended_action == (0,)
    assert result.requested_tree_fully_expanded_and_backed_up_within_budget is True
    assert result.direct_fallback_used is False
    assert result.action_authority_enabled is False
    assert forker.step_calls == [
        ("root", (0,)),
        ("deep", (2,)),
        ("deep", (3,)),
        ("root", (1,)),
    ]
    assert result.telemetry.actual_native_clone_calls == 4
    assert result.telemetry.actual_simulator_step_calls == 4
    assert result.telemetry.decision_nodes_expanded == 2
    assert result.telemetry.unique_tree_nodes_seen == 2
    assert result.telemetry.result_or_leaf_evaluations_seen == 3
    assert forker.close_calls == 4


def test_exact_chance_is_fully_enumerated_and_backed_up_as_fraction() -> None:
    root = _State("root")
    heads = _State("heads")
    tails = _State("tails")
    views = {
        "root": _view(
            "root",
            turn=(0, 1),
            actions=((0,), (1,)),
            direct=(1,),
        ),
        # Both chance children are an exact current-turn boundary, so their
        # values come from the policy-visible leaf evaluator rather than a
        # sampled future simulator trajectory.
        "heads": _view(
            "heads",
            turn=(0, 2),
            actions=((2,),),
            direct=(2,),
        ),
        "tails": _view(
            "tails",
            turn=(0, 2),
            actions=((3,),),
            direct=(3,),
        ),
    }
    chance = ExactChanceTransition(
        event_id="fair-but-not-sampled",
        distribution_receipt_sha256=_digest("fair-coin"),
        outcomes=(
            ExactChanceOutcome(
                "heads",
                Fraction(1, 3),
                heads,
                views["heads"].legal_actions_sha256,
            ),
            ExactChanceOutcome(
                "tails",
                Fraction(2, 3),
                tails,
                views["tails"].legal_actions_sha256,
            ),
        ),
    )
    forker = _Forker(
        views,
        {
            ("root", (0,)): chance,
            ("root", (1,)): TerminalTransition(Fraction(1)),
        },
    )
    planner = SimulatorBackedOneTurnMCTS(
        forker=forker,
        view_of=lambda state: views[state.name],
        leaf_value=lambda state, _view: {
            "heads": Fraction(9),
            "tails": Fraction(0),
        }[state.name],
    )

    result = planner.plan(root)

    values = {row.action: row.value for row in result.action_values}
    assert values[(0,)] == Fraction(3)
    assert values[(1,)] == Fraction(1)
    assert result.shadow_recommended_action == (0,)
    assert result.telemetry.finite_chance_outcomes_evaluated == 2
    assert result.telemetry.result_or_leaf_evaluations_seen == 3
    assert all(state != "heads" for state, _action in forker.step_calls)
    assert all(state != "tails" for state, _action in forker.step_calls)


def test_unknown_chance_or_information_boundary_discards_partial_tree() -> None:
    root = _State("root")
    view = _view(
        "root",
        turn=(0, 4),
        actions=((0,), (1,)),
        direct=(1,),
    )
    forker = _Forker(
        {"root": view},
        {
            ("root", (0,)): BoundaryTransition("unknown_or_hidden_chance"),
            ("root", (1,)): TerminalTransition(Fraction(1)),
        },
    )
    planner = SimulatorBackedOneTurnMCTS(
        forker=forker,
        view_of=lambda _state: view,
        leaf_value=lambda _state, _view: Fraction(0),
    )

    result = planner.plan(root)

    assert result.executed_action == (1,)
    assert result.shadow_recommended_action is None
    assert result.requested_tree_fully_expanded_and_backed_up_within_budget is False
    assert result.direct_fallback_used is True
    assert result.reason == "chance_or_information_boundary:unknown_or_hidden_chance"
    assert result.telemetry.boundary_leaf_results_seen == 1


def test_malformed_exact_chance_is_rejected_before_any_action_authority() -> None:
    root = _State("root")
    view = _view("root", turn=(0, 1), actions=((0,),), direct=(0,))
    with pytest.raises(R205FoundationError, match="sum exactly"):
        ExactChanceTransition(
            event_id="incomplete",
            distribution_receipt_sha256=_digest("bad"),
            outcomes=(
                ExactChanceOutcome("a", Fraction(1, 3), root, view.legal_actions_sha256),
                ExactChanceOutcome("b", Fraction(1, 3), root, view.legal_actions_sha256),
            ),
        )


def test_deadline_charges_real_simulator_work_and_returns_direct_action() -> None:
    clock = _ManualClock()
    root = _State("root")
    view = _view("root", turn=(0, 1), actions=((0,),), direct=(0,))
    forker = _Forker(
        {"root": view},
        {("root", (0,)): TerminalTransition(Fraction(2))},
        on_step=lambda: clock.advance(5.01),
    )
    planner = SimulatorBackedOneTurnMCTS(
        forker=forker,
        view_of=lambda _state: view,
        leaf_value=lambda _state, _view: Fraction(0),
        config=OneTurnSearchConfig(max_turn_seconds=20.0, max_action_seconds=5.0),
        clock=clock,
    )

    result = planner.plan(root)

    assert result.executed_action == (0,)
    assert result.direct_fallback_used is True
    assert result.telemetry.deadline_hit is True
    assert result.telemetry.action_budget_hit is True
    assert result.telemetry.actual_simulator_step_calls == 1
    assert result.reason == "action_budget_exhausted:after_simulator_step"


def test_sealed_start_replay_capability_is_not_accepted_as_midgame_clone() -> None:
    root = _State("root")
    view = _view("root", turn=(0, 1), actions=((0,),), direct=(0,))
    replay_only = MidgameCloneCapability(
        abi_name="poke_bot.rtp_pairing_snapshot_abi",
        abi_version=2,
        source_kind="sealed_start_replay",
        status="passed",
        arbitrary_midgame_policy_visible_decision=False,
        full_state_game_rng_config_counters=True,
        exact_future_legality=False,
        information_set_safe=False,
        independent_clone=True,
        engine_artifact_sha256=_digest("sealed-start-engine"),
    )
    forker = _Forker(
        {"root": view},
        {("root", (0,)): TerminalTransition(Fraction(1))},
        capability=replay_only,
    )
    planner = SimulatorBackedOneTurnMCTS(
        forker=forker,
        view_of=lambda _state: view,
        leaf_value=lambda _state, _view: Fraction(0),
    )

    result = planner.plan(root)

    assert result.executed_action == (0,)
    assert result.direct_fallback_used is True
    assert result.reason == "replay-from-sealed-start is not a native arbitrary-midgame clone"
    assert forker.clone_calls == []
    assert forker.step_calls == []


def test_bo1000_schedule_is_500_matched_seat_swapped_pairs() -> None:
    materials = tuple(
        SealedStartMaterial(
            pair_id=f"pair-{index:04d}",
            snapshot_sha256=_digest(f"snapshot:{index}"),
            initial_rng_state_sha256=_digest(f"rng:{index}"),
            deck_order_randomness_sha256=_digest(f"deck-order:{index}"),
        )
        for index in range(500)
    )

    games = build_seat_swapped_bo1000_schedule(materials)
    summary = validate_bo1000_schedule(games)

    assert summary.total_games == 1000
    assert summary.matched_rng_pairs == 500
    assert summary.mcts_as_seat_0 == 500
    assert summary.mcts_as_seat_1 == 500
    first_pair = games[:2]
    assert first_pair[0].start_material.identity_sha256 == first_pair[1].start_material.identity_sha256
    assert first_pair[0].arms_by_seat == (
        "chance_aware_inter_turn_mcts",
        "no_rtp_direct_policy",
    )
    assert first_pair[1].arms_by_seat == (
        "no_rtp_direct_policy",
        "chance_aware_inter_turn_mcts",
    )


def test_bo1000_schedule_rejects_a_crossed_pair_material() -> None:
    materials = tuple(
        SealedStartMaterial(
            pair_id=f"pair-{index:04d}",
            snapshot_sha256=_digest(f"snapshot:{index}"),
            initial_rng_state_sha256=_digest(f"rng:{index}"),
            deck_order_randomness_sha256=_digest(f"deck-order:{index}"),
        )
        for index in range(500)
    )
    games = list(build_seat_swapped_bo1000_schedule(materials))
    games[1] = replace(
        games[1],
        start_material=replace(
            games[1].start_material,
            initial_rng_state_sha256=_digest("crossed-rng"),
        ),
    )

    with pytest.raises(R205FoundationError, match="crossed"):
        validate_bo1000_schedule(games)


def test_sealed_start_material_is_only_restored_as_a_fresh_scheduled_root() -> None:
    material = SealedStartMaterial(
        pair_id="pair-0000",
        snapshot_sha256=_digest("snapshot"),
        initial_rng_state_sha256=_digest("rng"),
        deck_order_randomness_sha256=_digest("deck-order"),
    )
    game = build_seat_swapped_bo1000_schedule(
        (material,)
        + tuple(
            SealedStartMaterial(
                pair_id=f"pair-{index:04d}",
                snapshot_sha256=_digest(f"snapshot:{index}"),
                initial_rng_state_sha256=_digest(f"rng:{index}"),
                deck_order_randomness_sha256=_digest(f"deck-order:{index}"),
            )
            for index in range(1, 500)
        )
    )[0]

    class _Restorer:
        def __init__(self) -> None:
            self.materials: list[SealedStartMaterial] = []

        def restore_sealed_start(self, received: SealedStartMaterial) -> _State:
            self.materials.append(received)
            return _State("fresh-root")

    restorer = _Restorer()
    assert restore_scheduled_game_root(restorer, game) == _State("fresh-root")
    assert restorer.materials == [material]
