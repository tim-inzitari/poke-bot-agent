from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

from poke_bot.recursive_turn_planner.chance_aware_tree import ChanceAwareSearchConfig
from poke_bot.recursive_turn_planner.neural_leaf_reranker import (
    LeafDeadline,
    LeafEvaluation,
    LeafKind,
    LeafRequest,
    LeafRerankerResult,
    LeafRerankerTelemetry,
    RootPolicyPriors,
)
from poke_bot.recursive_turn_planner.r207_simulator_arena import (
    AbsolutePlannerDeadlineController,
    ExactChanceOutcome,
    OpaqueMidgameHandle,
    SuccessorTransition,
    TransitionKind,
    public_observation_sha256,
)
from poke_bot.recursive_turn_planner.simulator_one_turn_expectimax import (
    MCTSExpansionProfile,
    PolicyDecisionView,
    R207ArenaAdapter,
    SimulatorInterTurnMCTSSession,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, nanoseconds: int) -> None:
        self.now_ns += nanoseconds


@dataclass(frozen=True)
class _State:
    name: str
    actions: tuple[tuple[int, ...], ...]
    direct: tuple[int, ...]
    priors: tuple[float, ...]
    leaf_value: float
    actor: int = 0
    turn_key: tuple[int, int] = (0, 7)
    serial: int = 0
    terminal_result: int | None = None


class _Arena:
    def __init__(self, states: tuple[_State, ...], *, root: str) -> None:
        self._states = {state.name: state for state in states}
        self._root = root
        self._arena_sha = _digest("fake-r207-arena")
        self._handles = {
            state.name: OpaqueMidgameHandle(
                self._arena_sha,
                _digest(f"fake-handle:{state.name}"),
                public_observation_sha256(
                    {
                        "node": state.name,
                        "current": {
                            "result": (
                                -1
                                if state.terminal_result is None
                                else state.terminal_result
                            )
                        },
                    }
                ),
            )
            for state in states
        }
        self._names_by_handle = {
            handle.handle_id_sha256: name for name, handle in self._handles.items()
        }
        self._names_by_public = {
            handle.public_observation_sha256: name
            for name, handle in self._handles.items()
        }
        self.transitions: dict[tuple[str, tuple[int, ...]], SuccessorTransition] = {}
        self.expand_calls: list[tuple[str, tuple[int, ...]]] = []

    def state(self, name: str) -> _State:
        return self._states[name]

    def handle(self, name: str) -> OpaqueMidgameHandle:
        return self._handles[name]

    def name_from_public(self, fingerprint: str) -> str:
        return self._names_by_public[fingerprint]

    def _observation(self, name: str) -> dict[str, object]:
        state = self.state(name)
        return {
            "node": name,
            "current": {"result": -1 if state.terminal_result is None else state.terminal_result},
        }

    def capture_root(self, deadline):
        deadline.check("fake_capture_root")
        return self.handle(self._root), self._observation(self._root)

    def observe(self, handle, deadline):
        deadline.check("fake_observe")
        return self._observation(self._names_by_handle[handle.handle_id_sha256])

    def expand_action(self, handle, action, deadline):
        deadline.check("fake_expand")
        source = self._names_by_handle[handle.handle_id_sha256]
        exact_action = tuple(action)
        self.expand_calls.append((source, exact_action))
        return self.transitions[(source, exact_action)]

    def deterministic(self, source: str, action: tuple[int, ...], child: str) -> None:
        target = self.state(child)
        handle = self.handle(child)
        self.transitions[(source, action)] = SuccessorTransition(
            kind=TransitionKind.DETERMINISTIC_PUBLIC,
            transition_certificate_sha256=_digest(f"det:{source}:{action}:{child}"),
            public_observation_sha256=handle.public_observation_sha256,
            child_handle=handle,
            next_turn_key=target.turn_key,
            next_actor_seat=target.actor,
        )

    def terminal(self, source: str, action: tuple[int, ...], result: int, *, label: str) -> None:
        self.transitions[(source, action)] = SuccessorTransition(
            kind=TransitionKind.TERMINAL,
            transition_certificate_sha256=_digest(f"terminal:{source}:{action}:{label}"),
            public_observation_sha256=_digest(f"terminal-public:{label}"),
            terminal_result=result,
        )

    def boundary(self, source: str, action: tuple[int, ...], target: str) -> None:
        self.transitions[(source, action)] = SuccessorTransition(
            kind=TransitionKind.INFORMATION_BOUNDARY,
            transition_certificate_sha256=_digest(f"boundary:{source}:{action}:{target}"),
            public_observation_sha256=self.handle(target).public_observation_sha256,
            boundary_reason="unresolved_private_or_opponent_information",
        )

    def chance(self, source: str, action: tuple[int, ...], outcomes: tuple[tuple[str, Fraction], ...]) -> None:
        exact = tuple(
            ExactChanceOutcome(
                label=name,
                probability=probability,
                child_handle=self.handle(name),
                public_observation_sha256=self.handle(name).public_observation_sha256,
                outcome_certificate_sha256=_digest(f"chance-outcome:{source}:{name}"),
            )
            for name, probability in outcomes
        )
        self.transitions[(source, action)] = SuccessorTransition(
            kind=TransitionKind.FINITE_PUBLIC_CHANCE,
            transition_certificate_sha256=_digest(f"chance:{source}:{action}"),
            chance_outcomes=exact,
        )


class _Factory:
    def __init__(self, arena: _Arena) -> None:
        self._arena = arena

    def _view(self, name: str, handle: OpaqueMidgameHandle | None = None) -> PolicyDecisionView:
        state = self._arena.state(name)
        exact_handle = self._arena.handle(name) if handle is None else handle
        return PolicyDecisionView(
            handle=exact_handle,
            public_observation_sha256=exact_handle.public_observation_sha256,
            turn_key=state.turn_key,
            decision_serial=state.serial,
            acting_seat=state.actor,
            legal_actions=state.actions,
            option_encoding_sha256=_digest(f"encoding:{name}"),
            direct_action=state.direct,
            packet={"fake_policy_packet": name},
            future_legality_receipt_sha256=_digest(f"future-legality:{name}"),
            simulator_result_sha256=_digest(f"sim-result:{name}"),
        )

    def view(self, name: str) -> PolicyDecisionView:
        return self._view(name)

    def build_decision(self, *, handle, observation, root_seat, transition, deadline):
        deadline.check("fake_build_decision")
        return self._view(str(observation["node"]), handle)

    def build_boundary_leaf(self, *, parent, transition, root_seat, deadline):
        deadline.check("fake_build_boundary")
        assert transition.public_observation_sha256 is not None
        return self._view(self._arena.name_from_public(transition.public_observation_sha256))


class _Reranker:
    """Hermetic structural fake for the frozen r205 reranker surface."""

    def __init__(
        self,
        arena: _Arena,
        clock: _Clock,
        *,
        advances_ns: tuple[int, ...] = (),
    ) -> None:
        self._arena = arena
        self._clock = clock
        self._advances_ns = advances_ns
        self.calls: list[tuple[LeafRequest, ...]] = []

    def evaluate(self, requests, *, deadline: LeafDeadline) -> LeafRerankerResult:
        rows = tuple(requests)
        self.calls.append(rows)
        index = len(self.calls) - 1
        if index < len(self._advances_ns):
            self._clock.advance(self._advances_ns[index])
        evaluations: list[LeafEvaluation] = []
        for request in rows:
            state = self._arena.state(
                self._arena.name_from_public(request.public_state_sha256)
            )
            ranked = tuple(
                sorted(range(len(state.priors)), key=lambda value: (-state.priors[value], value))
            )
            priors = RootPolicyPriors(
                actions=request.expected_actions,
                probabilities=state.priors,
                raw_logits=state.priors,
                direct_action_index=request.expected_actions.index(request.direct_action),
                ranked_action_indices=ranked,
            )
            evaluations.append(
                LeafEvaluation(
                    request_id=request.request_id,
                    kind=request.kind,
                    simulator_result_sha256=request.simulator_result_sha256,
                    public_state_sha256=request.public_state_sha256,
                    exact_terminal_value=None,
                    root_policy_priors=priors,
                    raw_value_from_root=state.leaf_value,
                    outcome_probabilities_from_root=(0.1, 0.2, 0.7),
                    raw_outcome_expected_value_from_root=0.6,
                    selected_leaf_value=state.leaf_value,
                    value_source="hermetic_calibrated_fake",
                    search_value_eligible=True,
                )
            )
        count = len(rows)
        return LeafRerankerResult(
            evaluations=tuple(evaluations),
            telemetry=LeafRerankerTelemetry(
                simulator_results_seen=count,
                simulator_terminal_results_seen=0,
                simulator_boundary_leaves_seen=sum(
                    request.kind is LeafKind.BOUNDARY for request in rows
                ),
                neural_evaluations_started=count,
                neural_evaluations_accepted=count,
                neural_forward_batches=1,
                root_policy_prior_evaluations=count,
                outcome_head_evaluations=count,
                frozen_policy_prior_batches=1,
                frozen_policy_prior_evaluations=count,
                batched_frozen_outcome_value_leaf_reranking_batches=1,
                frozen_outcome_leaf_evaluations=count,
                frozen_value_leaf_evaluations=count,
                nonterminal_leaves_reranked=count,
                terminal_exact_results_not_reranked=0,
                result_or_leaf_evaluations_seen=count,
                exact_terminal_values_used=0,
                requested_leaf_batch_completed=True,
                every_leaf_has_search_eligible_value=True,
                deadline_hit=False,
                incomplete_reason=None,
                elapsed_seconds=0.0,
            ),
        )


def _session(
    arena: _Arena,
    factory: _Factory,
    reranker: _Reranker,
    clock: _Clock,
    *,
    profile: MCTSExpansionProfile,
) -> SimulatorInterTurnMCTSSession:
    config = ChanceAwareSearchConfig()
    return SimulatorInterTurnMCTSSession(
        adapter=R207ArenaAdapter(arena, factory),
        reranker=reranker,
        config=config,
        profile=profile,
        deadline_controller=AbsolutePlannerDeadlineController(config, clock_ns=clock),
    )


def _two_decision_toy() -> tuple[_Arena, _Factory, _Clock, _Reranker]:
    states = (
        _State("root", ((0,), (1,)), (1,), (0.99, 0.01), -0.4),
        # The one-step leaf value makes A look bad.  It becomes good only after
        # the second same-seat simulator decision selects B.
        _State("after-a", ((2,), (3,)), (3,), (0.99, 0.01), -0.9, serial=1),
        _State("wrong-real-child", ((4,),), (4,), (1.0,), 0.0, serial=9),
    )
    arena = _Arena(states, root="root")
    arena.deterministic("root", (0,), "after-a")
    arena.terminal("root", (1,), 1, label="root-direct-loss")
    arena.terminal("after-a", (2,), 0, label="two-step-win")
    arena.terminal("after-a", (3,), 1, label="two-step-direct-loss")
    clock = _Clock()
    factory = _Factory(arena)
    return arena, factory, clock, _Reranker(arena, clock)


def test_searches_two_same_turn_atomic_actions_before_returning_one_real_action() -> None:
    arena, factory, clock, reranker = _two_decision_toy()
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(
            requested_simulations=2,
            max_decision_depth=2,
            max_tree_nodes=12,
        ),
    )

    result = session.plan_next(planner_turn_id="turn-0-7", seat=0, real_view=factory.view("root"))

    assert result.selected_action == (0,)
    assert result.direct_action == (1,)
    assert result.telemetry.actions_dispatched == 1
    assert result.telemetry.requested_tree_fully_expanded_and_backed_up_within_budget
    assert result.telemetry.completed_simulations == 2
    assert result.telemetry.decision_nodes_expanded >= 2
    assert result.telemetry.simulator_transitions_seen >= 3
    scores = {score.action: score for score in result.root_action_scores}
    assert scores[(0,)].exact_value == Fraction(1)
    assert scores[(1,)].exact_value == Fraction(-1)
    assert ("after-a", (2,)) in arena.expand_calls
    canonical = result.telemetry.to_bo1000_turn_telemetry(
        game_nonce_sha256=_digest("bo1000-game"),
        pair_id="r207-pair-hermetic",
        mcts_seat=0,
        selected_action=result.selected_action,
        legal_actions=factory.view("root").legal_actions,
    )
    assert canonical.config_sha256 == ChanceAwareSearchConfig().identity_sha256
    assert canonical.nonterminal_leaves_reranked is True


def test_capture_root_adapter_uses_the_same_typed_action_lease() -> None:
    arena, factory, clock, reranker = _two_decision_toy()
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=12),
    )

    result = session.capture_and_plan(planner_turn_id="captured", seat=0, turn_key=(0, 7))

    assert result.selected_action == (0,)
    assert result.telemetry.actions_dispatched == 1
    assert result.telemetry.simulator_transitions_seen >= 3


def test_cache_reuse_requires_explicit_realized_deterministic_attestation() -> None:
    arena, factory, clock, reranker = _two_decision_toy()
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=12),
    )
    first = session.plan_next(planner_turn_id="turn-0-7-a", seat=0, real_view=factory.view("root"))
    assert first.selected_action == (0,)

    # A matching public view alone cannot prove that the real transition was
    # deterministic: a realized chance outcome may look identical.  Omitting
    # the execution-boundary attestation must rebuild rather than reuse.
    rebuilt = session.plan_next(
        planner_turn_id="turn-0-7-b", seat=0, real_view=factory.view("after-a")
    )
    assert rebuilt.telemetry.cache_hits == 0
    assert rebuilt.telemetry.deterministic_subtree_reuses == 0
    assert rebuilt.telemetry.tree_rebuilds == 1
    assert rebuilt.telemetry.cache_invalidation_reason == (
        "missing_realized_deterministic_attestation"
    )


def test_matching_real_child_reuses_only_a_verified_deterministic_subtree() -> None:
    arena, factory, clock, reranker = _two_decision_toy()
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=12),
    )
    first = session.plan_next(planner_turn_id="turn-0-7-a", seat=0, real_view=factory.view("root"))
    assert first.selected_action == (0,)

    session.observe_real_action(
        action=(0,),
        next_view=factory.view("after-a"),
        realized_transition_kind=TransitionKind.DETERMINISTIC_PUBLIC,
    )
    reused = session.plan_next(
        planner_turn_id="turn-0-7-b", seat=0, real_view=factory.view("after-a")
    )
    assert reused.telemetry.cache_hits == 1
    assert reused.telemetry.deterministic_subtree_reuses == 1
    assert reused.telemetry.tree_rebuilds == 0

    # A mismatched realised public child invalidates the same pending cache;
    # it cannot reuse a look-alike scratch handle or a chance/opponent edge.
    fresh = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=12),
    )
    fresh.plan_next(planner_turn_id="turn-0-7-c", seat=0, real_view=factory.view("root"))
    fresh.observe_real_action(
        action=(0,),
        next_view=factory.view("wrong-real-child"),
        realized_transition_kind=TransitionKind.DETERMINISTIC_PUBLIC,
    )
    rebuilt = fresh.plan_next(planner_turn_id="turn-0-7-d", seat=0, real_view=factory.view("root"))
    assert rebuilt.telemetry.cache_hits == 0
    assert rebuilt.telemetry.tree_rebuilds == 1
    assert rebuilt.telemetry.cache_invalidation_reason == "real_child_fingerprint_mismatch"

    chance_invalidated = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=12),
    )
    chance_invalidated.plan_next(
        planner_turn_id="turn-0-7-e", seat=0, real_view=factory.view("root")
    )
    chance_invalidated.observe_real_action(
        action=(0,),
        next_view=None,
        realized_transition_kind=TransitionKind.FINITE_PUBLIC_CHANCE,
    )
    chance_rebuilt = chance_invalidated.plan_next(
        planner_turn_id="turn-0-7-f", seat=0, real_view=factory.view("root")
    )
    assert chance_rebuilt.telemetry.cache_hits == 0
    assert chance_rebuilt.telemetry.cache_invalidation_reason == (
        "realized_non_deterministic_transition"
    )


def test_verified_deterministic_cache_can_advance_multiple_same_turn_hops() -> None:
    states = (
        _State("root", ((0,),), (0,), (1.0,), 0.0),
        _State("middle", ((1,),), (1,), (1.0,), 0.0, serial=1),
        _State("end", ((2,),), (2,), (1.0,), 0.0, serial=2),
    )
    arena = _Arena(states, root="root")
    arena.deterministic("root", (0,), "middle")
    arena.deterministic("middle", (1,), "end")
    arena.terminal("end", (2,), 0, label="same-turn-win")
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock)
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=1, max_decision_depth=3, max_tree_nodes=8),
    )

    first = session.plan_next(planner_turn_id="hop-root", seat=0, real_view=factory.view("root"))
    session.observe_real_action(
        action=first.selected_action,
        next_view=factory.view("middle"),
        realized_transition_kind=TransitionKind.DETERMINISTIC_PUBLIC,
    )
    middle = session.plan_next(
        planner_turn_id="hop-middle", seat=0, real_view=factory.view("middle")
    )
    assert middle.telemetry.cache_hits == 1

    session.observe_real_action(
        action=middle.selected_action,
        next_view=factory.view("end"),
        realized_transition_kind=TransitionKind.DETERMINISTIC_PUBLIC,
    )
    end = session.plan_next(planner_turn_id="hop-end", seat=0, real_view=factory.view("end"))
    assert end.telemetry.cache_hits == 1
    assert end.telemetry.deterministic_subtree_reuses == 1
    assert arena.expand_calls == [("root", (0,)), ("middle", (1,)), ("end", (2,))]


def test_finite_chance_expands_every_terminal_outcome_and_keeps_fraction_backup() -> None:
    states = (
        _State("root", ((0,),), (0,), (1.0,), 0.0),
        _State("heads", ((1,),), (1,), (1.0,), 0.0, terminal_result=0),
        _State("tails", ((1,),), (1,), (1.0,), 0.0, terminal_result=1),
    )
    arena = _Arena(states, root="root")
    arena.chance("root", (0,), (("heads", Fraction(1, 2)), ("tails", Fraction(1, 2))))
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock)
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=8),
    )

    result = session.plan_next(planner_turn_id="chance", seat=0, real_view=factory.view("root"))

    score = result.root_action_scores[0]
    assert score.exact_value == Fraction(0)
    assert result.telemetry.finite_chance_outcomes_evaluated == 2
    assert result.telemetry.terminal_exact_results_seen == 2
    assert result.telemetry.neural_leaf_evaluations_seen == 1  # root policy/value only
    assert result.telemetry.result_or_leaf_evaluations_seen == 3
    assert result.telemetry.completed_simulations == 2
    assert result.telemetry.terminal_exact_results_not_reranked is True
    assert [len(batch) for batch in reranker.calls] == [1]


def test_finite_chance_revisits_controlled_children_before_exact_backup() -> None:
    states = (
        _State("root", ((0,),), (0,), (1.0,), 0.0),
        _State("heads", ((1,), (2,)), (1,), (0.9, 0.1), 0.0, serial=1),
        _State("tails", ((1,), (2,)), (1,), (0.9, 0.1), 0.0, serial=2),
    )
    arena = _Arena(states, root="root")
    arena.chance("root", (0,), (("heads", Fraction(1, 2)), ("tails", Fraction(1, 2))))
    for child in ("heads", "tails"):
        arena.terminal(child, (1,), 1, label=f"{child}-first-prior-loss")
        arena.terminal(child, (2,), 0, label=f"{child}-second-prior-win")
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock)
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=8),
    )

    result = session.plan_next(
        planner_turn_id="chance-controlled-children",
        seat=0,
        real_view=factory.view("root"),
    )

    # The first visit follows each high-prior losing action.  The second visit
    # must retain those chance children, try their winning alternatives, and
    # recompute the exact 1/2 + 1/2 backup rather than freezing the first value.
    assert result.root_action_scores[0].exact_value == Fraction(1)
    assert ("heads", (1,)) in arena.expand_calls
    assert ("heads", (2,)) in arena.expand_calls
    assert ("tails", (1,)) in arena.expand_calls
    assert ("tails", (2,)) in arena.expand_calls
    assert result.telemetry.finite_chance_outcomes_evaluated == 2
    assert result.telemetry.completed_simulations == 2


def test_reused_chance_boundary_is_one_neural_leaf_in_telemetry() -> None:
    states = (
        _State("root", ((0,),), (0,), (1.0,), 0.0),
        _State("opponent-boundary", ((1,),), (1,), (1.0,), 0.25, actor=1, serial=1),
        _State("exact-win", ((1,),), (1,), (1.0,), 0.0, terminal_result=0),
    )
    arena = _Arena(states, root="root")
    arena.chance(
        "root",
        (0,),
        (("opponent-boundary", Fraction(1, 2)), ("exact-win", Fraction(1, 2))),
    )
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock)
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=8),
    )

    result = session.plan_next(
        planner_turn_id="chance-boundary-reuse",
        seat=0,
        real_view=factory.view("root"),
    )

    assert result.telemetry.requested_tree_fully_expanded_and_backed_up_within_budget
    assert result.telemetry.direct_fallback_used is False
    assert result.telemetry.boundary_leaf_results_seen == 1
    assert result.telemetry.neural_leaf_evaluations_seen == 2  # root + boundary
    assert result.telemetry.terminal_exact_results_seen == 1
    assert result.telemetry.result_or_leaf_evaluations_seen == 3


def test_information_boundary_is_one_neural_leaf_not_a_second_simulator_leaf() -> None:
    states = (
        _State("root", ((0,),), (0,), (1.0,), 0.0),
        _State("boundary", ((1,),), (1,), (1.0,), 0.25, actor=1),
    )
    arena = _Arena(states, root="root")
    arena.boundary("root", (0,), "boundary")
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock)
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=1, max_decision_depth=2, max_tree_nodes=8),
    )

    result = session.plan_next(planner_turn_id="boundary", seat=0, real_view=factory.view("root"))

    telemetry = result.telemetry
    assert telemetry.boundary_leaf_results_seen == 1
    assert telemetry.terminal_exact_results_seen == 0
    assert telemetry.simulator_leaf_evaluations_seen == 0
    assert telemetry.neural_leaf_evaluations_seen == 2
    assert telemetry.result_or_leaf_evaluations_seen == 2
    assert telemetry.frozen_outcome_leaf_evaluations == telemetry.neural_leaf_evaluations_seen
    assert telemetry.frozen_value_leaf_evaluations == telemetry.neural_leaf_evaluations_seen


def test_late_leaf_batch_is_discarded_and_returns_the_exact_direct_action() -> None:
    states = (_State("root", ((0,),), (0,), (1.0,), 0.0),)
    arena = _Arena(states, root="root")
    arena.terminal("root", (0,), 0, label="unused-because-leaf-is-late")
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock, advances_ns=(6_000_000_000,))
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=1, max_decision_depth=1, max_tree_nodes=4),
    )

    result = session.plan_next(planner_turn_id="late", seat=0, real_view=factory.view("root"))

    assert result.selected_action == result.direct_action == (0,)
    assert result.telemetry.direct_fallback_used is True
    assert result.telemetry.deadline_hit is True
    assert result.telemetry.tree_incomplete_reason is not None
    assert "deadline_action" in result.telemetry.tree_incomplete_reason
    assert result.telemetry.max_single_action_planner_wall_seconds == 5.0


def test_tree_digest_work_is_charged_to_the_action_deadline() -> None:
    states = (_State("root", ((0,),), (0,), (1.0,), 0.0),)
    arena = _Arena(states, root="root")
    arena.terminal("root", (0,), 0, label="on-time-terminal")
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock)
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=1, max_decision_depth=1, max_tree_nodes=4),
    )
    original_tree_sha = session._tree_sha256

    def late_tree_sha(root):
        clock.advance(6_000_000_000)
        return original_tree_sha(root)

    session._tree_sha256 = late_tree_sha  # type: ignore[method-assign]
    result = session.plan_next(
        planner_turn_id="late-tree-digest", seat=0, real_view=factory.view("root")
    )

    assert result.selected_action == result.direct_action == (0,)
    assert result.telemetry.direct_fallback_used is True
    assert result.telemetry.deadline_hit is True
    assert "deadline_action_mcts_after_tree_digest" == result.telemetry.tree_incomplete_reason


def test_turn_clock_persists_across_real_actions_and_profile_cap_falls_back() -> None:
    states = (
        _State("root", ((0,),), (0,), (1.0,), 0.0),
        _State("child", ((1,),), (1,), (1.0,), 0.0, serial=1),
    )
    arena = _Arena(states, root="root")
    arena.terminal("root", (0,), 0, label="win")
    clock = _Clock()
    factory = _Factory(arena)
    reranker = _Reranker(arena, clock, advances_ns=(4_000_000_000, 17_000_000_000))
    session = _session(
        arena,
        factory,
        reranker,
        clock,
        profile=MCTSExpansionProfile(requested_simulations=1, max_decision_depth=1, max_tree_nodes=4),
    )
    first = session.plan_next(planner_turn_id="same-turn-a", seat=0, real_view=factory.view("root"))
    second = session.plan_next(planner_turn_id="same-turn-b", seat=0, real_view=factory.view("root"))
    assert first.telemetry.requested_tree_fully_expanded_and_backed_up_within_budget
    assert second.telemetry.direct_fallback_used is True
    assert second.telemetry.deadline_hit is True
    assert "deadline_turn" in (second.telemetry.tree_incomplete_reason or "")
    assert second.telemetry.turn_planner_wall_seconds == 20.0

    cap_arena, cap_factory, cap_clock, cap_reranker = _two_decision_toy()
    cap_session = _session(
        cap_arena,
        cap_factory,
        cap_reranker,
        cap_clock,
        profile=MCTSExpansionProfile(requested_simulations=2, max_decision_depth=2, max_tree_nodes=1),
    )
    capped = cap_session.plan_next(
        planner_turn_id="node-cap", seat=0, real_view=cap_factory.view("root")
    )
    assert capped.selected_action == capped.direct_action == (1,)
    assert capped.telemetry.direct_fallback_used is True
    assert capped.telemetry.tree_incomplete_reason == "requested_profile_tree_node_cap_exhausted"
