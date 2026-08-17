from __future__ import annotations

import hashlib
from copy import deepcopy
from fractions import Fraction

import pytest

from poke_bot.recursive_turn_planner.chance_aware_tree import ChanceAwareSearchConfig
from poke_bot.recursive_turn_planner.r207_simulator_arena import (
    AbsolutePlannerDeadlineController,
    ExactChanceOutcome,
    OpaqueMidgameHandle,
    PlannerDeadlineExceeded,
    PublicTransitionAttestation,
    R207SimulatorArenaError,
    R207TurnSearchLedger,
    SuccessorTransition,
    TransitionKind,
    TruthfulReplayArena,
    controlled_successor_or_boundary,
    exact_chance_backup,
    public_observation_sha256,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Clock:
    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _SealedSnapshot:
    fingerprint_sha256 = _digest("sealed-initial-snapshot")

    @property
    def serialized_bytes(self) -> bytes:
        raise AssertionError("r207 replay arena must not read snapshot bytes")


class _FakeBattle:
    def __init__(self, engine: "_FakeEngine") -> None:
        self._engine = engine
        self.path: tuple[tuple[int, ...], ...] = ()
        self.closed = False

    def observation(self) -> dict:
        if self.closed:
            raise AssertionError("closed scratch battle was reused")
        return deepcopy(self._engine.states[self.path])

    def step(self, action: list[int] | tuple[int, ...]) -> dict:
        if self.closed:
            raise AssertionError("closed scratch battle was selected")
        next_path = self.path + (tuple(action),)
        if next_path not in self._engine.states:
            raise RuntimeError(f"unexpected simulator action path {next_path!r}")
        self.path = next_path
        return self.observation()

    def close(self) -> None:
        self.closed = True
        self._engine.closed_battles += 1


class _FakeEngine:
    def __init__(self) -> None:
        self.restore_calls = 0
        self.closed_battles = 0
        self.battles: list[_FakeBattle] = []
        self.states: dict[tuple[tuple[int, ...], ...], dict] = {
            (): {"state": "root", "current": {"result": -1}},
            ((0,),): {"state": "deterministic", "current": {"result": -1}},
            ((0,), (3,)): {"state": "terminal-after-replay", "current": {"result": 0}},
            ((1,),): {"state": "chance", "current": {"result": -1}},
            ((2,),): {"state": "terminal", "current": {"result": 1}},
            ((4,),): {"state": "private", "current": {"result": -1}},
            ((5,),): {"state": "opponent-next", "current": {"result": -1}},
        }

    def restore_snapshot(self, snapshot: object) -> _FakeBattle:
        assert isinstance(snapshot, _SealedSnapshot)
        self.restore_calls += 1
        battle = _FakeBattle(self)
        self.battles.append(battle)
        return battle


def _classifier(events: list[object]):
    def classify(event):
        events.append(event)
        assert isinstance(event.pre_observation, dict)
        assert isinstance(event.post_observation, dict)
        # The only classifier input is deep-copied public JSON.
        assert not hasattr(event, "battle")
        assert "hidden" not in event.pre_observation
        state = event.post_observation["state"]
        if state == "deterministic":
            return PublicTransitionAttestation(
                kind=TransitionKind.DETERMINISTIC_PUBLIC,
                certificate_sha256=_digest("deterministic-cert"),
                next_turn_key=(0, 4),
                next_actor_seat=0,
            )
        if state == "chance":
            return PublicTransitionAttestation(
                kind=TransitionKind.FINITE_PUBLIC_CHANCE,
                certificate_sha256=_digest("identified-fair-coin"),
            )
        if state == "opponent-next":
            return PublicTransitionAttestation(
                kind=TransitionKind.DETERMINISTIC_PUBLIC,
                certificate_sha256=_digest("opponent-next-cert"),
                next_turn_key=(1, 4),
                next_actor_seat=1,
            )
        return PublicTransitionAttestation(
            kind=TransitionKind.INFORMATION_BOUNDARY,
            certificate_sha256=_digest(f"boundary:{state}"),
            boundary_reason="private_information_boundary",
        )

    return classify


def _deadline(clock: _Clock):
    return AbsolutePlannerDeadlineController(
        ChanceAwareSearchConfig(), clock_ns=clock
    ).begin_action((0, 3))


def test_truthful_replay_uses_fresh_scratch_restores_and_opaque_handles_only() -> None:
    clock = _Clock()
    engine = _FakeEngine()
    events: list[object] = []
    snapshot = _SealedSnapshot()
    arena = TruthfulReplayArena(
        engine=engine,
        sealed_initial_snapshot=snapshot,
        classify_public_transition=_classifier(events),
    )
    deadline = _deadline(clock)

    root, root_observation = arena.capture_root(deadline)
    transition = arena.expand_action(root, (0,), deadline)
    assert transition.kind is TransitionKind.DETERMINISTIC_PUBLIC
    assert transition.child_handle is not None
    assert transition.child_handle.handle_id_sha256 != root.handle_id_sha256
    assert root_observation["state"] == "root"
    assert arena.observe(transition.child_handle, deadline)["state"] == "deterministic"

    assert len(events) == 1
    assert engine.restore_calls == 3  # root, root edge, child replay.
    assert arena.scratch_telemetry == {
        "scratch_restore_count": 3,
        "journal_replay_select_count": 1,
        "scratch_simulator_select_calls": 1,
        "simulator_observations_seen": 5,
    }
    assert all(battle.closed for battle in engine.battles)


def test_replay_fails_closed_when_a_replayed_public_checkpoint_diverges() -> None:
    clock = _Clock()
    engine = _FakeEngine()
    arena = TruthfulReplayArena(
        engine=engine,
        sealed_initial_snapshot=_SealedSnapshot(),
        classify_public_transition=_classifier([]),
    )
    deadline = _deadline(clock)
    root, _ = arena.capture_root(deadline)
    child = arena.expand_action(root, (0,), deadline).child_handle
    assert child is not None

    engine.states[((0,),)] = {"state": "tampered", "current": {"result": -1}}
    with pytest.raises(R207SimulatorArenaError, match="post-observation fingerprint mismatch"):
        arena.observe(child, deadline)


def test_replay_turns_finite_chance_and_private_paths_into_boundaries() -> None:
    clock = _Clock()
    engine = _FakeEngine()
    arena = TruthfulReplayArena(
        engine=engine,
        sealed_initial_snapshot=_SealedSnapshot(),
        classify_public_transition=_classifier([]),
    )
    deadline = _deadline(clock)
    root, _ = arena.capture_root(deadline)

    chance = arena.expand_action(root, (1,), deadline)
    assert chance.kind is TransitionKind.INFORMATION_BOUNDARY
    assert chance.boundary_reason == "finite_public_chance_requires_v3_outcome_enumerator"
    assert chance.child_handle is None
    private = arena.expand_action(root, (4,), deadline)
    assert private.kind is TransitionKind.INFORMATION_BOUNDARY
    assert private.boundary_reason == "private_information_boundary"
    assert private.child_handle is None


def test_opponent_next_decision_is_a_boundary_without_an_opponent_policy_call() -> None:
    clock = _Clock()
    engine = _FakeEngine()
    arena = TruthfulReplayArena(
        engine=engine,
        sealed_initial_snapshot=_SealedSnapshot(),
        classify_public_transition=_classifier([]),
    )
    deadline = _deadline(clock)
    root, _ = arena.capture_root(deadline)
    transition = arena.expand_action(root, (5,), deadline)
    assert transition.kind is TransitionKind.DETERMINISTIC_PUBLIC
    stopped = controlled_successor_or_boundary(transition, controlled_seat=0)
    assert stopped.kind is TransitionKind.INFORMATION_BOUNDARY
    assert stopped.boundary_reason == "opponent_decision_boundary"
    assert stopped.child_handle is None


def test_native_terminal_is_exact_and_classifier_cannot_replace_it() -> None:
    clock = _Clock()
    engine = _FakeEngine()
    classifier_events: list[object] = []
    arena = TruthfulReplayArena(
        engine=engine,
        sealed_initial_snapshot=_SealedSnapshot(),
        classify_public_transition=_classifier(classifier_events),
    )
    deadline = _deadline(clock)
    root, _ = arena.capture_root(deadline)
    terminal = arena.expand_action(root, (2,), deadline)

    assert terminal.kind is TransitionKind.TERMINAL
    assert terminal.terminal_result == 1
    assert classifier_events == []


def test_exact_finite_chance_requires_all_opaque_children_and_fraction_backup() -> None:
    arena_id = _digest("future-v3-arena")
    left = OpaqueMidgameHandle(arena_id, _digest("heads"), _digest("heads-public"))
    right = OpaqueMidgameHandle(arena_id, _digest("tails"), _digest("tails-public"))
    outcomes = (
        ExactChanceOutcome(
            "heads", Fraction(1, 2), left, _digest("heads-public"), _digest("heads-cert")
        ),
        ExactChanceOutcome(
            "tails", Fraction(1, 2), right, _digest("tails-public"), _digest("tails-cert")
        ),
    )
    transition = SuccessorTransition(
        kind=TransitionKind.FINITE_PUBLIC_CHANCE,
        transition_certificate_sha256=_digest("fair-coin-cert"),
        chance_outcomes=outcomes,
    )
    assert transition.kind is TransitionKind.FINITE_PUBLIC_CHANCE
    assert exact_chance_backup(
        outcomes,
        {
            left.handle_id_sha256: Fraction(1),
            right.handle_id_sha256: Fraction(-1),
        },
    ) == Fraction(0)
    with pytest.raises(R207SimulatorArenaError, match="missing an exact value"):
        exact_chance_backup(outcomes, {left.handle_id_sha256: Fraction(1)})


def test_absolute_deadline_discards_late_result_and_resets_only_on_real_turn_change() -> None:
    clock = _Clock()
    controller = AbsolutePlannerDeadlineController(
        ChanceAwareSearchConfig(), clock_ns=clock
    )
    first = controller.begin_action((0, 3))
    assert first.turn_deadline_ns == 20_000_000_000
    assert first.action_deadline_ns == 5_000_000_000
    clock.now_ns = 1_000_000_000
    same_turn = controller.begin_action((0, 3))
    assert same_turn.turn_deadline_ns == first.turn_deadline_ns
    assert same_turn.action_deadline_ns == 6_000_000_000
    clock.now_ns = 2_000_000_000
    next_turn = controller.begin_action((0, 4))
    assert next_turn.turn_deadline_ns == 22_000_000_000

    discarded: list[object] = []

    def slow_operation() -> str:
        clock.now_ns = 7_000_000_000
        return "late-scratch-result"

    with pytest.raises(PlannerDeadlineExceeded) as raised:
        same_turn.call("simulator_select", slow_operation, discard_late_result=discarded.append)
    assert raised.value.scope == "action"
    assert raised.value.late_result_discarded is True
    assert discarded == ["late-scratch-result"]


def test_turn_ledger_never_imputes_completion_and_separates_shadow_direct_from_fallback() -> None:
    ledger = R207TurnSearchLedger(
        planner_turn_id="r207-turn-0-3",
        seat=0,
        turn_key=(0, 3),
        selected_action_and_legality_fingerprint=_digest("selected-legal"),
        tree_and_config_sha256=_digest("tree-config"),
    )
    terminal = SuccessorTransition(
        kind=TransitionKind.TERMINAL,
        transition_certificate_sha256=_digest("terminal-cert"),
        public_observation_sha256=_digest("terminal-observation"),
        terminal_result=0,
    )
    boundary = SuccessorTransition(
        kind=TransitionKind.INFORMATION_BOUNDARY,
        transition_certificate_sha256=_digest("boundary-cert"),
        public_observation_sha256=_digest("boundary-observation"),
        boundary_reason="unresolved_chance",
    )
    ledger.record_simulator_transition(terminal)
    ledger.record_simulator_transition(boundary)
    ledger.record_frozen_policy_prior_batch(2)
    # The boundary classification overlaps one of the two neural-scored
    # leaves.  It must not add another result/leaf evaluation by itself.
    ledger.record_frozen_leaf_batch(2, 2, 2)
    ledger.complete_requested_expansion()
    ledger.complete_requested_expansion()
    receipt = ledger.finalize(
        requested_expansions=2,
        turn_planner_wall_seconds=0.25,
        max_single_action_planner_wall_seconds=0.1,
        deadline_hit=False,
        tree_incomplete_reason=None,
        direct_fallback_used=False,
        shadow_direct_action=True,
    )
    assert receipt["result_or_leaf_evaluations_seen"] == 3
    assert receipt["terminal_exact_results_seen"] == 1
    assert receipt["neural_leaf_evaluations_seen"] == 2
    assert receipt["boundary_leaf_results_seen"] == 1
    assert receipt["result_or_leaf_evaluations_seen"] == (
        receipt["terminal_exact_results_seen"]
        + receipt["neural_leaf_evaluations_seen"]
    )
    assert receipt["batched_frozen_outcome_value_leaf_reranking_batches"] == 1
    assert receipt["nonterminal_leaves_reranked"] == 2
    assert receipt["requested_tree_fully_expanded_and_backed_up_within_budget"] is True
    assert receipt["shadow_direct_action"] is True
    assert receipt["direct_fallback_used"] is False

    incomplete = R207TurnSearchLedger(
        planner_turn_id="r207-turn-0-4",
        seat=0,
        turn_key=(0, 4),
        selected_action_and_legality_fingerprint=_digest("selected-legal-2"),
        tree_and_config_sha256=_digest("tree-config-2"),
    )
    incomplete.complete_requested_expansion()
    fallback = incomplete.finalize(
        requested_expansions=2,
        turn_planner_wall_seconds=5.0,
        max_single_action_planner_wall_seconds=5.0,
        deadline_hit=True,
        tree_incomplete_reason="action_deadline",
        direct_fallback_used=True,
        shadow_direct_action=False,
    )
    assert fallback["requested_tree_fully_expanded_and_backed_up_within_budget"] is False
    assert fallback["direct_fallback_used"] is True


def test_frozen_leaf_batch_requires_one_outcome_and_value_per_leaf() -> None:
    ledger = R207TurnSearchLedger(
        planner_turn_id="r207-turn-1-0",
        seat=1,
        turn_key=(1, 0),
        selected_action_and_legality_fingerprint=_digest("selected-legal-3"),
        tree_and_config_sha256=_digest("tree-config-3"),
    )
    with pytest.raises(R207SimulatorArenaError, match="one outcome and value"):
        ledger.record_frozen_leaf_batch(2, 1, 2)


def test_public_observation_fingerprint_rejects_noncanonical_values() -> None:
    with pytest.raises(R207SimulatorArenaError, match="canonical JSON"):
        public_observation_sha256({"not-json": {1, 2, 3}})
