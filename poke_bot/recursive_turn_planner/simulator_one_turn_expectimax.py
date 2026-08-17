"""Simulator-backed, frozen-policy inter-turn MCTS for the r207 experiment.

The name is intentionally about the *one real action* returned by each call.
Internally this module can explore multiple same-seat simulator decisions before
that action is dispatched.  It is an offline, injected-component core only:
there is no agent bridge, serving import, selector mutation, engine mutation,
or remote-work control here.

The two external ABIs are deliberately reused rather than copied:

* :mod:`r207_simulator_arena` supplies opaque handles, exact classified
  successors, and one absolute turn/action deadline controller;
* :mod:`neural_leaf_reranker` supplies frozen policy priors plus calibrated
  outcome/value leaf scores in batches.

Every direct policy action is a mandatory legal MCTS candidate.  A missing
public successor/legality/packet, an incomplete requested simulation profile,
or either deadline discards the whole partial recommendation and returns that
exact direct action.  Finite chance explores every opaque outcome; terminal
only chance subtrees retain exact :class:`~fractions.Fraction` backup.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Protocol, TypeAlias, runtime_checkable

from .chance_aware_tree import ChanceAwareSearchConfig
from .neural_leaf_reranker import (
    LeafDeadline,
    LeafEvaluation,
    LeafKind,
    LeafRequest,
    LeafRerankerResult,
)
from .r207_simulator_arena import (
    AbsolutePlannerDeadlineController,
    ExactChanceOutcome,
    OpaqueMidgameHandle,
    PlannerActionDeadline,
    PlannerDeadlineExceeded,
    SuccessorArena,
    SuccessorTransition,
    TransitionKind,
    controlled_successor_or_boundary,
    public_observation_sha256,
)

Action: TypeAlias = tuple[int, ...]
TurnKey: TypeAlias = tuple[int, int]
SearchValue: TypeAlias = Fraction | float

_SHA256_PREFIX = "sha256:"
_PROFILE_SCHEMA = "poke_bot.r207_simulator_inter_turn_mcts_profile/v1"
_VIEW_SCHEMA = "poke_bot.r207_simulator_mcts_policy_decision_view/v1"
_TREE_SCHEMA = "poke_bot.r207_simulator_inter_turn_mcts_tree/v1"
_SELECTED_SCHEMA = "poke_bot.r207_selected_action_legality/v1"


class SimulatorMCTSError(RuntimeError):
    """The typed MCTS ABI cannot make a verified non-direct recommendation."""


class _IncompleteTree(SimulatorMCTSError):
    """A fail-closed private control flow carrying a receipt-worthy reason."""

    def __init__(self, reason: str, *, deadline_hit: bool = False) -> None:
        self.reason = reason
        self.deadline_hit = deadline_hit
        super().__init__(reason)


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SimulatorMCTSError("MCTS identity payload is not canonical JSON") from exc
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        raise SimulatorMCTSError(f"{label} must be a canonical sha256 digest")
    suffix = value[len(_SHA256_PREFIX) :]
    if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
        raise SimulatorMCTSError(f"{label} must be a lowercase sha256 digest")
    return value


def _normalise_action(value: Sequence[int] | Action, *, label: str) -> Action:
    if isinstance(value, (str, bytes)):
        raise SimulatorMCTSError(f"{label} must be a sequence of exact integers")
    try:
        action = tuple(value)
    except TypeError as exc:
        raise SimulatorMCTSError(f"{label} must be a sequence of exact integers") from exc
    if any(type(part) is not int or part < 0 for part in action):
        raise SimulatorMCTSError(f"{label} members must be non-negative exact integers")
    if len(set(action)) != len(action):
        raise SimulatorMCTSError(f"{label} must not repeat an option")
    return action


def _normalise_actions(value: tuple[Action, ...], *, label: str) -> tuple[Action, ...]:
    if not isinstance(value, tuple) or not value:
        raise SimulatorMCTSError(f"{label} must be a non-empty tuple")
    result = tuple(
        _normalise_action(action, label=f"{label}[{index}]")
        for index, action in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise SimulatorMCTSError(f"{label} must not contain duplicates")
    return result


def _require_turn_key(value: TurnKey) -> TurnKey:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(part) is not int or part < 0 for part in value)
    ):
        raise SimulatorMCTSError("turn_key must be a pair of non-negative exact ints")
    return value


def _legality_fingerprint(actions: tuple[Action, ...]) -> str:
    return _canonical_sha256(
        {
            "schema": "poke_bot.complete_ordered_action_fingerprint/v1",
            "actions": [list(action) for action in actions],
        }
    )


def _value_payload(value: SearchValue) -> object:
    if isinstance(value, Fraction):
        return {"exact_fraction": [value.numerator, value.denominator]}
    return {"float_hex": float(value).hex()}


def _chance_backup(weighted: Sequence[tuple[Fraction, SearchValue]]) -> SearchValue:
    """Exact rational backup, retaining Fraction arithmetic where possible."""

    if not weighted:
        raise SimulatorMCTSError("finite chance has no outcomes")
    if all(isinstance(value, Fraction) for _probability, value in weighted):
        return sum(
            (probability * value for probability, value in weighted),  # type: ignore[operator]
            Fraction(),
        )
    # Neural leaves are finite IEEE values.  Their probabilities are still
    # exact Fractions and ordering is fixed by the receipt-bound outcome order.
    return math.fsum(float(probability) * float(value) for probability, value in weighted)


def _terminal_value_from_winner(result: int, root_seat: int) -> Fraction:
    """Default exact terminal adapter for engines reporting a winning seat.

    An evaluation integration can inject a different adapter for a rules ABI
    whose terminal integer has an explicitly receipted draw convention.
    """

    if type(result) is not int or result < 0:
        raise SimulatorMCTSError("terminal simulator result must be non-negative")
    if type(root_seat) is not int or root_seat not in {0, 1}:
        raise SimulatorMCTSError("root_seat must be 0 or 1")
    if result == root_seat:
        return Fraction(1)
    if result in {0, 1}:
        return Fraction(-1)
    # A third explicit engine result can represent a draw only through the
    # caller's separately bound terminal adapter; the conservative default is
    # neutral rather than relabelling it as a win/loss.
    return Fraction(0)


def _public_terminal_result(observation: Mapping[str, Any]) -> int | None:
    """Read an explicitly public terminal result without inventing a world.

    Native ``SuccessorTransition`` edges already carry terminal results.  An
    exact finite-chance outcome carries an opaque child handle instead, so its
    public observation is the only truthful place to discover an immediately
    terminal child.  ``-1`` is the established public non-terminal sentinel.
    Unknown schemas deliberately return ``None`` and remain neural leaves.
    """

    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None
    result = current.get("result")
    if type(result) is int and result >= 0:
        return result
    return None


@dataclass(frozen=True, slots=True)
class MCTSExpansionProfile:
    """A bounded requested-simulation profile, distinct from 20s/5s config.

    ``ChanceAwareSearchConfig`` remains the sole timing/action-cap identity.
    This profile only says which finite MCTS work was requested; a result is
    complete only after every requested simulation reaches a backed-up leaf.
    """

    requested_simulations: int = 16
    max_decision_depth: int = 2
    max_tree_nodes: int = 128
    puct_exploration: float = 1.25

    def __post_init__(self) -> None:
        for label, value, maximum in (
            ("requested_simulations", self.requested_simulations, 4096),
            ("max_decision_depth", self.max_decision_depth, 64),
            ("max_tree_nodes", self.max_tree_nodes, 16384),
        ):
            if type(value) is not int or value < 1 or value > maximum:
                raise SimulatorMCTSError(
                    f"{label} must be an exact int in [1, {maximum}]"
                )
        if type(self.puct_exploration) not in {int, float} or not math.isfinite(
            float(self.puct_exploration)
        ):
            raise SimulatorMCTSError("puct_exploration must be finite")
        if float(self.puct_exploration) < 0.0:
            raise SimulatorMCTSError("puct_exploration must be non-negative")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": _PROFILE_SCHEMA,
                "requested_simulations": self.requested_simulations,
                "max_decision_depth": self.max_decision_depth,
                "max_tree_nodes": self.max_tree_nodes,
                "puct_exploration": float(self.puct_exploration),
            }
        )


@dataclass(frozen=True, slots=True)
class PolicyDecisionView:
    """One policy-visible simulator decision with exact future legality.

    This is the adapter boundary between the opaque r207 arena and the frozen
    reranker.  It does not duplicate successor types: every edge remains a
    :class:`SuccessorTransition` from ``r207_simulator_arena``.
    """

    handle: OpaqueMidgameHandle
    public_observation_sha256: str
    turn_key: TurnKey
    decision_serial: int
    acting_seat: int
    legal_actions: tuple[Action, ...]
    option_encoding_sha256: str
    direct_action: Action
    packet: object
    future_legality_receipt_sha256: str
    simulator_result_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.handle, OpaqueMidgameHandle):
            raise SimulatorMCTSError("policy decision needs an opaque arena handle")
        _require_digest(self.public_observation_sha256, label="public observation")
        if self.public_observation_sha256 != self.handle.public_observation_sha256:
            raise SimulatorMCTSError("view observation does not match opaque handle")
        object.__setattr__(self, "turn_key", _require_turn_key(self.turn_key))
        if type(self.decision_serial) is not int or self.decision_serial < 0:
            raise SimulatorMCTSError("decision_serial must be a non-negative exact int")
        if type(self.acting_seat) is not int or self.acting_seat not in {0, 1}:
            raise SimulatorMCTSError("acting_seat must be 0 or 1")
        actions = _normalise_actions(self.legal_actions, label="legal_actions")
        object.__setattr__(self, "legal_actions", actions)
        for label in (
            "option_encoding_sha256",
            "future_legality_receipt_sha256",
            "simulator_result_sha256",
        ):
            _require_digest(getattr(self, label), label=label)
        direct = _normalise_action(self.direct_action, label="direct_action")
        object.__setattr__(self, "direct_action", direct)
        if direct not in actions:
            raise SimulatorMCTSError(
                "the exact direct-policy action must be in complete legal actions"
            )
        if self.packet is None:
            raise SimulatorMCTSError("policy-visible decision requires a frozen-model packet")

    @property
    def legality_fingerprint_sha256(self) -> str:
        return _legality_fingerprint(self.legal_actions)

    @property
    def identity_sha256(self) -> str:
        # Deliberately excludes opaque handle ID: an independently captured real
        # child can reuse a deterministic subtree only when all public inputs,
        # legality, encoding, direct action, and receipt identity match.
        return _canonical_sha256(
            {
                "schema": _VIEW_SCHEMA,
                "public_observation_sha256": self.public_observation_sha256,
                "turn_key": list(self.turn_key),
                "decision_serial": self.decision_serial,
                "acting_seat": self.acting_seat,
                "legal_actions_sha256": self.legality_fingerprint_sha256,
                "option_encoding_sha256": self.option_encoding_sha256,
                "direct_action": list(self.direct_action),
                "future_legality_receipt_sha256": self.future_legality_receipt_sha256,
                "simulator_result_sha256": self.simulator_result_sha256,
            }
        )

    def leaf_request(self, *, request_id: str, root_seat: int, kind: LeafKind) -> LeafRequest:
        return LeafRequest(
            request_id=request_id,
            kind=kind,
            simulator_result_sha256=self.simulator_result_sha256,
            public_state_sha256=self.public_observation_sha256,
            root_seat=root_seat,
            packet=self.packet,
            expected_actions=self.legal_actions,
            direct_action=self.direct_action,
        )


@runtime_checkable
class PolicyDecisionFactory(Protocol):
    """Build exact policy-visible views without exposing native simulator state."""

    def build_decision(
        self,
        *,
        handle: OpaqueMidgameHandle,
        observation: Mapping[str, Any],
        root_seat: int,
        transition: SuccessorTransition | None,
        deadline: PlannerActionDeadline,
    ) -> PolicyDecisionView:
        """Build a view from a public root or certified deterministic/chance child."""

    def build_boundary_leaf(
        self,
        *,
        parent: PolicyDecisionView,
        transition: SuccessorTransition,
        root_seat: int,
        deadline: PlannerActionDeadline,
    ) -> PolicyDecisionView | None:
        """Return a public packet for a boundary, or ``None`` to fail closed."""


@runtime_checkable
class FrozenLeafReranker(Protocol):
    """The frozen r205 batch ABI used by this search core.

    Production supplies :class:`BatchedNeuralLeafReranker`; the structural
    protocol permits hermetic fake rerankers without a competing leaf type
    system.
    """

    def evaluate(
        self,
        requests: Sequence[LeafRequest],
        *,
        deadline: LeafDeadline,
    ) -> LeafRerankerResult:
        ...


class R207ArenaAdapter:
    """Thin native-ABI adapter; it never invents an arena state or transition."""

    def __init__(self, arena: SuccessorArena, factory: PolicyDecisionFactory) -> None:
        if not isinstance(arena, SuccessorArena):
            raise SimulatorMCTSError("arena does not implement r207 SuccessorArena")
        if not isinstance(factory, PolicyDecisionFactory):
            raise SimulatorMCTSError("factory does not implement PolicyDecisionFactory")
        self.arena = arena
        self.factory = factory

    def capture_root(
        self, *, root_seat: int, deadline: PlannerActionDeadline
    ) -> PolicyDecisionView:
        try:
            deadline.check("capture_root_before")
            handle, observation = self.arena.capture_root(deadline)
            deadline.check("capture_root_after")
            if public_observation_sha256(observation) != handle.public_observation_sha256:
                raise SimulatorMCTSError(
                    "captured public observation does not match opaque handle"
                )
            view = self.factory.build_decision(
                handle=handle,
                observation=observation,
                root_seat=root_seat,
                transition=None,
                deadline=deadline,
            )
            deadline.check("capture_root_view_after")
            return view
        except PlannerDeadlineExceeded as exc:
            raise _IncompleteTree("deadline_capture_root", deadline_hit=True) from exc

    def deterministic_child(
        self,
        transition: SuccessorTransition,
        *,
        root_seat: int,
        deadline: PlannerActionDeadline,
    ) -> PolicyDecisionView:
        if (
            transition.kind is not TransitionKind.DETERMINISTIC_PUBLIC
            or transition.child_handle is None
        ):
            raise SimulatorMCTSError("deterministic child adapter received another transition")
        try:
            deadline.check("observe_deterministic_child_before")
            observation = self.arena.observe(transition.child_handle, deadline)
            deadline.check("observe_deterministic_child_after")
            if public_observation_sha256(observation) != transition.public_observation_sha256:
                raise SimulatorMCTSError(
                    "observed deterministic child differs from transition observation"
                )
            view = self.factory.build_decision(
                handle=transition.child_handle,
                observation=observation,
                root_seat=root_seat,
                transition=transition,
                deadline=deadline,
            )
            deadline.check("observe_deterministic_child_view_after")
        except PlannerDeadlineExceeded as exc:
            raise _IncompleteTree(
                "deadline_observe_deterministic_child", deadline_hit=True
            ) from exc
        if transition.public_observation_sha256 != view.public_observation_sha256:
            raise SimulatorMCTSError("child view does not match transition observation")
        if transition.next_turn_key != view.turn_key:
            raise SimulatorMCTSError("child view turn key differs from transition")
        if transition.next_actor_seat != view.acting_seat:
            raise SimulatorMCTSError("child view actor differs from transition")
        return view

    def chance_child(
        self,
        outcome: ExactChanceOutcome,
        transition: SuccessorTransition,
        *,
        root_seat: int,
        deadline: PlannerActionDeadline,
    ) -> PolicyDecisionView | int:
        try:
            deadline.check("observe_exact_chance_child_before")
            observation = self.arena.observe(outcome.child_handle, deadline)
            deadline.check("observe_exact_chance_child_after")
            if public_observation_sha256(observation) != outcome.public_observation_sha256:
                raise SimulatorMCTSError(
                    "observed chance child differs from outcome observation"
                )
            terminal_result = _public_terminal_result(observation)
            if terminal_result is not None:
                return terminal_result
            view = self.factory.build_decision(
                handle=outcome.child_handle,
                observation=observation,
                root_seat=root_seat,
                transition=transition,
                deadline=deadline,
            )
            deadline.check("observe_exact_chance_child_view_after")
        except PlannerDeadlineExceeded as exc:
            raise _IncompleteTree(
                "deadline_observe_exact_chance_child", deadline_hit=True
            ) from exc
        if view.public_observation_sha256 != outcome.public_observation_sha256:
            raise SimulatorMCTSError("chance child view does not match outcome observation")
        return view

    def boundary_leaf(
        self,
        *,
        parent: PolicyDecisionView,
        transition: SuccessorTransition,
        root_seat: int,
        deadline: PlannerActionDeadline,
    ) -> PolicyDecisionView | None:
        try:
            deadline.check("build_boundary_leaf_before")
            view = self.factory.build_boundary_leaf(
                parent=parent,
                transition=transition,
                root_seat=root_seat,
                deadline=deadline,
            )
            deadline.check("build_boundary_leaf_after")
            return view
        except PlannerDeadlineExceeded as exc:
            raise _IncompleteTree("deadline_build_boundary_leaf", deadline_hit=True) from exc


@dataclass(frozen=True, slots=True)
class MCTSActionScore:
    action: Action
    prior: float
    visits: int
    mean_value: float | None
    exact_value: SearchValue | None


@dataclass(frozen=True, slots=True)
class MCTSTurnTelemetry:
    """Receipt-compatible r207 split telemetry for exactly one real action."""

    planner_turn_id: str
    seat: int
    turn_key: TurnKey
    actions_dispatched: int
    simulator_transitions_seen: int
    result_or_leaf_evaluations_seen: int
    terminal_exact_results_seen: int
    simulator_leaf_evaluations_seen: int
    neural_leaf_evaluations_seen: int
    boundary_leaf_results_seen: int
    unique_tree_nodes_seen: int
    decision_nodes_expanded: int
    finite_chance_outcomes_evaluated: int
    frozen_policy_prior_batches: int
    frozen_policy_prior_evaluations: int
    batched_frozen_outcome_value_leaf_reranking_batches: int
    frozen_outcome_leaf_evaluations: int
    frozen_value_leaf_evaluations: int
    nonterminal_leaves_reranked: int
    terminal_exact_results_not_reranked: bool
    cache_hits: int
    deterministic_subtree_reuses: int
    tree_rebuilds: int
    turn_planner_wall_seconds: float
    max_single_action_planner_wall_seconds: float
    requested_tree_fully_expanded_and_backed_up_within_budget: bool
    tree_incomplete_reason: str | None
    deadline_hit: bool
    direct_fallback_used: bool
    shadow_direct_action: bool
    selected_action_and_legality_fingerprint: str
    tree_and_config_sha256: str
    config_sha256: str
    profile_sha256: str
    requested_simulations: int
    completed_simulations: int
    cache_invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.planner_turn_id, str) or not self.planner_turn_id:
            raise SimulatorMCTSError("planner_turn_id must be non-empty")
        if type(self.seat) is not int or self.seat not in {0, 1}:
            raise SimulatorMCTSError("seat must be 0 or 1")
        object.__setattr__(self, "turn_key", _require_turn_key(self.turn_key))
        for label in (
            "actions_dispatched",
            "simulator_transitions_seen",
            "result_or_leaf_evaluations_seen",
            "terminal_exact_results_seen",
            "simulator_leaf_evaluations_seen",
            "neural_leaf_evaluations_seen",
            "boundary_leaf_results_seen",
            "unique_tree_nodes_seen",
            "decision_nodes_expanded",
            "finite_chance_outcomes_evaluated",
            "frozen_policy_prior_batches",
            "frozen_policy_prior_evaluations",
            "batched_frozen_outcome_value_leaf_reranking_batches",
            "frozen_outcome_leaf_evaluations",
            "frozen_value_leaf_evaluations",
            "nonterminal_leaves_reranked",
            "cache_hits",
            "deterministic_subtree_reuses",
            "tree_rebuilds",
            "requested_simulations",
            "completed_simulations",
        ):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise SimulatorMCTSError(f"{label} must be a non-negative exact int")
        if self.actions_dispatched != 1:
            raise SimulatorMCTSError("exactly one real atomic action must be returned")
        if self.result_or_leaf_evaluations_seen != (
            self.terminal_exact_results_seen + self.neural_leaf_evaluations_seen
        ):
            raise SimulatorMCTSError(
                "result_or_leaf_evaluations_seen must equal terminal exact plus neural leaves"
            )
        if self.simulator_leaf_evaluations_seen != self.terminal_exact_results_seen:
            raise SimulatorMCTSError(
                "simulator_leaf_evaluations_seen must equal terminal exact results"
            )
        if self.boundary_leaf_results_seen > self.neural_leaf_evaluations_seen:
            raise SimulatorMCTSError("boundary leaves must be a subset of neural leaves")
        if self.frozen_outcome_leaf_evaluations != self.neural_leaf_evaluations_seen:
            raise SimulatorMCTSError("every neural leaf must receive one outcome evaluation")
        if self.frozen_value_leaf_evaluations != self.neural_leaf_evaluations_seen:
            raise SimulatorMCTSError("every neural leaf must receive one value evaluation")
        if self.nonterminal_leaves_reranked != self.neural_leaf_evaluations_seen:
            raise SimulatorMCTSError("every neural leaf must be recorded as reranked")
        if self.frozen_policy_prior_evaluations < self.decision_nodes_expanded:
            raise SimulatorMCTSError("every expanded decision needs frozen policy priors")
        if (
            self.frozen_policy_prior_evaluations > 0
            and self.frozen_policy_prior_batches == 0
        ):
            raise SimulatorMCTSError("policy-prior evaluations require a batch")
        if (
            self.neural_leaf_evaluations_seen > 0
            and self.batched_frozen_outcome_value_leaf_reranking_batches == 0
        ):
            raise SimulatorMCTSError("neural leaves require an outcome/value batch")
        if self.terminal_exact_results_not_reranked is not True:
            raise SimulatorMCTSError("exact simulator terminal results may never be reranked")
        for label in ("turn_planner_wall_seconds", "max_single_action_planner_wall_seconds"):
            value = float(getattr(self, label))
            if not math.isfinite(value) or value < 0.0:
                raise SimulatorMCTSError(f"{label} must be finite and non-negative")
        if self.max_single_action_planner_wall_seconds > self.turn_planner_wall_seconds:
            raise SimulatorMCTSError(
                "single-action elapsed time cannot exceed elapsed real-turn time"
            )
        for label in (
            "selected_action_and_legality_fingerprint",
            "tree_and_config_sha256",
            "config_sha256",
            "profile_sha256",
        ):
            _require_digest(getattr(self, label), label=label)
        if self.requested_tree_fully_expanded_and_backed_up_within_budget:
            if self.completed_simulations != self.requested_simulations:
                raise SimulatorMCTSError("complete profile requires every requested simulation")
            if self.deadline_hit or self.tree_incomplete_reason is not None:
                raise SimulatorMCTSError("complete tree cannot have deadline/incomplete reason")
        else:
            if not isinstance(self.tree_incomplete_reason, str) or not self.tree_incomplete_reason:
                raise SimulatorMCTSError("incomplete tree requires an exact reason")
            if self.direct_fallback_used is not True:
                raise SimulatorMCTSError("incomplete tree must use the exact direct fallback")
        for label in (
            "deadline_hit",
            "direct_fallback_used",
            "shadow_direct_action",
            "terminal_exact_results_not_reranked",
        ):
            if type(getattr(self, label)) is not bool:
                raise SimulatorMCTSError(f"{label} must be an exact bool")
        if self.direct_fallback_used and not (
            self.deadline_hit or self.tree_incomplete_reason is not None
        ):
            raise SimulatorMCTSError("direct fallback requires an incomplete reason")
        if self.direct_fallback_used and self.shadow_direct_action:
            raise SimulatorMCTSError("shadow direct and fallback direct are distinct")

    def as_dict(self) -> dict[str, object]:
        return {
            field: (list(value) if field == "turn_key" else value)
            for field, value in (
                ("planner_turn_id", self.planner_turn_id),
                ("seat", self.seat),
                ("turn_key", self.turn_key),
                ("actions_dispatched", self.actions_dispatched),
                ("simulator_transitions_seen", self.simulator_transitions_seen),
                ("result_or_leaf_evaluations_seen", self.result_or_leaf_evaluations_seen),
                ("terminal_exact_results_seen", self.terminal_exact_results_seen),
                ("simulator_leaf_evaluations_seen", self.simulator_leaf_evaluations_seen),
                ("neural_leaf_evaluations_seen", self.neural_leaf_evaluations_seen),
                ("boundary_leaf_results_seen", self.boundary_leaf_results_seen),
                ("unique_tree_nodes_seen", self.unique_tree_nodes_seen),
                ("decision_nodes_expanded", self.decision_nodes_expanded),
                ("finite_chance_outcomes_evaluated", self.finite_chance_outcomes_evaluated),
                ("frozen_policy_prior_batches", self.frozen_policy_prior_batches),
                ("frozen_policy_prior_evaluations", self.frozen_policy_prior_evaluations),
                (
                    "batched_frozen_outcome_value_leaf_reranking_batches",
                    self.batched_frozen_outcome_value_leaf_reranking_batches,
                ),
                ("frozen_outcome_leaf_evaluations", self.frozen_outcome_leaf_evaluations),
                ("frozen_value_leaf_evaluations", self.frozen_value_leaf_evaluations),
                ("nonterminal_leaves_reranked", self.nonterminal_leaves_reranked),
                ("terminal_exact_results_not_reranked", self.terminal_exact_results_not_reranked),
                ("cache_hits", self.cache_hits),
                ("deterministic_subtree_reuses", self.deterministic_subtree_reuses),
                ("tree_rebuilds", self.tree_rebuilds),
                ("turn_planner_wall_seconds", self.turn_planner_wall_seconds),
                ("max_single_action_planner_wall_seconds", self.max_single_action_planner_wall_seconds),
                (
                    "requested_tree_fully_expanded_and_backed_up_within_budget",
                    self.requested_tree_fully_expanded_and_backed_up_within_budget,
                ),
                ("tree_incomplete_reason", self.tree_incomplete_reason),
                ("deadline_hit", self.deadline_hit),
                ("direct_fallback_used", self.direct_fallback_used),
                ("shadow_direct_action", self.shadow_direct_action),
                (
                    "selected_action_and_legality_fingerprint",
                    self.selected_action_and_legality_fingerprint,
                ),
                ("tree_and_config_sha256", self.tree_and_config_sha256),
                ("config_sha256", self.config_sha256),
                ("profile_sha256", self.profile_sha256),
                ("requested_simulations", self.requested_simulations),
                ("completed_simulations", self.completed_simulations),
                ("cache_invalidation_reason", self.cache_invalidation_reason),
            )
        }

    def to_bo1000_turn_telemetry(
        self,
        *,
        game_nonce_sha256: str,
        pair_id: str,
        mcts_seat: int,
        selected_action: Sequence[int],
        legal_actions: tuple[Action, ...],
    ) -> object:
        """Adapt this search receipt to the frozen BO1000 compiler ABI.

        The game envelope owns pair/game identifiers, while the search core
        owns simulator/leaf/cache/clock facts.  Keeping this narrow adapter
        explicit avoids a competing report type and makes the two telemetry
        schemas mechanically auditable.
        """

        if type(mcts_seat) is not int or mcts_seat != self.seat:
            raise SimulatorMCTSError("BO1000 MCTS seat must equal the search seat")
        action = _normalise_action(selected_action, label="BO1000 selected action")
        actions = _normalise_actions(legal_actions, label="BO1000 legal actions")
        if action not in actions:
            raise SimulatorMCTSError("BO1000 selected action must be exactly legal")
        # Import at the reporting seam, not module import time, so the search
        # core remains usable by the isolated simulator tests.
        from .bo1000_evaluation import MCTSTurnTelemetry as BO1000TurnTelemetry

        return BO1000TurnTelemetry(
            game_nonce_sha256=game_nonce_sha256,
            pair_id=pair_id,
            mcts_seat=mcts_seat,
            planner_turn_id=self.planner_turn_id,
            turn_key=self.turn_key,
            actions_dispatched=self.actions_dispatched,
            simulator_transitions_seen=self.simulator_transitions_seen,
            result_or_leaf_evaluations_seen=self.result_or_leaf_evaluations_seen,
            simulator_leaf_evaluations_seen=self.simulator_leaf_evaluations_seen,
            neural_leaf_evaluations_seen=self.neural_leaf_evaluations_seen,
            unique_tree_nodes_seen=self.unique_tree_nodes_seen,
            decision_nodes_expanded=self.decision_nodes_expanded,
            terminal_exact_results_seen=self.terminal_exact_results_seen,
            boundary_leaf_results_seen=self.boundary_leaf_results_seen,
            finite_chance_outcomes_evaluated=self.finite_chance_outcomes_evaluated,
            frozen_policy_prior_batches=self.frozen_policy_prior_batches,
            frozen_policy_prior_evaluations=self.frozen_policy_prior_evaluations,
            batched_frozen_outcome_value_leaf_reranking_batches=(
                self.batched_frozen_outcome_value_leaf_reranking_batches
            ),
            frozen_outcome_leaf_evaluations=self.frozen_outcome_leaf_evaluations,
            frozen_value_leaf_evaluations=self.frozen_value_leaf_evaluations,
            nonterminal_leaves_reranked=bool(self.neural_leaf_evaluations_seen),
            terminal_exact_results_not_reranked=self.terminal_exact_results_not_reranked,
            cache_hits=self.cache_hits,
            deterministic_subtree_reuses=self.deterministic_subtree_reuses,
            tree_rebuilds=self.tree_rebuilds,
            turn_planner_wall_seconds=self.turn_planner_wall_seconds,
            max_single_action_planner_wall_seconds=self.max_single_action_planner_wall_seconds,
            requested_tree_fully_expanded_and_backed_up_within_budget=(
                self.requested_tree_fully_expanded_and_backed_up_within_budget
            ),
            tree_incomplete_reason=self.tree_incomplete_reason,
            deadline_hit=self.deadline_hit,
            direct_fallback_used=self.direct_fallback_used,
            selected_action_legal=True,
            selected_action_sha256=_canonical_sha256(
                {
                    "schema": "poke_bot.r207_selected_action/v1",
                    "action": list(action),
                }
            ),
            legal_actions_sha256=_legality_fingerprint(actions),
            tree_sha256=self.tree_and_config_sha256,
            config_sha256=self.config_sha256,
        )


@dataclass(frozen=True, slots=True)
class MCTSTurnResult:
    """Offline-only one-real-action search result."""

    selected_action: Action
    direct_action: Action
    root_action_scores: tuple[MCTSActionScore, ...]
    root_value: SearchValue | None
    telemetry: MCTSTurnTelemetry

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_action", _normalise_action(self.selected_action, label="selected_action"))
        object.__setattr__(self, "direct_action", _normalise_action(self.direct_action, label="direct_action"))


@dataclass(slots=True)
class _Counters:
    simulator_transitions_seen: int = 0
    terminal_exact_results_seen: int = 0
    neural_leaf_evaluations_seen: int = 0
    boundary_leaf_results_seen: int = 0
    unique_tree_nodes_seen: int = 0
    decision_nodes_expanded: int = 0
    finite_chance_outcomes_evaluated: int = 0
    frozen_policy_prior_batches: int = 0
    frozen_policy_prior_evaluations: int = 0
    batched_frozen_outcome_value_leaf_reranking_batches: int = 0
    frozen_outcome_leaf_evaluations: int = 0
    frozen_value_leaf_evaluations: int = 0
    nonterminal_leaves_reranked: int = 0
    completed_simulations: int = 0


@dataclass(slots=True)
class _Edge:
    action: Action
    prior: float
    transition: SuccessorTransition | None = None
    child: _Node | None = None
    value: SearchValue | None = None
    # Finite chance is exact but not a terminal leaf: preserve every observed
    # outcome child so later requested simulations can keep searching each
    # controlled child before recomputing the exact weighted backup.
    chance_outcomes_resolved: bool = False
    chance_terminal_values: dict[str, SearchValue] = field(default_factory=dict)
    chance_children: dict[str, _Node] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    exact_value: SearchValue | None = None

    @property
    def mean_value(self) -> float | None:
        if self.visits <= 0:
            return None
        return self.value_sum / float(self.visits)


@dataclass(slots=True)
class _Node:
    view: PolicyDecisionView
    priors: tuple[float, ...] | None = None
    leaf_value: SearchValue | None = None
    evaluated: bool = False
    edges: dict[Action, _Edge] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    incoming_action: Action | None = None

    @property
    def mean_value(self) -> float | None:
        if self.visits <= 0:
            return None
        return self.value_sum / float(self.visits)


@dataclass(slots=True)
class _CacheEntry:
    expected_view_sha256: str
    node: _Node
    tree_sha256: str
    # A matching public child is not sufficient to prove that the real action
    # followed the cached deterministic edge: an exact chance outcome can have
    # the same public shape.  Reuse therefore needs an explicit post-action
    # deterministic attestation from the execution boundary.
    realized_deterministic_attested: bool = False


def _selected_action_fingerprint(view: PolicyDecisionView, action: Action) -> str:
    return _canonical_sha256(
        {
            "schema": _SELECTED_SCHEMA,
            "action": list(action),
            "legal_actions_sha256": view.legality_fingerprint_sha256,
            "option_encoding_sha256": view.option_encoding_sha256,
            "view_sha256": view.identity_sha256,
        }
    )


class SimulatorInterTurnMCTSSession:
    """Persistent offline MCTS session with exact deterministic cache reuse.

    ``plan_next`` returns one candidate action.  It keeps the 20-second clock
    in ``AbsolutePlannerDeadlineController`` across calls with the same *real*
    turn key; a 5-second action lease is nested inside that shared deadline for
    every real action.  Predicted child turn keys never reset this clock.
    """

    production_action_authority_enabled = False

    def __init__(
        self,
        *,
        adapter: R207ArenaAdapter,
        reranker: FrozenLeafReranker,
        config: ChanceAwareSearchConfig,
        profile: MCTSExpansionProfile | None = None,
        deadline_controller: AbsolutePlannerDeadlineController | None = None,
        terminal_value: Callable[[int, int], Fraction] = _terminal_value_from_winner,
    ) -> None:
        if not isinstance(adapter, R207ArenaAdapter):
            raise SimulatorMCTSError("adapter must be an R207ArenaAdapter")
        if not isinstance(reranker, FrozenLeafReranker):
            raise SimulatorMCTSError(
                "reranker must implement BatchedNeuralLeafReranker.evaluate"
            )
        if not isinstance(config, ChanceAwareSearchConfig):
            raise SimulatorMCTSError("config must be canonical ChanceAwareSearchConfig")
        profile = MCTSExpansionProfile() if profile is None else profile
        if not isinstance(profile, MCTSExpansionProfile):
            raise SimulatorMCTSError("profile must be MCTSExpansionProfile")
        if deadline_controller is not None and not isinstance(
            deadline_controller, AbsolutePlannerDeadlineController
        ):
            raise SimulatorMCTSError(
                "deadline_controller must be AbsolutePlannerDeadlineController"
            )
        if not callable(terminal_value):
            raise SimulatorMCTSError("terminal_value must be callable")
        self.adapter = adapter
        self.reranker = reranker
        self.config = config
        self.profile = profile
        self.deadlines = deadline_controller or AbsolutePlannerDeadlineController(config)
        if self.deadlines.config_sha256 != config.identity_sha256:
            raise SimulatorMCTSError("deadline controller belongs to another config")
        self._terminal_value = terminal_value
        self._cache: _CacheEntry | None = None
        self._last_cache_invalidation_reason: str | None = None

    def capture_and_plan(
        self,
        *,
        planner_turn_id: str,
        seat: int,
        turn_key: TurnKey,
    ) -> MCTSTurnResult:
        """Capture a fresh arena root and charge capture to the same action lease."""

        exact_turn_key = _require_turn_key(turn_key)
        lease = self.deadlines.begin_action(exact_turn_key)
        started_ns = lease._clock_ns()
        root = self.adapter.capture_root(root_seat=seat, deadline=lease)
        if root.turn_key != exact_turn_key:
            raise SimulatorMCTSError("captured root turn key differs from action lease")
        return self._plan_with_lease(
            planner_turn_id=planner_turn_id,
            seat=seat,
            view=root,
            lease=lease,
            started_ns=started_ns,
        )

    def observe_real_action(
        self,
        *,
        action: Sequence[int],
        next_view: PolicyDecisionView | None,
        realized_transition_kind: TransitionKind | None = None,
    ) -> None:
        """Optionally validate/invalidate a pending cache immediately.

        ``plan_next`` also performs this exact check defensively, so callers
        cannot gain cache reuse merely by omitting this notification.
        """

        exact_action = _normalise_action(action, label="real action")
        entry = self._cache
        if entry is None:
            return
        if realized_transition_kind is not TransitionKind.DETERMINISTIC_PUBLIC:
            self._cache = None
            self._last_cache_invalidation_reason = (
                "missing_realized_deterministic_attestation"
                if realized_transition_kind is None
                else "realized_non_deterministic_transition"
            )
            return
        expected_action = entry.node.incoming_action
        if expected_action is not None and exact_action != expected_action:
            self._cache = None
            self._last_cache_invalidation_reason = "real_action_differs_from_cached_edge"
            return
        if next_view is None or next_view.identity_sha256 != entry.expected_view_sha256:
            self._cache = None
            self._last_cache_invalidation_reason = "real_child_fingerprint_mismatch"
            return
        entry.realized_deterministic_attested = True

    def plan_next(
        self,
        *,
        planner_turn_id: str,
        seat: int,
        real_view: PolicyDecisionView,
    ) -> MCTSTurnResult:
        """Plan at one fresh real policy-visible decision or reuse a matching child."""

        lease = self.deadlines.begin_action(real_view.turn_key)
        return self._plan_with_lease(
            planner_turn_id=planner_turn_id,
            seat=seat,
            view=real_view,
            lease=lease,
            started_ns=lease._clock_ns(),
        )

    def _plan_with_lease(
        self,
        *,
        planner_turn_id: str,
        seat: int,
        view: PolicyDecisionView,
        lease: PlannerActionDeadline,
        started_ns: int | None = None,
    ) -> MCTSTurnResult:
        if not isinstance(planner_turn_id, str) or not planner_turn_id:
            raise SimulatorMCTSError("planner_turn_id must be non-empty")
        started_ns = lease._clock_ns() if started_ns is None else started_ns
        if type(seat) is not int or seat not in {0, 1}:
            raise SimulatorMCTSError("seat must be 0 or 1")
        if view.acting_seat != seat:
            return self._direct_failure(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                counters=_Counters(),
                reason="real_view_actor_is_not_controlled_seat",
                deadline_hit=False,
                cache_hits=0,
                deterministic_reuses=0,
                tree_rebuilds=1,
                root=None,
                started_ns=started_ns,
            )

        cache = self._cache
        if (
            cache is not None
            and cache.realized_deterministic_attested
            and cache.expected_view_sha256 == view.identity_sha256
        ):
            # Substitute the fresh actual opaque handle only after every public
            # fingerprint matched.  The old scratch handle is never reused.
            cache.node.view = view
            self._cache = None
            self._last_cache_invalidation_reason = None
            return self._cached_result(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                root=cache.node,
                tree_sha256=cache.tree_sha256,
            )

        invalidation = self._last_cache_invalidation_reason
        if cache is not None:
            invalidation = invalidation or (
                "missing_realized_deterministic_attestation"
                if not cache.realized_deterministic_attested
                else "cached_child_fingerprint_mismatch"
            )
        self._cache = None
        self._last_cache_invalidation_reason = None
        counters = _Counters()
        root = _Node(view=view)
        counters.unique_tree_nodes_seen = 1
        try:
            lease.check("mcts_root_start")
            for simulation_index in range(self.profile.requested_simulations):
                self._simulate(
                    root,
                    depth=0,
                    root_seat=seat,
                    lease=lease,
                    counters=counters,
                    force_root_direct=simulation_index == 0,
                )
                counters.completed_simulations += 1
                lease.check("mcts_after_simulation_backup")
            lease.check("mcts_before_root_selection")
            selected = self._select_root_action(root)
            lease.check("mcts_after_root_selection")
            tree_sha256 = self._tree_sha256(root)
            lease.check("mcts_after_tree_digest")
            self._install_cache_for_selected(root, selected, tree_sha256)
            lease.check("mcts_after_cache_install")
            return self._complete_result(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                started_ns=started_ns,
                counters=counters,
                root=root,
                selected_action=selected,
                tree_sha256=tree_sha256,
                cache_invalidation_reason=invalidation,
            )
        except _IncompleteTree as exc:
            return self._direct_failure(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                counters=counters,
                reason=exc.reason,
                deadline_hit=exc.deadline_hit,
                cache_hits=0,
                deterministic_reuses=0,
                tree_rebuilds=1,
                root=root,
                started_ns=started_ns,
                cache_invalidation_reason=invalidation,
            )
        except PlannerDeadlineExceeded as exc:
            return self._direct_failure(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                counters=counters,
                reason=f"deadline_{exc.scope}_{exc.operation}",
                deadline_hit=True,
                cache_hits=0,
                deterministic_reuses=0,
                tree_rebuilds=1,
                root=root,
                started_ns=started_ns,
                cache_invalidation_reason=invalidation,
            )
        except SimulatorMCTSError as exc:
            return self._direct_failure(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                counters=counters,
                reason=f"unverified_tree:{type(exc).__name__}",
                deadline_hit=False,
                cache_hits=0,
                deterministic_reuses=0,
                tree_rebuilds=1,
                root=root,
                started_ns=started_ns,
                cache_invalidation_reason=invalidation,
            )
        except Exception as exc:  # noqa: BLE001 - injected component must fail closed.
            return self._direct_failure(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                counters=counters,
                reason=f"injected_dependency_error:{type(exc).__name__}",
                deadline_hit=False,
                cache_hits=0,
                deterministic_reuses=0,
                tree_rebuilds=1,
                root=root,
                started_ns=started_ns,
                cache_invalidation_reason=invalidation,
            )

    def _simulate(
        self,
        node: _Node,
        *,
        depth: int,
        root_seat: int,
        lease: PlannerActionDeadline,
        counters: _Counters,
        force_root_direct: bool,
    ) -> SearchValue:
        self._ensure_evaluated(
            nodes=((node, LeafKind.SUCCESSOR),),
            root_seat=root_seat,
            lease=lease,
            counters=counters,
        )
        if depth >= self.profile.max_decision_depth:
            assert node.leaf_value is not None
            self._back_up_node(node, node.leaf_value)
            return node.leaf_value
        edge = self._select_edge(node, force_direct=force_root_direct)
        value = self._edge_value(
            parent=node,
            edge=edge,
            next_depth=depth + 1,
            root_seat=root_seat,
            lease=lease,
            counters=counters,
        )
        edge.visits += 1
        edge.value_sum += float(value)
        edge.exact_value = value
        self._back_up_node(node, value)
        return value

    def _back_up_node(self, node: _Node, value: SearchValue) -> None:
        node.visits += 1
        node.value_sum += float(value)

    def _select_edge(self, node: _Node, *, force_direct: bool) -> _Edge:
        if node.priors is None:
            raise _IncompleteTree("missing_frozen_policy_priors")
        if len(node.priors) != len(node.view.legal_actions):
            raise _IncompleteTree("frozen_policy_priors_do_not_match_complete_actions")
        if force_direct:
            index = node.view.legal_actions.index(node.view.direct_action)
            return self._edge_for(node, index)
        candidates = tuple(range(len(node.view.legal_actions)))
        # Unvisited actions get a prior-only UCB.  This avoids treating a hidden
        # random tie as an expansion rule and guarantees canonical ordering.
        parent_visits = max(1, node.visits)
        best_index = max(
            candidates,
            key=lambda index: self._puct_key(node, self._edge_for(node, index), parent_visits, index),
        )
        return self._edge_for(node, best_index)

    def _puct_key(
        self, node: _Node, edge: _Edge, parent_visits: int, index: int
    ) -> tuple[float, float, int]:
        mean = edge.mean_value if edge.mean_value is not None else 0.0
        explore = float(self.profile.puct_exploration) * edge.prior * math.sqrt(
            float(parent_visits)
        ) / float(1 + edge.visits)
        # Earlier complete-action index is the final deterministic tie break.
        return (mean + explore, edge.prior, -index)

    def _edge_for(self, node: _Node, index: int) -> _Edge:
        action = node.view.legal_actions[index]
        edge = node.edges.get(action)
        if edge is None:
            if node.priors is None:
                raise _IncompleteTree("missing_node_priors")
            edge = _Edge(action=action, prior=node.priors[index])
            node.edges[action] = edge
        return edge

    def _edge_value(
        self,
        *,
        parent: _Node,
        edge: _Edge,
        next_depth: int,
        root_seat: int,
        lease: PlannerActionDeadline,
        counters: _Counters,
    ) -> SearchValue:
        if edge.transition is None:
            transition = self._expand(parent, edge.action, lease=lease, counters=counters)
            # A deterministic simulator result that hands control to the
            # opponent is not a licence to call an opponent policy or to carry
            # an implicit determinization.  Reclassify it through the frozen
            # r207 helper before any child view is requested.
            edge.transition = controlled_successor_or_boundary(
                transition, controlled_seat=root_seat
            )
        transition = edge.transition
        assert transition is not None
        # Exact terminal/boundary/chance resolution is immutable for this
        # simulator edge.  Replaying it merely to inflate telemetry would make
        # one logical leaf look like several simulator results.
        if (
            edge.value is not None
            and transition.kind
            in {
                TransitionKind.TERMINAL,
                TransitionKind.INFORMATION_BOUNDARY,
            }
        ):
            return edge.value
        if transition.kind is TransitionKind.TERMINAL:
            assert transition.terminal_result is not None
            value = self._terminal_value(transition.terminal_result, root_seat)
            if not isinstance(value, Fraction) or value < -1 or value > 1:
                raise _IncompleteTree("terminal_value_adapter_returned_invalid_value")
            counters.terminal_exact_results_seen += 1
            edge.value = value
            return value
        if transition.kind is TransitionKind.INFORMATION_BOUNDARY:
            value = self._boundary_value(
                parent=parent,
                transition=transition,
                root_seat=root_seat,
                lease=lease,
                counters=counters,
            )
            edge.value = value
            return value
        if transition.kind is TransitionKind.DETERMINISTIC_PUBLIC:
            child = edge.child
            if child is None:
                child_view = self.adapter.deterministic_child(
                    transition, root_seat=root_seat, deadline=lease
                )
                if child_view.acting_seat != root_seat:
                    # It remains a public child packet but can be value-scored
                    # only as a boundary; no opponent policy call is permitted.
                    value = self._evaluate_boundary_view(
                        child_view,
                        root_seat=root_seat,
                        lease=lease,
                        counters=counters,
                    )
                    edge.value = value
                    return value
                child = self._new_child(child_view, counters)
                edge.child = child
            if next_depth >= self.profile.max_decision_depth:
                self._ensure_evaluated(
                    nodes=((child, LeafKind.SUCCESSOR),),
                    root_seat=root_seat,
                    lease=lease,
                    counters=counters,
                )
                assert child.leaf_value is not None
                edge.value = child.leaf_value
                return child.leaf_value
            value = self._simulate(
                child,
                depth=next_depth,
                root_seat=root_seat,
                lease=lease,
                counters=counters,
                force_root_direct=False,
            )
            edge.value = value
            return value
        value = self._finite_chance_value(
            parent=parent,
            edge=edge,
            transition=transition,
            next_depth=next_depth,
            root_seat=root_seat,
            lease=lease,
            counters=counters,
        )
        return value

    def _expand(
        self,
        parent: _Node,
        action: Action,
        *,
        lease: PlannerActionDeadline,
        counters: _Counters,
    ) -> SuccessorTransition:
        try:
            lease.check("mcts_expand_before")
            transition = self.adapter.arena.expand_action(parent.view.handle, action, lease)
            lease.check("mcts_expand_after")
        except PlannerDeadlineExceeded as exc:
            raise _IncompleteTree(
                f"deadline_{exc.scope}_{exc.operation}", deadline_hit=True
            ) from exc
        if not isinstance(transition, SuccessorTransition):
            raise _IncompleteTree("arena_returned_untyped_transition")
        counters.simulator_transitions_seen += 1
        return transition

    def _finite_chance_value(
        self,
        *,
        parent: _Node,
        edge: _Edge,
        transition: SuccessorTransition,
        next_depth: int,
        root_seat: int,
        lease: PlannerActionDeadline,
        counters: _Counters,
    ) -> SearchValue:
        if transition.kind is not TransitionKind.FINITE_PUBLIC_CHANCE:
            raise _IncompleteTree("finite_chance_dispatch_received_another_transition")
        outcomes = transition.chance_outcomes
        if not edge.chance_outcomes_resolved:
            counters.finite_chance_outcomes_evaluated += len(outcomes)
            # The classified chance edge is counted by ``_expand``.  Each exact
            # opaque outcome is then independently advanced/observed once and
            # retained as a separately auditable simulator result path.
            counters.simulator_transitions_seen += len(outcomes)
            for outcome in outcomes:
                child_or_result = self.adapter.chance_child(
                    outcome,
                    transition,
                    root_seat=root_seat,
                    deadline=lease,
                )
                if type(child_or_result) is int:
                    value = self._terminal_value(child_or_result, root_seat)
                    if not isinstance(value, Fraction) or value < -1 or value > 1:
                        raise _IncompleteTree("terminal_value_adapter_returned_invalid_value")
                    counters.terminal_exact_results_seen += 1
                    edge.chance_terminal_values[outcome.label] = value
                    continue
                edge.chance_children[outcome.label] = self._new_child(
                    child_or_result, counters
                )
            edge.chance_outcomes_resolved = True

        children: list[tuple[ExactChanceOutcome, _Node]] = []
        backed_up: list[tuple[Fraction, SearchValue]] = []
        for outcome in outcomes:
            terminal_value = edge.chance_terminal_values.get(outcome.label)
            child = edge.chance_children.get(outcome.label)
            if terminal_value is not None and child is not None:
                raise _IncompleteTree("chance_outcome_cache_has_conflicting_child")
            if terminal_value is not None:
                backed_up.append((outcome.probability, terminal_value))
                continue
            if child is None:
                raise _IncompleteTree("chance_outcome_cache_is_incomplete")
            children.append((outcome, child))

        # The frozen reranker is called once for every currently unresolved
        # outcome frontier, so exact chance never degenerates into sequential
        # sampled model calls.
        if children:
            self._ensure_evaluated(
                nodes=tuple(
                    (
                        child,
                        (
                            LeafKind.SUCCESSOR
                            if child.view.acting_seat == root_seat
                            else LeafKind.BOUNDARY
                        ),
                    )
                    for _outcome, child in children
                ),
                root_seat=root_seat,
                lease=lease,
                counters=counters,
            )
        for outcome, child in children:
            if child.view.acting_seat != root_seat:
                # The outcome was already batch-scored as a policy-visible
                # boundary above.  Boundary is an overlapping classification,
                # never a second neural leaf or a second boundary result when
                # this exact chance child is reused by a later simulation.
                assert child.leaf_value is not None
                value = child.leaf_value
            elif next_depth >= self.profile.max_decision_depth:
                assert child.leaf_value is not None
                value = child.leaf_value
            else:
                value = self._simulate(
                    child,
                    depth=next_depth,
                    root_seat=root_seat,
                    lease=lease,
                    counters=counters,
                    force_root_direct=False,
                )
            backed_up.append((outcome.probability, value))
        return _chance_backup(tuple(backed_up))

    def _boundary_value(
        self,
        *,
        parent: _Node,
        transition: SuccessorTransition,
        root_seat: int,
        lease: PlannerActionDeadline,
        counters: _Counters,
    ) -> SearchValue:
        boundary = self.adapter.boundary_leaf(
            parent=parent.view,
            transition=transition,
            root_seat=root_seat,
            deadline=lease,
        )
        if boundary is None:
            raise _IncompleteTree("boundary_has_no_policy_visible_frozen_leaf")
        if boundary.public_observation_sha256 != transition.public_observation_sha256:
            raise _IncompleteTree("boundary_leaf_does_not_match_transition_observation")
        return self._evaluate_boundary_view(
            boundary,
            root_seat=root_seat,
            lease=lease,
            counters=counters,
        )

    def _evaluate_boundary_view(
        self,
        view: PolicyDecisionView,
        *,
        root_seat: int,
        lease: PlannerActionDeadline,
        counters: _Counters,
    ) -> SearchValue:
        node = self._new_child(view, counters)
        self._ensure_evaluated(
            nodes=((node, LeafKind.BOUNDARY),),
            root_seat=root_seat,
            lease=lease,
            counters=counters,
        )
        assert node.leaf_value is not None
        return node.leaf_value

    def _new_child(self, view: PolicyDecisionView, counters: _Counters) -> _Node:
        if counters.unique_tree_nodes_seen >= self.profile.max_tree_nodes:
            raise _IncompleteTree("requested_profile_tree_node_cap_exhausted")
        counters.unique_tree_nodes_seen += 1
        return _Node(view=view)

    def _ensure_evaluated(
        self,
        *,
        nodes: Sequence[tuple[_Node, LeafKind]],
        root_seat: int,
        lease: PlannerActionDeadline,
        counters: _Counters,
    ) -> None:
        pending = tuple((node, kind) for node, kind in nodes if not node.evaluated)
        if not pending:
            return
        if any(
            len(node.view.legal_actions) > self.config.max_complete_actions
            for node, _kind in pending
        ):
            raise _IncompleteTree("complete_action_cap_exceeded")
        requests = tuple(
            node.view.leaf_request(
                request_id=f"leaf-{node.view.identity_sha256[7:23]}-{index}",
                root_seat=root_seat,
                kind=kind,
            )
            for index, (node, kind) in enumerate(pending)
        )
        if len({request.request_id for request in requests}) != len(requests):
            raise _IncompleteTree("generated_duplicate_leaf_request_id")
        deadline = LeafDeadline(
            turn_deadline_monotonic=lease.turn_deadline_ns / 1_000_000_000.0,
            action_deadline_monotonic=lease.action_deadline_ns / 1_000_000_000.0,
        )
        try:
            lease.check("frozen_leaf_batch_before")
            result = self.reranker.evaluate(requests, deadline=deadline)
            lease.check("frozen_leaf_batch_after")
        except PlannerDeadlineExceeded as exc:
            raise _IncompleteTree(
                f"deadline_{exc.scope}_{exc.operation}", deadline_hit=True
            ) from exc
        except Exception as exc:
            raise _IncompleteTree(f"frozen_leaf_reranker_error:{type(exc).__name__}") from exc
        self._accept_leaf_result(
            result,
            requests=requests,
            pending=pending,
            counters=counters,
        )

    def _accept_leaf_result(
        self,
        result: LeafRerankerResult,
        *,
        requests: tuple[LeafRequest, ...],
        pending: tuple[tuple[_Node, LeafKind], ...],
        counters: _Counters,
    ) -> None:
        if not isinstance(result, LeafRerankerResult):
            raise _IncompleteTree("reranker_returned_untyped_result")
        telemetry = result.telemetry
        if not telemetry.requested_leaf_batch_completed or telemetry.deadline_hit:
            raise _IncompleteTree(
                telemetry.incomplete_reason or "incomplete_frozen_leaf_batch",
                deadline_hit=telemetry.deadline_hit,
            )
        if not telemetry.every_leaf_has_search_eligible_value:
            raise _IncompleteTree("uncalibrated_frozen_leaf_value")
        if len(result.evaluations) != len(requests):
            raise _IncompleteTree("frozen_leaf_result_count_mismatch")
        expected_ids = tuple(request.request_id for request in requests)
        observed_ids = tuple(evaluation.request_id for evaluation in result.evaluations)
        if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
            raise _IncompleteTree("frozen_leaf_result_order_or_identity_mismatch")
        if telemetry.frozen_value_leaf_evaluations != len(requests):
            raise _IncompleteTree("frozen_leaf_value_count_mismatch")
        if telemetry.frozen_outcome_leaf_evaluations != len(requests):
            raise _IncompleteTree("frozen_leaf_outcome_count_mismatch")
        if telemetry.nonterminal_leaves_reranked != len(requests):
            raise _IncompleteTree("frozen_leaf_reranked_count_mismatch")
        if telemetry.frozen_policy_prior_evaluations != len(requests):
            raise _IncompleteTree("frozen_leaf_prior_count_mismatch")
        if telemetry.frozen_policy_prior_batches < 1:
            raise _IncompleteTree("frozen_leaf_prior_batch_missing")
        counters.frozen_policy_prior_batches += telemetry.frozen_policy_prior_batches
        counters.frozen_policy_prior_evaluations += telemetry.frozen_policy_prior_evaluations
        counters.batched_frozen_outcome_value_leaf_reranking_batches += (
            telemetry.batched_frozen_outcome_value_leaf_reranking_batches
        )
        counters.frozen_outcome_leaf_evaluations += telemetry.frozen_outcome_leaf_evaluations
        counters.frozen_value_leaf_evaluations += telemetry.frozen_value_leaf_evaluations
        counters.nonterminal_leaves_reranked += telemetry.nonterminal_leaves_reranked
        counters.neural_leaf_evaluations_seen += len(requests)
        counters.decision_nodes_expanded += len(pending)
        counters.boundary_leaf_results_seen += sum(
            kind is LeafKind.BOUNDARY for _node, kind in pending
        )
        for (node, kind), evaluation in zip(pending, result.evaluations):
            self._accept_evaluation(node, kind, evaluation)

    def _accept_evaluation(
        self, node: _Node, kind: LeafKind, evaluation: LeafEvaluation
    ) -> None:
        if evaluation.kind is not kind:
            raise _IncompleteTree("frozen_leaf_kind_mismatch")
        if evaluation.public_state_sha256 != node.view.public_observation_sha256:
            raise _IncompleteTree("frozen_leaf_public_state_mismatch")
        if evaluation.simulator_result_sha256 != node.view.simulator_result_sha256:
            raise _IncompleteTree("frozen_leaf_simulator_result_mismatch")
        if not evaluation.search_value_eligible or evaluation.selected_leaf_value is None:
            raise _IncompleteTree("frozen_leaf_has_no_calibrated_value")
        priors = evaluation.root_policy_priors
        if priors is None or priors.actions != node.view.legal_actions:
            raise _IncompleteTree("frozen_leaf_priors_do_not_match_complete_actions")
        if priors.direct_action_index != node.view.legal_actions.index(node.view.direct_action):
            raise _IncompleteTree("frozen_leaf_direct_action_mismatch")
        if len(priors.probabilities) != len(node.view.legal_actions):
            raise _IncompleteTree("frozen_leaf_prior_count_mismatch")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in priors.probabilities):
            raise _IncompleteTree("frozen_leaf_prior_is_invalid")
        if not math.isclose(math.fsum(priors.probabilities), 1.0, abs_tol=1e-9):
            raise _IncompleteTree("frozen_leaf_prior_mass_mismatch")
        node.priors = tuple(float(value) for value in priors.probabilities)
        node.leaf_value = float(evaluation.selected_leaf_value)
        node.evaluated = True

    def _select_root_action(self, root: _Node) -> Action:
        if not root.edges:
            raise _IncompleteTree("requested_profile_never_expanded_a_root_action")
        ordered = tuple(root.view.legal_actions)
        return max(
            ordered,
            key=lambda action: self._root_action_key(root, root.edges.get(action), ordered.index(action)),
        )

    def _root_action_key(
        self, root: _Node, edge: _Edge | None, index: int
    ) -> tuple[float, int, float, int]:
        if edge is None or edge.mean_value is None:
            return (float("-inf"), 0, root.priors[index] if root.priors else 0.0, -index)
        return (edge.mean_value, edge.visits, edge.prior, -index)

    def _root_scores(self, root: _Node) -> tuple[MCTSActionScore, ...]:
        priors = root.priors or tuple(0.0 for _ in root.view.legal_actions)
        return tuple(
            MCTSActionScore(
                action=action,
                prior=priors[index],
                visits=0 if (edge := root.edges.get(action)) is None else edge.visits,
                mean_value=None if edge is None else edge.mean_value,
                exact_value=None if edge is None else edge.exact_value,
            )
            for index, action in enumerate(root.view.legal_actions)
        )

    def _tree_sha256(self, root: _Node) -> str:
        return _canonical_sha256(
            {
                "schema": _TREE_SCHEMA,
                "config_sha256": self.config.identity_sha256,
                "profile_sha256": self.profile.identity_sha256,
                "root": self._node_payload(root, active=set()),
            }
        )

    def _node_payload(self, node: _Node, *, active: set[int]) -> dict[str, object]:
        identity = id(node)
        if identity in active:
            raise SimulatorMCTSError("MCTS subtree must be acyclic")
        active.add(identity)
        try:
            return {
                "view_sha256": node.view.identity_sha256,
                "visits": node.visits,
                "value_sum": float(node.value_sum).hex(),
                "edges": [
                    self._edge_payload(node.edges[action], active=active)
                    for action in node.view.legal_actions
                    if action in node.edges
                ],
            }
        finally:
            active.remove(identity)

    def _edge_payload(self, edge: _Edge, *, active: set[int]) -> dict[str, object]:
        transition = edge.transition
        return {
            "action": list(edge.action),
            "prior": float(edge.prior).hex(),
            "visits": edge.visits,
            "value_sum": float(edge.value_sum).hex(),
            "exact_value": None if edge.exact_value is None else _value_payload(edge.exact_value),
            "transition": None
            if transition is None
            else {
                "kind": transition.kind.value,
                "certificate_sha256": transition.transition_certificate_sha256,
                "public_observation_sha256": transition.public_observation_sha256,
                "boundary_reason": transition.boundary_reason,
                "terminal_result": transition.terminal_result,
                "chance_outcomes": [
                    {
                        "label": outcome.label,
                        "probability": [outcome.probability.numerator, outcome.probability.denominator],
                        "child_handle_sha256": outcome.child_handle.handle_id_sha256,
                        "public_observation_sha256": outcome.public_observation_sha256,
                    }
                    for outcome in transition.chance_outcomes
                ],
            },
            "child": None if edge.child is None else self._node_payload(edge.child, active=active),
            "chance_children": []
            if transition is None or transition.kind is not TransitionKind.FINITE_PUBLIC_CHANCE
            else [
                {
                    "label": outcome.label,
                    "terminal_value": (
                        None
                        if outcome.label not in edge.chance_terminal_values
                        else _value_payload(edge.chance_terminal_values[outcome.label])
                    ),
                    "child": (
                        None
                        if outcome.label not in edge.chance_children
                        else self._node_payload(
                            edge.chance_children[outcome.label], active=active
                        )
                    ),
                }
                for outcome in transition.chance_outcomes
            ],
        }

    def _install_cache_for_selected(
        self, root: _Node, action: Action, tree_sha256: str
    ) -> None:
        edge = root.edges.get(action)
        if (
            edge is None
            or edge.transition is None
            or edge.transition.kind is not TransitionKind.DETERMINISTIC_PUBLIC
            or edge.child is None
            or not edge.child.edges
        ):
            self._cache = None
            return
        # The child was constructed only after deterministic public transition
        # identity and same-seat actor checks.  Mark incoming action for the
        # optional explicit observation API without putting it in public digest.
        edge.child.incoming_action = action
        self._cache = _CacheEntry(
            expected_view_sha256=edge.child.view.identity_sha256,
            node=edge.child,
            tree_sha256=tree_sha256,
            realized_deterministic_attested=False,
        )

    def _cached_result(
        self,
        *,
        planner_turn_id: str,
        seat: int,
        view: PolicyDecisionView,
        lease: PlannerActionDeadline,
        root: _Node,
        tree_sha256: str,
    ) -> MCTSTurnResult:
        started_ns = lease._clock_ns()
        try:
            lease.check("cached_subtree_validation")
            selected = self._select_root_action(root)
            lease.check("cached_subtree_after_selection")
            # Keep advancing the already verified deterministic path.  Every
            # hop still needs its own real-world deterministic attestation in
            # ``observe_real_action`` before it can be consumed.
            self._install_cache_for_selected(root, selected, tree_sha256)
            lease.check("cached_subtree_after_cache_install")
            root_scores = self._root_scores(root)
            lease.check("cached_subtree_after_root_scores")
            elapsed = max(0.0, (lease._clock_ns() - started_ns) / 1_000_000_000.0)
            telemetry = self._telemetry(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                counters=_Counters(),
                elapsed=elapsed,
                complete=True,
                reason=None,
                deadline_hit=False,
                direct_fallback=False,
                shadow_direct=selected == view.direct_action,
                cache_hits=1,
                deterministic_reuses=1,
                tree_rebuilds=0,
                tree_sha256=tree_sha256,
                cache_invalidation_reason=None,
                requested_simulations=self.profile.requested_simulations,
                completed_simulations=self.profile.requested_simulations,
                selected_action=selected,
            )
            lease.check("cached_subtree_after_telemetry")
            return MCTSTurnResult(
                selected_action=selected,
                direct_action=view.direct_action,
                root_action_scores=root_scores,
                root_value=root.mean_value,
                telemetry=telemetry,
            )
        except PlannerDeadlineExceeded as exc:
            return self._direct_failure(
                planner_turn_id=planner_turn_id,
                seat=seat,
                view=view,
                lease=lease,
                counters=_Counters(),
                reason=f"deadline_{exc.scope}_{exc.operation}",
                deadline_hit=True,
                cache_hits=1,
                deterministic_reuses=0,
                tree_rebuilds=0,
                root=root,
                started_ns=started_ns,
            )

    def _complete_result(
        self,
        *,
        planner_turn_id: str,
        seat: int,
        view: PolicyDecisionView,
        lease: PlannerActionDeadline,
        started_ns: int,
        counters: _Counters,
        root: _Node,
        selected_action: Action,
        tree_sha256: str,
        cache_invalidation_reason: str | None,
    ) -> MCTSTurnResult:
        root_scores = self._root_scores(root)
        lease.check("mcts_after_root_scores")
        elapsed = max(0.0, (lease._clock_ns() - started_ns) / 1_000_000_000.0)
        telemetry = self._telemetry(
            planner_turn_id=planner_turn_id,
            seat=seat,
            view=view,
            lease=lease,
            counters=counters,
            elapsed=elapsed,
            complete=True,
            reason=None,
            deadline_hit=False,
            direct_fallback=False,
            shadow_direct=selected_action == view.direct_action,
            cache_hits=0,
            deterministic_reuses=0,
            tree_rebuilds=1,
            tree_sha256=tree_sha256,
            cache_invalidation_reason=cache_invalidation_reason,
            requested_simulations=self.profile.requested_simulations,
            completed_simulations=counters.completed_simulations,
            selected_action=selected_action,
        )
        lease.check("mcts_after_telemetry")
        return MCTSTurnResult(
            selected_action=selected_action,
            direct_action=view.direct_action,
            root_action_scores=root_scores,
            root_value=root.mean_value,
            telemetry=telemetry,
        )

    def _direct_failure(
        self,
        *,
        planner_turn_id: str,
        seat: int,
        view: PolicyDecisionView,
        lease: PlannerActionDeadline,
        counters: _Counters,
        reason: str,
        deadline_hit: bool,
        cache_hits: int,
        deterministic_reuses: int,
        tree_rebuilds: int,
        root: _Node | None,
        started_ns: int | None = None,
        cache_invalidation_reason: str | None = None,
    ) -> MCTSTurnResult:
        self._cache = None
        self._last_cache_invalidation_reason = reason
        start = lease._clock_ns() if started_ns is None else started_ns
        elapsed = max(0.0, (lease._clock_ns() - start) / 1_000_000_000.0)
        tree_sha256 = self._tree_sha256(root) if root is not None else _canonical_sha256(
            {
                "schema": _TREE_SCHEMA,
                "config_sha256": self.config.identity_sha256,
                "profile_sha256": self.profile.identity_sha256,
                "root_view_sha256": view.identity_sha256,
                "incomplete_reason": reason,
            }
        )
        telemetry = self._telemetry(
            planner_turn_id=planner_turn_id,
            seat=seat,
            view=view,
            lease=lease,
            counters=counters,
            elapsed=elapsed,
            complete=False,
            reason=reason,
            deadline_hit=deadline_hit,
            direct_fallback=True,
            shadow_direct=False,
            cache_hits=cache_hits,
            deterministic_reuses=deterministic_reuses,
            tree_rebuilds=tree_rebuilds,
            tree_sha256=tree_sha256,
            cache_invalidation_reason=cache_invalidation_reason,
            requested_simulations=self.profile.requested_simulations,
            completed_simulations=counters.completed_simulations,
            selected_action=view.direct_action,
        )
        return MCTSTurnResult(
            selected_action=view.direct_action,
            direct_action=view.direct_action,
            root_action_scores=() if root is None else self._root_scores(root),
            root_value=None if root is None else root.mean_value,
            telemetry=telemetry,
        )

    def _telemetry(
        self,
        *,
        planner_turn_id: str,
        seat: int,
        view: PolicyDecisionView,
        lease: PlannerActionDeadline,
        counters: _Counters,
        elapsed: float,
        complete: bool,
        reason: str | None,
        deadline_hit: bool,
        direct_fallback: bool,
        shadow_direct: bool,
        cache_hits: int,
        deterministic_reuses: int,
        tree_rebuilds: int,
        tree_sha256: str,
        cache_invalidation_reason: str | None,
        requested_simulations: int,
        completed_simulations: int,
        selected_action: Action,
    ) -> MCTSTurnTelemetry:
        action_elapsed = min(float(elapsed), float(self.config.max_action_seconds))
        turn_elapsed = self._turn_elapsed(lease)
        return MCTSTurnTelemetry(
            planner_turn_id=planner_turn_id,
            seat=seat,
            turn_key=view.turn_key,
            actions_dispatched=1,
            simulator_transitions_seen=counters.simulator_transitions_seen,
            result_or_leaf_evaluations_seen=(
                counters.terminal_exact_results_seen + counters.neural_leaf_evaluations_seen
            ),
            terminal_exact_results_seen=counters.terminal_exact_results_seen,
            simulator_leaf_evaluations_seen=counters.terminal_exact_results_seen,
            neural_leaf_evaluations_seen=counters.neural_leaf_evaluations_seen,
            boundary_leaf_results_seen=counters.boundary_leaf_results_seen,
            unique_tree_nodes_seen=counters.unique_tree_nodes_seen,
            decision_nodes_expanded=counters.decision_nodes_expanded,
            finite_chance_outcomes_evaluated=counters.finite_chance_outcomes_evaluated,
            frozen_policy_prior_batches=counters.frozen_policy_prior_batches,
            frozen_policy_prior_evaluations=counters.frozen_policy_prior_evaluations,
            batched_frozen_outcome_value_leaf_reranking_batches=(
                counters.batched_frozen_outcome_value_leaf_reranking_batches
            ),
            frozen_outcome_leaf_evaluations=counters.frozen_outcome_leaf_evaluations,
            frozen_value_leaf_evaluations=counters.frozen_value_leaf_evaluations,
            nonterminal_leaves_reranked=counters.nonterminal_leaves_reranked,
            terminal_exact_results_not_reranked=True,
            cache_hits=cache_hits,
            deterministic_subtree_reuses=deterministic_reuses,
            tree_rebuilds=tree_rebuilds,
            turn_planner_wall_seconds=max(turn_elapsed, action_elapsed),
            max_single_action_planner_wall_seconds=action_elapsed,
            requested_tree_fully_expanded_and_backed_up_within_budget=complete,
            tree_incomplete_reason=reason,
            deadline_hit=deadline_hit,
            direct_fallback_used=direct_fallback,
            shadow_direct_action=shadow_direct,
            selected_action_and_legality_fingerprint=_selected_action_fingerprint(
                view, selected_action
            ),
            tree_and_config_sha256=tree_sha256,
            config_sha256=self.config.identity_sha256,
            profile_sha256=self.profile.identity_sha256,
            requested_simulations=requested_simulations,
            completed_simulations=completed_simulations,
            cache_invalidation_reason=cache_invalidation_reason,
        )

    def _turn_elapsed(self, lease: PlannerActionDeadline) -> float:
        """Return charged elapsed time for the current *real* turn clock.

        The deadline controller only resets on the exact real ``turn_key``.
        Using its absolute deadline therefore includes prior atomic actions in
        this turn while predicted child keys inside a simulation cannot grant a
        fresh twenty seconds.  Values are capped at the hard lease for receipts
        after an uninterruptible late dependency has been discarded.
        """

        remaining = max(0.0, (lease.turn_deadline_ns - lease._clock_ns()) / 1e9)
        return min(
            float(self.config.max_turn_seconds),
            max(0.0, float(self.config.max_turn_seconds) - remaining),
        )

__all__ = [
    "Action",
    "FrozenLeafReranker",
    "MCTSActionScore",
    "MCTSExpansionProfile",
    "MCTSTurnResult",
    "MCTSTurnTelemetry",
    "PolicyDecisionFactory",
    "PolicyDecisionView",
    "R207ArenaAdapter",
    "SearchValue",
    "SimulatorInterTurnMCTSSession",
    "SimulatorMCTSError",
    "TurnKey",
]
