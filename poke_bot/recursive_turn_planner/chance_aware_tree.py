"""Offline phase-1 controller for revision-202 cached inter-turn trees.

This module deliberately does not import the policy, engine, simulator, Torch,
the legacy recursive executor, or managed-service code.  It can validate and
advance an immutable prebuilt tree, but it cannot build an MCTS tree, attest a
successor, or grant action authority.  Phase 1 therefore always dispatches the
freshly supplied direct-policy action and records a planner recommendation as
shadow diagnostics only.

The execution cache is reusable only over an attested chance-free transition
whose complete public-state key matches reality.  Finite chance nodes may be
valued exactly during offline search, but every realized chance event starts a
fresh public execution root.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import TypeAlias

Action: TypeAlias = tuple[int, ...]
TurnKey: TypeAlias = tuple[int, int]

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TREE_SCHEMA = "poke_bot.chance_aware_cached_inter_turn_tree/v1"
_LEGAL_SCHEMA = "poke_bot.complete_ordered_action_fingerprint/v1"
_HARD_MAX_COMPLETE_ACTIONS = 1024


class ChanceAwareTreeError(ValueError):
    """A typed phase-1 tree or public snapshot is malformed."""


class ControllerProtocolError(RuntimeError):
    """The caller violated the one-action-before-observation protocol."""


def _require_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChanceAwareTreeError(f"{name} must be a canonical sha256 digest")
    return value


def _require_turn_key(value: TurnKey) -> TurnKey:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(part) is not int for part in value)
        or value[0] < 0
        or value[1] < 0
    ):
        raise ChanceAwareTreeError("turn_key must be a non-negative (seat, turn) tuple")
    return value


def _normalise_action(action: Action, *, name: str = "action") -> Action:
    if not isinstance(action, tuple):
        raise ChanceAwareTreeError(f"{name} must be a tuple")
    if any(type(part) is not int or part < 0 for part in action):
        raise ChanceAwareTreeError(f"{name} members must be non-negative exact ints")
    return action


def _normalise_actions(
    actions: tuple[Action, ...],
    *,
    allow_empty_set: bool = False,
) -> tuple[Action, ...]:
    if not isinstance(actions, tuple):
        raise ChanceAwareTreeError("legal actions must be a tuple")
    normalised = tuple(
        _normalise_action(action, name=f"legal_actions[{index}]")
        for index, action in enumerate(actions)
    )
    if not normalised and not allow_empty_set:
        raise ChanceAwareTreeError("a decision requires at least one complete action")
    if len(set(normalised)) != len(normalised):
        raise ChanceAwareTreeError("complete legal actions must be unique")
    return normalised


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def complete_action_fingerprint(actions: tuple[Action, ...]) -> str:
    """Bind the exact ordered complete-action list, preserving ``()``."""

    normalised = _normalise_actions(actions)
    return _canonical_sha256(
        {
            "schema": _LEGAL_SCHEMA,
            "representation": "complete_ordered_actions",
            "actions": [list(action) for action in normalised],
        }
    )


@dataclass(frozen=True, slots=True)
class ChanceAwareSearchConfig:
    """Single, easy-to-change, identity-bound planner budget surface."""

    max_turn_seconds: float = 20.0
    max_action_seconds: float = 5.0
    max_complete_actions: int = 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("max_turn_seconds", self.max_turn_seconds),
            ("max_action_seconds", self.max_action_seconds),
        ):
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ChanceAwareTreeError(f"{name} must be finite")
            if float(value) <= 0.0:
                raise ChanceAwareTreeError(f"{name} must be positive")
        if self.max_action_seconds > self.max_turn_seconds:
            raise ChanceAwareTreeError(
                "per-action planner budget cannot exceed the per-turn budget"
            )
        if (
            type(self.max_complete_actions) is not int
            or self.max_complete_actions <= 0
            or self.max_complete_actions > _HARD_MAX_COMPLETE_ACTIONS
        ):
            raise ChanceAwareTreeError(
                "max_complete_actions must be an exact int in [1, 1024]"
            )

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "poke_bot.chance_aware_search_config/v1",
                "max_turn_seconds": float(self.max_turn_seconds),
                "max_action_seconds": float(self.max_action_seconds),
                "max_complete_actions": self.max_complete_actions,
            }
        )


@dataclass(frozen=True, slots=True)
class PublicStateKey:
    """Exact policy-visible identity required before a cached node can run."""

    turn_key: TurnKey
    decision_serial: int
    observation_sha256: str
    legal_actions_sha256: str
    option_encoding_sha256: str
    public_history_sha256: str
    source_sha256: str
    model_sha256: str
    rules_abi_sha256: str
    profile_sha256: str

    def __post_init__(self) -> None:
        _require_turn_key(self.turn_key)
        if type(self.decision_serial) is not int or self.decision_serial < 0:
            raise ChanceAwareTreeError("decision_serial must be a non-negative exact int")
        for name in (
            "observation_sha256",
            "legal_actions_sha256",
            "option_encoding_sha256",
            "public_history_sha256",
            "source_sha256",
            "model_sha256",
            "rules_abi_sha256",
            "profile_sha256",
        ):
            _require_digest(getattr(self, name), name=name)

    def as_payload(self) -> dict[str, object]:
        return {
            "turn_key": list(self.turn_key),
            "decision_serial": self.decision_serial,
            "observation_sha256": self.observation_sha256,
            "legal_actions_sha256": self.legal_actions_sha256,
            "option_encoding_sha256": self.option_encoding_sha256,
            "public_history_sha256": self.public_history_sha256,
            "source_sha256": self.source_sha256,
            "model_sha256": self.model_sha256,
            "rules_abi_sha256": self.rules_abi_sha256,
            "profile_sha256": self.profile_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "poke_bot.chance_aware_public_state_key/v1",
                **self.as_payload(),
            }
        )


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """Fresh current-decision inputs supplied by a policy-visible boundary."""

    key: PublicStateKey
    legal_actions: tuple[Action, ...]
    direct_action: Action

    def __post_init__(self) -> None:
        legal_actions = _normalise_actions(self.legal_actions)
        direct_action = _normalise_action(self.direct_action, name="direct_action")
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "direct_action", direct_action)
        if complete_action_fingerprint(legal_actions) != self.key.legal_actions_sha256:
            raise ChanceAwareTreeError(
                "snapshot legal actions do not match the public-state fingerprint"
            )
        if direct_action not in legal_actions:
            raise ChanceAwareTreeError(
                "the exact direct-policy action must belong to the complete legal set"
            )


def _require_exact_value(value: Fraction, *, name: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise ChanceAwareTreeError(f"{name} must be an exact Fraction")
    return value


@dataclass(frozen=True, slots=True)
class TerminalNode:
    value: Fraction
    reason: str = "terminal"

    def __post_init__(self) -> None:
        _require_exact_value(self.value, name="terminal value")
        if not self.reason:
            raise ChanceAwareTreeError("terminal reason must not be empty")


@dataclass(frozen=True, slots=True)
class BoundaryNode:
    """A calibrated leaf where exact expansion must stop."""

    value: Fraction
    reason: str

    def __post_init__(self) -> None:
        _require_exact_value(self.value, name="boundary value")
        if not self.reason:
            raise ChanceAwareTreeError("boundary reason must not be empty")


@dataclass(frozen=True, slots=True)
class ChanceOutcome:
    label: str
    probability: Fraction
    child: TreeNode

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ChanceAwareTreeError("chance outcome labels must not be empty")
        probability = _require_exact_value(
            self.probability, name="chance probability"
        )
        if probability <= 0:
            raise ChanceAwareTreeError("chance probabilities must be positive")
        if not isinstance(
            self.child,
            (DecisionNode, FiniteChanceNode, TerminalNode, BoundaryNode),
        ):
            raise ChanceAwareTreeError("chance outcomes require typed tree children")


@dataclass(frozen=True, slots=True)
class FiniteChanceNode:
    """A complete exact distribution used for expectimax-style backup only."""

    event_id: str
    distribution_receipt_sha256: str
    outcomes: tuple[ChanceOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ChanceAwareTreeError("finite chance event_id must not be empty")
        _require_digest(
            self.distribution_receipt_sha256,
            name="distribution_receipt_sha256",
        )
        if not isinstance(self.outcomes, tuple) or len(self.outcomes) < 2:
            raise ChanceAwareTreeError(
                "finite chance requires at least two fully enumerated outcomes"
            )
        labels = tuple(outcome.label for outcome in self.outcomes)
        if len(set(labels)) != len(labels):
            raise ChanceAwareTreeError("finite chance outcome labels must be unique")
        if sum((outcome.probability for outcome in self.outcomes), Fraction(0)) != 1:
            raise ChanceAwareTreeError(
                "finite chance probabilities must sum exactly to one"
            )

    @property
    def expected_value(self) -> Fraction:
        return sum(
            (
                outcome.probability * node_value(outcome.child)
                for outcome in self.outcomes
            ),
            Fraction(0),
        )


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    label: str
    child: TreeNode

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ChanceAwareTreeError("branch outcome labels must not be empty")
        if not isinstance(
            self.child,
            (DecisionNode, FiniteChanceNode, TerminalNode, BoundaryNode),
        ):
            raise ChanceAwareTreeError("branch outcomes require typed tree children")


@dataclass(frozen=True, slots=True)
class DeterministicTransition:
    certificate_sha256: str
    child: TreeNode

    def __post_init__(self) -> None:
        _require_digest(self.certificate_sha256, name="deterministic certificate")
        if not isinstance(
            self.child,
            (DecisionNode, FiniteChanceNode, TerminalNode, BoundaryNode),
        ):
            raise ChanceAwareTreeError(
                "deterministic transitions require a typed tree child"
            )


@dataclass(frozen=True, slots=True)
class ObservedPublicBranchTransition:
    """Chance-free public branch; missing labels select neither child."""

    predicate_id: str
    certificate_sha256: str
    outcomes: tuple[BranchOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.predicate_id, str) or not self.predicate_id:
            raise ChanceAwareTreeError("branch predicate_id must not be empty")
        _require_digest(self.certificate_sha256, name="branch certificate")
        if not isinstance(self.outcomes, tuple) or len(self.outcomes) < 2:
            raise ChanceAwareTreeError("a public branch needs at least two outcomes")
        labels = tuple(outcome.label for outcome in self.outcomes)
        if len(set(labels)) != len(labels):
            raise ChanceAwareTreeError("public branch outcome labels must be unique")

    def child_for(self, label: str | None) -> TreeNode | None:
        if label is None:
            return None
        for outcome in self.outcomes:
            if outcome.label == label:
                return outcome.child
        return None


@dataclass(frozen=True, slots=True)
class ObservedPublicBranchEvidence:
    """Receipt-bound evaluation of a public branch on the real snapshot."""

    predicate_id: str
    outcome_label: str
    certificate_sha256: str
    observation_sha256: str
    public_state_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.predicate_id, str) or not self.predicate_id:
            raise ChanceAwareTreeError("branch evidence predicate_id must not be empty")
        if not isinstance(self.outcome_label, str) or not self.outcome_label:
            raise ChanceAwareTreeError("branch evidence outcome_label must not be empty")
        for name in (
            "certificate_sha256",
            "observation_sha256",
            "public_state_sha256",
        ):
            _require_digest(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class FiniteChanceTransition:
    chance: FiniteChanceNode


@dataclass(frozen=True, slots=True)
class RebuildBoundaryTransition:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ChanceAwareTreeError("rebuild boundary reason must not be empty")


@dataclass(frozen=True, slots=True)
class TerminalTransition:
    terminal: TerminalNode


Transition: TypeAlias = (
    DeterministicTransition
    | ObservedPublicBranchTransition
    | FiniteChanceTransition
    | RebuildBoundaryTransition
    | TerminalTransition
)


@dataclass(frozen=True, slots=True)
class ActionEdge:
    action: Action
    transition: Transition

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _normalise_action(self.action))
        if not isinstance(
            self.transition,
            (
                DeterministicTransition,
                ObservedPublicBranchTransition,
                FiniteChanceTransition,
                RebuildBoundaryTransition,
                TerminalTransition,
            ),
        ):
            raise ChanceAwareTreeError("action edges require a typed transition")


@dataclass(frozen=True, slots=True)
class DecisionNode:
    expected_state: PublicStateKey
    expected_legal_actions: tuple[Action, ...]
    direct_action: Action
    shadow_recommended_action: Action
    search_value: Fraction
    edges: tuple[ActionEdge, ...]

    def __post_init__(self) -> None:
        legal_actions = _normalise_actions(self.expected_legal_actions)
        direct_action = _normalise_action(self.direct_action, name="direct_action")
        recommended = _normalise_action(
            self.shadow_recommended_action,
            name="shadow_recommended_action",
        )
        _require_exact_value(self.search_value, name="decision search_value")
        object.__setattr__(self, "expected_legal_actions", legal_actions)
        object.__setattr__(self, "direct_action", direct_action)
        object.__setattr__(self, "shadow_recommended_action", recommended)
        if complete_action_fingerprint(legal_actions) != self.expected_state.legal_actions_sha256:
            raise ChanceAwareTreeError(
                "decision legal actions do not match expected public-state key"
            )
        if direct_action not in legal_actions:
            raise ChanceAwareTreeError("direct action must be a complete legal action")
        if recommended not in legal_actions:
            raise ChanceAwareTreeError("shadow recommendation must be legal")
        if not isinstance(self.edges, tuple) or not self.edges:
            raise ChanceAwareTreeError("a decision node requires action edges")
        edge_actions = tuple(edge.action for edge in self.edges)
        if len(set(edge_actions)) != len(edge_actions):
            raise ChanceAwareTreeError("decision edge actions must be unique")
        if any(action not in legal_actions for action in edge_actions):
            raise ChanceAwareTreeError("decision edges must be drawn from legal actions")
        expected_edge_order = tuple(
            action for action in legal_actions if action in set(edge_actions)
        )
        if edge_actions != expected_edge_order:
            raise ChanceAwareTreeError(
                "decision edges must preserve complete-action canonical order"
            )
        if direct_action not in edge_actions:
            raise ChanceAwareTreeError("direct action requires an explicit tree edge")
        if recommended not in edge_actions:
            raise ChanceAwareTreeError("shadow recommendation requires a tree edge")

    def edge_for(self, action: Action) -> ActionEdge | None:
        for edge in self.edges:
            if edge.action == action:
                return edge
        return None


TreeNode: TypeAlias = DecisionNode | FiniteChanceNode | TerminalNode | BoundaryNode


def node_value(node: TreeNode) -> Fraction:
    if isinstance(node, DecisionNode):
        return node.search_value
    if isinstance(node, FiniteChanceNode):
        return node.expected_value
    return node.value


def _fraction_payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _node_payload(node: TreeNode, active: set[int]) -> dict[str, object]:
    identity = id(node)
    if identity in active:
        raise ChanceAwareTreeError("cached trees must be acyclic")
    active.add(identity)
    try:
        if isinstance(node, TerminalNode):
            return {
                "kind": "terminal",
                "value": _fraction_payload(node.value),
                "reason": node.reason,
            }
        if isinstance(node, BoundaryNode):
            return {
                "kind": "boundary",
                "value": _fraction_payload(node.value),
                "reason": node.reason,
            }
        if isinstance(node, FiniteChanceNode):
            return {
                "kind": "finite_chance",
                "event_id": node.event_id,
                "distribution_receipt_sha256": node.distribution_receipt_sha256,
                "outcomes": [
                    {
                        "label": outcome.label,
                        "probability": _fraction_payload(outcome.probability),
                        "child": _node_payload(outcome.child, active),
                    }
                    for outcome in node.outcomes
                ],
            }
        return {
            "kind": "decision",
            "expected_state": node.expected_state.as_payload(),
            "expected_legal_actions": [
                list(action) for action in node.expected_legal_actions
            ],
            "direct_action": list(node.direct_action),
            "shadow_recommended_action": list(node.shadow_recommended_action),
            "search_value": _fraction_payload(node.search_value),
            "edges": [
                {
                    "action": list(edge.action),
                    "transition": _transition_payload(edge.transition, active),
                }
                for edge in node.edges
            ],
        }
    finally:
        active.remove(identity)


def _transition_payload(
    transition: Transition,
    active: set[int],
) -> dict[str, object]:
    if isinstance(transition, DeterministicTransition):
        return {
            "kind": "deterministic_public",
            "certificate_sha256": transition.certificate_sha256,
            "child": _node_payload(transition.child, active),
        }
    if isinstance(transition, ObservedPublicBranchTransition):
        return {
            "kind": "observed_public_branch",
            "predicate_id": transition.predicate_id,
            "certificate_sha256": transition.certificate_sha256,
            "outcomes": [
                {
                    "label": outcome.label,
                    "child": _node_payload(outcome.child, active),
                }
                for outcome in transition.outcomes
            ],
        }
    if isinstance(transition, FiniteChanceTransition):
        return {
            "kind": "finite_chance",
            "event_id": transition.chance.event_id,
            "distribution_receipt_sha256": (
                transition.chance.distribution_receipt_sha256
            ),
            "outcomes": [
                {
                    "label": outcome.label,
                    "probability": _fraction_payload(outcome.probability),
                    "child": _node_payload(outcome.child, active),
                }
                for outcome in transition.chance.outcomes
            ],
        }
    if isinstance(transition, RebuildBoundaryTransition):
        return {"kind": "rebuild_boundary", "reason": transition.reason}
    return {
        "kind": "terminal_transition",
        "terminal": _node_payload(transition.terminal, active),
    }


@dataclass(frozen=True, slots=True)
class CachedActionTree:
    root: DecisionNode
    config_sha256: str
    planner_artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, DecisionNode):
            raise ChanceAwareTreeError("cached action tree root must be a decision node")
        _require_digest(self.config_sha256, name="config_sha256")
        _require_digest(
            self.planner_artifact_sha256,
            name="planner_artifact_sha256",
        )
        # Force full traversal now so malformed or cyclic trees never enter cache.
        _node_payload(self.root, set())

    @property
    def tree_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": _TREE_SCHEMA,
                "config_sha256": self.config_sha256,
                "planner_artifact_sha256": self.planner_artifact_sha256,
                "root": _node_payload(self.root, set()),
            }
        )


class ControllerState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    AWAITING_REAL_OBSERVATION = "awaiting_real_observation"
    REBUILD_REQUIRED = "rebuild_required"
    TERMINATED = "terminated"


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    turn_key: TurnKey | None
    turn_seconds_used: float
    action_seconds_used: float
    turn_exhausted: bool
    action_exhausted: bool


class _BudgetLedger:
    def __init__(self, config: ChanceAwareSearchConfig) -> None:
        self.config = config
        self.turn_key: TurnKey | None = None
        self.turn_seconds_used = 0.0
        self.action_seconds_used = 0.0
        self.turn_exhausted = False
        self.action_exhausted = False

    def begin_observed_decision(self, turn_key: TurnKey) -> None:
        turn_key = _require_turn_key(turn_key)
        if self.turn_key != turn_key:
            self.turn_key = turn_key
            self.turn_seconds_used = 0.0
            self.turn_exhausted = False
        self.action_seconds_used = 0.0
        self.action_exhausted = False

    def charge_elapsed(self, turn_key: TurnKey, seconds: float) -> bool:
        turn_key = _require_turn_key(turn_key)
        if self.turn_key != turn_key:
            self.begin_observed_decision(turn_key)
        if type(seconds) not in {int, float} or not math.isfinite(float(seconds)):
            raise ChanceAwareTreeError("measured planner elapsed time must be finite")
        if float(seconds) < 0.0:
            raise ChanceAwareTreeError(
                "monotonic planner clock moved backwards"
            )
        self.turn_seconds_used += float(seconds)
        self.action_seconds_used += float(seconds)
        self.turn_exhausted = (
            self.turn_seconds_used > float(self.config.max_turn_seconds)
        )
        self.action_exhausted = (
            self.action_seconds_used > float(self.config.max_action_seconds)
        )
        return not self.turn_exhausted and not self.action_exhausted

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            turn_key=self.turn_key,
            turn_seconds_used=self.turn_seconds_used,
            action_seconds_used=self.action_seconds_used,
            turn_exhausted=self.turn_exhausted,
            action_exhausted=self.action_exhausted,
        )


@dataclass(frozen=True, slots=True)
class DispatchResult:
    action: Action | None
    direct_action: Action
    shadow_recommended_action: Action | None
    mode: str
    reasons: tuple[str, ...]
    tree_sha256: str | None
    node_sha256: str | None
    action_authority_enabled: bool
    awaiting_real_observation: bool
    budget: BudgetSnapshot


@dataclass(frozen=True, slots=True)
class ObservationResult:
    state: ControllerState
    reused_subtree: bool
    rebuild_required: bool
    terminated: bool
    reason: str
    tree_sha256: str | None
    node_sha256: str | None
    chance_expected_value: Fraction | None
    budget: BudgetSnapshot


class ChanceAwareTreeController:
    """Validate/reuse prebuilt trees while retaining zero action authority."""

    action_authority_enabled = False

    def __init__(
        self,
        config: ChanceAwareSearchConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config or ChanceAwareSearchConfig()
        self._config_identity_sha256 = self._config.identity_sha256
        self._clock = clock or time.monotonic
        if not callable(self._clock):
            raise ChanceAwareTreeError("planner clock must be callable")
        self.state = ControllerState.IDLE
        self._tree: CachedActionTree | None = None
        self._node: DecisionNode | None = None
        self._pending_transition: Transition | None = None
        self._installed_tree_sha256: str | None = None
        self._latest_snapshot_key: PublicStateKey | None = None
        self._last_decision_serial: int | None = None
        self._budget = _BudgetLedger(self._config)
        self.tree_installations = 0
        self.subtree_reuses = 0

    @property
    def config(self) -> ChanceAwareSearchConfig:
        return self._config

    @property
    def tree_sha256(self) -> str | None:
        return self._tree.tree_sha256 if self._tree is not None else None

    @property
    def current_node_sha256(self) -> str | None:
        if self._node is None:
            return None
        return _canonical_sha256(_node_payload(self._node, set()))

    @property
    def budget(self) -> BudgetSnapshot:
        return self._budget.snapshot()

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ChanceAwareTreeError("planner monotonic clock must return a finite number")
        return float(value)

    def _charge_elapsed(self, turn_key: TurnKey, started_at: float) -> bool:
        return self._budget.charge_elapsed(turn_key, self._now() - started_at)

    def _config_integrity_reason(self) -> str | None:
        if self._config.identity_sha256 != self._config_identity_sha256:
            return "planner_config_identity_changed"
        return None

    def _tree_integrity_reason(self) -> str | None:
        if self._tree is None:
            return None
        if self._tree.config_sha256 != self._config_identity_sha256:
            return "cached_tree_config_identity_changed"
        if (
            self._installed_tree_sha256 is None
            or self._tree.tree_sha256 != self._installed_tree_sha256
        ):
            return "cached_tree_content_digest_changed"
        return None

    def _invalidate(self, *, terminated: bool = False) -> None:
        self._tree = None
        self._node = None
        self._pending_transition = None
        self._installed_tree_sha256 = None
        self.state = (
            ControllerState.TERMINATED
            if terminated
            else ControllerState.REBUILD_REQUIRED
        )

    def install_tree(
        self,
        tree: CachedActionTree,
        snapshot: DecisionSnapshot,
    ) -> bool:
        started_at = self._now()
        if self.state is ControllerState.AWAITING_REAL_OBSERVATION:
            raise ControllerProtocolError(
                "cannot install a tree before consuming the real observation"
            )
        if self.state is ControllerState.TERMINATED:
            raise ControllerProtocolError("cannot install a tree after a real terminal")
        if self.state is ControllerState.READY:
            raise ControllerProtocolError(
                "cannot replace a ready tree before dispatching its current action"
            )
        config_reason = self._config_integrity_reason()
        if config_reason is not None:
            self._charge_elapsed(snapshot.key.turn_key, started_at)
            self._invalidate()
            raise ChanceAwareTreeError(config_reason)
        if tree.config_sha256 != self._config_identity_sha256:
            self._charge_elapsed(snapshot.key.turn_key, started_at)
            raise ChanceAwareTreeError("tree/config budget identity mismatch")
        if (
            self._last_decision_serial is not None
            and snapshot.key.decision_serial < self._last_decision_serial
        ):
            self._charge_elapsed(snapshot.key.turn_key, started_at)
            self._invalidate()
            return False
        if (
            self.state is ControllerState.REBUILD_REQUIRED
            and self._latest_snapshot_key is not None
            and snapshot.key != self._latest_snapshot_key
        ):
            self._charge_elapsed(snapshot.key.turn_key, started_at)
            self._invalidate()
            return False
        if len(snapshot.legal_actions) > self._config.max_complete_actions:
            self._charge_elapsed(snapshot.key.turn_key, started_at)
            self._invalidate()
            return False
        if not self._node_matches_snapshot(tree.root, snapshot):
            self._charge_elapsed(snapshot.key.turn_key, started_at)
            self._invalidate()
            return False
        tree_sha256 = tree.tree_sha256
        if not self._charge_elapsed(snapshot.key.turn_key, started_at):
            self._invalidate()
            return False
        self._tree = tree
        self._node = tree.root
        self._pending_transition = None
        self._installed_tree_sha256 = tree_sha256
        self._latest_snapshot_key = snapshot.key
        self._last_decision_serial = snapshot.key.decision_serial
        self.state = ControllerState.READY
        self.tree_installations += 1
        return True

    @staticmethod
    def _node_matches_snapshot(
        node: DecisionNode,
        snapshot: DecisionSnapshot,
    ) -> bool:
        return (
            node.expected_state == snapshot.key
            and node.expected_legal_actions == snapshot.legal_actions
            and node.direct_action == snapshot.direct_action
        )

    def dispatch(
        self,
        snapshot: DecisionSnapshot,
    ) -> DispatchResult:
        if self.state is ControllerState.AWAITING_REAL_OBSERVATION:
            return DispatchResult(
                action=None,
                direct_action=snapshot.direct_action,
                shadow_recommended_action=None,
                mode="blocked_real_observation_required",
                reasons=("one_action_already_dispatched",),
                tree_sha256=self.tree_sha256,
                node_sha256=self.current_node_sha256,
                action_authority_enabled=False,
                awaiting_real_observation=True,
                budget=self.budget,
            )
        if self.state is ControllerState.TERMINATED:
            return DispatchResult(
                action=None,
                direct_action=snapshot.direct_action,
                shadow_recommended_action=None,
                mode="terminated",
                reasons=("tree_or_game_terminal",),
                tree_sha256=None,
                node_sha256=None,
                action_authority_enabled=False,
                awaiting_real_observation=False,
                budget=self.budget,
            )

        started_at = self._now()
        reasons: list[str] = []
        node = self._node
        config_reason = self._config_integrity_reason()
        if config_reason is not None:
            reasons.append(config_reason)
            node = None
        tree_reason = self._tree_integrity_reason()
        if tree_reason is not None:
            reasons.append(tree_reason)
            node = None
        if len(snapshot.legal_actions) > self._config.max_complete_actions:
            reasons.append("complete_action_cap_exceeded")
            node = None
        if node is not None and not self._node_matches_snapshot(node, snapshot):
            reasons.append("cached_node_fingerprint_mismatch")
            node = None
        if (
            self._last_decision_serial is not None
            and snapshot.key.decision_serial < self._last_decision_serial
        ):
            reasons.append("decision_serial_regressed")
            node = None
        if (
            self._latest_snapshot_key is not None
            and snapshot.key != self._latest_snapshot_key
        ):
            reasons.append("unobserved_decision_snapshot")
            node = None

        recommended: Action | None = None
        transition: Transition | None = None
        if node is not None:
            recommended = node.shadow_recommended_action
            direct_edge = node.edge_for(snapshot.direct_action)
            if direct_edge is None:
                reasons.append("direct_action_tree_edge_missing")
            else:
                transition = direct_edge.transition

        if not self._charge_elapsed(snapshot.key.turn_key, started_at):
            reasons.append("planner_budget_exhausted")
            node = None
            transition = None

        # Revision 202 phase 1 has zero action authority.  Even a valid tree is
        # diagnostics-only until later receipt-backed phases exist.
        action = snapshot.direct_action
        self._pending_transition = transition
        if (
            self._last_decision_serial is None
            or snapshot.key.decision_serial >= self._last_decision_serial
        ):
            self._latest_snapshot_key = snapshot.key
            self._last_decision_serial = snapshot.key.decision_serial
        self.state = ControllerState.AWAITING_REAL_OBSERVATION
        if reasons:
            self._tree = None
            self._node = None
            self._pending_transition = None
            self._installed_tree_sha256 = None
        return DispatchResult(
            action=action,
            direct_action=snapshot.direct_action,
            shadow_recommended_action=recommended,
            mode=(
                "phase1_direct_with_valid_shadow_tree"
                if node is not None and not reasons
                else "exact_direct_fallback"
            ),
            reasons=tuple(reasons),
            tree_sha256=self.tree_sha256,
            node_sha256=self.current_node_sha256,
            action_authority_enabled=False,
            awaiting_real_observation=True,
            budget=self.budget,
        )

    def observe(
        self,
        snapshot: DecisionSnapshot | None,
        *,
        branch_evidence: ObservedPublicBranchEvidence | None = None,
        chance_outcome_label: str | None = None,
        terminal: bool = False,
    ) -> ObservationResult:
        if self.state is not ControllerState.AWAITING_REAL_OBSERVATION:
            raise ControllerProtocolError(
                "a real observation may be consumed only after one dispatched action"
            )
        started_at = self._now()
        transition = self._pending_transition
        self._pending_transition = None

        if terminal:
            if snapshot is not None:
                self._budget.begin_observed_decision(snapshot.key.turn_key)
                self._charge_elapsed(snapshot.key.turn_key, started_at)
                self._latest_snapshot_key = snapshot.key
                if (
                    self._last_decision_serial is None
                    or snapshot.key.decision_serial > self._last_decision_serial
                ):
                    self._last_decision_serial = snapshot.key.decision_serial
                self._invalidate()
                return self._observation_result(
                    reused=False,
                    reason="terminal_claim_conflicts_with_nonterminal_snapshot",
                )
            if self._budget.turn_key is not None:
                self._charge_elapsed(self._budget.turn_key, started_at)
            self._invalidate(terminated=True)
            return self._observation_result(
                reused=False,
                reason="real_terminal_observed",
            )
        if snapshot is None:
            if self._budget.turn_key is not None:
                self._charge_elapsed(self._budget.turn_key, started_at)
            self._invalidate()
            return self._observation_result(
                reused=False,
                reason="missing_real_policy_visible_snapshot",
            )

        self._budget.begin_observed_decision(snapshot.key.turn_key)
        fresh_serial = True
        if (
            self._last_decision_serial is not None
            and snapshot.key.decision_serial <= self._last_decision_serial
        ):
            fresh_serial = False

        reason: str
        child: TreeNode | None = None
        chance_expected_value: Fraction | None = None
        config_reason = self._config_integrity_reason()
        tree_reason = self._tree_integrity_reason()
        if not fresh_serial:
            reason = "fresh_decision_serial_required"
        elif config_reason is not None:
            reason = config_reason
        elif tree_reason is not None:
            reason = tree_reason
        elif isinstance(transition, DeterministicTransition):
            child = transition.child
            reason = "deterministic_subtree_reused"
        elif isinstance(transition, ObservedPublicBranchTransition):
            if branch_evidence is None:
                reason = "missing_or_unknown_public_branch_outcome"
            elif (
                branch_evidence.predicate_id != transition.predicate_id
                or branch_evidence.certificate_sha256
                != transition.certificate_sha256
                or branch_evidence.observation_sha256
                != snapshot.key.observation_sha256
                or branch_evidence.public_state_sha256
                != snapshot.key.identity_sha256
            ):
                reason = "public_branch_evidence_mismatch"
            else:
                child = transition.child_for(branch_evidence.outcome_label)
                reason = (
                    "attested_public_branch_subtree_reused"
                    if child is not None
                    else "missing_or_unknown_public_branch_outcome"
                )
        elif isinstance(transition, FiniteChanceTransition):
            labels = {outcome.label for outcome in transition.chance.outcomes}
            chance_expected_value = transition.chance.expected_value
            reason = (
                "realized_finite_chance_starts_fresh_public_root"
                if chance_outcome_label in labels
                else "missing_or_unknown_chance_outcome_starts_fresh_public_root"
            )
        elif isinstance(transition, TerminalTransition):
            reason = "predicted_terminal_not_confirmed"
        elif isinstance(transition, RebuildBoundaryTransition):
            reason = f"rebuild_boundary:{transition.reason}"
        else:
            reason = "no_validated_cached_transition"

        reuse_child: DecisionNode | None = None
        if child is not None:
            if isinstance(child, DecisionNode):
                if self._node_matches_snapshot(child, snapshot):
                    reuse_child = child
                else:
                    reason = "cached_child_fingerprint_mismatch"
            elif isinstance(child, TerminalNode):
                reason = "predicted_terminal_not_confirmed"
            elif isinstance(child, BoundaryNode):
                reason = f"cached_child_boundary:{child.reason}"
            else:
                reason = "cached_child_chance_requires_fresh_public_root"

        budget_ok = self._charge_elapsed(snapshot.key.turn_key, started_at)
        if not budget_ok:
            reason = "planner_budget_exhausted_during_observation_validation"
            reuse_child = None

        if fresh_serial:
            self._last_decision_serial = snapshot.key.decision_serial
            self._latest_snapshot_key = snapshot.key
        if reuse_child is not None and budget_ok:
            self._node = reuse_child
            self.state = ControllerState.READY
            self.subtree_reuses += 1
            return self._observation_result(reused=True, reason=reason)

        self._invalidate()
        return self._observation_result(
            reused=False,
            reason=reason,
            chance_expected_value=chance_expected_value,
        )

    def _observation_result(
        self,
        *,
        reused: bool,
        reason: str,
        chance_expected_value: Fraction | None = None,
    ) -> ObservationResult:
        return ObservationResult(
            state=self.state,
            reused_subtree=reused,
            rebuild_required=self.state is ControllerState.REBUILD_REQUIRED,
            terminated=self.state is ControllerState.TERMINATED,
            reason=reason,
            tree_sha256=self.tree_sha256,
            node_sha256=self.current_node_sha256,
            chance_expected_value=chance_expected_value,
            budget=self.budget,
        )


__all__ = [
    "Action",
    "ActionEdge",
    "BoundaryNode",
    "BranchOutcome",
    "BudgetSnapshot",
    "CachedActionTree",
    "ChanceAwareSearchConfig",
    "ChanceAwareTreeController",
    "ChanceAwareTreeError",
    "ChanceOutcome",
    "ControllerProtocolError",
    "ControllerState",
    "DecisionNode",
    "DecisionSnapshot",
    "DeterministicTransition",
    "DispatchResult",
    "FiniteChanceNode",
    "FiniteChanceTransition",
    "ObservationResult",
    "ObservedPublicBranchEvidence",
    "ObservedPublicBranchTransition",
    "PublicStateKey",
    "RebuildBoundaryTransition",
    "TerminalNode",
    "TerminalTransition",
    "complete_action_fingerprint",
    "node_value",
]
