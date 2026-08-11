"""Shadow-only, goal-directed same-turn tactical sequence search.

This module is intentionally not wired into the serving agent.  It searches a
small deterministic state graph for an explicit goal and emits an auditable
proposal/certificate.  It never dispatches an action.  Native simulator
adapters must run in an owned bounded child process; deterministic in-process
backends are accepted only when explicitly marked as test fixtures.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


Action = tuple[int, ...]
IsolationMode = Literal["owned_bounded_child", "deterministic_test_fixture"]
GoalKind = Literal["exact_terminal_win", "public_fact", "visible_tutor_target"]

TACTICAL_SEQUENCE_RECEIPT_SCHEMA = "poke_bot.tactical_sequence_shadow/v1"
TACTICAL_SEQUENCE_PROOF_SCHEMA = "poke_bot.tactical_sequence_proof/v1"
DEFAULT_INTERNAL_ACTION_CEILING = 64


class TacticalSequenceError(RuntimeError):
    """The shadow planner or one of its typed inputs failed closed."""


def _action(value: Sequence[int]) -> Action:
    action = tuple(int(item) for item in value)
    if len(action) != len(set(action)) or any(item < 0 for item in action):
        raise TacticalSequenceError("actions must contain unique nonnegative indices")
    return action


def legal_action_order_fingerprint(actions: Sequence[Sequence[int]]) -> str:
    normalized = [_action(action) for action in actions]
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TacticalSearchState:
    """One exact state exposed by a simulator adapter or deterministic fixture."""

    observation_fingerprint: str
    semantic_fingerprint: str
    actor: int
    turn_id: int | str
    legal_actions: tuple[Action, ...]
    ordered_action_count: int
    terminal_winner: int | None = None
    explicit_chance_boundary: bool = False
    information_boundary: bool = False
    public_facts: Mapping[str, Any] = field(default_factory=dict)
    public_facts_are_observed: bool = True
    visible_tutor_cards: tuple[int, ...] = ()
    previous_action_token: Action | None = None
    raw_observation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.observation_fingerprint or not self.semantic_fingerprint:
            raise TacticalSequenceError("state fingerprints must be nonempty")
        if self.actor not in (0, 1):
            raise TacticalSequenceError("state actor must be 0 or 1")
        actions = tuple(_action(action) for action in self.legal_actions)
        if len(actions) != len(set(actions)):
            raise TacticalSequenceError("state legal actions must be unique")
        object.__setattr__(self, "legal_actions", actions)
        count = int(self.ordered_action_count)
        if count < len(actions) or count < 0:
            raise TacticalSequenceError(
                "ordered_action_count cannot be smaller than the supplied legal actions"
            )
        object.__setattr__(self, "ordered_action_count", count)
        if self.terminal_winner not in (None, 0, 1, 2):
            raise TacticalSequenceError("terminal_winner must be 0, 1, 2, or None")
        if self.previous_action_token is not None:
            object.__setattr__(
                self, "previous_action_token", _action(self.previous_action_token)
            )
        object.__setattr__(
            self, "visible_tutor_cards", tuple(int(card) for card in self.visible_tutor_cards)
        )

    @property
    def legal_order_fingerprint(self) -> str:
        return legal_action_order_fingerprint(self.legal_actions)


@dataclass(frozen=True, slots=True)
class TacticalTransition:
    """One verified deterministic simulator transition."""

    next_state: TacticalSearchState
    action_token: Action
    deterministic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_token", _action(self.action_token))


class TacticalTransitionBackend(Protocol):
    isolation_mode: IsolationMode

    def advance(
        self,
        state: TacticalSearchState,
        action: Action,
        *,
        deadline_monotonic: float,
    ) -> TacticalTransition:
        """Advance exactly one legal action before ``deadline_monotonic``."""


@dataclass(frozen=True, slots=True)
class RankedAction:
    action: Action
    probability: float
    sme_priority: float = 0.0
    tactical_head_hint: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _action(self.action))
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise TacticalSequenceError("ranked action probability must be finite in [0, 1]")
        object.__setattr__(self, "probability", probability)
        priority = float(self.sme_priority)
        if not math.isfinite(priority):
            raise TacticalSequenceError("SME priority must be finite")
        object.__setattr__(self, "sme_priority", priority)
        if self.tactical_head_hint is not None and not math.isfinite(
            float(self.tactical_head_hint)
        ):
            raise TacticalSequenceError("tactical-head hint must be finite when present")


PolicyRanker = Callable[[TacticalSearchState], Sequence[RankedAction]]


class TacticalGoal(Protocol):
    goal_id: str
    kind: GoalKind

    def satisfied(self, state: TacticalSearchState) -> bool:
        """Return whether public/exact state evidence satisfies this goal."""


@dataclass(frozen=True, slots=True)
class ExactTerminalWinGoal:
    root_actor: int
    goal_id: str = "terminal_win_this_turn"
    kind: GoalKind = "exact_terminal_win"

    def __post_init__(self) -> None:
        if self.root_actor not in (0, 1):
            raise TacticalSequenceError("root actor must be 0 or 1")

    def satisfied(self, state: TacticalSearchState) -> bool:
        return state.terminal_winner == self.root_actor


@dataclass(frozen=True, slots=True)
class PublicFactGoal:
    goal_id: str
    required_facts: Mapping[str, Any]
    kind: GoalKind = "public_fact"

    def __post_init__(self) -> None:
        if not self.goal_id or not self.required_facts:
            raise TacticalSequenceError("public-fact goals require an id and facts")

    def satisfied(self, state: TacticalSearchState) -> bool:
        return bool(state.public_facts_are_observed) and all(
            state.public_facts.get(key) == value
            for key, value in self.required_facts.items()
        )


@dataclass(frozen=True, slots=True)
class VisibleTutorTargetGoal:
    """Shadow goal used only after the real game exposes ``select.deck``."""

    target_card_ids: tuple[int, ...]
    goal_id: str = "visible_tutor_target"
    kind: GoalKind = "visible_tutor_target"

    def __post_init__(self) -> None:
        targets = tuple(int(card) for card in self.target_card_ids)
        if not targets:
            raise TacticalSequenceError("visible tutor goal needs at least one target")
        object.__setattr__(self, "target_card_ids", targets)

    def satisfied(self, state: TacticalSearchState) -> bool:
        if not state.public_facts_are_observed or state.information_boundary:
            return False
        available = set(state.visible_tutor_cards)
        return any(card in available for card in self.target_card_ids)


@dataclass(frozen=True, slots=True)
class TacticalSearchConfig:
    max_depth: int = 8
    max_nodes: int = 256
    max_discrepancies: int = 1
    internal_action_ceiling: int = DEFAULT_INTERNAL_ACTION_CEILING
    wall_seconds: float = 0.25
    shadow_only: bool = True
    allow_deterministic_test_fixture: bool = False

    def __post_init__(self) -> None:
        if self.max_depth < 1 or self.max_nodes < 1:
            raise TacticalSequenceError("depth and node caps must be positive")
        if self.max_discrepancies < 0:
            raise TacticalSequenceError("max_discrepancies cannot be negative")
        if self.internal_action_ceiling < 1:
            raise TacticalSequenceError("internal action ceiling must be positive")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0.0:
            raise TacticalSequenceError("wall_seconds must be finite and positive")
        if not self.shadow_only:
            raise TacticalSequenceError(
                "revision-257 tactical sequence planner is shadow-only"
            )


@dataclass(frozen=True, slots=True)
class TacticalPlanStep:
    from_observation_fingerprint: str
    from_legal_order_fingerprint: str
    action: Action
    policy_rank: int
    policy_probability: float
    sme_priority: float
    tactical_head_hint: float | None
    to_observation_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_observation_fingerprint": self.from_observation_fingerprint,
            "from_legal_order_fingerprint": self.from_legal_order_fingerprint,
            "action": list(self.action),
            "policy_rank": self.policy_rank,
            "policy_probability": self.policy_probability,
            "sme_priority": self.sme_priority,
            "tactical_head_hint": self.tactical_head_hint,
            "to_observation_fingerprint": self.to_observation_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class TacticalPlanResult:
    status: str
    goal_id: str
    goal_kind: GoalKind
    direct_action: Action
    proposed_action: Action | None
    path: tuple[TacticalPlanStep, ...]
    dispatch_authorized: bool
    receipt: Mapping[str, Any]


@dataclass(order=True)
class _FrontierNode:
    priority: tuple[int, float, float, int, int]
    state: TacticalSearchState = field(compare=False)
    path: tuple[TacticalPlanStep, ...] = field(compare=False)
    discrepancies: int = field(compare=False)
    negative_log_policy: float = field(compare=False)
    accumulated_sme_priority: float = field(compare=False)


class TacticalSequencePlanner:
    """Policy-ordered limited-discrepancy search with fail-closed boundaries."""

    def __init__(
        self,
        *,
        backend: TacticalTransitionBackend,
        rank_actions: PolicyRanker,
        config: TacticalSearchConfig | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.rank_actions = rank_actions
        self.config = config or TacticalSearchConfig()
        self.monotonic = monotonic
        mode = getattr(backend, "isolation_mode", None)
        if mode == "deterministic_test_fixture":
            if not self.config.allow_deterministic_test_fixture:
                raise TacticalSequenceError(
                    "deterministic test backends require explicit test-fixture opt-in"
                )
        elif mode != "owned_bounded_child":
            raise TacticalSequenceError(
                "native tactical search requires an owned bounded child process"
            )

    def _ranked(self, state: TacticalSearchState) -> tuple[RankedAction, ...]:
        ranked = tuple(self.rank_actions(state))
        if not ranked:
            raise TacticalSequenceError("policy ranker returned no candidates")
        actions = tuple(row.action for row in ranked)
        if len(actions) != len(set(actions)):
            raise TacticalSequenceError("policy ranker returned duplicate candidates")
        legal = set(state.legal_actions)
        if any(action not in legal for action in actions):
            raise TacticalSequenceError("policy ranker returned an illegal candidate")
        # The typed ranker owns the ordering.  The Alakazam wrapper keeps the
        # r195 principal action first and uses SME scores only among deviations.
        # Neither the ordering nor any score is proof.
        return ranked

    @staticmethod
    def _boundary_reason(
        state: TacticalSearchState,
        *,
        root_actor: int,
        root_turn_id: int | str,
        action_ceiling: int,
    ) -> str | None:
        if state.terminal_winner is not None:
            return "terminal_non_goal"
        if state.actor != root_actor:
            return "actor_change"
        if state.turn_id != root_turn_id:
            return "turn_change"
        if state.explicit_chance_boundary:
            return "explicit_chance_pre_random"
        if state.information_boundary:
            return "information_reobservation"
        if state.ordered_action_count > action_ceiling:
            return "deterministic_internal_fanout_over_64"
        if not state.legal_actions:
            return "no_legal_action"
        return None

    def search(
        self,
        *,
        root: TacticalSearchState,
        direct_action: Sequence[int],
        goal: TacticalGoal,
    ) -> TacticalPlanResult:
        direct = _action(direct_action)
        if direct not in root.legal_actions:
            raise TacticalSequenceError("direct action is not legal at the bound root")
        if isinstance(goal, ExactTerminalWinGoal) and goal.root_actor != root.actor:
            raise TacticalSequenceError("terminal goal actor does not match root actor")

        started = self.monotonic()
        deadline = started + self.config.wall_seconds
        serial = 0
        nodes_expanded = 0
        transitions = 0
        max_depth_seen = 0
        boundary_counts: Counter[str] = Counter()
        seen_best_discrepancy: dict[str, int] = {root.semantic_fingerprint: 0}
        frontier: list[_FrontierNode] = [
            _FrontierNode((0, 0.0, 0.0, 0, serial), root, (), 0, 0.0, 0.0)
        ]
        found_state: TacticalSearchState | None = None
        found_path: tuple[TacticalPlanStep, ...] = ()
        status = "no_goal_found"
        failure: str | None = None

        while frontier and nodes_expanded < self.config.max_nodes:
            if self.monotonic() >= deadline:
                status = "deadline"
                break
            node = heapq.heappop(frontier)
            state = node.state
            max_depth_seen = max(max_depth_seen, len(node.path))

            if goal.kind == "exact_terminal_win" and goal.satisfied(state):
                found_state, found_path = state, node.path
                status = "proven_exact_terminal_win_shadow"
                break

            boundary = self._boundary_reason(
                state,
                root_actor=root.actor,
                root_turn_id=root.turn_id,
                action_ceiling=self.config.internal_action_ceiling,
            )
            if boundary is not None:
                boundary_counts[boundary] += 1
                continue
            if goal.kind != "exact_terminal_win" and goal.satisfied(state):
                found_state, found_path = state, node.path
                status = "public_goal_reached_shadow"
                break
            if len(node.path) >= self.config.max_depth:
                boundary_counts["depth_cap"] += 1
                continue

            nodes_expanded += 1
            try:
                ranked = self._ranked(state)
                if not node.path and ranked[0].action != direct:
                    raise TacticalSequenceError(
                        "root policy principal does not match the bound r195 direct action"
                    )
            except Exception as exc:
                status = "invalid_policy_candidates"
                failure = f"{type(exc).__name__}: {exc}"
                break

            for rank, candidate in enumerate(ranked):
                discrepancies = node.discrepancies + int(rank != 0)
                if discrepancies > self.config.max_discrepancies:
                    continue
                if self.monotonic() >= deadline:
                    status = "deadline"
                    break
                try:
                    transition = self.backend.advance(
                        state,
                        candidate.action,
                        deadline_monotonic=deadline,
                    )
                    transitions += 1
                    if not transition.deterministic:
                        boundary_counts["stochastic_transition"] += 1
                        continue
                    if transition.action_token != candidate.action:
                        raise TacticalSequenceError("simulator action token mismatch")
                    if transition.next_state.previous_action_token != candidate.action:
                        raise TacticalSequenceError(
                            "simulated action was not preserved in previous-action history"
                        )
                except Exception as exc:
                    status = "backend_fault"
                    failure = f"{type(exc).__name__}: {exc}"
                    frontier.clear()
                    break

                step = TacticalPlanStep(
                    from_observation_fingerprint=state.observation_fingerprint,
                    from_legal_order_fingerprint=state.legal_order_fingerprint,
                    action=candidate.action,
                    policy_rank=rank + 1,
                    policy_probability=candidate.probability,
                    sme_priority=candidate.sme_priority,
                    tactical_head_hint=candidate.tactical_head_hint,
                    to_observation_fingerprint=(
                        transition.next_state.observation_fingerprint
                    ),
                )
                path = node.path + (step,)
                next_state = transition.next_state
                prior = seen_best_discrepancy.get(next_state.semantic_fingerprint)
                if prior is not None and prior <= discrepancies:
                    boundary_counts["duplicate_semantic_state"] += 1
                    continue
                seen_best_discrepancy[next_state.semantic_fingerprint] = discrepancies
                serial += 1
                probability = max(candidate.probability, 1e-12)
                negative_log_policy = node.negative_log_policy - math.log(probability)
                accumulated_sme_priority = (
                    node.accumulated_sme_priority + candidate.sme_priority
                )
                heapq.heappush(
                    frontier,
                    _FrontierNode(
                        (
                            discrepancies,
                            -accumulated_sme_priority,
                            negative_log_policy,
                            len(path),
                            serial,
                        ),
                        next_state,
                        path,
                        discrepancies,
                        negative_log_policy,
                        accumulated_sme_priority,
                    ),
                )

        if status == "no_goal_found" and nodes_expanded >= self.config.max_nodes:
            status = "node_cap"
        elapsed = max(0.0, self.monotonic() - started)
        proposed = found_path[0].action if found_path else None
        proof = None
        if found_state is not None and goal.kind == "exact_terminal_win":
            proof = {
                "schema": TACTICAL_SEQUENCE_PROOF_SCHEMA,
                "goal_id": goal.goal_id,
                "root_actor": root.actor,
                "root_observation_fingerprint": root.observation_fingerprint,
                "root_legal_order_fingerprint": root.legal_order_fingerprint,
                "terminal_winner": found_state.terminal_winner,
                "path": [step.to_dict() for step in found_path],
            }
        receipt = {
            "schema": TACTICAL_SEQUENCE_RECEIPT_SCHEMA,
            "mode": "shadow_only",
            "status": status,
            "goal_id": goal.goal_id,
            "goal_kind": goal.kind,
            "root_actor": root.actor,
            "root_observation_fingerprint": root.observation_fingerprint,
            "root_legal_order_fingerprint": root.legal_order_fingerprint,
            "direct_action": list(direct),
            "proposed_action": None if proposed is None else list(proposed),
            "action_changed": proposed is not None and proposed != direct,
            "dispatch_authorized": False,
            "tactical_outcome_head_is_proof": False,
            "backend_isolation_mode": self.backend.isolation_mode,
            "nodes_expanded": nodes_expanded,
            "transitions": transitions,
            "unique_semantic_states": len(seen_best_discrepancy),
            "max_depth_seen": max_depth_seen,
            "boundary_counts": dict(sorted(boundary_counts.items())),
            "elapsed_seconds": elapsed,
            "failure": failure,
            "proof": proof,
        }
        return TacticalPlanResult(
            status=status,
            goal_id=goal.goal_id,
            goal_kind=goal.kind,
            direct_action=direct,
            proposed_action=proposed,
            path=found_path,
            dispatch_authorized=False,
            receipt=receipt,
        )


__all__ = [
    "Action",
    "DEFAULT_INTERNAL_ACTION_CEILING",
    "ExactTerminalWinGoal",
    "PublicFactGoal",
    "RankedAction",
    "TACTICAL_SEQUENCE_PROOF_SCHEMA",
    "TACTICAL_SEQUENCE_RECEIPT_SCHEMA",
    "TacticalPlanResult",
    "TacticalPlanStep",
    "TacticalSearchConfig",
    "TacticalSearchState",
    "TacticalSequenceError",
    "TacticalSequencePlanner",
    "TacticalTransition",
    "TacticalTransitionBackend",
    "VisibleTutorTargetGoal",
    "legal_action_order_fingerprint",
]
