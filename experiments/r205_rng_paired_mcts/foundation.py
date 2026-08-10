"""Fail-closed substrate for a simulator-backed r205 one-turn search.

This is deliberately an **experiment-only** module.  It does not import the
competition ``cg`` package, load an RTP sidecar, start a service, or attach an
action override.  The only action it ever returns for execution is the supplied
exact direct-policy action.  A completed search is emitted as a shadow
recommendation so a later, separately receipted integration can inspect the
actual simulator work it performed.

The important distinction is structural rather than documentary:

* :class:`SealedStartMaterial` is a root-game material.  It can restore a fresh
  game only at its sealed BattleStart boundary.
* :class:`MidgameCloneCapability` is required before any search branch can be
  made from an arbitrary current decision.  A replay-from-sealed-start facility
  is rejected even if it happens to expose a similarly named method.

The tree builder is a bounded, full-expansion one-turn expectimax profile that
uses *real ``clone`` + ``step`` calls for every decision edge.  It is suitable
as the simulator/backup core underneath a later MCTS control policy, and it
already satisfies the stricter property that chance outcomes are enumerated
and backed up exactly rather than sampled.  It is not a r205 launch receipt or
an action-authoritative MCTS implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol, TypeAlias

Action: TypeAlias = tuple[int, ...]
TurnKey: TypeAlias = tuple[int, int]
StateToken: TypeAlias = object

MCTS_ARM = "chance_aware_inter_turn_mcts"
DIRECT_ARM = "no_rtp_direct_policy"
_MAX_COMPLETE_ACTIONS = 1024
_SHA256_PREFIX = "sha256:"


class R205FoundationError(ValueError):
    """A simulator/search artifact breaks a fail-closed r205 invariant."""


class MidgameCloneUnavailable(R205FoundationError):
    """The supplied facility is not an attested arbitrary-midgame cloner."""


class SearchDeadlineExceeded(R205FoundationError):
    """A charged search operation consumed the per-turn or per-action budget."""


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        raise R205FoundationError(f"{name} must be a sha256 digest")
    if len(value) != 71:
        raise R205FoundationError(f"{name} must be a sha256 digest")
    try:
        int(value[len(_SHA256_PREFIX) :], 16)
    except ValueError as exc:
        raise R205FoundationError(f"{name} must be a sha256 digest") from exc
    return value


def _normalise_action(action: Action, *, name: str) -> Action:
    if not isinstance(action, tuple):
        raise R205FoundationError(f"{name} must be a tuple")
    if any(type(component) is not int or component < 0 for component in action):
        raise R205FoundationError(
            f"{name} must contain only non-negative exact integer option indexes"
        )
    return action


def complete_action_fingerprint(actions: tuple[Action, ...]) -> str:
    """Hash the ordered complete-action list without dropping ``()`` actions."""

    normalised = tuple(
        _normalise_action(action, name=f"actions[{index}]")
        for index, action in enumerate(actions)
    )
    if not normalised:
        raise R205FoundationError("a decision must expose at least one legal action")
    if len(set(normalised)) != len(normalised):
        raise R205FoundationError("complete legal actions must be unique")
    return _sha256(
        {
            "schema": "poke_bot.r205.complete_ordered_actions/v1",
            "actions": [list(action) for action in normalised],
        }
    )


@dataclass(frozen=True, slots=True)
class SearchView:
    """The entire policy-visible view available to the search controller."""

    state_id: str
    turn_key: TurnKey
    acting_seat: int
    legal_actions: tuple[Action, ...]
    direct_action: Action
    observation_sha256: str
    legal_actions_sha256: str
    option_encoding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise R205FoundationError("state_id must be non-empty")
        if (
            not isinstance(self.turn_key, tuple)
            or len(self.turn_key) != 2
            or any(type(value) is not int or value < 0 for value in self.turn_key)
        ):
            raise R205FoundationError("turn_key must be a non-negative (seat, turn) tuple")
        if type(self.acting_seat) is not int or self.acting_seat not in (0, 1):
            raise R205FoundationError("acting_seat must be exactly 0 or 1")
        legal_actions = tuple(
            _normalise_action(action, name=f"legal_actions[{index}]")
            for index, action in enumerate(self.legal_actions)
        )
        if not legal_actions:
            raise R205FoundationError("a decision must expose at least one legal action")
        if len(set(legal_actions)) != len(legal_actions):
            raise R205FoundationError("complete legal actions must be unique")
        direct_action = _normalise_action(self.direct_action, name="direct_action")
        if direct_action not in legal_actions:
            raise R205FoundationError("direct_action must be in the complete legal actions")
        if complete_action_fingerprint(legal_actions) != self.legal_actions_sha256:
            raise R205FoundationError("legal action fingerprint does not match exact actions")
        for name in (
            "observation_sha256",
            "legal_actions_sha256",
            "option_encoding_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "direct_action", direct_action)


@dataclass(frozen=True, slots=True)
class MidgameCloneCapability:
    """Receipt facts required for an arbitrary policy-visible decision branch.

    The capability must mean an independent native clone of the actual current
    state.  A sealed BattleStart replay is intentionally represented by a
    different ``source_kind`` and can never satisfy this class's validator.
    """

    abi_name: str
    abi_version: int
    source_kind: str
    status: str
    arbitrary_midgame_policy_visible_decision: bool
    full_state_game_rng_config_counters: bool
    exact_future_legality: bool
    information_set_safe: bool
    independent_clone: bool
    engine_artifact_sha256: str

    def require_usable(self) -> None:
        if self.source_kind != "native_midgame_clone":
            raise MidgameCloneUnavailable(
                "replay-from-sealed-start is not a native arbitrary-midgame clone"
            )
        if self.status != "passed":
            raise MidgameCloneUnavailable("native midgame clone capability is not passed")
        if not isinstance(self.abi_name, str) or not self.abi_name:
            raise MidgameCloneUnavailable("native midgame clone ABI name is missing")
        if type(self.abi_version) is not int or self.abi_version <= 0:
            raise MidgameCloneUnavailable("native midgame clone ABI version is invalid")
        _require_digest(self.engine_artifact_sha256, name="engine_artifact_sha256")
        required = (
            self.arbitrary_midgame_policy_visible_decision,
            self.full_state_game_rng_config_counters,
            self.exact_future_legality,
            self.information_set_safe,
            self.independent_clone,
        )
        if not all(required):
            raise MidgameCloneUnavailable(
                "native clone lacks arbitrary state/RNG/future-legality/information-set proof"
            )


@dataclass(frozen=True, slots=True)
class SealedStartMaterial:
    """One immutable post-BattleStart root used by both games in a pair.

    It is deliberately *not* a ``MidgameCloneCapability``.  The same root
    material may be restored afresh for the two seat-swapped games, but it may
    not be replayed through an action history and relabelled as a branch from a
    later decision.
    """

    pair_id: str
    snapshot_sha256: str
    initial_rng_state_sha256: str
    deck_order_randomness_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise R205FoundationError("pair_id must be non-empty")
        for name in (
            "snapshot_sha256",
            "initial_rng_state_sha256",
            "deck_order_randomness_sha256",
        ):
            _require_digest(getattr(self, name), name=name)

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            {
                "schema": "poke_bot.r205.sealed_start_material/v1",
                "pair_id": self.pair_id,
                "snapshot_sha256": self.snapshot_sha256,
                "initial_rng_state_sha256": self.initial_rng_state_sha256,
                "deck_order_randomness_sha256": self.deck_order_randomness_sha256,
            }
        )


class SealedStartRestorer(Protocol):
    """Root-only interface.  It intentionally has no ``clone`` method."""

    def restore_sealed_start(self, material: SealedStartMaterial) -> StateToken:
        """Restore an independently fresh game at the sealed start boundary."""


class SimulatorBranch(Protocol):
    """A temporary independent native branch produced by a real clone ABI."""

    def step(self, action: Action) -> Transition:
        """Apply one complete legal action in this branch."""

    def close(self) -> None:
        """Release this temporary branch without changing the parent state."""


class NativeMidgameForker(Protocol):
    """The only interface accepted for same-turn search branching."""

    capability: MidgameCloneCapability

    def clone(self, state: StateToken) -> SimulatorBranch:
        """Return a fresh, independent clone of ``state``."""


@dataclass(frozen=True, slots=True)
class DeterministicTransition:
    next_state: StateToken
    successor_certificate_sha256: str
    future_legality_sha256: str

    def __post_init__(self) -> None:
        _require_digest(self.successor_certificate_sha256, name="successor_certificate_sha256")
        _require_digest(self.future_legality_sha256, name="future_legality_sha256")


@dataclass(frozen=True, slots=True)
class TerminalTransition:
    value: Fraction
    reason: str = "terminal"

    def __post_init__(self) -> None:
        if not isinstance(self.value, Fraction):
            raise R205FoundationError("terminal value must be an exact Fraction")
        if not isinstance(self.reason, str) or not self.reason:
            raise R205FoundationError("terminal reason must be non-empty")


@dataclass(frozen=True, slots=True)
class BoundaryTransition:
    """A chance/information boundary that cannot support an override."""

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise R205FoundationError("boundary reason must be non-empty")


@dataclass(frozen=True, slots=True)
class ExactChanceOutcome:
    label: str
    probability: Fraction
    next_state: StateToken
    future_legality_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise R205FoundationError("chance outcome label must be non-empty")
        if not isinstance(self.probability, Fraction) or self.probability <= 0:
            raise R205FoundationError("chance probability must be a positive exact Fraction")
        _require_digest(self.future_legality_sha256, name="future_legality_sha256")


@dataclass(frozen=True, slots=True)
class ExactChanceTransition:
    """A complete engine-enumerated finite chance distribution.

    ``outcomes`` must already be independently materialized successor states
    from an exact engine enumeration.  This module neither samples one outcome
    nor fabricates a child by replaying the BattleStart root.
    """

    event_id: str
    distribution_receipt_sha256: str
    outcomes: tuple[ExactChanceOutcome, ...]
    policy_visible_only: bool = True
    fully_enumerated: bool = True
    future_legality_attested: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise R205FoundationError("chance event_id must be non-empty")
        _require_digest(self.distribution_receipt_sha256, name="distribution_receipt_sha256")
        if not self.policy_visible_only:
            raise R205FoundationError("chance expansion cannot consume hidden information")
        if not self.fully_enumerated:
            raise R205FoundationError("chance distribution is not fully enumerated")
        if not self.future_legality_attested:
            raise R205FoundationError("chance outcomes lack future-legality attestation")
        if not isinstance(self.outcomes, tuple) or len(self.outcomes) < 2:
            raise R205FoundationError("finite chance needs at least two outcomes")
        labels = tuple(outcome.label for outcome in self.outcomes)
        if len(set(labels)) != len(labels):
            raise R205FoundationError("chance outcome labels must be unique")
        if sum((outcome.probability for outcome in self.outcomes), Fraction(0)) != 1:
            raise R205FoundationError("chance probabilities must sum exactly to one")


Transition: TypeAlias = (
    DeterministicTransition | TerminalTransition | BoundaryTransition | ExactChanceTransition
)


@dataclass(frozen=True, slots=True)
class OneTurnSearchConfig:
    """One identity-bound r205 budget surface for the isolated foundation."""

    max_turn_seconds: float = 20.0
    max_action_seconds: float = 5.0
    max_complete_actions: int = _MAX_COMPLETE_ACTIONS
    max_depth: int = 16
    max_tree_nodes: int = 4096

    def __post_init__(self) -> None:
        for name in ("max_turn_seconds", "max_action_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise R205FoundationError(f"{name} must be finite")
            if float(value) <= 0.0:
                raise R205FoundationError(f"{name} must be positive")
        if self.max_action_seconds > self.max_turn_seconds:
            raise R205FoundationError("per-action budget cannot exceed per-turn budget")
        if (
            type(self.max_complete_actions) is not int
            or not 1 <= self.max_complete_actions <= _MAX_COMPLETE_ACTIONS
        ):
            raise R205FoundationError("max_complete_actions must be an int in [1, 1024]")
        if type(self.max_depth) is not int or self.max_depth < 1:
            raise R205FoundationError("max_depth must be a positive exact int")
        if type(self.max_tree_nodes) is not int or self.max_tree_nodes < 1:
            raise R205FoundationError("max_tree_nodes must be a positive exact int")

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            {
                "schema": "poke_bot.r205.one_turn_search_config/v1",
                "max_turn_seconds": float(self.max_turn_seconds),
                "max_action_seconds": float(self.max_action_seconds),
                "max_complete_actions": self.max_complete_actions,
                "max_depth": self.max_depth,
                "max_tree_nodes": self.max_tree_nodes,
            }
        )


@dataclass(slots=True)
class SearchTelemetry:
    """Raw per-turn telemetry aligned to the r205 reporting requirements."""

    actual_native_clone_calls: int = 0
    actual_simulator_step_calls: int = 0
    result_or_leaf_evaluations_seen: int = 0
    decision_nodes_expanded: int = 0
    terminal_results_seen: int = 0
    boundary_leaf_results_seen: int = 0
    finite_chance_outcomes_evaluated: int = 0
    deadline_hit: bool = False
    turn_budget_hit: bool = False
    action_budget_hit: bool = False
    tree_incomplete_reason: str | None = None
    _state_ids: set[str] = field(default_factory=set)

    @property
    def unique_tree_nodes_seen(self) -> int:
        return len(self._state_ids)


@dataclass(frozen=True, slots=True)
class ActionValue:
    action: Action
    value: Fraction | None
    complete: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class ShadowSearchDecision:
    """Result of an isolated search; execution remains direct-policy only."""

    executed_action: Action
    direct_action: Action
    shadow_recommended_action: Action | None
    action_values: tuple[ActionValue, ...]
    requested_tree_fully_expanded_and_backed_up_within_budget: bool
    direct_fallback_used: bool
    reason: str | None
    telemetry: SearchTelemetry
    config_sha256: str
    action_authority_enabled: bool = False


@dataclass(frozen=True, slots=True)
class _Evaluation:
    value: Fraction | None
    complete: bool
    reason: str | None = None


class _RuntimeBudget:
    def __init__(
        self,
        config: OneTurnSearchConfig,
        clock: Callable[[], float],
        telemetry: SearchTelemetry,
    ) -> None:
        self._config = config
        self._clock = clock
        self._telemetry = telemetry
        self.turn_started = float(clock())
        self.action_started = self.turn_started

    def check(self, operation: str) -> None:
        now = float(self._clock())
        turn_elapsed = now - self.turn_started
        action_elapsed = now - self.action_started
        if turn_elapsed > self._config.max_turn_seconds:
            self._telemetry.deadline_hit = True
            self._telemetry.turn_budget_hit = True
            raise SearchDeadlineExceeded(f"turn_budget_exhausted:{operation}")
        if action_elapsed > self._config.max_action_seconds:
            self._telemetry.deadline_hit = True
            self._telemetry.action_budget_hit = True
            raise SearchDeadlineExceeded(f"action_budget_exhausted:{operation}")


class SimulatorBackedOneTurnMCTS:
    """Actual clone-and-step one-turn expectimax substrate, shadow-only.

    Each edge expansion obtains a new clone through ``forker.clone`` and calls
    ``branch.step``.  No action edge is supplied as a prebuilt tree.  Exact
    finite chance is fully enumerated; boundaries, clone gaps, malformed
    successor evidence, and deadlines all discard the shadow tree and retain
    the exact direct-policy action.
    """

    def __init__(
        self,
        *,
        forker: NativeMidgameForker | None,
        view_of: Callable[[StateToken], SearchView],
        leaf_value: Callable[[StateToken, SearchView], Fraction],
        config: OneTurnSearchConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._forker = forker
        self._view_of = view_of
        self._leaf_value = leaf_value
        self.config = config or OneTurnSearchConfig()
        self._clock = clock
        self._budget: _RuntimeBudget | None = None
        self._telemetry: SearchTelemetry | None = None

    def plan(self, root_state: StateToken) -> ShadowSearchDecision:
        """Build a shadow tree, but always execute the root direct action."""

        root = self._view(root_state)
        telemetry = SearchTelemetry()
        self._telemetry = telemetry
        self._budget = _RuntimeBudget(self.config, self._clock, telemetry)

        try:
            self._require_native_midgame_forker()
        except MidgameCloneUnavailable as exc:
            return self._direct_only(root, telemetry, reason=str(exc), fallback=True)

        try:
            evaluation, actions = self._expand_decision(
                root_state,
                root_turn_key=root.turn_key,
                controlled_seat=root.acting_seat,
                depth=0,
                path=frozenset(),
            )
        except SearchDeadlineExceeded as exc:
            return self._direct_only(root, telemetry, reason=str(exc), fallback=True)
        except R205FoundationError as exc:
            return self._direct_only(
                root,
                telemetry,
                reason=f"search_contract_violation:{exc}",
                fallback=True,
            )
        finally:
            self._budget = None

        if not evaluation.complete:
            return self._direct_only(
                root,
                telemetry,
                action_values=actions,
                reason=evaluation.reason or "tree_not_fully_expanded",
                fallback=True,
            )

        complete_values = tuple(item for item in actions if item.complete and item.value is not None)
        if not complete_values:
            return self._direct_only(
                root,
                telemetry,
                action_values=actions,
                reason="tree_has_no_complete_root_action",
                fallback=True,
            )
        best_value = max(item.value for item in complete_values if item.value is not None)
        # Preserve exact direct action on an exact tie.  It is mandatory as a
        # candidate but this isolated module still leaves execution to direct.
        best = next(
            item
            for item in complete_values
            if item.action == root.direct_action and item.value == best_value
        ) if any(
            item.action == root.direct_action and item.value == best_value
            for item in complete_values
        ) else next(item for item in complete_values if item.value == best_value)
        return ShadowSearchDecision(
            executed_action=root.direct_action,
            direct_action=root.direct_action,
            shadow_recommended_action=best.action,
            action_values=actions,
            requested_tree_fully_expanded_and_backed_up_within_budget=True,
            direct_fallback_used=False,
            reason=None,
            telemetry=telemetry,
            config_sha256=self.config.identity_sha256,
        )

    def _direct_only(
        self,
        root: SearchView,
        telemetry: SearchTelemetry,
        *,
        reason: str,
        fallback: bool,
        action_values: tuple[ActionValue, ...] = (),
    ) -> ShadowSearchDecision:
        telemetry.tree_incomplete_reason = reason
        return ShadowSearchDecision(
            executed_action=root.direct_action,
            direct_action=root.direct_action,
            shadow_recommended_action=None,
            action_values=action_values,
            requested_tree_fully_expanded_and_backed_up_within_budget=False,
            direct_fallback_used=fallback,
            reason=reason,
            telemetry=telemetry,
            config_sha256=self.config.identity_sha256,
        )

    def _require_native_midgame_forker(self) -> NativeMidgameForker:
        if self._forker is None:
            raise MidgameCloneUnavailable("native_midgame_clone_unavailable")
        capability = getattr(self._forker, "capability", None)
        if not isinstance(capability, MidgameCloneCapability):
            raise MidgameCloneUnavailable("midgame clone capability is absent or malformed")
        capability.require_usable()
        return self._forker

    def _view(self, state: StateToken) -> SearchView:
        view = self._view_of(state)
        if not isinstance(view, SearchView):
            raise R205FoundationError("view_of must return SearchView")
        return view

    def _check(self, operation: str) -> None:
        if self._budget is None:
            raise R205FoundationError("search budget is not active")
        self._budget.check(operation)

    def _expand_decision(
        self,
        state: StateToken,
        *,
        root_turn_key: TurnKey,
        controlled_seat: int,
        depth: int,
        path: frozenset[str],
    ) -> tuple[_Evaluation, tuple[ActionValue, ...]]:
        self._check("decision_entry")
        view = self._view(state)
        telemetry = self._telemetry
        if telemetry is None:
            raise R205FoundationError("telemetry is not active")
        telemetry._state_ids.add(view.state_id)

        # One-turn scope ends at the first exact real turn-key transition.
        if view.turn_key != root_turn_key:
            return self._leaf(state, view, reason="turn_boundary"), ()
        if view.acting_seat != controlled_seat:
            return (
                _Evaluation(
                    None,
                    False,
                    "opponent_or_private_information_without_receipted_distribution",
                ),
                (),
            )
        if depth >= self.config.max_depth:
            return _Evaluation(None, False, "max_depth_reached"), ()
        if view.state_id in path:
            return _Evaluation(None, False, "same_turn_cycle_detected"), ()
        if len(telemetry._state_ids) > self.config.max_tree_nodes:
            return _Evaluation(None, False, "max_tree_nodes_reached"), ()
        if len(view.legal_actions) > self.config.max_complete_actions:
            return _Evaluation(None, False, "complete_action_cap_exceeded"), ()

        telemetry.decision_nodes_expanded += 1
        child_path = path | {view.state_id}
        action_values: list[ActionValue] = []
        for action in view.legal_actions:
            transition = self._fork_and_step(state, action)
            child = self._evaluate_transition(
                transition,
                root_turn_key=root_turn_key,
                controlled_seat=controlled_seat,
                depth=depth + 1,
                path=child_path,
            )
            action_values.append(
                ActionValue(
                    action=action,
                    value=child.value,
                    complete=child.complete,
                    reason=child.reason,
                )
            )

        values = tuple(action_values)
        if any(not item.complete for item in values):
            first_reason = next(item.reason for item in values if not item.complete)
            return _Evaluation(None, False, first_reason), values
        if not values or any(item.value is None for item in values):
            return _Evaluation(None, False, "missing_complete_action_value"), values
        best_value = max(item.value for item in values if item.value is not None)
        return _Evaluation(best_value, True), values

    def _fork_and_step(self, state: StateToken, action: Action) -> Transition:
        self._check("before_native_clone")
        forker = self._require_native_midgame_forker()
        telemetry = self._telemetry
        if telemetry is None:
            raise R205FoundationError("telemetry is not active")
        try:
            branch = forker.clone(state)
        except Exception as exc:  # native adapters must fail closed into direct action
            raise R205FoundationError("native_midgame_clone_failed") from exc
        telemetry.actual_native_clone_calls += 1
        try:
            self._check("before_simulator_step")
            transition = branch.step(action)
            telemetry.actual_simulator_step_calls += 1
            self._check("after_simulator_step")
        except SearchDeadlineExceeded:
            raise
        except Exception as exc:
            raise R205FoundationError("simulator_step_failed") from exc
        finally:
            try:
                branch.close()
            except Exception as exc:
                raise R205FoundationError("simulator_branch_close_failed") from exc
        if not isinstance(
            transition,
            (
                DeterministicTransition,
                TerminalTransition,
                BoundaryTransition,
                ExactChanceTransition,
            ),
        ):
            raise R205FoundationError("simulator step returned an untyped transition")
        return transition

    def _evaluate_transition(
        self,
        transition: Transition,
        *,
        root_turn_key: TurnKey,
        controlled_seat: int,
        depth: int,
        path: frozenset[str],
    ) -> _Evaluation:
        self._check("transition_backup")
        telemetry = self._telemetry
        if telemetry is None:
            raise R205FoundationError("telemetry is not active")
        if isinstance(transition, TerminalTransition):
            telemetry.terminal_results_seen += 1
            telemetry.result_or_leaf_evaluations_seen += 1
            return _Evaluation(transition.value, True)
        if isinstance(transition, BoundaryTransition):
            telemetry.boundary_leaf_results_seen += 1
            return _Evaluation(None, False, f"chance_or_information_boundary:{transition.reason}")
        if isinstance(transition, DeterministicTransition):
            successor_view = self._view(transition.next_state)
            if successor_view.legal_actions_sha256 != transition.future_legality_sha256:
                return _Evaluation(None, False, "deterministic_future_legality_mismatch")
            child, _ = self._expand_decision(
                transition.next_state,
                root_turn_key=root_turn_key,
                controlled_seat=controlled_seat,
                depth=depth,
                path=path,
            )
            return child

        outcome_values: list[Fraction] = []
        for outcome in transition.outcomes:
            self._check("exact_chance_outcome")
            outcome_view = self._view(outcome.next_state)
            if outcome_view.legal_actions_sha256 != outcome.future_legality_sha256:
                return _Evaluation(None, False, "chance_future_legality_mismatch")
            telemetry.finite_chance_outcomes_evaluated += 1
            child, _ = self._expand_decision(
                outcome.next_state,
                root_turn_key=root_turn_key,
                controlled_seat=controlled_seat,
                depth=depth,
                path=path,
            )
            if not child.complete or child.value is None:
                return _Evaluation(None, False, child.reason or "incomplete_chance_child")
            outcome_values.append(outcome.probability * child.value)
        # Exact Fraction arithmetic is deliberate: no sampling or reweighting.
        return _Evaluation(sum(outcome_values, Fraction(0)), True)

    def _leaf(self, state: StateToken, view: SearchView, *, reason: str) -> _Evaluation:
        self._check(f"leaf_value:{reason}")
        value = self._leaf_value(state, view)
        if not isinstance(value, Fraction):
            raise R205FoundationError("leaf_value must return an exact Fraction")
        telemetry = self._telemetry
        if telemetry is None:
            raise R205FoundationError("telemetry is not active")
        telemetry.result_or_leaf_evaluations_seen += 1
        return _Evaluation(value, True)


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """One game in the exact r205 seat-swapped BO1000 design."""

    game_id: str
    pair_id: str
    mcts_seat: int
    start_material: SealedStartMaterial

    def __post_init__(self) -> None:
        if not isinstance(self.game_id, str) or not self.game_id:
            raise R205FoundationError("game_id must be non-empty")
        if self.pair_id != self.start_material.pair_id:
            raise R205FoundationError("scheduled game pair_id must match its start material")
        if self.mcts_seat not in (0, 1):
            raise R205FoundationError("mcts_seat must be 0 or 1")

    @property
    def arms_by_seat(self) -> tuple[str, str]:
        return (MCTS_ARM, DIRECT_ARM) if self.mcts_seat == 0 else (DIRECT_ARM, MCTS_ARM)


@dataclass(frozen=True, slots=True)
class BO1000ScheduleSummary:
    total_games: int
    matched_rng_pairs: int
    mcts_as_seat_0: int
    mcts_as_seat_1: int


def build_seat_swapped_bo1000_schedule(
    materials: Iterable[SealedStartMaterial],
) -> tuple[ScheduledGame, ...]:
    """Build exactly 500 material-matched pairs / 1,000 games.

    Both games in each pair restore the same sealed start material.  The arms,
    not the deck/RNG material, are swapped between seats.  This truthfully
    pairs the immutable initial state; divergent policies are *not* claimed to
    share later chance outcomes after their action histories diverge.
    """

    ordered = tuple(materials)
    if len(ordered) != 500:
        raise R205FoundationError("BO1000 requires exactly 500 sealed start materials")
    pair_ids = tuple(material.pair_id for material in ordered)
    if len(set(pair_ids)) != len(pair_ids):
        raise R205FoundationError("BO1000 start materials require unique pair ids")
    games: list[ScheduledGame] = []
    for index, material in enumerate(ordered):
        suffix = f"{index:04d}"
        games.append(
            ScheduledGame(
                game_id=f"r205-{suffix}-mcts-seat0",
                pair_id=material.pair_id,
                mcts_seat=0,
                start_material=material,
            )
        )
        games.append(
            ScheduledGame(
                game_id=f"r205-{suffix}-mcts-seat1",
                pair_id=material.pair_id,
                mcts_seat=1,
                start_material=material,
            )
        )
    validate_bo1000_schedule(games)
    return tuple(games)


def validate_bo1000_schedule(games: Iterable[ScheduledGame]) -> BO1000ScheduleSummary:
    """Fail closed on missing, duplicated, crossed, or non-seat-balanced pairs."""

    rows = tuple(games)
    if len(rows) != 1000:
        raise R205FoundationError("BO1000 requires exactly 1,000 games")
    ids = tuple(row.game_id for row in rows)
    if len(set(ids)) != len(ids):
        raise R205FoundationError("BO1000 game ids must be unique")
    by_pair: dict[str, list[ScheduledGame]] = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, []).append(row)
    if len(by_pair) != 500:
        raise R205FoundationError("BO1000 requires exactly 500 unique pair ids")
    for pair_id, pair_rows in by_pair.items():
        if len(pair_rows) != 2:
            raise R205FoundationError(f"pair {pair_id} does not have exactly two games")
        if {row.mcts_seat for row in pair_rows} != {0, 1}:
            raise R205FoundationError(f"pair {pair_id} is not seat-swapped")
        material_ids = {row.start_material.identity_sha256 for row in pair_rows}
        if len(material_ids) != 1:
            raise R205FoundationError(f"pair {pair_id} crossed its sealed RNG start material")
        if any(row.arms_by_seat[row.mcts_seat] != MCTS_ARM for row in pair_rows):
            raise R205FoundationError(f"pair {pair_id} does not place MCTS in its declared seat")
    mcts_seat_0 = sum(row.mcts_seat == 0 for row in rows)
    mcts_seat_1 = sum(row.mcts_seat == 1 for row in rows)
    if mcts_seat_0 != 500 or mcts_seat_1 != 500:
        raise R205FoundationError("MCTS must occupy each seat exactly 500 times")
    return BO1000ScheduleSummary(
        total_games=len(rows),
        matched_rng_pairs=len(by_pair),
        mcts_as_seat_0=mcts_seat_0,
        mcts_as_seat_1=mcts_seat_1,
    )


def restore_scheduled_game_root(
    restorer: SealedStartRestorer,
    game: ScheduledGame,
) -> StateToken:
    """Use a sealed root only for a fresh scheduled game, never for a branch."""

    if not hasattr(restorer, "restore_sealed_start"):
        raise R205FoundationError("scheduled root restorer lacks restore_sealed_start")
    return restorer.restore_sealed_start(game.start_material)


__all__ = [
    "DIRECT_ARM",
    "MCTS_ARM",
    "Action",
    "ActionValue",
    "BO1000ScheduleSummary",
    "BoundaryTransition",
    "DeterministicTransition",
    "ExactChanceOutcome",
    "ExactChanceTransition",
    "MidgameCloneCapability",
    "MidgameCloneUnavailable",
    "NativeMidgameForker",
    "OneTurnSearchConfig",
    "R205FoundationError",
    "ScheduledGame",
    "SealedStartMaterial",
    "SealedStartRestorer",
    "SearchTelemetry",
    "SearchView",
    "ShadowSearchDecision",
    "SimulatorBackedOneTurnMCTS",
    "TerminalTransition",
    "Transition",
    "build_seat_swapped_bo1000_schedule",
    "complete_action_fingerprint",
    "restore_scheduled_game_root",
    "validate_bo1000_schedule",
]
