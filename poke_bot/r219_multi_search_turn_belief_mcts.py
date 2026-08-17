"""Local r219 shared-turn controller for approximate BeliefMCTS evaluation.

This module deliberately builds on the inert, dependency-light r215 controller
instead of modifying the sealed r218 first-decision-only experiment.  Its only
authority is local evaluation action selection: no training, service, selector,
Kaggle, or RTP path is imported here.

The controller has one 45-second dynamically shrinking planner pool for an
actual turn.  Each meaningful private MCTS segment may use at most 15 seconds;
cache hops and forced/obvious actions consume only their real validation and
dispatch time.  A malformed, untrusted, or time-starved search always falls
back to the exact frozen direct policy supplied by the runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .r215_full_turn_belief_mcts import (
    R215ActionDecision,
    R215Error,
    R215FullTurnBeliefMCTS,
    R215Observation,
    R215PlannerProtocol,
    R215PlanResult,
    R215TimingConfig,
    R215TurnIdentity,
    _principal_continuation_from_diagnostics,
    _validated_root_stability_receipt,
    canonical_sha256,
)

R219_SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1"
R219_DEFAULT_TURN_POOL_SECONDS = 45.0
R219_DEFAULT_SEARCH_SEGMENT_SECONDS = 15.0
R219_FINITE_CHANCE_OUTCOME_CAP = 6


@dataclass(frozen=True, slots=True)
class R219FiniteChanceOutcome:
    """One engine-attested exact finite-chance child.

    Rational weights avoid floating-point normalization claims.  The controller
    validates only this receipt's structure; it never fabricates an engine
    enumeration capability or calls private simulator methods itself.
    """

    outcome_id: str
    probability_numerator: int
    probability_denominator: int
    successor_public_observation_fingerprint: str
    successor_legal_action_order_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome_id, str) or not self.outcome_id:
            raise R215Error("finite chance outcome_id must be nonempty")
        if int(self.probability_numerator) <= 0 or int(
            self.probability_denominator
        ) <= 0:
            raise R215Error("finite chance probabilities must be positive rationals")
        if int(self.probability_numerator) > int(self.probability_denominator):
            raise R215Error("finite chance probability may not exceed one")
        if not isinstance(
            self.successor_public_observation_fingerprint, str
        ) or not self.successor_public_observation_fingerprint:
            raise R215Error("finite chance successor observation fingerprint is required")
        if not isinstance(
            self.successor_legal_action_order_fingerprint, str
        ) or not self.successor_legal_action_order_fingerprint:
            raise R215Error("finite chance successor legal-order fingerprint is required")

    @property
    def probability(self) -> Fraction:
        return Fraction(
            int(self.probability_numerator), int(self.probability_denominator)
        )


@dataclass(frozen=True, slots=True)
class R219FiniteChanceReceipt:
    """Receipt required before a realized finite chance may retain its cache."""

    engine_force_enumeration_receipt_id: str
    engine_verified_exact_enumeration: bool
    outcomes: tuple[R219FiniteChanceOutcome, ...]
    realized_outcome_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.engine_force_enumeration_receipt_id, str) or not self.engine_force_enumeration_receipt_id:
            raise R215Error("finite chance receipt needs an engine receipt id")
        if self.engine_verified_exact_enumeration is not True:
            raise R215Error("finite chance cache reuse requires an exact engine attestation")
        outcomes = tuple(self.outcomes)
        if not 2 <= len(outcomes) <= R219_FINITE_CHANCE_OUTCOME_CAP:
            raise R215Error("finite chance outcome count is outside the r219 cap")
        if any(not isinstance(outcome, R219FiniteChanceOutcome) for outcome in outcomes):
            raise R215Error("finite chance receipt has an invalid outcome")
        if len({outcome.outcome_id for outcome in outcomes}) != len(outcomes):
            raise R215Error("finite chance outcome ids must be unique")
        if sum((outcome.probability for outcome in outcomes), Fraction(0, 1)) != 1:
            raise R215Error("finite chance probabilities must sum exactly to one")
        if self.realized_outcome_id not in {outcome.outcome_id for outcome in outcomes}:
            raise R215Error("realized finite chance outcome is absent from receipt")
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": "poke_bot.r219_exact_finite_chance_receipt/v1",
                "engine_force_enumeration_receipt_id": self.engine_force_enumeration_receipt_id,
                "engine_verified_exact_enumeration": True,
                "realized_outcome_id": self.realized_outcome_id,
                "outcomes": [
                    {
                        "outcome_id": outcome.outcome_id,
                        "probability_numerator": outcome.probability_numerator,
                        "probability_denominator": outcome.probability_denominator,
                        "successor_public_observation_fingerprint": outcome.successor_public_observation_fingerprint,
                        "successor_legal_action_order_fingerprint": outcome.successor_legal_action_order_fingerprint,
                    }
                    for outcome in self.outcomes
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class R219TimingConfig(R215TimingConfig):
    """Typed r219 values; every field participates in the timing identity."""

    default_turn_pool_seconds: float = R219_DEFAULT_TURN_POOL_SECONDS
    per_operation_ceiling_seconds: float = R219_DEFAULT_SEARCH_SEGMENT_SECONDS
    first_decision_search_ceiling_seconds: float = R219_DEFAULT_SEARCH_SEGMENT_SECONDS
    later_decision_search_ceiling_seconds: float = R219_DEFAULT_SEARCH_SEGMENT_SECONDS
    allow_later_same_turn_search: bool = True
    allow_verified_finite_chance_cache: bool = True
    enforce_component_operation_ceiling: bool = False
    finite_chance_outcome_cap: int = R219_FINITE_CHANCE_OUTCOME_CAP

    def __post_init__(self) -> None:
        R215TimingConfig.__post_init__(self)
        # Custom values are allowed in a separately checksum-bound config.
        if float(self.default_turn_pool_seconds) <= 0.0:
            raise R215Error("r219 turn pool must be positive")
        if float(self.first_decision_search_ceiling_seconds) > float(
            self.default_turn_pool_seconds
        ) or float(self.later_decision_search_ceiling_seconds) > float(
            self.default_turn_pool_seconds
        ):
            raise R215Error("r219 search segment may not exceed its turn pool")
        if not self.allow_later_same_turn_search:
            raise R215Error("r219 requires residual later meaningful searches")
        if not self.allow_verified_finite_chance_cache:
            raise R215Error("r219 requires a verified finite-chance cache gate")
        if self.enforce_component_operation_ceiling:
            raise R215Error("r219 does not impose an inherited per-call hard cap")
        if int(self.finite_chance_outcome_cap) != R219_FINITE_CHANCE_OUTCOME_CAP:
            raise R215Error("r219 finite chance cap must remain six")

    @property
    def identity_sha256(self) -> str:
        # Preserve the parent timing data but give the configuration an r219
        # identity so a 20/10 r218 controller cannot be mistaken for it.
        from dataclasses import asdict

        return canonical_sha256({"schema": R219_SCHEMA, "timing": asdict(self)})


class R219MultiSearchTurnBeliefMCTS(R215FullTurnBeliefMCTS):
    """Residual multi-search actual-turn controller for the r219 local mirror."""

    def __init__(
        self,
        planner: R215PlannerProtocol,
        *,
        direct_policy: Callable[[R215Observation], Sequence[int]],
        timing: R219TimingConfig | None = None,
        game_clock: Any | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        selected_timing = timing or R219TimingConfig()
        kwargs: dict[str, Any] = {
            "direct_policy": direct_policy,
            "timing": selected_timing,
            "game_clock": game_clock,
        }
        if monotonic is not None:
            kwargs["monotonic"] = monotonic
        super().__init__(planner, **kwargs)

    def finish_actual_turn(self) -> Mapping[str, Any] | None:
        """Close one actual r219 turn and expose canary aggregation facts."""

        base = super().finish_actual_turn()
        if base is None:
            return None
        receipt = dict(base)
        receipt.update(
            {
                "schema": R219_SCHEMA,
                "base_controller_schema": base.get("schema"),
                "shared_actual_turn_planner_pool_seconds": (
                    self.timing.default_turn_pool_seconds
                ),
                "per_meaningful_search_segment_ceiling_seconds": (
                    self.timing.later_decision_search_ceiling_seconds
                ),
                "only_first_segment_search": bool(
                    receipt.get("search_segments_this_turn", 0) == 1
                ),
            }
        )
        return receipt

    def act(
        self,
        identity: R215TurnIdentity,
        observation: R215Observation,
        *,
        dispatch: Callable[[list[int]], Any] | None = None,
        boundary_reason: str | None = None,
        finite_chance_receipt: R219FiniteChanceReceipt | None = None,
    ) -> R215ActionDecision:
        if finite_chance_receipt is not None and boundary_reason != "finite_chance":
            raise R215Error("finite chance receipt requires finite_chance boundary")
        if boundary_reason == "finite_chance" and finite_chance_receipt is None:
            verified_outcome = None
            chance_validation = "unavailable_or_unverified_research_boundary"
        elif finite_chance_receipt is not None:
            realized = next(
                outcome
                for outcome in finite_chance_receipt.outcomes
                if outcome.outcome_id == finite_chance_receipt.realized_outcome_id
            )
            if (
                observation.public_observation_fingerprint
                == realized.successor_public_observation_fingerprint
                and observation.legal_action_order_fingerprint
                == realized.successor_legal_action_order_fingerprint
            ):
                verified_outcome = finite_chance_receipt.realized_outcome_id
                chance_validation = "engine_receipt_and_realized_successor_validated"
            else:
                # A structurally valid receipt is not enough to reuse a
                # conditioned cache on a different real observation.  Pass no
                # verified id so the base controller clears it and re-searches
                # from the residual shared pool (or direct-falls back).
                verified_outcome = None
                chance_validation = "engine_receipt_realized_successor_mismatch_research_boundary"
        else:
            verified_outcome = None
            chance_validation = "not_a_finite_chance_boundary"
        decision = super().act(
            identity,
            observation,
            dispatch=dispatch,
            boundary_reason=boundary_reason,
            verified_finite_chance_outcome_id=verified_outcome,
        )
        receipt = dict(decision.receipt)
        receipt.update(
            {
                "schema": R219_SCHEMA,
                "base_controller_schema": receipt.get("schema"),
                "shared_actual_turn_planner_pool_seconds": self.timing.default_turn_pool_seconds,
                "per_meaningful_search_segment_ceiling_seconds": self.timing.later_decision_search_ceiling_seconds,
                "finite_chance_receipt_validation": chance_validation,
                "finite_chance_engine_receipt_id": (
                    finite_chance_receipt.engine_force_enumeration_receipt_id
                    if finite_chance_receipt is not None
                    else None
                ),
                "finite_chance_receipt_identity_sha256": (
                    finite_chance_receipt.identity_sha256
                    if finite_chance_receipt is not None
                    else None
                ),
                "finite_chance_exact_enumeration_engine_attested": bool(
                    finite_chance_receipt is not None
                    and verified_outcome is not None
                ),
                "only_first_segment_search": bool(
                    receipt.get("search_segments_this_turn", 0) == 1
                ),
            }
        )
        return R215ActionDecision(
            selected_action=decision.selected_action,
            source=decision.source,
            receipt=receipt,
        )


def r219_plan_result_from_mcts_result(
    result: Any,
    *,
    selected_action: Sequence[int] | None = None,
    extra_diagnostics: Mapping[str, Any] | None = None,
) -> R215PlanResult:
    """Convert one accepted native MCTS result to the public r219 plan ABI.

    A source-local transactional PolicyAgent planner calls this only after it
    has established that ``result`` is its fresh MCTS result (rather than its
    internal greedy fallback).  The helper preserves the same strict
    convergence receipt and cache parsing rules as the direct BeliefMCTS
    adapter, so runners never need to import r215's internal parser helpers.
    """

    target = getattr(result, "target", None)
    diagnostics_raw = getattr(target, "diagnostics", {})
    diagnostics = (
        dict(diagnostics_raw)
        if isinstance(diagnostics_raw, Mapping)
        else {}
    )
    if extra_diagnostics is not None:
        diagnostics.update(dict(extra_diagnostics))
    result_action = getattr(result, "select", None)
    if selected_action is None:
        selected_action = result_action
    elif result_action is None or tuple(selected_action) != tuple(result_action):
        raise R215Error(
            "transactional planner selected action disagrees with native MCTS result"
        )
    if selected_action is None:
        raise R215Error("native MCTS result has no selected action")
    stability_receipt = _validated_root_stability_receipt(
        diagnostics.get("root_stability_receipt"),
        selected_action=selected_action,
    )
    root_action_stable = bool(
        diagnostics.get(
            "root_action_stable",
            diagnostics.get("root_selected_action_stable", False),
        )
        and stability_receipt is not None
    )
    return R215PlanResult(
        selected_action=tuple(selected_action),
        sims_run=int(getattr(result, "sims_run", 0)),
        continuation=_principal_continuation_from_diagnostics(diagnostics),
        diagnostics=diagnostics,
        root_action_stable=root_action_stable,
        root_stability_receipt=stability_receipt,
    )


def r219_observation_from_raw(
    raw_observation: Mapping[str, Any],
    policy: Any,
) -> R215Observation:
    """Build the exact public/ordered action cache key without model inference."""

    # Imports stay inside the runtime helper so unit tests can import r219
    # without Torch or a native simulator installed.
    from poke_bot import features
    from poke_bot.belief_mcts import information_state_fingerprint

    raw = dict(raw_observation)
    try:
        combinations = features.enumerate_action_combos(raw)
    except Exception as exc:
        raise R215Error("r219 needs a complete ordered legal action space") from exc
    route_snapshot = None
    snapshot = getattr(policy, "matchup_adapter_shadow_snapshot", None)
    if callable(snapshot):
        route_snapshot = snapshot()
    return R215Observation(
        public_observation_fingerprint=information_state_fingerprint(raw),
        legal_actions=tuple(tuple(int(item) for item in action) for action in combinations),
        raw_observation=raw,
        action_space_mode="complete_ordered",
        matchup_adapter_route_receipt=route_snapshot,
    )


def commit_verified_cached_belief_action(
    policy: Any,
    raw_observation: Mapping[str, Any],
    action: Sequence[int],
) -> None:
    """Commit a cache-selected action through the PolicyAgent history path.

    A cache hop must not call greedy selection merely to update history.  This
    mirrors the committed portion of ``PolicyAgent.belief_mcts_select``:
    observe the causal Matchup Adapter route, append the real decision's
    history token, record the public action, then set the previous-action
    token.  It deliberately performs no model forward or private search.
    """

    from poke_bot import features

    raw = dict(raw_observation)
    selected = [int(index) for index in action]
    try:
        legal_actions = features.enumerate_action_combos(raw)
    except Exception as exc:
        raise R215Error("cached action has no complete legal-action proof") from exc
    if selected not in legal_actions:
        raise R215Error("cached action is absent from the fresh legal action order")
    features.assert_info_set(raw)

    router = getattr(policy, "_matchup_adapter_shadow_router", None)
    adapter_enabled = bool(
        getattr(policy, "matchup_adapter_shadow", False)
        or getattr(policy, "matchup_adapter_runtime", False)
    )
    if adapter_enabled:
        observe = getattr(router, "observe", None)
        if not callable(observe):
            raise R215Error("cached action cannot verify the active Matchup Adapter")
        observe(raw, scope="game_root", depth=len(getattr(policy, "board_history", ())))

    required = (
        "deck",
        "board_history",
        "previous_action_history",
        "belief_history",
        "_previous_action_token",
        "_history_context_limit",
    )
    if any(not hasattr(policy, name) for name in required):
        raise R215Error("policy lacks the required belief-history commit surface")
    board = features.build_board_tokens(raw, policy.deck)
    policy.board_history.append(board)
    policy.previous_action_history.append(policy._previous_action_token)
    max_context = int(policy._history_context_limit())
    policy.board_history = policy.board_history[-max_context:]
    policy.previous_action_history = policy.previous_action_history[-max_context:]
    policy.belief_history.record_action(raw, selected)
    policy._previous_action_token = features.build_option_tokens(raw, [selected])
    if hasattr(policy, "last_search_fallback_reason"):
        policy.last_search_fallback_reason = None


class R219PolicyTurnBridge:
    """Thin raw-observation bridge for a PolicyAgent-backed r219 evaluator."""

    _CACHE_COMMIT_SOURCES = frozenset(
        {
            "deterministic_cached_branch",
            "verified_finite_chance_cached_branch",
            "forced_legal_action",
        }
    )

    def __init__(self, controller: R219MultiSearchTurnBeliefMCTS, policy: Any) -> None:
        self.controller = controller
        self.policy = policy

    def act_from_raw(
        self,
        raw_observation: Mapping[str, Any],
        turn_identity: R215TurnIdentity,
        *,
        dispatch: Callable[[list[int]], Any] | None = None,
        boundary_reason: str | None = None,
        finite_chance_receipt: R219FiniteChanceReceipt | None = None,
    ) -> R215ActionDecision:
        bridge_started = self.controller._monotonic()
        observation = r219_observation_from_raw(raw_observation, self.policy)
        before_controller = self.controller._monotonic()
        decision = self.controller.act(
            turn_identity,
            observation,
            boundary_reason=boundary_reason,
            finite_chance_receipt=finite_chance_receipt,
        )
        after_controller = self.controller._monotonic()
        try:
            if decision.source in self._CACHE_COMMIT_SOURCES:
                commit_verified_cached_belief_action(
                    self.policy, raw_observation, decision.selected_action
                )
            if dispatch is not None:
                dispatch(list(decision.selected_action))
        finally:
            bridge_finished = self.controller._monotonic()
            # ``act`` has already charged its own selection interval.  Charge
            # only the non-overlapping raw-observation validation and the
            # post-selection cache commit/real dispatch interval here.
            bridge_pre_controller = max(0.0, before_controller - bridge_started)
            bridge_post_controller = max(0.0, bridge_finished - after_controller)
            bridge_external_elapsed = bridge_pre_controller + bridge_post_controller
            bridge_accounting = self.controller.charge_actual_turn_external_work(
                turn_identity, bridge_external_elapsed
            )
        receipt = dict(decision.receipt)
        receipt.update(bridge_accounting)
        receipt.update(
            {
                "bridge_public_observation_validation_wall_seconds": (
                    bridge_pre_controller
                ),
                "bridge_post_selection_commit_and_dispatch_wall_seconds": (
                    bridge_post_controller
                ),
                "bridge_validation_and_dispatch_wall_seconds": bridge_external_elapsed,
                "turn_planner_wall_seconds": float(
                    receipt.get("turn_planner_wall_seconds", 0.0)
                )
                + bridge_external_elapsed,
                "turn_budget_breach": bool(
                    receipt.get("turn_budget_breach", False)
                    or bridge_accounting["post_dispatch_turn_budget_breach"]
                ),
                "real_actions_dispatched": 1 if dispatch is not None else 0,
            }
        )
        receipt["policy_history_committed_without_model_forward"] = (
            decision.source in self._CACHE_COMMIT_SOURCES
        )
        return R215ActionDecision(
            selected_action=decision.selected_action,
            source=decision.source,
            receipt=receipt,
        )


__all__ = [
    "R219_DEFAULT_SEARCH_SEGMENT_SECONDS",
    "R219_DEFAULT_TURN_POOL_SECONDS",
    "R219_FINITE_CHANCE_OUTCOME_CAP",
    "R219_SCHEMA",
    "R219FiniteChanceOutcome",
    "R219FiniteChanceReceipt",
    "R219MultiSearchTurnBeliefMCTS",
    "R219PolicyTurnBridge",
    "R219TimingConfig",
    "commit_verified_cached_belief_action",
    "r219_observation_from_raw",
    "r219_plan_result_from_mcts_result",
]
