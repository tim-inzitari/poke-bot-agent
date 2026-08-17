"""Small, fail-closed controller for the r215 full-turn BeliefMCTS mirror.

This module owns *execution accounting*, not an alternate battle engine.  A
private planner receives a :class:`R215PlanRequest` and may return one next
action plus a verified deterministic continuation.  The controller is the
only code allowed to call the real-game dispatcher.  This keeps hypothetical
search actions private and makes the full-turn cache independently auditable.

The controller deliberately does not implement a public-observation
transposition table.  The packaged search API does not attest complete native
state identity (hidden state, RNG, pending effects, configuration and future
legal order), so cross-order merging is permanently disabled for r215.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:  # Keep the hermetic controller importable without Torch.
    from .belief_mcts import BeliefMCTS


R215_SCHEMA = "poke_bot.alakazam_full_turn_belief_mcts/v1"
R215_CHANCE_LABEL = "root_sampled_belief_mcts_non_r207_exact_chance"
R215_DEFAULT_GAME_SECONDS = 600.0
R215_DEFAULT_GAME_RESERVE_SECONDS = 30.0
R215_DEFAULT_EXPECTED_SEARCH_DECISIONS = 64
R215_DEFAULT_MINIMUM_REMAINING_DECISION_DIVISOR = 8
R215_DEFAULT_TURN_POOL_SECONDS = 20.0
R215_DEFAULT_OPERATION_SECONDS = 5.0
R215_DEFAULT_FIRST_DECISION_SEARCH_SECONDS = 10.0
R215_DEFAULT_LATER_DECISION_SEARCH_SECONDS = R215_DEFAULT_OPERATION_SECONDS
R215_MINIMUM_VALID_SIMULATIONS = 1
R215_EMERGENCY_SIMULATION_SAFETY_CEILING = 1_000_000
R215_EMERGENCY_DEPTH_SAFETY_CEILING = 1_000_000


class R215Error(ValueError):
    """An r215 boundary or receipt is malformed."""


@dataclass(slots=True)
class R215GameClock:
    """Dependency-light equivalent of ``mcts.GameClock`` for r215 control.

    Production may supply the project ``GameClock`` directly.  Keeping this
    tiny clock here makes the receipt/controller testable on a host that does
    not import the Torch-dependent search implementation.
    """

    total_s: float = R215_DEFAULT_GAME_SECONDS
    reserve_s: float = R215_DEFAULT_GAME_RESERVE_SECONDS
    expected_search_decisions: int = R215_DEFAULT_EXPECTED_SEARCH_DECISIONS
    remaining_s: float = field(init=False)
    decisions_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.remaining_s = float(self.total_s)

    def consume(self, used: float) -> None:
        self.remaining_s = max(0.0, self.remaining_s - max(0.0, float(used)))
        self.decisions_used += 1


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(payload: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _action_tuple(action: Sequence[int]) -> tuple[int, ...]:
    if isinstance(action, (str, bytes)):
        raise R215Error("an action must be a sequence of integer option indices")
    result = tuple(action)
    if not result or any(type(item) is not int or item < 0 for item in result):
        raise R215Error("an action must be a nonempty sequence of nonnegative ints")
    return result


def legal_action_order_fingerprint(actions: Sequence[Sequence[int]]) -> str:
    """Fingerprint the complete ordered legal space, not an unordered set."""

    return canonical_sha256(
        {
            "schema": "poke_bot.r215_complete_ordered_legal_actions/v1",
            "actions": [list(_action_tuple(action)) for action in actions],
        }
    )


@dataclass(frozen=True, slots=True)
class R215TimingConfig:
    """The one easy-to-change r215 timing surface.

    ``emergency_simulation_safety_ceiling`` is a stop guard only.  It must not
    be rendered as a requested-simulation target or completion rate.
    """

    total_game_wall_seconds: float = R215_DEFAULT_GAME_SECONDS
    reserve_wall_seconds: float = R215_DEFAULT_GAME_RESERVE_SECONDS
    expected_search_decisions: int = R215_DEFAULT_EXPECTED_SEARCH_DECISIONS
    minimum_remaining_decision_divisor: int = (
        R215_DEFAULT_MINIMUM_REMAINING_DECISION_DIVISOR
    )
    default_turn_pool_seconds: float = R215_DEFAULT_TURN_POOL_SECONDS
    per_operation_ceiling_seconds: float = R215_DEFAULT_OPERATION_SECONDS
    first_decision_search_ceiling_seconds: float = (
        R215_DEFAULT_FIRST_DECISION_SEARCH_SECONDS
    )
    # These two switches preserve the sealed r215/r218 controller behaviour by
    # default.  A separately versioned local controller can opt into residual
    # same-turn re-search without rewriting the earlier experiment's contract.
    later_decision_search_ceiling_seconds: float = (
        R215_DEFAULT_LATER_DECISION_SEARCH_SECONDS
    )
    allow_later_same_turn_search: bool = False
    allow_verified_finite_chance_cache: bool = False
    enforce_component_operation_ceiling: bool = True
    finite_chance_outcome_cap: int = 0
    minimum_valid_simulations: int = R215_MINIMUM_VALID_SIMULATIONS
    emergency_simulation_safety_ceiling: int = (
        R215_EMERGENCY_SIMULATION_SAFETY_CEILING
    )
    emergency_depth_safety_ceiling: int = R215_EMERGENCY_DEPTH_SAFETY_CEILING

    def __post_init__(self) -> None:
        for name in (
            "total_game_wall_seconds",
            "default_turn_pool_seconds",
            "per_operation_ceiling_seconds",
            "first_decision_search_ceiling_seconds",
            "later_decision_search_ceiling_seconds",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise R215Error(f"{name} must be positive")
        if not 0.0 <= float(self.reserve_wall_seconds) < float(
            self.total_game_wall_seconds
        ):
            raise R215Error("reserve_wall_seconds must be within the game clock")
        if int(self.expected_search_decisions) < 1:
            raise R215Error("expected_search_decisions must be positive")
        if int(self.minimum_remaining_decision_divisor) < 1:
            raise R215Error("minimum_remaining_decision_divisor must be positive")
        if int(self.minimum_valid_simulations) != 1:
            raise R215Error("r215 requires exactly one minimum valid simulation")
        if int(self.emergency_simulation_safety_ceiling) < 1:
            raise R215Error("emergency simulation ceiling must be positive")
        if int(self.emergency_depth_safety_ceiling) < 1:
            raise R215Error("emergency depth ceiling must be positive")
        if int(self.finite_chance_outcome_cap) < 0:
            raise R215Error("finite_chance_outcome_cap may not be negative")
        for name in (
            "allow_later_same_turn_search",
            "allow_verified_finite_chance_cache",
            "enforce_component_operation_ceiling",
        ):
            if type(getattr(self, name)) is not bool:
                raise R215Error(f"{name} must be a bool")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256({"schema": R215_SCHEMA, "timing": asdict(self)})

    def make_game_clock(self) -> Any:
        # Production gets the canonical source-backed clock when its runtime is
        # available.  Unit tests deliberately do not need Torch just to test
        # r215 cache/clock accounting.
        try:
            from .mcts import GameClock

            return GameClock(
                total_s=float(self.total_game_wall_seconds),
                reserve_s=float(self.reserve_wall_seconds),
                expected_search_decisions=int(self.expected_search_decisions),
            )
        except ModuleNotFoundError:
            return R215GameClock(
                total_s=float(self.total_game_wall_seconds),
                reserve_s=float(self.reserve_wall_seconds),
                expected_search_decisions=int(self.expected_search_decisions),
            )


@dataclass(frozen=True, slots=True)
class R215TurnIdentity:
    """The actual game turn; atomic steps intentionally do not change it."""

    seat: int
    actual_turn_id: int | str

    def __post_init__(self) -> None:
        if self.seat not in (0, 1):
            raise R215Error("seat must be 0 or 1")
        if not isinstance(self.actual_turn_id, (int, str)):
            raise R215Error("actual_turn_id must be an int or string")

    @property
    def key(self) -> tuple[int, int | str]:
        return (self.seat, self.actual_turn_id)


@dataclass(frozen=True, slots=True)
class R215Observation:
    """Public execution material needed to verify a real dispatch.

    Callers must obtain ``public_observation_fingerprint`` from the exact
    submitted runtime's policy-visible observation.  It is deliberately an
    explicit input: this controller must not derive an allegedly public key
    from a private simulator object, ``searchId``, seed, or opaque state token.
    """

    public_observation_fingerprint: str
    legal_actions: tuple[tuple[int, ...], ...]
    raw_observation: Mapping[str, Any] | None = None
    action_space_mode: str = "complete_ordered"
    matchup_adapter_route_receipt: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.public_observation_fingerprint, str) or not self.public_observation_fingerprint:
            raise R215Error("public_observation_fingerprint must be nonempty")
        normalized = tuple(_action_tuple(action) for action in self.legal_actions)
        if not normalized or len(set(normalized)) != len(normalized):
            raise R215Error("legal actions must be a nonempty unique ordered sequence")
        object.__setattr__(self, "legal_actions", normalized)
        if self.action_space_mode not in {"complete_ordered", "exact_factorized"}:
            raise R215Error("action_space_mode must be complete_ordered or exact_factorized")

    @property
    def legal_action_order_fingerprint(self) -> str:
        return legal_action_order_fingerprint(self.legal_actions)

    def contains(self, action: Sequence[int]) -> bool:
        return _action_tuple(action) in self.legal_actions


@dataclass(frozen=True, slots=True)
class R215CachedBranchStep:
    """One deterministic next real-game hop expected by a private tree."""

    expected_public_observation_fingerprint: str
    expected_legal_action_order_fingerprint: str
    selected_action: tuple[int, ...]
    deterministic: bool = True
    # A realized finite chance may reuse a conditioned child only when the
    # separately versioned caller has validated its exact engine receipt.  The
    # r215 default remains ordinary deterministic-only cache reuse.
    requires_verified_finite_chance: bool = False
    finite_chance_outcome_id: str | None = None

    def __post_init__(self) -> None:
        if not self.deterministic:
            raise R215Error("only deterministic steps may enter an r215 real cache")
        if not self.expected_public_observation_fingerprint:
            raise R215Error("cached step needs an observation fingerprint")
        if not self.expected_legal_action_order_fingerprint:
            raise R215Error("cached step needs a legal-order fingerprint")
        if self.requires_verified_finite_chance:
            if not isinstance(self.finite_chance_outcome_id, str) or not self.finite_chance_outcome_id:
                raise R215Error(
                    "finite-chance cached step needs a verified outcome identifier"
                )
        elif self.finite_chance_outcome_id is not None:
            raise R215Error(
                "finite_chance_outcome_id requires requires_verified_finite_chance"
            )
        object.__setattr__(self, "selected_action", _action_tuple(self.selected_action))

    def matches(
        self,
        observation: R215Observation,
        *,
        verified_finite_chance_outcome_id: str | None = None,
    ) -> bool:
        return (
            self.expected_public_observation_fingerprint
            == observation.public_observation_fingerprint
            and self.expected_legal_action_order_fingerprint
            == observation.legal_action_order_fingerprint
            and observation.contains(self.selected_action)
            and (
                not self.requires_verified_finite_chance
                or self.finite_chance_outcome_id
                == verified_finite_chance_outcome_id
            )
        )


@dataclass(frozen=True, slots=True)
class R215PlanRequest:
    """Budgeted private-planning request; it has no real-game dispatcher."""

    turn_identity: R215TurnIdentity
    observation: R215Observation
    turn_pool_remaining_seconds: float
    operation_allowance_seconds: float
    first_decision_search_allowance_seconds: float
    game_clock_remaining_seconds: float
    timing_identity_sha256: str
    emergency_simulation_safety_ceiling: int
    emergency_depth_safety_ceiling: int
    minimum_valid_simulations: int = R215_MINIMUM_VALID_SIMULATIONS
    chance_label: str = R215_CHANCE_LABEL
    allow_early_stop_when_root_action_stable: bool = True
    # ``first_decision_search_allowance_seconds`` is retained for the sealed
    # r215/r218 surface.  Later-version wrappers set the generic allowance for
    # every meaningful search segment while preserving that historical field.
    search_allowance_seconds: float | None = None
    search_segment_index: int = 1
    search_boundary_reason: str = "first_meaningful_choice"
    finite_chance_outcome_cap: int = 0
    allow_exact_finite_chance_enumeration: bool = False

    def __post_init__(self) -> None:
        for name in (
            "turn_pool_remaining_seconds",
            "operation_allowance_seconds",
            "first_decision_search_allowance_seconds",
            "game_clock_remaining_seconds",
        ):
            if float(getattr(self, name)) < 0.0:
                raise R215Error(f"{name} may not be negative")
        if self.search_allowance_seconds is not None and float(
            self.search_allowance_seconds
        ) < 0.0:
            raise R215Error("search_allowance_seconds may not be negative")
        if int(self.search_segment_index) < 1:
            raise R215Error("search_segment_index must be positive")
        if not isinstance(self.search_boundary_reason, str) or not self.search_boundary_reason:
            raise R215Error("search_boundary_reason must be nonempty")
        if int(self.finite_chance_outcome_cap) < 0:
            raise R215Error("finite_chance_outcome_cap may not be negative")

    @property
    def effective_search_allowance_seconds(self) -> float:
        if self.search_allowance_seconds is None:
            return float(self.first_decision_search_allowance_seconds)
        return float(self.search_allowance_seconds)


@dataclass(frozen=True, slots=True)
class R215PlanResult:
    """A private MCTS result plus only an attestable deterministic remainder."""

    selected_action: tuple[int, ...]
    sims_run: int
    continuation: tuple[R215CachedBranchStep, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    root_action_stable: bool = False
    root_stability_receipt: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_action", _action_tuple(self.selected_action))
        if int(self.sims_run) < 0:
            raise R215Error("sims_run may not be negative")
        if self.root_action_stable and not self.root_stability_receipt:
            raise R215Error(
                "an early stable-root result requires a root_stability_receipt"
            )
        object.__setattr__(self, "continuation", tuple(self.continuation))


@runtime_checkable
class R215PlannerProtocol(Protocol):
    """Injected for hermetic tests and implemented by the BeliefMCTS adapter.

    A planner that temporarily mutates a PolicyAgent while searching may also
    expose ``accept_plan(plan)`` and ``reject_plan(reason)``.  They are
    deliberately optional so pure planners stay dependency-light; when
    present, the controller resolves exactly one of them before it dispatches
    a selected MCTS action or invokes the direct fallback.
    """

    def plan_turn(self, request: R215PlanRequest) -> R215PlanResult: ...


@dataclass(frozen=True, slots=True)
class R215ActionDecision:
    selected_action: tuple[int, ...]
    source: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class R215TranspositionDecision:
    accepted: bool
    reason: str


def _validated_root_stability_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    selected_action: Sequence[int],
) -> dict[str, Any] | None:
    """Accept only a receipt that proves convergence used a backed-up root.

    This intentionally validates an *attestation emitted by the private
    search*, rather than attempting to infer a convergence claim from elapsed
    time or an incomplete tree.  Older result objects can still select a legal
    action, but they never acquire the stable-root early-stop label.
    """

    if not isinstance(receipt, Mapping):
        return None
    try:
        receipt_action = _action_tuple(receipt.get("selected_action", ()))
    except R215Error:
        return None
    if receipt_action != _action_tuple(selected_action):
        return None
    required = (
        "stable_root_convergence",
        "selected_action_legal",
        "selected_action_fully_backed_up",
    )
    if not all(receipt.get(key) is True for key in required):
        return None
    evidence = receipt.get("root_value_or_visit_stability_evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        return None
    try:
        completed_backups = int(receipt.get("completed_backups", 0))
        selected_visits = int(receipt.get("selected_action_visit_count", 0))
    except (TypeError, ValueError):
        return None
    if completed_backups < 1 or selected_visits < 1:
        return None
    return dict(receipt)


def _principal_continuation_from_diagnostics(
    diagnostics: Mapping[str, Any],
) -> tuple[R215CachedBranchStep, ...]:
    """Translate only fully backed-up, fingerprint-bound cache candidates.

    The search owns how a candidate is derived.  This controller only accepts
    a narrow serialization that it can independently re-check against a fresh
    real observation before dispatch.  Malformed or unavailable candidates are
    simply absent; a later residual search/direct fallback remains safe.
    """

    raw_rows = diagnostics.get("principal_continuation_by_fingerprint", ())
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return ()
    # The engine may phrase this invariant as an explicit boolean in newer
    # diagnostics, or as the fixed derivation label emitted by the first r219
    # principal-continuation implementation.  Do not accept an ordinary
    # ``deterministic`` flag by itself: that could still conceal a sampled
    # chance hop elsewhere on the principal path.
    no_sampled_chance_derivation = (
        "single_outcome_no_sampled_chance_principal_path_root_sampled_approximate"
    )
    steps: list[R215CachedBranchStep] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            break
        if raw.get("deterministic") is not True:
            break
        if raw.get("selected_action_fully_backed_up") is not True:
            break
        if not (
            raw.get("no_sampled_or_opaque_chance") is True
            or raw.get("derivation") == no_sampled_chance_derivation
        ):
            break
        expected = raw.get("expected_public_observation_fingerprint")
        raw_actions = raw.get("legal_actions")
        selected = raw.get("selected_action")
        if not isinstance(expected, str) or not expected:
            break
        if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
            break
        try:
            legal_actions = tuple(_action_tuple(action) for action in raw_actions)
            selected_action = _action_tuple(selected)
        except R215Error:
            break
        if not legal_actions or selected_action not in legal_actions:
            break
        finite_chance = raw.get("requires_verified_finite_chance") is True
        outcome_id = raw.get("finite_chance_outcome_id")
        if finite_chance and (
            not isinstance(outcome_id, str) or not outcome_id
        ):
            break
        try:
            steps.append(
                R215CachedBranchStep(
                    expected_public_observation_fingerprint=expected,
                    expected_legal_action_order_fingerprint=(
                        legal_action_order_fingerprint(legal_actions)
                    ),
                    selected_action=selected_action,
                    deterministic=True,
                    requires_verified_finite_chance=finite_chance,
                    finite_chance_outcome_id=(
                        outcome_id if finite_chance else None
                    ),
                )
            )
        except R215Error:
            break
    return tuple(steps)


@dataclass(slots=True)
class _ActiveTurn:
    identity: R215TurnIdentity
    effective_pool_seconds: float
    allocator_pool_seconds: float
    game_remaining_at_start_seconds: float
    work_used_seconds: float = 0.0
    continuation: list[R215CachedBranchStep] = field(default_factory=list)
    cache_hops: int = 0
    cache_only_later_step_count: int = 0
    cache_invalidations: int = 0
    cache_invalidation_reasons: dict[str, int] = field(default_factory=dict)
    searches_this_turn: int = 0
    first_segment_searches: int = 0
    later_researches: int = 0
    rebuild_searches: int = 0
    rebuild_search_reasons: dict[str, int] = field(default_factory=dict)
    finite_chance_enumeration_count: int = 0
    finite_chance_rebuild_count: int = 0
    direct_fallbacks: int = 0
    direct_fallback_reasons: dict[str, int] = field(default_factory=dict)
    search_segment_reasons: dict[str, int] = field(default_factory=dict)
    atomic_steps: int = 0


class BeliefMCTSPlannerAdapter:
    """Minimal adapter from the existing :class:`BeliefMCTS` to r215.

    The existing search object continues to own root-sampled particles,
    private libcg worlds, priors and leaf values.  A caller supplies the exact
    public belief/history material required by that object.  Its 1,000,000
    argument is an emergency stop guard; search remains deadline-driven.

    The legacy BeliefMCTS result format does not expose a certified real-turn
    continuation, so this adapter returns an empty continuation.  The
    controller then safely uses direct policy on later same-turn steps unless
    a newer planner supplies verified deterministic cached hops.
    """

    def __init__(
        self,
        planner: BeliefMCTS,
        *,
        belief_history_for: Callable[[R215PlanRequest], Any],
        root_history_for: Callable[[R215PlanRequest], tuple[Sequence[Any], Sequence[Any]]],
        temperature: float = 1.0,
    ) -> None:
        self.planner = planner
        self._belief_history_for = belief_history_for
        self._root_history_for = root_history_for
        self.temperature = float(temperature)

    def plan_turn(self, request: R215PlanRequest) -> R215PlanResult:
        if request.observation.raw_observation is None:
            raise R215Error("BeliefMCTS requires the raw public observation")
        # r215's minimum is one; an old 50/128 trust floor is not admissible.
        if int(getattr(self.planner, "min_trusted_sims", 1)) > 1:
            raise R215Error("BeliefMCTS must be constructed with min_trusted_sims=1")
        # The normal planner must be deadline/stability driven.  Its old
        # 256-depth default is not a practical r215 target; retain only the
        # deliberately unreachable emergency guard supplied in this request.
        if hasattr(self.planner, "max_depth"):
            self.planner.max_depth = int(request.emergency_depth_safety_ceiling)
        boards, previous_actions = self._root_history_for(request)
        try:
            result = self.planner.search(
                dict(request.observation.raw_observation),
                belief_history=self._belief_history_for(request),
                root_history_boards=boards,
                root_history_previous_actions=previous_actions,
                # The ceiling is intentionally a safety stop, not a goal.
                max_sims=int(request.emergency_simulation_safety_ceiling),
                # The sealed r215/r218 request leaves this equal to the first
                # decision allowance.  Later-version shared-turn wrappers use
                # the same deadline-driven field for a residual meaningful
                # search segment; no simulator action leaves this call.
                move_time_s=float(request.effective_search_allowance_seconds),
                temperature=self.temperature,
            )
        except Exception as exc:
            # Do not import the Torch-dependent implementation merely to name
            # its timeout exception.  The adapter translates only that
            # documented failure; every other exception remains visible to the
            # outer fail-closed controller as a planner error.
            if type(exc).__name__ != "TrustedSearchBudgetExhausted":
                raise
            return R215PlanResult(
                selected_action=request.observation.legal_actions[0],
                sims_run=0,
                diagnostics={"search_stop_reason": "no_valid_simulation", "error": str(exc)},
            )
        target_diagnostics = getattr(getattr(result, "target", None), "diagnostics", {})
        if not isinstance(target_diagnostics, Mapping):
            target_diagnostics = {}
        stability_receipt = _validated_root_stability_receipt(
            target_diagnostics.get("root_stability_receipt"),
            selected_action=tuple(result.select),
        )
        root_stable = bool(
            # BeliefMCTS emits ``root_action_stable``.  Accept the historical
            # spelling only as a compatibility alias, never as proof by itself.
            target_diagnostics.get(
                "root_action_stable",
                target_diagnostics.get("root_selected_action_stable", False),
            )
            and stability_receipt is not None
        )
        return R215PlanResult(
            selected_action=tuple(result.select),
            sims_run=int(result.sims_run),
            continuation=_principal_continuation_from_diagnostics(
                target_diagnostics
            ),
            diagnostics=dict(target_diagnostics),
            root_action_stable=root_stable,
            root_stability_receipt=stability_receipt,
        )


class R215FullTurnBeliefMCTS:
    """Full-actual-turn cache/controller around a private BeliefMCTS planner."""

    def __init__(
        self,
        planner: R215PlannerProtocol,
        *,
        direct_policy: Callable[[R215Observation], Sequence[int]],
        timing: R215TimingConfig | None = None,
        game_clock: Any | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        # Structural Protocol checks are intentionally only a friendly error;
        # test doubles need only implement plan_turn.
        if not isinstance(planner, R215PlannerProtocol) and not callable(
            getattr(planner, "plan_turn", None)
        ):
            raise TypeError("planner must implement plan_turn(request)")
        self.planner = planner
        self.direct_policy = direct_policy
        self.timing = timing or R215TimingConfig()
        self.game_clock = game_clock or self.timing.make_game_clock()
        self._monotonic = monotonic
        self._active: _ActiveTurn | None = None
        self._transposition_attempted = 0
        self._transposition_rejected: dict[str, int] = {}

    @staticmethod
    def _clock_remaining(clock: Any) -> float:
        return max(0.0, float(clock.remaining_s))

    @staticmethod
    def _clock_decisions_used(clock: Any) -> int:
        return max(0, int(getattr(clock, "decisions_used", 0)))

    def _allocator_pool(self, game_remaining: float) -> float:
        # Owner clarification (r215 live semantics): 20 s is the normal
        # healthy-game default.  The fair-share divisor is a near-clock guard,
        # not a reason to cut a 600 s game's first turn to 570 / 64 seconds.
        # It begins shrinking only when the usable game time cannot support
        # eight more default-size turn pools.
        available = max(0.0, game_remaining - self.timing.reserve_wall_seconds)
        divisor = int(self.timing.minimum_remaining_decision_divisor)
        return min(self.timing.default_turn_pool_seconds, available / divisor)

    def _start_turn(self, identity: R215TurnIdentity) -> _ActiveTurn:
        game_remaining = self._clock_remaining(self.game_clock)
        allocator = self._allocator_pool(game_remaining)
        effective = min(self.timing.default_turn_pool_seconds, allocator, game_remaining)
        active = _ActiveTurn(
            identity=identity,
            effective_pool_seconds=max(0.0, effective),
            allocator_pool_seconds=max(0.0, allocator),
            game_remaining_at_start_seconds=game_remaining,
        )
        self._active = active
        return active

    def finish_actual_turn(self) -> Mapping[str, Any] | None:
        """Charge all work once, discard the turn tree, and return its summary.

        The returned summary gives a runner one receipt-backed point at actual
        turn close for r219 canary aggregation.  Existing callers may ignore
        it just as they ignored the historical ``None`` return value.
        """

        active = self._active
        if active is None:
            return None
        consume = getattr(self.game_clock, "consume", None)
        if not callable(consume):
            raise R215Error("game_clock must expose consume(seconds)")
        consume(max(0.0, active.work_used_seconds))
        summary: dict[str, Any] = {
            "schema": R215_SCHEMA,
            "actual_turn_closed": True,
            "planner_turn_id": f"{active.identity.seat}:{active.identity.actual_turn_id}",
            "seat": active.identity.seat,
            "actual_turn_id": active.identity.actual_turn_id,
            "actual_atomic_steps": active.atomic_steps,
            "effective_actual_turn_planner_pool_seconds": active.effective_pool_seconds,
            "planner_seconds_used_this_turn": active.work_used_seconds,
            "planner_seconds_residual_this_turn": max(
                0.0, active.effective_pool_seconds - active.work_used_seconds
            ),
            "searches_this_turn": active.searches_this_turn,
            "search_segments_this_turn": active.searches_this_turn,
            "first_segment_search": bool(active.first_segment_searches),
            "later_research_count_this_turn": active.later_researches,
            "rebuild_searches_this_turn": active.rebuild_searches,
            "rebuild_count_and_reasons": dict(active.rebuild_search_reasons),
            "search_segment_reasons": dict(active.search_segment_reasons),
            "only_first_segment_search": active.searches_this_turn == 1,
            "cache_hops_this_turn": active.cache_hops,
            "cache_only_later_steps": active.cache_only_later_step_count,
            "finite_chance_enumeration_count_this_turn": active.finite_chance_enumeration_count,
            "finite_chance_rebuild_count_this_turn": active.finite_chance_rebuild_count,
            "direct_fallbacks_this_turn": active.direct_fallbacks,
            "direct_fallback_reasons": dict(active.direct_fallback_reasons),
        }
        self._active = None
        return summary

    def charge_actual_turn_external_work(
        self,
        identity: R215TurnIdentity,
        elapsed_seconds: float,
    ) -> Mapping[str, Any]:
        """Charge bridge-side validation/history/dispatch to the active pool.

        ``act`` owns selection and its optional dispatcher, but a thin runtime
        bridge may need to build a public observation or commit a verified
        cache action around that call.  Those operations are still actual-turn
        work.  This method never opens, resets, or aborts a turn; it only
        records elapsed monotonic work against the already active matching
        turn.  A post-dispatch overrun is telemetry, not a reason to undo an
        action that has already reached the real game.
        """

        if not isinstance(identity, R215TurnIdentity):
            raise R215Error("external work needs an R215TurnIdentity")
        if float(elapsed_seconds) < 0.0:
            raise R215Error("external actual-turn work may not be negative")
        active = self._active
        if active is None or active.identity != identity:
            raise R215Error("external actual-turn work has no matching active turn")
        charged = max(0.0, float(elapsed_seconds))
        active.work_used_seconds += charged
        return {
            "external_actual_turn_work_seconds": charged,
            "planner_seconds_used_this_turn": active.work_used_seconds,
            "planner_seconds_residual_this_turn": max(
                0.0, active.effective_pool_seconds - active.work_used_seconds
            ),
            "remaining_turn_planner_pool_seconds_after": max(
                0.0, active.effective_pool_seconds - active.work_used_seconds
            ),
            "game_clock_remaining_seconds_after": self._remaining_game_for_turn(
                active
            ),
            "post_dispatch_turn_budget_breach": (
                active.work_used_seconds > active.effective_pool_seconds
            ),
        }

    def transposition_decision(self, *_: Any, **__: Any) -> R215TranspositionDecision:
        """Record the fail-closed r215 answer for an A→B / B→A merge request."""

        reason = "native_complete_semantic_state_identity_unavailable"
        self._transposition_attempted += 1
        self._transposition_rejected[reason] = self._transposition_rejected.get(reason, 0) + 1
        return R215TranspositionDecision(accepted=False, reason=reason)

    # Alias useful to callers/tests that phrase the operation as an action.
    attempt_transposition_merge = transposition_decision

    def _remaining_game_for_turn(self, active: _ActiveTurn) -> float:
        return max(
            0.0,
            min(
                self._clock_remaining(self.game_clock),
                active.game_remaining_at_start_seconds - active.work_used_seconds,
            ),
        )

    def _operation_allowance(self, active: _ActiveTurn) -> float:
        return max(
            0.0,
            min(
                self.timing.per_operation_ceiling_seconds,
                active.effective_pool_seconds - active.work_used_seconds,
                self._remaining_game_for_turn(active),
            ),
        )

    def _first_decision_search_allowance(self, active: _ActiveTurn) -> float:
        """The historical first-search allowance for sealed r215/r218 runs."""

        return max(
            0.0,
            min(
                self.timing.first_decision_search_ceiling_seconds,
                active.effective_pool_seconds - active.work_used_seconds,
                self._remaining_game_for_turn(active),
            ),
        )

    def _later_decision_search_allowance(self, active: _ActiveTurn) -> float:
        """Residual allowance for an enabled later meaningful search segment."""

        return max(
            0.0,
            min(
                self.timing.later_decision_search_ceiling_seconds,
                active.effective_pool_seconds - active.work_used_seconds,
                self._remaining_game_for_turn(active),
            ),
        )

    @staticmethod
    def _invalidate_cached_branch(active: _ActiveTurn, reason: str) -> None:
        active.continuation.clear()
        active.cache_invalidations += 1
        active.cache_invalidation_reasons[reason] = (
            active.cache_invalidation_reasons.get(reason, 0) + 1
        )

    def _resolve_planner_result(
        self,
        *,
        accepted: bool,
        plan: R215PlanResult | None,
        reason: str | None = None,
    ) -> None:
        """Notify an optional transactional planner exactly once per call.

        A PolicyAgent-backed planner can mutate public-history state while it
        searches.  It therefore keeps its snapshot until this controller has
        either retained the result's action authority or rejected it before an
        exact direct fallback.  Plain planners do not implement these hooks.
        """

        hook_name = "accept_plan" if accepted else "reject_plan"
        hook = getattr(self.planner, hook_name, None)
        if not callable(hook):
            return
        if accepted:
            if plan is None:
                raise R215Error("cannot accept a missing planner result")
            hook(plan)
        else:
            hook(str(reason or "untrusted_or_rejected_plan"))

    def _direct_action(self, observation: R215Observation) -> tuple[int, ...]:
        action = _action_tuple(self.direct_policy(observation))
        if not observation.contains(action):
            raise R215Error("frozen direct policy returned an action outside exact legal order")
        return action

    def _base_receipt(
        self,
        active: _ActiveTurn,
        observation: R215Observation,
        *,
        remaining_before: float,
        operation_allowance: float,
        first_decision_search_allowance: float,
        later_decision_search_allowance: float,
    ) -> dict[str, Any]:
        return {
            "schema": R215_SCHEMA,
            "chance_label": R215_CHANCE_LABEL,
            "planner_turn_id": f"{active.identity.seat}:{active.identity.actual_turn_id}",
            "seat": active.identity.seat,
            "actual_turn_id": active.identity.actual_turn_id,
            "actual_atomic_step_index": active.atomic_steps,
            "configured_default_turn_planner_pool_seconds": self.timing.default_turn_pool_seconds,
            "configured_per_operation_ceiling_seconds": self.timing.per_operation_ceiling_seconds,
            "configured_first_decision_search_ceiling_seconds": self.timing.first_decision_search_ceiling_seconds,
            "configured_later_decision_search_ceiling_seconds": self.timing.later_decision_search_ceiling_seconds,
            "allow_later_same_turn_search": self.timing.allow_later_same_turn_search,
            "allow_verified_finite_chance_cache": self.timing.allow_verified_finite_chance_cache,
            "component_operation_ceiling_enforced": self.timing.enforce_component_operation_ceiling,
            "configured_outer_game_clock_identity": self.timing.identity_sha256,
            "timing_identity_sha256": self.timing.identity_sha256,
            "game_clock_remaining_seconds_before": self._remaining_game_for_turn(active),
            "game_clock_allocator_turn_pool_seconds": active.allocator_pool_seconds,
            "effective_actual_turn_planner_pool_seconds": active.effective_pool_seconds,
            "planner_wall_seconds_used_before": active.work_used_seconds,
            "remaining_turn_planner_pool_seconds_before": remaining_before,
            "effective_operation_allowance_seconds": operation_allowance,
            "effective_first_decision_search_allowance_seconds": first_decision_search_allowance,
            "effective_later_decision_search_allowance_seconds": later_decision_search_allowance,
            "first_decision_mcts_search_allowed": active.atomic_steps == 0,
            "later_meaningful_mcts_search_allowed": bool(
                active.atomic_steps > 0 and self.timing.allow_later_same_turn_search
            ),
            "complete_ordered_action_count": len(observation.legal_actions),
            "action_space_mode": observation.action_space_mode,
            "matchup_adapter_enabled_and_route_receipt": observation.matchup_adapter_route_receipt,
            "emergency_simulation_safety_ceiling": self.timing.emergency_simulation_safety_ceiling,
            "emergency_depth_safety_ceiling": self.timing.emergency_depth_safety_ceiling,
            "native_complete_semantic_state_identity_available": False,
            "native_actions_commute_certificate_available": False,
            "transposition_merges_attempted": self._transposition_attempted,
            "transposition_merges_accepted": 0,
            "transposition_merges_rejected": sum(self._transposition_rejected.values()),
            "transposition_merge_rejection_reasons": dict(self._transposition_rejected),
            "transposition_model_evaluation_savings": 0,
        }

    @staticmethod
    def _cache_matches(
        step: R215CachedBranchStep,
        observation: R215Observation,
        *,
        verified_finite_chance_outcome_id: str | None = None,
    ) -> bool:
        return step.matches(
            observation,
            verified_finite_chance_outcome_id=verified_finite_chance_outcome_id,
        )

    def act(
        self,
        identity: R215TurnIdentity,
        observation: R215Observation,
        *,
        dispatch: Callable[[list[int]], Any] | None = None,
        boundary_reason: str | None = None,
        verified_finite_chance_outcome_id: str | None = None,
    ) -> R215ActionDecision:
        """Choose and optionally dispatch exactly one verified real action.

        By default this preserves r215/r218: only the first atomic decision
        may call MCTS and later decisions are a verified cache hop or exact
        direct fallback.  Separately versioned controllers opt into residual
        searches by setting ``allow_later_same_turn_search`` in their typed
        timing configuration.  They may then re-search only at a meaningful
        branch endpoint/divergence, never merely because another atomic step
        occurred.

        ``verified_finite_chance_outcome_id`` is deliberately just a
        controller input, not a claimed engine capability.  A newer wrapper
        must validate an engine-issued exact finite-chance receipt before
        providing it; otherwise a chance is treated as an opaque boundary.
        """

        if boundary_reason not in {
            None,
            "chance",
            "finite_chance",
            "information",
            "divergence",
            "obvious",
        }:
            raise R215Error(
                "boundary_reason must be chance, finite_chance, information, "
                "divergence, obvious, or None"
            )
        if verified_finite_chance_outcome_id is not None and (
            not isinstance(verified_finite_chance_outcome_id, str)
            or not verified_finite_chance_outcome_id
        ):
            raise R215Error("verified_finite_chance_outcome_id must be nonempty")
        if self._active is not None and self._active.identity != identity:
            self.finish_actual_turn()
        active = self._active or self._start_turn(identity)
        remaining_before = max(0.0, active.effective_pool_seconds - active.work_used_seconds)
        allowance = self._operation_allowance(active)
        first_atomic_step = active.atomic_steps == 0
        first_search_segment = active.searches_this_turn == 0
        first_segment_is_eligible = first_atomic_step or (
            self.timing.allow_later_same_turn_search and first_search_segment
        )
        first_search_allowance = (
            self._first_decision_search_allowance(active)
            if first_segment_is_eligible
            else 0.0
        )
        later_search_allowance = (
            self._later_decision_search_allowance(active)
            if self.timing.allow_later_same_turn_search and not first_search_segment
            else 0.0
        )
        receipt = self._base_receipt(
            active,
            observation,
            remaining_before=remaining_before,
            operation_allowance=allowance,
            first_decision_search_allowance=first_search_allowance,
            later_decision_search_allowance=later_search_allowance,
        )
        started = self._monotonic()
        source = "direct_policy_fallback"
        fallback_reason: str | None = None
        plan: R215PlanResult | None = None
        plan_called = False
        planner_resolution: str | None = None
        search_trigger: str | None = None
        search_allowance = 0.0
        finite_chance_cache_reused = False
        selected: tuple[int, ...]

        def resolve_planner(*, accepted: bool, reason: str | None = None) -> None:
            """Resolve a mutating planner before any direct-policy fallback."""

            nonlocal planner_resolution
            if not plan_called or planner_resolution is not None:
                return
            self._resolve_planner_result(
                accepted=accepted,
                plan=plan,
                reason=reason,
            )
            planner_resolution = "accepted" if accepted else "rejected"

        if len(observation.legal_actions) == 1:
            # A forced step is not a meaningful search boundary.  It still
            # receives a fresh legal-action verification immediately before
            # the one real dispatch below.
            selected = observation.legal_actions[0]
            source = "forced_legal_action"
        elif allowance <= 0.0 or remaining_before <= 0.0:
            fallback_reason = "turn_or_game_clock_exhausted"
            selected = self._direct_action(observation)
        elif first_segment_is_eligible:
            search_trigger = (
                "first_meaningful_choice"
                if first_atomic_step
                else "first_meaningful_choice_after_forced_step"
            )
            search_allowance = first_search_allowance
        elif boundary_reason == "obvious":
            selected = self._direct_action(observation)
            source = "obvious_direct_policy"
        else:
            cached = active.continuation[0] if active.continuation else None
            if boundary_reason == "finite_chance":
                finite_cache_ok = bool(
                    self.timing.allow_verified_finite_chance_cache
                    and verified_finite_chance_outcome_id is not None
                )
                if (
                    cached is not None
                    and finite_cache_ok
                    and self._cache_matches(
                        cached,
                        observation,
                        verified_finite_chance_outcome_id=verified_finite_chance_outcome_id,
                    )
                ):
                    selected = cached.selected_action
                    active.continuation.pop(0)
                    active.cache_hops += 1
                    source = "verified_finite_chance_cached_branch"
                    finite_chance_cache_reused = True
                else:
                    invalidation_reason = (
                        "finite_chance_cached_child_mismatch"
                        if finite_cache_ok
                        else "unverified_finite_chance_boundary"
                    )
                    self._invalidate_cached_branch(active, invalidation_reason)
                    if self.timing.allow_later_same_turn_search:
                        search_trigger = invalidation_reason
                        search_allowance = later_search_allowance
                        active.finite_chance_rebuild_count += 1
                    else:
                        fallback_reason = "same_turn_boundary_requires_direct_fallback"
            elif boundary_reason in {"chance", "information", "divergence"}:
                self._invalidate_cached_branch(active, boundary_reason)
                if self.timing.allow_later_same_turn_search:
                    search_trigger = f"{boundary_reason}_boundary"
                    search_allowance = later_search_allowance
                else:
                    # Preserve the r218 sealed behaviour: a later opaque
                    # boundary never opens another private search allowance.
                    fallback_reason = "same_turn_boundary_requires_direct_fallback"
            elif cached is not None and self._cache_matches(cached, observation):
                selected = cached.selected_action
                active.continuation.pop(0)
                active.cache_hops += 1
                source = "deterministic_cached_branch"
            elif cached is not None:
                self._invalidate_cached_branch(
                    active, "cached_branch_fingerprint_mismatch"
                )
                if self.timing.allow_later_same_turn_search:
                    search_trigger = "cached_branch_fingerprint_mismatch"
                    search_allowance = later_search_allowance
                else:
                    fallback_reason = "cached_branch_fingerprint_mismatch_direct_fallback"
            else:
                if self.timing.allow_later_same_turn_search:
                    search_trigger = "validated_cached_plan_endpoint"
                    search_allowance = later_search_allowance
                else:
                    fallback_reason = "missing_or_untrusted_cached_branch"

        if search_trigger is not None:
            if search_allowance <= 0.0:
                fallback_reason = (
                    "first_decision_search_allowance_exhausted"
                    if first_search_segment
                    else "residual_turn_pool_exhausted_before_later_search"
                )
            else:
                request = R215PlanRequest(
                    turn_identity=identity,
                    observation=observation,
                    turn_pool_remaining_seconds=remaining_before,
                    operation_allowance_seconds=allowance,
                    first_decision_search_allowance_seconds=(
                        first_search_allowance if first_search_segment else 0.0
                    ),
                    search_allowance_seconds=search_allowance,
                    search_segment_index=active.searches_this_turn + 1,
                    search_boundary_reason=search_trigger,
                    finite_chance_outcome_cap=self.timing.finite_chance_outcome_cap,
                    allow_exact_finite_chance_enumeration=bool(
                        self.timing.finite_chance_outcome_cap > 0
                    ),
                    game_clock_remaining_seconds=self._remaining_game_for_turn(active),
                    timing_identity_sha256=self.timing.identity_sha256,
                    emergency_simulation_safety_ceiling=self.timing.emergency_simulation_safety_ceiling,
                    emergency_depth_safety_ceiling=self.timing.emergency_depth_safety_ceiling,
                )
                plan_called = True
                active.search_segment_reasons[search_trigger] = (
                    active.search_segment_reasons.get(search_trigger, 0) + 1
                )
                active.searches_this_turn += 1
                if first_search_segment:
                    active.first_segment_searches += 1
                else:
                    active.later_researches += 1
                    active.rebuild_searches += 1
                    active.rebuild_search_reasons[search_trigger] = (
                        active.rebuild_search_reasons.get(search_trigger, 0) + 1
                    )
                try:
                    plan = self.planner.plan_turn(request)
                except Exception as exc:  # noqa: BLE001 - fail closed at isolated planner boundary
                    fallback_reason = f"planner_error:{type(exc).__name__}"
                if plan is not None:
                    if int(plan.sims_run) < self.timing.minimum_valid_simulations:
                        fallback_reason = "no_valid_simulation"
                    elif int(plan.sims_run) > self.timing.emergency_simulation_safety_ceiling:
                        fallback_reason = "emergency_simulation_safety_ceiling_hit"
                    elif not observation.contains(plan.selected_action):
                        fallback_reason = "planner_selected_illegal_action"
                    else:
                        selected = plan.selected_action
                        source = "belief_mcts"
                        active.continuation = list(plan.continuation)

        if source == "direct_policy_fallback":
            # Every insufficient/invalid/untrusted private result falls back
            # to the exact legal frozen direct action.  There is intentionally
            # no turn or game abort branch here.
            resolve_planner(accepted=False, reason=fallback_reason)
            selected = self._direct_action(observation)

        # Take the authority checkpoint before a valid MCTS action can reach
        # the real game.  Any private segment that is already late loses
        # authority here, while its measured work remains charged to the one
        # actual-turn pool.
        pre_dispatch_elapsed = max(0.0, self._monotonic() - started)
        active.work_used_seconds += pre_dispatch_elapsed
        diagnostics = dict(plan.diagnostics) if plan is not None else {}
        try:
            finite_chance_enumerated_this_segment = max(
                0, int(diagnostics.get("finite_chance_outcomes_enumerated", 0))
            )
        except (TypeError, ValueError):
            finite_chance_enumerated_this_segment = 0
        active.finite_chance_enumeration_count += finite_chance_enumerated_this_segment
        try:
            reported_component_seconds = float(
                diagnostics.get("max_model_or_simulator_operation_wall_seconds", 0.0)
            )
        except (TypeError, ValueError):
            reported_component_seconds = float("inf")
        component_breach = bool(
            diagnostics.get("component_operation_budget_breach", False)
            or (
                self.timing.enforce_component_operation_ceiling
                and reported_component_seconds > self.timing.per_operation_ceiling_seconds
            )
            or (
                self.timing.enforce_component_operation_ceiling
                and not plan_called
                and pre_dispatch_elapsed > allowance
            )
        )
        first_search_breach = bool(
            plan_called
            and first_search_segment
            and pre_dispatch_elapsed > first_search_allowance
        )
        later_search_breach = bool(
            plan_called
            and not first_search_segment
            and pre_dispatch_elapsed > search_allowance
        )
        pre_dispatch_turn_breach = (
            active.work_used_seconds > active.effective_pool_seconds
        )
        if source != "direct_policy_fallback" and (
            component_breach
            or first_search_breach
            or later_search_breach
            or pre_dispatch_turn_breach
        ):
            # No action has been dispatched yet, so a late private result can
            # safely lose authority in favour of the exact direct policy.
            breach_reason = (
                "component_operation_budget_breach"
                if component_breach
                else (
                    "first_decision_search_budget_breach"
                    if first_search_breach
                    else (
                        "later_search_segment_budget_breach"
                        if later_search_breach
                        else "pre_dispatch_turn_budget_breach"
                    )
                )
            )
            resolve_planner(accepted=False, reason=breach_reason)
            selected = self._direct_action(observation)
            source = "direct_policy_fallback"
            fallback_reason = breach_reason
            active.continuation.clear()

        # This check deliberately occurs after all planner/cache work and
        # immediately before the one real dispatch.
        if not observation.contains(selected):
            raise R215Error("selected action failed fresh exact legal verification")
        if source == "belief_mcts":
            # The planner may now discard its transactional snapshot: this is
            # the one point at which its selected action retained authority.
            resolve_planner(accepted=True)
        # Final fresh legality validation, transactional acceptance, and the
        # optional real dispatcher are all real actual-turn work.  A late
        # fallback above therefore cannot hide its direct-policy cost, and a
        # slow dispatcher cannot make the receipt understate pool use.  If
        # dispatch itself raises, still charge its elapsed time before letting
        # the caller observe that error.
        try:
            if dispatch is not None:
                dispatch(list(selected))
        finally:
            total_elapsed = max(pre_dispatch_elapsed, self._monotonic() - started)
            finalization_elapsed = max(0.0, total_elapsed - pre_dispatch_elapsed)
            active.work_used_seconds += finalization_elapsed
        post_dispatch_turn_breach = (
            active.work_used_seconds > active.effective_pool_seconds
        )
        if source == "direct_policy_fallback":
            active.direct_fallbacks += 1
            fallback_key = fallback_reason or "unspecified_direct_fallback"
            active.direct_fallback_reasons[fallback_key] = (
                active.direct_fallback_reasons.get(fallback_key, 0) + 1
            )
        if active.atomic_steps > 0 and source in {
            "deterministic_cached_branch",
            "verified_finite_chance_cached_branch",
        }:
            active.cache_only_later_step_count += 1
        active.atomic_steps += 1

        receipt.update(
            {
                "planner_wall_seconds_used_after": active.work_used_seconds,
                "remaining_turn_planner_pool_seconds_after": max(0.0, active.effective_pool_seconds - active.work_used_seconds),
                "game_clock_remaining_seconds_after": self._remaining_game_for_turn(active),
                "turn_planner_wall_seconds": total_elapsed,
                "pre_dispatch_controller_wall_seconds": pre_dispatch_elapsed,
                "final_validation_and_dispatch_wall_seconds": finalization_elapsed,
                "max_model_or_simulator_operation_wall_seconds": (
                    reported_component_seconds
                    if first_atomic_step
                    else pre_dispatch_elapsed
                ),
                "component_operation_budget_breach": component_breach,
                "first_decision_search_budget_breach": first_search_breach,
                "later_search_segment_budget_breach": later_search_breach,
                "search_segment_budget_breach": bool(
                    first_search_breach or later_search_breach
                ),
                "turn_budget_breach": post_dispatch_turn_breach,
                "pre_dispatch_turn_budget_breach": pre_dispatch_turn_breach,
                "post_dispatch_turn_budget_breach": post_dispatch_turn_breach,
                "first_decision_mcts_search_executed": bool(
                    plan_called and first_atomic_step
                ),
                "fresh_mcts_search_executed": plan_called,
                "search_segment_index": (
                    active.searches_this_turn if plan_called else None
                ),
                "search_segment_boundary_reason": search_trigger,
                "effective_search_segment_allowance_seconds": search_allowance,
                "sims_run": int(plan.sims_run) if plan is not None else 0,
                "emergency_simulation_safety_ceiling_hit": fallback_reason == "emergency_simulation_safety_ceiling_hit",
                "leaf_evaluations": int(diagnostics.get("leaf_evaluations", 0)),
                "unique_nodes": int(diagnostics.get("unique_nodes", 0)),
                "unique_expanded_nodes": int(diagnostics.get("unique_expanded_nodes", 0)),
                "unique_deterministic_state_evaluation_keys": int(diagnostics.get("unique_deterministic_state_evaluation_keys", diagnostics.get("deterministic_state_model_evaluation_cache_misses", 0))),
                "deterministic_state_model_evaluation_cache_hits": int(diagnostics.get("deterministic_state_model_evaluation_cache_hits", 0)),
                "deterministic_state_model_evaluation_cache_misses": int(diagnostics.get("deterministic_state_model_evaluation_cache_misses", diagnostics.get("leaf_evaluations", 0))),
                "one_model_evaluation_per_unique_deterministic_state_key_verified": bool(diagnostics.get("one_model_evaluation_per_unique_deterministic_state_key_verified", False)),
                "simulator_transitions": int(diagnostics.get("simulator_transitions", 0)),
                "deterministic_successor_expansions": int(diagnostics.get("deterministic_successor_expansions", 0)),
                "exact_terminal_results_seen": int(diagnostics.get("exact_terminal_results_seen", 0)),
                "value_backups": int(diagnostics.get("value_backups", 0)),
                "max_simulator_search_depth": int(diagnostics.get("max_simulator_search_depth", diagnostics.get("max_depth", 0))),
                "multi_step_simulations": int(diagnostics.get("multi_step_simulations", 0)),
                "selected_branch_depth": len(plan.continuation) + 1 if plan is not None else 0,
                "root_selected_action_stable": bool(
                    plan.root_action_stable if plan is not None else False
                ),
                "root_stability_receipt": (
                    dict(plan.root_stability_receipt)
                    if plan is not None and plan.root_stability_receipt is not None
                    else None
                ),
                "cached_branch_hops": active.cache_hops,
                "cache_hops_this_turn": active.cache_hops,
                "cache_only_later_step": bool(
                    active.atomic_steps > 1
                    and source
                    in {
                        "deterministic_cached_branch",
                        "verified_finite_chance_cached_branch",
                    }
                ),
                "cache_only_later_steps_this_turn": bool(
                    active.cache_only_later_step_count
                ),
                "cache_only_later_step_count_this_turn": (
                    active.cache_only_later_step_count
                ),
                "cached_branch_fingerprint_verification_failures": active.cache_invalidations,
                "cached_branch_invalidations_and_reasons": dict(
                    active.cache_invalidation_reasons
                ),
                "searches_this_turn": active.searches_this_turn,
                "search_segments_this_turn": active.searches_this_turn,
                "search_segment_reasons": dict(active.search_segment_reasons),
                "first_segment_search": bool(active.first_segment_searches),
                "first_segment_searches_this_turn": active.first_segment_searches,
                "later_research_count": active.later_researches,
                "later_research_count_this_turn": active.later_researches,
                "rebuild_searches": active.rebuild_searches,
                "rebuild_searches_this_turn": active.rebuild_searches,
                "rebuild_count_and_reasons": dict(active.rebuild_search_reasons),
                "planner_seconds_used_this_turn": active.work_used_seconds,
                "planner_seconds_residual_this_turn": max(
                    0.0, active.effective_pool_seconds - active.work_used_seconds
                ),
                "finite_chance_outcomes_enumerated": finite_chance_enumerated_this_segment,
                "finite_chance_weighted_backup_count": int(diagnostics.get("finite_chance_weighted_backup_count", 0)),
                "finite_chance_enumeration_count_this_turn": active.finite_chance_enumeration_count,
                "finite_chance_rebuild_count_this_turn": active.finite_chance_rebuild_count,
                "finite_chance_cache_reused": finite_chance_cache_reused,
                "verified_finite_chance_outcome_id": (
                    verified_finite_chance_outcome_id
                    if finite_chance_cache_reused
                    else None
                ),
                "sampled_or_opaque_chance_boundaries": 1 if boundary_reason in {"chance", "information"} else 0,
                "search_stop_reason": diagnostics.get(
                    "search_stop_reason",
                    diagnostics.get("stop_reason", fallback_reason or "valid_private_search"),
                ),
                "full_finite_tree_completion_metric": "not_applicable_root_sampled_stochastic_belief_tree",
                "direct_policy_fallback_used": source == "direct_policy_fallback",
                "direct_fallbacks_this_turn": active.direct_fallbacks,
                "direct_fallback_reasons": dict(active.direct_fallback_reasons),
                "fallback_reason": fallback_reason,
                "planner_result_transaction_resolution": planner_resolution,
                "selected_action": list(selected),
                "selected_action_legal_verified": True,
                "real_actions_dispatched": 1 if dispatch is not None else 0,
            }
        )
        return R215ActionDecision(selected_action=selected, source=source, receipt=receipt)


__all__ = [
    "R215_CHANCE_LABEL",
    "R215_DEFAULT_FIRST_DECISION_SEARCH_SECONDS",
    "R215_DEFAULT_GAME_SECONDS",
    "R215_DEFAULT_LATER_DECISION_SEARCH_SECONDS",
    "R215_DEFAULT_OPERATION_SECONDS",
    "R215_DEFAULT_TURN_POOL_SECONDS",
    "R215_EMERGENCY_DEPTH_SAFETY_CEILING",
    "R215_EMERGENCY_SIMULATION_SAFETY_CEILING",
    "BeliefMCTSPlannerAdapter",
    "R215ActionDecision",
    "R215CachedBranchStep",
    "R215Error",
    "R215FullTurnBeliefMCTS",
    "R215GameClock",
    "R215Observation",
    "R215PlanRequest",
    "R215PlanResult",
    "R215PlannerProtocol",
    "R215TimingConfig",
    "R215TranspositionDecision",
    "R215TurnIdentity",
    "canonical_sha256",
    "legal_action_order_fingerprint",
]
