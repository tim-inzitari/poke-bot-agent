"""Static r219 fresh-process seeded mirror runtime.

This evaluator-only module is the runtime companion to the r219 BO1000
launcher.  It deliberately uses the frozen r195 direct and BeliefMCTS package
inputs, but its experimental action authority is the r219 shared-turn bridge:
one dynamic 45-second actual-turn pool and residual 15-second meaningful
search segments.  It has no training, service, selector, Kaggle, RTP, or
promotion path.
"""

from __future__ import annotations

import copy
import json
import random
import signal
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .r215_full_turn_belief_mcts import (
    R215GameClock,
    R215Observation,
    R215PlanRequest,
    R215PlanResult,
    R215TurnIdentity,
)
from .r215_seeded_mirror_runtime import (
    R215SeededMirrorRuntimeError,
    _first_player,
    _load_package_main,
    _new_seeded_env,
    _reset_seeded_game,
    _runtime_receipt,
    _seeded_engine_identity,
    _turn_order_action,
    _validate_action,
    _make_search_policy,
)
from .r219_multi_search_turn_belief_mcts import (
    R219MultiSearchTurnBeliefMCTS,
    R219PolicyTurnBridge,
    R219TimingConfig,
    r219_plan_result_from_mcts_result,
)
from .seeded_mirror_harness import engine_turn_identity_from_observation


SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219_game/v1"
STRATEGY = (
    "local_approximate_multi_search_per_actual_turn_public_history_root_sampled_"
    "belief_mcts_whole_frozen_model_wrapper"
)
EMERGENCY_SIMULATION_SAFETY_CEILING = 1_000_000


class R219SeededMirrorRuntimeError(R215SeededMirrorRuntimeError):
    """A static r219 worker cannot make a truthful local-only receipt."""


class R219SearchDeadline(TimeoutError):
    """Raised only inside a worker's private meaningful search segment."""


def _safe_copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _policy_snapshot(policy: Any) -> dict[str, Any]:
    """Capture every PolicyAgent field its own transactional wrapper restores."""

    router = getattr(policy, "_matchup_adapter_shadow_router", None)
    return {
        "board_history": list(getattr(policy, "board_history", ())),
        "previous_action_history": list(
            getattr(policy, "previous_action_history", ())
        ),
        "kv_cache": getattr(policy, "_kv_cache", None),
        "previous_action_token": _safe_copy(
            getattr(policy, "_previous_action_token", None)
        ),
        "last_result": getattr(policy, "last_result", None),
        "belief_history": _safe_copy(getattr(policy, "belief_history", None)),
        "router": (
            router.fork()
            if router is not None and callable(getattr(router, "fork", None))
            else router
        ),
        "clock": _safe_copy(getattr(policy, "clock", None)),
        "targets": list(getattr(policy, "targets", ())),
        "fail_closed_count": getattr(policy, "fail_closed_count", None),
        "fail_closed_logged": getattr(policy, "_fail_closed_logged", None),
        "matchup_shadow_failed_logged": getattr(
            policy, "_matchup_shadow_failed_logged", None
        ),
        "last_search_fallback_reason": getattr(
            policy, "last_search_fallback_reason", None
        ),
        "rng_state": (
            policy.rng.getstate()
            if callable(getattr(getattr(policy, "rng", None), "getstate", None))
            else None
        ),
        "strict_runtime": getattr(policy, "strict_runtime", None),
        "use_mcts": getattr(policy, "use_mcts", None),
        "max_sims": getattr(policy, "max_sims", None),
        "min_trusted_sims": getattr(policy, "min_trusted_sims", None),
        "move_time_s": getattr(policy, "move_time_s", None),
    }


def _restore_policy(policy: Any, snapshot: Mapping[str, Any]) -> None:
    policy.board_history = list(snapshot["board_history"])
    policy.previous_action_history = list(snapshot["previous_action_history"])
    policy._kv_cache = snapshot["kv_cache"]
    policy._previous_action_token = snapshot["previous_action_token"]
    policy.last_result = snapshot["last_result"]
    if snapshot["belief_history"] is not None:
        policy.belief_history = snapshot["belief_history"]
    if snapshot["router"] is not None:
        policy._matchup_adapter_shadow_router = snapshot["router"]
    policy.clock = snapshot["clock"]
    if hasattr(policy, "targets"):
        policy.targets[:] = list(snapshot["targets"])
    for name, key in (
        ("fail_closed_count", "fail_closed_count"),
        ("_fail_closed_logged", "fail_closed_logged"),
        ("_matchup_shadow_failed_logged", "matchup_shadow_failed_logged"),
        ("last_search_fallback_reason", "last_search_fallback_reason"),
        ("strict_runtime", "strict_runtime"),
        ("use_mcts", "use_mcts"),
        ("max_sims", "max_sims"),
        ("min_trusted_sims", "min_trusted_sims"),
        ("move_time_s", "move_time_s"),
    ):
        if snapshot.get(key) is not None and hasattr(policy, name):
            setattr(policy, name, snapshot[key])
    rng = getattr(policy, "rng", None)
    if snapshot.get("rng_state") is not None and callable(
        getattr(rng, "setstate", None)
    ):
        rng.setstate(snapshot["rng_state"])


@contextmanager
def _meaningful_segment_deadline(seconds: float) -> Iterator[None]:
    """Bound a private search segment, not individual model/simulator calls."""

    if float(seconds) <= 0.0:
        raise R219SearchDeadline("r219 private segment has no residual allowance")
    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        raise R219SeededMirrorRuntimeError(
            "r219 worker needs a Unix monotonic segment deadline"
        )
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise R219SeededMirrorRuntimeError(
            "r219 worker inherited an unrelated real-time deadline"
        )
    prior_handler = signal.getsignal(signal.SIGALRM)

    def expire(_signum: int, _frame: Any) -> None:
        raise R219SearchDeadline("r219 meaningful search segment deadline")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_handler)


class R219TransactionalPolicyPlanner:
    """Run native BeliefMCTS under r219 acceptance/rejection authority.

    The native policy commits public history as it searches.  We retain a
    complete pre-search snapshot until the r219 controller accepts its
    selected legal action.  On any controller rejection we restore that
    snapshot before exactly one direct-policy action is committed.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.pending_snapshot: dict[str, Any] | None = None
        self.last_native_result: Any | None = None
        self.last_direct_preview_action: tuple[int, ...] | None = None
        self.last_direct_preview_elapsed_s = 0.0
        self.last_direct_fallback_elapsed_s = 0.0
        self.last_fallback_reserve_s = 0.0
        self.last_private_search_allowance_s = 0.0
        self.last_rejection_reason: str | None = None
        self.last_resolution: str | None = None
        self.call_events: list[dict[str, Any]] = []

    def _record_call(
        self, kind: str, started: float, outcome: str, **extra: Any
    ) -> None:
        ended = time.monotonic()
        self.call_events.append(
            {
                "call_kind": kind,
                "started_monotonic_seconds": started,
                "ended_monotonic_seconds": ended,
                "elapsed_wall_seconds": max(0.0, ended - started),
                "outcome_or_deadline_reason": outcome,
                **extra,
            }
        )

    @staticmethod
    def _empty_plan(
        action: Sequence[int], reason: str, diagnostics: Mapping[str, Any] | None = None
    ) -> R215PlanResult:
        payload = {"search_stop_reason": reason}
        if diagnostics:
            payload.update(dict(diagnostics))
        return R215PlanResult(
            selected_action=tuple(int(index) for index in action),
            sims_run=0,
            diagnostics=payload,
        )

    def _preview_direct(
        self, raw: Mapping[str, Any], snapshot: Mapping[str, Any]
    ) -> tuple[tuple[int, ...], float]:
        started = time.monotonic()
        try:
            action = self.policy.trusted_search_or_greedy_select(dict(raw), search=False)
            selected = tuple(int(index) for index in action)
            self._record_call("frozen_direct_preview", started, "completed")
            return selected, max(0.0, time.monotonic() - started)
        except Exception as exc:
            self._record_call(
                "frozen_direct_preview", started, f"error:{type(exc).__name__}"
            )
            raise
        finally:
            _restore_policy(self.policy, snapshot)

    def plan_turn(self, request: R215PlanRequest) -> R215PlanResult:
        raw = request.observation.raw_observation
        if not isinstance(raw, Mapping):
            raise R219SeededMirrorRuntimeError(
                "r219 planner needs the current raw observation"
            )
        if self.pending_snapshot is not None:
            raise R219SeededMirrorRuntimeError(
                "r219 prior planner transaction was not resolved"
            )
        snapshot = _policy_snapshot(self.policy)
        self.last_native_result = None
        self.last_direct_preview_action = None
        self.last_direct_preview_elapsed_s = 0.0
        self.last_direct_fallback_elapsed_s = 0.0
        self.last_fallback_reserve_s = 0.0
        self.last_private_search_allowance_s = 0.0
        self.last_rejection_reason = None
        self.last_resolution = None
        try:
            preview, preview_elapsed = self._preview_direct(raw, snapshot)
        except Exception as exc:
            _restore_policy(self.policy, snapshot)
            return self._empty_plan(
                request.observation.legal_actions[0],
                "frozen_direct_preview_error",
                {"frozen_direct_preview_error": f"{type(exc).__name__}: {exc}"},
            )
        self.last_direct_preview_action = preview
        self.last_direct_preview_elapsed_s = preview_elapsed
        segment_allowance = max(0.0, float(request.search_allowance_seconds))
        reserve = min(2.0, max(0.25, 2.0 * preview_elapsed))
        private_allowance = max(0.0, segment_allowance - preview_elapsed - reserve)
        self.last_fallback_reserve_s = reserve
        self.last_private_search_allowance_s = private_allowance
        common = {
            "r219_frozen_direct_preview_action": list(preview),
            "r219_frozen_direct_preview_elapsed_s": preview_elapsed,
            "r219_direct_fallback_reserve_s": reserve,
            "r219_private_search_allowance_s": private_allowance,
            "r219_search_segment_allowance_s": segment_allowance,
        }
        if private_allowance <= 0.0:
            return self._empty_plan(
                preview, "no_residual_after_direct_preview_and_reserve", common
            )

        self.pending_snapshot = snapshot
        # This invokes the normal strict native BeliefMCTS path once.  It does
        # not use trusted_search_or_greedy_select(search=True), whose internal
        # fallback would otherwise create a second direct inference before r219
        # can reject and restore the transaction.
        self.policy.max_sims = EMERGENCY_SIMULATION_SAFETY_CEILING
        self.policy.min_trusted_sims = 1
        self.policy.clock = None
        self.policy.move_time_s = private_allowance
        self.policy.strict_runtime = True
        self.policy.use_mcts = True
        prior_result = snapshot["last_result"]
        started = time.monotonic()
        try:
            with _meaningful_segment_deadline(private_allowance):
                selected = tuple(int(index) for index in self.policy(dict(raw)))
        except Exception as exc:
            self._record_call(
                "private_belief_mcts_segment",
                started,
                f"error:{type(exc).__name__}",
                effective_search_allowance_seconds=private_allowance,
            )
            raise
        self._record_call(
            "private_belief_mcts_segment",
            started,
            "completed",
            effective_search_allowance_seconds=private_allowance,
        )
        result = getattr(self.policy, "last_result", None)
        fallback = getattr(self.policy, "last_search_fallback_reason", None)
        if result is None or result is prior_result or fallback:
            return self._empty_plan(
                preview,
                "native_search_untrusted_or_missing_result",
                {**common, "native_search_fallback_reason": fallback},
            )
        self.last_native_result = result
        return r219_plan_result_from_mcts_result(
            result,
            selected_action=selected,
            extra_diagnostics={
                **common,
                "r219_policy_clock_owner": "r219_shared_actual_turn_controller",
            },
        )

    def accept_plan(self, _plan: R215PlanResult) -> None:
        if self.pending_snapshot is None:
            raise R219SeededMirrorRuntimeError(
                "r219 accepted a planner result without a transaction"
            )
        self.pending_snapshot = None
        self.last_resolution = "accepted"

    def reject_plan(self, reason: str) -> None:
        if self.pending_snapshot is not None:
            _restore_policy(self.policy, self.pending_snapshot)
            self.pending_snapshot = None
        self.last_rejection_reason = str(reason)
        self.last_resolution = "rejected"

    def direct_policy(self, observation: R215Observation) -> tuple[int, ...]:
        if not isinstance(observation.raw_observation, Mapping):
            raise R219SeededMirrorRuntimeError(
                "r219 direct fallback needs a raw observation"
            )
        self.reject_plan("exact_frozen_r195_direct_policy")
        started = time.monotonic()
        try:
            action = self.policy.trusted_search_or_greedy_select(
                dict(observation.raw_observation), search=False
            )
            self._record_call("exact_frozen_direct_fallback", started, "completed")
            return tuple(int(index) for index in action)
        except Exception as exc:
            self._record_call(
                "exact_frozen_direct_fallback",
                started,
                f"error:{type(exc).__name__}",
            )
            raise
        finally:
            self.last_direct_fallback_elapsed_s = max(
                0.0, time.monotonic() - started
            )


def _manual_chance_context(observation: Mapping[str, Any]) -> bool:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return False
    try:
        return int(select.get("context", -1)) == 46
    except (TypeError, ValueError):
        return False


def _sample_manual_chance(
    observation: Mapping[str, Any], rng: random.Random
) -> tuple[list[int], int]:
    """Choose an engine-presented manual chance result uniformly, never by MCTS."""

    from poke_bot import features

    combinations = features.enumerate_action_combos(dict(observation))
    if not combinations:
        raise R219SeededMirrorRuntimeError(
            "manual chance prompt lacks a complete legal outcome set"
        )
    index = rng.randrange(len(combinations))
    return _validate_action(observation, combinations[index]), index


def _close_actual_turn(
    controller: R219MultiSearchTurnBeliefMCTS,
    receipts: list[dict[str, Any]],
    *,
    reason: str,
) -> None:
    """Capture every explicit close summary before a new turn or terminal end."""

    receipt = controller.finish_actual_turn()
    if receipt is not None:
        closed = dict(receipt)
        closed["close_reason"] = reason
        receipts.append(closed)


def _runtime_pair_receipt(
    *,
    direct_model: Any,
    direct_policy: Any,
    mcts_model: Any,
    mcts_policy: Any,
    seeded_engine: Mapping[str, Any],
) -> dict[str, Any]:
    direct = _runtime_receipt(
        model=direct_model, policy=direct_policy, seeded_engine=seeded_engine
    )
    experimental = _runtime_receipt(
        model=mcts_model, policy=mcts_policy, seeded_engine=seeded_engine
    )
    return {
        "direct_arm": direct,
        "mcts_arm": experimental,
        "same_frozen_r195_no_rtp_checkpoint_required": True,
        "matchup_adapter_runtime_on_both_arms": True,
        "adapter_bank_active_on_both_arms": True,
        "rtp_and_legacy_rtp_off_on_both_arms": True,
        "guide_linear_guide_logit_guide2vec_off_on_both_arms": True,
    }


def _row_from_decision(
    *,
    decision: Any,
    planner: R219TransactionalPolicyPlanner,
    policy: Any,
    identity: R215TurnIdentity,
    boundary_reason: str | None,
) -> dict[str, Any]:
    receipt = dict(decision.receipt)
    native = (
        planner.last_native_result
        if receipt.get("fresh_mcts_search_executed") is True
        else None
    )
    diagnostics_raw = getattr(getattr(native, "target", None), "diagnostics", {})
    diagnostics = dict(diagnostics_raw) if isinstance(diagnostics_raw, Mapping) else {}
    stability = receipt.get("root_stability_receipt")
    route = policy.matchup_adapter_shadow_snapshot()
    preview = (
        planner.last_direct_preview_action
        if receipt.get("fresh_mcts_search_executed") is True
        else None
    )
    return {
        "schema": SCHEMA,
        "actual_turn_key": [identity.seat, identity.actual_turn_id],
        "actual_atomic_step_index": int(receipt.get("actual_atomic_step_index", 0)),
        "mcts_decision_source": decision.source,
        "boundary_reason_supplied": boundary_reason,
        "fresh_mcts_search_executed": bool(
            receipt.get("fresh_mcts_search_executed", False)
        ),
        "search_segment_index": receipt.get("search_segment_index"),
        "search_segment_boundary_reason": receipt.get(
            "search_segment_boundary_reason"
        ),
        "search_segments_this_turn": int(
            receipt.get("search_segments_this_turn", 0) or 0
        ),
        "later_research_count_this_turn": int(
            receipt.get("later_research_count_this_turn", 0) or 0
        ),
        "effective_search_segment_allowance_s": float(
            receipt.get("effective_search_segment_allowance_seconds", 0.0) or 0.0
        ),
        "shared_turn_pool_s": float(
            receipt.get("effective_actual_turn_planner_pool_seconds", 0.0) or 0.0
        ),
        "shared_turn_pool_used_s": float(
            receipt.get("planner_seconds_used_this_turn", 0.0) or 0.0
        ),
        "shared_turn_pool_remaining_s": float(
            receipt.get("planner_seconds_residual_this_turn", 0.0) or 0.0
        ),
        "game_clock_remaining_s": float(
            receipt.get("game_clock_remaining_seconds_after", 0.0) or 0.0
        ),
        "cache_only_later_step": bool(
            receipt.get("cache_only_later_step", False)
        ),
        "cache_hops_this_turn": int(receipt.get("cache_hops_this_turn", 0) or 0),
        "cache_invalidation_reasons": receipt.get(
            "cached_branch_invalidations_and_reasons", {}
        ),
        "rebuild_count_and_reasons": receipt.get("rebuild_count_and_reasons", {}),
        "chance_or_information_rebuild": any(
            token in str(receipt.get("search_segment_boundary_reason") or "")
            for token in ("chance", "information", "divergence")
        ),
        "finite_chance_enumerations": int(
            receipt.get("finite_chance_outcomes_enumerated", 0) or 0
        ),
        "finite_chance_weighted_backups": int(
            receipt.get("finite_chance_weighted_backup_count", 0) or 0
        ),
        "sampled_or_opaque_chance_boundaries": int(
            receipt.get("sampled_or_opaque_chance_boundaries", 0) or 0
        ),
        "sims": int(receipt.get("sims_run", 0) or 0),
        "root_visits": int(diagnostics.get("root_visits", 0) or 0),
        "leaf_evaluations": int(receipt.get("leaf_evaluations", 0) or 0),
        "unique_nodes": int(receipt.get("unique_nodes", 0) or 0),
        "unique_expanded_nodes": int(
            receipt.get("unique_expanded_nodes", 0) or 0
        ),
        "max_depth": int(
            receipt.get(
                "max_simulator_search_depth", diagnostics.get("max_depth", 0)
            )
            or 0
        ),
        "simulator_transitions": int(
            receipt.get("simulator_transitions", 0) or 0
        ),
        "value_backups": int(receipt.get("value_backups", 0) or 0),
        "search_semantics": diagnostics.get("search_semantics"),
        "search_stop_reason": receipt.get("search_stop_reason"),
        "root_action_stable": bool(
            receipt.get("root_selected_action_stable", False)
        ),
        "stable_root_convergence": bool(
            isinstance(stability, Mapping)
            and stability.get("stable_root_convergence") is True
        ),
        "selected_action_fully_backed_up": bool(
            isinstance(stability, Mapping)
            and stability.get("selected_action_fully_backed_up") is True
        ),
        "direct_policy_fallback_used": bool(
            receipt.get("direct_policy_fallback_used", False)
            or decision.source == "direct_policy_fallback"
        ),
        "fallback_reason": receipt.get("fallback_reason"),
        "timing_breach_observed": bool(
            receipt.get("search_segment_budget_breach", False)
            or receipt.get("turn_budget_breach", False)
        ),
        "planner_result_transaction_resolution": receipt.get(
            "planner_result_transaction_resolution"
        ),
        "frozen_direct_policy_action": list(preview) if preview is not None else None,
        "mcts_changed_action_relative_to_frozen_direct_policy": bool(
            decision.source == "belief_mcts"
            and preview is not None
            and list(decision.selected_action) != list(preview)
        ),
        "direct_preview_elapsed_s": planner.last_direct_preview_elapsed_s,
        "direct_fallback_reserve_s": planner.last_fallback_reserve_s,
        "private_search_allowance_s": planner.last_private_search_allowance_s,
        "selected_action": list(decision.selected_action),
        "matchup_adapter_runtime": bool(
            getattr(policy, "matchup_adapter_runtime", False)
        ),
        "matchup_adapter_route_receipt": route,
    }


def _run_game(request: Mapping[str, Any]) -> dict[str, Any]:
    game = request.get("game")
    pair_seal = request.get("pair_first_player_seal")
    if not isinstance(game, Mapping) or not isinstance(pair_seal, Mapping):
        raise R219SeededMirrorRuntimeError("r219 game request lacks a sealed pair")
    direct_root = Path(str(request.get("direct_package_root", ""))).resolve()
    mcts_root = Path(str(request.get("mcts_package_root", ""))).resolve()
    if not direct_root.is_dir() or not mcts_root.is_dir():
        raise R219SeededMirrorRuntimeError("r219 game lacks physical package roots")
    seeded_engine = _seeded_engine_identity(request)

    # Loading direct first then mcts reproduces the ordinary package import
    # order.  The sealed source snapshot overlays the r219 stack into both
    # namespaces; the two independently loaded models remain the two frozen
    # r195 arm inputs.
    direct_module = _load_package_main(direct_root)
    direct_deck, direct_model, direct_policy = direct_module._ensure_runtime()
    mcts_module = _load_package_main(mcts_root)
    mcts_deck, mcts_model, _unused_mcts_policy = mcts_module._ensure_runtime()
    if list(direct_deck) != list(mcts_deck):
        raise R219SeededMirrorRuntimeError("r219 direct/mcts frozen deck bytes diverged")

    direct_policy.reset_game()
    direct_policy.strict_runtime = True
    direct_policy.rng = random.Random(int(game["control_rng_seed_u32"]))
    mcts_policy = _make_search_policy(
        mcts_module,
        deck=list(mcts_deck),
        model=mcts_model,
        rng_seed=int(game["experimental_rng_seed_u32"]),
    )
    mcts_policy.reset_game()
    mcts_policy.strict_runtime = True
    planner = R219TransactionalPolicyPlanner(mcts_policy)
    controller = R219MultiSearchTurnBeliefMCTS(
        planner,
        direct_policy=planner.direct_policy,
        timing=R219TimingConfig(),
        game_clock=R215GameClock(),
    )
    bridge = R219PolicyTurnBridge(controller, mcts_policy)
    runtime = _runtime_pair_receipt(
        direct_model=direct_model,
        direct_policy=direct_policy,
        mcts_model=mcts_model,
        mcts_policy=mcts_policy,
        seeded_engine=seeded_engine,
    )
    environment = _new_seeded_env(seeded_engine)
    rows: list[dict[str, Any]] = []
    close_receipts: list[dict[str, Any]] = []
    chance_events: list[dict[str, Any]] = []
    setup_actions: list[dict[str, Any]] = []
    direct_actions = 0
    first_player_verified = False
    active_experimental_identity: R215TurnIdentity | None = None
    pending_chance_boundary: R215TurnIdentity | None = None
    chance_rng = random.Random(int(game["chance_rng_seed_u32"]))
    steps = 0
    max_steps = int(request.get("max_atomic_steps", 4000))
    try:
        state = _reset_seeded_game(
            environment, list(direct_deck), int(game["engine_seed_u32"])
        )
        while not state.done and steps < max_steps:
            observation = state.obs
            first = _first_player(observation)
            if first is None:
                action = _turn_order_action(direct_module, observation)
                if action is None:
                    raise R219SeededMirrorRuntimeError(
                        "unsealed setup state has no frozen r195 action"
                    )
                setup_actions.append(
                    {
                        "action": action,
                        "acting_seat": (observation.get("current") or {}).get(
                            "yourIndex"
                        ),
                    }
                )
                state = environment.step_batch([action]).envs[0]
                steps += 1
                continue
            if first != pair_seal.get("first_player_seat"):
                raise R219SeededMirrorRuntimeError(
                    "game first player disagrees with sealed pair material"
                )
            first_player_verified = True
            turn_order = _turn_order_action(direct_module, observation)
            if turn_order is not None:
                setup_actions.append(
                    {
                        "action": turn_order,
                        "acting_seat": (observation.get("current") or {}).get(
                            "yourIndex"
                        ),
                    }
                )
                state = environment.step_batch([turn_order]).envs[0]
                steps += 1
                continue

            engine_identity = engine_turn_identity_from_observation(observation)
            identity = R215TurnIdentity(
                int(engine_identity.seat), int(engine_identity.actual_turn_id)
            )
            experimental_seat = int(game["experimental_seat"])
            if (
                active_experimental_identity is not None
                and (
                    identity != active_experimental_identity
                    or identity.seat != experimental_seat
                )
            ):
                _close_actual_turn(
                    controller,
                    close_receipts,
                    reason="engine_actual_turn_transition",
                )
                active_experimental_identity = None

            if _manual_chance_context(observation):
                action, outcome_index = _sample_manual_chance(observation, chance_rng)
                chance_events.append(
                    {
                        "actual_turn_key": [identity.seat, identity.actual_turn_id],
                        "manual_chance": True,
                        "selection_mode": "uniform_realized_outcome",
                        "legal_outcome_count": len(
                            __import__("poke_bot.features", fromlist=["x"]).enumerate_action_combos(
                                dict(observation)
                            )
                        ),
                        "selected_outcome_index": outcome_index,
                        "r207_exact_finite_chance_claim": False,
                    }
                )
                if identity.seat == experimental_seat:
                    pending_chance_boundary = identity
                state = environment.step_batch([action]).envs[0]
                steps += 1
                continue

            if identity.seat == experimental_seat:
                boundary = (
                    "chance"
                    if pending_chance_boundary is not None
                    and pending_chance_boundary == identity
                    else None
                )
                next_holder: dict[str, Any] = {}

                def dispatch(action: list[int]) -> None:
                    next_holder["state"] = environment.step_batch([action]).envs[0]

                try:
                    decision = bridge.act_from_raw(
                        dict(observation),
                        identity,
                        boundary_reason=boundary,
                        dispatch=dispatch,
                    )
                except Exception:
                    # No uncharged out-of-controller fallback is permitted.
                    # If the bridge could not create a legal/current controller
                    # action, this game is an explicit invalid worker receipt.
                    planner.reject_plan("bridge_error")
                    raise
                if "state" not in next_holder:
                    raise R219SeededMirrorRuntimeError(
                        "r219 bridge did not dispatch its verified action"
                    )
                rows.append(
                    _row_from_decision(
                        decision=decision,
                        planner=planner,
                        policy=mcts_policy,
                        identity=identity,
                        boundary_reason=boundary,
                    )
                )
                state = next_holder["state"]
                active_experimental_identity = identity
                if boundary is not None:
                    pending_chance_boundary = None
            elif identity.seat == int(game["control_seat"]):
                action = _validate_action(
                    observation,
                    direct_policy.trusted_search_or_greedy_select(
                        dict(observation), search=False
                    ),
                )
                state = environment.step_batch([action]).envs[0]
                direct_actions += 1
            else:
                raise R219SeededMirrorRuntimeError(
                    "engine emitted an unknown acting seat"
                )
            steps += 1

        _close_actual_turn(controller, close_receipts, reason="terminal_or_loop_end")
        active_experimental_identity = None
        if not first_player_verified:
            raise R219SeededMirrorRuntimeError(
                "game ended before first-player seal was verified"
            )
        if not state.done:
            raise R219SeededMirrorRuntimeError(
                "game exhausted max atomic steps before terminal state"
            )
        return {
            "schema": SCHEMA,
            "operation": "run_game",
            "evaluation_id": request["evaluation_id"],
            "source_identity_sha256": request["source_identity_sha256"],
            "output_identity_sha256": request["output_identity_sha256"],
            **dict(game),
            "first_player_seat": pair_seal["first_player_seat"],
            "experimental_actual_turn_order": (
                "first"
                if int(game["experimental_seat"])
                == int(pair_seal["first_player_seat"])
                else "second"
            ),
            "experimental_strategy": STRATEGY,
            "terminal_status": "completed",
            "winner_seat": state.winner,
            "steps": steps,
            "invalid_action": False,
            "crash": False,
            "packaged_turn_order_actions": setup_actions,
            "experimental_turn_receipts": rows,
            "experimental_actual_turn_close_receipts": close_receipts,
            "manual_chance_events": chance_events,
            "individual_call_telemetry": planner.call_events,
            "direct_atomic_actions": direct_actions,
            "runtime_receipt": runtime,
        }
    finally:
        environment.close()


def run_r219_seeded_mirror_operation(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute exactly one sealed-pair or game operation in a fresh child."""

    operation = request.get("operation")
    if operation == "seal_pair":
        # Pair sealing has no experimental action authority and may safely use
        # the direct package's frozen startup path.
        from .r215_seeded_mirror_runtime import _seal_pair

        return _seal_pair(request)
    if operation == "run_game":
        return _run_game(request)
    raise R219SeededMirrorRuntimeError("unsupported r219 seeded mirror operation")
