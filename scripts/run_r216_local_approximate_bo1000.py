#!/usr/bin/env python3
"""Run the owner-authorized r219 multi-search-turn BeliefMCTS mirror.

This is an evaluation-only launcher.  It never imports or invokes a Kaggle,
submission, selector, training, or promotion path.  Each game runs in a fresh
Python process against one seeded native engine.  Within every pair the seed
is identical and the experimental seat is swapped.

The worker deliberately adapts the already exercised frozen-r195 canary at
runtime rather than rewriting the archived submission.  It gives every actual
turn one source-backed, dynamically shrinking 45-second planner pool.  Every
meaningful search segment receives at most 15 seconds and later meaningful
boundaries may re-search only from the residual pool.  Valid deterministic
cache hops, forced steps, and exact direct fallbacks never open a fresh tree.

This file does not launch anything by itself.  Its parent mode prepares either
the required 10-game/5-pair r219 canary or an explicitly sharded BO1000 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Any

SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1"
EVALUATION_ID = "alakazam-r219-local-multi-search-turn-belief-mcts-bo1000"
CANARY_EVALUATION_ID = "alakazam-r219-local-multi-search-turn-belief-mcts-canary10"
OWNER_DECISION_REVISION = 219
TOTAL_GAMES = 1000
MATCHED_PAIRS = 500
CANARY_GAMES = 10
CANARY_PAIRS = 5
GAME_SECONDS = 600.0
GAME_RESERVE_SECONDS = 30.0
TURN_POOL_SECONDS = 45.0
TURN_POOL_DIVISOR = 8.0
SEARCH_SEGMENT_SECONDS = 15.0
EMERGENCY_SIMULATION_SAFETY_CEILING = 1_000_000
EXPECTED_CHECKPOINT = (
    "261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
EXPECTED_MATCHUP_TREE = (
    "e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
EXPECTED_ENGINE = (
    "b77afbd363fe80de968c7cf20a0bbf5eb616fefcacbeab7eeeda94213fad9ea6"
)
EXPECTED_GAME_SCAFFOLD = (
    "a06d67a7a2cef2cdb2b59447b3882c48e60c316e6b184a688594867219951b9a"
)


class R219RunError(RuntimeError):
    """The local evaluation boundary is malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(pair_index: int, evaluation_id: str) -> int:
    digest = hashlib.sha256(
        f"{evaluation_id}:pair:{pair_index}".encode()
    ).digest()
    return (int.from_bytes(digest[:4], "big") % 0xFFFFFFFF) + 1


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        destination.flush()
        os.fsync(destination.fileno())


R219_RUNTIME_PATCH = r'''
# r219 injected local-only turn controller.  This code is inserted into the
# hash-bound evaluator scaffold after its TimedPolicy class is defined and
# before main() constructs either arm.  It imports the r219 bridge lazily only
# after load_main() has placed the sealed experimental package on sys.path.

def _r219_safe_deepcopy(value):
    import copy
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _r219_policy_snapshot(policy):
    router = getattr(policy, "_matchup_adapter_shadow_router", None)
    router_snapshot = (
        router.fork() if router is not None and callable(getattr(router, "fork", None))
        else router
    )
    rng = getattr(policy, "rng", None)
    return {
        "board_history": list(getattr(policy, "board_history", ())),
        "previous_action_history": list(
            getattr(policy, "previous_action_history", ())
        ),
        "kv_cache": getattr(policy, "_kv_cache", None),
        "previous_action_token": _r219_safe_deepcopy(
            getattr(policy, "_previous_action_token", None)
        ),
        "last_result": getattr(policy, "last_result", None),
        "belief_history": _r219_safe_deepcopy(
            getattr(policy, "belief_history", None)
        ),
        "router": router_snapshot,
        "clock": _r219_safe_deepcopy(getattr(policy, "clock", None)),
        "targets": list(getattr(policy, "targets", ())),
        "fail_closed_count": getattr(policy, "fail_closed_count", None),
        "fail_closed_logged": getattr(policy, "_fail_closed_logged", None),
        "last_search_fallback_reason": getattr(
            policy, "last_search_fallback_reason", None
        ),
        "rng_state": (
            rng.getstate() if rng is not None and callable(getattr(rng, "getstate", None))
            else None
        ),
    }


def _r219_restore_policy(policy, snapshot):
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
        policy.targets[:] = snapshot["targets"]
    if snapshot["fail_closed_count"] is not None:
        policy.fail_closed_count = snapshot["fail_closed_count"]
    if snapshot["fail_closed_logged"] is not None:
        policy._fail_closed_logged = snapshot["fail_closed_logged"]
    if hasattr(policy, "last_search_fallback_reason"):
        policy.last_search_fallback_reason = snapshot["last_search_fallback_reason"]
    rng = getattr(policy, "rng", None)
    if snapshot["rng_state"] is not None and callable(getattr(rng, "setstate", None)):
        rng.setstate(snapshot["rng_state"])


class _R219TransactionalPolicyPlanner:
    """Use PolicyAgent search transactionally under r219 controller authority.

    PolicyAgent commits its public-history state while a native MCTS search is
    running.  The outer r219 controller can still reject that result for an
    invalid/untrusted/budget reason, so this planner keeps a full pre-search
    snapshot until the controller explicitly accepts or rejects it.  That
    prevents a direct fallback from appending a second history entry.
    """

    def __init__(self, policy):
        self.policy = policy
        self.pending_snapshot = None
        self.last_native_result = None
        self.last_direct_preview_action = None
        self.last_direct_preview_elapsed_s = 0.0
        self.last_rejection_reason = None
        self.last_resolution = None

    @staticmethod
    def _empty_plan(action, reason, diagnostics=None):
        from types import SimpleNamespace
        payload = {"search_stop_reason": reason}
        if diagnostics:
            payload.update(diagnostics)
        return SimpleNamespace(
            selected_action=tuple(int(index) for index in action),
            sims_run=0,
            continuation=(),
            diagnostics=payload,
            root_action_stable=False,
            root_stability_receipt=None,
        )

    def _preview_exact_direct(self, raw, snapshot):
        started = time.monotonic()
        try:
            action = self.policy.trusted_search_or_greedy_select(
                dict(raw), search=False
            )
            return tuple(int(index) for index in action), time.monotonic() - started
        finally:
            _r219_restore_policy(self.policy, snapshot)

    def plan_turn(self, request):
        from poke_bot.r219_multi_search_turn_belief_mcts import (
            r219_plan_result_from_mcts_result,
        )

        raw = dict(request.observation.raw_observation or {})
        if not raw:
            raise RuntimeError("r219 planner needs the current raw observation")
        if self.pending_snapshot is not None:
            raise RuntimeError("prior r219 planner result was not resolved")
        snapshot = _r219_policy_snapshot(self.policy)
        self.last_native_result = None
        self.last_direct_preview_action = None
        self.last_direct_preview_elapsed_s = 0.0
        self.last_rejection_reason = None
        self.last_resolution = None
        try:
            (
                self.last_direct_preview_action,
                self.last_direct_preview_elapsed_s,
            ) = self._preview_exact_direct(raw, snapshot)
        except Exception as exc:
            _r219_restore_policy(self.policy, snapshot)
            return self._empty_plan(
                request.observation.legal_actions[0],
                "frozen_direct_preview_error",
                {"frozen_direct_preview_error": f"{type(exc).__name__}: {exc}"},
            )

        remaining = max(
            0.0,
            float(request.effective_search_allowance_seconds)
            - float(self.last_direct_preview_elapsed_s),
        )
        if remaining <= 0.0:
            return self._empty_plan(
                self.last_direct_preview_action,
                "no_residual_after_frozen_direct_preview",
                {
                    "r219_frozen_direct_preview_action": list(
                        self.last_direct_preview_action
                    ),
                    "r219_frozen_direct_preview_elapsed_s": (
                        self.last_direct_preview_elapsed_s
                    ),
                },
            )

        self.pending_snapshot = snapshot
        self.policy.max_sims = 1000000
        self.policy.min_trusted_sims = 1
        # r219 owns the single 600/30/8 outer game clock; the submission
        # scheduler must not silently allocate another per-action budget.
        self.policy.clock = None
        self.policy.move_time_s = remaining
        prior = self.policy.last_result
        try:
            selected = self.policy.trusted_search_or_greedy_select(
                raw, search=True
            )
        except Exception:
            # The controller will invoke reject_plan before its exact direct
            # fallback.  Keep the snapshot until then.
            raise
        result = self.policy.last_result
        fallback_reason = getattr(self.policy, "last_search_fallback_reason", None)
        if result is prior or fallback_reason is not None:
            return self._empty_plan(
                self.last_direct_preview_action
                or request.observation.legal_actions[0],
                "policy_search_untrusted_or_fallback",
                {
                    "r219_frozen_direct_preview_action": (
                        list(self.last_direct_preview_action)
                        if self.last_direct_preview_action is not None
                        else None
                    ),
                    "r219_frozen_direct_preview_elapsed_s": (
                        self.last_direct_preview_elapsed_s
                    ),
                    "policy_search_fallback_reason": fallback_reason,
                },
            )
        self.last_native_result = result
        return r219_plan_result_from_mcts_result(
            result,
            selected_action=selected,
            extra_diagnostics={
                "r219_frozen_direct_preview_action": (
                    list(self.last_direct_preview_action)
                    if self.last_direct_preview_action is not None
                    else None
                ),
                "r219_frozen_direct_preview_elapsed_s": (
                    self.last_direct_preview_elapsed_s
                ),
                "r219_policy_clock_owner": "r219_shared_turn_controller",
            },
        )

    def accept_plan(self, _plan):
        if self.pending_snapshot is None:
            raise RuntimeError("r219 accepted a planner result without a snapshot")
        self.pending_snapshot = None
        self.last_resolution = "accepted"

    def reject_plan(self, reason):
        if self.pending_snapshot is not None:
            _r219_restore_policy(self.policy, self.pending_snapshot)
            self.pending_snapshot = None
        self.last_rejection_reason = str(reason)
        self.last_resolution = "rejected"

    def direct_policy(self, observation):
        # The controller calls reject_plan first for a search-origin fallback.
        # The extra guard also protects non-search fallback paths.
        self.reject_plan("exact_frozen_direct_policy")
        action = self.policy.trusted_search_or_greedy_select(
            dict(observation.raw_observation or {}), search=False
        )
        return tuple(int(index) for index in action)


_R219_ORIGINAL_TIMED_POLICY_INIT = TimedPolicy.__init__
_R219_ORIGINAL_TIMED_POLICY_CALL = TimedPolicy.__call__


def _r219_timed_policy_init(self, *args, **kwargs):
    _R219_ORIGINAL_TIMED_POLICY_INIT(self, *args, **kwargs)
    self._r219_turn_key = None
    self._r219_turn_closed_before_step = False
    self._r219_planner = None
    self._r219_bridge = None
    if not self.mcts:
        return
    from poke_bot.r219_multi_search_turn_belief_mcts import (
        R215TurnIdentity,
        R219MultiSearchTurnBeliefMCTS,
        R219PolicyTurnBridge,
        R219TimingConfig,
    )

    self.policy.max_sims = 1000000
    self.policy.min_trusted_sims = 1
    self.policy.clock = None
    self._r219_planner = _R219TransactionalPolicyPlanner(self.policy)
    controller = R219MultiSearchTurnBeliefMCTS(
        self._r219_planner,
        direct_policy=self._r219_planner.direct_policy,
        timing=R219TimingConfig(),
    )
    self._r219_identity_type = R215TurnIdentity
    self._r219_bridge = R219PolicyTurnBridge(controller, self.policy)


def _r219_boundary_reason(obs):
    # Current local BeliefMCTS exposes sampled rather than engine-attested
    # finite chance nodes.  Do not label these exact.  A coin/chance prompt
    # forces a cache invalidation and residual re-search, if any; ordinary
    # public-fingerprint mismatches are handled by the bridge/controller.
    selection = (obs or {}).get("select") or {}
    try:
        context = int(selection.get("context", -1))
    except (TypeError, ValueError):
        context = -1
    return "chance" if context == 46 else None


def _r219_timed_policy_call(self, obs):
    if not self.mcts:
        return _R219_ORIGINAL_TIMED_POLICY_CALL(self, obs)
    current = (obs or {}).get("current") or {}
    key = (int(current.get("yourIndex", -1)), int(current.get("turn", -1)))
    selection = (obs or {}).get("select") or {}
    context = selection.get("context")
    normalized_context = "".join(
        character for character in str(context).lower() if character.isalnum()
    )
    if context == 41 or normalized_context == "isfirst":
        return _R219_ORIGINAL_TIMED_POLICY_CALL(self, obs)

    self.game_trace.append(
        {"kind": "policy_decision", "seat": key[0], "turn": key[1]}
    )
    turn_closed_before_step = False
    if key != self._r219_turn_key:
        if self._r219_turn_key is not None and self._r219_bridge is not None:
            self._r219_bridge.controller.finish_actual_turn()
            turn_closed_before_step = True
        self._r219_turn_key = key

    started = time.monotonic()
    planner = self._r219_planner
    decision = None
    source = "direct_policy_fallback"
    fallback_reason = None
    try:
        decision = self._r219_bridge.act_from_raw(
            dict(obs),
            self._r219_identity_type(key[0], key[1]),
            boundary_reason=_r219_boundary_reason(obs),
        )
        action = list(decision.selected_action)
        source = decision.source
        receipt = dict(decision.receipt)
    except Exception as exc:
        if planner is not None:
            planner.reject_plan(f"bridge_error:{type(exc).__name__}")
        action = list(
            self.policy.trusted_search_or_greedy_select(dict(obs), search=False)
        )
        fallback_reason = f"bridge_error:{type(exc).__name__}"
        receipt = {
            "schema": "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1",
            "fresh_mcts_search_executed": False,
            "search_segments_this_turn": 0,
            "later_research_count_this_turn": 0,
            "direct_policy_fallback_used": True,
            "fallback_reason": fallback_reason,
            "planner_result_transaction_resolution": (
                planner.last_resolution if planner is not None else None
            ),
            "selected_action": list(action),
            "selected_action_legal_verified": False,
        }
    elapsed = max(0.0, time.monotonic() - started)
    result = (
        planner.last_native_result
        if planner is not None and source == "belief_mcts"
        else None
    )
    diagnostics = (
        dict(getattr(getattr(result, "target", None), "diagnostics", {}) or {})
        if result is not None
        else {}
    )
    direct_preview = (
        planner.last_direct_preview_action
        if planner is not None and bool(receipt.get("fresh_mcts_search_executed"))
        else None
    )
    root_receipt = receipt.get("root_stability_receipt")
    row = {
        "turn_key": list(key),
        "actual_turn_key": list(key),
        "actual_atomic_step_index": int(receipt.get("actual_atomic_step_index", 0)),
        "setup_turn_order_control": False,
        "actual_turn_closed_before_step": turn_closed_before_step,
        "elapsed_s": elapsed,
        "search_elapsed_s": float(
            diagnostics.get("elapsed_s", receipt.get("turn_planner_wall_seconds", 0.0))
            or 0.0
        ),
        "mcts_decision_source": source,
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
        "cache_only_later_step_count_this_turn": int(
            receipt.get("cache_only_later_step_count_this_turn", 0) or 0
        ),
        "cache_hops_this_turn": int(receipt.get("cache_hops_this_turn", 0) or 0),
        "cached_branch_fingerprint_verification_failures": int(
            receipt.get("cached_branch_fingerprint_verification_failures", 0) or 0
        ),
        "cache_invalidation_reasons": receipt.get(
            "cached_branch_invalidations_and_reasons", {}
        ),
        "rebuild_count_and_reasons": receipt.get("rebuild_count_and_reasons", {}),
        "plan_endpoint_rebuild": (
            receipt.get("search_segment_boundary_reason")
            == "validated_cached_plan_endpoint"
        ),
        "chance_or_information_rebuild": any(
            marker in str(receipt.get("search_segment_boundary_reason") or "")
            for marker in ("chance", "information", "divergence")
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
        "chance_samples": int(diagnostics.get("chance_samples", 0) or 0),
        "sims": int(receipt.get("sims_run", 0) or 0),
        "leaf_evaluations": int(receipt.get("leaf_evaluations", 0) or 0),
        "unique_nodes": int(receipt.get("unique_nodes", 0) or 0),
        "unique_expanded_nodes": int(
            receipt.get("unique_expanded_nodes", 0) or 0
        ),
        "root_visits": int(diagnostics.get("root_visits", 0) or 0),
        "max_depth": int(
            receipt.get("max_simulator_search_depth", diagnostics.get("max_depth", 0))
            or 0
        ),
        "simulator_transitions": int(
            receipt.get("simulator_transitions", 0) or 0
        ),
        "value_backups": int(receipt.get("value_backups", 0) or 0),
        "multi_step_simulations": int(
            receipt.get("multi_step_simulations", 0) or 0
        ),
        "search_semantics": diagnostics.get("search_semantics"),
        "search_stop_reason": receipt.get("search_stop_reason"),
        "root_action_stable": bool(
            receipt.get("root_selected_action_stable", False)
        ),
        "root_stability_receipt": root_receipt,
        "stable_root_convergence": bool(
            isinstance(root_receipt, dict)
            and root_receipt.get("stable_root_convergence") is True
        ),
        "selected_action_fully_backed_up": bool(
            isinstance(root_receipt, dict)
            and root_receipt.get("selected_action_fully_backed_up") is True
        ),
        "direct_policy_fallback_used": bool(
            receipt.get("direct_policy_fallback_used", False)
            or source == "direct_policy_fallback"
        ),
        "fallback_reason": receipt.get("fallback_reason") or fallback_reason,
        "timing_breach_observed": bool(
            receipt.get("search_segment_budget_breach", False)
            or receipt.get("turn_budget_breach", False)
        ),
        "planner_result_transaction_resolution": receipt.get(
            "planner_result_transaction_resolution"
        ),
        "frozen_direct_policy_action": (
            list(direct_preview) if direct_preview is not None else None
        ),
        "mcts_changed_action_relative_to_frozen_direct_policy": bool(
            source == "belief_mcts"
            and direct_preview is not None
            and list(action) != list(direct_preview)
        ),
        "selected_action": list(action),
        "matchup_adapter_runtime": bool(self.policy.matchup_adapter_runtime),
        "matchup_model_route": int(self.policy._matchup_model_route()),
    }
    self.rows.append(row)
    return action


def _r219_finish_actual_turn(self):
    if self.mcts and self._r219_bridge is not None:
        self._r219_bridge.controller.finish_actual_turn()
        self._r219_turn_key = None


TimedPolicy.__init__ = _r219_timed_policy_init
TimedPolicy.__call__ = _r219_timed_policy_call
TimedPolicy.r219_finish_actual_turn = _r219_finish_actual_turn
'''


def _patch_canary(source: str) -> str:
    """Inject only the separately versioned local r219 evaluator wrapper.

    The scaffold's SHA-256 is checked before this function is called.  Every
    replacement remains strict so a source drift cannot silently reinterpret
    a historical r214/r218 evaluator as r219.
    """

    anchors = {
        "TimedPolicy class": "class TimedPolicy:",
        "main boundary": "\n\ndef main() -> int:",
        "MCTS search settings": (
            "    mcts_policy.max_sims = 20\n"
            "    mcts_policy.min_trusted_sims = 20"
        ),
        "scaffold schema": "poke_bot.r214_simple_belief_mcts_canary/v1",
        "play_game boundary": (
            "    result = play_game(agents[0], agents[1], direct_deck, direct_deck)"
        ),
    }
    for label, anchor in anchors.items():
        if anchor not in source:
            raise R219RunError(f"canary patch target is absent: {label}")
    source = source.replace(
        anchors["MCTS search settings"],
        "    mcts_policy.max_sims = 1000000\n"
        "    mcts_policy.min_trusted_sims = 1\n"
        "    mcts_policy.clock = None",
        1,
    )
    source = source.replace(
        anchors["scaffold schema"],
        "poke_bot.alakazam_local_multi_search_turn_belief_mcts_game_r219/v1",
        1,
    )
    source = source.replace(
        anchors["play_game boundary"],
        anchors["play_game boundary"] + "\n"
        "    mcts.r219_finish_actual_turn()",
        1,
    )
    source = source.replace(
        anchors["main boundary"],
        "\n\n" + R219_RUNTIME_PATCH + anchors["main boundary"],
        1,
    )
    return source


def _worker(args: argparse.Namespace) -> int:
    source = _patch_canary(args.canary_source.read_text(encoding="utf-8"))
    worker_argv = [
        str(args.canary_source),
        "--direct",
        str(args.direct_package),
        "--mcts",
        str(args.mcts_package),
        "--seed",
        str(args.seed),
        "--mcts-seat",
        str(args.mcts_seat),
        "--output",
        str(args.output),
    ]
    prior_argv = sys.argv
    try:
        sys.argv = worker_argv
        namespace = {
            "__name__": "r216_local_game_worker",
            "__file__": str(args.canary_source),
        }
        exec(  # noqa: S102 - execute one pre-hashed local evaluator scaffold.
            compile(source, str(args.canary_source), "exec"), namespace
        )
        return int(namespace["main"]())
    finally:
        sys.argv = prior_argv


def _validate_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if _sha256(args.canary_source) != EXPECTED_GAME_SCAFFOLD:
        raise R219RunError("game scaffold identity drifted")
    for root in (args.direct_package, args.mcts_package):
        if not root.is_dir() or root.is_symlink():
            raise R219RunError(f"package root is not a physical directory: {root}")
        if _sha256(root / "model.pt") != EXPECTED_CHECKPOINT:
            raise R219RunError(f"checkpoint drift in {root}")
        if _sha256(root / "matchup_tree.json") != EXPECTED_MATCHUP_TREE:
            raise R219RunError(f"matchup tree drift in {root}")
        if (root / "rtp_shadow_planner.pt").exists():
            raise R219RunError(f"RTP sidecar is forbidden in {root}")
    if _sha256(args.direct_package / "cg" / "libcg.so") != EXPECTED_ENGINE:
        raise R219RunError("seeded engine identity drifted")
    if _sha256(args.mcts_package / "cg" / "libcg.so") != EXPECTED_ENGINE:
        raise R219RunError("MCTS seeded engine identity drifted")
    direct_search = json.loads(
        (args.direct_package / "search_config.json").read_text(encoding="utf-8")
    )
    mcts_search = json.loads(
        (args.mcts_package / "search_config.json").read_text(encoding="utf-8")
    )
    if direct_search.get("enabled") is not False:
        raise R219RunError("control package is not direct/no-search")
    if mcts_search.get("enabled") is not True:
        raise R219RunError("experimental package did not enable BeliefMCTS")
    forbidden = [
        key
        for key in os.environ
        if any(marker in key.upper() for marker in ("KAGGLE", "GUIDE", "RTP"))
    ]
    return {
        "checkpoint_sha256": f"sha256:{EXPECTED_CHECKPOINT}",
        "matchup_tree_sha256": f"sha256:{EXPECTED_MATCHUP_TREE}",
        "engine_sha256": f"sha256:{EXPECTED_ENGINE}",
        "game_scaffold_sha256": f"sha256:{EXPECTED_GAME_SCAFFOLD}",
        "mcts_enabled": True,
        "direct_search_enabled": False,
        "matchup_adapter_required_on_both_arms": True,
        "rtp_enabled": False,
        "guide_linear_enabled": False,
        "guide_logit_enabled": False,
        "guide2vec_enabled": False,
        "inherited_forbidden_environment_keys": sorted(forbidden),
        "kaggle_authority": False,
        "training_authority": False,
    }


def _game_valid(document: dict[str, Any]) -> tuple[bool, str | None]:
    result = document.get("result") or {}
    rows = document.get("mcts_rows") or []
    completed = result.get("failed_seat") is None and not result.get("incomplete")
    genuine = any(
        int(row.get("sims", 0) or 0) >= 1
        and int(row.get("root_visits", 0) or 0) >= 1
        and int(row.get("leaf_evaluations", 0) or 0) >= 1
        and int(row.get("max_depth", 0) or 0) >= 2
        and row.get("search_semantics")
        == "public_history_root_sampled_information_set_mcts"
        for row in rows
    )
    if not completed:
        return False, "game_not_terminal_or_agent_failure"
    if not genuine:
        return False, "no_genuine_multistep_belief_mcts_decision"
    return True, None


def _run_one(
    *,
    args: argparse.Namespace,
    pair_index: int,
    game_index: int,
    games_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    game_number = pair_index * 2 + game_index
    output = games_dir / f"game-{game_number:04d}.json"
    log = logs_dir / f"game-{game_number:04d}.log"
    seed = _seed(pair_index)
    mcts_seat = game_index
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--canary-source",
        str(args.canary_source),
        "--direct-package",
        str(args.direct_package),
        "--mcts-package",
        str(args.mcts_package),
        "--seed",
        str(seed),
        "--mcts-seat",
        str(mcts_seat),
        "--output",
        str(output),
    ]
    started = time.monotonic()
    environment = dict(os.environ)
    for key in list(environment):
        if any(marker in key.upper() for marker in ("KAGGLE", "GUIDE", "RTP")):
            environment.pop(key, None)
    devices = [
        item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()
    ]
    device = devices[pair_index % len(devices)]
    environment.update(
        {
            # Give each fresh game exactly one GPU and keep both games in a
            # matched seat-swapped pair on the same physical device.  Repeated
            # device entries provide an explicit weighting toward Blackwell.
            "CUDA_VISIBLE_DEVICES": device,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
        }
    )
    with log.open("wb") as destination:
        completed = subprocess.run(
            command,
            stdout=destination,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    elapsed = time.monotonic() - started
    if not output.is_file():
        return {
            "game_number": game_number,
            "pair_index": pair_index,
            "game_index": game_index,
            "seed": seed,
            "mcts_seat": mcts_seat,
            "valid": False,
            "reason": "worker_emitted_no_receipt",
            "exit_code": completed.returncode,
            "wall_seconds": elapsed,
        }
    document = json.loads(output.read_text(encoding="utf-8"))
    valid, reason = _game_valid(document)
    result = document.get("result") or {}
    rows = document.get("mcts_rows") or []
    return {
        "game_number": game_number,
        "pair_index": pair_index,
        "game_index": game_index,
        "seed": seed,
        "mcts_seat": mcts_seat,
        "physical_cuda_device": device,
        "mcts_actual_first": bool(document.get("mcts_actual_first")),
        "winner_seat": result.get("winner"),
        "valid": valid,
        "reason": reason,
        "exit_code": completed.returncode,
        "wall_seconds": elapsed,
        "mcts_simulations": int(document.get("mcts_simulations", 0) or 0),
        "mcts_search_decisions": int(document.get("mcts_search_decisions", 0) or 0),
        "mcts_leaf_evaluations": int(document.get("mcts_leaf_evaluations", 0) or 0),
        "mcts_root_visits": int(document.get("mcts_root_visits", 0) or 0),
        "max_depth": max((int(row.get("max_depth", 0) or 0) for row in rows), default=0),
        "fallbacks": sum(bool(row.get("direct_policy_fallback_used")) for row in rows),
        "root_prior_comparable_decisions": sum(
            row.get("root_policy_prior_top_action") is not None for row in rows
        ),
        "mcts_action_changes_from_root_prior": sum(
            bool(row.get("mcts_changed_action_from_root_policy_prior_top"))
            for row in rows
        ),
        "timing_breaches": sum(bool(row.get("timing_breach_observed")) for row in rows),
        "converged_searches": sum(bool(row.get("root_action_stable")) for row in rows),
    }


def _summary(
    *,
    started: float,
    results: list[dict[str, Any]],
    workers: int,
    pair_start: int,
    pair_count: int,
) -> dict[str, Any]:
    valid = [row for row in results if row["valid"]]
    experimental_wins = sum(
        row.get("winner_seat") == row.get("mcts_seat") for row in valid
    )
    direct_wins = sum(
        row.get("winner_seat") in {0, 1}
        and row.get("winner_seat") != row.get("mcts_seat")
        for row in valid
    )
    draws = len(valid) - experimental_wins - direct_wins
    wall = max(1e-9, time.monotonic() - started)
    rate = len(results) * 3600.0 / wall
    shard_games = pair_count * 2
    remaining = max(0, shard_games - len(results))
    return {
        "schema": SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "status": "running" if len(results) < shard_games else "complete",
        "completed_games": len(results),
        "valid_games": len(valid),
        "invalid_games": len(results) - len(valid),
        "total_games": 1000,
        "matched_pairs": 500,
        "shard_pair_start": pair_start,
        "shard_pair_count": pair_count,
        "shard_total_games": shard_games,
        "active_worker_limit": workers,
        "mcts_wins": experimental_wins,
        "direct_wins": direct_wins,
        "draws": draws,
        "mcts_actual_first_games": sum(bool(row.get("mcts_actual_first")) for row in valid),
        "mcts_actual_second_games": sum(not bool(row.get("mcts_actual_first")) for row in valid),
        "mcts_simulations": sum(row.get("mcts_simulations", 0) for row in results),
        "mcts_root_visits": sum(row.get("mcts_root_visits", 0) for row in results),
        "mcts_leaf_evaluations": sum(row.get("mcts_leaf_evaluations", 0) for row in results),
        "fallbacks": sum(row.get("fallbacks", 0) for row in results),
        "converged_searches": sum(row.get("converged_searches", 0) for row in results),
        "root_prior_comparable_decisions": sum(
            row.get("root_prior_comparable_decisions", 0) for row in results
        ),
        "mcts_action_changes_from_root_prior": sum(
            row.get("mcts_action_changes_from_root_prior", 0) for row in results
        ),
        "max_depth_seen": max((row.get("max_depth", 0) for row in results), default=0),
        "games_per_hour": rate,
        "eta_seconds": remaining * 3600.0 / rate if rate > 0 else None,
        "elapsed_seconds": wall,
        "approximate_non_exact": True,
        "r207_exact_chance": False,
        "training_eligible": False,
        "kaggle_submission_authority": False,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _parent(args: argparse.Namespace) -> int:
    runtime = _validate_runtime(args)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise R219RunError(f"refusing to reuse output root: {output_root}")
    games_dir = output_root / "games"
    logs_dir = output_root / "logs"
    games_dir.mkdir(parents=True)
    logs_dir.mkdir()
    _write_json_atomic(
        output_root / "run-contract.json",
        {
            "schema": SCHEMA,
            "evaluation_id": EVALUATION_ID,
            "owner_decision_revision": 218,
            "total_games": 1000,
            "matched_pairs": 500,
            "workers": args.workers,
            "runtime": runtime,
            "timing": {
                "game_seconds": 600.0,
                "reserve_seconds": 30.0,
                "default_turn_pool_seconds": 20.0,
                "first_decision_search_seconds": 10.0,
                "later_decision_fresh_search_allowed": False,
                "search_watchdog_seconds": 9.5,
                "fixed_simulation_target": None,
                "minimum_valid_simulations": 1,
                "emergency_simulation_ceiling": 1_000_000,
            },
            "no_kaggle_submission": True,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    progress = output_root / "progress.jsonl"
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    work = [
        (pair, game)
        for pair in range(args.pair_start, args.pair_start + args.pair_count)
        for game in (0, 1)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[dict[str, Any]], tuple[int, int]] = {}
        for pair, game in work:
            future = executor.submit(
                _run_one,
                args=args,
                pair_index=pair,
                game_index=game,
                games_dir=games_dir,
                logs_dir=logs_dir,
            )
            futures[future] = (pair, game)
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                pair, game = futures.pop(future)
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - seal failure and continue.
                    row = {
                        "game_number": pair * 2 + game,
                        "pair_index": pair,
                        "game_index": game,
                        "valid": False,
                        "reason": f"parent_worker_error:{type(exc).__name__}:{exc}",
                    }
                results.append(row)
                _append_event(progress, {"schema": SCHEMA, "kind": "game", **row})
                _write_json_atomic(
                    output_root / "summary.json",
                    _summary(
                        started=started,
                        results=results,
                        workers=args.workers,
                        pair_start=args.pair_start,
                        pair_count=args.pair_count,
                    ),
                )
    summary = _summary(
        started=started,
        results=results,
        workers=args.workers,
        pair_start=args.pair_start,
        pair_count=args.pair_count,
    )
    summary["status"] = "complete"
    _write_json_atomic(output_root / "summary.json", summary)
    return 0 if len(results) == args.pair_count * 2 else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--canary-source", type=Path, required=True)
    parser.add_argument("--direct-package", type=Path, required=True)
    parser.add_argument("--mcts-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=500)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mcts-seat", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.seed is None or args.mcts_seat is None or args.output is None:
            parser.error("--worker requires --seed, --mcts-seat, and --output")
    elif args.output_root is None:
        parser.error("parent requires --output-root")
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be in 1..8")
    if args.pair_count < 1 or args.pair_count > 500:
        parser.error("--pair-count must be in 1..500")
    if args.pair_start < 0 or args.pair_start + args.pair_count > 500:
        parser.error("global pair range must stay within 0..499")
    devices = [
        item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()
    ]
    if not devices or any(not item.isdigit() for item in devices):
        parser.error("--cuda-visible-devices must be a comma-separated index list")
    return args


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R219RunError as exc:
        print(f"r219 BO1000 refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
