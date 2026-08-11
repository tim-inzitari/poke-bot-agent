#!/usr/bin/env python3
"""Run one fail-closed r229 r228-MCTS versus frozen-r195-direct game."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r229_game/v1"
CHECKPOINT = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
MATCHUP_TREE = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
CANONICAL_NATIVE_LIBRARIES = {
    "linux_x86_64": ("cg/libcg.so", "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7", 1_342_400),
    "linux_aarch64": ("cg/libcg-arm64.so", "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2", 1_296_464),
    "macos_arm64": ("cg/libcg.dylib", "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30", 1_245_544),
    "windows_x86_64": ("cg/cg.dll", "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771", 1_525_248),
}
PLATFORM_LIBRARY = {
    ("linux", "x86_64"): "linux_x86_64",
    ("linux", "amd64"): "linux_x86_64",
    ("linux", "aarch64"): "linux_aarch64",
    ("linux", "arm64"): "linux_aarch64",
    ("darwin", "arm64"): "macos_arm64",
    ("darwin", "aarch64"): "macos_arm64",
    ("windows", "amd64"): "windows_x86_64",
    ("windows", "x86_64"): "windows_x86_64",
}


class R229GameError(RuntimeError):
    pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_bytes(encoded)
    os.replace(partial, path)


def _verify_canonical_native_set(stage: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    manifest_path = stage / "r252_fleet_evaluation_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise R229GameError("sealed package lacks a readable r252 manifest") from exc
    expected_manifest_libraries = {
        name: {"path": relative, "sha256": digest, "size_bytes": size}
        for name, (relative, digest, size) in CANONICAL_NATIVE_LIBRARIES.items()
    }
    if (
        manifest.get("schema")
        != "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r252_package/v1"
        or manifest.get("status") != "sealed_serial_bounded_leaf_evaluation_only"
        or manifest.get("owner_goal_revision") != 252
        or manifest.get("canonical_libcg_revision") != 236
        or manifest.get("superseded_two_lane_topology_revision") != 239
        or manifest.get("owner_handle_scoped_search_id_revision") != 244
        or manifest.get("owner_process_lane_recovery_revision") != 249
        or manifest.get("owner_serial_mcts_revision") != 250
        or manifest.get("owner_internal_leaf_boundary_revision") != 252
        or manifest.get("native_simulator_worker_process_count") != 1
        or manifest.get("shared_tree_and_frozen_model_remain_in_parent") is not True
        or manifest.get("native_search_calls_in_parent_worker_threads") is not False
        or manifest.get("concurrent_libcg_search_calls_allowed") is not False
        or manifest.get("complete_serial_retry_count_after_fault") != 1
        or manifest.get("failed_partial_tree_reuse_allowed") is not False
        or manifest.get("search_seconds_per_attempt") != 8.0
        or manifest.get("exhausted_recovery_direct_fallback_is_degraded") is not True
        or manifest.get("clean_full_game_preflight_max_exhausted_recovery_fallbacks") != 0
        or manifest.get("simulator_lane_count") != 1
        or manifest.get("internal_agent_start_arena_count") != 1
        or manifest.get("required_search_begin_call_count") != 1
        or manifest.get("required_handle_identity_count") != 1
        or manifest.get("required_handle_scoped_search_id_chain_count") != 1
        or manifest.get("required_handle_first_search_id_composite_count") != 1
        or manifest.get("handle_scoped_first_search_id_composite_state_array_field")
        != "handle_scoped_first_search_id_composite_states"
        or manifest.get(
            "handle_scoped_first_search_id_composite_state_entry_exact_keys_in_order"
        )
        != ["lane_id", "handle_identity", "first_search_id"]
        or manifest.get("search_begin_identity_scope")
        != "arena_handle_plus_handle_local_search_id"
        or manifest.get("raw_search_id_global_uniqueness_required") is not False
        or manifest.get("logical_frontier_leaf_count_per_frozen_model_batch") != 1
        or manifest.get("partial_frontier_batches_allowed") is not False
        or manifest.get("serial_one_lane_continuation_required") is not True
        or manifest.get("one_shared_logical_mcts_tree_required") is not True
        or manifest.get("process_parallel_node_evaluation_included") is not False
        or manifest.get("complete_ordered_action_ceiling") != 65536
        or manifest.get("internal_ordered_action_expansion_ceiling") != 64
        or manifest.get("every_explicit_chance_context_is_pre_random_value_boundary") is not True
        or manifest.get("explicit_chance_probability_distribution_assumed") is not False
        or manifest.get("deterministic_internal_fanout_over_64_is_value_only_boundary") is not True
        or manifest.get("internal_boundary_representative_action_has_no_tree_authority") is not True
        or manifest.get("internal_boundary_has_action_or_child_authority") is not False
        or manifest.get("internal_value_boundary_telemetry_required") is not True
        or manifest.get("bo_lifecycle_revision") != 233
        or manifest.get("r234_kaggle_broker_or_queue_lifecycle_included") is not False
        or manifest.get("kaggle_search_policy_changes_included") is not False
        or manifest.get("r249_bo_process_lane_boundary_included") is not True
        or manifest.get("r250_serial_process_lane_topology_included") is not True
        or manifest.get("r252_internal_leaf_boundary_included") is not True
        or manifest.get("canonical_native_libraries") != expected_manifest_libraries
    ):
        raise R229GameError("sealed r252 package manifest identity drifted")
    cg_root = (stage / "cg").resolve(strict=True)
    expected_paths = {row[0] for row in CANONICAL_NATIVE_LIBRARIES.values()}
    observed_paths = {
        path.relative_to(stage).as_posix()
        for path in cg_root.iterdir()
        if path.is_file() and (path.name.startswith("libcg") or path.name == "cg.dll")
    }
    if observed_paths != expected_paths:
        raise R229GameError("sealed package has a mixed or incomplete canonical libcg set")
    receipt: dict[str, dict[str, object]] = {}
    for platform_name, (relative, expected_sha, expected_size) in CANONICAL_NATIVE_LIBRARIES.items():
        path = (stage / relative).resolve(strict=True)
        if path.parent != cg_root or path.stat().st_size != expected_size or _sha256(path) != expected_sha:
            raise R229GameError(f"canonical libcg member drifted: {relative}")
        receipt[platform_name] = {
            "path": relative,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    host_key = (platform.system().lower(), platform.machine().lower())
    platform_name = PLATFORM_LIBRARY.get(host_key)
    if platform_name is None:
        raise R229GameError(f"unsupported canonical libcg platform: {host_key}")
    selected = {"platform_identity": platform_name, **receipt[platform_name]}
    return receipt, selected


def _load(stage: Path) -> Any:
    sys.dont_write_bytecode = True
    os.chdir(stage)
    sys.path[:] = [str(stage), *[item for item in sys.path if item and item != str(stage)]]
    for name in list(sys.modules):
        if name == "poke_bot" or name.startswith("poke_bot.") or name == "cg" or name.startswith("cg."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("r229_r228_package", stage / "main.py")
    if spec is None or spec.loader is None:
        raise R229GameError("cannot import sealed r228 main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_serial_receipts(receipts: Sequence[Mapping[str, Any]]) -> None:
    for row in receipts:
        boundary_count = row.get("internal_value_boundary_count")
        boundary_reasons = row.get("internal_value_boundary_reasons")
        max_internal = row.get("max_internal_ordered_action_count")
        if (
            isinstance(boundary_count, bool)
            or not isinstance(boundary_count, int)
            or boundary_count < 0
            or not isinstance(boundary_reasons, Mapping)
            or any(
                reason
                not in {
                    "explicit_chance_pre_random",
                    "deterministic_internal_fanout_over_64",
                }
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for reason, count in boundary_reasons.items()
            )
            or sum(boundary_reasons.values()) != boundary_count
            or isinstance(max_internal, bool)
            or not isinstance(max_internal, int)
            or max_internal < 0
            or row.get("internal_ordered_action_expansion_ceiling") != 64
            or row.get("explicit_chance_probability_distribution_assumed") is not False
            or row.get("explicit_chance_always_stops_before_random_resolution") is not True
            or row.get("internal_boundary_has_action_or_child_authority") is not False
        ):
            raise R229GameError("decision lacks exact r252 internal-boundary telemetry")
        if (
            boundary_reasons.get("deterministic_internal_fanout_over_64", 0) > 0
            and max_internal <= 64
        ):
            raise R229GameError("oversized internal boundary did not exceed 64")
        recovery = row.get("lane_process_recovery")
        if not isinstance(recovery, Mapping):
            raise R229GameError("decision lacks serial process-lane telemetry")
        attempt_count = recovery.get("attempt_count")
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count not in (1, 2)
            or not isinstance(recovery.get("attempts"), list)
            or len(recovery["attempts"]) != attempt_count
        ):
            raise R229GameError("decision has malformed serial recovery attempts")
        if recovery.get("serial_lane_count") != 1:
            raise R229GameError("decision recovery telemetry is not serial")
        exhausted = recovery.get("exhausted_direct_fallback") is True
        if row.get("mode") == "bounded_lane_recovery_exhausted_direct_fallback":
            if (
                not exhausted
                or attempt_count != 2
                or row.get("mcts_action_authority") is not False
                or row.get("action_changed") is not False
                or row.get("meaningful_choice_change") not in (None, False)
            ):
                raise R229GameError("exhausted lane recovery gained MCTS authority")
            continue
        if exhausted:
            raise R229GameError("non-degraded decision claims exhausted lane recovery")
        if row.get("mode") != "shared_tree_mcts":
            continue
        chains = row.get("per_lane_search_id_chains") or ()
        handles = row.get("per_lane_handle_identities") or ()
        composite_states = (
            row.get("handle_scoped_first_search_id_composite_states") or ()
        )
        valid_chains = [
            chain
            for chain in chains
            if isinstance(chain, list)
            and chain
            and isinstance(chain[0], int)
            and not isinstance(chain[0], bool)
        ]
        valid_handles = [
            handle
            for handle in handles
            if isinstance(handle, (int, str)) and not isinstance(handle, bool)
        ]
        scoped_search_states = [
            (str(handles[index]), chains[index][0])
            for index in range(1)
        ] if len(valid_handles) == len(valid_chains) == 1 else []
        valid_composite_states = (
            (
                isinstance(composite_states, list)
                and len(composite_states) == 1
                and all(
                    isinstance(state, dict)
                    and list(state)
                    == ["lane_id", "handle_identity", "first_search_id"]
                    and state["lane_id"] == lane_id
                    and state["handle_identity"] == handles[lane_id]
                    and state["first_search_id"] == chains[lane_id][0]
                    for lane_id, state in enumerate(composite_states)
                )
            )
            if len(valid_handles) == len(valid_chains) == 1
            else False
        )
        microbatches = row.get("microbatch_sizes") or ()
        if (
            row.get("requested_simulator_lane_count") != 1
            or row.get("active_simulator_lane_count") != 1
            or row.get("arena_count") != 1
            or row.get("unique_handle_count") != 1
            or len(valid_handles) != 1
            or len(set(map(str, valid_handles))) != 1
            or row.get("search_begin_calls") != 1
            or row.get("search_release_calls", 0) < 1
            or row.get("search_end_calls") != 1
            or row.get("max_simulator_calls_in_flight") != 1
            or len(row.get("per_lane_depth") or ()) != 1
            or len(chains) != 1
            or len(valid_chains) != 1
            or len(set(scoped_search_states)) != 1
            or not valid_composite_states
            or not microbatches
            or any(size != 1 for size in microbatches)
            or row.get("outstanding_virtual_loss") != 0
        ):
            raise R229GameError("searched decision lacks an exact serial receipt")


def _legal(obs: Mapping[str, Any], action: Sequence[int]) -> None:
    selection = obs.get("select")
    if not isinstance(selection, Mapping):
        raise R229GameError("agent action has no selection payload")
    options = selection.get("option")
    if not isinstance(options, list):
        raise R229GameError("selection payload has no option list")
    chosen = [int(value) for value in action]
    lower = int(selection.get("minCount", 0) or 0)
    upper = min(int(selection.get("maxCount", 0) or 0), len(options))
    if not lower <= len(chosen) <= upper:
        raise R229GameError("agent returned an illegal selection count")
    if len(set(chosen)) != len(chosen) or any(value < 0 or value >= len(options) for value in chosen):
        raise R229GameError("agent returned a duplicate or out-of-range option")


def run_game(*, stage: Path, pair_index: int, game_index: int, mcts_seat: int, host: str, max_steps: int) -> dict[str, Any]:
    if mcts_seat not in (0, 1) or game_index not in (0, 1) or pair_index < 0:
        raise R229GameError("invalid game identity")
    if _sha256(stage / "model.pt") != CHECKPOINT or _sha256(stage / "matchup_tree.json") != MATCHUP_TREE:
        raise R229GameError("frozen r195 model or Matchup Adapter tree drifted")
    native_set, host_native_library = _verify_canonical_native_set(stage)
    module = _load(stage)
    from poke_bot.r228_async_shared_tree_queue import LANES

    if LANES != 1:
        raise R229GameError("r252 experimental arm is not exactly one serial lane")
    direct_module = module._direct()
    deck, model, _base_policy = direct_module._ensure_runtime()
    runtime = module._runtime()
    runtime_stock = dict(runtime.stock_library_receipt)
    if (
        runtime_stock.get("member") != host_native_library["path"]
        or runtime_stock.get("sha256") != host_native_library["sha256"]
    ):
        raise R229GameError("r228 runtime loaded a non-canonical platform library")
    runtime.reset_game()
    from poke_bot.agent import PolicyAgent
    from poke_bot import cg_env
    direct_policy = PolicyAgent(model=model, deck=list(deck), use_mcts=False)
    direct_policy.strict_runtime = True
    direct_policy.reset_game()
    if getattr(direct_policy, "use_recursive_turn_planner", False):
        raise R229GameError("direct control unexpectedly enabled RTP")
    if not getattr(direct_policy, "matchup_adapter_runtime", False):
        raise R229GameError("direct control lacks the frozen Matchup Adapter runtime")

    started_at_utc = _utc()
    started = time.monotonic()
    decisions_seen = mcts_seen = forced = oversized = direct_seen = setup_seen = 0
    mcts_latencies: list[float] = []
    direct_latencies: list[float] = []
    setup_latencies: list[float] = []
    try:
        observation, start_data = cg_env.battle_start(list(deck), list(deck))
        if observation is None:
            raise R229GameError(
                f"stock BattleStart failed: {getattr(start_data, 'errorType', None)}"
            )
        steps = 0
        first_player = None
        while not cg_env.is_finished(observation) and steps < max_steps:
            obs = observation
            decision_started = time.monotonic()
            current = obs.get("current") if isinstance(obs, Mapping) else None
            seat = current.get("yourIndex") if isinstance(current, Mapping) else None
            first = current.get("firstPlayer") if isinstance(current, Mapping) else None
            if first in (0, 1):
                first_player = int(first)
            turn_order = direct_module._turn_order_choice(obs)
            arm = (
                "setup"
                if turn_order is not None
                else "mcts" if seat == mcts_seat else "direct"
            )
            print(
                f"R229_GAME_STEP_BEGIN step={steps + 1} seat={seat} arm={arm}",
                flush=True,
            )
            if turn_order is not None:
                action = list(direct_module._fail_closed(obs, turn_order))
                setup_seen += 1
                setup_latencies.append(time.monotonic() - decision_started)
            elif seat == mcts_seat:
                from poke_bot import features
                decisions_seen += 1
                mcts_seen += 1
                receipt_count_before = len(runtime.decision_receipts)
                selection = obs.get("select") if isinstance(obs, Mapping) else None
                options = selection.get("option") if isinstance(selection, Mapping) else None
                option_count = len(options) if isinstance(options, list) else -1
                print(
                    "R229_GAME_MCTS_ENUM_BEGIN "
                    f"step={steps + 1} option_count={option_count} "
                    f"min_count={selection.get('minCount') if isinstance(selection, Mapping) else None} "
                    f"max_count={selection.get('maxCount') if isinstance(selection, Mapping) else None}",
                    flush=True,
                )
                try:
                    legal = features.enumerate_action_combos(obs)
                except features.ActionSpaceTooLarge:
                    legal = None
                    oversized += 1
                print(
                    "R229_GAME_MCTS_ENUM_READY "
                    f"step={steps + 1} legal_action_count="
                    f"{len(legal) if legal is not None else 'oversized'}",
                    flush=True,
                )
                if legal is not None and len(legal) <= 1:
                    forced += 1
                print(f"R229_GAME_MCTS_AGENT_BEGIN step={steps + 1}", flush=True)
                action = list(module.agent(obs))
                decision_latency = time.monotonic() - decision_started
                mcts_latencies.append(decision_latency)
                if len(runtime.decision_receipts) == receipt_count_before + 1:
                    selection = obs.get("select") if isinstance(obs, Mapping) else None
                    context = (
                        selection.get("context")
                        if isinstance(selection, Mapping)
                        else None
                    )
                    runtime.decision_receipts[-1].update(
                        {
                            "actor_seat": int(seat),
                            "wall_latency_seconds": decision_latency,
                            "selection_context": (
                                context.name if hasattr(context, "name") else str(context)
                            ),
                        }
                    )
            elif seat in (0, 1):
                decisions_seen += 1
                direct_seen += 1
                try:
                    action = list(direct_policy.trusted_search_or_greedy_select(dict(obs), search=False))
                except Exception:
                    action = list(direct_module._fail_closed(obs, []))
                direct_latencies.append(time.monotonic() - decision_started)
            else:
                raise R229GameError("engine emitted an invalid acting seat")
            print(f"R229_GAME_ACTION_READY step={steps + 1} arm={arm}", flush=True)
            _legal(obs, action)
            print(f"R229_GAME_SELECT_BEGIN step={steps + 1}", flush=True)
            observation = cg_env.battle_select(action)
            steps += 1
            print(f"R229_GAME_PROGRESS step={steps}", flush=True)
        if not cg_env.is_finished(observation):
            raise R229GameError("game exceeded the atomic-step ceiling")
        receipts = [dict(row) for row in runtime.decision_receipts]
        if len(receipts) != mcts_seen - forced - oversized:
            raise R229GameError("MCTS receipt count does not match eligible branching decisions")
        if any(row.get("completed_backups", 0) < 1 and row.get("mode") == "shared_tree_mcts" for row in receipts):
            raise R229GameError("searched decision lacks a completed backup")
        _validate_serial_receipts(receipts)
        changed = sum(bool(row.get("action_changed")) for row in receipts)
        meaningful = sum(bool(row.get("meaningful_choice_change")) for row in receipts)
        recovered_searches = sum(
            bool((row.get("lane_process_recovery") or {}).get("recovered_search"))
            for row in receipts
        )
        exhausted_recovery_fallbacks = sum(
            row.get("mode") == "bounded_lane_recovery_exhausted_direct_fallback"
            for row in receipts
        )
        lane_faults = sum(
            len(attempt.get("new_lane_faults") or ())
            for row in receipts
            for attempt in (row.get("lane_process_recovery") or {}).get("attempts", ())
            if isinstance(attempt, Mapping)
        )
        internal_value_boundaries = sum(
            int(row["internal_value_boundary_count"]) for row in receipts
        )
        decisions_with_internal_value_boundary = sum(
            int(row["internal_value_boundary_count"] > 0) for row in receipts
        )
        internal_boundary_reasons: dict[str, int] = {}
        for row in receipts:
            for reason, count in row["internal_value_boundary_reasons"].items():
                internal_boundary_reasons[reason] = (
                    internal_boundary_reasons.get(reason, 0) + int(count)
                )
        max_internal_ordered_action_count = max(
            (int(row["max_internal_ordered_action_count"]) for row in receipts),
            default=0,
        )
        elapsed = max(1e-9, time.monotonic() - started)
        completed_at_utc = _utc()
        return {
            "schema": SCHEMA,
            "status": "complete",
            "game_id": f"r229-pair-{pair_index:04d}-game-{game_index}",
            "pair_index": pair_index,
            "game_index": game_index,
            "mcts_seat": mcts_seat,
            "direct_seat": 1 - mcts_seat,
            "first_player_seat": first_player,
            "winner_seat": cg_env.result_winner(observation),
            "host": host,
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "checkpoint_sha256": CHECKPOINT,
            "matchup_tree_sha256": MATCHUP_TREE,
            "stock_library": dict(runtime.stock_library_receipt),
            "canonical_libcg_revision": 236,
            "mcts_topology_revision": 250,
            "search_id_identity_revision": 244,
            "process_lane_recovery_revision": 249,
            "serial_mcts_revision": 250,
            "internal_leaf_boundary_revision": 252,
            "simulator_lane_count": 1,
            "canonical_native_libraries": native_set,
            "elapsed_seconds": elapsed,
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "steps": steps,
            "decision_metrics": {
                "decisions_seen": decisions_seen,
                "mcts_seat_decisions_seen": mcts_seen,
                "direct_seat_decisions_seen": direct_seen,
                "setup_decisions": setup_seen,
                "mcts_eligible": mcts_seen - forced - oversized,
                "searched": sum(row.get("mode") == "shared_tree_mcts" for row in receipts),
                "forced": forced,
                "oversized_direct_fallback": oversized,
                "fallback": sum(row.get("mode") != "shared_tree_mcts" for row in receipts),
                "recovered_searches": recovered_searches,
                "exhausted_recovery_direct_fallbacks": exhausted_recovery_fallbacks,
                "contained_native_lane_faults": lane_faults,
                "internal_value_boundaries": internal_value_boundaries,
                "decisions_with_internal_value_boundary": (
                    decisions_with_internal_value_boundary
                ),
                "internal_explicit_chance_boundaries": internal_boundary_reasons.get(
                    "explicit_chance_pre_random", 0
                ),
                "internal_deterministic_fanout_boundaries": (
                    internal_boundary_reasons.get(
                        "deterministic_internal_fanout_over_64", 0
                    )
                ),
                "max_internal_ordered_action_count": (
                    max_internal_ordered_action_count
                ),
                "action_changed": changed,
                "meaningful_choice_change": meaningful,
            },
            "decision_latency_seconds": {
                "mcts_seat_all": mcts_latencies,
                "direct_r195_seat_all": direct_latencies,
                "deterministic_setup": setup_latencies,
            },
            "mcts_decisions": [
                {
                    **row,
                    "search_elapsed_seconds": row.get("elapsed_seconds"),
                }
                for row in receipts
            ],
            "lane_process_recovery_summary": dict(runtime._search.summary()),
            "training_eligible": False,
        }
    finally:
        try:
            cg_env.battle_finish()
        except Exception:
            pass
        runtime.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--game-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mcts-seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args(argv)
    print(
        f"R229_GAME_START pair={args.pair_index} game={args.game_index} "
        f"mcts_seat={args.mcts_seat}",
        flush=True,
    )
    try:
        result = run_game(stage=args.stage.resolve(), pair_index=args.pair_index, game_index=args.game_index, mcts_seat=args.mcts_seat, host=args.host, max_steps=args.max_steps)
    except Exception as exc:
        result = {
            "schema": SCHEMA, "status": "failed_closed",
            "game_id": f"r229-pair-{args.pair_index:04d}-game-{args.game_index}",
            "pair_index": args.pair_index, "game_index": args.game_index,
            "mcts_seat": args.mcts_seat, "host": args.host,
            "error": f"{type(exc).__name__}: {exc}", "training_eligible": False,
        }
        _atomic(args.output, result)
        raise
    _atomic(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
