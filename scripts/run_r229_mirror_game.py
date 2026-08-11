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
    manifest_path = stage / "r239_fleet_evaluation_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise R229GameError("sealed package lacks a readable r239 manifest") from exc
    expected_manifest_libraries = {
        name: {"path": relative, "sha256": digest, "size_bytes": size}
        for name, (relative, digest, size) in CANONICAL_NATIVE_LIBRARIES.items()
    }
    if (
        manifest.get("schema")
        != "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r239_package/v1"
        or manifest.get("owner_goal_revision") != 239
        or manifest.get("canonical_libcg_revision") != 236
        or manifest.get("owner_two_lane_topology_revision") != 239
        or manifest.get("simulator_lane_count") != 2
        or manifest.get("internal_agent_start_arena_count") != 2
        or manifest.get("distinct_search_begin_id_count") != 2
        or manifest.get("search_begin_identity_scope")
        != "arena_handle_plus_handle_local_search_id"
        or manifest.get("raw_search_id_global_uniqueness_required") is not False
        or manifest.get("logical_frontier_leaf_count_per_frozen_model_batch") != 2
        or manifest.get("partial_frontier_batches_allowed") is not False
        or manifest.get("serial_one_lane_continuation_allowed") is not False
        or manifest.get("one_shared_logical_mcts_tree_required") is not True
        or manifest.get("bo_lifecycle_revision") != 233
        or manifest.get("r234_kaggle_broker_or_queue_lifecycle_included") is not False
        or manifest.get("canonical_native_libraries") != expected_manifest_libraries
    ):
        raise R229GameError("sealed r239 package manifest identity drifted")
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


def _validate_two_lane_receipts(receipts: Sequence[Mapping[str, Any]]) -> None:
    for row in receipts:
        if row.get("mode") != "shared_tree_mcts":
            continue
        chains = row.get("per_lane_search_id_chains") or ()
        handles = row.get("per_lane_handle_identities") or ()
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
            for index in range(2)
        ] if len(valid_handles) == len(valid_chains) == 2 else []
        microbatches = row.get("microbatch_sizes") or ()
        if (
            row.get("requested_simulator_lane_count") != 2
            or row.get("active_simulator_lane_count") != 2
            or row.get("arena_count") != 2
            or row.get("unique_handle_count") != 2
            or len(valid_handles) != 2
            or len(set(map(str, valid_handles))) != 2
            or row.get("search_begin_calls") != 2
            or row.get("search_release_calls", 0) < 2
            or row.get("search_end_calls") != 2
            or row.get("max_simulator_calls_in_flight") != 2
            or len(row.get("per_lane_depth") or ()) != 2
            or len(chains) != 2
            or len(valid_chains) != 2
            or len(set(scoped_search_states)) != 2
            or not microbatches
            or any(size != 2 for size in microbatches)
            or row.get("outstanding_virtual_loss") != 0
        ):
            raise R229GameError("searched decision lacks an exact two-lane receipt")


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

    if LANES != 2:
        raise R229GameError("r239 experimental arm is not exactly two lanes")
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
            if turn_order is not None:
                action = list(direct_module._fail_closed(obs, turn_order))
                setup_seen += 1
                setup_latencies.append(time.monotonic() - decision_started)
            elif seat == mcts_seat:
                from poke_bot import features
                decisions_seen += 1
                mcts_seen += 1
                receipt_count_before = len(runtime.decision_receipts)
                try:
                    legal = features.enumerate_action_combos(obs)
                except features.ActionSpaceTooLarge:
                    legal = None
                    oversized += 1
                if legal is not None and len(legal) <= 1:
                    forced += 1
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
            _legal(obs, action)
            observation = cg_env.battle_select(action)
            steps += 1
        if not cg_env.is_finished(observation):
            raise R229GameError("game exceeded the atomic-step ceiling")
        receipts = [dict(row) for row in runtime.decision_receipts]
        if len(receipts) != mcts_seen - forced - oversized:
            raise R229GameError("MCTS receipt count does not match eligible branching decisions")
        if any(row.get("completed_backups", 0) < 1 and row.get("mode") == "shared_tree_mcts" for row in receipts):
            raise R229GameError("searched decision lacks a completed backup")
        _validate_two_lane_receipts(receipts)
        changed = sum(bool(row.get("action_changed")) for row in receipts)
        meaningful = sum(bool(row.get("meaningful_choice_change")) for row in receipts)
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
            "mcts_topology_revision": 239,
            "simulator_lane_count": 2,
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
