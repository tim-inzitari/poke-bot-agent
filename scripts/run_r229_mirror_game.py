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


def _load(stage: Path) -> Any:
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
    module = _load(stage)
    direct_module = module._direct()
    deck, model, _base_policy = direct_module._ensure_runtime()
    runtime = module._runtime()
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
