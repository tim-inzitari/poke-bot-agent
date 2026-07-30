#!/usr/bin/env python3
"""Bounded outcome-stratified Slop Box guide-adherence audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from poke_bot import features
from poke_bot.pure_rl.shards import iter_shard_games
from poke_bot.teal_mask_ogerpon_heuristics import guide_scores


CONTEXT_NAMES = {
    0: "main",
    1: "setup_active",
    2: "setup_bench",
}


def _bucket() -> dict[str, int]:
    return {"labeled": 0, "agreed": 0, "disagreed": 0}


def _finish(rows: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, values in sorted(rows.items()):
        labeled = int(values["labeled"])
        result[name] = {
            **values,
            "agreement_rate": (
                float(values["agreed"]) / labeled if labeled else None
            ),
        }
    return result


def audit(path: Path, *, max_games: int) -> dict[str, Any]:
    by_context: dict[str, dict[str, int]] = defaultdict(_bucket)
    by_outcome: dict[str, dict[str, int]] = defaultdict(_bucket)
    by_opponent: dict[str, dict[str, int]] = defaultdict(_bucket)
    games = decisions = stages = labeled = agreed = 0
    selected_index_mismatches = 0
    setup_bench_prompts = setup_bench_declines = 0

    for game in iter_shard_games(path):
        if games >= max_games:
            break
        games += 1
        outcome = "win" if game.value > 0 else ("loss" if game.value < 0 else "draw")
        for decision in game.decisions:
            decisions += 1
            obs = decision.observation
            action = list(decision.action)
            stage_defs = features.factorized_teacher_forcing_stages(obs, action)
            if stage_defs and int(stage_defs[0][1]) != int(decision.selected_index):
                selected_index_mismatches += 1
            context = int((obs.get("select") or {}).get("context", -1))
            context_name = CONTEXT_NAMES.get(context, f"context_{context}")
            for combos, target_index in stage_defs:
                stages += 1
                normalized = [list(combo) for combo in combos]
                scores = guide_scores(
                    obs,
                    normalized,
                    deck=game.deck,
                    force_enabled=True,
                )
                if scores is None:
                    continue
                best = max(range(len(scores)), key=scores.__getitem__)
                is_agreement = int(target_index) == int(best)
                labeled += 1
                agreed += int(is_agreement)
                for group, name in (
                    (by_context, context_name),
                    (by_outcome, outcome),
                    (by_opponent, game.opp_archetype),
                ):
                    group[name]["labeled"] += 1
                    group[name]["agreed" if is_agreement else "disagreed"] += 1

                if context == 2 and normalized[best]:
                    setup_bench_prompts += 1
                    if not normalized[int(target_index)]:
                        setup_bench_declines += 1

    return {
        "schema": "poke_bot.teal_guide_adherence_audit/v1",
        "source_shard": str(path),
        "bounded_max_games": int(max_games),
        "games": games,
        "decisions": decisions,
        "policy_stages": stages,
        "guide_labeled_stages": labeled,
        "guide_agreed_stages": agreed,
        "guide_disagreed_stages": labeled - agreed,
        "guide_agreement_rate": float(agreed) / labeled if labeled else None,
        "selected_index_derivation_mismatches": selected_index_mismatches,
        "setup_bench_prompts_with_nonempty_guide_preference": setup_bench_prompts,
        "setup_bench_declines": setup_bench_declines,
        "setup_bench_decline_rate": (
            float(setup_bench_declines) / setup_bench_prompts
            if setup_bench_prompts
            else None
        ),
        "by_context": _finish(by_context),
        "by_outcome": _finish(by_outcome),
        "by_opponent": _finish(by_opponent),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_games <= 0:
        raise SystemExit("--max-games must be positive")
    result = audit(args.shard, max_games=args.max_games)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
