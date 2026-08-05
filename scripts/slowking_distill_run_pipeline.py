#!/usr/bin/env python3
"""Run Slowking distill stages A→E + eval gate (research-only, never promotes)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.slowking_distill.bc_stage import StageAConfig  # noqa: E402
from poke_bot.slowking_distill.eval_gate import EvalGateConfig  # noqa: E402
from poke_bot.slowking_distill.iql import IQLConfig  # noqa: E402
from poke_bot.slowking_distill.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-date", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--max-critical-search", type=int, default=0)
    parser.add_argument(
        "--eval-games-json",
        type=Path,
        default=None,
        help="Optional JSON list of paired training-ineligible game results",
    )
    parser.add_argument("--min-eval-games", type=int, default=64)
    parser.add_argument("--min-win-rate", type=float, default=0.55)
    args = parser.parse_args()

    eval_games = []
    if args.eval_games_json is not None:
        eval_games = json.loads(args.eval_games_json.read_text(encoding="utf-8"))
        if not isinstance(eval_games, list):
            raise SystemExit("eval-games-json must be a list")

    summary = run_pipeline(
        args.decisions_jsonl,
        config=PipelineConfig(
            out_dir=args.out_dir,
            val_dates=list(args.val_date),
            stage_a=StageAConfig(epochs=int(args.epochs)),
            stage_b=IQLConfig(epochs=int(args.epochs)),
            run_search=not args.skip_search,
            run_distill=not args.skip_distill,
            max_critical_search=int(args.max_critical_search),
            eval_games=eval_games,
            eval_gate=EvalGateConfig(
                min_games=int(args.min_eval_games),
                min_win_rate=float(args.min_win_rate),
            ),
        ),
    )
    print(json.dumps({"promoted": summary["promoted"], "eval_gate": summary["eval_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
