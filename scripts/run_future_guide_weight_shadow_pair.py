#!/usr/bin/env python3
"""Run one future-only guide-on/guide-off shadow study end to end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.guide_weight_shadow_pair import (  # noqa: E402
    finalize,
    prepare_manifest,
    run_evaluation,
    run_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--shadow-device", required=True)
    parser.add_argument("--production-device", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--game-timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    manifest = prepare_manifest(
        request_path=args.request,
        output_dir=args.output_dir,
        shadow_device=args.shadow_device,
        production_device=args.production_device,
        baseline_manifest=args.baseline_manifest,
    )
    training = run_training(manifest, device=args.shadow_device)
    rows = run_evaluation(
        manifest,
        training,
        workers=args.workers,
        game_timeout_seconds=args.game_timeout_seconds,
    )
    evidence, schedule = finalize(manifest, training, rows)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(manifest),
                "training_receipt": str(training),
                "evaluation_rows": str(rows),
                "evidence": str(evidence),
                "schedule": str(schedule),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
