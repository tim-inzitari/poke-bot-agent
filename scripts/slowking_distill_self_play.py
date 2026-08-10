#!/usr/bin/env python3
"""Stage D — population self-play vs frozen opponents (research-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.slowking_distill.self_play import (  # noqa: E402
    run_population_self_play,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--games-per-opponent", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_population_self_play(
        actor_checkpoint=args.actor_checkpoint,
        output_dir=args.out_dir,
        games_per_opponent=int(args.games_per_opponent),
        max_turns=int(args.max_turns),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {
                "win_rate": result.win_rate,
                "mean_prize_delta": result.mean_prize_delta,
                "n_games": len(result.games),
                "receipt_path": result.receipt_path,
                "promoted": False,
                "research_only": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
