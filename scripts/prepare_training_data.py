#!/usr/bin/env python3
"""Refresh bootstrap training data: episodes-index → rollouts → multideck → merge → validate.

Top competition games come from the official Kaggle episodes-index dataset:
https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index

Examples:
  python scripts/prepare_training_data.py
  python scripts/prepare_training_data.py --no-download   # use local kaggle/input only
  python scripts/prepare_training_data.py --no-generate   # merge+validate only
  python scripts/prepare_training_data.py --top-percent 5 --episodes 3000
  python scripts/prepare_training_data.py --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.config import build_config
from poke_agent.data_pipeline import prepare_bootstrap_training_data, validate_bootstrap_training_data
from poke_agent.paths import resolve_root
from poke_agent.simulator import load_simulator, print_simulator_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--download",
        dest="download_index",
        action="store_true",
        default=True,
        help="download episodes-index manifest + latest daily bundle if missing (default: on)",
    )
    parser.add_argument("--no-download", dest="download_index", action="store_false")
    parser.add_argument(
        "--generate",
        dest="generate",
        action="store_true",
        default=True,
        help="generate multideck CABT rollouts when cg-lib is available (default: on)",
    )
    parser.add_argument("--no-generate", dest="generate", action="store_false")
    parser.add_argument(
        "--scrape",
        dest="scrape",
        action="store_true",
        default=False,
        help="also scrape live leaderboard via Kaggle API (off by default; use episodes-index instead)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="multideck CABT games (default: DATASET_GAMES)")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers for replay conversion / multideck gen / merge (default: auto from CPU)",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=None,
        help="top N%% episodes by replay score within each daily bundle (default: TOP_EPISODE_PERCENT)",
    )
    parser.add_argument("--validate-only", action="store_true", help="only validate existing merged JSONL")
    parser.add_argument("--no-validate", action="store_true", help="skip ladder/matchup validation after merge")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_root()
    config = build_config(root)

    if args.validate_only:
        merged = Path(config["merged_rollout_path"])
        validate_bootstrap_training_data(config, merged)
        print(f"validate-only OK: {merged}")
        return 0

    simulator = load_simulator(root)
    print_simulator_status(simulator)

    print("prepare bootstrap training data (episodes-index + multideck)")
    print(f"  merged -> {config['merged_rollout_path']}")
    print(
        f"  download_index={'on' if args.download_index else 'off'}  "
        f"generate={'on' if args.generate else 'off'}  "
        f"scrape={'on' if args.scrape else 'off'}"
    )

    paths = prepare_bootstrap_training_data(
        config,
        root,
        simulator_available=simulator.available,
        scrape=args.scrape,
        download_index=args.download_index,
        generate_multideck=args.generate,
        episodes=args.episodes,
        workers=args.workers,
        top_percent=args.top_percent,
        validate=not args.no_validate,
    )

    print(f"ready for bootstrap train: {paths.get('training_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
