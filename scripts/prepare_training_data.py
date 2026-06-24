#!/usr/bin/env python3
"""Refresh bootstrap training data: ladder scrape → rollouts → multideck → merge → validate.

Run before ``scripts/train_agent.py`` or let ``scripts/run_full_pipeline.sh`` call
this automatically as step 0.

Examples:
  python scripts/prepare_training_data.py
  python scripts/prepare_training_data.py --no-scrape --no-generate   # merge+validate only
  python scripts/prepare_training_data.py --teams 15 --episodes 3000
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
        "--scrape",
        dest="scrape",
        action="store_true",
        default=True,
        help="incremental Kaggle ladder scrape (default: on)",
    )
    parser.add_argument("--no-scrape", dest="scrape", action="store_false")
    parser.add_argument(
        "--generate",
        dest="generate",
        action="store_true",
        default=True,
        help="generate multideck CABT rollouts when cg-lib is available (default: on)",
    )
    parser.add_argument("--no-generate", dest="generate", action="store_false")
    parser.add_argument("--teams", type=int, default=10, help="top N leaderboard teams to scrape")
    parser.add_argument(
        "--episodes-per-sub",
        type=int,
        default=20,
        help="recent episodes per submission when scraping",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="multideck CABT games to generate (default: DATASET_GAMES from config)",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=None,
        help="keep top N%% ladder episodes by score (default: TOP_EPISODE_PERCENT)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="skip scrape/generate/merge; only validate existing merged JSONL",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip ladder/matchup validation after merge",
    )
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

    print("prepare bootstrap training data")
    print(f"  merged -> {config['merged_rollout_path']}")
    print(f"  scrape={'on' if args.scrape else 'off'}  generate={'on' if args.generate else 'off'}")

    paths = prepare_bootstrap_training_data(
        config,
        root,
        simulator_available=simulator.available,
        scrape=args.scrape,
        scrape_teams=args.teams,
        scrape_episodes_per_sub=args.episodes_per_sub,
        generate_multideck=args.generate,
        episodes=args.episodes,
        top_percent=args.top_percent,
        validate=not args.no_validate,
    )

    train_path = paths.get("training_path")
    print(f"ready for bootstrap train: {train_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
