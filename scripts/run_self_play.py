#!/usr/bin/env python3
"""Run the AlphaGo-style self-play loop: collect → train → evaluate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.config import build_config
from poke_agent.deck import read_deck
from poke_agent.device import print_device_summary, torch_device
from poke_agent.paths import resolve_root
from poke_agent.self_play import (
    run_self_play_loop,
    self_play_settings_from_config,
)
from poke_agent.simulator import load_simulator, print_simulator_status


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaGo-style self-play: collect games, train, repeat.")
    parser.add_argument("--iterations", type=int, default=None, help="outer loop iterations")
    parser.add_argument("--games", type=int, default=None, help="CABT games per collect→train iteration")
    parser.add_argument("--eval-games", type=int, default=None, help="eval games vs random after each iteration")
    parser.add_argument("--train-epochs", type=int, default=None, help="transformer epochs per self-play retrain")
    parser.add_argument("--batch-games", type=int, default=None, help="games per temporal training batch")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="epochs without improvement before stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=None, help="minimum loss improvement for early stop")
    parser.add_argument("--target-win-rate", type=float, default=None, help="stop once eval winrate reaches this")
    parser.add_argument("--plateau-patience", type=int, default=None, help="stop after this many no-improvement iterations")
    parser.add_argument("--no-beam", action="store_true", help="policy-only during collection")
    parser.add_argument("--no-train", action="store_true", help="collect/eval only, skip training step")
    parser.add_argument(
        "--baselines",
        default=None,
        help="comma-separated public baseline names, or 'public' for the full accessible suite",
    )
    parser.add_argument("--no-baselines", action="store_true", help="disable public-baseline opponents")
    parser.add_argument("--checkpoint", type=Path, default=None, help="initial checkpoint path")
    parser.add_argument(
        "--field-deck-dir",
        type=Path,
        default=None,
        help="directory of opponent meta decks (default: decks/competitive/high_performing)",
    )
    parser.add_argument(
        "--matchup-mode",
        choices=["sample", "round-robin"],
        default=None,
        help="how to pick opponent decks from the field",
    )
    args = parser.parse_args()

    root = resolve_root()
    overrides: dict[str, object] = {}
    if args.iterations is not None:
        overrides["self_play_iterations"] = args.iterations
    if args.games is not None:
        overrides["self_play_games"] = args.games
    if args.eval_games is not None:
        overrides["self_play_eval_games"] = args.eval_games
    if args.train_epochs is not None:
        overrides["self_play_train_epochs"] = args.train_epochs
    if args.batch_games is not None:
        overrides["batch_games"] = args.batch_games
    if args.early_stop_patience is not None:
        overrides["early_stop_patience"] = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        overrides["early_stop_min_delta"] = args.early_stop_min_delta
    if args.target_win_rate is not None:
        overrides["self_play_target_win_rate"] = args.target_win_rate
    if args.plateau_patience is not None:
        overrides["self_play_plateau_patience"] = args.plateau_patience
    if args.no_beam:
        overrides["self_play_use_beam"] = False
    if args.matchup_mode is not None:
        overrides["self_play_matchup_mode"] = args.matchup_mode
    if args.baselines is not None:
        overrides["self_play_baselines"] = args.baselines
        overrides["self_play_train_vs_baselines"] = True
    if args.no_baselines:
        overrides["self_play_baselines"] = ""
        overrides["self_play_train_vs_baselines"] = False

    config = build_config(root, overrides=overrides or None)

    deck, deck_source = read_deck(config, root)
    deck_name = deck_source.stem if deck_source.suffix else str(deck_source)

    settings = self_play_settings_from_config(
        config,
        root,
        agent_name=deck_name,
        agent_deck=deck,
    )
    if args.no_train:
        settings.train_after_collect = False

    if args.field_deck_dir is not None:
        from poke_agent.deck_pool import resolve_field_pool

        settings.field_deck_dir = str(args.field_deck_dir)
        settings.field_pool, settings.use_field = resolve_field_pool(
            settings.field_deck_dir,
            root=root,
            agent_name=deck_name,
            agent_deck=deck,
        )

    device = torch_device()
    print("device", device)
    print_device_summary()

    simulator = load_simulator(root)
    print_simulator_status(simulator)
    if not simulator.available:
        raise SystemExit("CABT simulator (cg-lib) is required for self-play")

    print("agent deck", deck_name, len(deck))
    if settings.train_vs_baselines:
        print("baseline pool", len(settings.baseline_opponents), "agents")
    elif settings.use_field:
        print("field pool", len(settings.field_pool), "decks", f"mode={settings.matchup_mode}")
    else:
        print("field pool unavailable; using mirror matchups")

    reports = run_self_play_loop(
        config=config,
        simulator=simulator,
        agent_deck=deck,
        agent_name=deck_name,
        settings=settings,
        device=device,
        initial_checkpoint=args.checkpoint,
    )

    print("\nSelf-play summary")
    print("-" * 17)
    for report in reports:
        print(
            f"iter {report['iteration']}: rows={report['rows_collected']} "
            f"win_rate={report['eval_vs_random']['win_rate']:.1%} "
            f"checkpoint={Path(report['saved_checkpoint']).name}"
        )


if __name__ == "__main__":
    main()
