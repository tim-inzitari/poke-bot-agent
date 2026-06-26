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
from poke_agent.device import describe_torch_device, torch_device
from poke_agent.paths import resolve_root
from poke_agent.self_play import (
    run_baseline_phase_loop,
    run_curriculum_self_play,
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
    parser.add_argument("--no-beam", action="store_true", help="policy-only during collection")
    parser.add_argument("--no-train", action="store_true", help="collect/eval only, skip training step")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel CABT workers for collect/eval (0 or omit = auto from CPU count)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="baseline phase vs official Kaggle agents, then transformer self-play at 60%% gate",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="train/eval vs official baseline agents only (no transformer self-play)",
    )
    parser.add_argument(
        "--submit-on-stop",
        action="store_true",
        help="submit the champion checkpoint to Kaggle when self-play stops",
    )
    parser.add_argument(
        "--submit-after-baseline",
        action="store_true",
        help="submit the champion to Kaggle when the baseline gate is beaten (curriculum only)",
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
    if args.no_beam:
        overrides["self_play_use_beam"] = False
    if args.matchup_mode is not None:
        overrides["self_play_matchup_mode"] = args.matchup_mode
    if args.workers is not None:
        overrides["self_play_workers"] = args.workers

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
    print("device", describe_torch_device(device))

    simulator = load_simulator(root)
    print_simulator_status(simulator)
    if not simulator.available:
        raise SystemExit("CABT simulator (cg-lib) is required for self-play")

    print("agent deck", deck_name, len(deck))
    if settings.use_field:
        print("field pool", len(settings.field_pool), "decks", f"mode={settings.matchup_mode}")
    else:
        print("field pool unavailable; using mirror matchups")

    initial_checkpoint = args.checkpoint or config["output_path"]
    if args.baseline_only:
        from poke_agent.baseline_agents import load_baseline_agents

        settings.baseline_agents = load_baseline_agents(
            root,
            baseline_dir=settings.baseline_dir or "baselines/official",
            cg_lib_path=simulator.lib_path,
        )
        print("baseline-only: official heuristic agents vs our transformer")
        reports = run_baseline_phase_loop(
            config=config,
            simulator=simulator,
            agent_deck=deck,
            agent_name=deck_name,
            settings=settings,
            device=device,
            initial_checkpoint=initial_checkpoint,
            root=root,
        )
    elif args.curriculum:
        print("curriculum: official baseline agents → transformer self-play at gate")
        reports = run_curriculum_self_play(
            config=config,
            simulator=simulator,
            agent_deck=deck,
            agent_name=deck_name,
            settings=settings,
            device=device,
            initial_checkpoint=initial_checkpoint,
            submit_on_stop=args.submit_on_stop,
            submit_after_baseline=args.submit_after_baseline,
            root=root,
        )
    else:
        reports = run_self_play_loop(
            config=config,
            simulator=simulator,
            agent_deck=deck,
            agent_name=deck_name,
            settings=settings,
            device=device,
            initial_checkpoint=initial_checkpoint,
            submit_on_stop=args.submit_on_stop,
            root=root,
        )

    print("\nSelf-play summary")
    print("-" * 17)
    for report in reports:
        phase = report.get("phase", "transformer")
        if phase == "baseline":
            agg = report.get("eval_vs_baselines_aggregate", {})
            print(
                f"baseline iter {report['iteration']}: rows={report['rows_collected']} "
                f"baseline_wr={agg.get('win_rate', 0):.1%} "
                f"checkpoint={Path(report['saved_checkpoint']).name}"
            )
        else:
            print(
                f"iter {report['iteration']}: rows={report['rows_collected']} "
                f"win_rate={report['eval_vs_random']['win_rate']:.1%} "
                f"checkpoint={Path(report['saved_checkpoint']).name}"
            )


if __name__ == "__main__":
    main()
