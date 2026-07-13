#!/usr/bin/env python
"""Eval pure-Dragapult MCTS/greedy agent vs ALL baselines in the manifest.

Seat-swap, Wilson lower bounds, mirror/non-mirror split, MCTS-on vs greedy
ablation. Progress: outer tqdm over opponents, inner over games.

Pragmatic default N: ``--games-per-opp 8`` (= 4 seat-swapped pairs) so 29
opponents × 8 ≈ 232 games saturate 28 workers without blocking forever.
Raise to ≥100 for a formal gate pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from poke_bot import config, deck_pool, paths
from poke_bot.agent import PolicyAgent, play_game
from poke_bot.baselines_runtime import ensure_baselines_installed, load_baseline_agent, load_manifest
from poke_bot.device import leaf_eval_device
from poke_bot.eval_metrics import FieldReport
from poke_bot.train import load_model_from_checkpoint
from poke_bot.worker_pool import WorkerPool


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--games-per-opp", type=int, default=8, help="Must be even (seat-swap pairs).")
    p.add_argument("--workers", type=int, default=config.HARDWARE.sim_workers)
    p.add_argument("--mcts-sims", type=int, default=32)
    p.add_argument("--greedy-ablation", action="store_true", help="Also run greedy-only half.")
    p.add_argument("--gate", type=float, default=0.55)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--only", nargs="+", help="Subset of baseline ids")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _game_job(payload: dict) -> dict:
    """Worker entry: play one game (model loaded per worker — heavy but correct)."""
    import random

    from poke_bot.agent import PolicyAgent, play_game
    from poke_bot.baselines_runtime import BaselineSpec, load_baseline_agent
    from poke_bot.train import load_model_from_checkpoint

    ckpt = payload["checkpoint"]
    device = payload.get("device", "cpu")
    model = load_model_from_checkpoint(ckpt, device=__import__("torch").device(device))
    our_deck = payload["our_deck"]
    spec = BaselineSpec(**payload["spec"])
    opp_fn, opp_deck = load_baseline_agent(spec)
    our_seat = int(payload["our_seat"])
    use_mcts = bool(payload["use_mcts"])
    sims = int(payload["mcts_sims"])
    rng = random.Random(int(payload["seed"]))

    agent = PolicyAgent(
        model=model,
        deck=our_deck,
        use_mcts=use_mcts,
        max_sims=sims,
        rng=rng,
    )
    agent.reset_game()

    if our_seat == 0:
        result = play_game(agent, opp_fn, our_deck, opp_deck)
    else:
        result = play_game(opp_fn, agent, opp_deck, our_deck)

    is_mirror = sorted(our_deck) == sorted(opp_deck)
    return {
        "opponent_id": spec.id,
        "our_seat": our_seat,
        "winner": result["winner"],
        "steps": result["steps"],
        "is_mirror": is_mirror,
        "mcts_on": use_mcts,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    if args.games_per_opp % 2 != 0:
        print("ERROR: --games-per-opp must be even (seat-swap)", file=sys.stderr)
        return 2
    if not args.checkpoint.is_file():
        print(f"ERROR: missing checkpoint {args.checkpoint}", file=sys.stderr)
        return 2

    specs = ensure_baselines_installed(load_manifest())
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.id in wanted]
    our_deck = deck_pool.primary_deck()
    device = str(leaf_eval_device(prefer_name=config.HARDWARE.leaf_gpu_name))

    pairs = args.games_per_opp // 2
    jobs: list[dict] = []
    seed = args.seed
    for spec in specs:
        for pair_i in range(pairs):
            for our_seat in (0, 1):
                jobs.append(
                    {
                        "checkpoint": str(args.checkpoint),
                        "our_deck": our_deck,
                        "spec": {
                            "id": spec.id,
                            "name": spec.name,
                            "dir_name": spec.dir_name,
                            "group": spec.group,
                            "source": spec.source,
                            "path": str(spec.path),
                        },
                        "our_seat": our_seat,
                        "use_mcts": True,
                        "mcts_sims": args.mcts_sims,
                        "seed": seed,
                        "device": device,
                    }
                )
                seed += 1
                if args.greedy_ablation and pair_i == 0 and our_seat == 0:
                    # One greedy game per opp for ablation signal (cheap).
                    jobs.append(
                        {
                            "checkpoint": str(args.checkpoint),
                            "our_deck": our_deck,
                            "spec": {
                                "id": spec.id,
                                "name": spec.name,
                                "dir_name": spec.dir_name,
                                "group": spec.group,
                                "source": spec.source,
                                "path": str(spec.path),
                            },
                            "our_seat": 0,
                            "use_mcts": False,
                            "mcts_sims": 0,
                            "seed": seed,
                            "device": device,
                        }
                    )
                    seed += 1

    print(
        f"== eval_vs_baselines opps={len(specs)} jobs={len(jobs)} "
        f"games_per_opp={args.games_per_opp} workers={args.workers} "
        f"sims={args.mcts_sims} device={device}",
        flush=True,
    )
    print(
        f"   N note: {args.games_per_opp} games/opp (seat-swapped) × {len(specs)} "
        f"opponents = {args.games_per_opp * len(specs)} MCTS games "
        f"(+ greedy ablation extras if enabled).",
        flush=True,
    )

    report = FieldReport(gate_threshold=args.gate)
    # Group jobs by opponent for nested progress UX.
    by_opp: dict[str, list[dict]] = {}
    for j in jobs:
        by_opp.setdefault(j["spec"]["id"], []).append(j)

    # Serial-per-batch via WorkerPool for all jobs with a flat tqdm is simpler
    # and still shows live WR; nested bars over pool results are awkward.
    results: list[dict] = []
    with WorkerPool(num_workers=args.workers) as pool:
        bar = tqdm(total=len(jobs), desc="eval games", unit="game")
        wins = 0.0
        n = 0
        for res in pool.imap_unordered(_game_job, jobs):
            results.append(res)
            report.merge_game(
                res["opponent_id"],
                our_seat=res["our_seat"],
                winner=res["winner"],
                is_mirror=res["is_mirror"],
                mcts_on=res["mcts_on"],
            )
            n += 1
            if res["winner"] == 2:
                wins += 0.5
            elif res["winner"] == res["our_seat"]:
                wins += 1.0
            bar.update(1)
            bar.set_postfix(wr=f"{wins / max(n, 1):.1%}", n=n, pass_=len(report.opponents_passing()))
        bar.close()

    summary = report.summary()
    out = args.out or (paths.OUTPUTS_DIR / "eval" / "vs_baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f">> wrote {out}", flush=True)
    print(
        f">> evaluated={summary['n_evaluated']} passing_wilson>={args.gate}: "
        f"{summary['n_passing_wilson']} all_pass={summary['all_pass']}",
        flush=True,
    )
    for row in summary["matchups"]:
        print(
            f"   {row['opponent_id']:32} games={row['games']:3} "
            f"wr={row['wr']:.1%} wilson_lo={row['wilson_lo']:.1%}",
            flush=True,
        )
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
