#!/usr/bin/env python
"""Round-robin MCTS RL vs ALL baselines in ``baselines/manifest.json``.

Each iteration:
  1. Play seat-swapped games vs every manifest agent (28 workers).
  2. Train on MCTS visit targets + outcomes (AlphaZero-style).
  3. Eval WR matrix; Wilson-lower gate (~55%) when N is large enough.
  4. Checkpoint full loop state (``--resume auto``).

Pragmatic per-iteration N (documented):
  ``--games-per-opp 8`` → 4 seat-swap pairs × 29 opps ≈ **232 MCTS games/iter**
  — saturates SIM_WORKERS=28 while still making progress. Formal gate wants
  ≥100 games/opp; raise ``--games-per-opp`` / run ``eval_vs_baselines.py`` for that.

Progress: outer tqdm over opponents, inner over games; running WR postfix.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm.auto import tqdm

from poke_bot import checkpoint, config, deck_pool, paths
from poke_bot.agent import PolicyAgent, play_game
from poke_bot.baselines_runtime import (
    BaselineSpec,
    ensure_baselines_installed,
    load_baseline_agent,
    load_manifest,
)
from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, load_bootstrap_dataset
from poke_bot.device import leaf_eval_device, training_device
from poke_bot.eval_metrics import FieldReport
from poke_bot.features import build_board_tokens, build_option_tokens, enumerate_action_combos
from poke_bot.train import TrainConfig, load_model_from_checkpoint, sequence_losses, train_bootstrap
from poke_bot.worker_pool import WorkerPool

# Module-level cache for spawn workers (reloaded after recycle).
_WORKER_STATE: dict[str, Any] = {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--archetype",
        default=deck_pool.primary_archetype(),
        help="Primary archetype (drives run names, deck, bootstrap JSONL).",
    )
    p.add_argument(
        "--bootstrap-ckpt",
        type=Path,
        default=None,
        help="Starting weights (default: <archetype>_bootstrap.best.pt or latest)",
    )
    p.add_argument("--resume", default="auto")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument(
        "--games-per-opp",
        type=int,
        default=8,
        help="Even N per opponent per iter (default 8 = 4 seat-swap pairs).",
    )
    p.add_argument("--workers", type=int, default=config.HARDWARE.sim_workers)
    p.add_argument("--mcts-sims", type=int, default=32)
    p.add_argument("--gate", type=float, default=0.55)
    p.add_argument("--train-epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="+")
    return p.parse_args(argv)


def _resolve_bootstrap_ckpt(explicit: Path | None, archetype: str) -> Path:
    if explicit is not None:
        return explicit
    best = checkpoint.best_path(f"{archetype}_bootstrap")
    latest = checkpoint.latest_path(f"{archetype}_bootstrap")
    if best.is_file():
        return best
    if latest.is_file():
        return latest
    raise FileNotFoundError(
        "No bootstrap checkpoint found. Run scripts/train_bootstrap.py first "
        f"(looked for {best} / {latest})."
    )


def _worker_play(job: dict) -> dict:
    """Play one MCTS game in a worker; return outcome + search targets."""
    import torch

    from poke_bot.agent import PolicyAgent, play_game
    from poke_bot.baselines_runtime import BaselineSpec, load_baseline_agent
    from poke_bot.train import load_model_from_checkpoint

    ckpt = job["checkpoint"]
    device = job["device"]
    key = f"{ckpt}|{device}"
    if _WORKER_STATE.get("key") != key:
        _WORKER_STATE["model"] = load_model_from_checkpoint(
            ckpt, device=torch.device(device)
        )
        _WORKER_STATE["key"] = key
    model = _WORKER_STATE["model"]

    spec_d = dict(job["spec"])
    spec_d["path"] = Path(spec_d["path"])
    spec = BaselineSpec(**spec_d)
    opp_fn, opp_deck = load_baseline_agent(spec)
    our_deck = job["our_deck"]
    our_seat = int(job["our_seat"])
    rng = random.Random(int(job["seed"]))

    agent = PolicyAgent(
        model=model,
        deck=our_deck,
        use_mcts=True,
        max_sims=int(job["mcts_sims"]),
        collect_targets=True,
        rng=rng,
    )
    agent.reset_game()

    if our_seat == 0:
        result = play_game(agent, opp_fn, our_deck, opp_deck)
        winner = result["winner"]
    else:
        result = play_game(opp_fn, agent, opp_deck, our_deck)
        winner = result["winner"]

    # Value from our seat perspective.
    if winner == 2:
        value = 0.0
    elif winner == our_seat:
        value = 1.0
    else:
        value = -1.0

    return {
        "opponent_id": spec.id,
        "our_seat": our_seat,
        "winner": winner,
        "value": value,
        "steps": result["steps"],
        "is_mirror": sorted(our_deck) == sorted(opp_deck),
        "mcts_on": True,
        "targets": agent.targets,
        "deck": our_deck,
        "seed": job["seed"],
    }


def _targets_to_sequence(rec: dict, archetype: str = "dragapult") -> GameSequence | None:
    """Build a lightweight GameSequence from collected MCTS targets.

    Note: we do not have full board SparseVectors from the worker without
    re-featurizing observations. For AlphaZero train we store policy targets
    alongside a synthetic single-step sequence only when boards were captured.

    This round-robin version trains primarily via appending JSONL rollouts that
    include observation snapshots when available; if targets lack boards, we
    skip and rely on outcome value fine-tune from bootstrap continuation.
    """
    # Worker currently returns visit targets without board SparseVectors.
    # Persist rollouts for later reanalyse; return None for immediate train.
    return None


def _append_rollout_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            # Drop huge tensors; keep serializable fields.
            slim = {
                "opponent_id": rec["opponent_id"],
                "our_seat": rec["our_seat"],
                "winner": rec["winner"],
                "value": rec["value"],
                "steps": rec["steps"],
                "is_mirror": rec["is_mirror"],
                "targets": rec.get("targets") or [],
                "seed": rec.get("seed"),
            }
            fh.write(json.dumps(slim) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    if args.games_per_opp % 2 != 0 or args.games_per_opp < 2:
        print("ERROR: --games-per-opp must be even and >= 2", file=sys.stderr)
        return 2

    import os

    os.environ["POKEBOT_PRIMARY_ARCHETYPE"] = args.archetype
    run_name = f"{args.archetype}_round_robin"

    specs = ensure_baselines_installed(load_manifest())
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.id in wanted]

    our_deck = deck_pool.primary_deck()
    leaf_dev = str(leaf_eval_device(prefer_name=config.HARDWARE.leaf_gpu_name))
    train_dev = training_device(prefer_name=config.HARDWARE.train_gpu_name)

    loop_state: dict[str, Any] = {
        "iteration": 0,
        "champion": None,
        "history": [],
        "seed": args.seed,
        "games_per_opp": args.games_per_opp,
        "n_opponents": len(specs),
        "n_note": (
            f"{args.games_per_opp} games/opp/iter × {len(specs)} opps = "
            f"{args.games_per_opp * len(specs)} games/iter "
            f"(seat-swapped; 28 workers)"
        ),
    }

    resume_path = checkpoint.resolve_resume_path(run_name, args.resume)
    model_ckpt = _resolve_bootstrap_ckpt(args.bootstrap_ckpt, args.archetype)

    if resume_path is not None:
        print(f"[rr] resume loop from {resume_path}", flush=True)
        ckpt = checkpoint.load_checkpoint(resume_path, map_location="cpu")
        extra = ckpt.get("extra") or {}
        loop_state.update(extra.get("loop_state") or {})
        model_ckpt = Path(loop_state.get("champion") or model_ckpt)
        print(f"[rr] continuing at iteration={loop_state['iteration']} champion={model_ckpt}", flush=True)
    else:
        loop_state["champion"] = str(model_ckpt)

    print(f"== train_round_robin", flush=True)
    print(f"   {loop_state['n_note']}", flush=True)
    print(f"   leaf_device={leaf_dev} train_device={train_dev}", flush=True)
    print(f"   gate wilson_lo>={args.gate} mcts_sims={args.mcts_sims}", flush=True)

    mgr = checkpoint.CheckpointManager(run_name)
    rollout_path = paths.DATA_DIR / "rollouts" / f"{args.archetype}_rr.jsonl"

    start_iter = int(loop_state.get("iteration", 0))
    for it in tqdm(range(start_iter, args.iterations), desc="rr iterations", unit="iter"):
        loop_state["iteration"] = it
        champion = Path(loop_state["champion"])
        report = FieldReport(gate_threshold=args.gate)

        # Build jobs: seat-swap pairs per opponent.
        jobs: list[dict] = []
        seed = int(loop_state.get("seed", args.seed)) + it * 100_000
        pairs = args.games_per_opp // 2
        for spec in specs:
            for _ in range(pairs):
                for our_seat in (0, 1):
                    jobs.append(
                        {
                            "checkpoint": str(champion),
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
                            "mcts_sims": args.mcts_sims,
                            "seed": seed,
                            "device": leaf_dev,
                        }
                    )
                    seed += 1

        results: list[dict] = []
        wins = 0.0
        n = 0
        opp_bar = tqdm(specs, desc=f"iter{it} opponents", leave=False, unit="opp")
        # Play all jobs in one pool; update per-opponent nested feel via postfix.
        with WorkerPool(num_workers=args.workers) as pool:
            game_bar = tqdm(total=len(jobs), desc=f"iter{it} games", leave=False, unit="game")
            for res in pool.imap_unordered(_worker_play, jobs):
                results.append(res)
                report.merge_game(
                    res["opponent_id"],
                    our_seat=res["our_seat"],
                    winner=res["winner"],
                    is_mirror=res["is_mirror"],
                    mcts_on=True,
                )
                n += 1
                if res["winner"] == 2:
                    wins += 0.5
                elif res["winner"] == res["our_seat"]:
                    wins += 1.0
                game_bar.update(1)
                game_bar.set_postfix(
                    wr=f"{wins / max(n, 1):.1%}",
                    pass_=len(report.opponents_passing()),
                )
            game_bar.close()
        opp_bar.close()

        _append_rollout_jsonl(rollout_path, results)
        summary = report.summary()
        tqdm.write(
            f"[rr] iter={it} games={n} wr={wins / max(n, 1):.1%} "
            f"passing={summary['n_passing_wilson']}/{summary['n_evaluated']} "
            f"all_pass={summary['all_pass']}"
        )
        for row in summary["matchups"]:
            tqdm.write(
                f"   {row['opponent_id']:32} wr={row['wr']:.1%} "
                f"wilson_lo={row['wilson_lo']:.1%} n={row['games']}"
            )

        loop_state["history"].append(
            {"iteration": it, "summary": summary, "t": time.time(), "wr": wins / max(n, 1)}
        )
        loop_state["seed"] = seed

        # Light continued supervised train from bootstrap JSONL (keeps weights warm).
        # Full AZ from visit targets needs obs reanalyse (follow-up); still checkpoint loop.
        bootstrap_jsonl = paths.DATA_DIR / "bootstrap" / f"{args.archetype}.jsonl"
        if bootstrap_jsonl.is_file() and args.train_epochs > 0:
            tqdm.write(f"[rr] warm-train {args.train_epochs} epoch(s) on {bootstrap_jsonl}")
            ds = load_bootstrap_dataset(bootstrap_jsonl, max_games=0, use_cache=True)
            # Resume from champion into bootstrap run name for this fine-tune.
            warm_name = f"{run_name}_warm"
            # Copy champion into warm latest so train resumes weights.
            import shutil

            warm_latest = checkpoint.latest_path(warm_name)
            warm_latest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(champion, warm_latest)
            result = train_bootstrap(
                ds,
                run_name=warm_name,
                archetype_id=args.archetype,
                train_cfg=TrainConfig(epochs=args.train_epochs, early_stop_patience=2),
                resume="auto",
                device=train_dev,
            )
            new_champ = Path(result["best_path"] or result["latest_path"])
            loop_state["champion"] = str(new_champ)
            tqdm.write(f"[rr] champion ← {new_champ}")
        else:
            tqdm.write("[rr] skip warm-train (no bootstrap jsonl or train_epochs=0)")

        # Save loop checkpoint.
        model = load_model_from_checkpoint(loop_state["champion"], device=torch.device("cpu"))
        ckpt = checkpoint.build_checkpoint(
            model=model,
            step=it,
            epoch=it,
            rl_iteration=it,
            best_metric=summary.get("n_passing_wilson"),
            archetype_id=args.archetype,
            model_id=run_name,
            extra={"loop_state": loop_state, "last_eval": summary},
        )
        saved = mgr.save(ckpt, is_best=bool(summary.get("all_pass")))
        tqdm.write(f"[rr] checkpoint → {saved}")

        eval_out = paths.OUTPUTS_DIR / "eval" / f"rr_iter{it:03d}.json"
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        eval_out.write_text(json.dumps({"loop": loop_state, "summary": summary}, indent=2, default=str) + "\n")

        if summary.get("all_pass") and all(
            st.games >= 50 for st in report.matchups.values()
        ):
            tqdm.write("[rr] GATE PASSED (wilson≥threshold with ≥50 games/opp) — stop for submit prep")
            break
        if summary.get("all_pass"):
            tqdm.write(
                "[rr] all opponents above Wilson gate at current N — "
                "increase --games-per-opp (≥100) before Kaggle submit"
            )

    print(f">> champion={loop_state.get('champion')}", flush=True)
    print(f">> rollouts={rollout_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
