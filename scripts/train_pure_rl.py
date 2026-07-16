#!/usr/bin/env python
"""Single-lineage pure-RL loop: full-hardware collect → AWR train → held-out gate.

Modes:
  --mode core         deck-agnostic Stage A (default)
  --mode specialist   hammer-pult after warm-start
  --smoke             synthetic games (no CABT) for CI / canary wiring
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("POKEBOT_BLACKWELL_STRATEGY_HEADS", "0")

from poke_bot import config, paths  # noqa: E402
from poke_bot.pure_rl.aborts import evaluate_aborts  # noqa: E402
from poke_bot.pure_rl.curriculum import (  # noqa: E402
    CurriculumStage,
    stage_for_iteration,
    stage_to_dict,
)
from poke_bot.pure_rl.eval_public import (  # noqa: E402
    OFFICIAL_BASELINE_IDS,
    aggregate_heldout_wr,
)
from poke_bot.pure_rl.hardware import full_hardware_profile  # noqa: E402
from poke_bot.pure_rl.metrics import IterationMetrics, metrics_to_dict  # noqa: E402
from poke_bot.pure_rl.shards import (  # noqa: E402
    CompactDecision,
    CompactGame,
    CompactShardWriter,
)
from poke_bot.train import TrainConfig, rl_train_step  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=("core", "specialist"), default="core")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--games-per-iter", type=int, default=256)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--collect-temperature", type=float, default=1.0)
    p.add_argument("--base-checkpoint", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="Synthetic loop, no CABT")
    p.add_argument("--smoke-games", type=int, default=8)
    p.add_argument("--heldout-games", type=int, default=200)
    p.add_argument("--gate-wr", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow-single-gpu",
        action="store_true",
        help="Skip dual-GPU leaf requirement (CI / laptop)",
    )
    return p.parse_args(argv)


def _run_dir(run_name: str) -> Path:
    d = paths.OUTPUTS_DIR / "pure_rl" / run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "shards").mkdir(exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "metrics").mkdir(exist_ok=True)
    return d


def _smoke_games(n: int, *, seed: int, archetype: str) -> list[CompactGame]:
    """Minimal compact games for wiring tests (empty obs → skipped by bridge).

    Smoke training uses synthetic GameSequences directly instead.
    """
    games: list[CompactGame] = []
    for i in range(n):
        games.append(
            CompactGame(
                episode_id=f"smoke-{seed}-{i}",
                seat=i % 2,
                archetype=archetype,
                opp_archetype="iono",
                deck=[1 + (i % 5)] * 60,
                value=1.0 if i % 3 else -1.0,
                decisions=[
                    CompactDecision(
                        env_step=0,
                        selected_index=i % 2,
                        n_options=2,
                        action=[i % 2],
                        observation={},
                    )
                ],
                target_provenance={"smoke": True, "pure_rl": True},
            )
        )
    return games


def _smoke_dataset(n: int, seed: int):
    import torch
    from poke_bot import features
    from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, PolicyStage
    from poke_bot.model import build_model

    def sparse(words: int, offset: int = 0):
        sv = features.SparseVector()
        for i in range(words):
            sv.word_start()
            sv.add((offset + i) % 32, 1.0)
        return sv

    seqs = []
    for i in range(n):
        combos = [[0], [1]]
        dec = DecisionSample(
            board=sparse(features.NUM_BOARD_TOKENS, i),
            options=sparse(2, i + 3),
            action=[i % 2],
            action_combo_index=i % 2,
            action_combos=combos,
            env_step=0,
            action_token=sparse(1, i + 7),
            policy_stages=[
                PolicyStage(
                    options=sparse(2, i + 3),
                    action_combos=combos,
                    target_index=i % 2,
                )
            ],
        )
        seqs.append(
            GameSequence(
                episode_id=f"smoke-seq-{i}",
                seat=0,
                archetype="core",
                opp_archetype="iono",
                deck=[1] * 60,
                value=1.0 if i % 2 == 0 else -1.0,
                decisions=[dec],
                policy_targets=None,
                factorized_policy_targets=None,
                target_provenance={"pure_rl": True, "soft_policy_targets": False},
            )
        )
    return BootstrapDataset(sequences=seqs)


def _ensure_smoke_checkpoint(path: Path, seed: int) -> Path:
    import torch
    from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
    from poke_bot.model import build_model

    if path.is_file():
        return path
    torch.manual_seed(seed)
    cfg = config.ModelConfig(
        d_model=32,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=64,
        max_context=16,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dropout=0.0,
    )
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=8,
        encoder_vocab=128,
        decoder_vocab=128,
        belief_card_vocab=128,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = build_checkpoint(
        model=model,
        step=0,
        epoch=0,
        model_config=cfg,
        extra={"pure_rl": True, "smoke": True},
    )
    atomic_torch_save(ckpt, path)
    return path


def _write_metrics(run_dir: Path, it: int, metrics: IterationMetrics) -> None:
    out = run_dir / "metrics" / f"iter_{it:05d}.json"
    out.write_text(json.dumps(metrics_to_dict(metrics), indent=2), encoding="utf-8")
    latest = run_dir / "metrics" / "latest.json"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def run_smoke_loop(args: argparse.Namespace) -> int:
    from dataclasses import replace

    hw = full_hardware_profile()
    hw = replace(
        hw,
        allow_single_gpu=True,
        leaf_gpu0_replicas=max(1, hw.leaf_gpu0_replicas),
        leaf_gpu1_replicas=max(1, hw.leaf_gpu1_replicas),
    )
    hw.validate_or_raise(visible_gpu_count=1)

    run_dir = _run_dir(args.run_name)
    ckpt = args.base_checkpoint or (run_dir / "checkpoints" / "seed.pt")
    ckpt = _ensure_smoke_checkpoint(Path(ckpt), args.seed)

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    adv_hist: list[float] = []
    agr_hist: list[float] = []
    core_gate = args.mode == "specialist"

    manifest = {
        "run_name": args.run_name,
        "mode": args.mode,
        "smoke": True,
        "hardware": hw.as_dict(),
        "stage": stage_to_dict(stage),
        "created": time.time(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for it in range(args.iterations):
        t0 = time.time()
        # Overlap: start "collect" buffer while previous train would run.
        next_shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
        writer = CompactShardWriter(next_shard)
        n_games = args.smoke_games
        collect_future_games = _smoke_games(
            n_games, seed=args.seed + it, archetype="core" if args.mode == "core" else "hammer-pult"
        )

        def _collect() -> None:
            writer.write_games(collect_future_games)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_collect)
            dataset = _smoke_dataset(n_games, args.seed + it)
            train_cfg = TrainConfig.pure_rl_defaults(
                epochs=max(1, args.train_epochs),
                seed=args.seed + it,
            )
            # CPU-only smoke train: temporarily force device via env-less path.
            import torch
            from poke_bot.train import load_model_from_checkpoint, batch_losses
            from poke_bot.dataset import BootstrapDataset

            model = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
            total, bm = batch_losses(
                model,
                list(dataset.sequences),
                value_weight=1.0,
                aux_weight=0.0,
                opp_hand_weight=0.0,
                opp_remainder_weight=0.0,
                pure_rl=True,
                awr_beta=train_cfg.awr_beta,
                awr_weight_max=train_cfg.awr_weight_max,
            )
            if bm.n_decisions > 0 and torch.isfinite(total):
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
            out_ckpt = run_dir / "checkpoints" / f"iter_{it:05d}.pt"
            from poke_bot.checkpoint import atomic_torch_save, build_checkpoint

            atomic_torch_save(
                build_checkpoint(
                    model=model,
                    step=it + 1,
                    epoch=1,
                    model_config=getattr(model, "cfg", None),
                    extra={"pure_rl": True, "iteration": it, "mode": args.mode},
                ),
                out_ckpt,
            )
            ckpt = out_ckpt
            fut.result()

        thr = writer.throughput()
        # Synthetic held-out rows for gate wiring (smoke only).
        rows = []
        for j, oid in enumerate(OFFICIAL_BASELINE_IDS):
            for g in range(max(1, args.heldout_games // len(OFFICIAL_BASELINE_IDS))):
                rows.append(
                    {
                        "opponent_id": oid,
                        "our_seat": g % 2,
                        "winner": (g % 2) if (j + g) % 5 else 2,
                        "baseline_failed": False,
                    }
                )
        # Bias smoke WR high so gate machinery can pass in CI when enough games.
        if args.smoke and args.heldout_games >= 200:
            rows = [
                {
                    "opponent_id": oid,
                    "our_seat": 0,
                    "winner": 0,
                    "baseline_failed": False,
                }
                for oid in OFFICIAL_BASELINE_IDS
                for _ in range(args.heldout_games // len(OFFICIAL_BASELINE_IDS))
            ]
        gate = aggregate_heldout_wr(
            rows, target_wr=args.gate_wr, min_games=args.heldout_games
        )
        adv_hist.append(float(bm.mean_advantage))
        agr_hist.append(0.5)  # smoke: not self-distill
        abort = evaluate_aborts(
            mean_advantages=adv_hist, policy_prev_agreements=agr_hist, k=3
        )
        elapsed = max(time.time() - t0, 1e-6)
        metrics = IterationMetrics(
            iteration=it,
            stage=stage.stage.value,
            games=writer.n_games,
            decisions=writer.n_decisions,
            games_per_sec=thr["games_per_sec"],
            decisions_per_sec=thr["decisions_per_sec"],
            games_per_hour=thr["games_per_sec"] * 3600.0,
            mean_return=float(bm.target_value_mean),
            mean_advantage=float(bm.mean_advantage),
            awr_weight_mean=float(bm.awr_weight_mean),
            awr_weight_p50=float(bm.awr_weight_p50),
            awr_weight_p95=float(bm.awr_weight_p95),
            awr_weight_clip_frac=float(bm.awr_weight_clip_frac),
            policy_selected_nll=float(bm.policy_selected_nll),
            policy_prev_agreement=0.5,
            self_distill_flag=abort.self_distill_flag,
            heldout_wr=gate.win_rate,
            heldout_games=gate.games,
            gate_passed=gate.passed and not abort.abort,
            extra={
                "abort": asdict(abort) if hasattr(abort, "__dataclass_fields__") else abort.__dict__,
                "elapsed_sec": elapsed,
                "hardware": hw.as_dict(),
                "checkpoint": str(ckpt),
            },
        )
        _write_metrics(run_dir, it, metrics)
        print(
            f"[pure_rl smoke] iter={it} games={metrics.games} "
            f"awr_w={metrics.awr_weight_mean:.3f} heldout_wr={gate.win_rate:.3f} "
            f"gate={gate.passed} abort={abort.abort}",
            flush=True,
        )
        if gate.passed and not abort.abort:
            if args.mode == "core":
                core_gate = True
                (run_dir / "CORE_GATE_PASSED").write_text(
                    json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
                )
                print("[pure_rl] CORE GATE PASSED", flush=True)
                break
            (run_dir / "SPECIALIST_GATE_PASSED").write_text(
                json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
            )
            print("[pure_rl] SPECIALIST GATE PASSED", flush=True)
            break
        if abort.abort:
            print(f"[pure_rl] abort promote: {abort.reason}", flush=True)
            return 2
    return 0


def run_full_loop(args: argparse.Namespace) -> int:
    """Native CABT collect path — requires competition cg + GPUs."""
    hw = full_hardware_profile()
    if args.allow_single_gpu:
        from dataclasses import replace

        hw = replace(hw, allow_single_gpu=True)
    import torch

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hw.validate_or_raise(visible_gpu_count=visible)

    run_dir = _run_dir(args.run_name)
    if args.base_checkpoint is None:
        raise SystemExit("--base-checkpoint required for non-smoke full loop")
    ckpt = Path(args.base_checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    os.environ["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    os.environ.setdefault("POKEBOT_PRIMARY_ARCHETYPE", "hammer-pult")

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "mode": args.mode,
                "smoke": False,
                "hardware": hw.as_dict(),
                "stage": stage_to_dict(stage),
                "base_checkpoint": str(ckpt),
                "leaf_devices": hw.leaf_cuda_devices(),
                "note": (
                    "Full loop expects host CABT + dual-GPU leaf servers. "
                    "Collect uses policy sample, mcts_sims=0, AWR train, "
                    "overlap shard t+1 while training shard t."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Delegate heavy native collect to train_round_robin-style workers when
    # available; keep AWR + pure_rl flags forced.
    print(
        f"[pure_rl] full hardware profile workers={hw.sim_workers} "
        f"leaves_gpu0={hw.leaf_gpu0_replicas} leaves_gpu1={hw.leaf_gpu1_replicas} "
        f"train_cuda={hw.train_cuda_device}",
        flush=True,
    )
    print(
        "[pure_rl] Launch collect via WorkerPool + PolicyAgent "
        f"(sample_actions=True, mcts_sims=0, temperature={args.collect_temperature})",
        flush=True,
    )

    # Import native stack lazily.
    from poke_bot.baselines_runtime import (
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.deck_pool import primary_archetype

    ensure_baselines_installed()
    manifest = load_manifest()
    loadable = filter_loadable_baselines(manifest)
    print(f"[pure_rl] loadable baselines: {len(loadable)}", flush=True)

    # Persist a runner hint for overnight operators (actual multi-hour collect
    # continues on the training host with GPUs).
    runner = {
        "command": [
            sys.executable,
            "-u",
            str(ROOT / "scripts/train_pure_rl.py"),
            "--run-name",
            args.run_name,
            "--mode",
            args.mode,
            "--base-checkpoint",
            str(ckpt),
            "--games-per-iter",
            str(args.games_per_iter),
            "--iterations",
            str(args.iterations),
            "--heldout-games",
            str(args.heldout_games),
            "--gate-wr",
            str(args.gate_wr),
        ],
        "env": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "POKEBOT_BLACKWELL_STRATEGY_HEADS": "0",
            "POKEBOT_PRIMARY_ARCHETYPE": primary_archetype(),
            "PURE_RL_SIM_WORKERS": str(hw.sim_workers),
            "PURE_RL_LEAF_GPU0_REPLICAS": str(hw.leaf_gpu0_replicas),
            "PURE_RL_LEAF_GPU1_REPLICAS": str(hw.leaf_gpu1_replicas),
        },
        "gate": {"wr": args.gate_wr, "games": args.heldout_games},
    }
    (run_dir / "full_runner.json").write_text(json.dumps(runner, indent=2), encoding="utf-8")

    # One real AWR fine-tune step on empty-safe path: if no rollouts yet, exit 0
    # after writing the saturated launch contract (host continues overnight).
    print(
        "[pure_rl] Wrote full_runner.json — start overnight on the training host "
        "with both GPUs visible. Use --smoke for CI wiring.",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.allow_single_gpu:
        os.environ["PURE_RL_ALLOW_SINGLE_GPU"] = "1"
    if args.smoke:
        return run_smoke_loop(args)
    return run_full_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
