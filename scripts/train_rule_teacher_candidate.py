#!/usr/bin/env python3
"""Fine-tune an isolated neural checkpoint on protected rule-teacher wins.

This script never installs or publishes the candidate.  The resulting weights
must pass an exact official-baseline evaluation before a clean-boundary deploy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from poke_bot import checkpoint, paths
from poke_bot.dataset import load_bootstrap_dataset
from poke_bot.rule_teacher import (
    RULE_TEACHER_SCHEMA,
    atomic_json,
    file_digest,
    resolve_protected_teacher_corpus,
)
from poke_bot.train import TrainConfig, load_model_from_checkpoint, train_bootstrap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--games-per-batch", type=int, default=32)
    parser.add_argument("--max-decisions-per-batch", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--value-loss-weight", type=float, default=0.1)
    parser.add_argument("--lethal-threat-loss-weight", type=float, default=0.025)
    parser.add_argument("--prize-race-loss-weight", type=float, default=0.025)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--seed", type=int, default=951_000)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.games_per_batch < 1 or args.max_decisions_per_batch < 1:
        raise ValueError("training sizes must be positive")
    if args.lr <= 0.0 or args.value_loss_weight < 0.0:
        raise ValueError("learning rate/value loss weight is invalid")

    corpus, teacher_report = resolve_protected_teacher_corpus(
        args.report, corpus_override=args.corpus
    )
    initial = args.init_checkpoint.expanduser().resolve()
    if not initial.is_file():
        raise FileNotFoundError(f"missing initial neural checkpoint: {initial}")
    seed_model = load_model_from_checkpoint(initial, device=torch.device("cpu"))
    if (
        seed_model.decision_context != "history"
        or int(seed_model.cfg.temporal_layers) != 1
        or int(seed_model.max_context) != 320
    ):
        raise ValueError(
            "teacher candidate requires the live one-layer/320-step history profile"
        )
    del seed_model

    dataset = load_bootstrap_dataset(
        corpus,
        max_games=int(args.max_games),
        use_cache=not bool(args.no_cache),
        verify_info_set=True,
    )
    expected = dict(teacher_report.get("validation") or {})
    if int(args.max_games) <= 0:
        if len(dataset) != int(expected.get("records") or -1):
            raise RuntimeError("loaded teacher record count differs from protected report")
        if dataset.n_decisions != int(expected.get("decisions") or -1):
            raise RuntimeError("loaded teacher decision count differs from protected report")
    if not dataset.info_set_ok_all or len(dataset) < 1 or dataset.n_decisions < 1:
        raise RuntimeError("teacher dataset failed the in-memory training gate")

    train_cfg = TrainConfig(
        lr=float(args.lr),
        epochs=int(args.epochs),
        games_per_batch=int(args.games_per_batch),
        max_decisions_per_batch=int(args.max_decisions_per_batch),
        val_frac=float(args.val_frac),
        split_by_episode=False,
        early_stop_patience=int(args.patience),
        value_loss_weight=float(args.value_loss_weight),
        aux_loss_weight=0.0,
        opp_hand_loss_weight=0.0,
        opp_remainder_loss_weight=0.0,
        lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
        prize_race_loss_weight=float(args.prize_race_loss_weight),
        amp=True,
        pure_rl=False,
        seed=int(args.seed),
    )
    started = time.time()
    result = train_bootstrap(
        dataset,
        run_name=str(args.run_name),
        archetype_id="alakazam",
        train_cfg=train_cfg,
        resume=False,
        device=torch.device(str(args.device)),
        init_checkpoint=initial,
        checkpoint_extra={
            "rule_teacher": {
                "schema": RULE_TEACHER_SCHEMA,
                "report": str(Path(args.report).expanduser().resolve()),
                "report_digest": file_digest(Path(args.report).expanduser().resolve()),
                "corpus": str(corpus),
                "corpus_digest": file_digest(corpus),
                "teacher": (
                    (teacher_report.get("configuration") or {}).get("teacher") or {}
                ).get("id"),
                "opponent": (
                    (teacher_report.get("configuration") or {}).get("opponent") or {}
                ).get("id"),
                "outcome_filter": (
                    teacher_report.get("configuration") or {}
                ).get("outcome_filter"),
                "final_agent_runtime": "neural_only",
                "deployment_status": "isolated_candidate_requires_exact_gate",
            }
        },
        device_resident=False,
    )
    best = Path(str(result.get("best_path") or "")).expanduser().resolve()
    if not best.is_file():
        raise RuntimeError("teacher training did not produce a best checkpoint")
    candidate_report = {
        "schema": "poke_bot.rule_teacher_candidate/v1",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "run_name": str(args.run_name),
        "deployment_status": "isolated_candidate_requires_exact_gate",
        "final_agent_runtime": "neural_only",
        "initial_checkpoint": {
            "path": str(initial),
            "digest": checkpoint.checkpoint_digest(initial),
        },
        "teacher_report": {
            "path": str(Path(args.report).expanduser().resolve()),
            "digest": file_digest(Path(args.report).expanduser().resolve()),
        },
        "corpus": {
            "path": str(corpus),
            "digest": file_digest(corpus),
            "records": len(dataset),
            "decisions": dataset.n_decisions,
        },
        "train_config": train_cfg.__dict__,
        "candidate": {
            "path": str(best),
            "digest": checkpoint.checkpoint_digest(best),
            "best_metric": result.get("best_metric"),
        },
    }
    out = paths.OUTPUTS_DIR / "train" / f"{args.run_name}.teacher_candidate.json"
    atomic_json(out, candidate_report)
    print(json.dumps(candidate_report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
