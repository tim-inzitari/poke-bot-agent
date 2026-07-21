#!/usr/bin/env python
"""Supervised bootstrap training on per-archetype JSONL (Phase 3/4).

Default: pure Dragapult deck/archetype, Blackwell AMP, ``--resume auto``.

Example::

    /home/inzi/miniconda3/envs/poke-bot-agent/bin/python scripts/train_bootstrap.py \\
        --jsonl data/bootstrap/dragapult.jsonl --archetype dragapult --resume auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import config, paths
from poke_bot.dataset import load_bootstrap_dataset
from poke_bot.train import TrainConfig, train_bootstrap


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Bootstrap JSONL (default data/bootstrap/<archetype>.jsonl)",
    )
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=None,
        help="Validated compact feature-shard manifest (mutually exclusive with --jsonl).",
    )
    p.add_argument("--archetype", default="dragapult")
    p.add_argument("--run-name", default=None, help="Checkpoint run name")
    p.add_argument("--resume", default="auto", help="auto | 0 | path | best")
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Weights-only initialization; optimizer/epoch state starts fresh.",
    )
    p.add_argument(
        "--model-profile",
        choices=("default", "pure-rl"),
        default="default",
        help="Use the exact small pure-RL architecture for a compatible hot start.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--games-per-batch", type=int, default=4)
    p.add_argument("--max-decisions-per-batch", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument(
        "--split-by-episode",
        action="store_true",
        help="Keep both seats from an episode wholly in train or validation.",
    )
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-games", type=int, default=0, help="Cap games loaded (0=all)")
    p.add_argument(
        "--min-usable-record-frac",
        type=float,
        default=0.0,
        help="Fail before training if featurization keeps less than this fraction.",
    )
    p.add_argument(
        "--min-decisions",
        type=int,
        default=0,
        help="Fail before training below this many usable decisions.",
    )
    p.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.1,
        help="Archetype aux_head CE weight (default 0.1).",
    )
    p.add_argument(
        "--opp-hand-loss-weight",
        type=float,
        default=0.2,
        help="opp_hand_head multilabel BCE weight (default 0.2; masked if labels absent).",
    )
    p.add_argument(
        "--opp-remainder-loss-weight",
        type=float,
        default=0.15,
        help="opp_remainder_head multilabel BCE weight (default 0.15; masked if absent).",
    )
    p.add_argument(
        "--lethal-threat-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Scope B lethal_threat_head weight (default 0 — core/bootstrap off; "
            "enable on Blackwell Hammer specialist trains)."
        ),
    )
    p.add_argument(
        "--prize-race-loss-weight",
        type=float,
        default=0.0,
        help="Scope B prize_race_head weight (default 0 — core/bootstrap off).",
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--device-resident",
        action="store_true",
        help=(
            "Pack the complete hard-target stateless corpus onto the training "
            "GPU once and train without per-batch host transfers."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    if args.jsonl is not None and args.feature_manifest is not None:
        print("ERROR: --jsonl and --feature-manifest are mutually exclusive", file=sys.stderr)
        return 2
    jsonl = args.jsonl or (paths.DATA_DIR / "bootstrap" / f"{args.archetype}.jsonl")
    feature_manifest = args.feature_manifest
    if feature_manifest is None and not jsonl.is_file():
        print(f"ERROR: missing JSONL {jsonl}", file=sys.stderr)
        return 2
    if feature_manifest is not None and not feature_manifest.is_file():
        print(f"ERROR: missing feature manifest {feature_manifest}", file=sys.stderr)
        return 2
    if feature_manifest is not None:
        compact_incompatible = (
            args.model_profile != "pure-rl"
            or float(args.aux_loss_weight) != 0.0
            or float(args.opp_hand_loss_weight) != 0.0
            or float(args.opp_remainder_loss_weight) != 0.0
            or float(args.lethal_threat_loss_weight) != 0.0
            or float(args.prize_race_loss_weight) != 0.0
        )
        if compact_incompatible:
            print(
                "ERROR: compact feature manifests require --model-profile pure-rl "
                "and all auxiliary loss weights set to zero",
                file=sys.stderr,
            )
            return 2

    run_name = args.run_name or f"{args.archetype}_bootstrap"
    print(f"== train_bootstrap archetype={args.archetype}", flush=True)
    print(
        f"   dataset={feature_manifest if feature_manifest is not None else jsonl}",
        flush=True,
    )
    print(f"   run_name={run_name} resume={args.resume}", flush=True)
    print(f"   device_resident={bool(args.device_resident)}", flush=True)

    if feature_manifest is not None:
        if args.max_games > 0:
            print("ERROR: --max-games is not supported with --feature-manifest", file=sys.stderr)
            return 2
        from poke_bot.feature_shards import load_feature_manifest

        ds = load_feature_manifest(feature_manifest, verify_hashes=True)
    else:
        ds = load_bootstrap_dataset(
            jsonl,
            max_games=args.max_games,
            use_cache=not args.no_cache,
            verify_info_set=True,
        )
    summary = ds.summary()
    print(f"   dataset: {json.dumps(summary)}", flush=True)
    if len(ds) == 0:
        print("ERROR: empty dataset", file=sys.stderr)
        return 1
    if not ds.info_set_ok_all:
        print("ERROR: info-set integrity failed in dataset", file=sys.stderr)
        return 1
    conversion = dict(summary.get("conversion") or {})
    records_total = int(conversion.get("records_total") or len(ds))
    usable_record_frac = len(ds) / max(1, records_total)
    if usable_record_frac < float(args.min_usable_record_frac):
        print(
            f"ERROR: usable record fraction {usable_record_frac:.4f} < "
            f"minimum {float(args.min_usable_record_frac):.4f}",
            file=sys.stderr,
        )
        return 1
    if ds.n_decisions < int(args.min_decisions):
        print(
            f"ERROR: usable decisions {ds.n_decisions} < "
            f"minimum {int(args.min_decisions)}",
            file=sys.stderr,
        )
        return 1

    cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        games_per_batch=args.games_per_batch,
        max_decisions_per_batch=args.max_decisions_per_batch,
        val_frac=args.val_frac,
        split_by_episode=bool(args.split_by_episode),
        early_stop_patience=args.patience,
        aux_loss_weight=float(args.aux_loss_weight),
        opp_hand_loss_weight=float(args.opp_hand_loss_weight),
        opp_remainder_loss_weight=float(args.opp_remainder_loss_weight),
        lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
        prize_race_loss_weight=float(args.prize_race_loss_weight),
        amp=not args.no_amp,
        seed=args.seed,
    )
    model_cfg = None
    checkpoint_extra = None
    if args.model_profile == "pure-rl":
        from poke_bot.pure_rl.model_profile import (
            model_config_dict,
            pure_rl_model_config,
        )

        model_cfg = pure_rl_model_config()
        provenance_path = feature_manifest if feature_manifest is not None else jsonl
        digest = hashlib.sha256()
        with provenance_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        checkpoint_extra = {
            "pure_rl": True,
            "smoke": False,
            "model_profile": model_config_dict(model_cfg),
            "top_ladder_hot_start": {
                "dataset": str(provenance_path.resolve()),
                "dataset_sha256": "sha256:" + digest.hexdigest(),
                "split_by_episode": bool(args.split_by_episode),
                "archetype": str(args.archetype),
            },
        }
    result = train_bootstrap(
        ds,
        run_name=run_name,
        archetype_id=args.archetype,
        train_cfg=cfg,
        resume=args.resume,
        model_cfg=model_cfg,
        init_checkpoint=args.init_checkpoint,
        checkpoint_extra=checkpoint_extra,
        device_resident=bool(args.device_resident),
    )
    out = paths.OUTPUTS_DIR / "train" / f"{run_name}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Drop huge history tensors if any; history is JSON-safe already.
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f">> result → {out}", flush=True)
    print(f">> best={result.get('best_path')} metric={result.get('best_metric')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
