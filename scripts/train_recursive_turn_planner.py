#!/usr/bin/env python3
"""Train the Recursive Turn Planner (+ optional PokeRLM core) for multi-turn planning.

Default: synthetic smoke train (CI / no shards).
Host: pass --checkpoint and --shard to encode frozen CABT features from replay.

Does not rewrite GOAL.md, parent CABT tensors, or managed Alakazam services.
Serving remains opt-in via PolicyAgent RTP path + POKEBOT_RTP_CHECKPOINT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.poke_rlm.training.shadow_train import (  # noqa: E402
    PokeRLMTrainConfig,
    train_poke_rlm_shadow,
)
from poke_bot.recursive_turn_planner.training.shadow_train import (  # noqa: E402
    RTPTrainConfig,
    encode_sequences_to_batches,
    make_synthetic_batches,
    train_rtp_shadow,
)


def _load_shard_batches(args: argparse.Namespace):
    from poke_bot.pure_rl.dataset_bridge import compact_game_to_sequence
    from poke_bot.pure_rl.shards import iter_shard_games
    from poke_bot.train import load_model_from_checkpoint

    device = args.device
    model = load_model_from_checkpoint(Path(args.checkpoint), device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    sequences = []
    for index, game in enumerate(iter_shard_games(Path(args.shard))):
        if index >= int(args.max_games):
            break
        sequence = compact_game_to_sequence(
            game, verify_info_set=False, max_context=int(model.max_context)
        )
        if sequence is not None:
            sequences.append(sequence)
    if not sequences:
        raise SystemExit("no trainable sequences from shard")
    return encode_sequences_to_batches(
        model,
        sequences,
        option_threshold=int(args.complexity_option_threshold),
        entropy_threshold=float(args.complexity_entropy_threshold),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true", help="CI smoke path")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Frozen CABT parent")
    parser.add_argument("--shard", type=Path, default=None, help="Replay shard path")
    parser.add_argument("--max-games", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--profile", type=str, default="pure_rl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-synthetic", type=int, default=64)
    parser.add_argument("--complexity-option-threshold", type=int, default=8)
    parser.add_argument("--complexity-entropy-threshold", type=float, default=1.5)
    parser.add_argument(
        "--also-poke-rlm",
        action="store_true",
        help="Also train PokeRLMModelCore sidecar on the same batches",
    )
    args = parser.parse_args()

    if args.synthetic or (args.checkpoint is None and args.shard is None):
        batches = make_synthetic_batches(
            n_decisions=int(args.n_synthetic),
            d_model=int(args.d_model),
            seed=int(args.seed),
            option_threshold=int(args.complexity_option_threshold),
            entropy_threshold=float(args.complexity_entropy_threshold),
        )
        source = "synthetic"
    else:
        if args.checkpoint is None or args.shard is None:
            raise SystemExit("host train requires both --checkpoint and --shard")
        batches = _load_shard_batches(args)
        source = "shard"

    rtp = train_rtp_shadow(
        batches,
        output_dir=args.out_dir / "rtp",
        config=RTPTrainConfig(
            d_model=int(args.d_model),
            profile=str(args.profile),
            epochs=int(args.epochs),
            lr=float(args.lr),
            seed=int(args.seed),
            device=str(args.device),
            complexity_option_threshold=int(args.complexity_option_threshold),
            complexity_entropy_threshold=float(args.complexity_entropy_threshold),
        ),
    )
    summary: dict = {
        "source": source,
        "n_batches": len(batches),
        "rtp": {
            "checkpoint_path": rtp.checkpoint_path,
            "receipt_path": rtp.receipt_path,
            "metrics": rtp.metrics,
            "env_load": f"POKEBOT_RTP_CHECKPOINT={rtp.checkpoint_path}",
        },
        "serving_eligible": False,
    }
    if args.also_poke_rlm:
        poke = train_poke_rlm_shadow(
            batches,
            output_dir=args.out_dir / "poke_rlm",
            config=PokeRLMTrainConfig(
                profile="pure_rl_96" if int(args.d_model) == 96 else "base_384",
                d_model=int(args.d_model),
                epochs=int(args.epochs),
                lr=float(args.lr),
                seed=int(args.seed),
                device=str(args.device),
            ),
        )
        summary["poke_rlm"] = {
            "checkpoint_path": poke.checkpoint_path,
            "receipt_path": poke.receipt_path,
            "metrics": poke.metrics,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
