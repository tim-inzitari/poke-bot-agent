#!/usr/bin/env python3
"""Run the exact local-only 25-epoch r274 successor bootstrap on Inzi."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from poke_bot import checkpoint
from poke_bot.r241_own_deck_successor import load_r260_owner_contract
from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex
from poke_bot.train import TrainConfig, streaming_r260_host_rehearsal_step


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sidecar-binding", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--tactical-overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-games", type=int, default=16)
    parser.add_argument("--manifest-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=274)
    args = parser.parse_args()

    owner = load_r260_owner_contract()
    binding = json.loads(args.sidecar_binding.read_text(encoding="utf-8"))
    daily: dict[str, str] = {}
    for day, row in dict(binding["daily_sidecar_meta_receipts"]).items():
        meta = json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))
        daily[str(day)] = str(meta["meta_sha256"])
    index = R260InziSidecarIndex(
        args.index,
        source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    index.assert_verified(
        expected_source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )

    parent = checkpoint.load_checkpoint(args.base_checkpoint, map_location="cpu")
    expanded = dict(
        dict(parent.get("extra") or {})
        .get("expanded_head_training", {})
        .get("loss_weights", {})
    )
    cfg = TrainConfig(
        lr=1e-5,
        weight_decay=0.0,
        epochs=int(args.epochs),
        games_per_batch=int(args.batch_games),
        max_decisions_per_batch=2048,
        split_by_episode=True,
        value_loss_weight=1.0,
        aux_loss_weight=0.05,
        opp_hand_loss_weight=0.05,
        opp_remainder_loss_weight=0.05,
        lethal_threat_loss_weight=0.025,
        prize_race_loss_weight=0.025,
        setup_board_outcome_loss_weight=0.025,
        combo_state_loss_weight=0.025,
        visible_tutor_completion_loss_weight=0.025,
        terminal_conversion_loss_weight=0.025,
        tactical_sequence_outcome_loss_weight=0.025,
        expanded_head_loss_weights=expanded,
        collect_own_deck_promotion_metrics=True,
        pure_rl=False,
        entropy_bonus=0.0,
        amp=True,
        seed=int(args.seed),
    )
    result = streaming_r260_host_rehearsal_step(
        manifest_path=args.manifest,
        manifest_digest=sha256_file(args.manifest),
        sidecar_index=index,
        base_ckpt=args.base_checkpoint,
        output_path=args.output,
        archetype_id="alakazam",
        epochs=int(args.epochs),
        cfg=cfg,
        seed=int(args.seed),
        max_context=320,
        batch_games=int(args.batch_games),
        manifest_workers=int(args.manifest_workers),
        device=torch.device(args.device),
        tactical_overlay_path=args.tactical_overlay,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
