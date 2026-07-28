#!/usr/bin/env python3
"""Build and validate one bounded, reusable expert CPU pack."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT  # noqa: E402
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)


REQUIRED_TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=5_000_000)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--belief-card-vocab", type=int, required=True)
    parser.add_argument("--min-decisions", type=int, default=100_000)
    parser.add_argument("--archetype", default="alakazam")
    parser.add_argument("--pack-workers", type=int, default=1)
    parser.add_argument("--pack-memory-reserve-gib", type=float, default=12.0)
    parser.add_argument("--pack-disk-reserve-gib", type=float, default=16.0)
    parser.add_argument(
        "--allow-unprotected",
        action="store_true",
        help="Canary only: permit a direct manifest instead of a sealed pointer.",
    )
    args = parser.parse_args()

    started = time.time()
    identity = resolve_expert_manifest(
        args.corpus,
        min_decisions=int(args.min_decisions),
        require_protected=not bool(args.allow_unprotected),
        required_archetype=str(args.archetype),
        required_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        required_max_context=int(args.max_context),
        required_target_coverage=REQUIRED_TARGETS,
    )
    cache = ResidentExpertCorpusCache(cpu_pack_root=args.cache_root)
    corpus = cache.prepare(
        identity,
        device=torch.device("cpu"),
        seed=int(args.split_seed),
        val_frac=float(args.val_frac),
        max_context=int(args.max_context),
        belief_card_vocab=int(args.belief_card_vocab),
        pack_workers=int(args.pack_workers),
        pack_memory_reserve_gib=float(args.pack_memory_reserve_gib),
        pack_disk_reserve_gib=float(args.pack_disk_reserve_gib),
    )
    validate_cpu_corpus(corpus)
    if not corpus.has_temporal_layout or not corpus.has_exact_targets:
        raise RuntimeError("built expert CPU pack lacks temporal/all-head layout")
    if int(corpus.decisions) <= 0 or int(corpus.total_samples) <= 0:
        raise RuntimeError("built expert CPU pack is empty")
    pack_info = dict(cache.pack_info or {})
    receipt = {
        "schema": "poke_bot.expert_cpu_pack_prebuild/v1",
        "completed_at": time.time(),
        "elapsed_sec": time.time() - started,
        "manifest": identity.as_dict(),
        "split_seed": int(args.split_seed),
        "val_frac": float(args.val_frac),
        "max_context": int(args.max_context),
        "belief_card_vocab": int(args.belief_card_vocab),
        "pack_workers": int(args.pack_workers),
        "pack_memory_reserve_gib": float(args.pack_memory_reserve_gib),
        "pack_disk_reserve_gib": float(args.pack_disk_reserve_gib),
        "train_games": int(corpus.train_games),
        "val_games": int(corpus.val_games),
        "decisions": int(corpus.decisions),
        "samples": int(corpus.total_samples),
        "tensor_bytes": int(corpus.tensor_bytes),
        "temporal_layout": bool(corpus.has_temporal_layout),
        "exact_targets": bool(corpus.has_exact_targets),
        "pack": pack_info,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
