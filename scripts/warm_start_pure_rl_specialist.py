#!/usr/bin/env python
"""Warm-start an explicitly selected specialist from a pure-RL/core checkpoint.

Uses Hope ``CoreKernel.warm_start_specialist`` when the checkpoint is a core
kernel; otherwise copies a plain TemporalCabtTransformer checkpoint into the
specialist run directory for ``train_pure_rl.py --mode specialist``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-checkpoint", type=Path, required=True)
    p.add_argument(
        "--archetype",
        required=True,
        help="Specialist chosen from a versioned ladder-value evaluation",
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    import torch
    from poke_bot import paths
    from poke_bot.core_kernel import warm_start_specialist_from_checkpoint
    from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
    from poke_bot.train import load_model_from_checkpoint

    out_dir = paths.OUTPUTS_DIR / "pure_rl" / args.run_name / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.archetype}_warmstart.pt"
    device = torch.device(args.device)

    ckpt_path = Path(args.core_checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")

    # Prefer core-kernel API when the file carries core_kernel_state_dict.
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {
        "core_checkpoint": str(ckpt_path),
        "archetype": args.archetype,
        "run_name": args.run_name,
    }
    if isinstance(blob, dict) and (
        "core_kernel_state_dict" in blob or blob.get("is_core_kernel")
    ):
        result = warm_start_specialist_from_checkpoint(
            ckpt_path,
            args.archetype,
            run_name=args.run_name,
            device=device,
            write_checkpoint=True,
        )
        # Also copy/symlink into pure_rl run tree for launch_pure_rl.
        src = Path(result.get("latest_path") or result.get("path") or "")
        if src.is_file():
            out_path.write_bytes(src.read_bytes())
        meta["method"] = "core_kernel_warm_start"
        meta["source_result_keys"] = sorted(result.keys())
    else:
        model = load_model_from_checkpoint(ckpt_path, device=device)
        atomic_torch_save(
            build_checkpoint(
                model=model,
                step=0,
                epoch=0,
                model_config=getattr(model, "cfg", None),
                extra={
                    "pure_rl": True,
                    "warm_started_from": str(ckpt_path),
                    "archetype": args.archetype,
                },
            ),
            out_path,
        )
        meta["method"] = "plain_checkpoint_copy"

    meta["specialist_checkpoint"] = str(out_path)
    meta_path = out_dir / "warm_start_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
