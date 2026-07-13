#!/usr/bin/env python3
"""Prepare a new self-play segment after an early stop or iteration budget exhaustion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.self_play.rollout_io import load_manifest, save_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archetype", help="profile name, e.g. lucario or dragapult")
    args = parser.parse_args()

    arch = args.archetype.strip()
    manifest_path = ROOT / "outputs/checkpoints/self_play" / arch / "manifest.json"
    fresh_ckpt = ROOT / "outputs/checkpoints" / f"{arch}_fresh.pt"
    state_path = ROOT / "outputs/logs" / f"{arch}_continue_state.json"

    if not manifest_path.is_file() and not fresh_ckpt.is_file():
        print(f"no manifest or bootstrap checkpoint for {arch}; nothing to continue")
        return 1

    state: dict = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    if not manifest_path.is_file():
        print(f"manifest missing ({manifest_path}); will resume bootstrap/self-play from fresh checkpoint")
        return 0

    manifest = load_manifest(manifest_path)
    phase = str(manifest.get("phase", "baseline"))
    baseline_iters = manifest.get("baseline_iterations") or []
    transformer_iters = manifest.get("iterations") or []
    stop_reason = manifest.get("stop_reason")

    if phase == "baseline" and not stop_reason:
        aggregate = 0.0
        if baseline_iters:
            aggregate = float(baseline_iters[-1].get("eval_vs_baselines_aggregate", {}).get("win_rate", 0.0))
        if aggregate <= 0.0:
            print(f"baseline phase in progress for {arch}; continuation will resume baseline loop")
            return 0

    if stop_reason:
        print(f"clearing stop_reason for {arch}: {stop_reason}")
        manifest.pop("stop_reason", None)

    manifest["plateau_count"] = 0
    save_manifest(manifest_path, manifest)

    segment = int(state.get("segments", 0)) + 1
    state.update(
        {
            "archetype": arch,
            "segments": segment,
            "last_continue_utc": datetime.now(timezone.utc).isoformat(),
            "last_stop_reason": stop_reason,
            "baseline_iterations": len(baseline_iters),
            "transformer_iterations": len(transformer_iters),
            "phase": phase,
        }
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"prepared segment {segment} for {arch}: "
        f"phase={phase} baseline={len(baseline_iters)} transformer={len(transformer_iters)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
