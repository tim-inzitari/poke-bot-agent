#!/usr/bin/env python3
"""Stage checksummed, route-specific matchup-adapter expert shards.

This command never infers oracle labels from compact features.  It requires a
row-aligned oracle manifest rebuilt from the pinned raw archives and an active
gate contract with exact package exclusions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.matchup_adapter_corpus import stage_matchup_adapter_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--package-registry", type=Path, required=True)
    parser.add_argument("--active-gate-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disk-floor-gib", type=float, default=8.0)
    parser.add_argument("--require-public-router-digest", default=None)
    args = parser.parse_args()
    manifest = stage_matchup_adapter_corpus(
        args.feature_manifest,
        args.oracle_manifest,
        args.package_registry,
        args.active_gate_contract,
        args.output_dir,
        val_frac=args.val_frac,
        seed=args.seed,
        min_available_bytes=int(args.disk_floor_gib * 1024**3),
        required_public_router_digest=args.require_public_router_digest,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "membership_digest": payload["membership_digest"],
                "selected_sequences": payload["totals"]["selected_sequences"],
                "decisions": payload["totals"]["decisions"],
                "runtime_routes_enabled": payload["runtime_routes_enabled"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
