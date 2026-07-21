#!/usr/bin/env python3
"""Isolated throughput/parity probe for the pure-RL replay cache builder."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import config
from poke_bot.pure_rl.dataset_bridge import (
    _build_parallel_cache,
    _cache_paths,
    _cache_signature,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    shard = args.shard.expanduser().resolve()
    config.HARDWARE.cache_dir = args.cache_dir.expanduser().resolve()
    signature = _cache_signature(
        shard, verify_info_set=False, max_context=args.max_context
    )
    cache_dir, _manifest_path = _cache_paths(shard, signature)
    shutil.rmtree(cache_dir, ignore_errors=True)
    started = time.monotonic()
    manifest = _build_parallel_cache(
        shard,
        cache_dir=cache_dir,
        signature=signature,
        verify_info_set=False,
        max_context=args.max_context,
        workers=args.workers,
    )
    elapsed = max(1e-9, time.monotonic() - started)
    source_bytes = int(signature["source_size"])
    covered_bytes = sum(int(part.get("bytes", 0)) for part in manifest["parts"])
    if covered_bytes != source_bytes:
        raise RuntimeError(
            f"byte coverage mismatch: covered={covered_bytes} source={source_bytes}"
        )
    result = {
        "source": str(shard),
        "source_bytes": source_bytes,
        "covered_bytes": covered_bytes,
        "workers": len(manifest["parts"]),
        "records": int(manifest["records"]),
        "sequences": int(manifest["sequences"]),
        "dropped": int(manifest["dropped"]),
        "elapsed_s": elapsed,
        "source_mib_per_s": source_bytes / elapsed / (1024 * 1024),
        "max_rss_main_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "max_rss_children_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        / 1024,
        "manifest": str(cache_dir / "manifest.json"),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not args.keep:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
