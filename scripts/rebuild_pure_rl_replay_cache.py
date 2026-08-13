#!/usr/bin/env python3
"""Build and validate a derived compact Pure-RL replay cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from poke_bot.pure_rl.dataset_bridge import ensure_replay_cache_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--max-context", type=int, required=True)
    args = parser.parse_args()
    manifest = ensure_replay_cache_manifest(
        args.shard,
        verify_info_set=False,
        max_context=args.max_context,
    )
    print(
        "REPLAY_CACHE_READY "
        f"records={manifest['records']} sequences={manifest['sequences']} "
        f"dropped={manifest['dropped']} "
        f"schema={manifest['signature']['compact_cache_schema']} "
        f"manifest={manifest['manifest_path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
