#!/usr/bin/env python3
"""Build one immutable Inzi-local contiguous pure-RL replay window."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from poke_bot.pure_rl.replay_parallel_prepare import (
    SUPPORTED_WORKERS,
    build_parallel_replay_window_pack,
)


INZI_HOSTNAMES = frozenset({"inzi", "inzi-MS-7C35"})


def assert_inzi_host(*, allow_diagnostic_host: bool) -> None:
    hostname = socket.gethostname()
    if hostname not in INZI_HOSTNAMES and not allow_diagnostic_host:
        raise RuntimeError(
            f"production RL replay packing is Inzi-only; host={hostname!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--exact-card-vocab", type=int)
    parser.add_argument("--memory-reserve-gib", type=float, default=4.0)
    parser.add_argument("--force-strategic", action="store_true")
    parser.add_argument("--diagnostic-host", action="store_true")
    parser.add_argument("--diagnostic-worker-count", action="store_true")
    args = parser.parse_args()
    assert_inzi_host(allow_diagnostic_host=bool(args.diagnostic_host))
    if args.workers not in SUPPORTED_WORKERS:
        parser.error(f"--workers must be one of {sorted(SUPPORTED_WORKERS)}")
    if args.workers != 16 and not args.diagnostic_worker_count:
        parser.error("production revision-306 RL replay packing requires 16 workers")
    if args.exact_card_vocab is None and not args.diagnostic_host:
        parser.error("production RL replay packing requires --exact-card-vocab")
    _corpus, routing, manifest = build_parallel_replay_window_pack(
        args.source,
        args.output,
        component_root=args.component_root,
        workers=int(args.workers),
        max_context=int(args.max_context),
        exact_card_vocab=args.exact_card_vocab,
        force_strategic=bool(args.force_strategic),
        memory_reserve_gib=float(args.memory_reserve_gib),
        semantic_contract={
            "owner_revision": 306,
            "build_host": "inzi",
            "production_workers": 16,
            "batch_size_change_authorized": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "workers": int(args.workers),
                "games": int(manifest["games"]),
                "decisions": int(manifest["decisions"]),
                "adapter_ticketed_games": routing.ticketed_games,
                "adapter_ticketed_decisions": routing.ticketed_decisions,
                "output_digest": manifest["output_digest"],
                "cache_reused": bool(manifest.get("cache_reused")),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
