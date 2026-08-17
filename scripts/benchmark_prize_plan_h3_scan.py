#!/usr/bin/env python3
"""Benchmark serial versus byte-range H3 replay featurization on one cache part."""

from __future__ import annotations

import argparse
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from poke_bot.prize_plan_live_cache import (
    _scan_range_worker,
    _scan_tasks,
    _scan_worker_init,
    _wanted_actions,
)
from poke_bot.pure_rl.dataset_bridge import (
    _cache_paths,
    _cache_signature,
    _read_cached_part,
    _valid_manifest,
)


def _read_payload(path: Path) -> tuple[set[tuple[str, int, int]], int, int]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return set(payload["seen"]), len(payload["examples"]), int(payload["masked"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=64)
    args = parser.parse_args()
    if args.manifest is None:
        signature = _cache_signature(
            args.shard, verify_info_set=False, max_context=args.max_context
        )
        _cache_dir, manifest_path = _cache_paths(args.shard, signature)
        manifest = _valid_manifest(manifest_path, signature)
    else:
        manifest_path = args.manifest
        with manifest_path.open("r", encoding="utf-8") as handle:
            import json

            manifest = json.load(handle)
    if manifest is None:
        raise SystemExit("validated replay cache is absent")
    first_part = Path(str(manifest["parts"][0]["path"]))
    sequences = _read_cached_part(first_part)
    wanted = _wanted_actions(sequences)
    source_bytes = int(manifest["parts"][0]["bytes"])
    with tempfile.TemporaryDirectory(prefix="h3-scan-benchmark-") as root:
        root_path = Path(root)
        sample = root_path / "sample.jsonl"
        with args.shard.open("rb") as source, sample.open("wb") as output:
            remaining = source_bytes
            while remaining:
                block = source.read(min(4 * 1024 * 1024, remaining))
                if not block:
                    raise SystemExit("source ended before cache-part boundary")
                output.write(block)
                remaining -= len(block)

        _scan_worker_init(wanted)
        started = time.monotonic()
        serial_spool = root_path / "serial.pkl"
        _scan_range_worker(str(sample), 0, sample.stat().st_size, str(serial_spool))
        serial_seconds = time.monotonic() - started
        serial = _read_payload(serial_spool)

        tasks = _scan_tasks([sample], workers=args.workers)
        started = time.monotonic()
        parallel_seen: set[tuple[str, int, int]] = set()
        parallel_examples = 0
        parallel_masked = 0
        import multiprocessing as mp

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_scan_worker_init,
            initargs=(wanted,),
        ) as executor:
            futures = [
                executor.submit(
                    _scan_range_worker,
                    str(path),
                    start,
                    end,
                    str(root_path / f"parallel-{index:03d}.pkl"),
                )
                for index, (path, start, end) in enumerate(tasks)
            ]
            for future in as_completed(futures):
                result = future.result()
                seen, examples, masked = _read_payload(Path(result["spool"]))
                if parallel_seen.intersection(seen):
                    raise SystemExit("parallel ranges overlapped")
                parallel_seen.update(seen)
                parallel_examples += examples
                parallel_masked += masked
        parallel_seconds = time.monotonic() - started
        parallel = (parallel_seen, parallel_examples, parallel_masked)
        if serial != parallel:
            raise SystemExit("serial/parallel membership or count parity failed")
        print(
            {
                "sample_bytes": sample.stat().st_size,
                "wanted_actions": len(wanted),
                "workers": args.workers,
                "serial_seconds": serial_seconds,
                "parallel_seconds": parallel_seconds,
                "speedup": serial_seconds / parallel_seconds,
                "parity": True,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
