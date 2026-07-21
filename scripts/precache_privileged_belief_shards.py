#!/usr/bin/env python
"""Build exact-replay feature caches in parallel before Blackwell training."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import features
from poke_bot.dataset import (
    DATASET_CACHE_SCHEMA_VERSION,
    BootstrapDataset,
    _cache_key,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _cache_path(shard: Path, cache_dir: Path, max_context: int) -> Path:
    key = _cache_key(shard, max_context, True, 0)
    return cache_dir / f"bootstrap_{shard.stem}_{key}.pkl"


def _build_one(
    shard_text: str,
    expected_sha256: str,
    cache_dir_text: str,
    max_context: int,
) -> dict[str, Any]:
    # Avoid each featurizer process starting a large native thread pool.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    shard = Path(shard_text)
    cache_dir = Path(cache_dir_text)
    actual_sha256 = _sha256(shard)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"canonical checksum mismatch: {shard}")
    dataset = BootstrapDataset.from_jsonl(
        shard,
        max_context=max_context,
        verify_info_set=True,
        use_cache=True,
        cache_dir=cache_dir,
    )
    if not dataset.info_set_ok_all or dataset.n_decisions <= 0:
        raise RuntimeError(f"invalid or empty feature cache: {shard}")
    cache_path = _cache_path(shard, cache_dir, max_context)
    if not cache_path.is_file() or cache_path.stat().st_size <= 0:
        raise RuntimeError(f"feature cache was not published: {cache_path}")
    return {
        "shard": str(shard),
        "cache": str(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "games": len(dataset),
        "decisions": dataset.n_decisions,
    }


def main() -> int:
    args = _parse_args()
    if args.workers < 1 or args.max_context < 1:
        raise ValueError("workers and max-context must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    status_path = args.status_json.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "poke_bot.privileged_belief_corpus/v1":
        raise RuntimeError("unsupported privileged-belief corpus manifest")
    if manifest.get("storage_authority") != "inzi":
        raise RuntimeError("feature caches must be built from Inzi authority")
    rows = list(manifest.get("shards") or [])
    expected_count = int((manifest.get("totals") or {}).get("shards", -1))
    if not rows or len(rows) != expected_count:
        raise RuntimeError("manifest shard count mismatch")

    shards = [
        (manifest_path.parent / str(row["path"])).resolve() for row in rows
    ]
    expected_caches = [
        _cache_path(shard, cache_dir, int(args.max_context)) for shard in shards
    ]
    marker_contract = {
        "corpus_digest": manifest["corpus_digest"],
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
        "max_context": int(args.max_context),
        "shards": len(shards),
    }
    if status_path.is_file():
        try:
            old = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if (
            old.get("status") == "complete"
            and all(old.get(key) == value for key, value in marker_contract.items())
            and all(path.is_file() and path.stat().st_size > 0 for path in expected_caches)
        ):
            print(
                f"[belief-cache] {len(shards)}/{len(shards)} caches already ready",
                flush=True,
            )
            return 0

    started = time.time()
    completed: list[dict[str, Any]] = []
    base_status = {
        "schema": "poke_bot.privileged_belief_cache_status/v1",
        "status": "building",
        **marker_contract,
        "workers": int(args.workers),
        "completed_shards": 0,
        "started_unix": int(started),
        "updated_unix": int(started),
    }
    _atomic_json(status_path, base_status)
    print(
        f"[belief-cache] building {len(rows)} caches with {args.workers} workers",
        flush=True,
    )

    # Spawn keeps CUDA state out of workers when this is launched immediately
    # before the GPU trainer in the same service.
    context = multiprocessing.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.workers), mp_context=context
        ) as pool:
            futures = {
                pool.submit(
                    _build_one,
                    str(shard),
                    str(row["sha256"]),
                    str(cache_dir),
                    int(args.max_context),
                ): shard
                for row, shard in zip(rows, shards)
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed.append(result)
                now = time.time()
                done = len(completed)
                _atomic_json(
                    status_path,
                    {
                        **base_status,
                        "completed_shards": done,
                        "decisions": sum(int(row["decisions"]) for row in completed),
                        "cache_bytes": sum(int(row["cache_bytes"]) for row in completed),
                        "elapsed_seconds": now - started,
                        "updated_unix": int(now),
                    },
                )
                print(
                    f"[belief-cache] {done}/{len(rows)} "
                    f"{Path(result['shard']).parent.name}/{Path(result['shard']).name} "
                    f"decisions={result['decisions']}",
                    flush=True,
                )
    except BaseException as exc:
        _atomic_json(
            status_path,
            {
                **base_status,
                "status": "failed",
                "completed_shards": len(completed),
                "error": repr(exc),
                "updated_unix": int(time.time()),
            },
        )
        raise

    finished = time.time()
    final = {
        **base_status,
        "status": "complete",
        "completed_shards": len(completed),
        "games": sum(int(row["games"]) for row in completed),
        "decisions": sum(int(row["decisions"]) for row in completed),
        "cache_bytes": sum(int(row["cache_bytes"]) for row in completed),
        "elapsed_seconds": finished - started,
        "updated_unix": int(finished),
    }
    if final["decisions"] != int((manifest.get("totals") or {})["decisions"]):
        raise RuntimeError("feature-cache decision total differs from manifest")
    _atomic_json(status_path, final)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
