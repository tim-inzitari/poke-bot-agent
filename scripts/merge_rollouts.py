#!/usr/bin/env python3
"""Merge multiple rollout JSONL sources into one deck-agnostic training file.

Streams episode-by-episode (low RAM). Parallel workers process episode batches
or whole source files; worker count comes from --workers, TENSOR_BUILD_WORKERS,
or SELF_PLAY_WORKERS (same knobs as tensor build / self-play).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.rewards import is_complete_episode
from poke_agent.rollout_filter import episode_involves_archetype
from tqdm.auto import tqdm

_EPISODE_FIELD_RE = re.compile(r'"episode"\s*:\s*(-?\d+)')


def resolve_workers(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    for key in ("TENSOR_BUILD_WORKERS", "SELF_PLAY_WORKERS"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return max(1, int(raw))
    try:
        from poke_agent.config import build_config
        from poke_agent.features import default_tensor_build_workers
        from poke_agent.paths import resolve_root

        cfg = build_config(resolve_root())
        configured = cfg.get("tensor_build_workers")
        if configured is not None:
            return max(1, int(configured))
        return default_tensor_build_workers()
    except Exception:
        return max(1, (os.cpu_count() or 1) - 2)


def _count_episodes(path: Path) -> int:
    """Fast-ish episode count by scanning episode-id boundaries only."""
    count = 0
    current: int | None = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            match = _EPISODE_FIELD_RE.search(line)
            episode = int(match.group(1)) if match else int(json.loads(line)["episode"])
            if current is not None and episode != current:
                count += 1
            current = episode
    return count + (1 if current is not None else 0)


def _iter_episodes(path: Path, *, desc: str | None = None, total: int | None = None) -> Iterator[list[dict[str, Any]]]:
    """Yield one episode's rows at a time; JSONL must be grouped by episode id."""
    current_episode: int | None = None
    batch: list[dict[str, Any]] = []
    progress = tqdm(
        desc=desc or path.name,
        total=total,
        unit="ep",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                episode = int(row["episode"])
                if current_episode is not None and episode != current_episode:
                    progress.update(1)
                    yield batch
                    batch = []
                current_episode = episode
                batch.append(row)
        if batch:
            progress.update(1)
            yield batch
    finally:
        progress.close()


def dedupe_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("source", "")),
        str(row.get("source_episode_id", row.get("episode", ""))),
        int(row.get("step", 0)),
        int(row.get("player", 0)),
    )


def _episode_complete(episode_rows: list[dict[str, Any]]) -> bool:
    if not episode_rows:
        return False
    ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
    last = ordered[-1]
    raw_result = int(last.get("result", -1))
    terminal_obs = last.get("next_observation") or last.get("observation")
    truncated = bool(last.get("truncated"))
    return is_complete_episode(raw_result, terminal_obs=terminal_obs, truncated=truncated)


def _filter_episode_batch(
    payload: tuple[str, list[list[dict[str, Any]]], bool, str | None],
) -> list[list[dict[str, Any]]]:
    """Worker: filter/annotate a batch of episodes (no global dedupe)."""
    source_stem, episodes, require_complete, require_archetype = payload
    kept: list[list[dict[str, Any]]] = []
    for episode_rows in episodes:
        if require_complete and not _episode_complete(episode_rows):
            continue
        if require_archetype and not episode_involves_archetype(episode_rows, require_archetype):
            continue
        rows: list[dict[str, Any]] = []
        for row in episode_rows:
            tagged = dict(row)
            tagged.setdefault("source", source_stem)
            rows.append(tagged)
        if rows:
            kept.append(sorted(rows, key=lambda item: int(item["step"])))
    return kept


def _stream_source_to_temp(
    path: Path,
    temp_path: Path,
    require_complete: bool,
    require_archetype: str | None = None,
) -> tuple[int, int, int]:
    """Stream one source file to a temp JSONL (episode ids start at 0)."""
    seen: set[tuple[str, str, int, int]] = set()
    next_episode = 0
    total_rows = 0
    skipped = 0
    total_eps = _count_episodes(path)
    with temp_path.open("w", encoding="utf-8") as out:
        for episode_rows in _iter_episodes(path, desc=f"read {path.name}", total=total_eps):
            if require_complete and not _episode_complete(episode_rows):
                skipped += 1
                continue
            if require_archetype and not episode_involves_archetype(episode_rows, require_archetype):
                skipped += 1
                continue
            kept_rows: list[dict[str, Any]] = []
            for row in episode_rows:
                row.setdefault("source", path.stem)
                key = dedupe_key(row)
                if key in seen:
                    continue
                seen.add(key)
                kept_rows.append(row)
            if not kept_rows:
                continue
            for row in sorted(kept_rows, key=lambda item: int(item["step"])):
                row["episode"] = next_episode
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                total_rows += 1
            next_episode += 1
    return next_episode, total_rows, skipped


def _concat_temp_sources(
    temp_paths: list[Path],
    out_path: Path,
) -> tuple[int, int]:
    """Concatenate per-source temp files, reindexing episode ids."""
    seen: set[tuple[str, str, int, int]] = set()
    next_episode = 0
    total_rows = 0
    with out_path.open("w", encoding="utf-8") as out:
        for temp_path in temp_paths:
            total_eps = _count_episodes(temp_path)
            for episode_rows in _iter_episodes(
                temp_path,
                desc=f"concat {temp_path.name}",
                total=total_eps,
            ):
                kept_rows: list[dict[str, Any]] = []
                for row in episode_rows:
                    key = dedupe_key(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    kept_rows.append(row)
                if not kept_rows:
                    continue
                for row in sorted(kept_rows, key=lambda item: int(item["step"])):
                    row["episode"] = next_episode
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    total_rows += 1
                next_episode += 1
    return next_episode, total_rows


def _merge_source_parallel_batches(
    path: Path,
    out_handle: Any,
    *,
    require_complete: bool,
    require_archetype: str | None,
    workers: int,
    seen: set[tuple[str, str, int, int]],
    start_episode: int,
) -> tuple[int, int, int]:
    """Merge one source using a process pool over episode batches."""
    next_episode = start_episode
    total_rows = 0
    skipped = 0
    batch: list[list[dict[str, Any]]] = []
    batch_size = max(8, workers * 2)
    total_eps = _count_episodes(path)

    def flush_batch(pool: ProcessPoolExecutor, episodes: list[list[dict[str, Any]]]) -> None:
        nonlocal next_episode, total_rows, skipped
        if not episodes:
            return
        payload = (path.stem, episodes, require_complete, require_archetype)
        if workers <= 1:
            kept_episodes = _filter_episode_batch(payload)
        else:
            kept_episodes = pool.submit(_filter_episode_batch, payload).result()
        skipped += len(episodes) - len(kept_episodes)
        for episode_rows in kept_episodes:
            kept_rows: list[dict[str, Any]] = []
            for row in episode_rows:
                key = dedupe_key(row)
                if key in seen:
                    continue
                seen.add(key)
                kept_rows.append(row)
            if not kept_rows:
                continue
            for row in kept_rows:
                row["episode"] = next_episode
                out_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                total_rows += 1
            next_episode += 1

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for episode_rows in _iter_episodes(path, desc=f"merge {path.name}", total=total_eps):
            batch.append(episode_rows)
            if len(batch) >= batch_size:
                flush_batch(pool, batch)
                batch = []
        flush_batch(pool, batch)

    return next_episode, total_rows, skipped


def _stream_source_to_temp_task(
    args: tuple[Path, Path, bool, str | None],
) -> tuple[int, int, int]:
    path, temp_path, require_complete, require_archetype = args
    return _stream_source_to_temp(path, temp_path, require_complete, require_archetype)


def merge_sources_to_file(
    paths: list[Path],
    out_path: Path,
    *,
    require_complete: bool = True,
    require_archetype: str | None = None,
    workers: int | None = None,
) -> tuple[int, int]:
    """Stream-merge sources to ``out_path``. Returns (games, rows)."""
    worker_count = resolve_workers(workers)
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"no rollout sources found: {paths}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_episodes = 0

    # Multiple large sources: process each to a temp file in parallel, then concat.
    if len(existing) > 1 and worker_count > 1:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="merge_rollouts_"))
        temp_paths: list[Path] = []
        try:
            tasks = [
                (path, temp_dir / f"{index:03d}_{path.stem}.jsonl", require_complete, require_archetype)
                for index, path in enumerate(existing)
            ]
            file_workers = min(worker_count, len(tasks))
            print(f"merge rollouts: {len(tasks)} sources, {file_workers} file workers")
            with tqdm(total=len(tasks), desc="merge sources", unit="file", dynamic_ncols=True) as file_bar:
                with ProcessPoolExecutor(max_workers=file_workers) as pool:
                    results = list(pool.map(_stream_source_to_temp_task, tasks))
                file_bar.update(len(tasks))
            for (path, temp_path, _, _), (eps, rows, skipped) in zip(tasks, results):
                temp_paths.append(temp_path)
                skipped_episodes += skipped
                print(f"merged {path.name}: {eps:,} episodes, {rows:,} rows")
            games, total_rows = _concat_temp_sources(temp_paths, out_path)
            if require_complete and skipped_episodes:
                print(f"complete games filter: skipped {skipped_episodes:,} incomplete episodes")
            return games, total_rows
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            temp_dir.rmdir()

    # Single source (or workers=1): stream with optional episode-batch parallelism.
    seen: set[tuple[str, str, int, int]] = set()
    next_episode = 0
    total_rows = 0
    print(f"merge rollouts: {worker_count} workers")
    with out_path.open("w", encoding="utf-8") as out:
        for path in existing:
            if worker_count <= 1:
                source_episodes = 0
                source_rows = 0
                total_eps = _count_episodes(path)
                for episode_rows in _iter_episodes(path, desc=f"merge {path.name}", total=total_eps):
                    if require_complete and not _episode_complete(episode_rows):
                        skipped_episodes += 1
                        continue
                    if require_archetype and not episode_involves_archetype(episode_rows, require_archetype):
                        skipped_episodes += 1
                        continue
                    kept_rows: list[dict[str, Any]] = []
                    for row in episode_rows:
                        row.setdefault("source", path.stem)
                        key = dedupe_key(row)
                        if key in seen:
                            continue
                        seen.add(key)
                        kept_rows.append(row)
                    if not kept_rows:
                        continue
                    for row in sorted(kept_rows, key=lambda item: int(item["step"])):
                        row["episode"] = next_episode
                        out.write(json.dumps(row, separators=(",", ":")) + "\n")
                        source_rows += 1
                    next_episode += 1
                    source_episodes += 1
                print(f"merged {path.name}: {source_episodes:,} episodes, {source_rows:,} rows")
                total_rows += source_rows
            else:
                next_episode, source_rows, skipped = _merge_source_parallel_batches(
                    path,
                    out,
                    require_complete=require_complete,
                    require_archetype=require_archetype,
                    workers=worker_count,
                    seen=seen,
                    start_episode=next_episode,
                )
                skipped_episodes += skipped
                print(f"merged {path.name}: {source_rows:,} rows ({worker_count} workers)")

    if require_complete and skipped_episodes:
        print(f"complete games filter: skipped {skipped_episodes:,} incomplete episodes")
    return next_episode, total_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge rollout JSONL files for deck-agnostic training.")
    parser.add_argument("sources", nargs="+", type=Path, help="input JSONL paths")
    parser.add_argument("--out", type=Path, default="data/training_rollouts_merged.jsonl")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="keep truncated/draw/timeout episodes (default: complete decisive games only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers (default: TENSOR_BUILD_WORKERS / SELF_PLAY_WORKERS / config)",
    )
    parser.add_argument(
        "--require-archetype",
        default=None,
        help="keep episodes where either deck matches this heuristic archetype id (e.g. dragapult-ex)",
    )
    args = parser.parse_args()

    games, rows = merge_sources_to_file(
        args.sources,
        args.out,
        require_complete=not args.allow_incomplete,
        require_archetype=args.require_archetype,
        workers=args.workers,
    )
    print(f"merged {len(args.sources)} sources -> {games:,} games / {rows:,} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
