"""Convert compact pure-RL shards into :class:`GameSequence` for AWR train."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Optional

from poke_bot import config, features
from poke_bot.dataset import (
    BootstrapDataset,
    DATASET_CACHE_SCHEMA_VERSION,
    DecisionSample,
    GameSequence,
    PolicyStage,
    featurize_step,
)
from poke_bot.pure_rl.shards import CompactDecision, CompactGame, iter_shard_games
from poke_bot.blackwell_heads import attach_blackwell_strategy_labels


COMPACT_CACHE_SCHEMA_VERSION = 2
_STREAM_STAGING_NAME = re.compile(r"^iter_\d{5}\.(\d+)\.\d+$")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    return int(raw)


def _compact_game_from_raw(raw: bytes | str) -> CompactGame:
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise TypeError("compact shard row is not an object")
    decisions = [
        CompactDecision(
            env_step=int(d.get("env_step") or 0),
            selected_index=int(d.get("selected_index") or 0),
            n_options=int(d.get("n_options") or 1),
            action=[int(x) for x in (d.get("action") or [])],
            observation=dict(d.get("observation") or {}),
            aux_labels=dict(d.get("aux_labels") or {}),
        )
        for d in (obj.get("decisions") or [])
    ]
    return CompactGame(
        episode_id=str(obj.get("episode_id") or ""),
        seat=int(obj.get("seat") or 0),
        archetype=str(obj.get("archetype") or ""),
        opp_archetype=str(obj.get("opp_archetype") or ""),
        deck=[int(x) for x in (obj.get("deck") or [])],
        value=float(obj.get("value") or 0.0),
        decisions=decisions,
        source=str(obj.get("source") or "pure_rl"),
        target_provenance=dict(obj.get("target_provenance") or {}),
    )


def _cache_signature(
    path: Path,
    *,
    verify_info_set: bool,
    max_context: int,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": str(path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "verify_info_set": bool(verify_info_set),
        "max_context": int(max_context),
        "compact_cache_schema": COMPACT_CACHE_SCHEMA_VERSION,
        "dataset_cache_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
    }


def _cache_key(signature: dict[str, Any]) -> str:
    raw = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cache_paths(path: Path, signature: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(config.HARDWARE.cache_dir) / "pure_rl_compact"
    run_key = hashlib.sha1(str(path.parent.parent.resolve()).encode()).hexdigest()[:12]
    cache_dir = root / run_key / f"{path.stem}_{_cache_key(signature)}"
    return cache_dir, cache_dir / "manifest.json"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _status_path(shard: Path) -> Optional[Path]:
    if shard.parent.name != "shards":
        return None
    return shard.parent.parent / "replay_window.cache.status.json"


def _write_status(shard: Path, **values: Any) -> None:
    target = _status_path(shard)
    if target is None:
        return
    try:
        _atomic_json(
            target,
            {
                "source_shard": str(shard),
                "updated_at": __import__("time").time(),
                **values,
            },
        )
    except OSError:
        pass


def _range_worker(
    source: str,
    start: int,
    end: int,
    output: str,
    verify_info_set: bool,
    max_context: int,
) -> dict[str, int]:
    """Convert one newline-aligned byte range and write its cache part."""
    sequences: list[GameSequence] = []
    records = 0
    dropped = 0
    bytes_consumed = 0
    with Path(source).open("rb") as handle:
        if start > 0:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        actual_start = handle.tell()
        while handle.tell() < end:
            raw = handle.readline()
            if not raw:
                break
            records += 1
            # Fail closed on corrupt JSON or feature-schema drift.  The old
            # serial loader raised here; parallelism must not silently turn a
            # broken trajectory into a smaller, apparently-valid dataset.
            game = _compact_game_from_raw(raw)
            seq = compact_game_to_sequence(
                game,
                verify_info_set=verify_info_set,
                max_context=max_context,
            )
            if seq is None:
                dropped += 1
            else:
                sequences.append(seq)
        bytes_consumed = max(0, handle.tell() - actual_start)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "sequences": sequences,
                "records": records,
                "dropped": dropped,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(output_path)
    return {
        "records": records,
        "kept": len(sequences),
        "dropped": dropped,
        "bytes": bytes_consumed,
    }


def _read_cached_part(part_path: Path) -> list[GameSequence]:
    """Load one immutable cache part; kept separate for bounded prefetch."""
    with part_path.open("rb") as handle:
        payload = pickle.load(handle)
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        raise TypeError(f"invalid replay-cache payload: {part_path}")
    return sequences


def _load_cached_parts(
    shard: Path,
    manifest: dict[str, Any],
) -> list[GameSequence]:
    sequences: list[GameSequence] = []
    parts = list(manifest.get("parts") or [])
    from tqdm.auto import tqdm

    _write_status(
        shard,
        stage="cache_load",
        parts_complete=0,
        parts_total=len(parts),
        percent=0.0,
    )
    # Pickle decode plus disk reads measured ~30% faster with four threads on
    # Inzi.  Submit only one part per worker at a time: unlike executor.map,
    # this cannot decode the entire 25+ GiB replay cache behind a slow early
    # part and create an unbounded memory spike.  Results are consumed in
    # manifest order so train/validation splitting remains deterministic.
    workers = max(
        1,
        min(
            4,
            len(parts),
            _env_int("PURE_RL_REPLAY_CACHE_LOAD_WORKERS", 4),
        ),
    )

    def consume(index: int, loaded: list[GameSequence]) -> None:
        sequences.extend(loaded)
        _write_status(
            shard,
            stage="cache_load",
            workers=workers,
            parts_complete=index + 1,
            parts_total=len(parts),
            percent=100.0 * (index + 1) / max(1, len(parts)),
            sequences_loaded=len(sequences),
        )

    progress = tqdm(parts, desc=f"replay-cache load {shard.name}", unit="part")
    if workers == 1:
        for index, part in enumerate(progress):
            consume(index, _read_cached_part(Path(str(part["path"]))))
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="replay-cache-load",
        ) as executor:
            pending = {
                index: executor.submit(
                    _read_cached_part, Path(str(parts[index]["path"]))
                )
                for index in range(workers)
            }
            next_submit = workers
            for index, _part in enumerate(progress):
                loaded = pending.pop(index).result()
                consume(index, loaded)
                if next_submit < len(parts):
                    pending[next_submit] = executor.submit(
                        _read_cached_part,
                        Path(str(parts[next_submit]["path"])),
                    )
                    next_submit += 1
    return sequences


def _valid_manifest(
    manifest_path: Path,
    signature: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") != signature:
            return None
        parts = list(manifest.get("parts") or [])
        if not parts or not all(Path(str(part.get("path"))).is_file() for part in parts):
            return None
        return manifest
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def validated_replay_cache_manifest(
    shard: Path,
    *,
    verify_info_set: bool = False,
    max_context: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Return a fully reconciled cache manifest for one immutable shard.

    A completed stream-cache is also the strongest bounded proof that every
    newline-complete source record was parsed and featurized.  Recovery uses
    this helper before preserving a collected shard after a trainer crash.
    Merely finding ``manifest.json`` is not sufficient: all aggregate counters
    and byte coverage must agree with both the current source stat and the
    individual immutable cache parts.
    """
    shard = Path(shard)
    if not shard.is_file():
        return None
    max_ctx = int(
        max_context if max_context is not None else config.MODEL.max_context
    )
    signature = _cache_signature(
        shard,
        verify_info_set=verify_info_set,
        max_context=max_ctx,
    )
    _cache_dir, manifest_path = _cache_paths(shard, signature)
    manifest = _valid_manifest(manifest_path, signature)
    if manifest is None:
        return None
    parts = list(manifest.get("parts") or [])
    try:
        indices = [int(part["index"]) for part in parts]
        covered_bytes = sum(int(part["bytes"]) for part in parts)
        records = sum(int(part["records"]) for part in parts)
        kept = sum(int(part["kept"]) for part in parts)
        dropped = sum(int(part["dropped"]) for part in parts)
    except (KeyError, TypeError, ValueError):
        return None
    if indices != list(range(len(parts))):
        return None
    if covered_bytes != int(signature["source_size"]):
        return None
    if (
        records != int(manifest.get("records", -1))
        or kept != int(manifest.get("sequences", -1))
        or dropped != int(manifest.get("dropped", -1))
        or records != kept + dropped
    ):
        return None
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "covered_bytes": covered_bytes,
    }


def _build_parallel_cache(
    shard: Path,
    *,
    cache_dir: Path,
    signature: dict[str, Any],
    verify_info_set: bool,
    max_context: int,
    workers: int,
) -> dict[str, Any]:
    size = int(signature["source_size"])
    workers = max(1, min(int(workers), 16, max(1, size // (16 * 1024 * 1024))))
    boundaries = [size * index // workers for index in range(workers + 1)]
    part_paths = [cache_dir / f"part_{index:03d}.pkl" for index in range(workers)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    from tqdm.auto import tqdm

    _write_status(
        shard,
        stage="parallel_featurize",
        workers=workers,
        parts_complete=0,
        parts_total=workers,
        percent=0.0,
        bytes_total=size,
    )
    results: list[Optional[dict[str, int]]] = [None] * workers
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                _range_worker,
                str(shard),
                boundaries[index],
                boundaries[index + 1],
                str(part_paths[index]),
                verify_info_set,
                max_context,
            ): index
            for index in range(workers)
        }
        completed_bytes = 0
        with tqdm(
            total=workers,
            desc=f"replay-cache build {shard.name}",
            unit="part",
        ) as progress:
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result
                completed_bytes += int(result.get("bytes", 0))
                progress.update(1)
                done = sum(item is not None for item in results)
                _write_status(
                    shard,
                    stage="parallel_featurize",
                    workers=workers,
                    parts_complete=done,
                    parts_total=workers,
                    percent=100.0 * completed_bytes / max(1, size),
                    bytes_complete=completed_bytes,
                    bytes_total=size,
                )
    if any(item is None for item in results):
        raise RuntimeError("parallel replay cache build lost a worker result")
    parts = [
        {"index": index, "path": str(part_paths[index]), **dict(results[index] or {})}
        for index in range(workers)
    ]
    manifest = {
        "signature": signature,
        "parts": parts,
        "records": sum(int(part["records"]) for part in parts),
        "sequences": sum(int(part["kept"]) for part in parts),
        "dropped": sum(int(part["dropped"]) for part in parts),
    }
    _atomic_json(cache_dir / "manifest.json", manifest)
    return manifest


def _prune_cache_run(cache_dir: Path) -> None:
    keep = max(2, _env_int("PURE_RL_REPLAY_CACHE_KEEP_SHARDS", 4))
    run_root = cache_dir.parent
    try:
        candidates = sorted(
            (path for path in run_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for old in candidates[keep:]:
        try:
            shutil.rmtree(old)
        except OSError:
            pass


def _stream_staging_pid_is_alive(pid: int) -> bool:
    """Conservatively report whether a stream-cache owner may still exist."""
    if int(pid) == os.getpid():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        # Never delete another process's staging merely because its status
        # cannot be inspected under a different user/security context.
        return True
    return True


def _prune_stale_stream_staging() -> dict[str, int]:
    """Remove bounded-cache staging left behind by dead trainer PIDs.

    ``finish`` and ``abort`` already remove their own directory, but SIGKILL,
    OOM termination, or a host reboot cannot run either cleanup path.  Every
    staging leaf is PID-addressed, so a new collection can safely scavenge only
    leaves whose owner no longer exists.  Live/ambiguous owners fail closed.
    """
    root = Path(config.HARDWARE.cache_dir) / "pure_rl_compact_staging"
    removed = 0
    reclaimed = 0
    kept = 0
    try:
        run_roots = list(root.iterdir())
    except OSError:
        return {"removed": 0, "reclaimed_bytes": 0, "kept": 0}
    for run_root in run_roots:
        if not run_root.is_dir():
            continue
        try:
            leaves = list(run_root.iterdir())
        except OSError:
            continue
        for leaf in leaves:
            match = _STREAM_STAGING_NAME.fullmatch(leaf.name)
            if not match or not leaf.is_dir():
                continue
            if _stream_staging_pid_is_alive(int(match.group(1))):
                kept += 1
                continue
            try:
                size = sum(
                    path.stat().st_size
                    for path in leaf.rglob("*")
                    if path.is_file()
                )
                shutil.rmtree(leaf)
            except OSError:
                # Cleanup is an optimization. A permission or transient I/O
                # failure must not prevent training from starting.
                kept += 1
                continue
            removed += 1
            reclaimed += int(size)
        try:
            run_root.rmdir()
        except OSError:
            pass
    if removed:
        print(
            "[pure_rl] stale stream-cache cleanup "
            f"dirs={removed} reclaimed_gib={reclaimed / 2**30:.2f} "
            f"live_or_ambiguous={kept}",
            flush=True,
        )
    return {"removed": removed, "reclaimed_bytes": reclaimed, "kept": kept}


class StreamingReplayCache:
    """Featurize newline-complete shard prefixes while collection continues.

    The collector already receives every remote trajectory as it completes.
    This observer turns durable byte prefixes into bounded on-disk pickle parts;
    it never queues trajectory objects and never reads the writer's partial line.
    A manifest is published only after the final source stat and all parts agree.
    """

    def __init__(
        self,
        shard: Path,
        *,
        verify_info_set: bool,
        max_context: int,
        workers: int,
        chunk_mib: int,
    ) -> None:
        self.shard = Path(shard)
        self.verify_info_set = bool(verify_info_set)
        self.max_context = int(max_context)
        self.workers = max(1, min(16, int(workers)))
        self.chunk_bytes = max(1, int(chunk_mib)) * 1024 * 1024
        run_key = hashlib.sha1(
            str(self.shard.parent.parent.resolve()).encode()
        ).hexdigest()[:12]
        self.staging_dir = (
            Path(config.HARDWARE.cache_dir)
            / "pure_rl_compact_staging"
            / run_key
            / f"{self.shard.stem}.{os.getpid()}.{time.time_ns()}"
        )
        self.staging_dir.mkdir(parents=True, exist_ok=False)
        self._context = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=self._context,
        )
        self._futures: dict[Any, tuple[int, int, int, Path]] = {}
        self._results: dict[int, dict[str, int]] = {}
        self._submitted_end = 0
        self._part_index = 0
        self._closed = False
        self._failed: Optional[BaseException] = None
        self._max_pending = self.workers * 2
        _write_status(
            self.shard,
            stage="streaming_featurize",
            workers=self.workers,
            parts_complete=0,
            parts_total=0,
            bytes_complete=0,
            bytes_total=None,
            percent=None,
        )

    @classmethod
    def maybe_start(
        cls,
        shard: Path,
        *,
        verify_info_set: bool = False,
        max_context: Optional[int] = None,
    ) -> Optional["StreamingReplayCache"]:
        enabled = str(
            os.environ.get("PURE_RL_REPLAY_STREAM_CACHE", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        shard = Path(shard)
        if (
            not enabled
            or shard.parent.name != "shards"
            or not shard.name.startswith("iter_")
        ):
            return None
        _prune_stale_stream_staging()
        max_ctx = int(
            max_context if max_context is not None else config.MODEL.max_context
        )
        workers = _env_int("PURE_RL_REPLAY_FEATURIZE_WORKERS", 8)
        chunk_mib = _env_int("PURE_RL_REPLAY_STREAM_CHUNK_MIB", 64)
        return cls(
            shard,
            verify_info_set=verify_info_set,
            max_context=max_ctx,
            workers=workers,
            chunk_mib=chunk_mib,
        )

    def _publish_status(self, *, final_size: Optional[int] = None) -> None:
        completed_bytes = sum(int(row.get("bytes", 0)) for row in self._results.values())
        denominator = int(final_size or max(self._submitted_end, 1))
        _write_status(
            self.shard,
            stage="streaming_featurize",
            workers=self.workers,
            parts_complete=len(self._results),
            parts_total=self._part_index,
            bytes_complete=completed_bytes,
            bytes_total=final_size,
            percent=(100.0 * completed_bytes / denominator),
        )

    def _collect_done(self) -> bool:
        changed = False
        for future, meta in list(self._futures.items()):
            if not future.done():
                continue
            index, _start, _end, _part = meta
            try:
                self._results[index] = future.result()
            except BaseException as exc:  # surfaced in finish; no silent drops
                self._failed = exc
            del self._futures[future]
            changed = True
        if changed:
            self._publish_status()
        return changed

    def _submit(self, start: int, end: int) -> None:
        if end <= start:
            return
        index = self._part_index
        self._part_index += 1
        part = self.staging_dir / f"part_{index:04d}.pkl"
        future = self._executor.submit(
            _range_worker,
            str(self.shard),
            int(start),
            int(end),
            str(part),
            self.verify_info_set,
            self.max_context,
        )
        self._futures[future] = (index, int(start), int(end), part)
        self._submitted_end = int(end)

    def note_append(self) -> None:
        """Submit a durable newline boundary without blocking the collector."""
        if self._closed or self._failed is not None:
            return
        self._collect_done()
        if len(self._futures) >= self._max_pending:
            return
        try:
            end = int(self.shard.stat().st_size)
        except OSError:
            return
        if end - self._submitted_end >= self.chunk_bytes:
            self._submit(self._submitted_end, end)
            self._publish_status()

    def finish(self) -> Optional[dict[str, Any]]:
        """Close the tail and atomically publish the complete-shard manifest."""
        if self._closed:
            return None
        self._closed = True
        try:
            final_size = int(self.shard.stat().st_size)
            if final_size > self._submitted_end:
                self._submit(self._submitted_end, final_size)
            for future in as_completed(list(self._futures)):
                index, _start, _end, _part = self._futures[future]
                self._results[index] = future.result()
                self._publish_status(final_size=final_size)
            self._futures.clear()
            if self._failed is not None:
                raise self._failed
            self._executor.shutdown(wait=True, cancel_futures=False)

            signature = _cache_signature(
                self.shard,
                verify_info_set=self.verify_info_set,
                max_context=self.max_context,
            )
            cache_dir, manifest_path = _cache_paths(self.shard, signature)
            existing = _valid_manifest(manifest_path, signature)
            if existing is not None:
                shutil.rmtree(self.staging_dir, ignore_errors=True)
                return existing
            cache_dir.mkdir(parents=True, exist_ok=True)
            parts: list[dict[str, Any]] = []
            for index in range(self._part_index):
                source_part = self.staging_dir / f"part_{index:04d}.pkl"
                final_part = cache_dir / f"part_{index:04d}.pkl"
                source_part.replace(final_part)
                result = self._results[index]
                parts.append(
                    {"index": index, "path": str(final_part), **dict(result)}
                )
            covered_bytes = sum(int(part.get("bytes", 0)) for part in parts)
            if covered_bytes != final_size:
                raise RuntimeError(
                    "stream-cache byte coverage mismatch: "
                    f"covered={covered_bytes} source={final_size}"
                )
            manifest = {
                "signature": signature,
                "parts": parts,
                "records": sum(int(part["records"]) for part in parts),
                "sequences": sum(int(part["kept"]) for part in parts),
                "dropped": sum(int(part["dropped"]) for part in parts),
                "stream_built": True,
            }
            _atomic_json(manifest_path, manifest)
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            _prune_cache_run(cache_dir)
            _write_status(
                self.shard,
                stage="stream_cache_ready",
                workers=self.workers,
                parts_complete=len(parts),
                parts_total=len(parts),
                bytes_complete=final_size,
                bytes_total=final_size,
                percent=100.0,
                records=int(manifest["records"]),
                sequences=int(manifest["sequences"]),
            )
            print(
                f"[pure_rl] replay stream-cache ready shard={self.shard.name} "
                f"parts={len(parts)} records={manifest['records']} "
                f"sequences={manifest['sequences']}",
                flush=True,
            )
            return manifest
        except BaseException as exc:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            print(
                f"[pure_rl] WARN replay stream-cache abandoned for "
                f"{self.shard.name}: {type(exc).__name__}: {exc}; "
                "post-collect cache build will retry fail-closed",
                flush=True,
            )
            return None

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        shutil.rmtree(self.staging_dir, ignore_errors=True)

    def __del__(self) -> None:
        # CPython drops the collection frame promptly on an exception; this
        # prevents an optimization pool surviving a failed collect wave.
        try:
            self.abort()
        except Exception:
            pass


def compact_game_to_sequence(
    game: CompactGame,
    *,
    verify_info_set: bool = False,
    max_context: Optional[int] = None,
) -> Optional[GameSequence]:
    """Featurize compact decisions; never attaches soft policy targets."""
    max_ctx = int(
        max_context if max_context is not None else config.MODEL.max_context
    )
    if max_ctx <= 0:
        raise ValueError("max_context must be positive")
    source_decisions = game.decisions[:max_ctx]
    decisions_truncated = max(0, len(game.decisions) - len(source_decisions))
    # Build the complete acting-seat trajectory first.  Scope-B tactical
    # labels depend on later public prize-count observations, while privileged
    # belief labels (when present) remain isolated under aux_labels.
    steps = [
        {
            "observation": d.observation,
            "action": list(d.action),
            "env_step": d.env_step,
            "aux_labels": dict(d.aux_labels or {}),
        }
        for d in source_decisions
    ]
    attach_blackwell_strategy_labels(steps)
    decisions: list[DecisionSample] = []
    for d, step in zip(source_decisions, steps):
        try:
            sample = featurize_step(
                step,
                list(game.deck) if game.deck else [0] * 60,
                verify_info_set=verify_info_set,
            )
        except Exception:
            # Fall back: keep selected_index via synthetic single stage when
            # observation is too thin for full featurization (unit tests).
            if not d.observation:
                continue
            raise
        # Force hard target from compact selected_index when stages exist.
        if sample.policy_stages:
            stages = []
            for i, stage in enumerate(sample.policy_stages):
                idx = int(d.selected_index) if i == 0 else int(stage.target_index)
                n = max(1, len(stage.action_combos))
                idx = max(0, min(idx, n - 1))
                stages.append(
                    PolicyStage(
                        options=stage.options,
                        action_combos=stage.action_combos,
                        target_index=idx,
                        guide_target_index=int(
                            getattr(stage, "guide_target_index", -1)
                        ),
                        guide_confidence=float(
                            getattr(stage, "guide_confidence", 0.0)
                        ),
                    )
                )
            sample.policy_stages = stages
            sample.action_combo_index = int(stages[0].target_index)
        decisions.append(sample)
    if not decisions:
        return None
    return GameSequence(
        episode_id=game.episode_id,
        seat=int(game.seat),
        archetype=game.archetype,
        opp_archetype=game.opp_archetype,
        deck=list(game.deck) if game.deck else [0] * 60,
        value=float(game.value),
        decisions=decisions,
        source=game.source or "pure_rl",
        policy_targets=None,
        factorized_policy_targets=None,
        target_provenance={
            **dict(game.target_provenance),
            "pure_rl": True,
            "soft_policy_targets": False,
            "max_context": max_ctx,
            "decisions_truncated": decisions_truncated,
        },
    )


def dataset_from_shard(
    path,
    *,
    verify_info_set: bool = False,
    max_games: Optional[int] = None,
    max_context: Optional[int] = None,
) -> BootstrapDataset:
    path = Path(path)
    # Small diagnostic/smoke reads preserve their original early-stop
    # semantics and avoid creating misleading partial caches.
    if max_games is not None:
        seqs: list[GameSequence] = []
        for i, game in enumerate(iter_shard_games(path)):
            if i >= max_games:
                break
            seq = compact_game_to_sequence(
                game,
                verify_info_set=verify_info_set,
                max_context=max_context,
            )
            if seq is not None:
                seqs.append(seq)
        return BootstrapDataset(sequences=seqs)

    max_ctx = int(
        max_context if max_context is not None else config.MODEL.max_context
    )
    signature = _cache_signature(
        path,
        verify_info_set=verify_info_set,
        max_context=max_ctx,
    )
    cache_dir, manifest_path = _cache_paths(path, signature)
    manifest = _valid_manifest(manifest_path, signature)
    cache_hit = manifest is not None
    if manifest is None:
        threshold = max(
            0, _env_int("PURE_RL_REPLAY_CACHE_PARALLEL_MIN_MIB", 64)
        ) * 1024 * 1024
        requested_workers = max(
            1, min(16, _env_int("PURE_RL_REPLAY_FEATURIZE_WORKERS", 8))
        )
        workers = requested_workers if path.stat().st_size >= threshold else 1
        started = time.monotonic()
        manifest = _build_parallel_cache(
            path,
            cache_dir=cache_dir,
            signature=signature,
            verify_info_set=verify_info_set,
            max_context=max_ctx,
            workers=workers,
        )
        print(
            f"[pure_rl] replay-cache built shard={path.name} workers={workers} "
            f"records={manifest.get('records')} sequences={manifest.get('sequences')} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    started = time.monotonic()
    seqs = _load_cached_parts(path, manifest)
    _write_status(
        path,
        stage="complete",
        cache_hit=cache_hit,
        percent=100.0,
        parts_complete=len(manifest.get("parts") or []),
        parts_total=len(manifest.get("parts") or []),
        records=int(manifest.get("records") or 0),
        sequences_loaded=len(seqs),
    )
    print(
        f"[pure_rl] replay-cache {'hit' if cache_hit else 'ready'} "
        f"shard={path.name} sequences={len(seqs)} "
        f"load_elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    _prune_cache_run(cache_dir)
    return BootstrapDataset(sequences=seqs)


def _dataset_from_shard_serial(
    path,
    *,
    verify_info_set: bool = False,
    max_context: Optional[int] = None,
) -> BootstrapDataset:
    """Reference path retained for parity tests and emergency diagnosis."""
    seqs: list[GameSequence] = []
    for game in iter_shard_games(path):
        seq = compact_game_to_sequence(
            game,
            verify_info_set=verify_info_set,
            max_context=max_context,
        )
        if seq is not None:
            seqs.append(seq)
    return BootstrapDataset(sequences=seqs)


def iter_sequences_from_shard(
    path,
    *,
    verify_info_set: bool = False,
    max_context: Optional[int] = None,
) -> Iterator[GameSequence]:
    for game in iter_shard_games(path):
        seq = compact_game_to_sequence(
            game,
            verify_info_set=verify_info_set,
            max_context=max_context,
        )
        if seq is not None:
            yield seq
