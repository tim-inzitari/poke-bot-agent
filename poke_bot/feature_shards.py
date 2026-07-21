"""Bounded, portable bootstrap feature shards.

Large ladder corpora must not be materialized as Python objects while they are
being featurized.  This module converts JSONL records with a bounded process
pool and writes each :class:`~poke_bot.dataset.GameSequence` to an append-only
pickle stream immediately.  Shards can therefore be built on multiple hosts,
validated by digest, and loaded on the trainer only when training starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from array import array
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Optional

from . import features
from .dataset import (
    DATASET_CACHE_SCHEMA_VERSION,
    BootstrapDataset,
    GameSequence,
    convert_record,
)


SHARD_FORMAT = "pokebot-bootstrap-feature-shard"
SHARD_FORMAT_VERSION = 1
MANIFEST_FORMAT = "pokebot-bootstrap-feature-manifest"
MANIFEST_FORMAT_VERSION = 1
COMPACT_MODE = "stateless-core-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _compact_sparse(vector: features.SparseVector) -> None:
    """Replace Python number lists with equivalent fixed-width buffers."""
    vector.index = array("I", vector.index)
    vector.value = array("f", vector.value)
    vector.offset = array("I", vector.offset)


def compact_stateless_sequence(sequence: GameSequence) -> GameSequence:
    """Losslessly compact tensors and drop fields unused by stateless BC.

    The current pure-RL model consumes board/option sparse vectors and hard
    factorized target indices.  Realized previous-action tokens, source text,
    deck copies, and hard-target candidate lists are redundant for that model.
    Soft factorized targets retain their candidate lists for ordering checks.
    """
    sequence.deck = array("I", sequence.deck)  # type: ignore[assignment]
    sequence.source = ""
    sequence.target_provenance = {}
    hard_targets_only = sequence.factorized_policy_targets is None
    for decision in sequence.decisions:
        _compact_sparse(decision.board)
        _compact_sparse(decision.options)
        decision.action = []
        decision.action_token = None
        decision.aux_labels = {}
        if hard_targets_only:
            decision.action_combos = []
        for stage in decision.policy_stages:
            _compact_sparse(stage.options)
            if hard_targets_only:
                stage.action_combos = []
                # Guide labels are aligned to the candidate list and cannot be
                # consumed after stateless expert-shard compaction drops it.
                stage.guide_target_index = -1
                stage.guide_confidence = 0.0
    return sequence


def _convert_raw(
    raw: str,
    max_context: int,
    verify_info_set: bool,
    allowed_sources: tuple[str, ...],
) -> tuple[Optional[GameSequence], Optional[str], dict[str, int]]:
    try:
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise TypeError("record is not an object")
    except Exception:
        return None, "invalid_json", {
            "decisions_truncated": 0,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
        }
    if str(record.get("source") or "") not in allowed_sources:
        return None, "source_date_mismatch", {
            "decisions_truncated": 0,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
        }
    sequence, reason, details = convert_record(
        record,
        max_context=max_context,
        verify_info_set=verify_info_set,
    )
    if sequence is not None:
        compact_stateless_sequence(sequence)
    return sequence, reason, details


def _iter_nonempty_lines(path: Path, max_records: int = 0) -> Iterator[str]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            yield raw
            emitted += 1
            if max_records > 0 and emitted >= max_records:
                return


def _new_stats() -> dict[str, Any]:
    return {
        "records_total": 0,
        "records_kept": 0,
        "records_dropped": 0,
        "decisions_kept": 0,
        "drop_reasons": {},
        "decisions_truncated": 0,
        "policy_targets_padded": 0,
        "policy_targets_truncated": 0,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
        "compact_mode": COMPACT_MODE,
    }


def _account(
    result: tuple[Optional[GameSequence], Optional[str], dict[str, int]],
    stats: dict[str, Any],
) -> Optional[GameSequence]:
    sequence, reason, details = result
    stats["records_total"] += 1
    for key in (
        "decisions_truncated",
        "policy_targets_padded",
        "policy_targets_truncated",
    ):
        stats[key] += int(details.get(key, 0))
    if sequence is None:
        reason = reason or "unknown"
        drops = stats["drop_reasons"]
        drops[reason] = int(drops.get(reason, 0)) + 1
        stats["records_dropped"] += 1
        return None
    stats["records_kept"] += 1
    stats["decisions_kept"] += len(sequence)
    return sequence


def write_feature_shard(
    jsonl_path: Path,
    output_path: Path,
    *,
    source_dates: list[str],
    max_context: int,
    workers: int,
    max_in_flight: int = 0,
    max_records: int = 0,
    verify_info_set: bool = True,
) -> dict[str, Any]:
    """Build one atomic feature stream with bounded parent/worker memory."""
    jsonl_path = Path(jsonl_path).resolve()
    output_path = Path(output_path).resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.partial.{os.getpid()}")
    in_flight = max(workers, max_in_flight or workers * 2)
    stats = _new_stats()
    header = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
        "compact_mode": COMPACT_MODE,
        "source_jsonl": jsonl_path.name,
        "source_jsonl_bytes": jsonl_path.stat().st_size,
        "source_dates": list(source_dates),
        "max_context": int(max_context),
    }
    allowed_sources = tuple(
        f"pokemon-tcg-ai-battle-episodes-{day}" for day in source_dates
    )
    started = time.time()
    from tqdm.auto import tqdm

    lines = iter(_iter_nonempty_lines(jsonl_path, max_records=max_records))
    try:
        with partial.open("xb") as output, ProcessPoolExecutor(
            max_workers=workers
        ) as pool:
            pickle.dump(header, output, protocol=pickle.HIGHEST_PROTOCOL)
            pending: deque[Future] = deque()

            def submit_one() -> bool:
                try:
                    raw = next(lines)
                except StopIteration:
                    return False
                pending.append(
                    pool.submit(
                        _convert_raw,
                        raw,
                        max_context,
                        verify_info_set,
                        allowed_sources,
                    )
                )
                return True

            while len(pending) < in_flight and submit_one():
                pass
            with tqdm(desc=f"feature {jsonl_path.name}", unit="seq") as progress:
                while pending:
                    result = pending.popleft().result()
                    sequence = _account(result, stats)
                    if sequence is not None:
                        pickle.dump(sequence, output, protocol=pickle.HIGHEST_PROTOCOL)
                    progress.update(1)
                    progress.set_postfix(
                        kept=stats["records_kept"],
                        drop=stats["records_dropped"],
                    )
                    submit_one()
            footer = {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            }
            pickle.dump(footer, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(output_path)
        digest = _sha256(output_path)
        metadata = {
            **header,
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": digest,
            "stats": stats,
            "workers": workers,
            "max_in_flight": in_flight,
            "elapsed_seconds": time.time() - started,
        }
        sidecar_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sidecar_tmp.replace(sidecar)
        return metadata
    except BaseException:
        partial.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
        raise


def iter_feature_shard(path: Path) -> Iterator[GameSequence]:
    """Yield a fully validated shard stream, rejecting truncation/trailing data."""
    with Path(path).open("rb") as handle:
        header = pickle.load(handle)
        if not isinstance(header, dict) or header.get("format") != SHARD_FORMAT:
            raise ValueError(f"invalid feature shard header: {path}")
        if int(header.get("format_version", -1)) != SHARD_FORMAT_VERSION:
            raise ValueError(f"unsupported feature shard version: {path}")
        if int(header.get("dataset_schema", -1)) != DATASET_CACHE_SCHEMA_VERSION:
            raise ValueError(f"dataset schema mismatch: {path}")
        if int(header.get("feature_schema", -1)) != features.FEATURE_SCHEMA_VERSION:
            raise ValueError(f"feature schema mismatch: {path}")
        count = 0
        while True:
            try:
                item = pickle.load(handle)
            except EOFError as exc:
                raise ValueError(f"feature shard is missing its footer: {path}") from exc
            if isinstance(item, dict) and item.get("format") == SHARD_FORMAT + "-footer":
                expected = int((item.get("stats") or {}).get("records_kept", -1))
                if expected != count:
                    raise ValueError(
                        f"feature shard count mismatch: expected {expected}, loaded {count}"
                    )
                if handle.read(1):
                    raise ValueError(f"trailing bytes after feature shard footer: {path}")
                return
            if not isinstance(item, GameSequence):
                raise ValueError(f"unexpected feature shard item: {type(item)!r}")
            count += 1
            yield item


def load_feature_manifest(
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> BootstrapDataset:
    """Load ordered compact shards described by a portable JSON manifest."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError("invalid feature manifest format")
    if int(payload.get("format_version", -1)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported feature manifest version")
    shards = list(payload.get("shards") or [])
    if not shards:
        raise ValueError("feature manifest contains no shards")
    sequences: list[GameSequence] = []
    combined = _new_stats()
    combined["records_total"] = 0
    from tqdm.auto import tqdm

    for row in shards:
        path = (manifest_path.parent / str(row["path"])).resolve()
        expected_hash = str(row.get("sha256") or "")
        if verify_hashes and _sha256(path) != expected_hash:
            raise ValueError(f"feature shard digest mismatch: {path}")
        stats = dict(row.get("stats") or {})
        for key in (
            "records_total",
            "records_kept",
            "records_dropped",
            "decisions_kept",
            "decisions_truncated",
            "policy_targets_padded",
            "policy_targets_truncated",
        ):
            combined[key] += int(stats.get(key, 0))
        for reason, count in dict(stats.get("drop_reasons") or {}).items():
            combined["drop_reasons"][reason] = (
                int(combined["drop_reasons"].get(reason, 0)) + int(count)
            )
        expected = int(stats.get("records_kept", 0))
        before = len(sequences)
        for sequence in tqdm(
            iter_feature_shard(path),
            total=expected or None,
            desc=f"load {path.name}",
            unit="seq",
        ):
            sequences.append(sequence)
        if expected and len(sequences) - before != expected:
            raise ValueError(f"manifest count mismatch for {path}")
    return BootstrapDataset(sequences=sequences, conversion_stats=combined)
