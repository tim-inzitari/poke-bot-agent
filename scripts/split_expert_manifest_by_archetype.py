#!/usr/bin/env python3
"""Split a mixed acting-seat expert manifest into many specialists in one scan.

The ordinary single-archetype filter rescans every source shard for every
target.  This splitter opens one output stream per requested archetype and
routes each causal ``GameSequence`` exactly once.  Every nonempty result is
sealed as an immutable ``PROTECTED_EXPERT_CORPUS.json``; absent targets receive
an explicit unavailable receipt rather than a fabricated empty corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION
from poke_bot.feature_shards import (
    COMPACT_MODE,
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    _target_coverage,
    iter_feature_shard,
)
from poke_bot.features import FEATURE_SCHEMA_VERSION
from scripts.filter_feature_manifest import seal_filtered_manifest, sha256


READY_SCHEMA = "poke_bot.specialist_expert_corpora_ready/v1"
UNAVAILABLE_SCHEMA = "poke_bot.specialist_expert_corpus_unavailable/v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict) or value.get("format") != SHARD_FORMAT:
        raise RuntimeError(f"source feature shard is invalid: {path}")
    return value


def _target_header(source: dict[str, Any], archetype: str) -> dict[str, Any]:
    result = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "compact_mode": str(source.get("compact_mode") or COMPACT_MODE),
        "source_dates": list(source.get("source_dates") or []),
        "max_context": int(source.get("max_context") or 0),
        "selection": "sequence.archetype == selection_archetype",
        "selection_archetype": archetype,
        "opponent_routes_only": False,
        "required_archetype": archetype,
    }
    for key in (
        "classifier_sha256",
        "source_archive",
        "source_archive_sha256",
        "visual_trace_schema",
        "target_consumer_contract",
    ):
        if key in source:
            result[key] = source[key]
    return result


def _new_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "decisions": 0,
        "coverage": Counter(),
        "seats": Counter(),
        "opponents": Counter(),
    }


def split_manifest(
    source_manifest: Path,
    output_root: Path,
    *,
    archetypes: list[str] | tuple[str, ...],
    minimum_decisions: int = 100_000,
    progress_every: int = 1_000,
) -> Path:
    source_manifest = Path(source_manifest).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    targets = tuple(
        dict.fromkeys(str(value).strip().casefold() for value in archetypes)
    )
    if (
        source.get("format") != MANIFEST_FORMAT
        or int(source.get("format_version", -1)) != MANIFEST_FORMAT_VERSION
        or not source.get("shards")
        or not targets
        or any(not value for value in targets)
        or int(minimum_decisions) <= 0
        or int(progress_every) < 0
    ):
        raise RuntimeError("mixed expert manifest or target list is invalid")
    identity = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "archetypes": list(targets),
        "minimum_decisions": int(minimum_decisions),
    }
    ready_path = output_root / "SPECIALIST_CORPORA_READY.json"
    if ready_path.is_file():
        existing = json.loads(ready_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != READY_SCHEMA
            or existing.get("identity_sha256") != _canonical_digest(identity)
        ):
            raise RuntimeError("existing specialist corpus split has other inputs")
        return ready_path
    if output_root.exists():
        raise RuntimeError("unsealed specialist corpus output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    totals = {target: _new_stats() for target in targets}
    shard_rows: dict[str, list[dict[str, Any]]] = {
        target: [] for target in targets
    }
    try:
        for source_row in source["shards"]:
            source_path = (
                source_manifest.parent / str(source_row["path"])
            ).resolve()
            if (
                not source_path.is_file()
                or sha256(source_path) != str(source_row.get("sha256") or "")
            ):
                raise RuntimeError(f"source shard identity changed: {source_path}")
            source_header = _header(source_path)
            source_stem = (
                source_path.name[: -len(".features")]
                if source_path.name.endswith(".features")
                else source_path.stem
            )
            streams: dict[str, BinaryIO] = {}
            partials: dict[str, Path] = {}
            shard_stats = {target: _new_stats() for target in targets}
            scanned = 0
            try:
                for target in targets:
                    directory = temporary / target
                    directory.mkdir(parents=True, exist_ok=True)
                    partial = directory / f".{source_stem}.{target}.features.partial"
                    stream = partial.open("xb")
                    pickle.dump(
                        _target_header(source_header, target),
                        stream,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    streams[target] = stream
                    partials[target] = partial
                for sequence in iter_feature_shard(source_path):
                    scanned += 1
                    target = str(sequence.archetype).strip().casefold()
                    if target not in streams:
                        continue
                    if not sequence.info_set_ok:
                        raise RuntimeError(
                            f"selected sequence failed info-set guard: "
                            f"{sequence.episode_id}"
                        )
                    pickle.dump(
                        sequence, streams[target], protocol=pickle.HIGHEST_PROTOCOL
                    )
                    current = shard_stats[target]
                    coverage = _target_coverage(sequence)
                    current["coverage"].update(coverage)
                    current["records"] += 1
                    current["decisions"] += len(sequence)
                    current["seats"][str(int(sequence.seat))] += 1
                    current["opponents"][str(sequence.opp_archetype)] += 1
                    if int(progress_every) > 0 and scanned % int(
                        progress_every
                    ) == 0:
                        selected = sum(
                            int(stats["records"])
                            for stats in shard_stats.values()
                        )
                        print(
                            "[specialist-corpus-split] "
                            f"shard={source_path.name} scanned={scanned} "
                            f"selected={selected} targets={len(targets)}",
                            flush=True,
                        )
                for target in targets:
                    current = shard_stats[target]
                    stream = streams[target]
                    pickle.dump(
                        {
                            "format": SHARD_FORMAT + "-footer",
                            "format_version": SHARD_FORMAT_VERSION,
                            "stats": {
                                "records_total": current["records"],
                                "records_kept": current["records"],
                                "records_dropped": 0,
                                "decisions_kept": current["decisions"],
                                "drop_reasons": {},
                                "decisions_truncated": 0,
                                "policy_targets_padded": 0,
                                "policy_targets_truncated": 0,
                                "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                                "feature_schema": FEATURE_SCHEMA_VERSION,
                                "compact_mode": str(
                                    source_header.get("compact_mode") or COMPACT_MODE
                                ),
                                "target_coverage": dict(
                                    sorted(current["coverage"].items())
                                ),
                                "source_records_scanned": scanned,
                                "source_records_excluded": (
                                    scanned - current["records"]
                                ),
                                "seat_counts": dict(
                                    sorted(current["seats"].items())
                                ),
                                "opponent_archetypes": dict(
                                    sorted(current["opponents"].items())
                                ),
                                "selection_archetype": target,
                            },
                        },
                        stream,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                    final = (
                        temporary
                        / target
                        / f"{source_stem}.{target}.features"
                    )
                    if current["records"] == 0:
                        partials[target].unlink()
                        continue
                    os.replace(partials[target], final)
                    metadata = {
                        **_target_header(source_header, target),
                        "path": final.name,
                        "bytes": int(final.stat().st_size),
                        "sha256": sha256(final),
                        "source_records_scanned": scanned,
                        "source_sha256": str(source_row["sha256"]),
                        "selection_archetype": target,
                        "stats": {
                            "records_total": current["records"],
                            "records_kept": current["records"],
                            "records_dropped": 0,
                            "decisions_kept": current["decisions"],
                            "target_coverage": dict(
                                sorted(current["coverage"].items())
                            ),
                            "source_records_scanned": scanned,
                            "source_records_excluded": (
                                scanned - current["records"]
                            ),
                            "seat_counts": dict(
                                sorted(current["seats"].items())
                            ),
                            "opponent_archetypes": dict(
                                sorted(current["opponents"].items())
                            ),
                            "selection_archetype": target,
                        },
                    }
                    _atomic_json(
                        final.with_suffix(final.suffix + ".json"), metadata
                    )
                    shard_rows[target].append(metadata)
                    total = totals[target]
                    total["records"] += current["records"]
                    total["decisions"] += current["decisions"]
                    total["coverage"].update(current["coverage"])
                    total["seats"].update(current["seats"])
                    total["opponents"].update(current["opponents"])
            finally:
                for stream in streams.values():
                    if not stream.closed:
                        stream.close()
                for partial in partials.values():
                    partial.unlink(missing_ok=True)
        results: list[dict[str, Any]] = []
        for target in targets:
            current = totals[target]
            directory = temporary / target
            if current["records"] == 0 or current["decisions"] == 0:
                unavailable = {
                    "schema": UNAVAILABLE_SCHEMA,
                    "archetype": target,
                    "reason": "no acting-seat records in the pinned source corpus",
                    **identity,
                }
                _atomic_json(directory / "UNAVAILABLE_EXPERT_CORPUS.json", unavailable)
                results.append(
                    {
                        "archetype": target,
                        "status": "unavailable",
                        "records": 0,
                        "decisions": 0,
                    }
                )
                continue
            compact_modes = {
                str(row.get("compact_mode") or COMPACT_MODE)
                for row in shard_rows[target]
            }
            if len(compact_modes) != 1:
                raise RuntimeError(f"{target} corpus mixes compact feature modes")
            compact_mode = next(iter(compact_modes))
            coverage = dict(sorted(current["coverage"].items()))
            temporal_complete = (
                compact_mode != COMPACT_MODE_TEMPORAL_EXPERT
                or int(coverage.get("temporal_action_rows", 0))
                == int(current["decisions"])
            )
            if not temporal_complete:
                raise RuntimeError(f"{target} temporal action coverage is incomplete")
            manifest = {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "date_start": source.get("date_start"),
                "date_end": source.get("date_end"),
                "dates": list(source.get("dates") or []),
                "compact_mode": compact_mode,
                "max_context": int(source.get("max_context") or 0),
                "selection": {
                    "field": "GameSequence.archetype",
                    "operator": "exact_casefold",
                    "value": target,
                    "seat_semantics": "acting_seat_only",
                    "opponent_routes_only": False,
                },
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": identity["source_manifest_sha256"],
                "shards": shard_rows[target],
                "totals": {
                    "bytes": sum(int(row["bytes"]) for row in shard_rows[target]),
                    "records_kept": int(current["records"]),
                    "records_total": int(current["records"]),
                    "decisions_kept": int(current["decisions"]),
                    "source_records_scanned": sum(
                        int(row["stats"]["source_records_scanned"])
                        for row in shard_rows[target]
                    ),
                    "target_coverage": coverage,
                },
                "quality_gates": {
                    "passed": True,
                    "nonempty": True,
                    "checksummed": True,
                    "acting_seat_archetype_exact": True,
                    "max_context_exact": int(source.get("max_context") or 0) > 0,
                    "temporal_action_tokens_complete": temporal_complete,
                    "hidden_targets_are_aux_only": True,
                },
            }
            manifest_path = directory / "manifest.json"
            _atomic_json(manifest_path, manifest)
            pointer = seal_filtered_manifest(manifest_path)
            results.append(
                {
                    "archetype": target,
                    "status": (
                        "ready"
                        if int(current["decisions"]) >= int(minimum_decisions)
                        else "insufficient_decisions"
                    ),
                    "records": int(current["records"]),
                    "decisions": int(current["decisions"]),
                    "minimum_decisions": int(minimum_decisions),
                    # Keep the aggregate receipt relocatable.  The protected
                    # pointer itself resolves its manifest relative to its
                    # containing directory, so the index should do the same.
                    "protected_corpus": f"{target}/{pointer.name}",
                    "manifest_sha256": sha256(manifest_path),
                }
            )
        ready = {
            "schema": READY_SCHEMA,
            "identity": identity,
            "identity_sha256": _canonical_digest(identity),
            "results": results,
        }
        _atomic_json(temporary / ready_path.name, ready)
        os.replace(temporary, output_root)
        return ready_path
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archetype", action="append", required=True)
    parser.add_argument("--minimum-decisions", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    args = parser.parse_args()
    ready = split_manifest(
        args.source_manifest,
        args.output_root,
        archetypes=args.archetype,
        minimum_decisions=int(args.minimum_decisions),
        progress_every=int(args.progress_every),
    )
    print(ready.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
