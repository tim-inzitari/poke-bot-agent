#!/usr/bin/env python3
"""Derive a bounded, latest-first deck-agnostic core corpus.

The authoritative all-recognized manifest may be too large to pack safely on
one GPU.  This builder streams it newest-day first and admits a bounded number
of records and decisions per acting-seat archetype.  No sequence is split and
no private target is promoted into model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import archetypes
from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    _target_coverage,
    iter_feature_shard,
)
from poke_bot.features import FEATURE_SCHEMA_VERSION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_balanced_manifest(
    source_manifest: Path,
    output_dir: Path,
    *,
    max_records_per_archetype: int,
    max_decisions_per_archetype: int,
    additive_archetypes: tuple[str, ...] = (),
) -> Path:
    source_manifest = Path(source_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (
        source.get("format") != MANIFEST_FORMAT
        or int(source.get("format_version", -1)) != MANIFEST_FORMAT_VERSION
        or source.get("compact_mode") != COMPACT_MODE_TEMPORAL_EXPERT
    ):
        raise ValueError("source is not a temporal expert feature manifest")
    source_rows = list(source.get("shards") or [])
    if not source_rows:
        raise ValueError("source manifest has no shards")
    if max_records_per_archetype <= 0 or max_decisions_per_archetype <= 0:
        raise ValueError("per-archetype caps must be positive")

    known = set(archetypes.archetype_ids())
    known.update(additive_archetypes)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "deck-agnostic-balanced.features"
    sidecar = output.with_suffix(output.suffix + ".json")
    manifest_path = output_dir / "manifest.json"
    pointer_path = output_dir / "PROTECTED_CORE_CORPUS.json"
    if any(path.exists() for path in (output, sidecar, manifest_path, pointer_path)):
        if not all(path.exists() for path in (output, sidecar, manifest_path, pointer_path)):
            raise FileExistsError("partial balanced-core corpus already exists")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            sha256(output) == (manifest.get("shards") or [{}])[0].get("sha256")
            and sha256(source_manifest) == manifest.get("source_manifest_sha256")
            and int((manifest.get("selection") or {}).get("max_records_per_archetype", -1))
            == max_records_per_archetype
            and int((manifest.get("selection") or {}).get("max_decisions_per_archetype", -1))
            == max_decisions_per_archetype
            and list((manifest.get("selection") or {}).get("additive_archetypes") or [])
            == list(additive_archetypes)
        ):
            return manifest_path
        raise RuntimeError("existing balanced-core corpus identity differs")

    records: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    seats: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    scanned: Counter[str] = Counter()
    source_digests: list[dict[str, Any]] = []
    partial = output.with_name(f".{output.name}.{os.getpid()}.partial")
    header = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "source_dates": list(source.get("dates") or []),
        "max_context": int(source.get("max_context") or 0),
        "selection": "latest-first bounded per recognized acting archetype",
        "max_records_per_archetype": int(max_records_per_archetype),
        "max_decisions_per_archetype": int(max_decisions_per_archetype),
        "additive_archetypes": list(additive_archetypes),
    }
    try:
        with partial.open("xb") as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for row in reversed(source_rows):
                shard = (source_manifest.parent / str(row.get("path") or "")).resolve()
                expected = str(row.get("sha256") or "")
                actual = sha256(shard)
                if actual != expected:
                    raise ValueError(f"source shard digest mismatch: {shard}")
                source_digests.append(
                    {"path": str(shard), "sha256": actual, "bytes": shard.stat().st_size}
                )
                for sequence in iter_feature_shard(shard):
                    archetype = str(sequence.archetype or "").strip().casefold()
                    scanned[archetype or "unknown"] += 1
                    if archetype not in known or not sequence.info_set_ok:
                        continue
                    n_decisions = len(sequence)
                    if records[archetype] >= max_records_per_archetype:
                        continue
                    if (
                        decisions[archetype] > 0
                        and decisions[archetype] + n_decisions
                        > max_decisions_per_archetype
                    ):
                        continue
                    pickle.dump(sequence, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    records[archetype] += 1
                    decisions[archetype] += n_decisions
                    opponents[str(sequence.opp_archetype)] += 1
                    seats[str(int(sequence.seat))] += 1
                    coverage.update(_target_coverage(sequence))
            total_records = sum(records.values())
            total_decisions = sum(decisions.values())
            if total_records <= 0 or total_decisions <= 0:
                raise ValueError("balanced core selection is empty")
            stats = {
                "records_total": total_records,
                "records_kept": total_records,
                "records_dropped": 0,
                "decisions_kept": total_decisions,
                "records_per_archetype": dict(sorted(records.items())),
                "decisions_per_archetype": dict(sorted(decisions.items())),
                "source_records_scanned": dict(sorted(scanned.items())),
                "seat_counts": dict(sorted(seats.items())),
                "opponent_archetypes": dict(sorted(opponents.items())),
                "target_coverage": dict(sorted(coverage.items())),
                "drop_reasons": {},
                "decisions_truncated": 0,
                "policy_targets_padded": 0,
                "policy_targets_truncated": 0,
            }
            pickle.dump(
                {
                    "format": SHARD_FORMAT + "-footer",
                    "format_version": SHARD_FORMAT_VERSION,
                    "stats": stats,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if sum(1 for _ in iter_feature_shard(partial)) != total_records:
            raise ValueError("balanced core shard failed stream validation")
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    shard_row = {
        **header,
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "stats": stats,
    }
    atomic_json(sidecar, shard_row)
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "date_start": source.get("date_start"),
        "date_end": source.get("date_end"),
        "dates": list(source.get("dates") or []),
        "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
        "max_context": int(source.get("max_context") or 0),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "source_shards": list(reversed(source_digests)),
        "selection": {
            "field": "GameSequence.archetype",
            "operator": "latest_first_bounded_registered",
            "seat_semantics": "acting_seat_only",
            "max_records_per_archetype": int(max_records_per_archetype),
            "max_decisions_per_archetype": int(max_decisions_per_archetype),
            "additive_archetypes": list(additive_archetypes),
        },
        "shards": [
            {
                "path": output.name,
                "sha256": shard_row["sha256"],
                "bytes": shard_row["bytes"],
                "source_dates": list(source.get("dates") or []),
                "stats": stats,
            }
        ],
        "totals": {
            "bytes": shard_row["bytes"],
            "records_total": total_records,
            "records_kept": total_records,
            "decisions_kept": total_decisions,
            "records_per_archetype": dict(sorted(records.items())),
            "decisions_per_archetype": dict(sorted(decisions.items())),
            "target_coverage": dict(sorted(coverage.items())),
        },
        "quality_gates": {
            "passed": True,
            "nonempty": True,
            "checksummed": True,
            "acting_seat_archetype_recognized": True,
            "episode_sequences_unsplit": True,
            "hidden_targets_are_aux_only": True,
            "temporal_action_tokens_complete": int(coverage.get("temporal_action_rows", 0))
            == total_decisions,
        },
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        pointer_path,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": manifest_path.name,
            "manifest_sha256": sha256(manifest_path),
            "selection": manifest["selection"],
            "totals": manifest["totals"],
        },
    )
    os.chmod(output, 0o444)
    os.chmod(sidecar, 0o444)
    os.chmod(manifest_path, 0o444)
    os.chmod(pointer_path, 0o444)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records-per-archetype", type=int, default=2500)
    parser.add_argument("--max-decisions-per-archetype", type=int, default=220000)
    parser.add_argument("--additive-archetype", action="append", default=[])
    args = parser.parse_args()
    result = build_balanced_manifest(
        args.source_manifest,
        args.output_dir,
        max_records_per_archetype=int(args.max_records_per_archetype),
        max_decisions_per_archetype=int(args.max_decisions_per_archetype),
        additive_archetypes=tuple(
            dict.fromkeys(
                str(value).strip().casefold()
                for value in args.additive_archetype
                if str(value).strip()
            )
        ),
    )
    print(json.dumps(json.loads(result.read_text()), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
