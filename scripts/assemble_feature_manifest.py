#!/usr/bin/env python
"""Verify compact feature shards and atomically assemble a training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.feature_shards import (
    COMPACT_MODE,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    SUPPORTED_COMPACT_MODES,
)
from poke_bot.strategic_heads import (
    StrategicTargetContractError,
    masked_expanded_strategic_coverage,
    merge_expanded_strategic_coverages,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _verified_digest(
    shard: Path,
    sidecar: Path,
    metadata: dict[str, Any],
    verified_dir: Path | None,
) -> str:
    """Hash once on arrival and safely reuse that result during final assembly."""
    expected = str(metadata.get("sha256") or "")
    stat = shard.stat()
    sidecar_digest = _sha256(sidecar)
    cache_path = (
        verified_dir / f"{sidecar.name}.verified.json"
        if verified_dir is not None
        else None
    )
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if (
            cached.get("path") == shard.name
            and cached.get("sha256") == expected
            and cached.get("sidecar_sha256") == sidecar_digest
            and int(cached.get("bytes", -1)) == stat.st_size
            and int(cached.get("ctime_ns", -1)) == stat.st_ctime_ns
        ):
            return expected

    digest = _sha256(shard)
    if digest != expected:
        raise SystemExit(f"digest mismatch: {shard}")
    if cache_path is not None:
        _atomic_json(
            cache_path,
            {
                "path": shard.name,
                "sha256": digest,
                "sidecar_sha256": sidecar_digest,
                "bytes": stat.st_size,
                "ctime_ns": stat.st_ctime_ns,
            },
        )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-date", action="append", required=True)
    parser.add_argument(
        "--only-date",
        action="append",
        default=[],
        help="Verify/assemble only these dates (used for per-shard pipelining).",
    )
    parser.add_argument(
        "--verified-dir",
        type=Path,
        default=None,
        help="Cache post-transfer shard digests for the final manifest pass.",
    )
    parser.add_argument("--min-free-gib", type=float, default=25.0)
    parser.add_argument(
        "--compact-mode",
        choices=sorted(SUPPORTED_COMPACT_MODES),
        default=COMPACT_MODE,
    )
    parser.add_argument(
        "--required-archetype",
        default="",
        help="Require every shard to contain only this acting-seat archetype.",
    )
    parser.add_argument(
        "--expected-max-context",
        type=int,
        default=None,
        help="Require the exact temporal context recorded by every shard.",
    )
    parser.add_argument(
        "--require-target-coverage",
        action="append",
        default=[],
        help="Require this target counter to equal decisions_kept (repeatable).",
    )
    parser.add_argument(
        "--seal-protected",
        action="store_true",
        help="Write an immutable PROTECTED_EXPERT_CORPUS.json beside the manifest.",
    )
    parser.add_argument(
        "--allow-empty-shards",
        action="store_true",
        help=(
            "Count checksum-valid zero-record shards toward expected date coverage "
            "without adding them to the training manifest."
        ),
    )
    args = parser.parse_args()

    staging = args.staging_dir.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging).free
    minimum = int(args.min_free_gib * (1024**3))
    if free < minimum:
        raise SystemExit(
            f"free-space guard failed: {free / 1024**3:.1f} GiB < "
            f"{args.min_free_gib:.1f} GiB"
        )

    expected_dates = list(args.expected_date)
    if len(expected_dates) != len(set(expected_dates)):
        raise SystemExit("duplicate --expected-date values")
    only_dates = set(args.only_date)
    if not only_dates.issubset(set(expected_dates)):
        raise SystemExit("--only-date must be included in --expected-date")
    verified_dir = args.verified_dir.resolve() if args.verified_dir else None
    if verified_dir is not None:
        verified_dir.mkdir(parents=True, exist_ok=True)
    required_archetype = str(args.required_archetype).strip().casefold()
    required_targets = tuple(str(value).strip() for value in args.require_target_coverage)
    if len(required_targets) != len(set(required_targets)) or any(
        not value for value in required_targets
    ):
        raise SystemExit("--require-target-coverage values must be unique/nonempty")
    if args.expected_max_context is not None and int(args.expected_max_context) <= 0:
        raise SystemExit("--expected-max-context must be positive")
    if args.seal_protected and not required_archetype:
        raise SystemExit("--seal-protected requires --required-archetype")
    rows: list[dict] = []
    actual_dates: list[str] = []
    empty_dates: list[str] = []
    target_coverage: Counter[str] = Counter()
    expanded_strategic_targets = masked_expanded_strategic_coverage(0)
    for sidecar in sorted(staging.glob("*.features.json")):
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("format") != SHARD_FORMAT:
            raise SystemExit(f"invalid shard format in {sidecar}")
        if int(metadata.get("format_version", -1)) != SHARD_FORMAT_VERSION:
            raise SystemExit(f"invalid shard version in {sidecar}")
        if metadata.get("compact_mode") != args.compact_mode:
            raise SystemExit(f"invalid compact mode in {sidecar}")
        if required_archetype and str(
            metadata.get("required_archetype") or ""
        ).strip().casefold() != required_archetype:
            raise SystemExit(
                f"acting-seat archetype mismatch in {sidecar}: "
                f"expected={required_archetype!r} "
                f"actual={metadata.get('required_archetype')!r}"
            )
        if args.expected_max_context is not None and int(
            metadata.get("max_context", -1)
        ) != int(args.expected_max_context):
            raise SystemExit(
                f"max-context mismatch in {sidecar}: "
                f"expected={int(args.expected_max_context)} "
                f"actual={metadata.get('max_context')!r}"
            )
        shard = staging / str(metadata.get("path") or "")
        if not shard.is_file():
            raise SystemExit(f"missing shard for {sidecar}: {shard}")
        dates = [str(value) for value in metadata.get("source_dates") or []]
        if only_dates and not only_dates.intersection(dates):
            continue
        if only_dates and not set(dates).issubset(only_dates):
            raise SystemExit(f"selected shard contains unexpected dates: {sidecar}")
        digest = _verified_digest(shard, sidecar, metadata, verified_dir)
        overlap = set(actual_dates).intersection(dates)
        if overlap:
            raise SystemExit(f"overlapping shard dates: {sorted(overlap)}")
        actual_dates.extend(dates)
        stats = dict(metadata.get("stats") or {})
        total = int(stats.get("records_total", 0))
        kept = int(stats.get("records_kept", 0))
        shard_decisions = int(stats.get("decisions_kept", 0))
        if kept <= 0:
            if not args.allow_empty_shards:
                raise SystemExit(f"empty feature shard: {shard}")
            empty_target_coverage = dict(stats.get("target_coverage") or {})
            if (
                total != 0
                or kept != 0
                or shard_decisions != 0
                or any(int(value or 0) != 0 for value in empty_target_coverage.values())
            ):
                raise SystemExit(
                    f"invalid zero-record feature shard metadata: {shard}"
                )
            shard_expanded = stats.get("expanded_strategic_targets")
            if shard_expanded is not None:
                try:
                    normalized_expanded = merge_expanded_strategic_coverages(
                        (shard_expanded,)
                    )
                except StrategicTargetContractError as exc:
                    raise SystemExit(
                        "invalid expanded strategic target coverage in "
                        f"{sidecar}: {exc}"
                    ) from exc
                if int(normalized_expanded["decisions"]) != 0:
                    raise SystemExit(
                        f"zero-record shard has nonzero expanded targets: {sidecar}"
                    )
            empty_dates.extend(dates)
            continue
        if total <= 0 or kept / total < 0.98:
            raise SystemExit(
                f"usable-record gate failed for {shard}: kept={kept} total={total}"
            )
        target_coverage.update(dict(stats.get("target_coverage") or {}))
        shard_expanded = stats.get("expanded_strategic_targets")
        if shard_expanded is None:
            # Legacy/missing target metadata means target absence.  Preserve
            # those rows as masked rather than implying numerical zero labels.
            shard_expanded = masked_expanded_strategic_coverage(
                shard_decisions
            )
        try:
            normalized_expanded = merge_expanded_strategic_coverages(
                (shard_expanded,)
            )
            if int(normalized_expanded["decisions"]) != shard_decisions:
                raise StrategicTargetContractError(
                    "expanded target decisions do not match decisions_kept"
                )
            expanded_strategic_targets = (
                merge_expanded_strategic_coverages(
                    (
                        expanded_strategic_targets,
                        normalized_expanded,
                    )
                )
            )
        except StrategicTargetContractError as exc:
            raise SystemExit(
                f"invalid expanded strategic target coverage in {sidecar}: {exc}"
            ) from exc
        rows.append(
            {
                "path": shard.name,
                "sha256": digest,
                "bytes": shard.stat().st_size,
                "source_dates": dates,
                "stats": stats,
            }
        )

    if sorted(actual_dates) != sorted(expected_dates):
        raise SystemExit(
            f"date coverage mismatch: expected={sorted(expected_dates)} "
            f"actual={sorted(actual_dates)}"
        )
    if not rows:
        raise SystemExit("no feature shard sidecars found")
    rows.sort(key=lambda row: min(row["source_dates"]))
    total_decisions = sum(
        int(row["stats"].get("decisions_kept", 0)) for row in rows
    )
    incomplete_targets = {
        name: int(target_coverage.get(name, 0))
        for name in required_targets
        if int(target_coverage.get(name, 0)) != total_decisions
    }
    if incomplete_targets:
        raise SystemExit(
            "required target coverage is incomplete: "
            f"decisions={total_decisions} coverage={incomplete_targets}"
        )
    payload = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "date_start": min(actual_dates),
        "date_end": max(actual_dates),
        "dates": sorted(actual_dates),
        "compact_mode": args.compact_mode,
        "max_context": (
            int(args.expected_max_context)
            if args.expected_max_context is not None
            else None
        ),
        "empty_dates": sorted(empty_dates),
        "shards": rows,
        "expanded_strategic_targets": expanded_strategic_targets,
        "totals": {
            "bytes": sum(int(row["bytes"]) for row in rows),
            "records_total": sum(
                int(row["stats"].get("records_total", 0)) for row in rows
            ),
            "records_kept": sum(
                int(row["stats"].get("records_kept", 0)) for row in rows
            ),
            "decisions_kept": total_decisions,
            "target_coverage": dict(sorted(target_coverage.items())),
        },
    }
    if required_archetype:
        wildcard = required_archetype == "*"
        payload["selection"] = {
            "field": "GameSequence.archetype",
            "operator": (
                "registered_non_unknown" if wildcard else "exact_casefold"
            ),
            "value": required_archetype,
            "seat_semantics": "acting_seat_only",
        }
        payload["quality_gates"] = {
            "passed": True,
            "nonempty": total_decisions > 0,
            "checksummed": True,
            "acting_seat_archetype_exact": not wildcard,
            "acting_seat_archetype_recognized": wildcard,
            "max_context_exact": args.expected_max_context is not None,
            "required_target_rows_complete": not incomplete_targets,
            "required_target_names": list(required_targets),
            "empty_expected_dates_allowed": bool(args.allow_empty_shards),
            "empty_dates": sorted(empty_dates),
            "hidden_targets_are_aux_only": True,
        }
    out = args.out.resolve()
    if args.seal_protected and out.parent != staging:
        raise SystemExit("a protected manifest must be written inside --staging-dir")
    if args.seal_protected and out.exists():
        if json.loads(out.read_text(encoding="utf-8")) != payload:
            raise SystemExit(f"protected manifest already differs: {out}")
    else:
        _atomic_json(out, payload)
    if args.seal_protected:
        pointer = {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": out.name,
            "manifest_sha256": _sha256(out),
            "selection": payload["selection"],
            "totals": {
                "bytes": int(payload["totals"]["bytes"]),
                "records_kept": int(payload["totals"]["records_kept"]),
                "decisions_kept": int(payload["totals"]["decisions_kept"]),
            },
        }
        pointer_path = out.parent / "PROTECTED_EXPERT_CORPUS.json"
        if pointer_path.exists():
            if json.loads(pointer_path.read_text(encoding="utf-8")) != pointer:
                raise SystemExit(f"protected pointer already differs: {pointer_path}")
        else:
            _atomic_json(pointer_path, pointer)
        print(f"protected_pointer={pointer_path}", flush=True)
    print(json.dumps(payload["totals"], sort_keys=True), flush=True)
    print(f"manifest={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
