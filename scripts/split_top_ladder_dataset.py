#!/usr/bin/env python
"""Split one validated top-ladder JSONL corpus into bounded date shards.

The replay converter deliberately emits one record at a time, but bootstrap
feature construction still materializes a shard in memory.  This helper keeps
larger replay pulls useful without reintroducing the old host-RAM spike: it
streams the parent JSONL once, routes each record by its source date, and
writes independently checksummed/quality-gated child artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _source_date(source: str) -> str:
    match = DATE_RE.search(str(source))
    if match is None:
        raise ValueError(f"source has no terminal YYYY-MM-DD date: {source!r}")
    return match.group(1)


@dataclass
class _Shard:
    start_date: str
    end_date: str
    output: Path
    partial: Path
    handle: TextIO | None = None
    records: int = 0
    decisions: int = 0
    dropped_frames: int = 0
    info_set_failures: int = 0
    archetypes: Counter[str] = field(default_factory=Counter)
    records_by_source: Counter[str] = field(default_factory=Counter)

    def accepts(self, date: str) -> bool:
        return self.start_date <= date <= self.end_date

    def add(self, record: dict[str, Any], raw_line: str) -> None:
        assert self.handle is not None
        self.handle.write(raw_line)
        self.records += 1
        self.decisions += int(record.get("n_decisions") or 0)
        self.dropped_frames += int(
            record.get("dropped_incompatible_action_frames") or 0
        )
        if not bool(record.get("info_set_ok", False)):
            self.info_set_failures += 1
        self.archetypes[str(record.get("archetype") or "unknown")] += 1
        self.records_by_source[str(record.get("source") or "")] += 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--shard",
        nargs=3,
        metavar=("START_DATE", "END_DATE", "OUTPUT"),
        action="append",
        required=True,
        help="Repeat for each inclusive, non-overlapping child date range.",
    )
    parser.add_argument("--min-sequences", type=int, default=1)
    parser.add_argument("--min-recognized-seat-frac", type=float, default=0.90)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def _build_shards(args: argparse.Namespace) -> list[_Shard]:
    shards: list[_Shard] = []
    seen_outputs: set[Path] = set()
    for raw_start, raw_end, raw_output in args.shard:
        start, end = str(raw_start), str(raw_end)
        if start > end:
            raise SystemExit(f"shard start is after end: {start}..{end}")
        output = Path(raw_output).expanduser().resolve()
        if output in seen_outputs:
            raise SystemExit(f"duplicate shard output: {output}")
        seen_outputs.add(output)
        for prior in shards:
            if not (end < prior.start_date or start > prior.end_date):
                raise SystemExit(
                    f"overlapping shard ranges: {start}..{end} and "
                    f"{prior.start_date}..{prior.end_date}"
                )
        if output.exists() and not args.replace:
            raise SystemExit(f"output exists (use --replace): {output}")
        partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
        shards.append(_Shard(start, end, output, partial))
    return shards


def _validate_shard(
    shard: _Shard,
    *,
    parent_meta: dict[str, Any],
    min_sequences: int,
    min_recognized_frac: float,
) -> float:
    """Run every quality gate before any partial file replaces a final artifact."""
    active = set(
        str(value)
        for value in (parent_meta.get("classifier") or {}).get(
            "active_deck_ids", []
        )
    )
    recognized = sum(
        count for archetype, count in shard.archetypes.items() if archetype in active
    )
    recognized_frac = recognized / max(1, shard.records)
    failures: list[str] = []
    if shard.records < min_sequences:
        failures.append(f"records {shard.records} < minimum {min_sequences}")
    if recognized_frac < min_recognized_frac:
        failures.append(
            f"recognized seat fraction {recognized_frac:.4f} < "
            f"minimum {min_recognized_frac:.4f}"
        )
    if shard.info_set_failures:
        failures.append(f"info-set failures={shard.info_set_failures}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return recognized_frac


def _write_meta(
    shard: _Shard,
    *,
    parent: Path,
    parent_meta: dict[str, Any],
    parent_digest: str,
    min_sequences: int,
    min_recognized_frac: float,
    recognized_frac: float,
) -> None:
    parent_sources = list(parent_meta.get("sources") or [])
    sources = [
        row
        for row in parent_sources
        if shard.start_date <= str(row.get("date") or "") <= shard.end_date
    ]
    meta = {
        "schema": "poke_bot.top_ladder_bootstrap/v1",
        "output": str(shard.output),
        "output_bytes": shard.output.stat().st_size,
        "output_sha256": _sha256(shard.output),
        "sources": sources,
        "classifier": dict(parent_meta.get("classifier") or {}),
        "policy_scope": str(
            parent_meta.get("policy_scope") or "all_valid_top_ladder_seats"
        ),
        "split_provenance": {
            "parent_dataset": str(parent),
            "parent_sha256": parent_digest,
            "start_date": shard.start_date,
            "end_date": shard.end_date,
        },
        "quality_gates": {
            "min_sequences": min_sequences,
            "min_recognized_seat_frac": min_recognized_frac,
            "recognized_seat_frac": recognized_frac,
            "passed": True,
        },
        "stats": {
            "records_written": shard.records,
            "decisions_written": shard.decisions,
            "dropped_incompatible_action_frames": shard.dropped_frames,
            "info_set_failures": shard.info_set_failures,
            "record_archetypes": dict(shard.archetypes),
            "records_by_source": dict(shard.records_by_source),
        },
    }
    meta_path = shard.output.with_suffix(".meta.json")
    tmp_meta = meta_path.with_name(f".{meta_path.name}.tmp.{os.getpid()}")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_meta, meta_path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.min_sequences <= 0:
        raise SystemExit("--min-sequences must be positive")
    if not 0.0 <= args.min_recognized_seat_frac <= 1.0:
        raise SystemExit("--min-recognized-seat-frac must be in [0, 1]")

    dataset = args.dataset.expanduser().resolve()
    meta_path = dataset.with_suffix(".meta.json")
    if not dataset.is_file() or not meta_path.is_file():
        raise SystemExit(f"dataset/meta pair is incomplete: {dataset}")
    parent_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parent_digest = _sha256(dataset)
    if parent_digest != str(parent_meta.get("output_sha256") or ""):
        raise RuntimeError("parent dataset checksum does not match its metadata")
    if not bool((parent_meta.get("quality_gates") or {}).get("passed", False)):
        raise RuntimeError("parent dataset did not pass its quality gates")

    shards = _build_shards(args)
    source_dates = {
        str(row.get("date") or "") for row in (parent_meta.get("sources") or [])
    }
    uncovered_dates = sorted(
        date for date in source_dates if not any(shard.accepts(date) for shard in shards)
    )
    if uncovered_dates:
        raise RuntimeError(f"parent source dates are not assigned: {uncovered_dates}")

    total_records = 0
    try:
        for shard in shards:
            shard.output.parent.mkdir(parents=True, exist_ok=True)
            shard.handle = shard.partial.open("w", encoding="utf-8")
        with dataset.open("r", encoding="utf-8") as source_handle:
            for line_number, raw_line in enumerate(source_handle, 1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                date = _source_date(str(record.get("source") or ""))
                destinations = [shard for shard in shards if shard.accepts(date)]
                if len(destinations) != 1:
                    raise RuntimeError(
                        f"line {line_number} date {date} has "
                        f"{len(destinations)} destinations"
                    )
                destinations[0].add(record, raw_line)
                total_records += 1

        expected_records = int(
            (parent_meta.get("stats") or {}).get("records_written") or 0
        )
        if expected_records and total_records != expected_records:
            raise RuntimeError(
                f"routed records {total_records} != parent {expected_records}"
            )
        for shard in shards:
            assert shard.handle is not None
            shard.handle.flush()
            os.fsync(shard.handle.fileno())
            shard.handle.close()
            shard.handle = None

        # Fail the whole split before replacing any destination.  In
        # particular, a too-small or low-quality later shard must not leave an
        # apparently complete earlier child behind.
        recognized_fractions = {
            shard.output: _validate_shard(
                shard,
                parent_meta=parent_meta,
                min_sequences=int(args.min_sequences),
                min_recognized_frac=float(args.min_recognized_seat_frac),
            )
            for shard in shards
        }
        for shard in shards:
            os.replace(shard.partial, shard.output)
            _write_meta(
                shard,
                parent=dataset,
                parent_meta=parent_meta,
                parent_digest=parent_digest,
                min_sequences=int(args.min_sequences),
                min_recognized_frac=float(args.min_recognized_seat_frac),
                recognized_frac=recognized_fractions[shard.output],
            )
            print(
                f"[ready] {shard.start_date}..{shard.end_date} "
                f"records={shard.records} decisions={shard.decisions} "
                f"output={shard.output}",
                flush=True,
            )
        return 0
    finally:
        for shard in shards:
            if shard.handle is not None:
                shard.handle.close()
            if shard.partial.exists():
                shard.partial.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
