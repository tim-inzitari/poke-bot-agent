#!/usr/bin/env python3
"""Seal completed Archaludon schema-7 days into an immutable training corpus."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--cg-runtime", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--minimum-records", type=int, default=16_639)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    os.environ["CG_LIB_PATH"] = str(args.cg_runtime.resolve())
    from scripts import materialize_archaludon_ex_full_public_schema7_corpus as full

    source_snapshot = full._validate_source_snapshot(
        source_root,
        args.assembler.resolve(),
        args.cg_runtime.resolve(),
        args.source_lock.resolve(),
    )
    status = full._read_json(args.identity_root / "status/window.json")
    completed = list(status.get("completed_days") or ())
    if not completed:
        raise RuntimeError("stored Archaludon corpus has no completed days")
    dates = [str(row["date"]) for row in completed]
    expected_dates = full._days(
        date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
    )
    records = sum(int(row["records"]) for row in completed)
    if dates != expected_dates or records < args.minimum_records:
        raise RuntimeError(
            f"stored corpus is not contiguous or below floor: {records}"
        )

    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"stored corpus output already exists: {output}")
    output.mkdir(parents=True)
    for day in dates:
        feature, sidecar, receipt = full._new_paths(args.identity_root, day)
        for source in (feature, sidecar, receipt):
            if not source.is_file():
                raise RuntimeError(f"completed stored shard is missing: {source}")
            destination = output / source.name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

    sources = full._read_json(args.identity_root / "SOURCE_ARCHIVES.json")
    sources["archives"] = [
        row for row in sources.get("archives") or () if row.get("date") in dates
    ]
    sources["date_start"] = dates[0]
    sources["date_end"] = dates[-1]
    sources["days"] = len(dates)
    sources["materialization_dates"] = dates
    sources["owner_override"] = {
        "goal_revision": 68,
        "mode": "completed_stored_days_only",
        "unmaterialized_or_partial_days_excluded": True,
    }
    _atomic_json(output / "SOURCE_ARCHIVES.json", sources)

    full.END = date.fromisoformat(dates[-1])
    run_args = SimpleNamespace(
        output_root=output,
        assembler=args.assembler.resolve(),
        managed_service=(
            "pokebot-archaludon-ex-stored-schema7-r68-finalize.service"
        ),
        day_parallelism=0,
        workers_per_day=0,
        runtime_memory_floor_gib=0.0,
    )
    full._assemble(run_args)
    ready = full._validate_and_seal(run_args, sources, source_snapshot)
    ready["corpus_kind"] = "stored_completed_days_schema7_owner_r68"
    ready["owner_override"] = {
        "goal_revision": 68,
        "date_start": dates[0],
        "date_end": dates[-1],
        "days": len(dates),
        "records": int(ready["records"]),
        "partial_or_in_progress_days_excluded": True,
    }
    ready_path = output / "ARCHALUDON_EX_FULL_PUBLIC_CORPUS_READY.json"
    _atomic_json(ready_path, ready)
    full._make_read_only(output)
    print(json.dumps(ready, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
