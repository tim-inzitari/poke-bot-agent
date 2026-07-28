#!/usr/bin/env python
"""Atomically extract exact official source days from a combined JSONL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--day", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.jsonl.resolve()
    if not source.is_file():
        raise SystemExit(f"missing source JSONL: {source}")
    days = list(args.day)
    if len(days) != len(set(days)):
        raise SystemExit("duplicate --day values")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    markers = {
        day: f'"source":"pokemon-tcg-ai-battle-episodes-{day}"'.encode()
        for day in days
    }
    outputs = {
        day: (args.out_dir / f"top_ladder_all_{day}.jsonl").resolve()
        for day in days
    }
    for path in outputs.values():
        if path.exists():
            raise SystemExit(f"refusing to replace existing output: {path}")
    partials = {
        day: path.with_name(f".{path.name}.partial.{os.getpid()}")
        for day, path in outputs.items()
    }
    handles = {day: path.open("xb") for day, path in partials.items()}
    counts = {day: 0 for day in days}
    try:
        with source.open("rb") as input_handle:
            for raw in input_handle:
                matched = [day for day, marker in markers.items() if marker in raw]
                if not matched:
                    continue
                if len(matched) != 1:
                    raise ValueError("record matched multiple source-day tags")
                day = matched[0]
                handles[day].write(raw)
                counts[day] += 1
        for day, handle in handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if counts[day] <= 0:
                raise ValueError(f"no records found for {day}")
        for day in days:
            partials[day].replace(outputs[day])
            print(
                f"[ready] day={day} records={counts[day]} "
                f"bytes={outputs[day].stat().st_size} path={outputs[day]}",
                flush=True,
            )
    except BaseException:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for path in partials.values():
            path.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
