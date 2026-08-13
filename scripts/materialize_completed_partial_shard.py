#!/usr/bin/env python3
"""Copy only source-game-complete rows from a stopped partial shard."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-behavior-digest", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    jobs: dict[int, list[tuple[int, bool, str]]] = {}
    with args.source.open("rb") as source:
        for raw in source:
            if not raw.endswith(b"\n"):
                raise RuntimeError("source shard ends with an incomplete row")
            row = json.loads(raw)
            provenance = dict(row.get("target_provenance") or {})
            if str(provenance.get("behavior_checkpoint_digest") or "") != args.expected_behavior_digest:
                raise RuntimeError("source shard behavior digest changed")
            job = int(provenance.get("collection_job_index", -1))
            jobs.setdefault(job, []).append(
                (
                    int(row.get("seat", -1)),
                    bool(provenance.get("self_play")),
                    str(provenance.get("opponent_checkpoint_digest") or ""),
                )
            )

    complete: set[int] = set()
    for job, rows in jobs.items():
        if all(item[1] for item in rows):
            opponent_digests = {item[2] for item in rows}
            collect_both = opponent_digests == {args.expected_behavior_digest}
            expected_records = 2 if collect_both else 1
            if (
                "" not in opponent_digests
                and len(opponent_digests) == 1
                and len(rows) == expected_records
                and {item[0] for item in rows}.issubset({0, 1})
                and (not collect_both or {item[0] for item in rows} == {0, 1})
            ):
                complete.add(job)
        elif len(rows) == 1 and rows[0][0] in (0, 1):
            complete.add(job)

    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    kept_records = 0
    excluded_records = 0
    with os.fdopen(fd, "wb") as output, args.source.open("rb") as source:
        for raw in source:
            row = json.loads(raw)
            job = int((row.get("target_provenance") or {}).get("collection_job_index", -1))
            if job in complete:
                output.write(raw)
                kept_records += 1
            else:
                excluded_records += 1
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({
        "complete_source_games": len(complete),
        "excluded_source_games": len(jobs) - len(complete),
        "kept_trajectory_records": kept_records,
        "excluded_trajectory_records": excluded_records,
        "excluded_job_indices": sorted(set(jobs) - complete),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
