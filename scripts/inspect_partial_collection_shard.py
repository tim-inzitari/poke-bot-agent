#!/usr/bin/env python3
"""Read-only summary of a compact partial collection shard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--expected-behavior-digest", required=True)
    args = parser.parse_args()

    jobs: dict[int, list[tuple[int, bool, str]]] = {}
    decisions = 0
    records = 0
    with args.shard.open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                raise RuntimeError("partial shard ends with an incomplete row")
            row = json.loads(raw)
            provenance = dict(row.get("target_provenance") or {})
            behavior = str(provenance.get("behavior_checkpoint_digest") or "")
            if behavior != args.expected_behavior_digest:
                raise RuntimeError(f"unexpected behavior digest: {behavior}")
            job = int(provenance.get("collection_job_index", -1))
            if job < 0:
                raise RuntimeError("missing collection_job_index")
            jobs.setdefault(job, []).append(
                (
                    int(row.get("seat", -1)),
                    bool(provenance.get("self_play")),
                    str(provenance.get("opponent_checkpoint_digest") or ""),
                )
            )
            decisions += len(row.get("decisions") or [])
            records += 1

    self_play = 0
    public = 0
    incomplete: list[int] = []
    for job, rows in jobs.items():
        if all(item[1] for item in rows):
            opponent_digests = {item[2] for item in rows}
            collect_both = opponent_digests == {args.expected_behavior_digest}
            expected_records = 2 if collect_both else 1
            if (
                "" in opponent_digests
                or len(opponent_digests) != 1
                or len(rows) != expected_records
                or not {item[0] for item in rows}.issubset({0, 1})
                or (collect_both and {item[0] for item in rows} != {0, 1})
            ):
                incomplete.append(job)
                continue
            self_play += 1
        else:
            if len(rows) != 1 or rows[0][0] not in (0, 1):
                incomplete.append(job)
                continue
            public += 1
    print(json.dumps({
        "observed_source_jobs": len(jobs),
        "retained_source_games": self_play + public,
        "retained_self_play_source_games": self_play,
        "retained_public_source_games": public,
        "trajectory_records": records,
        "decisions": decisions,
        "minimum_job_index": min(jobs) if jobs else None,
        "maximum_job_index": max(jobs) if jobs else None,
        "incomplete_job_indices": sorted(incomplete),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
