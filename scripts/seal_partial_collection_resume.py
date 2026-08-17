#!/usr/bin/env python3
"""Seal an exact partial compact shard for append-only collection resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "poke_bot.partial_collection_resume_authorization/v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--requested-games", type=int, required=True)
    parser.add_argument("--expected-self-play", type=int, required=True)
    parser.add_argument("--expected-public", type=int, required=True)
    parser.add_argument("--expected-behavior-digest", required=True)
    args = parser.parse_args()

    shard = args.shard.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    digest = hashlib.sha256()
    episode_ids: set[str] = set()
    by_job: dict[int, list[tuple[int, bool, str, str]]] = {}
    decisions = 0
    records = 0
    with shard.open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                raise RuntimeError("partial shard ends with an incomplete row")
            digest.update(raw)
            obj = json.loads(raw)
            episode_id = str(obj.get("episode_id") or "")
            provenance = dict(obj.get("target_provenance") or {})
            rows = obj.get("decisions")
            job_index = int(provenance.get("collection_job_index", -1))
            behavior = str(provenance.get("behavior_checkpoint_digest") or "")
            if (
                not episode_id
                or episode_id in episode_ids
                or job_index < 0
                or not isinstance(rows, list)
                or not rows
                or behavior != args.expected_behavior_digest
            ):
                raise RuntimeError(
                    f"invalid partial row episode={episode_id!r} job={job_index}"
                )
            episode_ids.add(episode_id)
            self_play = bool(provenance.get("self_play"))
            opponent_id = str(provenance.get("opponent_id") or "")
            opponent_digest = str(
                provenance.get("opponent_checkpoint_digest") or ""
            )
            by_job.setdefault(job_index, []).append(
                (int(obj.get("seat", -1)), self_play, opponent_id, opponent_digest)
            )
            records += 1
            decisions += len(rows)

    self_play_jobs = 0
    public_jobs = 0
    for job_index, rows in by_job.items():
        is_self_play = all(row[1] for row in rows)
        if is_self_play:
            opponent_digests = {row[3] for row in rows}
            if "" in opponent_digests or len(opponent_digests) != 1:
                raise RuntimeError(
                    f"self-play job {job_index} has invalid opponent digest"
                )
            collect_both = opponent_digests == {args.expected_behavior_digest}
            expected_records = 2 if collect_both else 1
            if (
                len(rows) != expected_records
                or not {row[0] for row in rows}.issubset({0, 1})
                or (collect_both and {row[0] for row in rows} != {0, 1})
            ):
                raise RuntimeError(
                    f"self-play job {job_index} has invalid trajectory cardinality"
                )
            self_play_jobs += 1
        else:
            if len(rows) != 1 or rows[0][0] not in (0, 1):
                raise RuntimeError(f"public job {job_index} is not singleton complete")
            public_jobs += 1
    if self_play_jobs != args.expected_self_play or public_jobs != args.expected_public:
        raise RuntimeError(
            "partial source counts differ: "
            f"self={self_play_jobs}/{args.expected_self_play} "
            f"public={public_jobs}/{args.expected_public}"
        )
    retained = sorted(by_job)
    requested = int(args.requested_games)
    if any(value < 0 or value >= requested for value in retained):
        raise RuntimeError("partial shard contains an out-of-schedule job index")
    missing = sorted(set(range(requested)) - set(retained))
    stat = shard.stat()
    payload = {
        "schema": SCHEMA,
        "status": "authorized_append_only_resume",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": int(args.iteration),
        "requested_games": requested,
        "retained_source_games": len(retained),
        "retained_self_play_source_games": self_play_jobs,
        "retained_public_source_games": public_jobs,
        "retained_job_indices": retained,
        "missing_job_indices": missing,
        "trajectory_records": records,
        "decisions": decisions,
        "behavior_checkpoint_digest": args.expected_behavior_digest,
        "shard_path": str(shard),
        "shard_size": int(stat.st_size),
        "shard_mtime_ns": int(stat.st_mtime_ns),
        "shard_sha256": f"sha256:{digest.hexdigest()}",
        "resume_policy": "append_only_missing_job_indices_then_full_reaudit",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: payload[k] for k in (
        "shard_sha256", "retained_source_games", "retained_self_play_source_games",
        "retained_public_source_games", "trajectory_records", "decisions"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
