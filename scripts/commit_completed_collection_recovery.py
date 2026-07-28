#!/usr/bin/env python3
"""Publish a verified completed-collection receipt without recollecting games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train_pure_rl import _commit_completed_collection_receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--design-migration-receipt", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--collect-elapsed-seconds", type=float, required=True)
    parser.add_argument("--metadata-repair-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    state = json.loads((run_dir / "loop_state.json").read_text(encoding="utf-8"))
    migration = json.loads(
        args.design_migration_receipt.resolve().read_text(encoding="utf-8")
    )
    contract = dict(migration.get("current_contract") or {})
    if (
        int(state.get("next_iteration", -1)) != int(args.iteration)
        or str(state.get("design_fingerprint") or "")
        != str(migration.get("current_fingerprint") or "")
    ):
        raise RuntimeError(
            "ledger boundary/design fingerprint does not match migration receipt"
        )
    learner = dict(state.get("learner") or state.get("champion") or {})
    shard = run_dir / "shards" / f"iter_{int(args.iteration):05d}.jsonl"
    expected_games = int(dict(contract.get("games") or {}).get("per_iteration") or 0)
    elapsed = float(args.collect_elapsed_seconds)
    stats = {
        "ok": expected_games,
        "with_record": expected_games,
        "requested_games": expected_games,
        "retained_source_games": expected_games,
        "retained_trajectories": expected_games,
        "usable_game_fraction": 1.0,
        "collect_elapsed_sec": elapsed,
        "claimed_games_per_sec": expected_games / elapsed,
        "valid_source_games_per_sec": expected_games / elapsed,
        "trajectory_games_per_sec": expected_games / elapsed,
        "recovered_completed_collection": True,
        "metadata_repair_receipt": (
            str(args.metadata_repair_receipt.resolve())
            if args.metadata_repair_receipt is not None
            else None
        ),
    }
    receipt = _commit_completed_collection_receipt(
        run_dir=run_dir,
        state=state,
        contract=contract,
        iteration=int(args.iteration),
        shard=shard,
        checkpoint=Path(str(learner.get("path") or "")),
        checkpoint_digest=str(learner.get("digest") or ""),
        stats=stats,
        started_at=float(shard.stat().st_mtime) - elapsed,
        writer=None,
        recovery_derived=True,
    )
    print(
        json.dumps(
            {
                "receipt": receipt["receipt_path"],
                "games": receipt["shard"]["games"],
                "decisions": receipt["shard"]["decisions"],
                "sha256": receipt["shard"]["sha256"],
                "runtime_enforcement": receipt["stats"][
                    "matchup_runtime_enforcement"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
