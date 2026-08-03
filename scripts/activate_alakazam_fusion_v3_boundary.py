#!/usr/bin/env python3
"""Atomically point a clean Alakazam loop boundary at its Fusion-v3 child."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"activation receipt changed: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    partial.write_text(body, encoding="utf-8")
    os.link(partial, path)
    partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--migrated-checkpoint", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--superseded-migration-receipt", type=Path)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    loop_path = run_dir / "loop_state.json"
    commit_path = run_dir / "commits" / f"iter_{args.iteration:05d}.json"
    migrated = args.migrated_checkpoint.expanduser().resolve()
    migration_path = args.migration_receipt.expanduser().resolve()
    if not all(path.is_file() for path in (loop_path, commit_path, migrated, migration_path)):
        raise RuntimeError("fusion-v3 boundary inputs are incomplete")
    loop = json.loads(loop_path.read_text(encoding="utf-8"))
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    current_learner = dict(loop.get("learner") or {})
    source = dict(commit.get("learner") or {})
    allowed_current_learners = [source]
    if args.superseded_migration_receipt is not None:
        superseded_path = args.superseded_migration_receipt.expanduser().resolve()
        superseded = json.loads(superseded_path.read_text(encoding="utf-8"))
        allowed_current_learners.append(
            {
                "path": superseded.get("output_checkpoint"),
                "digest": superseded.get("output_checkpoint_sha256"),
            }
        )
    if (
        int(loop.get("last_completed_iteration", -1)) != args.iteration
        or int(loop.get("next_iteration", -1)) != args.iteration + 1
        or int(commit.get("last_completed_iteration", -1)) != args.iteration
        or int(commit.get("next_iteration", -1)) != args.iteration + 1
        or dict(commit.get("learner") or {}) != source
        or current_learner not in allowed_current_learners
        or migration.get("status") != "validated"
        or migration.get("source_checkpoint") != source.get("path")
        or migration.get("source_checkpoint_sha256") != source.get("digest")
        or migration.get("output_checkpoint") != str(migrated)
        or migration.get("output_checkpoint_sha256") != _sha256(migrated)
    ):
        raise RuntimeError("fusion-v3 activation is not at the exact committed boundary")
    updated = dict(loop)
    updated["learner"] = {
        "path": str(migrated),
        "digest": migration["output_checkpoint_sha256"],
    }
    partial = loop_path.with_name(f".{loop_path.name}.fusion-v3.{os.getpid()}")
    partial.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, loop_path)
    receipt = {
        "schema": "poke_bot.alakazam_fusion_v3_boundary_activation/v1",
        "status": "active_for_next_iteration",
        "completed_iteration": args.iteration,
        "next_iteration": args.iteration + 1,
        "source_checkpoint": source,
        "learner": updated["learner"],
        "migration_receipt": str(migration_path),
        "migration_receipt_sha256": _sha256(migration_path),
        "loop_state_sha256_after_activation": _sha256(loop_path),
        "guide_training_mode": "strategic_directional_v2",
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "revision_103_iter20_boundary_preserved": True,
    }
    _write_once(args.activation_receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
