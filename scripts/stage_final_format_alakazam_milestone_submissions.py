#!/usr/bin/env python3
"""Queue nonblocking final-format milestone snapshots from durable commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.handle_passed_gate import (  # noqa: E402
    _copy_submission_slot,
    build_submission_bundle,
    materialize_pinned_specialist_deck,
    queue_submission_copies,
)

SCHEMA = "poke_bot.alakazam_milestone_submission/v1"
GENERIC_SCHEMA = "poke_bot.final_format_milestone_submission/v1"


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def eligible_iterations(
    run_dir: Path,
    *,
    maximum_iteration: int = 184,
    include_iteration_zero: bool = False,
) -> list[int]:
    found: list[int] = []
    for path in sorted((run_dir / "commits").glob("iter_*.json")):
        try:
            iteration = int(path.stem.removeprefix("iter_"))
        except ValueError:
            continue
        if (
            (include_iteration_zero and iteration == 0)
            or (
                4 <= iteration <= maximum_iteration
                and iteration % 5 == 4
            )
        ):
            found.append(iteration)
    return found


def receipt_schema(specialist_id: str) -> str:
    return SCHEMA if specialist_id == "alakazam" else GENERIC_SCHEMA


def validate_commit(
    run_dir: Path,
    iteration: int,
    *,
    prefer_committed_learner: bool = False,
) -> tuple[Path, Path, str, str]:
    commit_path = run_dir / "commits" / f"iter_{iteration:05d}.json"
    checkpoint = run_dir / "checkpoints" / f"iter_{iteration:05d}.pt"
    commit = read_json(commit_path)
    rows = [
        row
        for row in (commit.get("history") or [])
        if isinstance(row, dict) and int(row.get("iteration", -1)) == iteration
    ]
    if (
        int(commit.get("last_completed_iteration", -1)) != iteration
        or int(commit.get("next_iteration", -1)) != iteration + 1
        or len(rows) != 1
        or rows[0].get("completed") is not True
        or not checkpoint.is_file()
    ):
        raise RuntimeError(f"iteration {iteration} has no exact durable commit")
    checkpoint_role = "iteration_candidate"
    if prefer_committed_learner:
        learner = dict(commit.get("learner") or {})
        learner_path = Path(str(learner.get("path") or "")).expanduser().resolve()
        learner_digest = str(learner.get("digest") or "")
        if (
            not learner_path.is_file()
            or not learner_digest.startswith("sha256:")
            or sha256(learner_path) != learner_digest
        ):
            raise RuntimeError(
                f"iteration {iteration} committed learner is missing or digest-mismatched"
            )
        checkpoint = learner_path
        checkpoint_role = "committed_learner"
    digest = sha256(checkpoint)
    candidate = dict(rows[0].get("candidate") or {})
    known_digests = {
        str(candidate.get("digest") or ""),
        str((commit.get("learner") or {}).get("digest") or ""),
        str((commit.get("heldout_champion") or {}).get("digest") or ""),
    }
    known_digests.discard("")
    if known_digests and digest not in known_digests:
        raise RuntimeError(f"iteration {iteration} checkpoint digest is not commit-bound")
    return commit_path, checkpoint, digest, checkpoint_role


def stage_one(args: argparse.Namespace, iteration: int) -> dict[str, Any]:
    schema = receipt_schema(args.specialist_id)
    receipt = args.receipts / f"iter_{iteration:05d}.json"
    prefer_committed_learner = (
        args.specialist_id == "marnie-s-grimmsnarl-ex" and iteration == 9
    )
    if receipt.is_file():
        existing = read_json(receipt)
        if existing.get("schema") != schema or existing.get("status") != "queued":
            raise RuntimeError(f"invalid existing milestone receipt: {receipt}")
        if prefer_committed_learner:
            _, learner_path, learner_digest, learner_role = validate_commit(
                args.run_dir,
                iteration,
                prefer_committed_learner=True,
            )
            if (
                Path(str(existing.get("checkpoint") or "")).resolve()
                != learner_path
                or str(existing.get("checkpoint_sha256") or "")
                != learner_digest
                or existing.get("checkpoint_role") != learner_role
            ):
                raise RuntimeError(
                    "existing Marnie iteration-9 milestone does not bind the "
                    "exact committed learner"
                )
        return existing

    commit, checkpoint, checkpoint_digest, checkpoint_role = validate_commit(
        args.run_dir,
        iteration,
        prefer_committed_learner=prefer_committed_learner,
    )
    root = args.submission_root / f"iter-{iteration:05d}"
    deck = materialize_pinned_specialist_deck(
        run_dir=args.run_dir,
        representatives_path=args.representatives,
        archetype=args.specialist_id,
        output_path=root / f"pinned-{args.specialist_id}.deck.csv",
    )
    bundle = build_submission_bundle(
        repo_root=args.runtime_root,
        frozen_manifest={
            "model_path": str(checkpoint.resolve()),
            "checkpoint_digest": checkpoint_digest,
        },
        deck_receipt=deck,
        output_dir=root / "build",
        python=args.python,
        archetype=args.specialist_id,
        matchup_tree=args.matchup_tree,
        cg_root=args.cg_root,
        turn_order_preference="first_if_allowed",
        rtp_mode="off",
        direct_no_search_assets=True,
    )
    copy = _copy_submission_slot(bundle, root, 1)
    queued = queue_submission_copies(
        queue_path=args.queue,
        copies=[copy],
        gate_plan={
            "checkpoint_digest": checkpoint_digest,
            "gate_id": (
                f"final-format-{args.specialist_id}-training-snapshot-r"
                f"{args.owner_decision_revision}"
            ),
            "iteration": iteration,
            "completion_authority": "training_milestone_snapshot",
        },
        specialist_id=args.specialist_id,
        competition="pokemon-tcg-ai-battle",
    )
    payload = {
        "schema": schema,
        "status": "queued",
        "owner_decision_revision": args.owner_decision_revision,
        "iteration": iteration,
        "commit": str(commit.resolve()),
        "commit_sha256": sha256(commit),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_role": checkpoint_role,
        "bundle": bundle,
        "queue_entry": queued[0],
        "training_stop_or_freeze_authority": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(receipt, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--cg-root", type=Path)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--specialist-id", default="alakazam")
    parser.add_argument("--maximum-iteration", type=int, default=184)
    parser.add_argument("--include-iteration-zero", action="store_true")
    parser.add_argument(
        "--iteration",
        type=int,
        help="Stage one exact durable iteration, including an owner-authorized non-cadence snapshot.",
    )
    parser.add_argument("--owner-decision-revision", type=int, default=97)
    args = parser.parse_args()
    for key in (
        "runtime_root", "run_dir", "representatives", "matchup_tree",
        "submission_root", "receipts", "queue", "python",
    ):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.cg_root is not None:
        args.cg_root = args.cg_root.expanduser().resolve()
    iterations = (
        [args.iteration]
        if args.iteration is not None
        else eligible_iterations(
            args.run_dir,
            maximum_iteration=args.maximum_iteration,
            include_iteration_zero=args.include_iteration_zero,
        )
    )
    args.specialist_id = args.specialist_id.strip().casefold()
    if (
        any(iteration < 0 for iteration in iterations)
        or args.maximum_iteration < 0
        or not args.specialist_id.strip()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in args.specialist_id
        )
    ):
        raise ValueError("iteration must be nonnegative")
    results = [stage_one(args, iteration) for iteration in iterations]
    print(json.dumps({"status": "ready", "staged": [r["iteration"] for r in results]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
