#!/usr/bin/env python3
"""Continuously service the four post-bootstrap r274 submission boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from scripts.process_kaggle_submission_queue import process_once
from scripts.stage_r274_bootstrap_submission import OWNER_SOURCE


BOUNDARIES = (1, 5, 10, 15, 20)


def _emit(status: str, **fields: Any) -> None:
    print(json.dumps({"status": status, **fields}, sort_keys=True), flush=True)


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"r274 RL submission worker failed rc={completed.returncode}: "
            + " ".join(command)
        )


def _paths(args: argparse.Namespace, boundary: int) -> dict[str, Path]:
    exchange = args.boundary_exchange / f"boundary-{boundary:05d}"
    output = args.submission_root / f"boundary-{boundary:05d}"
    return {
        "request": exchange / "request.json",
        "upload": exchange / "upload.json",
        "output": output,
        "stage": output / "stage.json",
    }


def advance(args: argparse.Namespace) -> str:
    for boundary in BOUNDARIES:
        paths = _paths(args, boundary)
        if paths["upload"].is_file():
            continue
        if not paths["request"].is_file():
            return f"waiting_boundary_{boundary:05d}_request"
        if not paths["stage"].is_file():
            _run(
                [
                    str(args.python),
                    "scripts/stage_r274_rl_submission.py",
                    "--runtime-root",
                    str(args.runtime_root),
                    "--python",
                    str(args.python),
                    "--contract",
                    str(args.contract),
                    "--request",
                    str(paths["request"]),
                    "--active-matchup-tree",
                    str(args.active_matchup_tree),
                    "--matchup-roster",
                    str(args.matchup_roster),
                    "--cg-root",
                    str(args.cg_root),
                    "--output-root",
                    str(paths["output"]),
                    "--queue",
                    str(args.queue),
                    "--receipt",
                    str(paths["stage"]),
                ],
                cwd=args.runtime_root,
            )
            return f"boundary_{boundary:05d}_staged"

        queue_result = process_once(
            queue_path=args.queue,
            kaggle=args.kaggle,
            default_competition="pokemon-tcg-ai-battle",
            authorization_path=args.authorization,
            required_owner_decision_source=(
                "GOAL.md#/revision-304-TRAINING"
                if boundary == 1
                else OWNER_SOURCE
            ),
        )
        status = str(queue_result.get("status") or "unknown")
        if status in {
            "failed",
            "failed_identity",
            "failed_unknown_prior_attempt",
            "processor_error",
        }:
            raise RuntimeError(
                f"r274 boundary {boundary} upload failed: {queue_result}"
            )
        from scripts.materialize_r274_rl_upload import materialize

        materialized = materialize(
            queue_path=args.queue,
            stage_receipt=paths["stage"],
            request_path=paths["request"],
            attempt_receipts=args.attempt_receipts,
            output=paths["upload"],
        )
        if paths["upload"].is_file():
            return f"boundary_{boundary:05d}_complete"
        _emit(
            f"waiting_boundary_{boundary:05d}_upload",
            queue=queue_result,
            materialize=materialized,
        )
        return f"waiting_boundary_{boundary:05d}_upload"
    return "complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime_root",
        "python",
        "kaggle",
        "authorization",
        "attempt_receipts",
        "contract",
        "active_matchup_tree",
        "matchup_roster",
        "cg_root",
        "boundary_exchange",
        "submission_root",
        "queue",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main() -> int:
    args = parse_args()
    while True:
        status = advance(args)
        _emit(status)
        if status == "complete" or args.once:
            return 0
        time.sleep(max(15.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
