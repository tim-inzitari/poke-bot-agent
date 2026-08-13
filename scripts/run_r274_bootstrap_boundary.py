#!/usr/bin/env python3
"""Advance the r274 bootstrap through its exact upload/activation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any


OWNER_SOURCE = "GOAL.md#/revision-264-TRAINING"


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"r274 bootstrap boundary command failed rc={completed.returncode}: "
            + " ".join(command)
        )


def _emit(status: str, **fields: Any) -> None:
    print(json.dumps({"status": status, **fields}, sort_keys=True), flush=True)


def advance(args: argparse.Namespace) -> str:
    if args.tactical_activation_receipt.is_file():
        from poke_bot.r274_bootstrap_handoff import validate_handoff_receipt

        validate_handoff_receipt(
            args.tactical_activation_receipt,
            expected_initial_checkpoint=args.tactical_activation_checkpoint,
        )
        return "complete"
    if not args.gpu_bootstrap_result.is_file():
        return "waiting_bootstrap"
    if (
        not args.tactical_repair_receipt.is_file()
        or not args.adapter_training_receipt.is_file()
        or not args.bootstrap_checkpoint.is_file()
    ):
        return "waiting_matchup_adapter_bootstrap"

    if not args.bootstrap_submission_receipt.is_file():
        _run(
            [
                str(args.python),
                "scripts/finalize_r274_bootstrap_submission_checkpoint.py",
                "--base-checkpoint",
                str(args.activated_parent),
                "--bootstrap-checkpoint",
                str(args.bootstrap_checkpoint),
                "--expert-manifest",
                str(args.expert_manifest),
                "--tactical-overlay",
                str(args.tactical_overlay),
                "--gpu-bootstrap-result-receipt",
                str(args.gpu_bootstrap_result),
                "--tactical-repair-receipt",
                str(args.tactical_repair_receipt),
                "--adapter-training-receipt",
                str(args.adapter_training_receipt),
                "--output-checkpoint",
                str(args.bootstrap_submission_checkpoint),
                "--output-receipt",
                str(args.bootstrap_submission_receipt),
            ],
            cwd=args.runtime_root,
        )
        return "bootstrap_submission_checkpoint_ready"

    if not args.stage_receipt.is_file():
        _run(
            [
                str(args.python),
                "scripts/stage_r274_bootstrap_submission.py",
                "--runtime-root",
                str(args.runtime_root),
                "--contract",
                str(args.contract),
                "--submission-checkpoint",
                str(args.bootstrap_submission_checkpoint),
                "--bootstrap-submission-receipt",
                str(args.bootstrap_submission_receipt),
                "--inactive-matchup-tree",
                str(args.inactive_matchup_tree),
                "--matchup-roster",
                str(args.matchup_roster),
                "--cg-root",
                str(args.cg_root),
                "--output-root",
                str(args.submission_root),
                "--queue",
                str(args.queue),
                "--python",
                str(args.python),
                "--receipt",
                str(args.stage_receipt),
            ],
            cwd=args.runtime_root,
        )
        return "submission_staged"

    from scripts.process_kaggle_submission_queue import process_once

    queue_result = process_once(
        queue_path=args.queue,
        kaggle=args.kaggle,
        default_competition="pokemon-tcg-ai-battle",
        authorization_path=args.authorization,
        required_owner_decision_source=OWNER_SOURCE,
    )
    queue_status = str(queue_result.get("status") or "unknown")
    if queue_status in {
        "failed",
        "failed_identity",
        "failed_unknown_prior_attempt",
        "processor_error",
    }:
        raise RuntimeError(f"r274 bootstrap upload failed: {queue_result}")

    from scripts.materialize_r274_bootstrap_upload import materialize

    upload_result = materialize(
        queue_path=args.queue,
        stage_receipt=args.stage_receipt,
        attempt_receipts=args.attempt_receipts,
        output=args.upload_receipt,
    )
    if not args.upload_receipt.is_file():
        _emit(
            "waiting_upload",
            queue=queue_result,
            upload=upload_result,
        )
        return "waiting_upload"

    _run(
        [
            str(args.python),
            "scripts/activate_r274_post_bootstrap_tactical_route.py",
            "--submission-checkpoint",
            str(args.bootstrap_submission_checkpoint),
            "--bootstrap-submission-receipt",
            str(args.bootstrap_submission_receipt),
            "--upload-receipt",
            str(args.upload_receipt),
            "--manifest",
            str(args.expert_manifest),
            "--sidecar-binding",
            str(args.sidecar_binding),
            "--index",
            str(args.sidecar_index),
            "--tactical-overlay",
            str(args.tactical_overlay),
            "--output-checkpoint",
            str(args.tactical_activation_checkpoint),
            "--output-receipt",
            str(args.tactical_activation_receipt),
            "--device",
            args.device,
        ],
        cwd=args.runtime_root,
    )
    from poke_bot.r274_bootstrap_handoff import validate_handoff_receipt

    validate_handoff_receipt(
        args.tactical_activation_receipt,
        expected_initial_checkpoint=args.tactical_activation_checkpoint,
    )
    return "complete"


def _path(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime_root",
        "python",
        "kaggle",
        "authorization",
        "attempt_receipts",
        "contract",
        "activated_parent",
        "bootstrap_checkpoint",
        "gpu_bootstrap_result",
        "tactical_repair_receipt",
        "adapter_training_receipt",
        "expert_manifest",
        "tactical_overlay",
        "sidecar_binding",
        "sidecar_index",
        "bootstrap_submission_checkpoint",
        "bootstrap_submission_receipt",
        "inactive_matchup_tree",
        "matchup_roster",
        "cg_root",
        "submission_root",
        "queue",
        "stage_receipt",
        "upload_receipt",
        "tactical_activation_checkpoint",
        "tactical_activation_receipt",
    ):
        _path(parser, name)
    parser.add_argument("--device", default="cuda:1")
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
        time.sleep(max(15.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
