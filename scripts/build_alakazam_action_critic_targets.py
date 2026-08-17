"""Seal revision-21 target-only action-critic artifacts under the current contract.

The command reads one immutable complete-action JSONL overlay and its exact
raw episode ZIP.  It does not start a model, simulator, search, planner, or
training job.  The destination must be a new output root and is published only
after the content-addressed target shard, manifest, and receipt are durable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.action_critic_targets import (
    ActionCriticTargetError,
    build_action_critic_target_overlay_day,
    finalize_action_critic_target_set,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complete-action-overlay",
        type=Path,
        required=True,
        help="one sealed complete-action overlay JSONL day",
    )
    parser.add_argument(
        "--raw-episode-zip",
        type=Path,
        required=True,
        help="the exact raw episode ZIP whose SHA-256 is carried by every overlay row",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new create-only per-day artifact root",
    )
    parser.add_argument("--utc-day", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "evaluation"),
    )
    parser.add_argument(
        "--goal-contract",
        type=Path,
        required=True,
        help="the current canonical contract carrying the embedded revision-21 critic authority",
    )
    parser.add_argument(
        "--expected-goal-contract-sha256",
        required=True,
        help="full sha256:<hex> identity of --goal-contract",
    )
    parser.add_argument(
        "--expected-complete-action-overlay-sha256",
        required=True,
        help="full sha256:<hex> identity of --complete-action-overlay",
    )
    parser.add_argument(
        "--expected-raw-episode-zip-sha256",
        required=True,
        help="full sha256:<hex> identity of --raw-episode-zip",
    )
    return parser


def _finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate exactly 20 sealed target-overlay days and seal their aggregate manifest."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--goal-contract", type=Path, required=True)
    parser.add_argument("--expected-goal-contract-sha256", required=True)
    parser.add_argument("--base-pack-completion", type=Path, required=True)
    parser.add_argument("--expected-base-pack-completion-sha256", required=True)
    parser.add_argument("--complete-action-overlay-manifest", type=Path, required=True)
    parser.add_argument("--expected-complete-action-overlay-manifest-sha256", required=True)
    parser.add_argument(
        "--day-artifact-root",
        type=Path,
        action="append",
        required=True,
        help="one sealed per-day target artifact root; pass exactly 20 times",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    finalizing = bool(supplied and supplied[0] == "finalize")
    args = (_finalize_parser() if finalizing else _parser()).parse_args(
        supplied[1:] if finalizing else supplied
    )
    try:
        if finalizing:
            result = finalize_action_critic_target_set(
                day_artifact_roots=args.day_artifact_root,
                output_root=args.output_root,
                goal_contract_path=args.goal_contract,
                expected_goal_contract_sha256=args.expected_goal_contract_sha256,
                base_pack_completion_path=args.base_pack_completion,
                expected_base_pack_completion_sha256=(
                    args.expected_base_pack_completion_sha256
                ),
                complete_action_overlay_manifest_path=(
                    args.complete_action_overlay_manifest
                ),
                expected_complete_action_overlay_manifest_sha256=(
                    args.expected_complete_action_overlay_manifest_sha256
                ),
            )
        else:
            result = build_action_critic_target_overlay_day(
                complete_action_overlay_path=args.complete_action_overlay,
                raw_episode_zip_path=args.raw_episode_zip,
                output_root=args.output_root,
                utc_day=args.utc_day,
                split=args.split,
                goal_contract_path=args.goal_contract,
                expected_goal_contract_sha256=args.expected_goal_contract_sha256,
                expected_complete_action_overlay_sha256=(
                    args.expected_complete_action_overlay_sha256
                ),
                expected_raw_episode_zip_sha256=args.expected_raw_episode_zip_sha256,
            )
    except ActionCriticTargetError as exc:
        raise SystemExit(f"action-critic target overlay refused: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
