#!/usr/bin/env python3
"""Build sealed public Prize-plan-v2 targets; no model or training actions.

The command has three explicit phases:

``fit-phi``
    checks all 20 immutable source identities but opens raw payloads only for
    the fourteen train days, then seals the monotone public Phi table.

``build-day``
    reads one exact raw day and its aligned complete-action overlay to label
    H1/H3/H6/H12 causal segment returns with a frozen Phi table.

``finalize``
    verifies all twenty day roots, split isolation, shared segment proofs, and
    analytic model target transform, then publishes a compact portable set.

Every output root is create-only.  This command never launches a critic,
optimizer, simulator, search, RTP, MCTS, or actor integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.prize_plan_targets_v2 import (  # noqa: E402
    PrizePlanTargetError,
    build_prize_plan_target_overlay_day,
    finalize_prize_plan_target_set,
    fit_prize_plan_potential_v2,
)


def _day_input_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--day-input must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--day-input must decode to a JSON object")
    return parsed


def _fit_config_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--fit-configuration must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--fit-configuration must decode to a JSON object")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit-phi", help="fit train-only frozen monotone public Phi")
    fit.add_argument("--output-root", type=Path, required=True)
    fit.add_argument("--goal-contract", type=Path, required=True)
    fit.add_argument("--expected-goal-contract-sha256", required=True)
    fit.add_argument(
        "--fit-configuration",
        type=_fit_config_json,
        required=True,
        help=(
            "exact JSON: {\"algorithm\":\"alternating_weighted_2d_isotonic_pava/v1\","
            "\"smoothing_prior_strength\":8.0,\"max_iterations\":10000,"
            "\"convergence_tolerance\":1e-10}"
        ),
    )
    fit.add_argument(
        "--day-input",
        type=_day_input_json,
        action="append",
        required=True,
        help=(
            "repeat exactly 20 times: {utc_day,split,complete_action_overlay_path,"
            "complete_action_overlay_sha256,raw_episode_zip_path,raw_episode_zip_sha256}"
        ),
    )

    day = commands.add_parser("build-day", help="materialize one frozen-Phi target day")
    day.add_argument("--output-root", type=Path, required=True)
    day.add_argument("--utc-day", required=True)
    day.add_argument("--split", choices=("train", "validation", "evaluation"), required=True)
    day.add_argument("--complete-action-overlay", type=Path, required=True)
    day.add_argument("--expected-complete-action-overlay-sha256", required=True)
    day.add_argument("--raw-episode-zip", type=Path, required=True)
    day.add_argument("--expected-raw-episode-zip-sha256", required=True)
    day.add_argument("--goal-contract", type=Path, required=True)
    day.add_argument("--expected-goal-contract-sha256", required=True)
    day.add_argument("--phi-fit-manifest", type=Path, required=True)
    day.add_argument("--expected-phi-fit-manifest-sha256", required=True)
    day.add_argument(
        "--gamma",
        type=float,
        required=True,
        help="explicit receipt-bound gamma; first owner-selected launch value is 1.0",
    )

    final = commands.add_parser("finalize", help="seal exact portable recent-20 target set")
    final.add_argument("--output-root", type=Path, required=True)
    final.add_argument("--goal-contract", type=Path, required=True)
    final.add_argument("--expected-goal-contract-sha256", required=True)
    final.add_argument("--phi-fit-manifest", type=Path, required=True)
    final.add_argument("--expected-phi-fit-manifest-sha256", required=True)
    final.add_argument("--complete-action-overlay-manifest", type=Path, required=True)
    final.add_argument("--expected-complete-action-overlay-manifest-sha256", required=True)
    final.add_argument("--gamma", type=float, required=True)
    final.add_argument(
        "--day-artifact-root",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly 20 create-only day roots",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "fit-phi":
            result = fit_prize_plan_potential_v2(
                day_inputs=args.day_input,
                output_root=args.output_root,
                goal_contract_path=args.goal_contract,
                expected_goal_contract_sha256=args.expected_goal_contract_sha256,
                fit_configuration=args.fit_configuration,
            )
        elif args.command == "build-day":
            result = build_prize_plan_target_overlay_day(
                complete_action_overlay_path=args.complete_action_overlay,
                raw_episode_zip_path=args.raw_episode_zip,
                output_root=args.output_root,
                utc_day=args.utc_day,
                split=args.split,
                goal_contract_path=args.goal_contract,
                expected_goal_contract_sha256=args.expected_goal_contract_sha256,
                phi_fit_manifest_path=args.phi_fit_manifest,
                expected_phi_fit_manifest_sha256=args.expected_phi_fit_manifest_sha256,
                gamma=args.gamma,
                expected_complete_action_overlay_sha256=args.expected_complete_action_overlay_sha256,
                expected_raw_episode_zip_sha256=args.expected_raw_episode_zip_sha256,
            )
        else:
            result = finalize_prize_plan_target_set(
                day_artifact_roots=args.day_artifact_root,
                output_root=args.output_root,
                goal_contract_path=args.goal_contract,
                expected_goal_contract_sha256=args.expected_goal_contract_sha256,
                phi_fit_manifest_path=args.phi_fit_manifest,
                expected_phi_fit_manifest_sha256=args.expected_phi_fit_manifest_sha256,
                complete_action_overlay_manifest_path=args.complete_action_overlay_manifest,
                expected_complete_action_overlay_manifest_sha256=(
                    args.expected_complete_action_overlay_manifest_sha256
                ),
                gamma=args.gamma,
            )
    except PrizePlanTargetError as exc:
        raise SystemExit(f"Prize-plan-v2 target pipeline refused: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

