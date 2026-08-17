#!/usr/bin/env python3
"""Explicit, isolated r212 Guide2Vec BO1000 plan/preflight/runner.

This is intentionally not a managed service and does not launch anything by
default.  ``--plan`` only creates the immutable r212 plan/output index;
``--preflight`` loads and audits frozen artifacts but opens no battle; and
``--run`` is the sole explicit mode allowed to start the 500 paired native
mirrors after a completed frozen Guide2Vec training receipt exists.
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

from poke_bot.guide2vec_bo1000_runtime import (  # noqa: E402
    R212ArtifactIdentity,
    build_guide2vec_bo1000_plan,
    materialize_guide2vec_bo1000_plan,
    preflight_guide2vec_bo1000_runtime,
    run_guide2vec_bo1000,
    verify_guide2vec_bo1000_plan,
)


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read r212 plan: {path}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("r212 plan must be a JSON object")
    verify_guide2vec_bo1000_plan(raw)
    return raw


def _plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        "r195_bundle",
        "r195_package_root",
        "r195_checkpoint",
        "guide2vec_checkpoint",
        "guide2vec_training_receipt",
        "seed_identity_sha256",
    )
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        raise SystemExit(
            "--plan requires " + ", ".join("--" + item.replace("_", "-") for item in missing)
        )
    artifacts = R212ArtifactIdentity(
        r195_bundle=args.r195_bundle,
        r195_package_root=args.r195_package_root,
        r195_checkpoint=args.r195_checkpoint,
        guide2vec_checkpoint=args.guide2vec_checkpoint,
        guide2vec_training_receipt=args.guide2vec_training_receipt,
        owner_contract=ROOT / "state/alakazam-guide2vec-no-mcts-bo1000-r212.json",
        r195_contract=ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json",
    )
    return build_guide2vec_bo1000_plan(
        artifacts=artifacts,
        seed_identity_sha256=args.seed_identity_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="write only PLAN.json and OUTPUT_IDENTITY.json")
    mode.add_argument("--preflight", action="store_true", help="audit frozen runtime only; never open a battle")
    mode.add_argument("--run", action="store_true", help="explicitly execute the isolated native 500-pair BO1000")
    parser.add_argument("--output-root", type=Path, required=True, help="dedicated r212 evaluation root")
    parser.add_argument("--plan-file", type=Path, help="immutable PLAN.json for --preflight or --run")
    parser.add_argument("--r195-bundle", type=Path)
    parser.add_argument("--r195-package-root", type=Path)
    parser.add_argument("--r195-checkpoint", type=Path)
    parser.add_argument("--guide2vec-checkpoint", type=Path)
    parser.add_argument("--guide2vec-training-receipt", type=Path)
    parser.add_argument("--seed-identity-sha256")
    parser.add_argument("--max-atomic-actions", type=int, default=4000)
    parser.add_argument("--native-seed-attempts", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.plan:
        if args.plan_file is not None:
            raise SystemExit("--plan-file is only valid with --preflight or --run")
        plan = _plan_from_args(args)
        output = materialize_guide2vec_bo1000_plan(plan=plan, output_root=args.output_root)
        print(json.dumps({"mode": "plan", "output": str(output), "plan_sha256": plan["canonical_sha256"]}, sort_keys=True))
        return 0
    if args.plan_file is None:
        raise SystemExit("--preflight and --run require --plan-file")
    plan = _read_plan(args.plan_file)
    if args.preflight:
        receipt = preflight_guide2vec_bo1000_runtime(
            plan=plan,
            output_root=args.output_root,
        )
        output = materialize_guide2vec_bo1000_plan(plan=plan, output_root=args.output_root)
        print(json.dumps({"mode": "preflight", "output": str(output), "receipt": receipt}, sort_keys=True))
        return 0
    result = run_guide2vec_bo1000(
        plan=plan,
        output_root=args.output_root,
        max_atomic_actions=args.max_atomic_actions,
        native_seed_attempts=args.native_seed_attempts,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
