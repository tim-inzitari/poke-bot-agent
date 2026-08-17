#!/usr/bin/env python3
"""Generate immutable checkpoint-derived r241 launch/terminal receipts.

This helper is deliberately offline: it reads the supplied immutable files,
reconstructs the checkpoint model for its evidence, and creates only the
specified write-once JSON receipt(s).  It never starts a trainer, service,
collector, queue, or submission client.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import r241_checkpoint_receipts as receipts  # noqa: E402


def _direct_environment(official_cg_root: Path) -> dict[str, str]:
    """Set required direct keys but retain inherited forbidden keys to fail closed."""

    result = dict(os.environ)
    result.update(
        {
            "CG_LIB_PATH": str(official_cg_root),
            "POKEBOT_R241_DIRECT_POLICY_ONLY": "1",
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
            "POKEBOT_SEARCH_MODE": "policy",
            "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
            "POKEBOT_COMBO_STATE_ROUTE_ENABLED": "0",
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("peak", "terminal"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--learner-matchup-tree", type=Path, required=True)
    parser.add_argument("--h10-matchup-tree", type=Path, required=True)
    parser.add_argument("--official-cg-root", type=Path, required=True)
    parser.add_argument("--expert-window-receipt", type=Path, required=True)
    parser.add_argument("--source-snapshot-root", type=Path, required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--source-outputs-root", type=Path, required=True)
    parser.add_argument("--source-host", choices=("inzi", "elmo"), required=True)
    parser.add_argument(
        "--baseline-adapter-roster",
        type=Path,
        default=receipts.BASELINE_ADAPTER_ROSTER_PATH,
    )

    peak = parser.add_argument_group("peak receipt")
    peak.add_argument("--output", type=Path)
    peak.add_argument("--protected-expert-pointer", type=Path)
    peak.add_argument("--h10-adapter-receipt", type=Path)
    peak.add_argument("--active-gate-contract", type=Path)
    peak.add_argument("--frozen-specialist-registry", type=Path)
    peak.add_argument("--research-control-registry", type=Path)
    peak.add_argument("--adapter-training-activation", type=Path)

    terminal = parser.add_argument_group("terminal receipts")
    terminal.add_argument("--model-output", type=Path)
    terminal.add_argument("--matchup-output", type=Path)
    terminal.add_argument("--terminal-parent-checkpoint", type=Path)
    terminal.add_argument("--terminal-checkpoint", type=Path)
    terminal.add_argument("--terminal-refresh-receipt", type=Path)
    terminal.add_argument("--terminal-rehearsal-receipt", type=Path)
    return parser


def _require(args: argparse.Namespace, *names: str) -> None:
    missing = ["--" + name.replace("_", "-") for name in names if getattr(args, name) is None]
    if missing:
        raise receipts.R241CheckpointReceiptError(
            f"{args.kind} receipt generation requires " + ", ".join(missing)
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    environment = _direct_environment(args.official_cg_root)
    common = {
        "contract_path": args.contract,
        "learner_matchup_tree": args.learner_matchup_tree,
        "h10_matchup_tree": args.h10_matchup_tree,
        "official_cg_root": args.official_cg_root,
        "environment": environment,
        "expert_window_receipt": args.expert_window_receipt,
        "source_snapshot_root": args.source_snapshot_root,
        "source_snapshot_manifest": args.source_snapshot_manifest,
        "source_outputs_root": args.source_outputs_root,
        "source_snapshot_host": args.source_host,
        "baseline_adapter_roster": args.baseline_adapter_roster,
    }
    if args.kind == "peak":
        _require(
            args,
            "output",
            "protected_expert_pointer",
            "h10_adapter_receipt",
            "active_gate_contract",
            "frozen_specialist_registry",
            "research_control_registry",
            "adapter_training_activation",
        )
        result = receipts.generate_peak_r195_preservation_receipt(
            output_path=args.output,
            parent_checkpoint=args.parent_checkpoint,
            protected_expert_pointer=args.protected_expert_pointer,
            h10_adapter_receipt=args.h10_adapter_receipt,
            active_gate_contract=args.active_gate_contract,
            frozen_specialist_registry=args.frozen_specialist_registry,
            research_control_registry=args.research_control_registry,
            adapter_training_activation=args.adapter_training_activation,
            **common,
        )
    else:
        _require(
            args,
            "model_output",
            "matchup_output",
            "terminal_parent_checkpoint",
            "terminal_checkpoint",
            "terminal_refresh_receipt",
            "terminal_rehearsal_receipt",
        )
        result = receipts.generate_terminal_runtime_receipts(
            model_output_path=args.model_output,
            matchup_output_path=args.matchup_output,
            r195_parent_checkpoint=args.parent_checkpoint,
            terminal_parent_checkpoint=args.terminal_parent_checkpoint,
            terminal_checkpoint=args.terminal_checkpoint,
            terminal_refresh_receipt=args.terminal_refresh_receipt,
            terminal_rehearsal_receipt=args.terminal_rehearsal_receipt,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except receipts.R241CheckpointReceiptError as exc:
        print(f"r241 checkpoint receipt generation failed: {exc}", file=sys.stderr)
        raise SystemExit(78)
