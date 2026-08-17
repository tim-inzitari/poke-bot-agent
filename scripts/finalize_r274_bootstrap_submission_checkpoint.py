#!/usr/bin/env python3
"""Seal the exact r274 bootstrap into its tactical-route-off submission child."""

from __future__ import annotations

import argparse
import json

from poke_bot.r274_bootstrap_handoff import (
    materialize_bootstrap_submission_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--bootstrap-checkpoint", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--tactical-overlay", required=True)
    parser.add_argument("--gpu-bootstrap-result-receipt")
    parser.add_argument("--tactical-repair-receipt")
    parser.add_argument("--adapter-training-receipt")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-receipt", required=True)
    args = parser.parse_args()
    receipt = materialize_bootstrap_submission_checkpoint(
        base_checkpoint=args.base_checkpoint,
        bootstrap_checkpoint=args.bootstrap_checkpoint,
        expert_manifest=args.expert_manifest,
        tactical_overlay=args.tactical_overlay,
        gpu_bootstrap_result_receipt=args.gpu_bootstrap_result_receipt,
        tactical_repair_receipt=args.tactical_repair_receipt,
        adapter_training_receipt=args.adapter_training_receipt,
        output_checkpoint=args.output_checkpoint,
        output_receipt=args.output_receipt,
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
