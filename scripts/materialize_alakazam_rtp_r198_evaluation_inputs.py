#!/usr/bin/env python3
"""Create the immutable r198 A/B/C evaluation-input bundle.

This command intentionally performs no service, selector, trainer, promotion,
or Kaggle operation.  It consumes a pre-built/probed private pairing engine
capability and a sealed evaluator base specification, then creates the
evaluation-only cohort, 1,000 sealed cell snapshots, two separate planner
preflight fixtures, the pass preflight receipt, evaluator-v2 manifest, and
evaluation-only authorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.rtp_r198_evaluation_input_materializer import (  # noqa: E402
    R198EvaluationInputError,
    materialize_r198_evaluation_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--research-control-registry", type=Path, required=True)
    parser.add_argument("--pairing-capability", type=Path, required=True)
    parser.add_argument("--evaluator-base-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-nonce",
        help="Optional unique safe token; omit to generate a non-reusable nonce.",
    )
    args = parser.parse_args()
    try:
        result = materialize_r198_evaluation_inputs(
            completion_receipt=args.completion_receipt,
            research_control_registry=args.research_control_registry,
            pairing_capability=args.pairing_capability,
            evaluator_base_spec=args.evaluator_base_spec,
            output_root=args.output_root,
            run_nonce=args.run_nonce,
        )
    except R198EvaluationInputError as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
