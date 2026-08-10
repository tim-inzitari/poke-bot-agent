"""Materialize one sealed, offline r207 frozen-leaf calibration bundle.

The three inputs are existing immutable r195 NO-RTP game/inference captures.
This command does not start a simulator, model server, remote job, BO1000, or
training workload.  It only creates a fresh output directory and fails closed
if the captures cannot pass the existing source-exclusion and calibration
compiler.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from poke_bot.recursive_turn_planner.r207_frozen_leaf_calibration import (
    FrozenLeafCalibrationError,
)
from poke_bot.recursive_turn_planner.r207_frozen_leaf_materialization import (
    R207FrozenLeafMaterializationError,
    materialize_r207_frozen_leaf_calibration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new sealed r207 frozen-leaf calibration bundle from existing "
            "r195 NO-RTP captures; does not run a model, simulator, service, or remote job."
        )
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--terminal-capture", type=Path, required=True)
    parser.add_argument("--frozen-inference-capture", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="A new, non-existing directory. Existing output is never overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize_r207_frozen_leaf_calibration(
            source_index_path=args.source_index,
            terminal_capture_path=args.terminal_capture,
            frozen_inference_capture_path=args.frozen_inference_capture,
            output_dir=args.output_dir,
        )
    except (FrozenLeafCalibrationError, R207FrozenLeafMaterializationError) as exc:
        print(f"r207 frozen-leaf materialization failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
