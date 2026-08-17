#!/usr/bin/env python3
"""Create an immutable adapter-only rehearsal authorization at a clean boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from poke_bot.matchup_adapter_activation import (
    build_adapter_rehearsal_authorization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--completed-iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build_adapter_rehearsal_authorization(
        run_dir=args.run_dir,
        completed_iteration=args.completed_iteration,
        output_path=args.output,
    )
    print(proof.path)


if __name__ == "__main__":
    main()
