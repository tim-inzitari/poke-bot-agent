#!/usr/bin/env python3
"""Managed entrypoint for the r274 expert tactical shadow overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poke_bot.tactical_sequence_materialization import (
    materialize_tactical_record_stream_overlay,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-roots", type=int, default=1200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = materialize_tactical_record_stream_overlay(
        args.records,
        checkpoint_path=args.checkpoint,
        checkpoint_digest=args.checkpoint_sha256,
        output_path=args.output,
        minimum_roots=args.minimum_roots,
        device=torch.device(args.device),
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
