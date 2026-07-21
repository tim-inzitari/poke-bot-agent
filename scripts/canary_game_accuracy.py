#!/usr/bin/env python3
"""Live CABT game-accuracy canary for multi-env / throughput deploys.

Fail-closed: exit 2 if isolation / wrapper / playthrough checks fail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CG = ROOT / (
    "kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission"
)
DEFAULT_DECK = DEFAULT_CG / "deck.csv"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cg-parent",
        type=Path,
        default=Path(
            os.environ.get("CG_LIB_PATH", str(DEFAULT_CG / "cg"))
        ).resolve().parent
        if (DEFAULT_CG / "cg").is_dir()
        else DEFAULT_CG,
        help="Directory that contains the competition ``cg`` package",
    )
    p.add_argument("--deck-csv", type=Path, default=DEFAULT_DECK)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument(
        "--expected-libcg-sha256",
        default=os.environ.get("POKEBOT_EXPECTED_LIBCG_SHA256", ""),
        help="Fail unless the loaded original libcg has this SHA-256",
    )
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    cg_parent = args.cg_parent
    # Allow passing …/cg directly.
    if (cg_parent / "sim.py").is_file() and cg_parent.name == "cg":
        cg_parent = cg_parent.parent
    deck = args.deck_csv
    if not (cg_parent / "cg" / "libcg.so").is_file() and not (
        cg_parent / "cg" / "libcg.dylib"
    ).is_file():
        # Also accept CG_LIB_PATH pointing at cg/
        alt = Path(os.environ.get("CG_LIB_PATH", ""))
        if alt.is_dir() and (
            (alt / "libcg.so").is_file() or (alt / "libcg.dylib").is_file()
        ):
            cg_parent = alt.parent
            if not deck.is_file():
                cand = cg_parent / "deck.csv"
                if cand.is_file():
                    deck = cand

    if not (cg_parent / "cg").is_dir():
        print(f"ERROR: cg package not found under {cg_parent}", file=sys.stderr)
        return 2
    if not deck.is_file():
        print(f"ERROR: deck csv missing: {deck}", file=sys.stderr)
        return 2

    from poke_bot.engine_rebuild.live_accuracy import run_live_accuracy_suite

    report = run_live_accuracy_suite(
        cg_parent=cg_parent,
        deck_csv=deck,
        num_envs=max(2, int(args.num_envs)),
        expected_libcg_sha256=args.expected_libcg_sha256 or None,
    )
    print(json.dumps(report.to_dict(), indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    if report.ok:
        print("GAME_ACCURACY_OK")
        return 0
    print("GAME_ACCURACY_FAIL", file=sys.stderr)
    for c in report.checks:
        if not c.ok:
            print(f"  FAIL {c.name}: {c.detail}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
