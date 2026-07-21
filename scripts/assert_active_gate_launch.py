#!/usr/bin/env python3
"""Fail closed unless the selected production gate is exact and non-legacy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE_PATH = ROOT / "poke_bot" / "pure_rl" / "strong_public_gate.py"
_SPEC = importlib.util.spec_from_file_location("pokebot_gate_contract", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery
    raise RuntimeError(f"cannot load gate contract validator: {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
load_active_gate_contract = _MODULE.load_active_gate_contract


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = _args()
    path = args.contract.expanduser().resolve()
    raw = path.read_bytes()
    contract = load_active_gate_contract(path)
    gate = contract["next_gate"]
    evaluation = gate["evaluation"]
    roster = gate["roster"]
    receipt = {
        "schema": "poke_bot.active_gate_launch_assertion/v1",
        "asserted_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(path),
        "contract_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "active_gate_id": str(gate["id"]),
        "opponent_ids": [str(row["opponent_id"]) for row in roster],
        "opponents": len(roster),
        "games_total": int(evaluation["games_total"]),
        "games_per_opponent": int(evaluation["games_per_opponent"]),
        "seat0_games_per_opponent": int(
            evaluation["seat0_games_per_opponent"]
        ),
        "seat1_games_per_opponent": int(
            evaluation["seat1_games_per_opponent"]
        ),
        "mode": str(evaluation["mode"]),
        "passed": True,
    }
    if args.receipt is not None:
        _atomic_json(args.receipt.expanduser().resolve(), receipt)
    print(
        "ACTIVE_GATE_LAUNCH_ASSERT PASS "
        f"id={receipt['active_gate_id']} opponents={receipt['opponents']} "
        f"games={receipt['games_total']} "
        f"per_opponent={receipt['games_per_opponent']} "
        f"seats={receipt['seat0_games_per_opponent']}/"
        f"{receipt['seat1_games_per_opponent']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
