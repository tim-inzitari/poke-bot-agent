#!/usr/bin/env python3
"""Publish a versioned terminal marker from one exact committed gate pass.

The marker is only a wake-up signal for ``handle_passed_gate.py``.  This
bridge is needed when a stricter gate supersedes an already archived first
pass while the learner keeps the same append-only run lineage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.handle_passed_gate import (
    _canonical_digest,
    _read_json,
    validate_exact_pass,
)


def _atomic_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"versioned gate marker identity changed: {path}")


def materialize_if_passed(
    run_dir: Path,
    contract_path: Path,
    marker_name: str,
) -> Path | None:
    run_dir = Path(run_dir).expanduser().resolve()
    contract_path = Path(contract_path).expanduser().resolve()
    contract = _read_json(contract_path)
    gate = dict(contract.get("next_gate") or {})
    evaluation = dict(gate.get("evaluation") or {})
    criteria = dict(gate.get("pass_criteria") or {})
    if (
        not gate
        or str(contract.get("active_gate_id") or "") != str(gate.get("id") or "")
        or not evaluation
        or not criteria
    ):
        raise RuntimeError("active gate contract is absent/invalid")
    if (
        not marker_name
        or Path(marker_name).name != marker_name
        or not marker_name.startswith("SPECIALIST_GATE_PASSED")
    ):
        raise RuntimeError("terminal marker name is invalid")

    pointer_path = Path(str(gate.get("exact_result_pointer") or "")).expanduser()
    pointer = _read_json(pointer_path)
    if (
        not pointer
        or pointer.get("committed") is not True
        or str(pointer.get("gate_id") or "") != str(gate["id"])
        or pointer.get("passed") is not True
    ):
        return None

    iteration = int(pointer.get("iteration", -1))
    commit_path = (run_dir / "commits" / f"iter_{iteration:05d}.json").resolve()
    commit = _read_json(commit_path)
    pointer_core = {
        key: value
        for key, value in pointer.items()
        if key not in {"committed", "commit", "commit_digest", "created_at_utc"}
    }
    rows = [
        dict(row)
        for row in (commit.get("history") or [])
        if isinstance(row, dict) and int(row.get("iteration", -2)) == iteration
    ]
    if (
        len(rows) != 1
        or str(pointer.get("commit") or "") != str(commit_path)
        or str(pointer.get("commit_digest") or "") != _canonical_digest(commit)
        or rows[0].get("completed") is not True
        or dict(rows[0].get("active_gate_result") or {}) != pointer_core
    ):
        raise RuntimeError("exact result pointer is not bound to one committed pass")

    result = pointer_core
    checks = dict(result.get("checks") or {})
    required_checks = {
        "audit",
        "skill_weighted_win_rate",
        "skill_weighted_confidence_lower",
        "s_tier_mean_floor",
        "individual_opponent_floor",
    }
    if "s_plus_individual_floor" in criteria:
        required_checks.add("s_plus_matchup_floor_allowance")
    if set(checks) != required_checks or not all(checks.values()):
        raise RuntimeError("exact pass does not satisfy the complete active gate")
    if float(result.get("confidence_lower", -1.0)) < float(
        criteria["skill_weighted_confidence_lower"]
    ):
        raise RuntimeError("exact pass is below the active lower-confidence threshold")

    candidate = dict(rows[0].get("candidate") or {})
    marker = {
        "iteration": iteration,
        "wr": float(result["skill_weighted_wr"]),
        "confidence_lower": float(result["confidence_lower"]),
        "games": int(result["games"]),
        "checkpoint": str(candidate["path"]),
        "checkpoint_digest": str(candidate["digest"]),
    }
    marker_path = run_dir / marker_name
    _atomic_exclusive(marker_path, marker)
    validate_exact_pass(run_dir, contract_path, marker_name=marker_name)
    return marker_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--marker-name", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    while True:
        marker = materialize_if_passed(
            args.run_dir,
            args.contract,
            args.marker_name,
        )
        if marker is not None:
            print(f"versioned exact gate marker ready: {marker}", flush=True)
            return 0
        if args.once:
            return 0
        time.sleep(max(float(args.poll_seconds), 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
