#!/usr/bin/env python3
"""Print held-out win-rate progression for a pure-RL run (stdlib only).

Reads ``outputs/pure_rl/<run>/metrics/iter_*.json`` (already written every
iteration by ``train_pure_rl.py``) and prints one compact trend line. No
torch / project imports required, so it stays safe to call from the bash
watch script even while the trainer process is mid-restart.

Usage::

    python3 scripts/pure_rl_wr_trend.py
    python3 scripts/pure_rl_wr_trend.py --run-dir outputs/pure_rl/<run_name>
    python3 scripts/pure_rl_wr_trend.py --last 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def _auto_run_dir(pure_rl_dir: Path) -> Optional[Path]:
    """Most recently touched run (by metrics/latest.json mtime)."""
    candidates = sorted(
        pure_rl_dir.glob("*/metrics/latest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent.parent if candidates else None


def _load_iter_metrics(run_dir: Path) -> list[dict]:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return []
    rows = []
    for path in sorted(metrics_dir.glob("iter_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    rows.sort(key=lambda r: int(r.get("iteration", -1)))
    return rows


def _wr_trend_line(run_dir: Path, *, last_n: int, gate_wr: Optional[float]) -> str:
    rows = _load_iter_metrics(run_dir)
    if not rows:
        return "wr_trend: (no iter metrics yet)"
    known = [r for r in rows if r.get("heldout_wr") is not None]
    if not known:
        return f"wr_trend: (iter 0..{rows[-1].get('iteration')} — no heldout_wr yet)"
    tail = known[-max(1, int(last_n)):]
    steps = " -> ".join(
        f"{int(r.get('iteration', -1))}:{float(r.get('heldout_wr') or 0.0):.1%}"
        for r in tail
    )
    best = max(known, key=lambda r: float(r.get("heldout_wr") or 0.0))
    latest = known[-1]
    gate_bit = f" gate={float(gate_wr):.0%}" if gate_wr is not None else ""
    passed_bit = " GATE_PASSED" if latest.get("gate_passed") else ""
    return (
        f"wr_trend[{steps}] "
        f"best={float(best.get('heldout_wr') or 0.0):.1%}@iter{best.get('iteration')} "
        f"latest_games={int(latest.get('heldout_games') or 0)}"
        f"{gate_bit}{passed_bit}"
    )


def _gate_wr_from_manifest(run_dir: Path) -> Optional[float]:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        gw = data.get("gate_wr")
        return float(gw) if gw is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="outputs/pure_rl/<run_name> (default: most recently active run)",
    )
    p.add_argument("--last", type=int, default=12, help="iters to show in the trend")
    p.add_argument("--gate-wr", type=float, default=None, help="override gate target")
    args = p.parse_args(argv)

    pure_rl_dir = ROOT / "outputs" / "pure_rl"
    run_dir = args.run_dir
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if run_dir is None:
        run_dir = _auto_run_dir(pure_rl_dir)
    if run_dir is None or not run_dir.is_dir():
        print("wr_trend: no pure_rl runs found under outputs/pure_rl/", file=sys.stderr)
        return 1

    gate_wr = args.gate_wr if args.gate_wr is not None else _gate_wr_from_manifest(run_dir)
    print(f"[{run_dir.name}] " + _wr_trend_line(run_dir, last_n=args.last, gate_wr=gate_wr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
