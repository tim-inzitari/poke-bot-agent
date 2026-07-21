"""Cross-iteration held-out win-rate trend (lightweight, stdlib-only).

Reads the per-iteration metrics JSON files a pure-RL run already writes
(``<run_dir>/metrics/iter_*.json``) and renders a compact WR progression
line for logs / watch scripts. No new dashboards, no extra state files —
just a summary of data that already exists on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_iter_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Return iteration metrics dicts sorted by iteration (best-effort)."""
    metrics_dir = Path(run_dir) / "metrics"
    if not metrics_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("iter_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    rows.sort(key=lambda r: int(r.get("iteration", -1)))
    return rows


def wr_trend_line(
    run_dir: Path,
    *,
    last_n: int = 12,
    gate_wr: Optional[float] = None,
) -> str:
    """One-line WR progression summary: last N iters + best-so-far + gate."""
    rows = load_iter_metrics(run_dir)
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
