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


def _gate_cohort(row: dict[str, Any]) -> Optional[str]:
    """Return the immutable gate identity represented by one metrics row.

    A run may migrate from one held-out roster to another without resetting its
    iteration counter.  Win rates from those contracts are not comparable, so
    trend summaries must never select a historical best from a different gate.
    """

    extra = row.get("extra")
    if not isinstance(extra, dict):
        return None
    for field in ("active_gate_result", "raw_heldout_gate"):
        result = extra.get(field)
        if not isinstance(result, dict):
            continue
        gate_id = str(result.get("gate_id") or "").strip()
        if gate_id:
            return gate_id
    return None


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

    # Restrict the trend to the latest row's gate contract.  In particular,
    # this prevents an old four-agent score from being presented as the best
    # result for a later eight-agent gate merely because both live in the same
    # run directory.
    cohort = _gate_cohort(known[-1])
    comparable = [row for row in known if _gate_cohort(row) == cohort]
    tail = comparable[-max(1, int(last_n)):]
    steps = " -> ".join(
        f"{int(r.get('iteration', -1))}:{float(r.get('heldout_wr') or 0.0):.1%}"
        for r in tail
    )
    best = max(comparable, key=lambda r: float(r.get("heldout_wr") or 0.0))
    latest = comparable[-1]
    gate_bit = f" gate={float(gate_wr):.0%}" if gate_wr is not None else ""
    passed_bit = " GATE_PASSED" if latest.get("gate_passed") else ""
    cohort_bit = f" cohort={cohort}" if cohort is not None else ""
    return (
        f"wr_trend[{steps}] "
        f"best={float(best.get('heldout_wr') or 0.0):.1%}@iter{best.get('iteration')} "
        f"latest_games={int(latest.get('heldout_games') or 0)}"
        f"{gate_bit}{cohort_bit}{passed_bit}"
    )
