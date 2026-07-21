"""Live win-rate progression surfacing: tqdm postfix, heldout live WR, trend line.

User ask: "id like to see winrate progressing too when possible" — these tests
cover the three lightweight additions:
  1. ``_TqdmProgress.set_wr`` puts a WR readout on the bar postfix.
  2. ``_consume_results(..., live_wr_gate=...)`` streams a running WR as
     heldout games land (official baselines only; regular collect is unaffected).
  3. ``poke_bot.pure_rl.wr_trend.wr_trend_line`` renders WR history across
     iterations from the metrics JSON files already written each iter.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_train_pure_rl():
    path = ROOT / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_wr", path)
    assert spec is not None and spec.loader is not None
    sys.modules.pop("train_pure_rl_wr", None)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeWriter:
    """Minimal stand-in: only ``n_decisions`` / ``write_game`` are touched."""

    def __init__(self) -> None:
        self.n_decisions = 0
        self.games_written: list[Any] = []

    def write_game(self, game: Any) -> None:
        self.games_written.append(game)


def test_tqdm_set_wr_puts_readout_on_postfix() -> None:
    mod = _load_train_pure_rl()
    prog = mod._TqdmProgress(
        stage="heldout",
        iteration=3,
        total=200,
        remotes=0,
        inplace=False,
        mininterval=60.0,
    )
    try:
        assert prog._bar.ncols >= 160
        assert "wr" not in prog._postfix(sps="0")
        prog.set_wr(0.42, 50, target=0.70)
        pf = prog._postfix(sps="0")
        assert "wr" in pf
        assert "42.00%" in pf["wr"]
        assert "50g" in pf["wr"]
        assert "70%" in pf["wr"]
    finally:
        prog.close()


def test_tqdm_set_wr_noop_on_zero_games() -> None:
    mod = _load_train_pure_rl()
    prog = mod._TqdmProgress(
        stage="heldout", iteration=0, total=10, inplace=False, mininterval=60.0
    )
    try:
        prog.set_wr(0.0, 0, target=0.70)
        assert "wr" not in prog._postfix(sps="0")
    finally:
        prog.close()


def test_consume_results_live_wr_gate_tracks_running_winrate() -> None:
    """Heldout gate stream: WR should update after each official-baseline result."""
    mod = _load_train_pure_rl()
    prog = mod._TqdmProgress(
        stage="heldout", iteration=1, total=4, inplace=False, mininterval=60.0
    )
    writer = _FakeWriter()
    rows: list[dict[str, Any]] = []
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
    }
    # 2 wins, 1 loss, 1 forfeit (excluded) vs official baselines.
    results = [
        {"opponent_id": "iono", "our_seat": 0, "winner": 0, "baseline_failed": False},
        {"opponent_id": "dragapult-ex", "our_seat": 1, "winner": 1, "baseline_failed": False},
        {"opponent_id": "mega-abomasnow-ex", "our_seat": 0, "winner": 1, "baseline_failed": False},
        {"opponent_id": "mega-lucario-ex", "our_seat": 0, "winner": 0, "baseline_failed": True},
    ]
    try:
        mod._consume_results(
            iter(results),
            writer,
            rows,
            stats,
            progress=prog,
            live_wr_gate=(0.70, 4),
        )
    finally:
        prog.close()
    assert len(rows) == 4
    # win, win, loss counted (3 games); forfeit excluded from the WR readout.
    assert prog.wr is not None
    assert "3g" in prog.wr
    assert "66.67%" in prog.wr


def test_consume_results_without_live_wr_gate_leaves_wr_unset() -> None:
    """Regular collect waves (no live_wr_gate) must not paint a misleading WR."""
    mod = _load_train_pure_rl()
    prog = mod._TqdmProgress(
        stage="collect:public_mix", iteration=1, total=1, inplace=False, mininterval=60.0
    )
    writer = _FakeWriter()
    rows: list[dict[str, Any]] = []
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
    }
    try:
        mod._consume_results(
            iter([{"opponent_id": "iono", "our_seat": 0, "winner": 0}]),
            writer,
            rows,
            stats,
            progress=prog,
        )
    finally:
        prog.close()
    assert prog.wr is None


def _write_iter_metrics(run_dir: Path, iteration: int, heldout_wr, games=200, gate_passed=False):
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "heldout_wr": heldout_wr,
        "heldout_games": games,
        "gate_passed": gate_passed,
    }
    (metrics_dir / f"iter_{iteration:05d}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_wr_trend_line_renders_progression(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from poke_bot.pure_rl.wr_trend import wr_trend_line

    run_dir = tmp_path / "pure_rl_test_run"
    for it, wr in enumerate([0.01, 0.15, 0.22, 0.71]):
        _write_iter_metrics(run_dir, it, wr, gate_passed=(wr >= 0.70))

    line = wr_trend_line(run_dir, gate_wr=0.70)
    assert "0:1.0%" in line
    assert "3:71.0%" in line
    assert "best=71.0%@iter3" in line
    assert "gate=70%" in line
    assert "GATE_PASSED" in line


def test_wr_trend_line_handles_missing_metrics(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from poke_bot.pure_rl.wr_trend import wr_trend_line

    line = wr_trend_line(tmp_path / "no_such_run")
    assert "no iter metrics yet" in line


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
