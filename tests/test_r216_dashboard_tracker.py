"""Focused checks for the read-only r216 BO1000 dashboard projection."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import dashboard_snapshot as dashboard


def _r216_root(root: Path) -> Path:
    return (
        root
        / "outputs/evaluations/alakazam-local-approximate-belief-mcts-r216-bo1000"
    )


def test_r216_tracker_projects_launcher_summary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    output = _r216_root(tmp_path)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "status": "running",
                "completed_games": 412,
                "total_games": 1000,
                "active_workers": 12,
                "experimental": {"wins": 210, "draws": 18, "losses": 184},
                "control": {"wins": 184, "draws": 18, "losses": 210},
                "fallback_count": 9,
                "simulations_total": 91_234,
                "genuine_mcts_turns": 2_842,
                "average_depth": 5.75,
                "max_depth": 18,
                "games_per_hour": 40.0,
                "started_at_utc": "2026-08-10T19:00:00Z",
                "updated_at_utc": "2026-08-10T20:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    state = dashboard.r216_local_approximate_belief_mcts_state()

    assert state["available"] is True
    assert state["status"] == "running"
    assert state["completed_games"] == 412
    assert state["total_games"] == 1000
    assert state["active_workers"] == 12
    assert state["mcts"] == {"wins": 210, "draws": 18, "losses": 184}
    assert state["direct"] == {"wins": 184, "draws": 18, "losses": 210}
    assert state["fallbacks"] == 9
    assert state["simulations_total"] == 91_234
    assert state["genuine_mcts_turns"] == 2_842
    assert state["average_depth"] == 5.75
    assert state["max_depth"] == 18
    assert state["games_per_hour"] == 40.0
    assert state["eta_seconds"] == 52_920.0
    assert state["no_kaggle"] is True
    assert state["training_eligible"] is False
    assert "local_approximate_belief_mcts_non_exact" in state["labels"]


def test_r216_tracker_uses_jsonl_when_summary_is_not_written(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    output = _r216_root(tmp_path)
    output.mkdir(parents=True)
    (output / "progress.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"event": "game_completed", "game_id": "pair-001-a"}),
                json.dumps({"event": "game_completed", "game_id": "pair-001-b"}),
                json.dumps(
                    {
                        "event": "heartbeat",
                        "status": "running",
                        "active_workers": {"count": 4},
                        "mcts_wins": 1,
                        "mcts_draws": 0,
                        "mcts_losses": 1,
                        "direct_wins": 1,
                        "direct_draws": 0,
                        "direct_losses": 1,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    state = dashboard.r216_local_approximate_belief_mcts_state()

    assert state["available"] is True
    assert state["completed_games"] == 2
    assert state["active_workers"] == 4
    assert state["mcts"] == {"wins": 1, "draws": 0, "losses": 1}
    assert state["direct"] == {"wins": 1, "draws": 0, "losses": 1}


def test_dashboard_replaces_the_stale_r216_card_with_turn_pool_tracker() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'data-card="mctseval" data-widget="mctseval"' in html
    assert 'data-widget-toggle="mctseval"' in html
    assert "mcts-simulations" in html
    assert "mcts-decisions" in html
    assert "mcts-depth" in html
    assert "NO KAGGLE API / QUEUE / UPLOAD / SUBMISSION" in html
