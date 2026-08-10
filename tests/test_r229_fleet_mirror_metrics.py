from __future__ import annotations

import pytest

from poke_bot.r229_fleet_mirror_metrics import R229MetricsError, summarize_games


def _rows():
    rows = []
    for pair in range(500):
        for mcts_seat in (0, 1):
            rows.append({
                "game_id": f"p{pair}-s{mcts_seat}", "pair_index": pair,
                "mcts_seat": mcts_seat, "winner_seat": mcts_seat,
                "host": ("elmo", "bert", "train_inzi")[pair % 3],
                "elapsed_seconds": 2.0,
                "decision_metrics": {"decisions_seen": 20, "mcts_eligible": 10, "searched": 8, "forced": 2, "fallback": 0, "action_changed": 3, "meaningful_choice_change": 2},
                "mcts_decisions": [{"search_elapsed_seconds": 1.0, "completed_backups": 16, "microbatch_sizes": [8, 8]}],
            })
    return rows


def test_complete_summary_reports_decision_averages_changes_and_throughput():
    summary = summarize_games(_rows())
    assert summary["games"] == 1000
    assert summary["outcomes"]["mcts_win"] == 1000
    assert summary["decisions"]["seen_per_game"]["mean"] == 20
    assert summary["decisions"]["action_changed_total"] == 3000
    assert summary["decisions"]["meaningful_choice_change_total"] == 2000
    assert set(summary["throughput"]["by_host"]) == {"elmo", "bert", "train_inzi"}


def test_duplicate_game_or_broken_influence_counts_fail_closed():
    rows = _rows()
    rows[1]["game_id"] = rows[0]["game_id"]
    with pytest.raises(R229MetricsError, match="duplicated"):
        summarize_games(rows)
    rows = _rows()
    rows[0]["decision_metrics"]["meaningful_choice_change"] = 9
    with pytest.raises(R229MetricsError, match="inconsistent"):
        summarize_games(rows)
