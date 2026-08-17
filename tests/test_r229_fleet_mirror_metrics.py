from __future__ import annotations

import pytest

from poke_bot.r229_fleet_mirror_metrics import R229MetricsError, summarize_games


def _rows():
    rows = []
    for pair in range(500):
        for mcts_seat in (0, 1):
            decisions = []
            for index in range(10):
                searched = index < 8
                decisions.append({
                    "mode": (
                        "shared_tree_mcts" if searched
                        else "clean_deadline_zero_backup_frozen_model_fallback"
                    ),
                    "selection_context": "SelectPokemon",
                    "actor_seat": mcts_seat,
                    "action_changed": index < 3,
                    "meaningful_choice_change": index < 2,
                    "internal_value_boundary_count": int(index == 0),
                    "internal_value_boundary_reasons": (
                        {"explicit_chance_pre_random": 1} if index == 0 else {}
                    ),
                    "max_internal_ordered_action_count": 50,
                    "internal_ordered_action_expansion_ceiling": 64,
                    "explicit_chance_probability_distribution_assumed": False,
                    "explicit_chance_always_stops_before_random_resolution": True,
                    "internal_boundary_has_action_or_child_authority": False,
                    "lane_process_recovery": {
                        "serial_lane_count": 1,
                        "attempt_count": 1,
                        "recovered_search": False,
                        "exhausted_direct_fallback": False,
                        "attempts": [
                            {"attempt": 1, "status": "complete", "new_lane_faults": []}
                        ],
                    },
                    **({
                        "search_elapsed_seconds": 1.0,
                        "completed_backups": 16,
                        "root_visits": 16,
                        "search_begin_calls": 16,
                        "search_release_calls": 32,
                        "search_end_calls": 16,
                        "rollout_count": 16,
                        "rollout_search_id_chains": [[0, 1]] * 16,
                        "root_action_visit_counts": [12, 4],
                        "distinct_root_actions_visited": 2,
                        "legal_action_count": 2,
                        "rollout_ceiling": 1000,
                        "rollout_stop_reason": "decision_deadline",
                        "microbatch_sizes": [1] * 16,
                    } if searched else {}),
                })
            rows.append({
                "game_id": f"p{pair}-s{mcts_seat}", "pair_index": pair,
                "mcts_seat": mcts_seat, "winner_seat": mcts_seat,
                "serial_rollout_revision": 253,
                "host": ("elmo", "bert", "train_inzi")[pair % 3],
                "elapsed_seconds": 2.0,
                "started_at_utc": "2026-08-10T20:00:00Z",
                "completed_at_utc": "2026-08-10T20:00:02Z",
                "decision_metrics": {"decisions_seen": 20, "mcts_seat_decisions_seen": 12, "direct_seat_decisions_seen": 8, "setup_decisions": 1, "mcts_eligible": 10, "searched": 8, "forced": 2, "fallback": 2, "recovered_searches": 0, "exhausted_recovery_direct_fallbacks": 0, "contained_native_lane_faults": 0, "internal_value_boundaries": 1, "decisions_with_internal_value_boundary": 1, "internal_explicit_chance_boundaries": 1, "internal_deterministic_fanout_boundaries": 0, "max_internal_ordered_action_count": 50, "serial_root_rollouts": 128, "decisions_visiting_multiple_root_actions": 8, "max_distinct_root_actions_visited": 2, "action_changed": 3, "meaningful_choice_change": 2},
                "decision_latency_seconds": {
                    "mcts_seat_all": [2.0] * 12,
                    "direct_r195_seat_all": [0.25] * 8,
                    "deterministic_setup": [0.001],
                },
                "mcts_decisions": decisions,
            })
    return rows


def test_complete_summary_reports_decision_averages_changes_and_throughput():
    summary = summarize_games(_rows())
    assert summary["games"] == 1000
    assert summary["outcomes"]["mcts_win"] == 1000
    assert summary["decisions"]["seen_per_game"]["mean"] == 20
    assert summary["decisions"]["seen_total"] == 20000
    assert summary["decisions"]["action_changed_total"] == 3000
    assert summary["decisions"]["meaningful_choice_change_total"] == 2000
    assert summary["decisions"]["serial_root_rollouts_total"] == 128000
    assert summary["decisions"]["decisions_visiting_multiple_root_actions_total"] == 8000
    assert set(summary["throughput"]["by_host"]) == {"elmo", "bert", "train_inzi"}
    assert summary["decisions"]["influence_by_stage"]["SelectPokemon"]["decisions"] == 10000
    assert summary["search"]["mean_mcts_to_direct_decision_latency_ratio"] == 8.0
    assert summary["process_lane_recovery"]["recovered_searches_total"] == 0
    assert summary["internal_leaf_boundaries"]["total"] == 1000
    assert summary["internal_leaf_boundaries"]["reasons"] == {
        "explicit_chance_pre_random": 1000
    }


def test_duplicate_game_or_broken_influence_counts_fail_closed():
    rows = _rows()
    rows[1]["game_id"] = rows[0]["game_id"]
    with pytest.raises(R229MetricsError, match="duplicated"):
        summarize_games(rows)
    rows = _rows()
    rows[0]["decision_metrics"]["meaningful_choice_change"] = 9
    with pytest.raises(R229MetricsError, match="inconsistent"):
        summarize_games(rows)


def test_clean_fallback_without_search_fields_is_counted_not_misparsed():
    rows = _rows()
    rows[0]["mcts_decisions"][0] = {
        "mode": "clean_deadline_zero_backup_frozen_model_fallback",
        "selection_context": "SelectPokemon", "actor_seat": 0,
        "action_changed": False,
        "internal_value_boundary_count": 1,
        "internal_value_boundary_reasons": {"explicit_chance_pre_random": 1},
        "max_internal_ordered_action_count": 50,
        "internal_ordered_action_expansion_ceiling": 64,
        "explicit_chance_probability_distribution_assumed": False,
        "explicit_chance_always_stops_before_random_resolution": True,
        "internal_boundary_has_action_or_child_authority": False,
        "lane_process_recovery": {
            "serial_lane_count": 1,
            "attempt_count": 1, "recovered_search": False,
            "exhausted_direct_fallback": False,
            "attempts": [{"attempt": 1, "status": "failed", "new_lane_faults": []}],
        },
    }
    rows[0]["mcts_decisions"][8] = {
        "mode": "shared_tree_mcts", "search_elapsed_seconds": 1.0,
        "completed_backups": 16, "root_visits": 16,
        "search_begin_calls": 16, "search_release_calls": 32,
        "search_end_calls": 16, "rollout_count": 16,
        "rollout_search_id_chains": [[0, 1]] * 16,
        "root_action_visit_counts": [12, 4],
        "distinct_root_actions_visited": 2, "legal_action_count": 2,
        "rollout_ceiling": 1000, "rollout_stop_reason": "decision_deadline",
        "microbatch_sizes": [1] * 16,
        "selection_context": "SelectPokemon", "actor_seat": 0,
        "action_changed": True, "meaningful_choice_change": True,
        "internal_value_boundary_count": 0,
        "internal_value_boundary_reasons": {},
        "max_internal_ordered_action_count": 50,
        "internal_ordered_action_expansion_ceiling": 64,
        "explicit_chance_probability_distribution_assumed": False,
        "explicit_chance_always_stops_before_random_resolution": True,
        "internal_boundary_has_action_or_child_authority": False,
        "lane_process_recovery": {
            "serial_lane_count": 1,
            "attempt_count": 1, "recovered_search": False,
            "exhausted_direct_fallback": False,
            "attempts": [{"attempt": 1, "status": "complete", "new_lane_faults": []}],
        },
    }
    summary = summarize_games(rows)
    assert summary["search"]["latency_seconds"]["mean"] == 1.0
    assert summary["decisions"]["influence_by_stage"]["SelectPokemon"]["decisions"] == 10000
