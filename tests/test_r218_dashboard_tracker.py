"""Focused coverage for the read-only distributed turn-pool MCTS tracker."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import dashboard_snapshot as dashboard

R222_SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r222/v1"
R222_BO1000 = "alakazam-r222-local-multi-search-turn-belief-mcts-bo1000"


def _root(root: Path) -> Path:
    return (
        root
        / "outputs/evaluations"
        / "alakazam-local-first-decision-belief-mcts-r218-bo1000-attempt2"
    )


def _contract() -> dict[str, object]:
    return {
        "owner_decision_revision": 222,
        "evaluation_id": R222_BO1000,
        "no_kaggle_submission": True,
        "runtime": {
            "matchup_adapter_required_on_both_arms": True,
            "rtp_enabled": False,
            "guide_linear_enabled": False,
            "guide_logit_enabled": False,
            "guide2vec_enabled": False,
            "kaggle_authority": False,
        },
        "timing": {
            "default_turn_pool_seconds": 45.0,
            "meaningful_search_segment_seconds": 15.0,
            "later_residual_search_allowed": True,
            "deterministic_cache_skips": True,
            "dynamic_game_clock_shrink": True,
        },
    }


def _write_shard(
    root: Path,
    host: str,
    *,
    completed: int,
    valid: int,
    invalid: int,
    target: int,
    mcts_wins: int,
    direct_wins: int,
    draws: int,
    selected: int,
    fallbacks: int,
    simulations: int,
    average_depth: float,
    max_depth: int,
    workers: int,
    pair_index: int = 0,
    game_number: int = 0,
    updated_at_utc: str = "2026-08-10T21:00:00Z",
    finite_enumerations: int | None = None,
    unforceable_boundaries: int | None = None,
) -> None:
    shard = root / "shards" / host
    shard.mkdir(parents=True)
    (shard / "run-contract.json").write_text(
        json.dumps(_contract()), encoding="utf-8"
    )
    (shard / "summary.json").write_text(
        json.dumps(
            {
                "schema": R222_SCHEMA,
                "evaluation_id": R222_BO1000,
                "status": "running",
                "completed_games": completed,
                "valid_games": valid,
                "invalid_games": invalid,
                "total_games": 1000,
                "shard_total_games": target,
                "active_worker_limit": workers,
                "current_pair_index": pair_index,
                "current_game_number": game_number,
                "prefix_target_games": 10,
                "prefix_completed_games": min(10, completed),
                "runtime_transport_identity": "stock_r195_libcg",
                "pair_rng_streams": "independent_unmatched",
                "shared_tree_lane_telemetry": {
                    "requested_lane_count": 8,
                    "active_lane_count": 8,
                    "shared_logical_tree_id": f"tree-{host}",
                    "shared_root_visits": completed * 10,
                    "completed_backed_simulations": completed * 8,
                    "backed_simulations_per_second": 24.5,
                    "outstanding_reservations": 0,
                    "outstanding_virtual_loss": 0,
                    "leaf_microbatch_sizes": [2, 4, 8],
                    "throughput_ratio_eight_over_one": 1.75,
                    "lane_trajectory_counts": {
                        str(lane): completed for lane in range(8)
                    },
                    "lane_backup_counts": {
                        str(lane): completed for lane in range(8)
                    },
                },
                "mcts_wins": mcts_wins,
                "direct_wins": direct_wins,
                "draws": draws,
                "mcts_search_decisions": selected,
                "fallbacks": fallbacks,
                "mcts_simulations": simulations,
                "average_depth": average_depth,
                "max_depth_seen": max_depth,
                "converged_searches": 3,
                "finite_chance_enumerations": finite_enumerations,
                "unforceable_random_branch_boundaries": unforceable_boundaries,
                "elapsed_seconds": 3600.0,
                "games_per_hour": float(completed),
                "eta_seconds": (
                    (target - completed) * 3600.0 / completed if completed else None
                ),
                "updated_at_utc": updated_at_utc,
            }
        ),
        encoding="utf-8",
    )


def test_turn_pool_tracker_aggregates_train_bert_elmo_attempt2(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    root = _root(tmp_path)
    _write_shard(
        root,
        "train",
        completed=400,
        valid=390,
        invalid=10,
        target=700,
        mcts_wins=200,
        direct_wins=185,
        draws=5,
        selected=3600,
        fallbacks=80,
        simulations=100_000,
        average_depth=4.5,
        max_depth=11,
        workers=8,
        pair_index=199,
        game_number=399,
        updated_at_utc="2026-08-10T21:00:00Z",
        finite_enumerations=100,
        unforceable_boundaries=5,
    )
    _write_shard(
        root,
        "bert",
        completed=100,
        valid=100,
        invalid=0,
        target=200,
        mcts_wins=45,
        direct_wins=52,
        draws=3,
        selected=810,
        fallbacks=16,
        simulations=30_000,
        average_depth=5.0,
        max_depth=13,
        workers=4,
        pair_index=249,
        game_number=499,
        updated_at_utc="2026-08-10T21:05:00Z",
        finite_enumerations=30,
        unforceable_boundaries=2,
    )
    _write_shard(
        root,
        "elmo",
        completed=50,
        valid=50,
        invalid=0,
        target=100,
        mcts_wins=25,
        direct_wins=24,
        draws=1,
        selected=400,
        fallbacks=7,
        simulations=10_000,
        average_depth=6.0,
        max_depth=16,
        workers=2,
        pair_index=274,
        game_number=549,
        updated_at_utc="2026-08-10T21:10:00Z",
        finite_enumerations=10,
        unforceable_boundaries=1,
    )

    state = dashboard.local_turn_pool_belief_mcts_bo1000_state()

    assert state["status"] == "running"
    assert state["completed_games"] == 550
    assert state["total_games"] == 1000
    assert state["valid_games"] == 540
    assert state["invalid_games"] == 10
    assert state["unreceipted_games"] == 450
    assert state["mcts"] == {"wins": 270, "draws": 9, "losses": 261}
    assert state["direct"] == {"wins": 261, "draws": 9, "losses": 270}
    assert state["mcts_selected_decisions"] == 4810
    assert state["fallbacks"] == 103
    assert state["simulations_total"] == 140_000
    assert state["average_depth"] == (4.5 * 400 + 5.0 * 100 + 6.0 * 50) / 550
    assert state["max_depth"] == 16
    assert state["active_worker_limit"] == 14
    assert [row["host"] for row in state["hosts"][:3]] == ["train", "bert", "elmo"]
    assert state["timing"]["shared_turn_pool_seconds"] == 45.0
    assert state["timing"]["search_segment_cap_seconds"] == 15.0
    assert state["timing"]["later_residual_searches"] is True
    assert state["timing"]["deterministic_cache_skips"] is True
    assert state["safety"] == {
        "adapter_on": True,
        "rtp_off": True,
        "guides_off": True,
        "kaggle_off": True,
        "training_eligible": False,
    }
    panel = state["progress_panel"]
    assert panel["available"] is True
    assert panel["revision"] == 222
    assert panel["phase"] == "bo1000"
    assert panel["completed_games"] == 550
    assert panel["target_games"] == 1000
    assert panel["percent"] == 55.0
    assert panel["prefix_completed_games"] == 10
    assert panel["prefix_target_games"] == 10
    assert panel["prefix_percent"] == 100.0
    assert panel["runtime"] == "stock r195 libcg"
    assert panel["rng"] == "independent/unmatched"
    assert panel["valid_games"] == 540
    assert panel["invalid_games"] == 10
    assert panel["finite_chance_enumerations"] == 140
    assert panel["unforceable_random_branch_boundaries"] == 8
    assert panel["active_workers"] == 14
    assert panel["games_per_hour"] == 550.0
    assert panel["elapsed_seconds"] == 3600.0
    assert panel["eta_seconds"] == 3600.0
    assert panel["cursor_host"] == "elmo"
    assert panel["current_pair_index"] == 274
    assert panel["current_game_number"] == 549
    assert panel["last_receipt_source"].endswith("shards/elmo/summary.json")
    shared = panel["shared_tree"]
    assert shared["available"] is True
    assert shared["requested_lane_count"] == 8
    assert shared["active_lane_count"] == 8
    assert shared["shared_logical_tree_id"] == "tree-elmo"
    assert shared["outstanding_reservations"] == 0
    assert shared["outstanding_virtual_loss"] == 0
    assert shared["leaf_microbatch_count"] == 3
    assert shared["leaf_microbatch_mean_size"] == 14 / 3
    assert shared["leaf_microbatch_p95_size"] == 8
    assert shared["leaf_microbatch_max_size"] == 8
    assert shared["throughput_ratio_eight_over_one"] == 1.75
    assert len(shared["lanes"]) == 8
    assert shared["lanes"][0] == {
        "lane_id": 0,
        "trajectories": 50,
        "backups": 50,
    }
    assert [row["host"] for row in panel["hosts"]] == ["train", "bert", "elmo"]
    assert [row["target_games"] for row in panel["hosts"]] == [700, 200, 100]


def test_turn_pool_tracker_uses_game_jsonl_before_a_summary_exists(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    shard = _root(tmp_path) / "shards" / "train"
    shard.mkdir(parents=True)
    (shard / "run-contract.json").write_text(json.dumps(_contract()), encoding="utf-8")
    (shard / "progress.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "schema": R222_SCHEMA,
                    "evaluation_id": R222_BO1000,
                    "kind": "game",
                    "game_number": 0,
                    "pair_index": 0,
                    "valid": True,
                    "mcts_search_decisions": 7,
                    "mcts_simulations": 42,
                    "fallbacks": 1,
                    "max_depth": 4,
                },
                {
                    "schema": R222_SCHEMA,
                    "evaluation_id": R222_BO1000,
                    "kind": "game",
                    "game_number": 1,
                    "pair_index": 0,
                    "total_games": 1000,
                    "target_games": 1000,
                    "active_worker_limit": 3,
                    "games_per_hour": 60.0,
                    "elapsed_seconds": 12.0,
                    "eta_seconds": 120.0,
                    "runtime_transport_identity": "stock_r195_libcg",
                    "pair_rng_streams": "independent_unmatched",
                    "shared_tree_lane_telemetry": {
                        "requested_lane_count": 8,
                        "active_lane_count": 8,
                        "shared_logical_tree_id": "tree-jsonl",
                        "shared_root_visits": 80,
                        "completed_backed_simulations": 64,
                        "backed_simulations_per_second": 32.0,
                        "outstanding_reservations": 0,
                        "outstanding_virtual_loss": 0,
                        "leaf_microbatch_sizes": [8, 8],
                        "lane_trajectory_counts": {
                            str(lane): 1 for lane in range(8)
                        },
                        "lane_backup_counts": {
                            str(lane): 1 for lane in range(8)
                        },
                    },
                    "finite_chance_enumerations": 4,
                    "unforceable_random_branch_boundaries": 1,
                    "updated_at_utc": "2026-08-10T21:15:00Z",
                    "valid": False,
                    "mcts_search_decisions": 2,
                    "mcts_simulations": 11,
                    "fallbacks": 0,
                    "max_depth": 8,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    state = dashboard.local_turn_pool_belief_mcts_bo1000_state()

    assert state["completed_games"] == 2
    assert state["valid_games"] == 1
    assert state["invalid_games"] == 1
    assert state["mcts_selected_decisions"] == 9
    assert state["simulations_total"] == 53
    assert state["fallbacks"] == 1
    assert state["average_depth"] == 6.0
    assert state["max_depth"] == 8
    panel = state["progress_panel"]
    assert panel["available"] is True
    assert panel["phase"] == "bo1000"
    assert panel["completed_games"] == 2
    assert panel["target_games"] == 1000
    assert panel["percent"] == 0.2
    assert panel["prefix_completed_games"] == 2
    assert panel["prefix_target_games"] == 10
    assert panel["prefix_percent"] == 20.0
    assert panel["runtime"] == "stock r195 libcg"
    assert panel["rng"] == "independent/unmatched"
    assert panel["current_pair_index"] == 0
    assert panel["current_game_number"] == 1
    assert panel["active_workers"] == 3
    assert panel["games_per_hour"] == 60.0
    assert panel["finite_chance_enumerations"] == 4
    assert panel["unforceable_random_branch_boundaries"] == 1
    assert panel["shared_tree"]["shared_logical_tree_id"] == "tree-jsonl"
    assert panel["shared_tree"]["shared_root_visits"] == 80
    assert panel["shared_tree"]["shared_root_backups"] == 64
    assert panel["shared_tree"]["backed_simulations_per_second"] == 32.0


def test_turn_pool_progress_panel_awaits_receipts_without_false_zero(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)

    panel = dashboard.local_turn_pool_belief_mcts_bo1000_state()["progress_panel"]

    assert panel["available"] is False
    assert panel["phase"] is None
    assert panel["completed_games"] is None
    assert panel["target_games"] is None
    assert panel["percent"] is None
    assert panel["prefix_completed_games"] is None
    assert panel["prefix_target_games"] is None
    assert panel["prefix_percent"] is None
    assert panel["runtime"] is None
    assert panel["rng"] is None
    assert panel["shared_tree"]["available"] is False
    assert panel["shared_tree"]["requested_lane_count"] is None
    assert panel["shared_tree"]["outstanding_reservations"] is None
    assert panel["shared_tree"]["outstanding_virtual_loss"] is None
    assert panel["shared_tree"]["lanes"] == []
    assert panel["valid_games"] is None
    assert panel["invalid_games"] is None
    assert all(row["available"] is False for row in panel["hosts"])


def test_turn_pool_progress_panel_ignores_pre_r222_receipts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    root = (
        tmp_path
        / "outputs/evaluations"
        / "alakazam-local-multi-search-turn-belief-mcts-r221-bo1000"
    )
    root.mkdir(parents=True)
    (root / "run-contract.json").write_text(
        json.dumps(
            {
                **_contract(),
                "owner_decision_revision": 221,
                "evaluation_id": (
                    "alakazam-r221-local-multi-search-turn-belief-mcts-bo1000"
                ),
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.alakazam_local_multi_search_turn_belief_mcts_"
                    "bo1000_r221/v1"
                ),
                "evaluation_id": (
                    "alakazam-r221-local-multi-search-turn-belief-mcts-bo1000"
                ),
                "completed_games": 10,
                "total_games": 1000,
            }
        ),
        encoding="utf-8",
    )

    panel = dashboard.local_turn_pool_belief_mcts_bo1000_state()["progress_panel"]

    assert panel["available"] is False
    assert panel["revision"] is None
    assert panel["completed_games"] is None
    assert panel["percent"] is None


def test_turn_pool_progress_panel_reads_flat_r222_continuous_receipts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    root = (
        tmp_path
        / "outputs/evaluations"
        / "alakazam-local-multi-search-turn-belief-mcts-r222-bo1000"
    )
    root.mkdir(parents=True)
    (root / "run-contract.json").write_text(
        json.dumps(_contract()), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema": R222_SCHEMA,
                "evaluation_id": R222_BO1000,
                "status": "running",
                "completed_games": 3,
                "valid_games": 3,
                "invalid_games": 0,
                "total_games": 1000,
                "shard_total_games": 1000,
                "prefix_target_games": 10,
                "prefix_completed_games": 3,
                "runtime_transport_identity": "stock_r195_libcg",
                "pair_rng_streams": "independent_unmatched",
                "active_worker_limit": 2,
                "finite_chance_enumerations": 12,
                "unforceable_random_branch_boundaries": 2,
                "games_per_hour": 30.0,
                "elapsed_seconds": 360.0,
                "eta_seconds": 840.0,
                "updated_at_utc": "2026-08-10T21:20:00Z",
            }
        ),
        encoding="utf-8",
    )
    (root / "progress.jsonl").write_text(
        json.dumps(
            {
                "schema": R222_SCHEMA,
                "evaluation_id": R222_BO1000,
                "kind": "game",
                "pair_index": 1,
                "game_number": 2,
                "valid": True,
                "updated_at_utc": "2026-08-10T21:19:59Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    panel = dashboard.local_turn_pool_belief_mcts_bo1000_state()["progress_panel"]

    assert panel["available"] is True
    assert panel["revision"] == 222
    assert panel["phase"] == "bo1000"
    assert panel["completed_games"] == 3
    assert panel["target_games"] == 1000
    assert panel["percent"] == 0.3
    assert panel["prefix_completed_games"] == 3
    assert panel["prefix_target_games"] == 10
    assert panel["prefix_percent"] == 30.0
    assert panel["runtime"] == "stock r195 libcg"
    assert panel["rng"] == "independent/unmatched"
    assert panel["finite_chance_enumerations"] == 12
    assert panel["unforceable_random_branch_boundaries"] == 2
    assert panel["current_pair_index"] == 1
    assert panel["current_game_number"] == 2
    assert panel["hosts"][0]["host"] == "train"
    assert panel["hosts"][0]["completed_games"] == 3
    assert panel["hosts"][1]["available"] is False


def test_turn_pool_dashboard_widget_names_required_live_metrics() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'data-card="mctseval" data-widget="mctseval"' in html
    assert 'data-card="mctsprogress" data-widget="mctsprogress"' in html
    assert 'data-widget-toggle="mctsprogress"' in html
    assert "receipt-backed tqdm" in html
    assert "Latest receipt cursor" in html
    assert "Train / Bert / Elmo shard progress" in html
    assert "mcts-progress-fill" in html
    assert "mcts-progress-tqdm" in html
    assert "First-10 prefix diagnostic · no pause or gate" in html
    assert "mcts-progress-prefix-fill" in html
    assert "stock r195 libcg" in html
    assert "independent/unmatched" in html
    assert "Shared-tree eight-lane search · receipt only" in html
    assert "Requested / active lanes" in html
    assert "One shared tree ID" in html
    assert "Reservations / virtual loss" in html
    assert "Leaf microbatches" in html
    assert "8-lane / 1-lane viability" in html
    assert "Per-lane trajectories / backups" in html
    assert "Exact finite enumerations" in html
    assert "Unforceable random branch boundaries" in html
    assert "never labeled samples" in html
    assert "MCTS-selected decisions" in html
    assert "Direct fallbacks" in html
    assert "Valid / invalid receipts" in html
    assert "Unreceipted" in html
    assert "Shard receipts" in html
    assert "NO KAGGLE API / QUEUE / UPLOAD / SUBMISSION" in html
