from pathlib import Path

import pytest

from scripts.train_round_robin import (
    _build_selfplay_record,
    _validate_contract_migration_boundary,
)


def test_contract_migration_requires_clean_completed_n_plus_one(
    tmp_path: Path,
) -> None:
    target = tmp_path / "iter103.jsonl"
    _validate_contract_migration_boundary(
        completed_iteration=102,
        effective_iteration=103,
        finalized_paths=[target],
    )
    with pytest.raises(RuntimeError, match="N\\+1 boundary"):
        _validate_contract_migration_boundary(
            completed_iteration=101,
            effective_iteration=103,
            finalized_paths=[target],
        )
    target.write_text("{}\n")
    with pytest.raises(RuntimeError, match="finalized"):
        _validate_contract_migration_boundary(
            completed_iteration=102,
            effective_iteration=103,
            finalized_paths=[target],
        )


def test_adaptive_move_budget_is_diagnostic_not_mixed_provenance() -> None:
    provenance = {
        "checkpoint_digest": "sha256:current",
        "model_generation": 3,
        "search_config": {
            "algorithm": "public_history_root_sampled_information_set_mcts",
            "max_sims": 128,
            "min_trusted_sims": 128,
            "move_time_s": 8.0,
            "tree_reuse": False,
            "adaptive_sequential_updates": True,
            "cross_game_batching_only": True,
            "clock_allocation": (
                "adaptive_per_game_fair_share_with_watchdog_reserve"
            ),
        },
        "belief_config": {
            "sampler": "public-particles-v2",
            "mode": "particles",
            "model_digest": "sha256:belief",
            "conserves_card_multiplicity": True,
            "uses_baseline_identity": False,
        },
        "simulator_version": "competition-libcg-sha256:test",
    }
    observation = {
        "select": {
            "option": [{"type": 14}, {"type": 14}],
            "minCount": 1,
            "maxCount": 1,
        }
    }

    def target(move_budget: float) -> dict:
        return {
            "observation": observation,
            "action": [0],
            "factorized_stages": [
                {
                    "action_combos": [[0], [1]],
                    "policy": [1.0, 0.0],
                }
            ],
            "target_source": "belief_mcts",
            "provenance": provenance,
            "diagnostics": {
                "sims_run": 128,
                "sims_planned": 128,
                "unique_expanded_nodes": 2,
                "max_depth": 2,
                "mean_depth": 1.0,
                "mean_branching": 2.0,
                "leaf_evaluations": 128,
                "chance_samples": 0,
                "unique_particles": 2,
                "root_visits": 128,
                "queue_wait_ms_mean": 0.0,
                "inference_batch_size_mean": 1.0,
                "sims_per_s": 16.0,
                "elapsed_s": move_budget,
                "move_budget_s": move_budget,
                "trusted": True,
            },
        }

    record = _build_selfplay_record(
        [target(8.0), target(6.5)],
        our_deck=[1] * 60,
        our_seat=0,
        value=1.0,
        opp_id="test",
        archetype="test",
        seed=1,
        target_provenance={
            "incumbent_checkpoint": {"digest": "sha256:current"},
            "model_generation": 3,
        },
    )
    assert record is not None
    assert len(record["steps"]) == 2
