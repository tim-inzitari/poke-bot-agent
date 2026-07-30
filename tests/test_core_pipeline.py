import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from poke_bot import config, features
from poke_bot.core_kernel import (
    CoreKernel,
    core_batch_losses,
    core_kernel_config_small_3080ti,
)
from poke_bot.core_pipeline import (
    DEFAULT_CORE_LOG,
    CollectionAccounting,
    advance_pipeline_state,
    auto_size_core_workers,
    build_core_bc_jobs,
    core_to_specialist_transfer_report,
    load_pipeline_state,
    require_trusted_search,
    validate_search_target_identity,
    validate_gpu0_isolation,
)
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage


def _spec(idx: int):
    return SimpleNamespace(
        id=f"agent-{idx}",
        name=f"Agent {idx}",
        dir_name=f"agent-{idx}",
        group="test",
        source="test",
        path=f"/tmp/agent-{idx}",
    )


def test_core_contract_migration_quarantines_partial_at_phase_boundary(
    tmp_path, monkeypatch
) -> None:
    from poke_bot import paths
    from scripts import train_core_pipeline

    outputs = tmp_path / "outputs"
    data = tmp_path / "data"
    monkeypatch.setattr(paths, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(paths, "DATA_DIR", data)
    run_dir = outputs / "runs" / "migration-test"
    replay_dir = data / "core_pipeline" / "migration-test"
    run_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    (run_dir / "pipeline_state.json").write_text(
        json.dumps(
            {
                "run_name": "migration-test",
                "current_phase": "core_deep_search",
                "completed_phases": ["core_bc"],
                "artifacts": {},
            }
        )
    )
    old_contract = {
        "schema": "belief_decision_budget_v1",
        "nominal_games_per_opponent": 16,
        "min_games_per_opponent": 12,
        "max_games_per_opponent": 24,
        "target_search_decisions": 2080,
        "target_derivation": {"method": "old-pilot"},
        "balanced_seats": True,
        "old_policy_targets_equivalent": False,
        "policy_anchor_is_explicit": True,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_name": "migration-test",
                "iteration_contract": old_contract,
            }
        )
    )
    for name in (
        "core_deep_search.jsonl.partial",
        "core_deep_search.jsonl.partial.journal",
        "core_deep_search.jsonl.partial.writer.json",
    ):
        (replay_dir / name).write_text("partial")

    assert (
        train_core_pipeline.main(
            [
                "--run-name",
                "migration-test",
                "--dry-run",
                "--restart-partial-search",
                "--core-target-search-decisions",
                "3664",
            ]
        )
        == 0
    )
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["iteration_contract"]["target_search_decisions"] == 3664
    migration = metadata["contract_migration_history"][-1]
    assert migration["previous_contract"] == old_contract
    assert migration["finalized_search_reinterpreted"] is False
    assert len(migration["quarantined_partial_artifacts"]) == 3
    assert not (replay_dir / "core_deep_search.jsonl.partial").exists()

    assert (
        train_core_pipeline.main(
            [
                "--run-name",
                "migration-test",
                "--dry-run",
                "--core-target-search-decisions",
                "3664",
                "--workers",
                "24",
                "--worker-ceiling",
                "24",
                "--inference-servers",
                "3",
            ]
        )
        == 0
    )
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["active_command_config"]["workers"] == 24
    assert metadata["active_command_config"]["inference_servers"] == 3


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    vector = features.SparseVector()
    for word in range(words):
        vector.word_start()
        vector.add((offset + word) % 32, 1.0)
    return vector


def _sequence() -> GameSequence:
    decision = DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS),
        options=_sparse(2, 3),
        action=[1],
        action_combo_index=1,
        action_combos=[[0], [1]],
        env_step=0,
        action_token=_sparse(1, 7),
        policy_stages=[
            PolicyStage(
                options=_sparse(2, 3),
                action_combos=[[0], [1]],
                target_index=1,
            )
        ],
    )
    return GameSequence(
        episode_id="core",
        seat=0,
        archetype="unknown",
        opp_archetype="not-a-registered-archetype",
        deck=[1] * 60,
        value=1.0,
        decisions=[decision],
        factorized_policy_targets=[
            [
                {
                    "action_combos": [[0], [1]],
                    "policy": [0.25, 0.75],
                    "selected_index": 1,
                }
            ]
        ],
    )


def test_core_pipeline_worker_sizing_reserves_host_headroom() -> None:
    assert auto_size_core_workers(
        cpu_count=32, load_1m=9.1, reserve_threads=8, ceiling=10
    ) == 10
    assert auto_size_core_workers(
        cpu_count=16, load_1m=12.0, reserve_threads=8, ceiling=10
    ) == 2


def test_core_bc_jobs_cover_multiple_pairings() -> None:
    jobs = build_core_bc_jobs([_spec(i) for i in range(6)], games=24, seed=7)
    assert len(jobs) == 24
    assert all(job["spec0"]["id"] != job["spec1"]["id"] for job in jobs)
    assert len({job["spec0"]["id"] for job in jobs}) == 6
    assert len(
        {(job["spec0"]["id"], job["spec1"]["id"]) for job in jobs}
    ) > 6


def test_collection_accounting_rejects_incomplete_games() -> None:
    accounting = CollectionAccounting(scheduled_games=2)
    accounting.add(
        {
            "ok": True,
            "records": [{"steps": [1]}, {"steps": [2]}],
            "decisions": 3,
            "deck_hashes": ["a", "b"],
            "agents": ["x", "y"],
        }
    )
    accounting.add({"ok": False, "error": "timeout"})
    assert accounting.completed_games == 2
    assert accounting.accepted_games == 1
    assert accounting.dropped_games == 1
    assert accounting.records == 2
    assert accounting.decisions == 3


def test_phase_resume_advances_only_at_boundary(tmp_path) -> None:
    state = load_pipeline_state(tmp_path, "test-run")
    assert state["current_phase"] == "core_bc"
    advanced = advance_pipeline_state(
        tmp_path, state, "core_bc", {"checkpoint": {"digest": "sha256:x"}}
    )
    assert advanced["current_phase"] == "core_deep_search"
    resumed = load_pipeline_state(tmp_path, "test-run")
    assert resumed == advanced
    with pytest.raises(ValueError):
        advance_pipeline_state(tmp_path, resumed, "core_bc", {})


def test_trusted_deep_search_guard_accepts_shared_belief_engine() -> None:
    require_trusted_search("core_deep_search")


def test_search_target_identity_rejects_stale_and_incomplete_targets() -> None:
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
        },
        "belief_config": {
            "sampler": "public-particles-v1",
            "mode": "particles",
            "model_digest": "sha256:belief",
            "conserves_card_multiplicity": True,
            "uses_baseline_identity": False,
        },
        "simulator_version": "competition-libcg-sha256:test",
    }
    diagnostics = [
        {
            "sims_run": 128,
            "sims_planned": 128,
            "unique_expanded_nodes": 70,
            "max_depth": 9,
            "mean_depth": 3.2,
            "mean_branching": 4.0,
            "leaf_evaluations": 128,
            "chance_samples": 4,
            "unique_particles": 16,
            "root_visits": 128,
            "queue_wait_ms_mean": 1.0,
            "inference_batch_size_mean": 32.0,
            "sims_per_s": 20.0,
            "elapsed_s": 6.4,
            "trusted": True,
        }
    ]
    validate_search_target_identity(
        provenance,
        diagnostics,
        expected_checkpoint_digest="sha256:current",
        expected_model_generation=3,
    )
    with pytest.raises(ValueError, match="stale"):
        validate_search_target_identity(
            provenance,
            diagnostics,
            expected_checkpoint_digest="sha256:other",
            expected_model_generation=3,
        )
    with pytest.raises(ValueError, match="missing"):
        validate_search_target_identity(
            provenance,
            [
                {
                    key: value
                    for key, value in diagnostics[0].items()
                    if key != "leaf_evaluations"
                }
            ],
            expected_checkpoint_digest="sha256:current",
            expected_model_generation=3,
        )


def test_gpu0_environment_contract(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert validate_gpu0_isolation()["cuda_visible_devices"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=0"):
        validate_gpu0_isolation()


def test_stable_core_console_path_is_not_timestamped() -> None:
    assert DEFAULT_CORE_LOG.name == "core_kernel.log"
    assert DEFAULT_CORE_LOG.parent.name == "logs"


def test_small_core_loss_uses_factorized_history_contract() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        decision_context="history",
        kv_cache=True,
        dropout=0.0,
    )
    kernel = CoreKernel(cfg=cfg, device=torch.device("cpu"))
    loss, metrics = core_batch_losses(
        kernel, [_sequence()], condition=False, aux_weight=0.0
    )
    assert torch.isfinite(loss)
    assert metrics.n_games == 1
    assert metrics.n_decisions == 1
    assert metrics.policy_kl >= 0.0


def test_small_core_to_hammer_transfer_is_shape_complete(tmp_path) -> None:
    cfg = core_kernel_config_small_3080ti()
    kernel = CoreKernel(cfg=cfg, device=torch.device("cpu"))
    checkpoint_path = tmp_path / "small-core.pt"
    kernel.save_core_kernel(checkpoint_path)
    report = core_to_specialist_transfer_report(checkpoint_path)
    assert report["complete_shape_transfer"] is True
    assert report["loaded_count"] == report["target_tensor_count"]
    assert report["skipped_tensors"] == {}


def test_historical_core_kernel_ignores_ambient_future_head_flags(tmp_path) -> None:
    cfg = core_kernel_config_small_3080ti()
    kernel = CoreKernel(cfg=cfg, device=torch.device("cpu"))
    checkpoint_path = tmp_path / "historical-core-kernel.pt"
    kernel.save_core_kernel(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for field in (
        "setup_board_outcome_head_enabled",
        "decision_fusion_dedicated_routes_enabled",
        "decision_fusion_dedicated_routes_runtime_enabled",
    ):
        payload["model_config"].pop(field, None)
    torch.save(payload, checkpoint_path)

    source = """
import json
from poke_bot.core_kernel import CoreKernel
k = CoreKernel.load_core_kernel(PATH)
print(json.dumps({
    "setup": k.cfg.setup_board_outcome_head_enabled,
    "routes": k.cfg.decision_fusion_dedicated_routes_enabled,
    "runtime": k.cfg.decision_fusion_dedicated_routes_runtime_enabled,
}))
""".replace("PATH", repr(str(checkpoint_path)))
    environment = dict(os.environ)
    environment.update(
        {
            "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED": "1",
            "POKEBOT_DECISION_FUSION_DEDICATED_ROUTES_ENABLED": "1",
            "POKEBOT_DECISION_FUSION_DEDICATED_ROUTES_RUNTIME_ENABLED": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == {"setup": False, "routes": False, "runtime": False}
