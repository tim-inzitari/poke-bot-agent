from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_r229_fleet_bo1000.py"
spec = importlib.util.spec_from_file_location("r229_runner", SCRIPT)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _package_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "package-manifest.json"
    path.write_text(json.dumps({
        "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r239_package/v1",
        "status": "sealed_evaluation_only",
        "owner_goal_revision": 239,
        "bo_lifecycle_revision": 233,
        "canonical_libcg_revision": 236,
        "owner_two_lane_topology_revision": 239,
        "simulator_lane_count": 2,
        "internal_agent_start_arena_count": 2,
        "distinct_search_begin_id_count": 2,
        "search_begin_identity_scope": "arena_handle_plus_handle_local_search_id",
        "raw_search_id_global_uniqueness_required": False,
        "logical_frontier_leaf_count_per_frozen_model_batch": 2,
        "partial_frontier_batches_allowed": False,
        "serial_one_lane_continuation_allowed": False,
        "one_shared_logical_mcts_tree_required": True,
        "checkpoint_sha256": runner.CHECKPOINT,
        "complete_ordered_action_ceiling": 65536,
        "canonical_libcg_wheel": {"sha256": runner.CANONICAL_LIBCG_WHEEL},
        "canonical_native_libraries": {
            name: {"path": path_name, "sha256": digest, "size_bytes": size}
            for name, (path_name, digest, size) in runner.CANONICAL_NATIVE_LIBRARIES.items()
        },
        "r234_kaggle_broker_or_queue_lifecycle_included": False,
        "package_payload_tree_sha256": "sha256:" + "a" * 64,
        "training_eligible": False,
    }))
    return path


def test_schedule_is_exactly_500_seat_swapped_pairs():
    rows = runner.schedule()
    assert len(rows) == 1000
    assert len({row["game_id"] for row in rows}) == 1000
    for pair in range(500):
        pair_rows = [row for row in rows if row["pair_index"] == pair]
        assert {row["mcts_seat"] for row in pair_rows} == {0, 1}
        assert {row["game_index"] for row in pair_rows} == {0, 1}


def test_revision_230_fully_enumerates_observed_6720_action_prompt():
    from poke_bot import features

    observation = SimpleNamespace(
        select=SimpleNamespace(option=[object()] * 8, minCount=5, maxCount=5)
    )
    actions = features.enumerate_action_combos(observation)
    assert features.MAX_ACTION_COMBOS == 65536
    assert actions.total_count == len(actions) == 6720


def test_run_parses_remote_stdout_and_commits_exact_receipt(monkeypatch, tmp_path):
    job = runner.schedule()[0]
    receipt = {
        "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r229_game/v1",
        "status": "complete", **job, "canonical_libcg_revision": 236,
        "mcts_topology_revision": 239, "simulator_lane_count": 2,
        "training_eligible": False,
    }
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="ADMITTED\n"),
    )

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout):
            return "noise\n" + json.dumps(receipt) + "\n", None

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: FakeProcess())
    output = tmp_path / "game.json"
    result = runner._run(
        {"id": "elmo", "admission_command": ["probe"], "command": ["worker", "{game_id}"], "slots": 1},
        job, output, tmp_path / "game.log",
    )
    assert result["disposition"] == "complete"
    assert json.loads(output.read_text())["game_id"] == job["game_id"]


def test_host_admission_is_exact_and_fail_closed(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="ADMITTED but busy\n"))
    admitted, reason = runner._host_admitted({"admission_command": ["probe"]})
    assert admitted is False
    assert "admission_refused" in reason


def test_one_worker_failure_is_receipted_requeued_and_does_not_abort(monkeypatch, tmp_path):
    job = runner.schedule()[0]
    monkeypatch.setattr(runner, "schedule", lambda: [job])
    monkeypatch.setattr(runner, "summarize_games", lambda rows, require_complete: {"throughput": {}})
    calls = {"count": 0}

    def fake_run(host, current, output, log):
        calls["count"] += 1
        if calls["count"] == 1:
            raise runner.R229FleetError("stuck worker")
        runner._atomic(output, {
            "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r229_game/v1",
            "status": "complete", **current, "canonical_libcg_revision": 236,
            "mcts_topology_revision": 239, "simulator_lane_count": 2,
            "training_eligible": False,
        })
        return {"disposition": "complete", "host": host["id"], "wall_seconds": 1.0}

    monkeypatch.setattr(runner, "_run", fake_run)
    config = tmp_path / "fleet.json"
    config.write_text(json.dumps({
        "package_manifest_path": str(_package_manifest(tmp_path)),
        "hosts": [
        {"id": "elmo-slot", "role": "elmo", "slots": 1},
        {"id": "bert-slot", "role": "bert", "slots": 0},
        {"id": "train-slot", "role": "train_inzi", "slots": 0},
    ]}))
    result = runner.run(SimpleNamespace(
        config=config, output_root=tmp_path / "out",
        admission_retry_seconds=0.0, quarantine_after_failures=3,
    ))
    assert result["status"] == "complete"
    attempts = sorted((tmp_path / "out" / "attempts").glob("*.json"))
    assert len(attempts) == 2
    first = json.loads(attempts[0].read_text())
    second = json.loads(attempts[1].read_text())
    assert first["disposition"] == "failed_attempt_requeued"
    assert second["disposition"] == "complete"
    assert first["attempt"] == 1 and second["attempt"] == 2
    assert first["attempt_wall_seconds"] >= 0
    assert second["log_sha256"] is None


def test_process_watchdog_bounds_only_its_spawned_child():
    watchdog = Path(__file__).parents[1] / "scripts/run_r229_process_watchdog.py"
    result = subprocess.run(
        [
            sys.executable, str(watchdog), "--timeout-seconds", "0.05",
            "--grace-seconds", "0.05", "--", sys.executable, "-c",
            "import time; time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=2,
        check=False,
    )
    assert result.returncode == 124
    assert "R229_GAME_WATCHDOG_TIMEOUT" in result.stdout


def test_failed_remote_child_runs_only_configured_exact_cleanup(monkeypatch, tmp_path):
    job = runner.schedule()[0]
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        stdout = "ADMITTED\n" if argv == ["probe"] else "removed exact child\n"
        return SimpleNamespace(returncode=0, stdout=stdout)

    class FailedProcess:
        returncode = 124

        def communicate(self, timeout):
            return "R229_GAME_WATCHDOG_TIMEOUT\n", None

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: FailedProcess())
    with pytest.raises(runner.R229FleetError, match="cleanup_exit:0"):
        runner._run(
            {
                "id": "elmo",
                "admission_command": ["probe"],
                "command": ["worker", "{game_id}"],
                "failed_child_cleanup_command": ["cleanup", "pokebot-{game_id}"],
            },
            job,
            tmp_path / "game.json",
            tmp_path / "game.log",
        )
    assert calls[-1] == ["cleanup", f"pokebot-{job['game_id']}"]


def test_resume_attempt_number_never_reuses_orphaned_log(tmp_path):
    game_id = "r229-pair-0000-game-0"
    logs = tmp_path / "logs"
    attempts = tmp_path / "attempts"
    logs.mkdir()
    attempts.mkdir()
    (attempts / f"{game_id}.attempt-001.json").write_text("{}\n")
    (logs / f"{game_id}.attempt-004.log").write_text("orphaned\n")
    assert runner._attempt_number(tmp_path, game_id) == 4


def test_package_identity_is_required_and_checksum_bound(tmp_path):
    with pytest.raises(runner.R229FleetError, match="package_manifest_path"):
        runner._package_identity({})
    manifest = _package_manifest(tmp_path)
    identity = runner._package_identity({"package_manifest_path": str(manifest)})
    assert identity["package_manifest_sha256"].startswith("sha256:")
    assert identity["package_payload_tree_sha256"] == "sha256:" + "a" * 64
    assert identity["canonical_libcg_revision"] == 236
    assert identity["mcts_topology_revision"] == 239
    assert identity["simulator_lane_count"] == 2


def test_package_identity_rejects_mixed_libcg_manifest(tmp_path):
    manifest = _package_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["canonical_native_libraries"]["linux_x86_64"]["sha256"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(runner.R229FleetError, match="mixed or incomplete"):
        runner._package_identity({"package_manifest_path": str(manifest)})


def test_fleet_config_uses_sealed_admission_script_for_every_host():
    config = json.loads(
        (Path(__file__).parents[1] / "config/r229_fleet_bo1000.json").read_text()
    )
    expected = (
        "/home/inzi/poke-bot-agent/outputs/evaluations/r229-sealed/"
        "source/scripts/r229_host_admission.py"
    )
    assert all(expected in row["admission_command"] for row in config["hosts"])
    endpoint_rows = [
        row for row in config["hosts"] if "--endpoint" in row["admission_command"]
    ]
    assert all(
        "PYTHONPATH=/home/inzi/poke-bot-agent" in row["admission_command"]
        for row in endpoint_rows
    )
