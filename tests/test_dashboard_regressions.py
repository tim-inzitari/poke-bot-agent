"""Regression coverage for the live LAN training dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
import plistlib
import re
import sys
import types

import pytest

from scripts import dashboard_snapshot as dashboard_snapshot_module
from dashboard.lan.server import (
    DashboardHTTPServer,
    SnapshotCache,
    dashboard_ui_version,
    rendered_index,
)
from scripts.dashboard_snapshot import (
    canonical_next_prestage_overlay,
    COMPETITION_GATE_PROGRAM,
    _active_curriculum_services,
    _metric_iteration_wall_seconds,
    _offset_public_mix_iterations,
    _run_name_from_command,
    _specialist_id_from_runtime,
    _select_curriculum_run_dir,
    active_gate_contract_for_run,
    active_specialist_commit_overlay,
    annotate_collection_budget,
    annotate_expert_optimizer_sps,
    authoritative_training_state,
    checkpoint_parameter_telemetry,
    committed_official_heldout_state,
    competition_gate_program_state,
    effective_design_contract_for_run,
    expert_rehearsal_state,
    learner_model_state,
    latest_committed_active_gate_result,
    latest_committed_formal_holdout_state,
    latest_committed_official_heldout_state,
    latest_committed_research_control_result,
    iteration_timing_state,
    matchup_runtime_collection_state,
    parse_curriculum_progress,
    prestage_receipt_is_current,
    reconcile_frozen_specialist_label,
    reconcile_canonical_router_candidate,
    reconcile_completed_train_epoch,
    reconcile_frozen_specialist_rows,
    research_control_registry_state,
    replay_window_state,
    scheduler_queue_state,
    service_state,
    specialist_protocol_state,
    strong_public_practice_plan_state,
    strong_public_gate_runtime_state,
)


def test_canonical_v6_prestage_blocks_legacy_v5_ready_projection() -> None:
    payload = {
        "expert_corpus_archive": {
            "canonical_policy": {
                "next_specialist_prestage": {
                    "status": "blocked_waiting_for_expanded_v6_corpus",
                    "blocker": "protocol_valid_expert_corpus_not_ready",
                    "intended_next_specialist_after_corpus_validation": (
                        "dudunsparce"
                    ),
                    "receipt": "/state/next-specialist-prestage-v1.json",
                    "cpu_pack_status": "not_built",
                }
            }
        }
    }

    state = canonical_next_prestage_overlay(payload)

    assert state == {
        "status": "blocked_waiting_for_expanded_v6_corpus",
        "blocker": "protocol_valid_expert_corpus_not_ready",
        "intended_specialist": "dudunsparce",
        "blocks_v6_handoff": True,
        "receipt": "/state/next-specialist-prestage-v1.json",
        "cpu_pack_status": "not_built",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_blocked_prestage_receipt_is_current_without_stale_selection() -> None:
    blocked = {
        "schema": "poke_bot.next_specialist_prestage/v1",
        "status": "blocked",
        "active_specialist": "dragapult-dusknoir",
        "selected_specialist": None,
        "selection": None,
        "blockers": ["protocol_valid_expert_corpus_not_ready"],
    }
    assert prestage_receipt_is_current(blocked, "dragapult-dusknoir")
    assert not prestage_receipt_is_current(blocked, "starmie")


def test_ready_prestage_receipt_requires_selected_specialist() -> None:
    ready = {
        "schema": "poke_bot.next_specialist_prestage/v1",
        "status": "ready",
        "active_specialist": "dragapult-dusknoir",
        "selected_specialist": None,
    }
    assert not prestage_receipt_is_current(ready, "dragapult-dusknoir")
    ready["selected_specialist"] = "dudunsparce"
    assert prestage_receipt_is_current(ready, "dragapult-dusknoir")


def test_latest20_dashboard_names_atomic_sync_state() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/dashboard_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "Inzi checksum sync active" in source
    assert "atomic pointer withheld until " in source
    assert '"validation"' in source


def test_curriculum_worker_reads_effective_environment_file_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_file = tmp_path / "specialist_runtime.env"
    environment_file.write_text(
        "\n".join(
            (
                "PURE_RL_SIM_WORKERS=128",
                "PURE_RL_LEAF_GPU0_REPLICAS=10",
                "PURE_RL_LEAF_GPU1_REPLICAS=24",
                "POKEBOT_MULTI_ENV_PER_WORKER=4",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "_unit_values",
        lambda *_args, **_kwargs: {
            "MainPID": "123",
            "ControlGroup": "/user.slice/test.service",
            "MemoryCurrent": "1024",
            "TasksCurrent": "8",
            "Environment": "",
            "EnvironmentFiles": f"{environment_file} (ignore_errors=no)",
        },
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "_cgroup_pids",
        lambda _group: {123},
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "_process_environment",
        lambda _pid: {},
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "process_rows",
        lambda: {123: (1, 12.0, 2048, "trainer --run current")},
    )

    worker = dashboard_snapshot_module.curriculum_worker_state(
        ["production.service"],
        [123],
    )

    assert worker["workers"] == 128
    assert worker["multi_env_per_worker"] == 4
    assert worker["leaf_gpu0_replicas"] == 10
    assert worker["leaf_gpu1_replicas"] == 24
    assert worker["leaf_servers"] == 34
    assert (
        worker["topology_source"]
        == "active managed trainer effective environment"
    )


def test_gpu0_assignment_uses_active_managed_trainer_leaf_topology() -> None:
    gpus = [
        {"index": 0, "name": "NVIDIA GeForce RTX 3080 Ti"},
        {"index": 1, "name": "NVIDIA RTX PRO 5000 Blackwell"},
    ]
    curriculum = {
        "active": True,
        "worker": {
            "leaf_gpu0_replicas": 10,
            "leaf_gpu1_replicas": 24,
            "topology_source": "active managed trainer effective environment",
        },
    }

    dashboard_snapshot_module.annotate_gpu_production_assignments(
        gpus,
        curriculum,
        {"active": False},
    )

    assert gpus[0]["production_active"] is True
    assert gpus[0]["production_leaf_replicas"] == 10
    assert gpus[0]["assignment"] == "PRODUCTION · 10 policy leaf replicas"
    assert (
        gpus[0]["assignment_source"]
        == "active managed trainer effective environment"
    )
    assert gpus[1]["production_active"] is True
    assert gpus[1]["production_leaf_replicas"] == 24
    assert gpus[1]["assignment"] == "PRODUCTION · policy leaves + trainer"


def test_gpu0_assignment_is_out_of_fleet_only_with_zero_effective_replicas() -> None:
    gpus = [{"index": 0, "name": "NVIDIA GeForce RTX 3080 Ti"}]

    dashboard_snapshot_module.annotate_gpu_production_assignments(
        gpus,
        {
            "active": True,
            "worker": {
                "leaf_gpu0_replicas": 0,
                "topology_source": "active managed trainer effective environment",
            },
        },
        {"active": False},
    )

    assert gpus[0]["production_active"] is False
    assert gpus[0]["production_leaf_replicas"] == 0
    assert (
        gpus[0]["assignment"]
        == "OUT OF FLEET · no active trainer leaf replicas"
    )


def test_frozen_specialist_label_cannot_cross_contaminate_archetypes() -> None:
    row = reconcile_frozen_specialist_label(
        {
            "frozen_specialist": True,
            "archetype_id": "lucario",
            "archetype_label": "Frozen Dragapult Dusknoir specialist",
        }
    )
    assert row["archetype_label"] == "Frozen Mega Lucario ex specialist"
    assert row["source_archetype_label"] == (
        "Frozen Dragapult Dusknoir specialist"
    )
    assert row["archetype_label_reconciled"] is True


def test_canonical_router_candidate_supersedes_old_elmo_refresh() -> None:
    old = {
        "router_refresh": {
            "name": "pokebot-public-tree-latest22-v32",
            "calibrated_route_count": 19,
            "calibrated_route_ids": ["incorrect-compressed-class"],
        }
    }
    protocol = {
        "head_requirements": {
            "staged_router_candidate": {
                "version": 33,
                "status": "validated_inactive",
                "runtime_enabled": False,
                "accepted_route_count": 16,
                "accepted_routes": ["starmie", "lucario"],
                "evidence_blocked_routes": ["walrein"],
                "artifact": "/canonical/v33.json",
                "artifact_checksum": "sha256:" + "a" * 64,
                "calibration_contract": "expanded_to_canonical_class_indexes",
            }
        }
    }

    reconciled = reconcile_canonical_router_candidate(old, protocol)
    refresh = reconciled["router_refresh"]

    assert refresh["canonical"] is True
    assert refresh["name"] == "canonical-public-matchup-tree-v33"
    assert refresh["calibrated_route_count"] == 19
    assert refresh["calibrated_route_ids"] == ["incorrect-compressed-class"]
    assert refresh["protocol_ready_route_count"] == 16
    assert refresh["protocol_ready_route_ids"] == ["starmie", "lucario"]
    assert refresh["protocol_blocked_route_count"] == 1
    assert refresh["evidence_blocked_routes"] == ["walrein"]
    assert refresh["candidate_runtime_enabled"] is False
    assert (
        refresh["superseded_observation"]["name"]
        == "pokebot-public-tree-latest22-v32"
    )


def test_frozen_registry_overlay_supersedes_stale_mutable_roster_status() -> None:
    rows = [
        {
            "id": "starmie",
            "status": "rl_training",
            "active": True,
            "frozen": False,
            "public_mix_eligible": False,
        },
        {
            "id": "lucario",
            "status": "unstarted",
            "active": False,
            "frozen": False,
            "public_mix_eligible": False,
        },
    ]

    reconciled, counts, active = reconcile_frozen_specialist_rows(
        rows,
        {"starmie"},
        "starmie",
    )

    starmie = next(row for row in reconciled if row["id"] == "starmie")
    assert starmie["status"] == "passed_frozen"
    assert starmie["frozen"] is True
    assert starmie["active"] is False
    assert starmie["public_mix_eligible"] is True
    assert starmie["immutable_registry_overlay"] is True
    assert counts == {"passed_frozen": 1, "unstarted": 1}
    assert active == ""


def test_completed_specialist_handoff_does_not_surface_stale_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff = tmp_path / "handoff.json"
    core = tmp_path / "core.json"
    bootstrap = tmp_path / "bootstrap.json"
    log = tmp_path / "handoff.log"
    _write_json(handoff, {"phase": "next_specialist_rl_started"})
    _write_json(core, {"history": [{"epoch": 25}], "epochs_max": 25})
    _write_json(bootstrap, {"history": [{"epoch": 25}], "epochs_max": 25})
    log.write_text("RuntimeError: prior launch failed\n", encoding="utf-8")
    monkeypatch.setattr(
        dashboard_snapshot_module, "OWNER_SPECIALIST_HANDOFF_STATE", handoff
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "OWNER_CORE_DISTILL_STATE", core
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "OWNER_TREVENANT_BOOTSTRAP_STATE", bootstrap
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "OWNER_SPECIALIST_HANDOFF_LOG", log
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda *_args, **_kwargs: {
            "active": False,
            "pid": 0,
            "load_state": "loaded",
        },
    )

    result = dashboard_snapshot_module.owner_specialist_handoff_state()

    assert result["phase"] == "next_specialist_rl_started"
    assert result["stage"] == "next_specialist_rl_started"
    assert result["latest_line"] == "Hop's Trevenant specialist RL is active."


def test_iteration_timing_subtracts_expert_rehearsal_from_history(
    tmp_path: Path,
) -> None:
    payload = {
        "iteration": 5,
        "games": 100,
        "decisions": 1_000,
        "extra": {
            "iteration_wall_sec": 1_000.0,
            "collect_stats": {"collect_elapsed_sec": 50.0},
            "expert_rehearsal": {"before_iteration": 5},
        },
    }
    _write_json(
        tmp_path / "metrics/iter_00004.json",
        {
            "iteration": 4,
            "games": 300,
            "decisions": 6_000,
            "extra": {
                "iteration_wall_sec": 500.0,
                "collect_stats": {"collect_elapsed_sec": 100.0},
            },
        },
    )
    _write_json(tmp_path / "metrics/iter_00005.json", payload)
    _write_json(
        tmp_path / "collection_receipts/iter_00005.json",
        {"completed_at": 2_000.0},
    )
    rehearsal = tmp_path / "rehearsals/before_iter_00005.json"
    _write_json(rehearsal, {"before_iteration": 5})
    os.utime(rehearsal, (2_300.0, 2_300.0))

    assert _metric_iteration_wall_seconds(payload, run_dir=tmp_path) == 700.0
    state = iteration_timing_state(
        tmp_path,
        active=False,
        global_iteration_offset=0,
    )
    assert state["latest_seconds"] == 700.0
    assert state["rolling5_seconds"] == 600.0
    assert state["history"][-1]["expert_rehearsal_excluded_seconds"] == 300.0
    assert state["latest_gps"] == 2.0
    assert state["latest_sps"] == 20.0
    assert state["rolling5_gps"] == 400.0 / 150.0
    assert state["rolling5_sps"] == 7_000.0 / 150.0
    assert state["rolling5_throughput_samples"] == 2


def test_live_iteration_clock_freezes_during_expert_rehearsal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("scripts.dashboard_snapshot.time.time", lambda: 10_000.0)
    _write_json(
        tmp_path / "iteration_runtime.json",
        {
            "iteration": 5,
            "phase": "collect",
            "started_at": 9_000.0,
        },
    )
    _write_json(
        tmp_path / "collection_receipts/iter_00005.json",
        {"stats": {"collect_elapsed_sec": 123.5}},
    )

    state = iteration_timing_state(
        tmp_path,
        active=True,
        global_iteration_offset=0,
        next_iteration=5,
        progress_iteration=5,
        progress_stage="train:expert",
    )
    assert state["current_seconds"] == 123.5
    assert state["current_paused_for_expert_rehearsal"] is True
    assert "expert rehearsal excluded" in state["current_source"]


def test_dashboard_labels_iteration_time_as_rehearsal_excluded() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "RL iteration time" in html
    assert "expert rehearsal excluded" in html
    assert "current_paused_for_expert_rehearsal" in html
    assert 'id="rl-gps-note"' in html
    assert 'id="rl-sps-note"' in html
    assert "weighted avg" in html


def test_matchup_adapter_fit_owns_hero_progress_while_production_is_gated() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "matchupFitActive?num(matchupFit.percent,1)+'%'" in html
    assert "Exact all-22 matchup-adapter bootstrap · base model frozen" in html
    assert "Exact all-22 matchup-adapter validation · base model frozen" in html
    assert "training pass complete · heldout pass active" in html
    assert "matchupFit.phase==='validation'" in html
    assert "matchupFitActive?duration(matchupFit.eta_seconds)" in html
    assert "matchupFitActive?matchupFit.latest_line" in html
    snapshot = (
        Path(__file__).resolve().parents[1] / "scripts/dashboard_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "pokebot-matchup-adapter-v31-recovery.service" in snapshot
    assert '"auxiliary_fleet": bool(fit_service.get("active"))' in snapshot
    assert 'if fit_service.get("active"):' in snapshot
    assert '"fleet": False' in snapshot
    assert "fleet_eta_seconds" in html
    assert "active_route_games" in html
    assert "gate_handler_minimum_completed_iteration" in snapshot


def test_dashboard_uses_only_committed_matchup_runtime_receipts(
    tmp_path: Path,
) -> None:
    assert matchup_runtime_collection_state(tmp_path)["available"] is False
    audit = {
        "schema": "poke_bot.matchup_runtime_collection_audit/v1",
        "games": 8192,
        "audited_games": 8192,
        "all_runtime_enabled": True,
        "contract_clean": True,
    }
    receipt = tmp_path / "collection_receipts/iter_00027.json"
    _write_json(
        receipt,
        {
            "schema": "poke_bot.completed_collection/v1",
            "iteration": 27,
            "checkpoint_digest": "sha256:" + "a" * 64,
            "stats": {
                "matchup_runtime": audit,
                "matchup_runtime_self_play": {**audit, "games": 1024},
                "matchup_runtime_public_mix": {**audit, "games": 7168},
                "matchup_runtime_enforcement": {
                    "schema": (
                        "poke_bot.matchup_runtime_collection_enforcement/v1"
                    ),
                    "required": True,
                    "passed": True,
                    "assertions": {},
                },
            },
        },
    )
    state = matchup_runtime_collection_state(tmp_path)
    assert state["available"] is True
    assert state["iteration"] == 27
    assert state["combined"]["audited_games"] == 8192

    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "CAUSAL ROUTING VERIFIED" in html
    assert "every audited game started at exact bypass" in html
    assert "matchupRuntime=c.matchup_runtime||{}" in html


def test_truncated_live_tqdm_remains_current_and_splits_queue_grain() -> None:
    initial = (
        "pure_rl collect:self_play iter=0:   0%| | 0/6963 "
        "[00:07<?, ?game/s, rdmd=64, remotes=128, sps=0]\r"
    )
    current = (
        "pure_rl collect:self_play iter=0:  52%|████| 3604/6963 "
        "[05:43<05:29, 10.18game/s, rdmd=64, remotes=128, sps="
    )
    parsed = parse_curriculum_progress("", initial + current, iteration_hint=0)

    assert parsed["current"] == 3604
    assert parsed["total"] == 6963
    assert parsed["percent"] == 52.0
    assert parsed["gps"] == 10.18
    assert parsed["remotes"] == 64
    assert parsed["metrics"]["remote_request_sockets"] == 128
    assert parsed["metrics"]["remote_queue_capacity"] == 64


def test_self_play_reserve_progress_is_labeled_as_attempts_not_training_games() -> None:
    parsed = parse_curriculum_progress(
        "",
        "pure_rl collect:self_play iter=27: 50%|██| 768/1536 "
        "[01:00<01:00, 12.8game/s]",
        iteration_hint=27,
    )
    annotated = annotate_collection_budget(
        parsed,
        "[pure_rl] collect iter=27 bounded self-play refill capacity=512 "
        "primary_self_play=1024 target_games=8192",
    )

    assert annotated["total"] == 1536
    assert annotated["metrics"]["simulation_attempts"] is True
    assert annotated["metrics"]["primary_retained_target"] == 1024
    assert annotated["metrics"]["reserve_attempt_capacity"] == 512
    assert annotated["metrics"]["iteration_retained_target"] == 8192
    assert annotated["metrics"]["unused_reserve_training_eligible"] is False

    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "simulation attempts · exact retained target" in html
    assert "isolated reserve attempts" in html


def test_shutdown_warning_cannot_corrupt_last_progress_frame() -> None:
    line = (
        "pure_rl collect:self_play iter=0:  54%|████| 3744/6963 "
        "[05:51<03:59, 13.46game/s, rdmd=64, remotes=128, sps="
        "/opt/python/multiprocessing/resource_tracker.py:254: UserWarning: "
        "resource_tracker leaked semaphores"
    )
    parsed = parse_curriculum_progress("", line, iteration_hint=0)
    assert parsed["current"] == 3744
    assert "resource_tracker" not in parsed["line"]


def test_curriculum_dashboard_parses_every_active_head() -> None:
    line = (
        "rl-train ep0:  50%|████| 5/10 [00:01<00:01, 4.0batch/s, "
        "acc=42.0%, aux=1.2/5, guide=0.7/9, hand=0.6/21, "
        "lethal=0.4/30, loss=2.0, p=1.0, prize=0.3/30, "
        "rem=0.5/21, sps=12345, v=0.2]"
    )
    parsed = parse_curriculum_progress("", line, iteration_hint=3)
    assert parsed["stage"] == "train:policy"
    assert parsed["sps"] == 12345.0
    assert parsed["metrics"] == {
        "acc": 42.0,
        "loss": 2.0,
        "p": 1.0,
        "v": 0.2,
        "hand": 0.6,
        "rem": 0.5,
        "aux": 1.2,
        "lethal": 0.4,
        "prize": 0.3,
        "guide": 0.7,
    }


def test_replay_cache_tqdm_is_live_training_preparation() -> None:
    line = (
        "replay-cache load iter_00000.jsonl:  41%|████| 153/374 "
        "[02:11<03:09, 1.17part/s]"
    )
    parsed = parse_curriculum_progress("", line, iteration_hint=0)
    assert parsed["stage"] == "train:preparing"
    assert parsed["current"] == 153
    assert parsed["total"] == 374
    assert parsed["unit"] == "parts"
    assert parsed["metrics"]["replay_shard"] == "iter_00000.jsonl"


def test_resident_rl_corpus_pack_is_live_training_progress() -> None:
    line = (
        "pack Blackwell corpus:  63%|██████▎| 52250/82944 "
        "[00:49<00:28, 1082.4game/s]"
    )
    parsed = parse_curriculum_progress("", line, iteration_hint=4)
    assert parsed["stage"] == "train:packing"
    assert parsed["iteration"] == 4
    assert parsed["current"] == 52250
    assert parsed["total"] == 82944
    assert parsed["unit"] == "games"
    assert parsed["rate"] == 1082.4
    assert parsed["rate_unit"] == "game/s"


def test_recovery_replay_panel_uses_inherited_shard_and_its_cache(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source-run"
    source_shard = source_run / "shards" / "iter_00000.jsonl"
    source_shard.parent.mkdir(parents=True)
    source_shard.write_bytes(b"one\ntwo\n")
    (source_run / "replay_window.cache.status.json").write_text(
        json.dumps(
            {
                "source_shard": str(source_shard),
                "updated_at": time.time(),
                "stage": "cache_load",
                "parts_complete": 153,
                "parts_total": 374,
                "percent": 40.9,
                "sequences_loaded": 34_000,
            }
        ),
        encoding="utf-8",
    )
    recovery_run = tmp_path / "recovery-run"
    recovery_run.mkdir()
    manifest = {
        "design_contract": {
            "games": {"per_iteration": 0},
            "collection": {"replay_window_shards": 2},
            "learner": {
                "initial_replay_shards": [{"path": str(source_shard)}]
            },
        }
    }
    state = replay_window_state(
        recovery_run,
        {"next_iteration": 0},
        manifest,
        {"iteration": 0, "stage": "train:preparing"},
        "",
    )
    assert state["target_shards"] == 1
    assert state["ready_shards"] == 1
    assert state["bytes_total"] == source_shard.stat().st_size
    assert state["shards"][0]["inherited"] is True
    assert state["cache"]["parts_complete"] == 153


def test_active_service_run_name_wins_over_newer_historical_json(tmp_path: Path) -> None:
    old = tmp_path / "run-a"
    active = tmp_path / "run-b"
    old.mkdir()
    active.mkdir()
    (old / "manifest.json").write_text("{}")
    command = "/python scripts/launch_pure_rl.py --run-name run-b --resume auto"

    assert _run_name_from_command(command) == "run-b"
    assert _select_curriculum_run_dir(tmp_path, {old}, "run-b") == active


def test_live_trainer_pid_wins_over_remain_after_exit_bootstrap(monkeypatch) -> None:
    def fake_run(argv, timeout=3.0):
        del timeout
        command = " ".join(argv)
        if "list-units" in command:
            return (
                "pokebot-pure-rl-alakazam-bootstrap.service loaded active exited\n"
                "pokebot-pure-rl-alakazam.service loaded active running"
            )
        if "--property=MainPID" in command:
            return "222" if argv[-3] == "pokebot-pure-rl-alakazam.service" else "0"
        if "--property=ExecStart" in command:
            return "python launch.py --run-name completed-bootstrap"
        return ""

    real_read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        if str(path) == "/proc/222/cmdline":
            return b"python\0launch.py\0--run-name\0live-specialist\0"
        return real_read_bytes(path)

    monkeypatch.setattr("scripts.dashboard_snapshot.run", fake_run)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    units, pids, run_name = _active_curriculum_services()
    assert units == ["pokebot-pure-rl-alakazam.service"]
    assert pids == [222]
    assert run_name == "live-specialist"


def test_remain_after_exit_bootstrap_is_not_a_live_curriculum(monkeypatch) -> None:
    def fake_run(argv, timeout=3.0):
        del timeout
        command = " ".join(argv)
        if "list-units" in command:
            return (
                "pokebot-pure-rl-alakazam-bootstrap.service "
                "loaded active exited"
            )
        if "--property=MainPID" in command:
            return "0"
        if "--property=ExecStart" in command:
            return "python launch.py --run-name completed-bootstrap"
        return ""

    monkeypatch.setattr("scripts.dashboard_snapshot.run", fake_run)

    assert _active_curriculum_services() == ([], [], None)


def test_remain_after_exit_bootstrap_is_not_the_live_training_service(
    monkeypatch,
) -> None:
    rows = {
        "pokebot-pure-rl-alakazam.service": {
            "name": "pokebot-pure-rl-alakazam.service",
            "active": False,
            "active_state": "failed",
            "sub_state": "failed",
            "pid": 0,
            "memory_bytes": None,
            "memory_peak_bytes": 123,
            "cpu_ns": 456,
            "started": "now",
        },
        "pokebot-pure-rl-alakazam-bootstrap.service": {
            "name": "pokebot-pure-rl-alakazam-bootstrap.service",
            "active": True,
            "active_state": "active",
            "sub_state": "exited",
            "pid": 0,
            "memory_bytes": None,
            "memory_peak_bytes": 789,
            "cpu_ns": 1011,
            "started": "earlier",
        },
    }

    def fake_unit_state(name, user=False):
        del user
        return rows.get(
            name,
            {
                "name": name,
                "active": False,
                "active_state": "inactive",
                "sub_state": "dead",
                "pid": 0,
                "memory_bytes": None,
                "memory_peak_bytes": None,
                "cpu_ns": None,
                "started": "",
            },
        )

    monkeypatch.setattr(dashboard_snapshot_module, "unit_state", fake_unit_state)
    monkeypatch.setattr(dashboard_snapshot_module, "run", lambda *args, **kwargs: "")

    state = service_state()
    assert state["name"] == "pokebot-pure-rl-alakazam.service"
    assert state["active"] is False
    assert state["active_state"] == "failed"


def test_live_trevenant_service_is_the_top_level_training_service(
    monkeypatch,
) -> None:
    trevenant = "pokebot-pure-rl-trevenant-staged.service"
    rows = {
        trevenant: {
            "name": trevenant,
            "active": True,
            "active_state": "active",
            "sub_state": "running",
            "pid": 4242,
            "memory_bytes": 96_000_000_000,
            "memory_peak_bytes": 100_000_000_000,
            "cpu_ns": 123,
            "started": "now",
        },
        "pokebot-pure-rl-alakazam.service": {
            "name": "pokebot-pure-rl-alakazam.service",
            "active": False,
            "active_state": "inactive",
            "sub_state": "dead",
            "pid": 0,
            "memory_bytes": None,
            "memory_peak_bytes": 100_000_000_000,
            "cpu_ns": 456,
            "started": "earlier",
        },
    }

    def fake_unit_state(name, user=False):
        del user
        return rows.get(
            name,
            {
                "name": name,
                "active": False,
                "active_state": "inactive",
                "sub_state": "dead",
                "pid": 0,
                "memory_bytes": None,
                "memory_peak_bytes": None,
                "cpu_ns": None,
                "started": "",
            },
        )

    monkeypatch.setattr(
        dashboard_snapshot_module,
        "_active_curriculum_services",
        lambda: ([trevenant], [4242], "trevenant-run"),
    )
    monkeypatch.setattr(dashboard_snapshot_module, "unit_state", fake_unit_state)
    monkeypatch.setattr(dashboard_snapshot_module, "run", lambda *args, **kwargs: "")

    state = service_state()
    assert state["name"] == trevenant
    assert state["active"] is True
    assert state["pid"] == 4242


def test_expert_loading_bar_is_labeled_as_periodic_tune_up() -> None:
    line = (
        "pure_rl train:expert iter=5:   0%| | 0/1 "
        "[00:00<?, ?expert pass/s, loading corpus]"
    )

    parsed = parse_curriculum_progress("", line, iteration_hint=5)

    assert parsed["stage"] == "train:expert:loading"
    assert parsed["iteration"] == 5
    assert parsed["unit"] == "expert pass"
    assert parsed["eta"] == "loading corpus"


def test_expert_rehearsal_sps_uses_exact_device_corpus_split() -> None:
    progress = parse_curriculum_progress(
        "",
        (
            "expert rehearsal before iter5 ep5/5:  46%|████▌     | "
            "102/223 [00:35<00:40, 2.96batch/s, acc=54.57%]"
        ),
        iteration_hint=5,
    )
    enriched = annotate_expert_optimizer_sps(
        progress,
        (
            "[device-corpus] CPU pack=3.75 GiB decisions=2579178 "
            "samples=3033497 train=2729429 val=304068\n"
        ),
    )

    assert enriched["sps"] == pytest.approx(2.96 * 2_729_429 / 223)
    assert enriched["optimizer_samples"] == 2_729_429
    assert enriched["sps_source"] == (
        "exact device-corpus split × live tqdm batch rate"
    )


def test_expert_adapter_rehearsal_has_explicit_live_progress(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rehearsals" / "matchup_adapters"
    _write_json(
        root / "before_iter_00040.authorization.json",
        {"schema": "poke_bot.matchup_adapter_rehearsal_authorization/v1"},
    )
    _write_json(
        root / "before_iter_00040.fit" / "progress.json",
        {
            "epoch": 2,
            "epochs": 5,
            "step": 123,
            "train_sequences_consumed": 100,
            "train_sequences": 1_000,
            "complete": False,
        },
    )
    state = expert_rehearsal_state(
        tmp_path,
        {
            "every_iterations": 5,
            "epochs": 5,
            "matchup_adapters": {
                "enabled": True,
                "epochs": 5,
                "learning_rate": 1e-4,
                "games_per_batch": 4,
                "max_decisions_per_batch": 512,
                "optimizer_scope": "matchup_adapter_bank_only",
                "staged_manifest": "/data/causal/manifest.json",
            },
        },
        {"next_iteration": 40},
        {},
        global_iteration_offset=0,
        trainer_active=True,
    )

    adapter = state["matchup_adapter_rehearsal"]
    assert state["active"] is True
    assert state["state"] == "running · adapter-only phase"
    assert state["current"]["stage"] == "train:expert:matchup-adapters"
    assert state["current"]["percent"] == pytest.approx(42.0)
    assert adapter["active"] is True
    assert adapter["state"] == "running"
    assert adapter["optimizer_scope"] == "matchup_adapter_bank_only"
    assert adapter["base_frozen"] is True
    assert adapter["runtime_enabled_during_fit"] is False


def test_expert_panel_uses_verified_design_migration_chain(
    tmp_path: Path,
) -> None:
    initial = {
        "expert_rehearsal": {
            "every_iterations": 5,
            "epochs": 5,
        }
    }
    current = {
        "expert_rehearsal": {
            "every_iterations": 5,
            "epochs": 5,
            "matchup_adapters": {
                "enabled": True,
                "epochs": 5,
                "optimizer_scope": "matchup_adapter_bank_only",
            },
        }
    }
    initial_digest = dashboard_snapshot_module._canonical_design_digest(initial)
    current_digest = dashboard_snapshot_module._canonical_design_digest(current)
    manifest = {
        "design_contract": initial,
        "design_fingerprint": initial_digest,
    }
    _write_json(
        tmp_path / "design_migrations/migration_0001.json",
        {
            "schema": 1,
            "previous_contract": initial,
            "current_contract": current,
            "previous_fingerprint": initial_digest,
            "current_fingerprint": current_digest,
        },
    )

    effective = effective_design_contract_for_run(tmp_path, manifest)

    assert effective == current
    assert effective["expert_rehearsal"]["matchup_adapters"]["enabled"] is True


def test_expert_validation_sps_uses_validation_split_and_seconds_per_batch() -> None:
    progress = {
        "stage": "train:expert:validation",
        "rate": 0.5,
        "rate_unit": "s/batch",
        "total": 25,
        "sps": None,
    }
    enriched = annotate_expert_optimizer_sps(
        progress,
        (
            "[device-corpus] CPU pack=3.75 GiB decisions=2579178 "
            "samples=3033497 train=2729429 val=304068\n"
        ),
    )

    assert enriched["sps"] == pytest.approx(2.0 * 304_068 / 25)
    assert enriched["optimizer_samples"] == 304_068


def test_expert_optimizer_sps_never_reuses_non_expert_or_stale_values() -> None:
    log = (
        "[device-corpus] CPU pack=1 GiB decisions=100 samples=120 "
        "train=108 val=12\n"
    )
    collecting = {"stage": "collect:public_mix", "sps": None, "rate": 2.0, "total": 2}
    measured = {"stage": "train:expert", "sps": 999.0, "rate": 2.0, "total": 2}

    assert annotate_expert_optimizer_sps(collecting, log) is collecting
    assert annotate_expert_optimizer_sps(measured, log) is measured


def test_official_gateline_uses_reconciled_heldout_champion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    (run_dir / "commits" / "iter_00025.json").write_text("{}")
    loop = {
        "heldout_champion": {"path": "/checkpoints/iter_00025.pt", "digest": "sha256:a"},
        "heldout_champion_evidence": {
            "iteration": 25,
            "checkpoint_digest": "sha256:a",
            "games": 1000,
            "win_rate": 0.2925,
            "confidence_lower": 0.2651,
            "confidence_upper": 0.3214,
        },
        "history": [
            {
                "iteration": 25,
                "candidate": {"digest": "sha256:a"},
                "heldout_audit": {
                    "passed": True,
                    "checkpoint_digest": "sha256:a",
                    "valid_games": 1000,
                    "exact_distribution": True,
                    "exact_weights": True,
                    "greedy_required": True,
                    "per_opponent": {
                        "iono": {"games": 250, "seat0": 125, "seat1": 125}
                    },
                },
                "raw_heldout_gate": {
                    "games": 1000,
                    "win_rate": 0.2925,
                    "passed": False,
                    "reason": "per_opponent_floor",
                    "per_opponent": {
                        "iono": {
                            "games": 250,
                            "win_rate": 0.128,
                            "wins": 32,
                            "draws": 0,
                            "losses": 218,
                            "seat0_games": 125,
                            "seat1_games": 125,
                        }
                    },
                },
            }
        ],
    }

    result = committed_official_heldout_state(loop, run_dir)

    assert result["available"] is True
    assert result["wr"] == 0.2925
    assert result["games"] == 1000
    assert result["iteration"] == 25
    assert result["audit_passed"] is True
    assert result["matchups"] == [
        {
            "opponent_id": "iono",
            "games": 250,
            "wr": 0.128,
            "wins": 32.0,
            "draws": 0.0,
            "losses": 218.0,
            "seat0": 125,
            "seat1": 125,
        }
    ]


def test_official_gateline_accepts_committed_premium_gate_v1_shape(
    tmp_path: Path,
) -> None:
    """The premium gate stores its aggregate/rows under the v1 field names."""

    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    (run_dir / "commits" / "iter_00008.json").write_text("{}")
    digest = "sha256:" + "8" * 64
    rates = {
        "archaludon-ex": 0.416,
        "lucifer19-battlecore": 0.480,
        "pilkwang-meta-20260708": 0.324,
        "specialist-alakazam": 0.148,
        "specialist-dragapult-dusknoir": 0.780,
        "specialist-hops-trevenant": 0.192,
        "specialist-lucario": 0.172,
        "specialist-starmie": 0.416,
    }
    weighted_wr = 0.35428571428571426
    audit_rows = {
        opponent_id: {"games": 250, "seat0": 125, "seat1": 125}
        for opponent_id in rates
    }
    loop = {
        "heldout_champion": {
            "path": "/checkpoints/iter_00008.pt",
            "digest": digest,
        },
        "heldout_champion_evidence": {
            "iteration": 8,
            "checkpoint_digest": digest,
            "games": 2000,
            "win_rate": weighted_wr,
            "confidence_lower": 0.33885714285714286,
            "confidence_upper": 0.37057142857142855,
        },
        "history": [
            {
                "iteration": 8,
                "candidate": {"digest": digest},
                "heldout_audit": {
                    "passed": True,
                    "checkpoint_digest": digest,
                    "valid_games": 2000,
                    "exact_distribution": True,
                    "exact_weights": True,
                    "greedy_required": True,
                    "per_opponent": audit_rows,
                },
                "raw_heldout_gate": {
                    "schema": "poke_bot.public_agent_gate_result/v1",
                    "games": 2000,
                    "skill_weighted_wr": weighted_wr,
                    "passed": False,
                    "reason": "s_plus_matchup_floor_allowance",
                    "matchups": [
                        {
                            "opponent_id": opponent_id,
                            "games": 250,
                            "wr": wr,
                            "wins": int(wr * 250),
                            "draws": 0,
                            "losses": 250 - int(wr * 250),
                            "seat0": 125,
                            "seat1": 125,
                        }
                        for opponent_id, wr in rates.items()
                    ],
                },
            }
        ],
    }

    result = committed_official_heldout_state(loop, run_dir)

    assert result["available"] is True
    assert result["iteration"] == 8
    assert result["games"] == 2000
    assert result["wr"] == weighted_wr
    assert result["audit_passed"] is True
    assert result["opponent_count"] == 8
    assert {row["opponent_id"] for row in result["matchups"]} == set(rates)
    assert sum(int(row["games"]) for row in result["matchups"]) == 2000
    assert sum(int(row["seat0"]) for row in result["matchups"]) == 1000
    assert sum(int(row["seat1"]) for row in result["matchups"]) == 1000
    assert next(
        row
        for row in result["matchups"]
        if row["opponent_id"] == "specialist-dragapult-dusknoir"
    )["wr"] == 0.780


def test_official_gateline_refuses_unreconciled_evidence(tmp_path: Path) -> None:
    loop = {
        "heldout_champion": {"path": "/checkpoints/iter_00025.pt", "digest": "sha256:a"},
        "heldout_champion_evidence": {
            "iteration": 25,
            "checkpoint_digest": "sha256:a",
            "games": 1000,
            "win_rate": 0.2925,
        },
        "history": [
            {
                "iteration": 25,
                "candidate": {"digest": "sha256:a"},
                "heldout_audit": {
                    "passed": True,
                    "checkpoint_digest": "sha256:a",
                    "valid_games": 999,
                    "per_opponent": {},
                },
                "raw_heldout_gate": {
                    "games": 1000,
                    "win_rate": 0.2925,
                    "per_opponent": {},
                },
            }
        ],
    }

    result = committed_official_heldout_state(loop, tmp_path)

    assert result["available"] is False
    assert result["reason"] == "heldout evidence failed reconciliation"


def test_latest_exact_holdout_is_visible_when_candidate_is_not_champion(
    tmp_path: Path,
) -> None:
    from scripts.dashboard_snapshot import latest_committed_official_heldout_state

    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    (run_dir / "commits" / "iter_00002.json").write_text("{}")
    audit_rows = {
        opponent_id: {"games": 250, "seat0": 125, "seat1": 125}
        for opponent_id in (
            "iono",
            "dragapult-ex",
            "mega-abomasnow-ex",
            "mega-lucario-ex",
        )
    }
    rates = {
        "iono": 0.376,
        "dragapult-ex": 0.516,
        "mega-abomasnow-ex": 0.58,
        "mega-lucario-ex": 0.656,
    }
    gate_rows = {
        opponent_id: {
            "games": 250,
            "seat0_games": 125,
            "seat1_games": 125,
            "win_rate": wr,
            "wins": int(wr * 250),
            "draws": 0,
            "losses": 250 - int(wr * 250),
        }
        for opponent_id, wr in rates.items()
    }
    loop = {
        "heldout_champion": {"digest": "sha256:protected"},
        "history": [
            {
                "iteration": 2,
                "completed": True,
                "candidate": {
                    "digest": "sha256:candidate",
                    "path": "/checkpoints/iter_00002.pt",
                },
                "heldout_audit": {
                    "passed": True,
                    "checkpoint_digest": "sha256:candidate",
                    "valid_games": 1000,
                    "exact_distribution": True,
                    "exact_weights": True,
                    "greedy_required": True,
                    "per_opponent": audit_rows,
                },
                "raw_heldout_gate": {
                    "games": 1000,
                    "win_rate": 0.532,
                    "confidence_lower": 0.501,
                    "confidence_upper": 0.563,
                    "passed": False,
                    "reason": "per_opponent_floor",
                    "per_opponent": gate_rows,
                },
                "learner_after": {"digest": "sha256:protected"},
                "heldout_champion_updated": False,
            }
        ],
    }

    result = latest_committed_official_heldout_state(loop, run_dir)

    assert result["available"] is True
    assert result["iteration"] == 2
    assert result["games"] == 1000
    assert result["wr"] == 0.532
    assert result["protected_champion"] is False
    assert result["learner_retained"] is False
    assert {row["opponent_id"] for row in result["matchups"]} == set(rates)


def test_baseline_card_separates_latest_attempt_from_protected_champion() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'id="baseline-protected"' in html
    assert 'id="baseline-latest"' in html
    assert "latestHeldout=c.latest_official_heldout||{}" in html
    assert "shownHeldout=latestHeldout.available?latestHeldout:gate" in html
    assert "exact research-control games" in html
    assert 'data-widget="protocol"' in html
    assert "protocol=d.specialist_protocol||{}" in html


def test_latest_formal_holdout_displays_failed_gate_when_audit_passed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    (run_dir / "commits" / "iter_00009.json").write_text("{}")
    digest = "sha256:" + "b" * 64
    ids = tuple(f"premium-{index}" for index in range(8))
    audit_rows = {
        opponent_id: {"games": 250, "seat0": 125, "seat1": 125}
        for opponent_id in ids
    }
    matchups = [
        {
            "opponent_id": opponent_id,
            "games": 250,
            "seat0": 125,
            "seat1": 125,
            "wins": 83,
            "draws": 0,
            "losses": 167,
            "wr": 0.332,
        }
        for opponent_id in ids
    ]
    loop = {
        "heldout_champion": {"digest": "sha256:" + "a" * 64},
        "history": [
            {
                "iteration": 9,
                "completed": True,
                "candidate": {"digest": digest, "path": "/checkpoints/iter_00009.pt"},
                "heldout_audit": {
                    "passed": True,
                    "checkpoint_digest": digest,
                    "valid_games": 2000,
                    "exact_distribution": True,
                    "exact_weights": True,
                    "greedy_required": True,
                    "per_opponent": audit_rows,
                },
                "active_gate_result": {
                    "games": 2000,
                    "skill_weighted_wr": 0.332,
                    "confidence_lower": 0.3163,
                    "confidence_upper": 0.348,
                    "passed": False,
                    "pipeline_gate_reason": "active_gate_criteria_failed",
                    "matchups": matchups,
                },
                "learner_after": {"digest": digest},
                "heldout_champion_updated": False,
            }
        ],
    }

    state = latest_committed_formal_holdout_state(loop, run_dir)

    assert state["available"] is True
    assert state["iteration"] == 9
    assert state["games"] == 2000
    assert state["wr"] == 0.332
    assert state["passed"] is False
    assert state["audit_passed"] is True
    assert state["protected_champion"] is False
    assert len(state["matchups"]) == 8


def test_outcomes_panel_prefers_latest_formal_holdout() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert "latestFormal=c.latest_formal_holdout||{}" in html
    assert "committedHoldout=latestFormalComplete?latestFormal:" in html


def test_latest_official_panel_prefers_new_committed_research_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "research_controls/iter_00023.json"
    _write_json(source, {"committed": True})
    ids = ("iono", "dragapult-ex", "mega-abomasnow-ex", "mega-lucario-ex")
    rows = [
        {
            "opponent_id": opponent_id,
            "games": 250,
            "seat0": 125,
            "seat1": 125,
            "wins": 125.0,
            "draws": 0,
            "losses": 125,
            "win_rate": 0.5,
        }
        for opponent_id in ids
    ]
    result = {
        "iteration": 23,
        "checkpoint": "/checkpoints/iter_00023.pt",
        "checkpoint_digest": "sha256:" + "2" * 64,
        "games": 1000,
        "wins": 500.0,
        "draws": 0,
        "losses": 500,
        "win_rate": 0.5,
        "action_selection": "greedy",
        "training_eligible": False,
        "replay_eligible": False,
        "matchups": rows,
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
        },
    }
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "latest_committed_research_control_result",
        lambda _run_dir: (result, source),
    )
    loop = {
        "heldout_champion": {"digest": "sha256:" + "1" * 64},
        "history": [
            {
                "iteration": 23,
                "completed": True,
                "learner_after": {"digest": result["checkpoint_digest"]},
            }
        ],
    }

    state = latest_committed_official_heldout_state(loop, tmp_path)

    assert state["available"] is True
    assert state["kind"] == "latest_committed_research_control_result"
    assert state["iteration"] == 23
    assert state["games"] == 1000
    assert state["wr"] == 0.5
    assert state["learner_retained"] is True
    assert state["training_eligible"] is False
    assert state["replay_eligible"] is False
    assert {row["opponent_id"] for row in state["matchups"]} == set(ids)


def test_specialist_protocol_state_validates_roster_and_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "specialists.yaml"
    path.write_text("schema_version: placeholder\n", encoding="utf-8")
    payload = {
        "schema_version": "poke_bot.specialist_state/v2",
        "protocol_schema_version": "poke_bot.rl_protocol/v1",
        "last_verified_at_utc": "2026-07-22T16:47:00Z",
        "allowed_status_values": {
            "specialist": [
                "rl_training",
                "restart_required",
                "passed_frozen",
                "unstarted",
            ],
        },
        "current": {
            "phase": "specialist_baseline_rl",
            "active_specialist": "alakazam",
            "next_action": "continue iteration 24",
        },
        "target_registry": {"required_target_count": 2},
        "training_priority": {
            "policy": "unfinished_without_passing_checkpoint_then_meta_share_descending",
            "handoff_override": {"priority_prefix": ["hammer-pult"]},
            "strict_post_spidops_prefix": {
                "decision_revision": 29,
                "ids": ["hammer-pult"],
                "missing_input_behavior": (
                    "block_fallback_and_recover_public_inputs"
                ),
            },
            "owner_removal": {
                "decision_revision": 28,
                "specialist_ids": [
                    "dragapult-blaziken",
                    "dragapult-dudunsparce",
                ],
                "counts_toward_completion": False,
            },
            "ordered_unfinished_ids_after_active": ["hammer-pult"],
            "source": {
                "generated_at_utc": "2026-07-22T17:09:07.357Z",
                "snapshot_id": "2026-06-22T05:47:35",
            },
            "rows": [
                {
                    "id": "hammer-pult",
                    "source_archetype": "Dragapult ex",
                    "share": 0.055105348460291734,
                    "mapping": "aggregate_family",
                }
            ],
        },
        "shared_core": {"status": "pending", "checkpoint": None},
        "heads_and_datasets": {
            "specialist_head_template": {
                "archetype_policy_head_required": True,
                "game_plan_policy_heads_required": True,
                "matchup_policy_heads_required": True,
                "matchup_routing": {
                    "causal_observable_state_only": True,
                    "opponent_package_identity_allowed": False,
                    "relevant_matchup_sequences_only": True,
                },
            }
        },
        "specialists": [
            {
                "id": "alakazam",
                "name": "Alakazam",
                "status": "rl_training",
                "active": True,
                "counters": {"bootstrap_epochs_completed": 25, "rl_iterations_completed": 24},
            },
            {
                "id": "hammer-pult",
                "name": "Hammer Pult",
                "status": "restart_required",
                "active": False,
                "counters": {"bootstrap_epochs_completed": 18, "rl_iterations_completed": 0},
            },
        ],
        "population_training": {
            "status": "unstarted",
            "enabled": False,
            "all_required_specialists_passed": False,
        },
        "unresolved_facts": ["shared core pending"],
    }
    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda _text: payload
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    state = specialist_protocol_state(path)

    assert state["available"] is True
    assert state["active_specialist"] == "alakazam"
    assert state["status_counts"] == {"rl_training": 1, "restart_required": 1}
    assert state["head_requirements"]["matchup_policies"] is True
    assert state["head_requirements"]["causal_observable_state_only"] is True
    assert state["head_requirements"]["opponent_package_identity_allowed"] is False
    assert state["training_priority"]["next_specialist"] == "hammer-pult"
    assert state["training_priority"]["strict_post_spidops_prefix"] == {
        "decision_revision": 29,
        "ids": ["hammer-pult"],
        "missing_input_behavior": "block_fallback_and_recover_public_inputs",
    }
    assert state["training_priority"]["owner_removal"]["specialist_ids"] == [
        "dragapult-blaziken",
        "dragapult-dudunsparce",
    ]
    hammer = next(row for row in state["specialists"] if row["id"] == "hammer-pult")
    assert hammer["status"] == "restart_required"
    assert hammer["bootstrap_epochs_completed"] == 18
    assert hammer["rank_after_active"] == 1
    assert hammer["meta_share"] == pytest.approx(0.055105348460291734)

    population_path = tmp_path / "population-state.json"
    _write_json(
        population_path,
        {
            "schema": "poke_bot.population_round_robin_state/v1",
            "status": "training",
            "member_count": 2,
            "population_cycle": 4,
            "active_member_index": 1,
            "active_specialist_id": "hammer-pult",
            "members": [
                {
                    "specialist_id": f"specialist-{index:02d}",
                    "cycles_completed": 1 if index < 1 else 0,
                    "rl_epochs_completed": 5 if index < 1 else 0,
                    "rehearsal_epochs_completed": 5 if index < 1 else 0,
                }
                for index in range(2)
            ],
        },
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "POPULATION_ROUND_ROBIN_STATE",
        population_path,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda _name, user=False: {"active": True},
    )
    population_overlay = specialist_protocol_state(path)
    runtime = population_overlay["population_training"]["runtime"]
    assert population_overlay["population_training"]["status"] == "training"
    assert population_overlay["population_training"]["enabled"] is True
    assert runtime["active_specialist_id"] == "hammer-pult"
    assert runtime["completed_member_cycles"] == 1
    assert runtime["rl_epochs_completed"] == 5
    assert runtime["rehearsal_epochs_completed"] == 5

    runtime_state = specialist_protocol_state(
        path,
        runtime_specialist_id="hammer-pult",
        runtime_run_name="pure_rl_hammer_pult_live",
    )
    assert runtime_state["available"] is True
    assert runtime_state["active_specialist"] == "hammer-pult"
    assert runtime_state["canonical_active_specialist"] == "alakazam"
    assert runtime_state["runtime_reconciled"] is True
    assert runtime_state["canonical_pointer_stale"] is True
    assert "live Hammer Pult specialist" in runtime_state["next_action"]
    assert "2 specialists remain unfinished including the active" in (
        runtime_state["next_action"]
    )
    assert "Population training remains blocked" in runtime_state["next_action"]
    assert next(
        row for row in runtime_state["specialists"] if row["id"] == "hammer-pult"
    )["active"] is True
    assert next(
        row for row in runtime_state["specialists"] if row["id"] == "alakazam"
    )["active"] is False

    same_identity_state = specialist_protocol_state(
        path,
        runtime_specialist_id="alakazam",
        runtime_run_name="pure_rl_alakazam_live",
    )
    assert same_identity_state["runtime_identity_reconciled"] is False
    assert same_identity_state["canonical_pointer_stale"] is False

    payload["current"] = {
        "phase": "shared_core_derivation",
        "active_specialist": None,
        "next_action": "distill shared core",
    }
    payload["specialists"][0]["status"] = "passed_frozen"
    payload["specialists"][0]["active"] = False
    payload["specialists"][1]["status"] = "unstarted"

    transition_state = specialist_protocol_state(path)

    assert transition_state["available"] is True
    assert transition_state["active_specialist"] == ""
    assert transition_state["phase"] == "shared_core_derivation"
    assert transition_state["training_priority"]["next_specialist"] == "hammer-pult"

    payload["current"]["phase"] = "specialist_core_refresh_handoff"

    cycle_transition_state = specialist_protocol_state(path)

    assert cycle_transition_state["available"] is True
    assert cycle_transition_state["active_specialist"] == ""
    assert cycle_transition_state["phase"] == "specialist_core_refresh_handoff"


def test_runtime_specialist_identity_prefers_explicit_command() -> None:
    command = (
        "python scripts/train_pure_rl.py --run-name misleading_alakazam "
        "--specialist-archetype starmie"
    )
    assert _specialist_id_from_runtime(command, "misleading_alakazam") == "starmie"
    assert (
        _specialist_id_from_runtime("", "pure_rl_hops_trevenant_temporal1")
        == "hops-trevenant"
    )


def test_active_specialist_commit_overlay_supersedes_stale_yaml_counters(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "active-run"
    checkpoint = run_dir / "checkpoints/iter_00002.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"candidate")
    digest = "sha256:" + hashlib.sha256(b"candidate").hexdigest()
    _write_json(
        run_dir / "loop_state.json",
        {"last_completed_iteration": 2, "next_iteration": 3},
    )
    history = []
    for iteration in range(3):
        candidate = {
            "path": str(
                checkpoint
                if iteration == 2
                else run_dir / f"checkpoints/iter_{iteration:05d}.pt"
            ),
            "digest": digest,
        }
        history.append(
            {
                "iteration": iteration,
                "completed": True,
                "candidate": candidate,
                "active_gate_result": (
                    {
                        "iteration": iteration,
                        "checkpoint_digest": digest,
                        "games": 2250,
                    }
                    if iteration == 2
                    else None
                ),
                "incumbent_after": candidate,
                "heldout_champion_updated": iteration == 2,
                "promoted": iteration == 2,
                "promotion": {
                    "continuous_learner": {
                        "exact_gate_regression": {
                            "enabled": True,
                            "streak": 1,
                            "patience": 2,
                        }
                    }
                },
            }
        )
    _write_json(
        run_dir / "commits/iter_00002.json",
        {
            "history": history,
            "champion": {
                "path": str(checkpoint),
                "digest": digest,
            },
            "learner": {
                "path": str(checkpoint),
                "digest": digest,
            },
        },
    )
    _write_json(
        run_dir / "research_controls/iter_00002.json",
        {
            "iteration": 2,
            "checkpoint_digest": digest,
            "games": 1000,
        },
    )

    overlay = active_specialist_commit_overlay({"path": str(run_dir)})

    assert overlay["available"] is True
    assert overlay["last_completed_iteration"] == 2
    assert overlay["next_iteration"] == 3
    assert overlay["rl_iterations_completed"] == 3
    assert overlay["learner_checkpoint"] == str(checkpoint)
    assert overlay["learner_digest"] == digest
    assert overlay["premium_holdout"]["games"] == 2250
    assert overlay["official_research"]["games"] == 1000
    assert overlay["checkpoint_digest"] == digest
    assert overlay["protected_champion"]["digest"] == digest
    assert overlay["candidate_promoted"] is True
    assert overlay["heldout_champion_updated"] is True
    assert overlay["exact_gate_regression"]["streak"] == 1


def test_active_specialist_commit_overlay_preserves_rolled_back_learner(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "active-run"
    candidate_digest = "sha256:" + "a" * 64
    learner_digest = "sha256:" + "b" * 64
    learner = {
        "path": str(run_dir / "checkpoints/iter_00001.pt"),
        "digest": learner_digest,
    }
    _write_json(
        run_dir / "loop_state.json",
        {"last_completed_iteration": 2, "next_iteration": 3},
    )
    _write_json(
        run_dir / "commits/iter_00002.json",
        {
            "history": [
                {
                    "iteration": 2,
                    "completed": True,
                    "candidate": {
                        "path": str(run_dir / "checkpoints/iter_00002.pt"),
                        "digest": candidate_digest,
                    },
                    "learner_after": learner,
                }
            ],
            "learner": learner,
        },
    )

    overlay = active_specialist_commit_overlay({"path": str(run_dir)})

    assert overlay["available"] is True
    assert overlay["checkpoint_digest"] == candidate_digest
    assert overlay["learner_digest"] == learner_digest
    assert overlay["learner_checkpoint"].endswith("iter_00001.pt")


def test_active_specialist_commit_overlay_rejects_cross_checkpoint_results(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "active-run"
    digest = "sha256:" + "a" * 64
    _write_json(
        run_dir / "loop_state.json",
        {"last_completed_iteration": 0, "next_iteration": 1},
    )
    _write_json(
        run_dir / "commits/iter_00000.json",
        {
            "history": [
                {
                    "iteration": 0,
                    "completed": True,
                    "candidate": {
                        "path": str(run_dir / "checkpoints/iter_00000.pt"),
                        "digest": digest,
                    },
                    "active_gate_result": {
                        "iteration": 0,
                        "checkpoint_digest": "sha256:" + "b" * 64,
                    },
                }
            ]
        },
    )

    overlay = active_specialist_commit_overlay({"path": str(run_dir)})

    assert overlay["available"] is False
    assert "identity is inconsistent" in overlay["reason"]


def test_active_gate_is_current_public_roster_plus_s_plus_and_research_is_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "ops/alakazam_gate_program_v1.json").read_text(encoding="utf-8")
    )

    assert contract["active_gate_id"] == contract["next_gate"]["id"]
    semantics = contract["active_gate_semantics"]
    assert semantics["gate_roster_size"] == len(contract["next_gate"]["roster"])
    assert semantics["games_per_opponent"] == 250
    assert semantics["gate_games_total"] == 250 * len(
        contract["next_gate"]["roster"]
    )
    assert semantics["frozen_specialist_tier"] == "S+"
    assert semantics["original_four_role"] == "research_control_only"
    assert semantics["original_four_gate_weight"] == 0.0
    active = contract["next_gate"]
    assert active["id"] == contract["active_gate_id"]
    assert len(active["roster"]) == semantics["gate_roster_size"]
    assert active["evaluation"]["games_total"] == semantics["gate_games_total"]
    assert active["roster"][-1]["tier"] == "S+"
    assert active["roster"][-1]["frozen_specialist"] is True
    assert all(row["gate_weight"] == 0.0 for row in active["research_measurements"])
    assert all(
        row["included_in_gate_pass"] is False
        for row in active["research_measurements"]
    )


def test_dashboard_has_committed_holdout_by_deck_and_specialist_mix_panel() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'data-card="outcomes"' in html
    assert "Latest completed formal holdout" in html
    assert "Latest completed holdout by opponent deck" in html
    assert 'id="last-holdout-percent"' in html
    assert 'id="last-holdout-rows"' in html


def test_dashboard_head_loss_panel_is_driven_by_canonical_head_registry() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'id="head-loss-grid"' in html
    assert "Object.entries(heads)" in html
    assert "renderAllHeadLosses(model,m)" in html
    for head_id in (
        "policy",
        "value",
        "archetype",
        "opponent_hand",
        "opponent_remainder",
        "lethal_threat",
        "prize_race",
        "action_q",
        "action_type",
        "action_target",
        "action_resource",
        "action_utility",
        "tactical_outcome",
        "opponent_response",
        "resource_forecast",
        "game_phase",
        "outcome_distribution",
        "remaining_turns",
    ):
        assert head_id in html


def test_existing_gate_roster_cards_fall_back_to_latest_committed_holdout() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert "committedGateById" in html
    assert "displayCommittedGate" in html
    assert "committed iteration " in html
    assert "formalHoldoutComplete=!!(heldout.available" in html
    assert (
        "committedHoldout=latestFormalComplete?latestFormal:"
        "formalHoldoutComplete?heldout" in html
    )
    assert "gateRosterById=new Map" in html
    assert "exact greedy games" in html
    assert "candidate first/second" in html
    assert "NO COMMITTED RESULT" in html
    assert "immutable completed active-gate result" in html
    assert 'id="specialist-mix-summary"' in html
    assert 'id="specialist-mix-rows"' in html
    assert "AWAITING NEXT ITERATION" in html
    assert "this is not a zero-game result" in html
    assert "sampled replay-eligible public mix" in html
    assert "public_mix_opponent_ids" in html
    assert "separate exact holdout is gate evidence" in html
    assert COMPETITION_GATE_PROGRAM.parts[-2:] == (
        "ops",
        "alakazam_gate_program_v1.json",
    )


def _practice_gate_contract() -> dict:
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "ops/alakazam_gate_program_v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    gate = dict(contract["next_gate"])
    gate["available"] = True
    gate["contract_valid"] = True
    return gate


def _write_practice_plan(run_dir: Path, *, iteration: int = 9) -> Path:
    gate = _practice_gate_contract()
    roster = gate["roster"]
    per_opponent = {
        row["opponent_id"]: {
            "archetype_id": row["archetype_id"],
            "games": 2,
            "seat0": 1,
            "seat1": 1,
        }
        for row in roster
    }
    path = run_dir / "collection_plans" / f"iter_{iteration:05d}.json"
    _write_json(
        path,
        {
            "schema": "poke_bot.strong_public_practice_plan/v1",
            "iteration": iteration,
            "active_gate_id": gate["id"],
            "games": 2 * len(roster),
            "per_opponent": per_opponent,
            "adaptive_weights": {
                row["opponent_id"]: 1.0 / len(roster) for row in roster
            },
            "sampled_policy": True,
            "temperature": 0.35,
            "training_eligible": True,
            "formal_eval": False,
            "seed_disjoint": True,
            "seed_namespace": "train/strong-public-practice-v1",
            "formal_seed_namespace": "eval/strong-public-fixed-manifest-v1",
        },
    )
    return path


def test_practice_plan_is_current_training_only_and_gate_reconciled(
    tmp_path: Path,
) -> None:
    source = _write_practice_plan(tmp_path)
    state = strong_public_practice_plan_state(
        tmp_path,
        9,
        _practice_gate_contract(),
    )

    assert state["available"] is True
    assert state["games"] == 2 * len(_practice_gate_contract()["roster"])
    assert state["roster_size"] == len(_practice_gate_contract()["roster"])
    assert state["temperature"] == 0.35
    assert state["training_eligible"] is True
    assert state["formal_eval"] is False
    assert state["seed_disjoint"] is True
    assert state["source"] == str(source)
    assert {row["opponent_id"] for row in state["per_opponent"]} == {
        row["opponent_id"] for row in _practice_gate_contract()["roster"]
    }


def test_practice_plan_fails_closed_on_matchup_archetype_cross_contamination(
    tmp_path: Path,
) -> None:
    path = _write_practice_plan(tmp_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    first_id = next(iter(plan["per_opponent"]))
    plan["per_opponent"][first_id]["archetype_id"] = "wrong-matchup"
    _write_json(path, plan)

    state = strong_public_practice_plan_state(
        tmp_path,
        9,
        _practice_gate_contract(),
    )

    assert state["available"] is False
    assert "per-opponent archetype/seat/weight" in state["reason"]


def test_practice_plan_never_falls_back_to_a_prior_iteration(tmp_path: Path) -> None:
    _write_practice_plan(tmp_path, iteration=8)

    state = strong_public_practice_plan_state(
        tmp_path,
        9,
        _practice_gate_contract(),
    )

    assert state["available"] is False
    assert state["iteration"] == 9
    assert "not written yet" in state["reason"]
    assert state["source"].endswith("collection_plans/iter_00009.json")


def test_official_gateline_labels_exact_external_seed_as_nonterminal_anchor(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    rates = {
        "iono": 0.408,
        "dragapult-ex": 0.52,
        "mega-abomasnow-ex": 0.552,
        "mega-lucario-ex": 0.612,
    }
    report = {
        "valid": True,
        "trusted_formal": True,
        "formal_mode": "policy",
        "failures": [],
        "scheduled_jobs": 1000,
        "completed_jobs": 1000,
        "checkpoint": {"digest": digest},
        "pooled_formal": {
            "games": 1000,
            "wr": sum(rates.values()) / 4,
        },
        "deck_agnostic_gate": {"exact_deck_seat_balance": True},
        "matchups": [
            {
                "opponent_id": opponent_id,
                "games": 250,
                "wins": rate * 250,
                "losses": (1.0 - rate) * 250,
                "draws": 0.0,
            }
            for opponent_id, rate in rates.items()
        ],
    }
    report_path = tmp_path / "seed-audit.json"
    payload = json.dumps(report).encode()
    report_path.write_bytes(payload)
    loop = {
        "heldout_champion": {"path": "/protected/seed.pt", "digest": digest},
        "heldout_champion_evidence": {
            "iteration": -1,
            "checkpoint_digest": digest,
            "games": 1000,
            "win_rate": sum(rates.values()) / 4,
            "confidence_lower": 0.491,
            "confidence_upper": 0.553,
            "per_opponent": {
                opponent_id: {
                    "games": 250,
                    "wins": rate * 250,
                    "win_rate": rate,
                }
                for opponent_id, rate in rates.items()
            },
            "audit": {
                "passed": True,
                "source": "trusted_external_new_lineage_anchor",
                "terminal_gate_eligible": False,
                "report": {
                    "path": str(report_path),
                    "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                },
            },
        },
        "history": [],
    }

    result = committed_official_heldout_state(loop, tmp_path)

    assert result["available"] is True
    assert result["kind"] == "external_seed_official_heldout_anchor"
    assert result["passed"] is False
    assert result["terminal_gate_eligible"] is False
    assert result["wr"] == sum(rates.values()) / 4
    assert result["matchups"][0]["opponent_id"] == "iono"
    assert result["matchups"][0]["seat0"] == 125
    assert result["matchups"][0]["seat1"] == 125

    report_path.write_text("{}")
    failed = committed_official_heldout_state(loop, tmp_path)
    assert failed["available"] is False
    assert failed["reason"] == "seed audit report failed reconciliation"


def test_official_gateline_keeps_exact_result_across_lineage_handoff() -> None:
    digest = "sha256:a"
    loop = {
        "heldout_champion": {"path": "/checkpoints/source.pt", "digest": digest},
        "heldout_champion_evidence": None,
    }
    audit = {
        "passed": True,
        "valid_games": 1000,
        "exact_distribution": True,
        "exact_weights": True,
        "greedy_required": True,
        "per_opponent": {
            "iono": {"games": 1000, "seat0": 500, "seat1": 500}
        },
    }
    handoff = {
        "source_run": "source-v6",
        "source_global_iteration_offset": 11,
        "inherited_official_heldout": {
            "games": 1000,
            "wr": 0.299,
            "lower": 0.2714,
            "upper": 0.3281,
            "passed": False,
            "reason": "per_opponent_floor",
            "lineage_iteration": 26,
            "checkpoint_digest": digest,
            "audit": audit,
            "per_opponent": {
                "iono": {
                    "games": 1000,
                    "win_rate": 0.299,
                    "wins": 299,
                    "draws": 0,
                    "losses": 701,
                }
            },
        },
    }

    result = committed_official_heldout_state(loop, None, handoff=handoff)

    assert result["available"] is True
    assert result["kind"] == "inherited_official_heldout_champion"
    assert result["wr"] == 0.299
    assert result["iteration"] == 37
    assert result["matchups"][0]["seat0"] == 500


def test_dashboard_official_gateline_prefers_curriculum_heldout() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "dashboard/lan/index.html").read_text(encoding="utf-8")
    assert "const heldout=c.official_heldout||{}" in html
    assert "Archived original-four selection history" in html
    assert "latestHeldout=c.latest_official_heldout||{}" in html
    assert "exact audit" in html
    assert "exact research-control games · seats ${x.seat0" in html


def test_competition_gate_program_reconciles_accepted_and_sampled_evidence(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "ops/alakazam_gate_program_v1.json"
    contract = json.loads(contract_path.read_text())
    accepted = contract["accepted_gate"]
    digest = accepted["checkpoint_digest"]
    exact = accepted["exact_holdout"]
    registry_path = tmp_path / "registry.json"
    registry = {
        "checkpoint_digest": digest,
        "immutable": True,
        "automatic_pruning_allowed": False,
        "evidence": {
            "checkpoint_digest": digest,
            "games": exact["games"],
            "win_rate": exact["win_rate"],
            "confidence_lower": exact["confidence_lower"],
            "audit": {
                "checkpoint_digest": digest,
                "passed": True,
                "exact_distribution": True,
                "exact_weights": True,
                "greedy_required": True,
                "valid_games": exact["games"],
            },
        },
    }
    _write_json(registry_path, registry)
    official = {
        "available": True,
        "valid": True,
        "audit_passed": True,
        "exact_distribution": True,
        "exact_weights": True,
        "greedy_required": True,
        "checkpoint_digest": digest,
        "games": exact["games"],
        "wr": exact["win_rate"],
        "lower": exact["confidence_lower"],
    }
    matchups = [
        {
            "opponent_id": row["opponent_id"],
            "games": 10,
            "win_rate": 0.25 + index * 0.01,
            "seat0": 5,
            "seat1": 5,
        }
        for index, row in enumerate(contract["next_gate"]["roster"])
    ]
    public_mix = {
        "available": True,
        "checkpoint_mixed": False,
        "checkpoint_digest": digest,
        "games": 10 * len(matchups),
        "iteration": 1,
        "matchups": matchups,
    }

    state = competition_gate_program_state(
        official,
        public_mix,
        contract_path=contract_path,
        registry_path=registry_path,
        exact_result_override={},
    )

    assert state["accepted_gate"]["accepted"] is True
    assert state["accepted_gate"]["registry_protected"] is True
    assert state["accepted_gate"]["exact_holdout"] == exact
    submissions = state["accepted_gate"]["submissions"]
    assert len(submissions) == 3
    assert submissions[-1]["ref"] == 54883731
    assert submissions[-1]["authorization"].startswith("unauthorized")
    next_gate = state["next_gate"]
    assert next_gate["available"] is True
    assert state["active_gate_id"] == contract["active_gate_id"]
    assert state["active_gate_semantics"]["gate_games_total"] == (
        contract["next_gate"]["evaluation"]["games_total"]
    )
    assert next_gate["status"] == "queued"
    assert len(next_gate["roster"]) == len(contract["next_gate"]["roster"])
    assert len({row["content_digest"] for row in next_gate["roster"]}) == len(
        contract["next_gate"]["roster"]
    )
    assert all(row["archetype_label"] for row in next_gate["roster"])
    assert next_gate["diagnostic"]["available"] is True
    assert next_gate["diagnostic"]["games"] == 10 * len(matchups)
    assert next_gate["diagnostic"]["roster_coverage"] == 1.0
    assert next_gate["exact_result_available"] is False
    assert next_gate["research_measurements_valid"] is True
    assert len(next_gate["research_measurements"]) == 4
    assert sum(row["games"] for row in next_gate["research_measurements"]) == 1000
    assert all(row["gate_weight"] == 0 for row in next_gate["research_measurements"])
    assert all(row["archetype_label"] for row in next_gate["research_measurements"])

    fallback_state = competition_gate_program_state(
        official,
        public_mix,
        contract_path=contract_path,
        registry_path=registry_path,
        exact_result_override={},
        completed_iteration=4,
    )
    fallback_gate = fallback_state["next_gate"]
    assert fallback_gate["fallback_active"] is True
    assert fallback_gate["effective_gate_id"] == (
        contract["fallback_transition"]["id"]
    )
    assert fallback_gate["pass_criteria"][
        "skill_weighted_confidence_lower"
    ] == 0.55
    assert fallback_gate["effective_pass_criteria"][
        "skill_weighted_confidence_lower"
    ] == 0.50
    assert fallback_gate["threshold_transition"]["status"] == "active"

    # The archived accepted milestone reconciles against its immutable model
    # registry, not whichever newer active-gate checkpoint the curriculum has.
    official["checkpoint_digest"] = "sha256:new-active-gate-candidate"
    mismatch = competition_gate_program_state(
        official,
        public_mix,
        contract_path=contract_path,
        registry_path=registry_path,
        exact_result_override={},
    )
    assert mismatch["accepted_gate"]["accepted"] is True

    corrupted_registry = dict(registry)
    corrupted_registry["checkpoint_digest"] = "sha256:corrupted-registry"
    _write_json(registry_path, corrupted_registry)
    corrupted = competition_gate_program_state(
        official,
        public_mix,
        contract_path=contract_path,
        registry_path=registry_path,
        exact_result_override={},
    )
    assert corrupted["accepted_gate"]["accepted"] is False
    assert corrupted["accepted_gate"]["exact_holdout"] == {}

    invalid_contract = json.loads(contract_path.read_text())
    invalid_contract["active_gate_semantics"]["gate_games_total"] = 1000
    invalid_contract_path = tmp_path / "invalid-gate-program.json"
    _write_json(invalid_contract_path, invalid_contract)
    invalid = competition_gate_program_state(
        official,
        public_mix,
        contract_path=invalid_contract_path,
        registry_path=registry_path,
        exact_result_override={},
    )
    assert invalid["next_gate"]["available"] is False
    assert "active gate identity" in invalid["next_gate"]["contract_reason"]


def test_active_gate_runtime_keeps_contract_target_during_research_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "ops/alakazam_gate_program_v1.json"
    gate_state = competition_gate_program_state(
        {},
        {},
        contract_path=contract_path,
        registry_path=root / "missing-protected-registry.json",
    )["next_gate"]
    monkeypatch.setattr(dashboard_snapshot_module, "run", lambda *_args, **_kwargs: "active")
    monkeypatch.setattr(dashboard_snapshot_module, "read_tail", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "parse_curriculum_progress",
        lambda *_args, **_kwargs: {
            "stage": "measure:research_controls",
            "iteration": 7,
            "current": 375,
            "total": 1000,
            "percent": 37.5,
            "gps": 4.25,
            "sps": 0.0,
            "remotes": 12,
            "line": "research-only progress",
        },
    )

    runtime = strong_public_gate_runtime_state(gate_state)

    assert runtime["available"] is True
    assert runtime["contract_aligned"] is True
    assert runtime["phase"] == "research_controls"
    gate_total = gate_state["evaluation"]["games_total"]
    assert runtime["current"] == gate_total
    assert runtime["total"] == gate_total
    assert runtime["percent"] == 100.0
    assert runtime["roster_size"] == len(gate_state["roster"])
    assert runtime["games_per_opponent"] == 250
    assert runtime["phase_current"] == 375
    assert runtime["phase_total"] == 1000


def test_active_gate_runtime_prefers_main_curriculum_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    gate_state = competition_gate_program_state(
        {},
        {},
        contract_path=root / "ops/alakazam_gate_program_v1.json",
        registry_path=root / "missing-protected-registry.json",
    )["next_gate"]

    def standalone_service_must_not_be_queried(*_args, **_kwargs) -> str:
        raise AssertionError("main trainer progress should be authoritative")

    monkeypatch.setattr(
        dashboard_snapshot_module,
        "run",
        standalone_service_must_not_be_queried,
    )
    runtime = strong_public_gate_runtime_state(
        gate_state,
        curriculum_progress={
            "stage": "heldout:strong_public_gate",
            "iteration": 8,
            "current": 625,
            "total": gate_state["evaluation"]["games_total"],
            "percent": (
                100.0
                * 625
                / gate_state["evaluation"]["games_total"]
            ),
            "gps": 5.5,
            "sps": 0.0,
            "remotes": 16,
            "line": "main strong-public progress",
        },
        curriculum_active=True,
    )

    assert runtime["active"] is True
    assert runtime["phase"] == "active_gate"
    assert runtime["current"] == 625
    assert runtime["total"] == gate_state["evaluation"]["games_total"]
    assert runtime["percent"] == pytest.approx(
        100.0 * 625 / gate_state["evaluation"]["games_total"]
    )
    assert runtime["source"] == "main curriculum run-bound progress"


def test_active_gate_runtime_does_not_fall_back_while_curriculum_collects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    gate_state = competition_gate_program_state(
        {},
        {},
        contract_path=root / "ops/alakazam_gate_program_v1.json",
        registry_path=root / "missing-protected-registry.json",
    )["next_gate"]

    def stale_standalone_must_not_be_queried(*_args, **_kwargs) -> str:
        raise AssertionError("a live curriculum owns all gate progress telemetry")

    monkeypatch.setattr(
        dashboard_snapshot_module,
        "run",
        stale_standalone_must_not_be_queried,
    )
    runtime = strong_public_gate_runtime_state(
        gate_state,
        curriculum_progress={
            "stage": "collect:self_play",
            "iteration": 9,
            "current": 512,
            "total": 1024,
        },
        curriculum_active=True,
    )

    assert runtime["active"] is False
    assert runtime["phase"] == "idle"
    assert runtime["current"] == 0
    assert runtime["total"] == gate_state["evaluation"]["games_total"]
    assert runtime["source"] == "main curriculum run-bound progress"


def test_latest_active_gate_result_ignores_uncommitted_eval(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    _write_json(
        eval_dir / "iter_00006.json",
        {"active_gate_result": {"gate_id": "gate", "iteration": 6}},
    )

    result, source = latest_committed_active_gate_result(tmp_path)
    assert result == {}
    assert source is None


def test_active_gate_contract_follows_verified_latest_design_migration(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contracts" / "lc55.json"
    payload = b'{"schema":"poke_bot.competition_gate_program/v1"}'
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    _write_json(
        tmp_path / "design_migrations" / "migration_0001.json",
        {
            "current_contract": {
                "gates": {
                    "active_contract": {
                        "path": str(contract_path),
                        "digest": digest,
                        "size": len(payload),
                    }
                }
            }
        },
    )
    assert active_gate_contract_for_run(tmp_path) == contract_path.resolve()

    contract_path.write_bytes(payload + b"\n")
    assert active_gate_contract_for_run(tmp_path) == COMPETITION_GATE_PROGRAM


def test_active_gate_contract_uses_verified_initial_specialist_manifest(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contracts" / "specialist-lc50.json"
    payload = b'{"schema":"poke_bot.competition_gate_program/v1"}'
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    _write_json(
        tmp_path / "manifest.json",
        {
            "design_contract": {
                "gates": {
                    "active_contract": {
                        "path": str(contract_path),
                        "digest": digest,
                        "size": len(payload),
                    }
                }
            }
        },
    )

    assert active_gate_contract_for_run(tmp_path) == contract_path.resolve()

    contract_path.write_bytes(payload + b"\n")
    assert active_gate_contract_for_run(tmp_path) == COMPETITION_GATE_PROGRAM


def test_latest_active_gate_result_sources_immutable_history_and_exact_pointer(
    tmp_path: Path,
) -> None:
    result_core = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "gate_id": "gate",
        "iteration": 6,
        "checkpoint_digest": "sha256:" + "a" * 64,
        "games": 2000,
    }
    commit = {
        "last_completed_iteration": 6,
        "next_iteration": 7,
        "history": [
            {
                "iteration": 6,
                "completed": True,
                "active_gate_result": result_core,
            }
        ],
    }
    commit_path = tmp_path / "commits/iter_00006.json"
    _write_json(commit_path, commit)
    _write_json(
        tmp_path / "eval/iter_00006.json",
        {"active_gate_result": {**result_core, "games": 1}},
    )

    result, source = latest_committed_active_gate_result(tmp_path)
    assert result == result_core
    assert source == commit_path

    commit_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            commit,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    pointer_path = tmp_path / "active_gate_result.json"
    pointer = {
        **result_core,
        "committed": True,
        "commit": str(commit_path.resolve()),
        "commit_digest": commit_digest,
        "created_at_utc": "2026-07-21T12:00:00Z",
    }
    _write_json(pointer_path, pointer)
    result, source = latest_committed_active_gate_result(
        tmp_path,
        mutable_result_pointer=pointer_path,
    )
    assert result == pointer
    assert source == pointer_path.resolve()

    _write_json(pointer_path, {**pointer, "games": 1999})
    result, source = latest_committed_active_gate_result(
        tmp_path,
        mutable_result_pointer=pointer_path,
    )
    assert result == result_core
    assert source == commit_path

    _write_json(pointer_path, {**pointer, "commit_digest": "sha256:" + "0" * 64})
    result, source = latest_committed_active_gate_result(
        tmp_path,
        mutable_result_pointer=pointer_path,
    )
    assert result == result_core
    assert source == commit_path


def test_dashboard_uses_only_committed_dedicated_research_artifact(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "ops/research_control_registry_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    iteration = 4
    result_path = (
        tmp_path / "research_controls" / f"iter_{iteration:05d}.json"
    ).resolve()
    matchups = [
        {
            "opponent_id": row["opponent_id"],
            "content_digest": row["content_digest"],
            "games": 250,
            "wins": 125.0,
            "draws": 0,
            "losses": 125,
            "seat0": 125,
            "seat1": 125,
            "win_rate": 0.5,
        }
        for row in registry["controls"]
    ]
    result = {
        "schema": "poke_bot.research_control_measurement_result/v1",
        "iteration": iteration,
        "registry_id": registry["registry_id"],
        "registry_version": registry["version"],
        "checkpoint_digest": "sha256:" + "a" * 64,
        "schedule_digest": "sha256:" + "b" * 64,
        "seed_namespace": "eval/research-controls-fixed-manifest-v1",
        "training_eligible": False,
        "replay_eligible": False,
        "diagnostic_only": True,
        "formal_eval": False,
        "included_in_gate_pass": False,
        "gate_weight": 0.0,
        "action_selection": "greedy",
        "games": 1000,
        "wins": 500.0,
        "draws": 0,
        "losses": 500,
        "win_rate": 0.5,
        "matchups": matchups,
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "seed_disjoint": True,
            "package_disjoint_from_active_gate": True,
            "replay_records_written": 0,
        },
        "result_path": str(result_path),
    }
    _write_json(result_path, result)
    commit = {
        "last_completed_iteration": iteration,
        "next_iteration": iteration + 1,
        "history": [
            {
                "iteration": iteration,
                "completed": True,
                "research_control_result": result,
            }
        ],
    }
    commit_path = tmp_path / "commits" / f"iter_{iteration:05d}.json"
    _write_json(commit_path, commit)

    committed, source = latest_committed_research_control_result(tmp_path)
    assert committed == result
    assert source == result_path
    conflicting_legacy = _research_telemetry(
        [_research_row("iono", content_digest=registry["controls"][0]["content_digest"])]
    )
    state = research_control_registry_state(
        conflicting_legacy,
        registry_path=registry_path,
        measurement_result=committed,
        measurement_source=source,
    )
    assert state["available"] is True
    assert state["games"] == 1000
    assert state["win_rate"] == 0.5
    assert state["stage"] == "measure:research_controls:complete"
    assert state["result_source"] == str(result_path)

    # A newer loose artifact has no authority until an immutable iteration
    # commit binds it. A tampered committed artifact is rejected entirely.
    _write_json(
        tmp_path / "research_controls/iter_00005.json",
        {**result, "iteration": 5},
    )
    assert latest_committed_research_control_result(tmp_path) == (
        result,
        result_path,
    )
    _write_json(result_path, {**result, "games": 999})
    assert latest_committed_research_control_result(tmp_path) == ({}, None)


def _research_telemetry(
    rows: list[dict],
    *,
    iteration: int = 3,
    checkpoint_digest: str | None = None,
) -> dict:
    checkpoint_digest = checkpoint_digest or "sha256:" + "9" * 64
    games = sum(int(row["games"]) for row in rows)
    wins = sum(float(row["wins"]) for row in rows)
    draws = sum(int(row["draws"]) for row in rows)
    losses = sum(int(row["losses"]) for row in rows)
    return {
        "research_controls": {
            "available": games > 0,
            "active": False,
            "iteration": iteration,
            "stage": "measure:research_controls:complete",
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / games if games else None,
            "checkpoint_digest": checkpoint_digest,
            "checkpoint_mixed": False,
            "checkpoint_digests": {checkpoint_digest: games},
            "matchups": rows,
        }
    }


def _research_row(
    opponent_id: str,
    *,
    content_digest: str | None = None,
) -> dict:
    row = {
        "opponent_id": opponent_id,
        "games": 2,
        "wins": 1.0,
        "draws": 0,
        "losses": 1,
        "seat0": 1,
        "seat1": 1,
        "win_rate": 0.5,
    }
    if content_digest is not None:
        row["opponent_content_digest"] = content_digest
    return row


@pytest.mark.parametrize("bad_weight", [None, "0", False, "invalid"])
def test_research_registry_rejects_missing_or_malformed_gate_weight(
    tmp_path: Path,
    bad_weight: object,
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "ops/research_control_registry_v1.json").read_text())
    if bad_weight is None:
        registry["controls"][0].pop("gate_weight")
    else:
        registry["controls"][0]["gate_weight"] = bad_weight
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)

    state = research_control_registry_state({}, registry_path=registry_path)
    assert state["available"] is False
    assert state["reason"] == "research-control registry failed validation"


def test_research_registry_rejects_forged_legacy_seed_digest(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "ops/research_control_registry_v1.json").read_text())
    registry["controls"][0]["content_digest"] = "sha256:" + "e" * 64
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)

    state = research_control_registry_state({}, registry_path=registry_path)
    assert state["available"] is False
    assert state["reason"] == (
        "research-control retirement history failed validation"
    )


def test_research_registry_surfaces_valid_retired_controls_dynamically(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "ops/research_control_registry_v1.json").read_text())
    retired_digest = "sha256:" + "5" * 64
    exact_digest = "sha256:" + "6" * 64
    checkpoint_digest = "sha256:" + "7" * 64
    registry["version"] = 2
    registry["controls"].append(
        {
            "opponent_id": "retired-strong-agent",
            "content_digest": retired_digest,
            "source_gate_id": "passed-gate-v2",
            "source": "immutable/package/path",
            "archetype_id": "retired-archetype",
            "archetype_label": "Retired Strong Agent",
            "retired_at_utc": "2026-07-21T12:00:00Z",
            "retired_exact_result_digest": exact_digest,
            "retired_checkpoint_digest": checkpoint_digest,
            "gate_weight": 0.0,
            "included_in_gate_pass": False,
            "formal_eval": False,
            "training_eligible": False,
        }
    )
    registry["retirements"].append(
        {
            "gate_id": "passed-gate-v2",
            "retired_at_utc": "2026-07-21T12:00:00Z",
            "exact_result_digest": exact_digest,
            "checkpoint_digest": checkpoint_digest,
            "iteration": 20,
            "opponent_ids": ["retired-strong-agent"],
        }
    )
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)
    telemetry = _research_telemetry(
        [_research_row("retired-strong-agent", content_digest=retired_digest)]
    )

    state = research_control_registry_state(
        telemetry,
        registry_path=registry_path,
    )

    assert state["available"] is True
    assert state["registry_version"] == 2
    assert state["control_count"] == 5
    by_id = {row["opponent_id"]: row for row in state["controls"]}
    assert by_id["retired-strong-agent"]["games"] == 2
    assert by_id["retired-strong-agent"]["win_rate"] == 0.5


def test_research_telemetry_fails_closed_on_identity_digest_or_aggregate_mix(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "ops/research_control_registry_v1.json").read_text())
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)
    iono_digest = registry["controls"][0]["content_digest"]
    valid = _research_telemetry(
        [_research_row("iono", content_digest=iono_digest)]
    )
    assert research_control_registry_state(
        valid,
        registry_path=registry_path,
    )["available"] is True

    unexpected = _research_telemetry([_research_row("unregistered-agent")])
    unexpected_state = research_control_registry_state(
        unexpected,
        registry_path=registry_path,
    )
    assert unexpected_state["available"] is False
    assert unexpected_state["unexpected_opponents"] == ["unregistered-agent"]

    duplicate = _research_telemetry(
        [_research_row("iono"), _research_row("iono")]
    )
    assert research_control_registry_state(
        duplicate,
        registry_path=registry_path,
    )["available"] is False

    wrong_package = _research_telemetry(
        [_research_row("iono", content_digest="sha256:" + "0" * 64)]
    )
    assert research_control_registry_state(
        wrong_package,
        registry_path=registry_path,
    )["available"] is False

    mixed = json.loads(json.dumps(valid))
    mixed_native = mixed["research_controls"]
    mixed_native["checkpoint_mixed"] = True
    mixed_native["checkpoint_digests"] = {
        "sha256:" + "8" * 64: 1,
        "sha256:" + "9" * 64: 1,
    }
    assert research_control_registry_state(
        mixed,
        registry_path=registry_path,
    )["available"] is False

    broken_aggregate = json.loads(json.dumps(valid))
    broken_aggregate["research_controls"]["wins"] = 0.0
    assert research_control_registry_state(
        broken_aggregate,
        registry_path=registry_path,
    )["available"] is False


def test_public_mix_iteration_offset_includes_nested_research_and_is_idempotent(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "ops/research_control_registry_v1.json"
    registry = json.loads(registry_path.read_text())
    iono_digest = registry["controls"][0]["content_digest"]
    telemetry = {
        "iteration": 4,
        **_research_telemetry(
            [_research_row("iono", content_digest=iono_digest)],
            iteration=3,
        ),
    }

    shifted = _offset_public_mix_iterations(telemetry, 11)
    shifted_again = _offset_public_mix_iterations(shifted, 11)
    state = research_control_registry_state(
        shifted,
        registry_path=registry_path,
    )

    assert shifted["lineage_iteration"] == 4
    assert shifted["iteration"] == 15
    assert shifted["research_controls"]["lineage_iteration"] == 3
    assert shifted["research_controls"]["iteration"] == 14
    assert shifted_again == shifted
    assert state["iteration"] == 14


def test_competition_gate_program_accepts_runner_fixed_seed_manifest(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "ops/alakazam_gate_program_v1.json").read_text())
    result_path = tmp_path / "strong_gate_result.json"
    contract["next_gate"]["exact_result_pointer"] = str(result_path)
    contract_path = tmp_path / "gate_program.json"
    _write_json(contract_path, contract)

    roster = contract["next_gate"]["roster"]
    games_per_opponent = contract["next_gate"]["evaluation"]["games_per_opponent"]
    total_games = contract["next_gate"]["evaluation"]["games_total"]
    checks = {
        "audit": True,
        "skill_weighted_win_rate": False,
        "skill_weighted_confidence_lower": False,
        "s_tier_mean_floor": False,
        "individual_opponent_floor": True,
        "accepted_official_holdout_non_regression": True,
        "research_control_audit": True,
    }
    _write_json(
        result_path,
        {
            "schema": "poke_bot.public_agent_gate_result/v1",
            "gate_id": contract["next_gate"]["id"],
            "checkpoint_digest": "sha256:candidate",
            "games": total_games,
            "passed": False,
            "skill_weighted_wr": 0.425,
            "confidence_lower": 0.406,
            "checks": checks,
            "matchups": [
                {
                    "opponent_id": row["opponent_id"],
                    "games": games_per_opponent,
                    "seat0": games_per_opponent // 2,
                    "seat1": games_per_opponent // 2,
                    "wr": 0.425,
                }
                for row in roster
            ],
            "audit": {
                "passed": True,
                "checkpoint_digest": "sha256:candidate",
                "exact_distribution": True,
                "both_seats": True,
                "greedy": True,
                "fixed_seed_manifest": {
                    "gate_games": total_games,
                    "mapping": "seed_base + job_index",
                },
                "fixed_seed_manifest_digest": "sha256:seed-manifest",
            },
        },
    )

    candidate_heldout = {"checkpoint_digest": "sha256:candidate"}
    state = competition_gate_program_state(
        candidate_heldout,
        {},
        contract_path=contract_path,
        registry_path=tmp_path / "missing-registry.json",
    )

    assert state["next_gate"]["exact_result_available"] is True
    assert state["next_gate"]["status"] == "failed"
    assert state["next_gate"]["exact_result"]["games"] == total_games

    active_only_pass = json.loads(result_path.read_text())
    active_only_pass["passed"] = True
    active_only_pass["checks"] = {
        "audit": True,
        "skill_weighted_win_rate": True,
        "skill_weighted_confidence_lower": True,
        "s_tier_mean_floor": True,
        "individual_opponent_floor": True,
        "s_plus_matchup_floor_allowance": True,
    }
    active_only_pass["research_checks"] = {
        "research_control_audit": False,
        "accepted_official_holdout_non_regression": False,
    }
    _write_json(result_path, active_only_pass)
    active_only = competition_gate_program_state(
        candidate_heldout,
        {},
        contract_path=contract_path,
        registry_path=tmp_path / "missing-registry.json",
    )
    assert active_only["next_gate"]["status"] == "passed"

    malformed = json.loads(result_path.read_text())
    malformed["matchups"][0]["games"] = games_per_opponent - 1
    _write_json(result_path, malformed)
    rejected = competition_gate_program_state(
        candidate_heldout,
        {},
        contract_path=contract_path,
        registry_path=tmp_path / "missing-registry.json",
    )
    assert rejected["next_gate"]["exact_result_available"] is False

    stale = json.loads(result_path.read_text())
    stale["matchups"][0]["games"] = games_per_opponent
    stale["checkpoint_digest"] = "sha256:prior-candidate"
    stale["audit"]["checkpoint_digest"] = "sha256:prior-candidate"
    _write_json(result_path, stale)
    stale_state = competition_gate_program_state(
        candidate_heldout,
        {},
        contract_path=contract_path,
        registry_path=tmp_path / "missing-registry.json",
    )
    assert stale_state["next_gate"]["exact_result_available"] is True
    assert stale_state["next_gate"]["latest_exact_attempt_available"] is True
    assert stale_state["next_gate"]["latest_exact_attempt_current"] is True
    assert (
        stale_state["next_gate"]["latest_exact_attempt"]["checkpoint_digest"]
        == "sha256:prior-candidate"
    )
    assert stale_state["next_gate"]["status"] == "passed"


def test_dashboard_separates_accepted_holdout_next_gate_and_sampled_progress() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "effective_pass_criteria" in html
    assert "LC50 fallback active from iteration 5" in html

    assert 'data-widget-toggle="nextgate"' in html
    assert 'data-card="nextgate"' in html
    assert "nextgate-exact" in html
    assert "nextgate-diagnostic" in html
    assert "nextgate-roster" in html
    assert "nextgate-research" in html
    assert "nextgate-research-roster" in html
    assert "nextgate-practice" in html
    assert "nextgate-practice-roster" in html
    assert "Exact gate allocation" in html
    assert "c.strong_public_practice||{}" in html
    assert "Current iteration strong-public practice" in html
    assert "LIVE COMMIT iter " in html
    assert "live commit overlay unavailable" in html
    assert "practiceLive=publicMixTelemetry.strong_public_practice||{}" in html
    assert "practiceLiveCurrent=!!" in html
    assert "current outcomes waiting; last iter" in html
    assert "LIVE sampled WR" in html
    assert "sampled training only · NON-GATE" in html
    assert "no prior iteration substituted" in html
    assert "TRAINING ONLY · formal evaluation off" in html
    assert "researchArchetypeById" in html
    assert "Archived original-four selection history" in html
    assert "toLocaleString()+' games'" in html
    assert "latestExact=nextGate.latest_exact_attempt||{}" in html
    assert "latestAttemptAvailable=!!(nextGate.latest_exact_attempt_available" in html
    assert "LATEST AUDITED · " in html
    assert "prior candidate; current gate remains queued" in html
    assert "exact active-gate games" in html
    assert "'NOT RUN'" in html
    assert "(nextDiag.rows||[])" not in html
    assert "acceptedGate=gateProgram.accepted_gate||{}" in html
    assert "acceptedExact=acceptedGate.exact_holdout||{}" in html
    assert "acceptedEvidenceAvailable=!!(accepted&&" in html
    assert "$('baseline-protected').textContent=acceptedEvidenceAvailable" in html
    assert "$('nextgate-anchor').textContent=acceptedEvidenceAvailable" in html
    assert "exact accepted-gate games · immutable contract + registry" in html
    assert "owner decision ACCEPTED · protected registry identity reconciled" in html
    assert "Last completed strong-baseline holdout" in html
    assert "Holdout games audited" in html
    assert "Current / last exact research controls" in html
    assert "Per-iteration research controls · additive greedy diagnostics" in html
    assert "researchStatePresent=Object.keys(researchState).length>0" in html
    assert "researchStatePresent?[]:originalResearchRows" in html
    assert "nativeResearchTelemetry=publicMixTelemetry.research_controls||{}" in html
    assert "!researchStatePresent&&nativeResearchTelemetry.available" in html
    assert "researchUnavailableReason=researchStatePresent&&!researchState.available" in html
    assert "legacyResearchRows=(publicMixTelemetry.matchups||[])" in html
    assert "researchResultById=new Map" in html
    assert "additive greedy diagnostics · no replay · non-gate" in html
    assert "per-iteration research controls: " in html
    assert "exact greedy diagnostic games" in html
    assert "greedy · both seats · fixed seeds" in html
    assert "sampled temperature policy · NON-GATE" not in html
    assert "live public-mix win rate" in html
    assert "fixed disjoint seeds · checkpoint digest pinned" in html
    assert "all matchups complete · no early stop" in html
    assert "excluded from every gate calculation" in html
    assert "alias duplicates excluded" in html
    assert "UNAUTHORIZED RETRY INCIDENT" in html
    assert "active gate held-out win rate" in html
    assert "activeGateAllocation=activeGateTotal.toLocaleString()+' total · '" in html
    assert "+' opponents × '+activeGatePerOpponent+' each · '" in html
    assert "activeGateSeat0=Number(nextEval.seat0_games_per_opponent||0)" in html
    assert "activeGateSeat1=Number(nextEval.seat1_games_per_opponent||0)" in html
    assert "candidate-first + '+activeGateSeat1+' candidate-second" in html
    assert "${row.seat0??activeGateSeat0} candidate-first + ${row.seat1??activeGateSeat1} candidate-second" in html
    assert "exact greedy gate games" in html
    assert "no early stop" in html
    assert "original-four research excluded" in html
    assert "requires ops/alakazam_gate_program_v1.json" in html
    assert "Number.isFinite(c.heldout_wr)" not in html
    assert "+' gate + '+researchGames" not in html
    assert "Dashboard UI v15" in html


def test_live_curriculum_wins_over_stale_bootstrap_in_hero_status() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "dashboard/lan/index.html").read_text(encoding="utf-8")
    stage_assignment = next(
        line for line in html.splitlines() if "$('stage').textContent=" in line
    )
    assert stage_assignment.index("curriculumActive?") < stage_assignment.index(
        "svc.active?"
    )


def test_dashboard_http_server_has_refresh_tab_headroom() -> None:
    assert DashboardHTTPServer.allow_reuse_address is True
    assert DashboardHTTPServer.daemon_threads is True
    assert DashboardHTTPServer.request_queue_size >= 64


def _fleet_payload(*, active: bool, elmo_jobs: int, bert_jobs: int) -> dict:
    return {
        "observed_at": time.time(),
        "curriculum": {
            "active": active,
            "run": "test-run",
            "stage": "collect:self_play",
            "progress": {
                "stage": "collect:self_play",
                "iteration": 11,
                "percent": 52.0,
                "current": 3604,
                "total": 6963,
            },
            "remote_dispatch": {
                "elmo_request_sockets": 96,
                "bert_request_sockets": 32,
            },
        },
        "fleet": {
            "inzi": {"worker": {"workers": 96, "leaf_servers": 24}},
            "elmo": {
                "reachable": True,
                "worker": {
                    "active": True,
                    "workers": 48,
                    "active_jobs": elmo_jobs,
                    "jobs_completed": 100,
                    "decisions_completed": 1000,
                    "command": "elmo-worker",
                },
            },
            "bert": {
                "reachable": True,
                "production_active": True,
                "worker": {
                    "active": True,
                    "workers": 16,
                    "active_jobs": bert_jobs,
                    "jobs_completed": 50,
                    "decisions_completed": 500,
                    "command": "bert-worker",
                },
            },
        },
        "recent_events": ["[remote] scheduled_dispatch kind=self_play"],
    }


def test_worker_cards_distinguish_execution_from_queued_requests() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=94, bert_jobs=27)
    SnapshotCache()._annotate_fleet_rates(payload)

    elmo = payload["fleet"]["elmo"]["worker"]
    bert = payload["fleet"]["bert"]["worker"]
    assert elmo["feed_coverage"] == 1.0
    assert elmo["queued_jobs"] == 46
    assert elmo["allocation_state"] == "WORKING · 48/48 workers fed · 46 queued"
    assert bert["feed_coverage"] == 1.0
    assert bert["queued_jobs"] == 11
    assert bert["allocation_state"] == "WORKING · 16/16 workers fed · 11 queued"


def test_public_mix_with_remote_sockets_is_not_labeled_local_only() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=96, bert_jobs=32)
    payload["curriculum"]["stage"] = "collect:public_mix"
    payload["curriculum"]["progress"].update(
        stage="collect:public_mix",
        remotes=64,
    )
    SnapshotCache()._annotate_fleet_rates(payload)

    assert payload["fleet"]["elmo"]["worker"]["allocation_state"] == (
        "WORKING · 48/48 workers fed · 48 queued"
    )
    assert payload["fleet"]["bert"]["worker"]["allocation_state"] == (
        "WORKING · 16/16 workers fed · 16 queued"
    )


def test_namespaced_heldout_with_remote_work_is_labeled_active() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=38, bert_jobs=64)
    payload["fleet"]["elmo"]["worker"]["workers"] = 36
    payload["curriculum"]["stage"] = "heldout:strong_public_gate"
    payload["curriculum"]["progress"].update(
        stage="heldout:strong_public_gate",
        remotes=52,
    )
    SnapshotCache()._annotate_fleet_rates(payload)

    assert payload["fleet"]["elmo"]["worker"]["allocation_state"] == (
        "WORKING · 36/36 workers fed · 2 queued"
    )
    assert payload["fleet"]["bert"]["worker"]["allocation_state"] == (
        "WORKING · 16/16 workers fed · 48 queued"
    )


def test_real_refill_shortfall_is_visible() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=4, bert_jobs=0)
    SnapshotCache()._annotate_fleet_rates(payload)

    assert payload["fleet"]["elmo"]["worker"]["allocation_state"] == (
        "REFILLING · 4/48 workers fed · queue empty"
    )
    assert payload["fleet"]["bert"]["worker"]["allocation_state"] == (
        "STARVED · 0/16 workers fed · refill pending"
    )


def test_zero_queue_without_log_hint_is_still_starved_mid_collection() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=0, bert_jobs=0)
    payload["recent_events"] = []
    SnapshotCache()._annotate_fleet_rates(payload)

    assert payload["fleet"]["elmo"]["worker"]["allocation_state"] == (
        "STARVED · 0/48 workers fed · refill pending"
    )
    assert payload["fleet"]["bert"]["worker"]["allocation_state"] == (
        "STARVED · 0/16 workers fed · refill pending"
    )


def test_scheduler_rates_and_ingest_backlog_replace_false_per_host_zero() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=0, bert_jobs=0)
    payload["curriculum"]["source_current"] = True
    payload["curriculum"]["progress"].update(gps=2.90, sps=467.5)
    payload["recent_events"] = [
        "[pure_rl] mid_iter_rebalance=scheduler=mid_iter "
        "wave_gps=9.98 local_gps=3.33 remote_gps=6.65 "
        "wave_sps=1342.4 local_sps=270.7 remote_sps=1071.7 "
        "result_buffer={'memory_items': 320, 'spool_files': 4050}"
    ]
    SnapshotCache()._annotate_fleet_rates(payload)

    assert payload["fleet"]["inzi"]["worker"]["gps"] == 3.33
    assert payload["fleet"]["inzi"]["worker"]["sps"] == 270.7
    assert payload["fleet_rates"]["total_gps"] == 9.98
    assert payload["fleet_rates"]["total_sps"] == 1342.4
    assert payload["fleet_rates"]["display_sps"] == 467.5
    assert payload["fleet_rates"]["ingest_gps"] == 2.90
    assert payload["fleet_rates"]["buffered_results"] == 4370
    assert payload["fleet"]["bert"]["worker"]["allocation_state"] == (
        "RESULTS BUFFERED · 4370 fleet games awaiting ingest"
    )


def test_collection_sps_survives_intermittent_zero_for_same_phase() -> None:
    cache = SnapshotCache()
    cache.last_valid_rates = {}
    first = _fleet_payload(active=True, elmo_jobs=48, bert_jobs=16)
    first["observed_at"] = 100.0
    first["curriculum"]["source_current"] = True
    first["curriculum"]["progress"].update(gps=10.0, sps=900.0)
    first["recent_events"] = []
    cache._annotate_fleet_rates(first)

    second = _fleet_payload(active=True, elmo_jobs=48, bert_jobs=16)
    second["observed_at"] = 102.0
    second["curriculum"]["source_current"] = True
    second["curriculum"]["progress"].update(gps=11.0, sps=0.0)
    second["recent_events"] = []
    cache._annotate_fleet_rates(second)

    assert second["fleet_rates"]["display_sps"] == 900.0
    assert second["fleet_rates"]["display_sps_held"] is True
    assert second["fleet_rates"]["display_sps_age_s"] == 2.0
    assert "same-phase" in second["fleet_rates"]["display_sps_source"]


def test_holdout_zero_sps_is_replaced_by_labeled_committed_density_estimate() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=96, bert_jobs=32)
    payload["curriculum"]["stage"] = "heldout"
    payload["curriculum"]["progress"].update(
        stage="heldout",
        gps=12.0,
        sps=0.0,
    )
    payload["curriculum"]["iteration_timing"] = {
        "latest_gps": 10.0,
        "latest_sps": 700.0,
    }

    SnapshotCache()._annotate_fleet_rates(payload)

    assert payload["fleet_rates"]["display_sps"] == 840.0
    assert payload["fleet_rates"]["display_sps_estimated"] is True
    assert payload["fleet_rates"]["decisions_per_game"] == 70.0
    assert "estimated from live GPS" in payload["fleet_rates"]["display_sps_source"]


def test_collection_sps_never_leaks_into_next_iteration() -> None:
    cache = SnapshotCache()
    cache.last_valid_rates = {}
    first = _fleet_payload(active=True, elmo_jobs=48, bert_jobs=16)
    first["observed_at"] = 100.0
    first["curriculum"]["source_current"] = True
    first["curriculum"]["progress"].update(iteration=5, gps=10.0, sps=900.0)
    first["recent_events"] = []
    cache._annotate_fleet_rates(first)

    next_iteration = _fleet_payload(active=True, elmo_jobs=48, bert_jobs=16)
    next_iteration["observed_at"] = 102.0
    next_iteration["curriculum"]["source_current"] = True
    next_iteration["curriculum"]["progress"].update(
        iteration=6, gps=1.0, sps=0.0
    )
    next_iteration["recent_events"] = []
    cache._annotate_fleet_rates(next_iteration)

    assert next_iteration["fleet_rates"]["display_sps"] == 0.0
    assert next_iteration["fleet_rates"]["display_sps_held"] is False


def test_stopped_trainer_never_reports_remote_starvation() -> None:
    payload = _fleet_payload(active=False, elmo_jobs=0, bert_jobs=0)
    SnapshotCache()._annotate_fleet_rates(payload)

    assert payload["fleet"]["elmo"]["worker"]["allocation_state"] == (
        "READY · next self-play allocation"
    )
    assert payload["fleet"]["bert"]["worker"]["allocation_state"] == (
        "READY · next self-play allocation"
    )


def test_stale_scheduler_startup_zero_does_not_hide_live_inzi_gps() -> None:
    cache = SnapshotCache()
    first = _fleet_payload(active=True, elmo_jobs=48, bert_jobs=16)
    first["observed_at"] = 100.0
    first["curriculum"]["source_current"] = True
    first["curriculum"]["progress"]["current"] = 100
    first["recent_events"] = [
        "[pure_rl] mid_iter_rebalance=start local_gps=0.00 local_sps=0.0 "
        "remote_gps=0.00"
    ]
    cache._annotate_fleet_rates(first)

    second = _fleet_payload(active=True, elmo_jobs=48, bert_jobs=16)
    second["observed_at"] = 102.0
    second["curriculum"]["source_current"] = True
    second["curriculum"]["progress"]["current"] = 120
    second["fleet"]["elmo"]["worker"]["jobs_completed"] = 104
    second["fleet"]["bert"]["worker"]["jobs_completed"] = 52
    second["fleet"]["elmo"]["worker"]["decisions_completed"] = 1400
    second["fleet"]["bert"]["worker"]["decisions_completed"] = 700
    second["recent_events"] = [
        "[pure_rl] mid_iter_rebalance=start local_gps=0.00 local_sps=0.0 "
        "remote_gps=0.00"
    ]
    cache._annotate_fleet_rates(second)

    assert second["fleet"]["inzi"]["worker"]["gps"] == 7.0
    assert second["fleet"]["inzi"]["worker"]["rate_source"].startswith(
        "collector total minus remote counters"
    )
    assert second["fleet"]["inzi"]["worker"]["sps"] == 700.0
    assert second["fleet"]["inzi"]["worker"]["sps_estimated"] is True


def test_bert_launchd_has_descriptor_budget_for_four_x_socket_queue() -> None:
    root = Path(__file__).resolve().parents[1]
    plist_path = root / ".staging/com.pokebot.remote-worker-8766.plist"
    if not plist_path.is_file():
        pytest.skip("host-only Bert LaunchAgent staging artifact is unavailable")
    with plist_path.open("rb") as fh:
        plist = plistlib.load(fh)
    assert plist["EnvironmentVariables"]["POKEBOT_REMOTE_MAX_CONNECTIONS"] == "150"
    assert plist["SoftResourceLimits"]["NumberOfFiles"] >= 1024
    assert plist["HardResourceLimits"]["NumberOfFiles"] >= 1024
    supervisor = (
        root / "scripts/run_bert_remote_worker_supervised.sh"
    ).read_text(encoding="utf-8")
    assert 'max_connections="${POKEBOT_REMOTE_MAX_CONNECTIONS:-150}"' in supervisor
    assert "max_connections=68" not in supervisor


def test_trainer_service_has_descriptor_budget_for_four_x_remote_queues() -> None:
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/systemd/pokebot-pure-rl-continuous-rehearsal.service").read_text()
    match = re.search(r"^LimitNOFILE=(\d+)$", unit, re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) >= 4096


def test_trainer_service_uses_blackwell_batches_with_oom_headroom() -> None:
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/systemd/pokebot-pure-rl-continuous-rehearsal.service").read_text()
    assert "--train-games-per-batch 240" in unit
    # 20,480 repeatedly left only 10-23 MiB free and ended in CUDA illegal
    # accesses. 12,288 retains large resident batches with ~40% activation
    # headroom instead of relying on recovery after a poisoned CUDA context.
    assert "--train-max-decisions-per-batch 12288" in unit
    assert "--train-max-decisions-per-batch 20480" not in unit


def test_resumable_trainer_service_pins_the_audited_lineage_handoff() -> None:
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/systemd/pokebot-pure-rl-continuous-rehearsal.service").read_text()
    assert "--resume auto" in unit
    checkpoint = (
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "pure_rl_core_continuous_rehearsal_v6_20260719/"
        "checkpoints/iter_00026.pt"
    )
    assert unit.count(f"--base-checkpoint {checkpoint}") == 1
    assert unit.count(f"--initial-learner-checkpoint {checkpoint}") == 1
    assert "--run-name pure_rl_core_baseline50_v7_20260720" in unit


def test_measurement_games_use_the_requested_four_deck_pool() -> None:
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/systemd/pokebot-pure-rl-continuous-rehearsal.service").read_text()
    assert (
        "Environment=PURE_RL_MEASUREMENT_DECKS="
        "lucario,alakazam,starmie,crustle"
    ) in unit


def test_trainer_prefills_remote_queue_while_results_are_returning() -> None:
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/systemd/pokebot-pure-rl-continuous-rehearsal.service").read_text()
    assert "Environment=POKEBOT_REMOTE_SOCKET_PREFETCH=5" in unit
    assert "Environment=POKEBOT_REMOTE_SOCKET_PREFETCH_MAX=8" in unit
    assert "Environment=POKEBOT_REMOTE_REFILL_GAMES=1" in unit
    assert "Environment=POKEBOT_REMOTE_QUEUE_LOW_WATER_FRAC=0.5" in unit
    assert "Environment=POKEBOT_REMOTE_QUEUE_PROBE_S=0.2" in unit
    assert "Environment=POKEBOT_REMOTE_RESULT_SPOOL_MAX_GB=16" in unit
    assert (
        'Environment="POKEBOT_REMOTE_ENDPOINT_CHUNKS='
        '192.168.1.143:8765=192,bert.local:8766=64"'
    ) in unit


def test_live_model_footer_contract_includes_every_parameterized_head() -> None:
    manifest = {
        "run_name": "current-run",
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 0},
                "trainable_parameters": 1_489_505,
                "max_decisions_per_batch": 32_768,
                "warmup_max_decisions_per_batch": 8_192,
                "warmup_iterations": 30,
                "initial_checkpoint": {"path": "/tmp/alakazam.pt", "digest": "sha256:abc"},
                "archetype_aux_loss_weight": 0.05,
                "opp_hand_loss_weight": 0.05,
                "opp_remainder_loss_weight": 0.05,
                "lethal_threat_loss_weight": 0.025,
                "prize_race_loss_weight": 0.025,
                "alakazam_guide_targets_enabled": True,
                "alakazam_guide_loss_weight": 0.05,
            }
        }
    }
    model = learner_model_state(manifest, iteration=4)
    assert model["trainable_parameters"] == 1_489_505
    assert model["parameter_source"] == "manifest.design_contract.learner"
    assert model["profile"]["temporal_layers"] == 0
    assert model["run"] == "current-run"
    assert model["training_schedule"]["phase"] == "head_focus"
    assert model["training_schedule"]["active_max_decisions_per_batch"] == 8_192
    assert model["optimizer"]["curriculum"]["name"] == "AdamW"
    assert model["optimizer"]["curriculum"]["learning_rate"] == 3e-4
    assert model["optimizer"]["curriculum"]["awr_beta"] == 0.5
    assert model["optimizer"]["expert_rehearsal"]["name"] == "AdamW"
    assert set(model["heads"]) == {
        "policy", "value", "archetype", "opponent_hand",
        "opponent_remainder", "lethal_threat", "prize_race",
    }
    assert all(row["enabled"] for row in model["heads"].values())
    assert model["training_targets"]["alakazam_guide"]["parameterized_head"] is False
    assert model["training_targets"]["current_deck_guide"]["parameterized_head"] is False
    assert model["seed_checkpoint"] == "/tmp/alakazam.pt"


def test_live_model_footer_prefers_checkpoint_loaded_parameter_count(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "[pure_rl] loaded checkpoint params=1555000 path=/tmp/old.pt\n"
        "[pure_rl] loaded checkpoint params=1601345 path=/tmp/current.pt\n",
        encoding="utf-8",
    )
    runtime = checkpoint_parameter_telemetry(log)
    manifest = {
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 1},
                "trainable_parameters": 1_923_425,
            }
        }
    }

    model = learner_model_state(
        manifest,
        runtime_parameter_contract=runtime,
    )

    assert model["trainable_parameters"] == 1_601_345
    assert model["parameter_source"] == "runtime checkpoint load"
    assert model["parameter_evidence_checkpoint"] == "/tmp/current.pt"
    assert model["parameter_evidence_source"] == str(log)


def test_live_model_footer_prefers_active_awr_environment_over_old_metrics() -> None:
    manifest = {
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 1},
                "awr_beta": 0.5,
                "awr_weight_max": 20.0,
            }
        }
    }

    model = learner_model_state(
        manifest,
        iteration=4,
        runtime_optimizer={"awr_beta": 0.75, "awr_weight_max": 10.0},
    )

    optimizer = model["optimizer"]
    assert optimizer["curriculum"]["awr_beta"] == 0.75
    assert optimizer["curriculum"]["awr_weight_max"] == 10.0
    assert optimizer["source"] == (
        "live systemd environment + immutable manifest contract"
    )


def test_live_model_footer_contract_separates_staged_dormant_parameters(
    tmp_path: Path,
) -> None:
    dormant_contract = tmp_path / "dormant.json"
    _write_json(
        dormant_contract,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [
                {
                    "id": "matchup_adapter_bank_v1",
                    "status": "staged_non_active",
                    "present_in_active_checkpoint": False,
                    "runtime_enabled": False,
                    "optimizer_active": False,
                    "parameter_count": 11_480,
                    "expert_count": 7,
                    "hidden_dim": 96,
                    "bottleneck_dim": 8,
                    "architecture": "7 × 96→8→96 residual MLP",
                }
            ],
        },
    )
    manifest = {
        "run_name": "current-run",
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 1},
                "trainable_parameters": 1_923_425,
            }
        },
    }

    model = learner_model_state(
        manifest,
        iteration=4,
        dormant_modules_path=dormant_contract,
    )

    assert model["parameter_breakdown"] == {
        "current_checkpoint_total": 1_923_425,
        "optimizer_active_current": 1_923_425,
        "current_non_active": 0,
        "staged_non_active": 11_480,
        "staged_architecture_total": 1_934_905,
        "staged_modules": 1,
        "deployed_dormant_modules": 0,
        "source": str(dormant_contract),
    }
    assert model["dormant_modules"][0]["runtime_enabled"] is False


def test_live_model_footer_counts_deployed_zero_output_adapters_as_non_active(
    tmp_path: Path,
) -> None:
    dormant_contract = tmp_path / "dormant.json"
    _write_json(
        dormant_contract,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [
                {
                    "id": "matchup_adapter_bank_v1",
                    "status": "deployed_dormant",
                    "present_in_active_checkpoint": True,
                    "runtime_enabled": False,
                    "optimizer_active": False,
                    "zero_output": True,
                    "parameter_count": 11_480,
                    "expert_count": 7,
                    "hidden_dim": 96,
                    "bottleneck_dim": 8,
                }
            ],
        },
    )
    manifest = {
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 1},
                "trainable_parameters": 1_612_825,
            }
        }
    }

    model = learner_model_state(
        manifest,
        dormant_modules_path=dormant_contract,
    )

    assert model["parameter_breakdown"] == {
        "current_checkpoint_total": 1_612_825,
        "optimizer_active_current": 1_601_345,
        "current_non_active": 11_480,
        "staged_non_active": 0,
        "staged_architecture_total": 1_612_825,
        "staged_modules": 0,
        "deployed_dormant_modules": 1,
        "source": str(dormant_contract),
    }
    assert model["dormant_modules"][0]["zero_output"] is True


def test_live_model_footer_exposes_validated_22_route_stage_receipt(
    tmp_path: Path,
) -> None:
    staged_roster = tmp_path / "matchup_adapter_roster_v4.json"
    expert_ids = [f"route-{index:02d}" for index in range(22)]
    _write_json(
        staged_roster,
        {
            "schema": "poke_bot.matchup_adapter_roster_stage/v1",
            "status": "tested_staged_not_active",
            "runtime_enabled": False,
            "mutually_exclusive_route_per_decision": True,
            "unknown_route_exact_bypass": True,
            "expert_count": 22,
            "expert_ids": expert_ids,
            "parameter_count": 36_080,
            "validation": {"tests_passed": 65},
        },
    )

    model = learner_model_state(
        {"design_contract": {"learner": {"profile": {}}}},
        staged_adapter_roster_path=staged_roster,
    )

    assert model["matchup_adapter_roster_stage"]["expert_count"] == 22
    assert model["matchup_adapter_roster_stage"]["expert_ids"] == expert_ids
    assert model["matchup_adapter_roster_stage"]["runtime_enabled"] is False


def test_live_model_footer_accepts_checksum_pinned_specialist_runtime(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "iter_00005.pt"
    checkpoint.write_bytes(b"specialist-checkpoint")
    checkpoint_digest = (
        "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    expert_ids = [f"route-{index:02d}" for index in range(21)]
    expert_ids.append("hops-trevenant")
    accepted_ids = expert_ids[:5] + ["hops-trevenant"]
    tree = tmp_path / "tree.json"
    authorization = tmp_path / "authorization.json"
    registry = tmp_path / "registry.json"
    staged = tmp_path / "staged.json"
    _write_json(
        tree,
        {
            "schema": "poke_bot.public_matchup_decision_tree/v1",
            "runtime_enabled": True,
            "targets": expert_ids,
            "runtime_contract": {
                "accepted_archetype_ids": accepted_ids,
                "checkpoint": str(checkpoint),
                "checkpoint_digest": checkpoint_digest,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    _write_json(
        authorization,
        {
            "schema": "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1",
            "first_eligible_iteration": 0,
            "optimizer_scope": "matchup_adapter_bank_only",
            "parent_checkpoint": str(checkpoint),
            "parent_checkpoint_digest": checkpoint_digest,
            "runtime_enabled": False,
        },
    )
    _write_json(
        registry,
        {
            "schema": "poke_bot.specialist_runtime_registry/v1",
            "specialists": {
                "hops-trevenant": {
                    "status": "ready",
                    "run_name": "trevenant-live",
                    "matchup_runtime_tree": str(tree),
                    "matchup_runtime_tree_sha256": (
                        "sha256:" + hashlib.sha256(tree.read_bytes()).hexdigest()
                    ),
                    "matchup_adapter_authorization": str(authorization),
                    "matchup_adapter_authorization_sha256": (
                        "sha256:"
                        + hashlib.sha256(authorization.read_bytes()).hexdigest()
                    ),
                    "matchup_adapter_epochs_per_rl_iteration": 1,
                }
            },
        },
    )
    _write_json(
        staged,
        {
            "schema": "poke_bot.matchup_adapter_roster_stage/v1",
            "status": "tested_staged_not_active",
            "runtime_enabled": False,
            "mutually_exclusive_route_per_decision": True,
            "unknown_route_exact_bypass": True,
            "expert_count": 22,
            "expert_ids": expert_ids,
            "parameter_count": 36_080,
            "validation": {"tests_passed": 147},
        },
    )

    model = learner_model_state(
        {
            "run_name": "trevenant-live",
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                    "trainable_parameters": 1_637_425,
                    "dormant_matchup_adapter": {"epochs": 1},
                }
            },
        },
        {"learner": {"path": str(checkpoint), "digest": checkpoint_digest}},
        staged_adapter_roster_path=staged,
        specialist_runtime_registry_path=registry,
        matchup_runtime_ready_path=tmp_path / "old-ready.json",
        matchup_runtime_boundary_path=tmp_path / "old-boundary.json",
    )

    runtime = model["matchup_adapter_runtime"]
    assert runtime["enabled"] is True
    assert runtime["checkpoint"] == str(checkpoint)
    assert runtime["expert_count"] == 22
    assert runtime["accepted_archetype_ids"] == accepted_ids
    adapter = model["dormant_modules"][0]
    assert adapter["runtime_enabled"] is True
    assert adapter["expert_ids"] == expert_ids
    assert adapter["isolated_adapter_updates_enabled"] is True


def test_live_model_footer_uses_committed_checkpoint_over_stale_ten_route_marker(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "iter_00001.pt"
    checkpoint.write_bytes(b"immutable-current-checkpoint")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expert_ids = [f"route-{index:02d}" for index in range(22)]
    stale = tmp_path / "stale-dormant.json"
    _write_json(
        stale,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [
                {
                    "id": "old-ten-route-marker",
                    "present_in_active_checkpoint": True,
                    "runtime_enabled": False,
                    "parameter_count": 16_400,
                    "expert_count": 10,
                }
            ],
        },
    )
    loop = {
        "learner": {"path": str(checkpoint), "digest": digest},
        "dormant_matchup_adapter_fit": {
            "trained_archetype_ids": expert_ids[:7],
            "route_decisions": {
                route: (100 if route in expert_ids[:7] else 0)
                for route in expert_ids
            },
        },
    }
    runtime_collection = {
        "available": True,
        "checkpoint_digest": digest,
        "iteration": 1,
        "combined": {
            "games": 8192,
            "audited_games": 8192,
            "all_games_audited": True,
            "all_runtime_enabled": True,
            "contract_clean": True,
            "accepted_roster_counts": {"|".join(expert_ids): 8192},
        },
        "enforcement": {"required": True, "passed": True},
    }
    structure = {
        "verified": True,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "model_parameters": 1_637_910,
        "state_tensor_elements": 1_959_991,
        "adapter_parameters": 36_080,
        "adapter_expert_count": 22,
        "adapter_expert_ids": expert_ids,
        "source": "test committed checkpoint",
    }

    model = learner_model_state(
        {
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                    "dormant_matchup_adapter": {"epochs": 1},
                }
            }
        },
        loop,
        runtime_collection=runtime_collection,
        checkpoint_structure=structure,
        dormant_modules_path=stale,
        staged_adapter_roster_path=tmp_path / "absent-roster.json",
        matchup_runtime_ready_path=tmp_path / "absent-ready.json",
        matchup_runtime_boundary_path=tmp_path / "absent-boundary.json",
        specialist_runtime_registry_path=tmp_path / "absent-registry.json",
    )

    assert model["trainable_parameters"] == 1_637_910
    assert model["parameter_breakdown"]["current_non_active"] == 36_080
    assert model["parameter_breakdown"]["optimizer_active_current"] == 1_601_830
    assert model["checkpoint_structure"]["state_tensor_elements"] == 1_959_991
    assert model["matchup_adapter_runtime"]["enabled"] is True
    assert model["matchup_adapter_runtime"]["expert_ids"] == expert_ids
    assert model["dormant_modules"][0]["expert_count"] == 22
    assert model["dormant_modules"][0]["expert_ids"] == expert_ids


def test_live_model_footer_accepts_checksum_pinned_descendant_of_clean_collection(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "iter_00001.pt"
    checkpoint.write_bytes(b"descendant-checkpoint")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prior_digest = "sha256:" + "a" * 64
    expert_ids = [f"route-{index:02d}" for index in range(22)]
    trained_ids = expert_ids[:7]
    loop = {
        "learner": {"path": str(checkpoint), "digest": digest},
        "dormant_matchup_adapter_fit": {
            "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
            "base_frozen": True,
            "optimizer_scope": "matchup_adapter_bank_only",
            "checkpoint_path": str(checkpoint),
            "checkpoint_digest": digest,
            "steps": 158,
            "rows": 1_341_398,
            "trained_archetype_ids": trained_ids,
            "route_decisions": {
                route: (100 if route in trained_ids else 0)
                for route in expert_ids
            },
        },
    }
    runtime_collection = {
        "available": True,
        "checkpoint_digest": prior_digest,
        "iteration": 1,
        "combined": {
            "games": 8192,
            "audited_games": 8192,
            "all_games_audited": True,
            "all_runtime_enabled": True,
            "contract_clean": True,
            "accepted_roster_counts": {"|".join(expert_ids): 8192},
        },
        "enforcement": {"required": True, "passed": True},
    }
    structure = {
        "verified": True,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "model_parameters": 1_637_910,
        "state_tensor_elements": 1_959_991,
        "adapter_parameters": 36_080,
        "adapter_expert_count": 22,
        "adapter_expert_ids": expert_ids,
        "source": "test committed checkpoint",
    }

    model = learner_model_state(
        {"design_contract": {"learner": {"profile": {}}}},
        loop,
        runtime_collection=runtime_collection,
        checkpoint_structure=structure,
        dormant_modules_path=tmp_path / "absent-dormant.json",
        staged_adapter_roster_path=tmp_path / "absent-roster.json",
        matchup_runtime_ready_path=tmp_path / "absent-ready.json",
        matchup_runtime_boundary_path=tmp_path / "absent-boundary.json",
        specialist_runtime_registry_path=tmp_path / "absent-registry.json",
    )

    runtime = model["matchup_adapter_runtime"]
    assert runtime["enabled"] is True
    assert runtime["expert_count"] == 22
    assert runtime["checkpoint_descendant_chain_verified"] is True
    assert runtime["live_collection_verified"] is False
    assert model["parameter_breakdown"]["current_non_active"] == 36_080


def test_live_model_footer_promotes_exact_v31_runtime_receipts_over_stale_shadow(
    tmp_path: Path,
) -> None:
    dormant_contract = tmp_path / "dormant.json"
    staged_roster = tmp_path / "roster.json"
    ready = tmp_path / "ready.json"
    boundary = tmp_path / "boundary.json"
    expert_ids = [f"route-{index:02d}" for index in range(22)]
    trained_ids = expert_ids[:15]
    accepted_ids = expert_ids[:14]
    parent_path = "/tmp/iter_00026.pt"
    active_path = "/tmp/iter_00026_matchup_v31.pt"
    digest = "sha256:" + "a" * 64
    _write_json(
        dormant_contract,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [
                {
                    "id": "matchup_adapter_bank_v2",
                    "status": "deployed_dormant",
                    "present_in_active_checkpoint": True,
                    "runtime_enabled": False,
                    "optimizer_active": False,
                    "zero_output": True,
                    "parameter_count": 16_400,
                    "expert_count": 10,
                    "hidden_dim": 96,
                    "bottleneck_dim": 8,
                }
            ],
        },
    )
    _write_json(
        staged_roster,
        {
            "schema": "poke_bot.matchup_adapter_roster_stage/v1",
            "status": "tested_staged_not_active",
            "runtime_enabled": False,
            "mutually_exclusive_route_per_decision": True,
            "unknown_route_exact_bypass": True,
            "expert_count": 22,
            "expert_ids": expert_ids,
            "parameter_count": 36_080,
            "validation": {"tests_passed": 147},
        },
    )
    _write_json(
        ready,
        {
            "schema": "poke_bot.matchup_runtime_production_ready/v1",
            "runtime_enabled": True,
            "iteration": 27,
            "artifacts": {
                "merged_checkpoint": {"path": active_path, "digest": digest}
            },
        },
    )
    _write_json(
        boundary,
        {
            "schema": "poke_bot.matchup_runtime_boundary_activation/v1",
            "activated_learner": {"path": active_path, "digest": digest},
            "parent_learner": {"path": parent_path},
            "boundary": {"next_iteration": 27},
            "adapter_fit": {
                "trained_archetype_ids": trained_ids,
                "route_decisions": {
                    route: 1000 if route in trained_ids else 0
                    for route in expert_ids
                },
            },
            "runtime_tree": {
                "accepted_archetype_ids": accepted_ids,
                "continuous_re_evaluation": True,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
                "digest": "sha256:" + "b" * 64,
            },
        },
    )

    model = learner_model_state(
        {
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                    "trainable_parameters": 1_637_425,
                    "dormant_matchup_adapter": {"epochs": 1},
                }
            }
        },
        {"learner": {"path": active_path, "digest": digest}},
        runtime_parameter_contract={
            "trainable_parameters": 1_637_425,
            "checkpoint": parent_path,
            "source": "/tmp/run.log",
        },
        dormant_modules_path=dormant_contract,
        staged_adapter_roster_path=staged_roster,
        matchup_runtime_ready_path=ready,
        matchup_runtime_boundary_path=boundary,
    )

    assert model["trainable_parameters"] == 1_637_425
    assert model["parameter_breakdown"]["optimizer_active_current"] == 1_601_345
    assert model["parameter_breakdown"]["current_non_active"] == 36_080
    assert model["matchup_adapter_roster_stage"] == {}
    assert model["matchup_adapter_runtime"]["enabled"] is True
    assert model["matchup_adapter_runtime"]["accepted_runtime_count"] == 14
    adapter = model["dormant_modules"][0]
    assert adapter["expert_count"] == 22
    assert adapter["runtime_enabled"] is True
    assert adapter["router_model_application_enabled"] is True
    assert adapter["isolated_adapter_updates_enabled"] is True
    assert len(adapter["zero_example_archetype_ids"]) == 7


def test_live_model_footer_keeps_v31_runtime_for_receipted_descendant(
    tmp_path: Path,
) -> None:
    dormant_contract = tmp_path / "dormant.json"
    staged_roster = tmp_path / "roster.json"
    ready = tmp_path / "ready.json"
    boundary = tmp_path / "boundary.json"
    expert_ids = [f"route-{index:02d}" for index in range(22)]
    trained_ids = expert_ids[:15]
    accepted_ids = expert_ids[:14]
    origin_path = "/tmp/iter_00026_matchup_v31.pt"
    origin_digest = "sha256:" + "a" * 64
    active_path = "/tmp/iter_00035.pt"
    active_digest = "sha256:" + "c" * 64
    _write_json(
        dormant_contract,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [
                {
                    "id": "matchup_adapter_bank_v2",
                    "status": "deployed_dormant",
                    "present_in_active_checkpoint": True,
                    "runtime_enabled": False,
                    "optimizer_active": False,
                    "zero_output": True,
                    "parameter_count": 16_400,
                    "expert_count": 10,
                    "hidden_dim": 96,
                    "bottleneck_dim": 8,
                }
            ],
        },
    )
    _write_json(
        staged_roster,
        {
            "schema": "poke_bot.matchup_adapter_roster_stage/v1",
            "status": "tested_staged_not_active",
            "runtime_enabled": False,
            "mutually_exclusive_route_per_decision": True,
            "unknown_route_exact_bypass": True,
            "expert_count": 22,
            "expert_ids": expert_ids,
            "parameter_count": 36_080,
            "validation": {"tests_passed": 147},
        },
    )
    _write_json(
        ready,
        {
            "schema": "poke_bot.matchup_runtime_production_ready/v1",
            "runtime_enabled": True,
            "iteration": 27,
            "artifacts": {
                "merged_checkpoint": {
                    "path": origin_path,
                    "digest": origin_digest,
                }
            },
        },
    )
    _write_json(
        boundary,
        {
            "schema": "poke_bot.matchup_runtime_boundary_activation/v1",
            "activated_learner": {
                "path": origin_path,
                "digest": origin_digest,
            },
            "parent_learner": {"path": "/tmp/iter_00026.pt"},
            "boundary": {"next_iteration": 27},
            "adapter_fit": {
                "trained_archetype_ids": trained_ids,
                "route_decisions": {
                    route: 1000 if route in trained_ids else 0
                    for route in expert_ids
                },
            },
            "runtime_tree": {
                "accepted_archetype_ids": accepted_ids,
                "continuous_re_evaluation": True,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
                "digest": "sha256:" + "b" * 64,
            },
        },
    )
    loop = {
        "learner": {"path": active_path, "digest": active_digest},
        "dormant_matchup_adapter_fit": {
            "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
            "runtime_enabled": False,
            "base_frozen": True,
            "optimizer_scope": "matchup_adapter_bank_only",
            "checkpoint_path": active_path,
            "checkpoint_digest": active_digest,
            "trained_archetype_ids": trained_ids,
            "route_decisions": {
                route: 2000 if route in trained_ids else 0
                for route in expert_ids
            },
            "epochs": 28,
            "steps": 5842,
            "rows": 6_879_140,
        },
    }

    model = learner_model_state(
        {
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                    "trainable_parameters": 1_637_425,
                    "dormant_matchup_adapter": {"epochs": 1},
                }
            }
        },
        loop,
        runtime_parameter_contract={
            "trainable_parameters": 1_637_425,
            "checkpoint": active_path,
            "source": "/tmp/run.log",
        },
        dormant_modules_path=dormant_contract,
        staged_adapter_roster_path=staged_roster,
        matchup_runtime_ready_path=ready,
        matchup_runtime_boundary_path=boundary,
    )

    assert model["matchup_adapter_runtime"]["enabled"] is True
    assert model["matchup_adapter_runtime"]["checkpoint"] == active_path
    assert model["dormant_modules"][0]["expert_count"] == 22
    assert model["dormant_modules"][0]["route_decisions"][trained_ids[0]] == 2000
    assert model["parameter_breakdown"]["current_non_active"] == 36_080
    assert model["parameter_breakdown"]["optimizer_active_current"] == 1_601_345


def test_live_model_footer_keeps_v31_runtime_after_safe_learner_rollback(
    tmp_path: Path,
) -> None:
    dormant_contract = tmp_path / "dormant.json"
    staged_roster = tmp_path / "roster.json"
    ready = tmp_path / "ready.json"
    boundary = tmp_path / "boundary.json"
    expert_ids = [f"route-{index:02d}" for index in range(22)]
    trained_ids = expert_ids[:15]
    accepted_ids = expert_ids[:14]
    origin_path = "/tmp/iter_00026_matchup_v31.pt"
    origin_digest = "sha256:" + "a" * 64
    rollback_path = "/tmp/iter_00032.pt"
    rollback_digest = "sha256:" + "c" * 64
    _write_json(
        dormant_contract,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [],
        },
    )
    _write_json(
        staged_roster,
        {
            "schema": "poke_bot.matchup_adapter_roster_stage/v1",
            "status": "tested_staged_not_active",
            "runtime_enabled": False,
            "mutually_exclusive_route_per_decision": True,
            "unknown_route_exact_bypass": True,
            "expert_count": 22,
            "expert_ids": expert_ids,
            "parameter_count": 36_080,
            "validation": {"tests_passed": 147},
        },
    )
    _write_json(
        ready,
        {
            "schema": "poke_bot.matchup_runtime_production_ready/v1",
            "runtime_enabled": True,
            "iteration": 27,
            "artifacts": {
                "merged_checkpoint": {
                    "path": origin_path,
                    "digest": origin_digest,
                }
            },
        },
    )
    _write_json(
        boundary,
        {
            "schema": "poke_bot.matchup_runtime_boundary_activation/v1",
            "activated_learner": {
                "path": origin_path,
                "digest": origin_digest,
            },
            "parent_learner": {"path": "/tmp/iter_00026.pt"},
            "boundary": {"next_iteration": 27},
            "adapter_fit": {
                "trained_archetype_ids": trained_ids,
                "route_decisions": {
                    route: 1000 if route in trained_ids else 0
                    for route in expert_ids
                },
            },
            "runtime_tree": {
                "accepted_archetype_ids": accepted_ids,
                "continuous_re_evaluation": True,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
                "digest": "sha256:" + "b" * 64,
            },
        },
    )
    runtime_collection = {
        "available": True,
        "iteration": 39,
        "checkpoint_digest": rollback_digest,
        "combined": {
            "all_games_audited": True,
            "all_runtime_enabled": True,
            "contract_clean": True,
            "audited_games": 8192,
            "games": 8192,
            "accepted_roster_counts": {
                "|".join(accepted_ids): 8192,
            },
        },
        "enforcement": {"required": True, "passed": True},
    }

    model = learner_model_state(
        {
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                    "trainable_parameters": 1_637_425,
                    "dormant_matchup_adapter": {"epochs": 1},
                }
            }
        },
        {"learner": {"path": rollback_path, "digest": rollback_digest}},
        runtime_parameter_contract={
            "trainable_parameters": 1_637_425,
            "checkpoint": rollback_path,
            "source": "/tmp/run.log",
        },
        runtime_collection=runtime_collection,
        dormant_modules_path=dormant_contract,
        staged_adapter_roster_path=staged_roster,
        matchup_runtime_ready_path=ready,
        matchup_runtime_boundary_path=boundary,
    )

    assert model["matchup_adapter_roster_stage"] == {}
    assert model["matchup_adapter_runtime"]["enabled"] is True
    assert model["matchup_adapter_runtime"]["iteration"] == 39
    assert model["matchup_adapter_runtime"]["live_collection_verified"] is True
    assert model["matchup_adapter_runtime"]["expert_count"] == 22
    assert model["matchup_adapter_runtime"]["trained_count"] == 15
    assert model["matchup_adapter_runtime"]["accepted_runtime_count"] == 14
    assert model["dormant_modules"][0]["zero_example_archetype_ids"] == expert_ids[15:]


def test_live_model_footer_accepts_receipt_backed_trained_shadow_adapters(
    tmp_path: Path,
) -> None:
    dormant_contract = tmp_path / "dormant.json"
    _write_json(
        dormant_contract,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [
                {
                    "id": "matchup_adapter_bank_v2",
                    "status": "deployed_dormant",
                    "present_in_active_checkpoint": True,
                    "runtime_enabled": False,
                    "optimizer_active": False,
                    "zero_output": False,
                    "trained_shadow": True,
                    "fit_receipt_valid": True,
                    "parameter_count": 16_400,
                    "expert_count": 10,
                    "hidden_dim": 96,
                    "bottleneck_dim": 8,
                    "expert_ids": ["lucario", "alakazam", "archaludon-ex"],
                }
            ],
        },
    )
    manifest = {
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 1},
                "trainable_parameters": 1_617_745,
            }
        }
    }

    model = learner_model_state(
        manifest,
        dormant_modules_path=dormant_contract,
    )

    assert model["parameter_breakdown"]["current_checkpoint_total"] == 1_617_745
    assert model["parameter_breakdown"]["optimizer_active_current"] == 1_601_345
    assert model["parameter_breakdown"]["current_non_active"] == 16_400
    assert model["dormant_modules"][0]["trained_shadow"] is True


def test_live_model_footer_promotes_matching_loop_fit_to_trained_shadow(
    tmp_path: Path,
) -> None:
    dormant_contract = tmp_path / "dormant.json"
    _write_json(
        dormant_contract,
        {
            "schema": "poke_bot.dormant_model_modules/v1",
            "modules": [
                {
                    "id": "matchup_adapter_bank_v2",
                    "status": "deployed_dormant",
                    "present_in_active_checkpoint": True,
                    "runtime_enabled": False,
                    "optimizer_active": False,
                    "zero_output": True,
                    "parameter_count": 16_400,
                    "expert_count": 10,
                    "hidden_dim": 96,
                    "bottleneck_dim": 8,
                }
            ],
        },
    )
    digest = "sha256:" + "a" * 64
    loop = {
        "learner": {"path": "/tmp/iter_00022.pt", "digest": digest},
        "dormant_matchup_adapter_fit": {
            "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
            "runtime_enabled": False,
            "base_frozen": True,
            "optimizer_scope": "matchup_adapter_bank_only",
            "checkpoint_digest": digest,
            "epochs": 1,
            "steps": 12,
            "rows": 4096,
            "route_sequences": {"lucario": 2903},
            "route_decisions": {"lucario": 12000},
        },
    }
    manifest = {
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 1},
                "trainable_parameters": 1_617_745,
            }
        }
    }

    model = learner_model_state(
        manifest,
        loop,
        dormant_modules_path=dormant_contract,
    )

    adapter = model["dormant_modules"][0]
    assert adapter["trained_shadow"] is True
    assert adapter["zero_output"] is False
    assert adapter["route_sequences"]["lucario"] == 2903


def test_model_panel_labels_current_and_staged_profiles_separately() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "Current production model" in html
    assert 'id="model-params-source"' in html
    assert 'id="model-runtime-roles"' in html
    assert 'id="model-training"' in html
    assert 'id="model-optimizer"' in html
    assert 'id="model-active-params"' in html
    assert 'id="model-dormant-params"' in html
    assert 'id="model-staged-params"' in html
    assert 'id="model-adapter-status"' in html
    assert "Matchup adapter parameters" in html
    assert "MATCHUP ADAPTER BANK (history remains active)" in html
    assert "DEPLOYED DORMANT · RUNTIME OFF" in html
    assert "ROUTER ACTIVE · OUTPUT DORMANT" in html
    assert "TRAINED SHADOW · RUNTIME OFF" in html
    assert "RUNTIME ON · RECEIPT VERIFIED" in html
    assert "continuous per-decision re-evaluation ON" in html
    assert "routes: " in html
    assert "causal public-prefix router ACTIVE in shadow" in html
    assert "FROZEN ROUTED OPPONENTS" in html
    assert "MODEL+DECK+ROUTER VERIFIED" in html


def test_curriculum_progress_parses_shadow_adapter_epoch() -> None:
    progress = parse_curriculum_progress(
        "",
        "rl-adapters ep0:  40%|████| 4/10 [00:05<00:07, 1.25s/batch, loss=0.812, rows=320]",
        iteration_hint=22,
    )

    assert progress["stage"] == "train:matchup-adapters:shadow"
    assert progress["iteration"] == 22
    assert progress["epoch"] == 1
    assert progress["current"] == 4
    assert progress["total"] == 10
    assert progress["metrics"] == {"loss": 0.812, "rows": 320.0}


def test_completed_epoch_line_advances_stale_validation_tqdm() -> None:
    stale = parse_curriculum_progress(
        "",
        "rl-val ep0:  14%|#| 2/14 [00:01<00:10, 1.13batch/s]",
        iteration_hint=26,
    )
    updated = reconcile_completed_train_epoch(
        stale,
        "\n".join(
            (
                "[pure_rl] train begin iter=26 seqs=17663",
                "[rl-train] NEW BEST epoch=0 val_loss=0.8871 acc=90.00%",
                "[rl-train] epoch=1 train_loss=0.8948 val_loss=0.8931 "
                "acc=90.08% patience=2",
            )
        ),
        iteration_hint=26,
        train_epochs=2,
    )

    assert updated["stage"] == "train:policy"
    assert updated["epoch"] == 2
    assert updated["percent"] == 100.0
    assert updated["current"] == 2
    assert updated["total"] == 2
    assert updated["metrics"]["loss"] == 0.8931
    assert updated["metrics"]["acc"] == 90.08


def test_dashboard_holds_scheduler_sps_through_collection_ingest_tail() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert "fleetRates.display_sps" in html
    assert "fleetRates.display_sps_held" in html
    assert "fleetRates.display_sps_estimated" in html
    assert "collectionSpsHeld" in html
    assert "last measured SPS held across telemetry sample gap" in html
    assert "$('rl-sps').textContent=num(collectionSps,1)" in html
    assert "$('batch-sps').textContent=handoffActive?'—':curriculumActive?(rlTraining?num(cp.sps,0):num(collectionSps,0))" in html
    assert "AWR β" in html
    assert 'id="model-next"' in html
    assert "STAGED, NOT ACTIVE" in html
    assert "ACTIVE · " in html
    active_assignment = next(
        line for line in html.splitlines() if "$('model-next').textContent=" in line
    )
    assert active_assignment.index("activeProfile?") < active_assignment.index(
        "planned.id?"
    )
    assert "full-game history ACTIVE" in active_assignment
    assert "KV cache '+(profile.kv_cache?'ACTIVE'" in active_assignment


def test_curriculum_panel_separates_live_public_mix_wr_from_exact_gate() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "active gate held-out win rate" in html
    assert "original-four research excluded" in html
    assert "Number.isFinite(c.heldout_wr)" not in html
    assert 'id="rl-public-wr"' in html
    assert 'id="rl-public-wr-note"' in html
    assert "sampled training behavior · non-gate" in html
    assert "publicWr=c.public_mix_live||{}" in html


def test_strong_public_gate_runtime_progress_is_separate_from_public_mix() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert "nextRuntime=nextGate.runtime||{}" in html
    assert "displayGateResult=!!((activeGateResultAvailable||latestAttemptAvailable)" in html
    assert "displayGateResultCurrent=!!(displayGateResult&&activeGateResultAvailable)" in html
    assert "exactRosterTotal+'-AGENT EXACT GATE" in html
    assert "strong-public holdout games" in html
    assert "researchPhaseLive" in html
    assert "excluded games" in html


def test_expert_panel_separates_refresh_tune_up_and_one_time_bootstrap() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert 'id="expert-window-label"' in html
    assert 'id="expert-tuneup"' in html
    assert 'id="expert-cadence"' in html
    assert "periodic correction only · not the one-time bootstrap" in html
    assert "d.expert_refresh&&d.expert_refresh.available" in html
    assert "prep.active&&!curriculumActive&&!c.run" in html
    assert "Active-run expert replay corpus" in html
    assert "Validated calendar-day sources" in html
    assert "selected games / optimizer decisions" in html
    assert "Active specialist selection" in html


def test_expert_refresh_reports_all_twenty_days(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "expert-latest20-20260702-20260721"
    feature_dir = root / "features-inzi"
    ready = [
        {"date": f"2026-07-{day:02d}", "bytes": 1000 + day}
        for day in range(2, 22)
    ]
    _write_json(
        root / "refresh.status.json",
        {
            "state": "ready_for_featurization",
            "window": {"start": "2026-07-02", "end": "2026-07-21", "days": 20},
            "ready": ready,
        },
    )
    _write_json(
        root / "inzi.features.status.json",
        {
            "state": "running",
            "current_date": "2026-07-04",
            "completed": [
                {"date": "2026-07-02", "records": 9, "decisions": 99},
                {"date": "2026-07-03", "records": 8, "decisions": 88},
            ],
        },
    )
    monkeypatch.setattr(dashboard_snapshot_module, "EXPERT20_ROOT", root)
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "EXPERT20_CURRENT_RECEIPT",
        root / "current.json",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "EXPERT20_REFRESH_STATUS", root / "refresh.status.json"
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "EXPERT20_INZI_STATUS", root / "inzi.features.status.json"
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "EXPERT20_INZI_FINAL_STATUS",
        root / "inzi-final.features.status.json",
    )
    monkeypatch.setattr(dashboard_snapshot_module, "EXPERT20_FEATURE_DIR", feature_dir)
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "EXPERT20_ASSEMBLED_MANIFEST",
        feature_dir / "all-recognized-latest20.manifest.json",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "EXPERT20_ALAKAZAM_CORPUS",
        root / "alakazam/PROTECTED_EXPERT_CORPUS.json",
    )
    monkeypatch.setattr(dashboard_snapshot_module, "run", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda *_args, **_kwargs: {"active": True},
    )

    state = dashboard_snapshot_module.expert_refresh_state()

    assert state["total_days"] == 20
    assert state["archive_ready_days"] == 20
    assert state["feature_ready_days"] == 2
    assert len(state["days"]) == 20
    assert state["days"][2]["stage"] == "featurizing"
    # Overall progress weights verified archives 25%, completed features 50%,
    # and shards landed on Inzi 25%. Two remote-only feature shards therefore
    # add 5 points to the completed 25-point archive phase.
    assert state["percent"] == 30.0


def test_expert_refresh_overlays_live_elmo_daily_materialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    days = [f"2026-07-{day:02d}" for day in range(4, 24)]
    current = tmp_path / "expert-latest20-current.json"
    _write_json(
        current,
        {
            "schema": "poke_bot.expert_latest20_receipt/v1",
            "status": "ready",
            "days": 20,
            "committed_at": "2026-07-24T20:00:00Z",
            "archives": [
                {
                    "date": day,
                    "bytes": 1_000,
                    "episode_count": 10,
                    "sha256": "sha256:" + "a" * 64,
                    "validated": True,
                }
                for day in days
            ],
        },
    )
    remote_status = "".join(
        json.dumps(
            {
                "schema": "pokebot-authoritative-archetype-window-status/v1",
                "state": "running",
                "current_date": day,
                "date_window": {"start": day, "end": day, "days": 1},
                "completed": [],
            }
        )
        for day in days[:2]
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "EXPERT20_CURRENT_RECEIPT",
        current,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "LATEST20_SPECIALIST_SYNC_STATE",
        tmp_path / "missing-specialist-sync.json",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "LATEST20_SPECIALIST_CURRENT",
        tmp_path / "missing-specialist-current",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "V6_STRATEGIC_SPECIALIST_SYNC_STATE",
        tmp_path / "missing-v6-sync.json",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "V6_STRATEGIC_SPECIALIST_CURRENT",
        tmp_path / "missing-v6-current",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "run",
        lambda *_args, **_kwargs: remote_status,
    )

    state = dashboard_snapshot_module.expert_refresh_state()

    assert state["active"] is True
    assert state["stage"] == "featurizing"
    assert state["phase"] == "parallel_daily_materialization"
    assert state["archive_ready_days"] == 20
    assert state["feature_ready_days"] == 0
    assert state["daily_materialization"] == {
        "selected_days": 2,
        "completed_days": 0,
        "running_days": 2,
        "failed_days": 0,
        "ready": False,
        "finalization_pending": False,
        "finalization_ready": False,
    }
    assert "0/2 selected missing daily features complete" in state["latest_line"]
    assert len(state["days"]) == 20
    assert [row["stage"] for row in state["days"][:2]] == [
        "featurizing",
        "featurizing",
    ]
    assert all(row["service"]["active"] for row in state["days"][:2])
    assert all(
        row["stage"] == "archive_ready" for row in state["days"][2:]
    )


def test_v6_strategic_corpus_progress_is_separate_and_receipt_backed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    days = [f"2026-07-{day:02d}" for day in range(4, 24)]
    statuses = {
        day: {
            "state": "complete" if index < 2 else "running" if index < 4 else "waiting",
            "completed": [{"date": day}] if index < 2 else [],
        }
        for index, day in enumerate(days)
    }
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "V6_STRATEGIC_SPECIALIST_SYNC_STATE",
        tmp_path / "missing-v6-sync.json",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "V6_STRATEGIC_SPECIALIST_CURRENT",
        tmp_path / "missing-v6-current",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "_elmo_latest20_daily_materialization",
        lambda status_glob=dashboard_snapshot_module.EXPERT20_ELMO_DAILY_STATUS_GLOB: (
            statuses
            if status_glob
            == dashboard_snapshot_module.EXPERT20_V6_STRATEGIC_ELMO_DAILY_STATUS_GLOB
            else {}
        ),
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "run",
        lambda command, **_kwargs: (
            "active\n"
            if command[:3] == ["systemctl", "--user", "is-active"]
            and command[-1]
            == dashboard_snapshot_module.V6_STRATEGIC_SPECIALIST_SYNC_SERVICE
            else ""
        ),
    )

    state = dashboard_snapshot_module.v6_strategic_corpus_state(days)

    assert state["available"] is True
    assert state["active"] is True
    assert state["complete"] is False
    assert state["phase"] == "parallel_daily_materialization"
    assert state["completed_days"] == 2
    assert state["running_days"] == 2
    assert state["total_days"] == 20
    assert state["target_schema"] == "poke_bot.expanded_strategic_targets/v2"
    assert state["target_digest"].startswith("sha256:")
    assert "2/20 daily feature shards ready" in state["latest_line"]


def test_expert_refresh_uses_finalized_sync_receipt_for_twenty_days(
    tmp_path: Path,
    monkeypatch,
) -> None:
    days = [f"2026-07-{day:02d}" for day in range(4, 24)]
    current = tmp_path / "expert-latest20-current.json"
    sync = tmp_path / "expert-latest20-specialist-sync.json"
    pointer = tmp_path / "current-specialist-latest20"
    _write_json(
        current,
        {
            "schema": "poke_bot.expert_latest20_receipt/v1",
            "status": "ready",
            "days": 20,
            "committed_at": "2026-07-24T20:00:00Z",
            "archives": [
                {
                    "date": day,
                    "bytes": 1_000,
                    "episode_count": 10,
                    "sha256": "sha256:" + "a" * 64,
                    "validated": True,
                }
                for day in days
            ],
        },
    )
    _write_json(
        sync,
        {
            "schema": "poke_bot.latest20_specialist_sync/v1",
            "status": "syncing",
            "dates": days,
            "specialist_count": 18,
            "source_bytes": 1_000,
            "copied_bytes": 400,
            "bandwidth_limit_kib_per_second": 8_000,
        },
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "EXPERT20_CURRENT_RECEIPT",
        current,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "LATEST20_SPECIALIST_SYNC_STATE",
        sync,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "LATEST20_SPECIALIST_CURRENT",
        pointer,
    )

    def fake_run(command, **_kwargs):
        if command[:3] == ["systemctl", "--user", "is-active"]:
            return "activating\n"
        return ""

    monkeypatch.setattr(dashboard_snapshot_module, "run", fake_run)

    state = dashboard_snapshot_module.expert_refresh_state()

    assert state["active"] is True
    assert state["complete"] is False
    assert state["stage"] == "syncing_specialist_corpora"
    assert state["phase"] == "atomic_checksum_sync_to_inzi"
    assert state["archive_ready_days"] == 20
    assert state["feature_ready_days"] == 20
    assert state["completed_days"] == 20
    assert state["percent"] == 85.0
    assert all(row["stage"] == "feature_ready" for row in state["days"])
    assert state["specialist_sync"]["current_bytes"] == 400
    assert state["specialist_sync"]["total_bytes"] == 1_000
    assert state["specialist_sync"]["percent"] == 40.0
    assert "400/1,000 bytes (40.0%)" in state["latest_line"]


def test_live_curriculum_outranks_durable_bootstrap_marker(monkeypatch) -> None:
    def stale_bootstrap_must_not_run() -> dict:
        raise AssertionError("completed bootstrap shadowed live curriculum")

    monkeypatch.setattr(
        "scripts.dashboard_snapshot.alakazam_bootstrap_progress",
        stale_bootstrap_must_not_run,
    )
    curriculum = {
        "active": True,
        "active_pids": [4242],
        "run": "temporal-live",
        "iteration": 3,
        "progress_source": "/tmp/temporal.progress.log",
        "progress_log_source": "/tmp/temporal.progress.log",
        "progress_updated_at": 123.0,
        "source_current": True,
        "progress": {
            "line": "rl-train ep1: 50%",
            "stage": "train:policy",
            "iteration": 3,
            "epoch": 2,
            "current": 50,
            "total": 100,
            "percent": 50.0,
            "sps": 5_700.0,
            "metrics": {"loss": 3.1},
        },
        "worker": {"rss_bytes": 12_345, "source": "systemd-user-cgroup"},
    }

    state = authoritative_training_state(curriculum, {"bootstrap": {"active": True}})

    assert state["mode"] == "curriculum_rl"
    assert state["status"] == "running"
    assert state["run"] == "temporal-live"
    assert state["phase"] == "train:policy"
    assert state["samples_per_second"] == 5_700.0
    assert state["service"]["pid"] == 4242
    assert state["source"] == "/tmp/temporal.progress.log"


def test_stopped_curriculum_progress_is_retained_but_never_marked_fresh() -> None:
    curriculum = {
        "active": False,
        "active_pids": [],
        "run": "interrupted-iteration-27",
        "progress_source": "/tmp/interrupted.progress.status",
        "progress_log_source": "/tmp/interrupted.progress.log",
        "progress_updated_at": 123.0,
        "source_current": True,
        "progress": {
            "line": "pure_rl collect:self_play iter=27: 24%",
            "stage": "collect:self_play",
            "iteration": 27,
            "current": 369,
            "total": 1536,
            "percent": 24.0,
        },
        "worker": {},
    }

    state = authoritative_training_state(curriculum, {})

    assert state["status"] == "stopped"
    assert state["fresh"] is False
    assert state["latest_line"].startswith(
        "Last stopped-run progress (historical):"
    )
    assert "iter=27: 24%" in state["latest_line"]


def test_model_panel_never_defaults_missing_temporal_telemetry_to_zero() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "profile.temporal_layers??model.temporal_layers??'—'" in html
    assert "profile.temporal_layers??model.temporal_layers??0" not in html
    assert "refusing to substitute an older/default profile" in html


def test_dashboard_page_reloads_when_server_ui_version_changes() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    rendered = rendered_index().decode("utf-8")

    assert "__DASHBOARD_UI_VERSION__" in html
    assert "__DASHBOARD_UI_VERSION__" not in rendered
    assert f"const DASHBOARD_UI_VERSION='{dashboard_ui_version()}'" in rendered
    assert "d.dashboard_ui_version!==DASHBOARD_UI_VERSION" in html
    assert "window.location.replace" in html


def test_active_manifest_profile_is_not_reported_as_staged(
    tmp_path: Path, monkeypatch
) -> None:
    profile = {
        "d_model": 96,
        "decision_context": "history",
        "kv_cache": True,
        "max_context": 320,
        "temporal_layers": 1,
    }
    registry = tmp_path / "profiles.json"
    registry.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "full_game_temporal_v1",
                        "status": "staged_for_four_iteration_boundary",
                        "trainable_parameters": 1_601_345,
                        "profile": profile,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.dashboard_snapshot.MODEL_PROFILE_REGISTRY", registry
    )
    manifest = {
        "run_name": "temporal-live",
        "design_contract": {
            "learner": {
                "profile": profile,
                "trainable_parameters": 1_601_345,
            }
        },
    }

    model = learner_model_state(manifest, iteration=0)

    assert model["profile_id"] == "full_game_temporal_v1"
    assert model["planned_profile"] == {}


def test_deployed_profile_registry_marks_temporal_history_active() -> None:
    registry = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "deploy/state/pure_rl_model_profiles.json"
        ).read_text(encoding="utf-8")
    )
    profiles = {row["id"]: row for row in registry["profiles"]}

    active = profiles["full_game_temporal_v1"]
    assert active["status"] == "active_production"
    assert active["profile"]["decision_context"] == "history"
    assert active["profile"]["temporal_layers"] == 1
    assert active["profile"]["kv_cache"] is True
    assert profiles["state_evaluator_v1"]["status"].startswith("superseded")


def test_scheduler_queue_panel_is_separate_and_shows_flow_and_latency() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert 'data-card="scheduler" data-widget="scheduler"' in html
    assert 'id="scheduler-elmo-flow"' in html
    assert 'id="scheduler-bert-flow"' in html
    assert 'id="scheduler-result-flow"' in html
    assert 'id="scheduler-latency"' in html
    assert "generation → ingest" in html
    assert "0 · collection complete / 0" in html
    assert "endpoint queues are intentionally idle" in html


def test_scheduler_unassigned_never_leaks_from_preceding_phase(monkeypatch) -> None:
    queue = (
        "[remote] endpoint_owned_queues "
        "depths={'192.168.1.143:8765': 376} "
        "high_water={'192.168.1.143:8765': 376} "
        "shared_endpoint_race=disabled"
    )
    raw = "\n".join(
        [
            queue,
            "[pure_rl] mid_iter_rebalance=done remaining=0",
            "[pure_rl] tqdm stage=collect:public_mix jobs=64512",
            queue,
            "[remote] queue_refill_controller interval=1.000s low_water=50% "
            "action=fill_to_high_water endpoints=parallel ingest_coupled=false",
        ]
    )
    monkeypatch.setattr("scripts.dashboard_snapshot.read_tail", lambda *_args: raw)

    state = scheduler_queue_state("active-run")

    assert state["available"] is True
    assert state["unassigned"] is None


def test_scheduler_unassigned_uses_current_phase_heartbeat(monkeypatch) -> None:
    queue = (
        "[remote] endpoint_owned_queues "
        "depths={'192.168.1.143:8765': 376} "
        "high_water={'192.168.1.143:8765': 376} "
        "shared_endpoint_race=disabled"
    )
    raw = "\n".join(
        [
            queue,
            "[pure_rl] mid_iter_rebalance=done remaining=0",
            queue,
            "[pure_rl] mid_iter_rebalance=scheduler=mid_iter remaining=62000",
        ]
    )
    monkeypatch.setattr("scripts.dashboard_snapshot.read_tail", lambda *_args: raw)

    state = scheduler_queue_state("active-run")

    assert state["unassigned"] == 62000


def test_optimizer_batches_are_never_reported_as_unassigned_games() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=0, bert_jobs=0)
    payload["curriculum"]["stage"] = "train:prep:baseline"
    payload["curriculum"]["progress"].update(
        stage="train:prep:baseline",
        current=271,
        total=387,
        unit="batches",
    )
    payload["curriculum"]["scheduler_queues"] = {
        "available": False,
        "mode": "legacy_or_starting",
    }

    SnapshotCache()._annotate_scheduler_queues(payload)

    queues = payload["scheduler_queues"]
    assert queues["unassigned"] == 0
    assert queues["unassigned_estimated"] is False
    assert queues["unassigned_source"] == (
        "scheduler idle outside game-generation phase (train:prep:baseline)"
    )


def test_collection_games_can_still_use_preheartbeat_unassigned_estimate() -> None:
    payload = _fleet_payload(active=True, elmo_jobs=0, bert_jobs=0)
    payload["curriculum"]["progress"].update(current=3604, total=6963, unit="games")
    payload["curriculum"]["scheduler_queues"] = {
        "available": False,
        "mode": "legacy_or_starting",
    }

    SnapshotCache()._annotate_scheduler_queues(payload)

    queues = payload["scheduler_queues"]
    assert queues["unassigned"] == 3359
    assert queues["unassigned_estimated"] is True


def test_elmo_has_bounded_task_headroom_for_constant_refill() -> None:
    root = Path(__file__).resolve().parents[1]
    compose_path = root / ".staging/elmo-docker-compose.production.yml"
    if not compose_path.is_file():
        pytest.skip("host-only Elmo Compose staging artifact is unavailable")
    compose = compose_path.read_text(encoding="utf-8")
    assert 'POKEBOT_REMOTE_MAX_CONNECTIONS: "420"' in compose
    assert re.search(r"^\s*pids_limit:\s*1536\s*$", compose, re.MULTILINE)
    assert re.search(r"^\s*mem_limit:\s*64g\s*$", compose, re.MULTILINE)
    assert 'POKEBOT_REMOTE_TREE_RSS_LIMIT_GB: "30"' in compose


def test_active_specialist_handoff_owns_the_top_progress_card() -> None:
    """A stopped prior curriculum run must not overwrite live handoff tqdm."""

    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert (
        "handoffActive?(Number.isFinite(handoff.percent)"
        in html
    )
    assert (
        "handoffActive?(Number.isFinite(handoff.epoch)"
        in html
    )
    assert (
        "handoffActive?(handoff.current??'—')+' / '+(handoff.total??'—')"
        in html
    )
    assert (
        "handoffActive?handoff.percent||0:isPrep?prep.percent"
        in html
    )
    assert (
        "handoffActive?handoff.latest_line:isPrep?prep.latest_line"
        in html
    )
    assert "handoffActive?(Number.isFinite(p.percent)" not in html


def test_completed_trevenant_handoff_is_not_current_during_starmie() -> None:
    historical = {
        "available": True,
        "active": False,
        "phase": "next_specialist_rl_started",
        "stage": "next_specialist_rl_started",
        "label": "Alakazam frozen → deck-agnostic core → Hop's Trevenant",
        "latest_line": "Hop's Trevenant specialist RL is active.",
    }

    result = (
        dashboard_snapshot_module.reconcile_current_specialist_handoff(
            historical,
            active_specialist="starmie",
            program_progress={"remaining_after_active": 19},
            next_specialist="lucario",
        )
    )

    assert result["phase"] == "waiting_for_active_specialist_gate"
    assert result["label"] == "Starmie → next unfinished specialist"
    assert "19 specialists remain" in result["latest_line"]
    assert result["historical_source_suppressed"] is True
    assert result["source_specialist_id"] == "starmie"
    assert result["next_specialist_id"] == "lucario"
    assert result["historical_source_specialist_id"] is None


def test_dashboard_source_integrity_covers_every_visible_card() -> None:
    payload = {
        "observed_at": time.time(),
        "dashboard_sampled_at": time.time(),
        "service": {
            "active": True,
            "pid": 123,
            "restart_count": 0,
            "name": "production.service",
            "command": "trainer --run run-a",
        },
        "training": {"phase": "heldout"},
        "bootstrap": {
            "phase": "heldout",
            "compatibility_alias": True,
            "alias_of": "training",
        },
        "transition": {"active": False, "historical": True},
        "specialist_handoff": {"active": False},
        "baseline_eval": {"historical": True},
        "expert_refresh": {
            "available": True,
            "complete": True,
            "authoritative_for_active_run": True,
            "archive_window_ready": True,
            "assembled_manifest_ready": True,
            "filtered_corpus_ready": True,
            "window_start": "2026-07-02",
            "window_end": "2026-07-21",
            "source": "/expert/status.json",
        },
        "specialist_protocol": {
            "available": True,
            "runtime_active_specialist": "dragapult-dusknoir",
            "canonical_active_specialist": "dragapult-dusknoir",
            "active_specialist": "dragapult-dusknoir",
            "source": "/state/specialists.yaml",
        },
        "curriculum": {
            "active": True,
            "source_current": True,
            "run": "run-a",
            "iteration": 1,
            "stage": "heldout:strong_public_gate",
            "progress_status_source": "/run/progress.status",
            "last_completed_iteration": 0,
            "heldout_source": "/run/commit.json",
            "gate_program": {
                "source": "/config/gate.json",
                "next_gate": {
                    "available": True,
                    "contract_valid": True,
                    "contract_source": "/config/gate.json",
                },
            },
        },
        "model": {
            "active_checkpoint": "/run/iter_00000.pt",
            "active_checkpoint_digest": "sha256:" + "a" * 64,
            "checkpoint_structure": {
                "verified": True,
                "checkpoint": "/run/iter_00000.pt",
                "checkpoint_digest": "sha256:" + "a" * 64,
            },
        },
        "gpus": [{"index": 1}],
        "scheduler_queues": {
            "available": True,
            "updated_at": time.time(),
            "source": "/run/queues.json",
        },
        "fleet": {
            key: {
                "reachable": True,
                "production_active": True,
                "name": key,
                "worker": {
                    "active": True,
                    "health_current": True,
                    "command": f"{key}-worker",
                    "rate_source": f"/{key}/rate.json",
                },
            }
            for key in ("inzi", "elmo", "bert")
        },
    }

    SnapshotCache._annotate_source_integrity(payload)

    integrity = payload["source_integrity"]
    visible_cards = {
        "stage",
        "bootstrap",
        "throughput",
        "blackwell",
        "outcomes",
        "latest10",
        "progress",
        "adapterfleet",
        "replay",
        "baseline",
        "nextgate",
        "protocol",
        "hardware",
        "pure",
        "fleet",
        "scheduler",
        "curriculum",
        "model",
        "command",
        "raw",
    }
    assert visible_cards.issubset(integrity["rows"])
    assert integrity["current"] is True
    payload["model"]["checkpoint_structure"]["checkpoint_digest"] = (
        "sha256:" + "b" * 64
    )
    SnapshotCache._annotate_source_integrity(payload)
    assert payload["source_integrity"]["current"] is False
    assert "model" in payload["source_integrity"]["failed"]


def test_dashboard_source_integrity_keeps_canonical_protocol_current_while_stopped() -> None:
    payload = {
        "observed_at": time.time(),
        "dashboard_sampled_at": time.time(),
        "service": {
            "active": False,
            "active_state": "failed",
            "sub_state": "failed",
            "pid": 0,
            "restart_count": 3,
            "name": "production.service",
            "command": "trainer --run run-a",
        },
        "training": {"phase": "collect:public_mix"},
        "bootstrap": {
            "phase": "collect:public_mix",
            "compatibility_alias": True,
            "alias_of": "training",
        },
        "specialist_protocol": {
            "available": True,
            "canonical_pointer_stale": False,
            "runtime_active_specialist": None,
            "canonical_active_specialist": "dragapult-dusknoir",
            "active_specialist": "dragapult-dusknoir",
            "source": "/state/specialists.yaml",
        },
        "curriculum": {
            "active": False,
            "source_current": True,
            "run": "run-a",
            "iteration": 2,
            "stage": "collect:public_mix",
            "progress_status_source": "/run/progress.status",
            "last_completed_iteration": 1,
            "gate_program": {
                "next_gate": {
                    "available": True,
                    "contract_valid": True,
                    "contract_source": "/config/gate.json",
                }
            },
        },
        "model": {
            "active_checkpoint": "/run/iter_00001.pt",
            "active_checkpoint_digest": "sha256:" + "a" * 64,
            "checkpoint_structure": {
                "verified": True,
                "checkpoint": "/run/iter_00001.pt",
                "checkpoint_digest": "sha256:" + "a" * 64,
            },
        },
        "expert_refresh": {
            "available": True,
            "complete": True,
            "authoritative_for_active_run": True,
            "archive_window_ready": True,
            "assembled_manifest_ready": True,
            "filtered_corpus_ready": True,
        },
        "gpus": [{"index": 1}],
        "fleet": {
            "inzi": {"reachable": True, "worker": {"active": False}},
            "elmo": {"reachable": True, "worker": {"active": True}},
            "bert": {"reachable": True, "worker": {"active": True}},
        },
    }

    SnapshotCache._annotate_source_integrity(payload)

    integrity = payload["source_integrity"]
    assert integrity["rows"]["protocol"]["current"] is True
    assert "protocol" not in integrity["failed"]
    # Operational cards remain honestly red; canonical protocol availability
    # does not pretend that a failed production controller is running.
    assert integrity["rows"]["stage"]["current"] is False
    assert integrity["rows"]["progress"]["current"] is False
    assert integrity["rows"]["bootstrap"]["current"] is False
    assert integrity["rows"]["throughput"]["current"] is False
    assert integrity["rows"]["curriculum"]["current"] is False
    assert integrity["rows"]["pure"]["current"] is False


def test_dashboard_source_integrity_accepts_fresh_receipt_backed_handoff_interval() -> None:
    now = time.time()
    payload = {
        "observed_at": now,
        "dashboard_sampled_at": now,
        "service": {
            "active": False,
            "pid": 0,
            "restart_count": 0,
            "name": "production.service",
        },
        "specialist_handoff": {
            "active": True,
            "pid": 321,
            "updated_at": now,
            "phase": "source_specialist_verified",
            "stage": "deck_agnostic_cumulative_core_training",
            "source": "/state/post-source-core-handoff.json",
            "source_specialist_id": "source-deck",
            "next_specialist_id": "target-deck",
            "service": {"name": "specialist-handoff.service"},
        },
        "training": {
            "mode": "specialist_handoff",
            "source": "/state/post-source-core-handoff.json",
        },
        "specialist_protocol": {
            "available": True,
            "canonical_pointer_stale": True,
            "runtime_identity_reconciled": False,
            "runtime_active_specialist": None,
            "canonical_active_specialist": "source-deck",
            "active_specialist": "",
            "required_target_count": 2,
            "program_progress": {"completed_specialist_ids": []},
            "specialists": [
                {
                    "id": "source-deck",
                    "active": True,
                    "frozen": False,
                    "public_mix_eligible": False,
                },
                {
                    "id": "target-deck",
                    "active": False,
                    "frozen": False,
                    "public_mix_eligible": False,
                },
            ],
            "frozen_inference_opponents": [],
            "source": "/state/specialists.yaml",
        },
        "curriculum": {
            "active": False,
            "run": "source-run",
            "iteration": 5,
            "stage": "measure:research_controls",
        },
        "model": {
            "checkpoint_structure": {"adapter_expert_count": 2},
        },
        "gpus": [{"index": 1}],
        "fleet": {
            "inzi": {"reachable": True, "worker": {"active": False}},
            "elmo": {"reachable": True, "worker": {"active": True}},
            "bert": {"reachable": True, "worker": {"active": True}},
        },
    }

    SnapshotCache._annotate_source_integrity(payload)

    integrity = payload["source_integrity"]
    assert integrity["rows"]["stage"]["current"] is True
    assert integrity["rows"]["progress"]["current"] is True
    assert integrity["rows"]["protocol"]["current"] is True
    assert integrity["rows"]["protocol"]["checks"]["canonical_pointer"] is True
    assert integrity["rows"]["protocol"]["checks"]["specialist_roster"] is True
    assert integrity["rows"]["handoff"]["current"] is True
    assert "protocol" not in integrity["failed"]
    assert "handoff" not in integrity["failed"]


def test_dashboard_source_integrity_rejects_live_protocol_specialist_mismatch() -> None:
    payload = {
        "dashboard_sampled_at": time.time(),
        "service": {
            "active": True,
            "pid": 456,
            "restart_count": 0,
            "name": "production.service",
        },
        "specialist_protocol": {
            "available": True,
            "canonical_pointer_stale": False,
            "runtime_active_specialist": "starmie",
            "canonical_active_specialist": "dragapult-dusknoir",
            "source": "/state/specialists.yaml",
        },
    }

    SnapshotCache._annotate_source_integrity(payload)

    assert payload["source_integrity"]["rows"]["protocol"]["current"] is False
    assert "protocol" in payload["source_integrity"]["failed"]


def test_dashboard_latest20_archive_is_current_while_filtered_corpus_builds() -> None:
    payload = {
        "dashboard_sampled_at": time.time(),
        "service": {"active": True, "pid": 10, "restart_count": 0},
        "specialist_protocol": {
            "available": True,
            "runtime_active_specialist": "dragapult-dusknoir",
            "canonical_active_specialist": "dragapult-dusknoir",
            "source": "/state/specialists.yaml",
        },
        "curriculum": {
            "active": True,
            "source_current": True,
            "run": "run-a",
            "iteration": 2,
            "stage": "train:baseline",
            "progress_status_source": "/run/progress.status",
        },
        "model": {},
        "expert_refresh": {
            "available": True,
            "complete": False,
            "authoritative_for_active_run": True,
            "archive_window_ready": True,
            "assembled_manifest_ready": False,
            "filtered_corpus_ready": False,
            "window_start": "2026-07-04",
            "window_end": "2026-07-23",
            "total_days": 20,
            "days": [
                {"day": f"2026-07-{day:02d}", "stage": "source_ready_unfiltered"}
                for day in range(4, 24)
            ],
            "source": "/state/expert-latest20-current.json",
        },
        "fleet": {},
    }

    SnapshotCache._annotate_source_integrity(payload)

    latest = payload["source_integrity"]["rows"]["latest10"]
    assert latest["current"] is True
    assert latest["checks"]["archive_window"] is True
    assert latest["checks"]["filtered_corpus_ready"] is False
    assert "latest10" not in payload["source_integrity"]["failed"]


def test_dashboard_source_integrity_rejects_canonical_frozen_pool_drift() -> None:
    payload = {
        "dashboard_sampled_at": time.time(),
        "service": {
            "active": False,
            "pid": 0,
            "restart_count": 0,
            "name": "production.service",
        },
        "specialist_protocol": {
            "available": True,
            "canonical_pointer_stale": False,
            "canonical_active_specialist": "dragapult-dusknoir",
            "required_target_count": 2,
            "specialists": [
                {
                    "id": "alakazam",
                    "active": False,
                    "frozen": False,
                    "public_mix_eligible": False,
                },
                {
                    "id": "dragapult-dusknoir",
                    "active": True,
                    "frozen": False,
                    "public_mix_eligible": False,
                },
            ],
            "frozen_inference_opponents": [
                {"specialist_id": "alakazam", "inference_only": True}
            ],
            "source": "/state/specialists.yaml",
        },
        "model": {
            "checkpoint_structure": {
                "adapter_expert_count": 2,
            },
        },
    }

    SnapshotCache._annotate_source_integrity(payload)

    protocol_row = payload["source_integrity"]["rows"]["protocol"]
    assert protocol_row["current"] is False
    assert protocol_row["checks"]["specialist_roster"] is True
    assert protocol_row["checks"]["model_roster"] is True
    assert protocol_row["checks"]["frozen_pool"] is False


def test_dashboard_uses_training_environment_for_checkpoint_snapshot() -> None:
    server_source = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/server.py"
    ).read_text(encoding="utf-8")

    assert (
        'REMOTE_PYTHON = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"'
        in server_source
    )
    assert (
        '"inzi@192.168.1.151",\n'
        "            REMOTE_PYTHON,\n"
        "            REMOTE_SNAPSHOT,"
    ) in server_source


def test_active_expert_card_uses_run_pinned_specialist_corpus(
    tmp_path: Path,
) -> None:
    corpus_dir = (
        tmp_path
        / "expert-evidence28-20260626-20260723"
        / "specialist-corpora-v2"
        / "dragapult-dusknoir"
    )
    corpus_dir.mkdir(parents=True)
    shard = corpus_dir / "day.features"
    shard.write_bytes(b"feature-data")
    manifest = corpus_dir / "manifest.json"
    manifest_payload = {
        "format": "pokebot-bootstrap-feature-manifest",
        "date_start": "2026-06-26",
        "date_end": "2026-06-26",
        "dates": ["2026-06-26"],
        "selection": {"value": "dragapult-dusknoir"},
        "quality_gates": {"passed": True},
        "shards": [
            {
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(shard.read_bytes()).hexdigest(),
                "source_dates": ["2026-06-26"],
                "stats": {"records_kept": 168, "decisions_kept": 10_946},
            }
        ],
        "totals": {"records_kept": 168, "decisions_kept": 10_946},
    }
    _write_json(manifest, manifest_payload)
    manifest_digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    protected = corpus_dir / "PROTECTED_EXPERT_CORPUS.json"
    _write_json(
        protected,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": manifest.name,
            "manifest_sha256": manifest_digest,
        },
    )

    result = dashboard_snapshot_module.active_expert_corpus_state(
        {
            "expert_rehearsal": {
                "manifest": str(protected),
                "active": False,
                "state": "scheduled",
            }
        },
        {
            "source": "/old/refresh.status.json",
            "window_start": "2026-07-02",
            "window_end": "2026-07-21",
            "complete": True,
            "days": [
                {
                    "day": f"2026-07-{day:02d}",
                    "stage": "archive_ready",
                    "percent": 50.0,
                }
                for day in range(2, 22)
            ],
        },
    )

    assert result["authoritative_for_active_run"] is True
    assert result["complete"] is False
    assert result["specialist_id"] == "dragapult-dusknoir"
    assert result["records_kept"] == 168
    assert result["decisions_kept"] == 10_946
    assert result["source_day_contract_satisfied"] is False
    assert result["total_days"] == 20
    assert len(result["days"]) == 20
    assert result["window_start"] == "2026-07-02"
    assert result["window_end"] == "2026-07-21"
    assert all(
        row["active_specialist_filter_receipt"] is False
        for row in result["days"]
    )
    assert all(
        row["specialist_id"] == "dragapult-dusknoir"
        and row["matching_status"] == "filter_receipt_missing"
        for row in result["days"]
    )
    assert all(
        row["stage"] == "source_ready_unfiltered" for row in result["days"]
    )
    assert result["historical_fallback"]["used"] is False
    assert result["historical_fallback"]["not_latest20"] is True
    assert result["evidence_window_end"] == "2026-07-23"
    assert result["archive_refresh_history"]["superseded_by"] == (
        "active_run_pinned_expert_corpus"
    )


def test_active_expert_card_preserves_all_20_calendar_days_after_filtering(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "latest20" / "dragapult-dusknoir"
    corpus_dir.mkdir(parents=True)
    shard = corpus_dir / "matching.features"
    shard.write_bytes(b"filtered-feature-data")
    days = [f"2026-07-{day:02d}" for day in range(2, 22)]
    manifest = corpus_dir / "manifest.json"
    manifest_payload = {
        "format": "pokebot-bootstrap-feature-manifest",
        "date_start": days[0],
        "date_end": days[-1],
        "dates": days,
        "selection": {"value": "dragapult-dusknoir"},
        "source_window": {
            "unit": "calendar_day",
            "selection": "latest_available_fully_validated_daily_sources",
            "days": 20,
            "dates": days,
            "filter_applied_after_window_selection": True,
            "filter_archetype": "dragapult-dusknoir",
        },
        "source_days": [
            {
                "date": day,
                "source_feature_sha256": "sha256:" + "a" * 64,
                "source_feature_validated": True,
                "source_archive_sha256": "sha256:" + "b" * 64,
                "source_archive_validated": True,
                "matching_games": 1 if index == 0 else 0,
                "matching_decisions": 8 if index == 0 else 0,
                "filtered_feature_present": index == 0,
            }
            for index, day in enumerate(days)
        ],
        "quality_gates": {"passed": True},
        "shards": [
            {
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(shard.read_bytes()).hexdigest(),
                "source_dates": [days[0]],
                "stats": {"records_kept": 1, "decisions_kept": 8},
            }
        ],
        "totals": {"records_kept": 1, "decisions_kept": 8},
    }
    _write_json(manifest, manifest_payload)
    protected = corpus_dir / "PROTECTED_EXPERT_CORPUS.json"
    _write_json(
        protected,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": manifest.name,
            "manifest_sha256": (
                "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
            ),
        },
    )

    result = dashboard_snapshot_module.active_expert_corpus_state(
        {
            "expert_rehearsal": {
                "manifest": str(protected),
                "active": False,
                "state": "scheduled",
            }
        },
        {},
    )

    assert result["complete"] is True
    assert result["source_day_contract_satisfied"] is True
    assert result["total_days"] == 20
    assert result["feature_ready_days"] == 20
    assert len(result["days"]) == 20
    assert sum(row["zero_match_present"] for row in result["days"]) == 19
    assert result["latest20"]["dates"] == days
    assert result["latest20"]["matching_games"] == 1
    assert result["latest20"]["all_zero_matches"] is False
    assert result["days"][0]["matching_status"] == "matches_present"
    assert all(
        row["matching_status"] == "zero_matches"
        for row in result["days"][1:]
    )
    assert result["historical_fallback"]["used"] is False


def test_active_expert_card_displays_finalized_next_boundary_window(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "historical-bound-corpus"
    corpus_dir.mkdir()
    shard = corpus_dir / "bound.features"
    shard.write_bytes(b"immutable-bound-corpus")
    manifest = corpus_dir / "manifest.json"
    _write_json(
        manifest,
        {
            "format": "pokebot-bootstrap-feature-manifest",
            "selection": {"value": "dragapult-dusknoir"},
            "quality_gates": {"passed": True},
            "shards": [
                {
                    "path": shard.name,
                    "bytes": shard.stat().st_size,
                    "sha256": (
                        "sha256:"
                        + hashlib.sha256(shard.read_bytes()).hexdigest()
                    ),
                    "stats": {"records_kept": 27, "decisions_kept": 2_581},
                }
            ],
            "totals": {"records_kept": 27, "decisions_kept": 2_581},
        },
    )
    protected = corpus_dir / "PROTECTED_EXPERT_CORPUS.json"
    _write_json(
        protected,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": manifest.name,
            "manifest_sha256": (
                "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
            ),
        },
    )
    days = [
        (date(2026, 7, 4) + timedelta(days=offset)).isoformat()
        for offset in range(20)
    ]
    refresh = {
        "available": True,
        "active": True,
        "complete": False,
        "archive_window_ready": True,
        "stage": "syncing_specialist_corpora",
        "phase": "atomic_checksum_sync_to_inzi",
        "window_start": days[0],
        "window_end": days[-1],
        "feature_ready_days": 20,
        "local_feature_ready_days": 0,
        "total_days": 20,
        "percent": 86.0,
        "latest_line": "20/20 finalized; checksum sync active",
        "days": [
            {
                "day": day,
                "stage": "feature_ready",
                "percent": 100.0,
                "service": {"active": False},
            }
            for day in days
        ],
    }

    result = dashboard_snapshot_module.active_expert_corpus_state(
        {
            "expert_rehearsal": {
                "manifest": str(protected),
                "active": False,
                "state": "scheduled",
            }
        },
        refresh,
    )

    assert result["stage"] == "syncing_specialist_corpora"
    assert result["phase"] == "atomic_checksum_sync_to_inzi"
    assert result["current"] == 20
    assert result["total"] == 20
    assert result["feature_ready_days"] == 20
    assert result["percent"] == 86.0
    assert all(
        row["binding_status"] == "staged_for_next_safe_boundary"
        for row in result["days"]
    )
    assert result["active_bound_corpus"]["records_kept"] == 27
    assert result["active_bound_corpus"]["decisions_kept"] == 2_581
    assert result["active_bound_corpus"]["immutable_until_safe_boundary"] is True
    assert "immutable run binding retained" in result["latest_line"]


def test_active_expert_card_separates_receipted_all_zero_historical_fallback(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "latest20" / "team-rockets-spidops"
    corpus_dir.mkdir(parents=True)
    fallback_shard = corpus_dir / "historical-fallback.features"
    fallback_shard.write_bytes(b"checksum-bound-historical-features")
    fallback_receipt = corpus_dir / "historical-fallback.receipt.json"
    _write_json(
        fallback_receipt,
        {
            "schema": "poke_bot.historical_expert_fallback/v1",
            "records_kept": 12,
            "decisions_kept": 640,
        },
    )
    days = [
        (date(2026, 7, 2) + timedelta(days=offset)).isoformat()
        for offset in range(20)
    ]
    manifest = corpus_dir / "manifest.json"
    manifest_payload = {
        "format": "pokebot-bootstrap-feature-manifest",
        "date_start": days[0],
        "date_end": days[-1],
        "dates": days,
        "selection": {"value": "team-rockets-spidops"},
        "source_window": {
            "unit": "calendar_day",
            "selection": "latest_available_fully_validated_daily_sources",
            "days": 20,
            "dates": days,
            "filter_applied_after_window_selection": True,
            "filter_archetype": "team-rockets-spidops",
        },
        "source_days": [
            {
                "date": day,
                "source_feature_sha256": "sha256:" + "a" * 64,
                "source_feature_validated": True,
                "source_archive_sha256": "sha256:" + "b" * 64,
                "source_archive_validated": True,
                "matching_games": 0,
                "matching_decisions": 0,
                "filtered_feature_present": False,
            }
            for day in days
        ],
        "historical_fallback": {
            "used": True,
            "reason": "latest20_all_zero_matches",
            "receipt": fallback_receipt.name,
            "receipt_sha256": (
                "sha256:"
                + hashlib.sha256(fallback_receipt.read_bytes()).hexdigest()
            ),
        },
        "quality_gates": {"passed": True},
        "shards": [
            {
                "path": fallback_shard.name,
                "bytes": fallback_shard.stat().st_size,
                "sha256": (
                    "sha256:"
                    + hashlib.sha256(fallback_shard.read_bytes()).hexdigest()
                ),
                "source_dates": ["historical"],
                "stats": {"records_kept": 12, "decisions_kept": 640},
            }
        ],
        "totals": {"records_kept": 12, "decisions_kept": 640},
    }
    _write_json(manifest, manifest_payload)
    protected = corpus_dir / "PROTECTED_EXPERT_CORPUS.json"
    _write_json(
        protected,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": manifest.name,
            "manifest_sha256": (
                "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
            ),
        },
    )

    result = dashboard_snapshot_module.active_expert_corpus_state(
        {
            "expert_rehearsal": {
                "manifest": str(protected),
                "active": False,
                "state": "scheduled",
            }
        },
        {},
    )

    assert result["complete"] is True
    assert result["total_days"] == 20
    assert result["latest20"]["label"] == "Latest 20 calendar days"
    assert result["latest20"]["dates"] == days
    assert result["latest20"]["matching_games"] == 0
    assert result["latest20"]["all_zero_matches"] is True
    assert all(row["zero_match_present"] is True for row in result["days"])
    fallback = result["historical_fallback"]
    assert fallback["used"] is True
    assert fallback["not_latest20"] is True
    assert fallback["label"] == (
        "Historical checksum-receipted fallback · not latest20"
    )
    assert fallback["records_kept"] == 12
    assert fallback["decisions_kept"] == 640
    assert "HISTORICAL FALLBACK USED (not latest20)" in result["latest_line"]


def test_active_expert_card_rejects_historical_fallback_when_latest20_is_nonzero(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "fallback.json"
    _write_json(receipt, {"records_kept": 10, "decisions_kept": 100})
    protected = tmp_path / "PROTECTED_EXPERT_CORPUS.json"
    manifest = {
        "historical_fallback": {
            "used": True,
            "reason": "latest20_all_zero_matches",
            "receipt": receipt.name,
            "receipt_sha256": (
                "sha256:" + hashlib.sha256(receipt.read_bytes()).hexdigest()
            ),
        }
    }

    fallback = (
        dashboard_snapshot_module._checksum_receipted_historical_fallback(
            protected,
            {},
            manifest,
            latest20_all_zero=False,
        )
    )

    assert fallback["available"] is True
    assert fallback["used"] is False
    assert fallback["not_latest20"] is True
    assert "requires 20 validated zero-match" in fallback["rejection_reason"]


def test_model_card_derives_deployed_adapter_version_from_checkpoint() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert "ACTIVE '+esc(adapterVersion)+' SLOT" in html
    assert "physical '+adapterVersion+' slots from active checkpoint" in html
    assert "stable roster and checkpoint agree" in html
    assert "adapterSlotCount" in html
    snapshot_source = (
        Path(__file__).resolve().parents[1] / "scripts/dashboard_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "checkpoint_is_canonical_roster" in snapshot_source
    assert "CANONICAL_MATCHUP_ADAPTER_ROSTER" in snapshot_source


def test_live_post_starmie_handoff_reports_remaining_program(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Keep the historical-service fixture isolated from any reusable cycle
    # receipts present on the machine running the test.
    monkeypatch.setattr(dashboard_snapshot_module, "ROOT", tmp_path)
    state = tmp_path / "post-starmie.json"
    log = tmp_path / "post-starmie.log"
    roster = tmp_path / "matchup-adapter-roster.json"
    frozen_registry = tmp_path / "frozen-specialists.json"
    state.write_text(
        json.dumps(
            {
                "schema": "poke_bot.post_starmie_core_handoff_state/v1",
                "phase": "starmie_pass_verified",
            }
        ),
        encoding="utf-8",
    )
    log.write_text(
        "pack Blackwell corpus:  50%|#####| 16547/33095 "
        "[01:00<01:00, 275.50game/s]\n",
        encoding="utf-8",
    )
    roster.write_text(
        json.dumps(
            {
                "required_specialist_count": 18,
                "expert_ids": [f"specialist-{index}" for index in range(18)],
            }
        ),
        encoding="utf-8",
    )
    frozen_registry.write_text(
        json.dumps({"specialists": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "POST_STARMIE_HANDOFF_STATE", state
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "POST_STARMIE_HANDOFF_LOG", log
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "CANONICAL_MATCHUP_ADAPTER_ROSTER", roster
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "FROZEN_SPECIALIST_REGISTRY", frozen_registry
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda service, **_kwargs: {
            "load_state": "loaded",
            "active": service
            == dashboard_snapshot_module.POST_STARMIE_HANDOFF_SERVICE,
            "active_state": (
                "activating"
                if service
                == dashboard_snapshot_module.POST_STARMIE_HANDOFF_SERVICE
                else "inactive"
            ),
            "pid": (
                123
                if service
                == dashboard_snapshot_module.POST_STARMIE_HANDOFF_SERVICE
                else 0
            ),
            "memory_bytes": 456,
        },
    )

    result = dashboard_snapshot_module.post_starmie_specialist_handoff_state()

    assert result["active"] is True
    assert result["stage"] == "deck_agnostic_core_v2_corpus_pack"
    assert result["current"] == 16547
    assert result["total"] == 33095
    assert result["remaining_specialists_after_starmie"] == 18
    assert result["program_complete"] is False
    assert result["population_transition_ready"] is False


def test_live_generic_cycle_handoff_reports_cumulative_core_training(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent"
    state_root = root / "outputs/state"
    log_root = root / "outputs/logs"
    state_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    state_path = state_root / "post-lucario-core-v3-handoff.json"
    _write_json(
        state_path,
        {
            "schema": "poke_bot.post_starmie_core_handoff_state/v1",
            "phase": "starmie_pass_verified",
            "source": {"specialist_id": "lucario"},
        },
    )
    _write_json(
        state_root / "post-lucario-cumulative-core-handoff.json",
        {
            "schema": "poke_bot.post_specialist_core_refresh_handoff/v1",
            "core_refresh": {"max_epochs": 25},
        },
    )
    log_path = log_root / "specialist-cycle-handoff.log"
    log_path.write_text(
        "expert rehearsal before iter5 ep1/1:  50%|#####| "
        "94/188 [00:35<00:35, 2.65batch/s]\n",
        encoding="utf-8",
    )
    frozen_registry = tmp_path / "frozen.json"
    _write_json(
        frozen_registry,
        {
            "specialists": [
                {"specialist_id": "alakazam"},
                {"specialist_id": "hops-trevenant"},
                {"specialist_id": "starmie"},
            ]
        },
    )
    monkeypatch.setattr(dashboard_snapshot_module, "ROOT", root)
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "SPECIALIST_CYCLE_HANDOFF_LOG",
        log_path,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "FROZEN_SPECIALIST_REGISTRY",
        frozen_registry,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda name, **_kwargs: {
            "name": name,
            "load_state": "loaded",
            "active": True,
            "active_state": "activating",
            "pid": 123,
        },
    )

    result = dashboard_snapshot_module.post_starmie_specialist_handoff_state()

    assert result["active"] is True
    assert result["source_specialist_id"] == "lucario"
    assert result["stage"] == "deck_agnostic_cumulative_core_training"
    assert result["epoch"] == 5
    assert result["epochs_target"] == 25
    assert result["current"] == 94
    assert result["total"] == 188


def test_new_core_version_does_not_reuse_prior_version_tqdm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent"
    state_root = root / "outputs/state"
    log_root = root / "outputs/logs"
    state_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    _write_json(
        state_root / "post-lucario-core-v4-handoff.json",
        {
            "schema": "poke_bot.post_starmie_core_handoff_state/v1",
            "phase": "starmie_pass_verified",
            "source": {"specialist_id": "lucario"},
        },
    )
    _write_json(
        state_root / "post-lucario-cumulative-core-v4-handoff.json",
        {
            "schema": "poke_bot.post_specialist_core_refresh_handoff/v1",
            "core_refresh": {
                "max_epochs": 25,
                "run_dir": str(root / "outputs/bootstrap/core-v4"),
            },
        },
    )
    log_path = log_root / "specialist-cycle-handoff.log"
    log_path.write_text(
        "expert rehearsal before iter25 ep1/1: 100%|#####| "
        "188/188 [01:13<00:00, 2.57batch/s]\n"
        "[core-refresh] loading protected balanced corpus "
        "records=33095 decisions=2362796 archetypes=20 teachers=4 "
        "device=cuda:1\n"
        "[expert-cpu-pack] MISS/REBUILD key=abc reason=missing\n",
        encoding="utf-8",
    )
    frozen_registry = tmp_path / "frozen.json"
    _write_json(
        frozen_registry,
        {
            "specialists": [
                {"specialist_id": "alakazam"},
                {"specialist_id": "hops-trevenant"},
                {"specialist_id": "starmie"},
                {"specialist_id": "lucario"},
            ]
        },
    )
    monkeypatch.setattr(dashboard_snapshot_module, "ROOT", root)
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "SPECIALIST_CYCLE_HANDOFF_LOG",
        log_path,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "FROZEN_SPECIALIST_REGISTRY",
        frozen_registry,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda name, **_kwargs: {
            "name": name,
            "load_state": "loaded",
            "active": True,
            "active_state": "activating",
            "pid": 123,
        },
    )

    result = dashboard_snapshot_module.post_starmie_specialist_handoff_state()

    assert result["source"].endswith("post-lucario-core-v4-handoff.json")
    assert result["stage"] == "deck_agnostic_cumulative_core_corpus_pack"
    assert result["epoch"] is None
    assert result["percent"] is None
    assert "MISS/REBUILD" in result["latest_line"]


def test_live_cycle_selects_newest_transition_by_receipt_not_sorted_digest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent"
    state_root = root / "outputs/state"
    log_root = root / "outputs/logs"
    state_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    _write_json(
        state_root / "specialist-transition-graph.json",
        {
            "transitions": {
                "sha256:zzzz": {
                    "active_specialist": "dragapult-dusknoir",
                    "receipts": {
                        "specialist_transition": {
                            "failed_at": "2026-07-25T04:00:00+00:00"
                        }
                    },
                },
                "sha256:0000": {
                    "active_specialist": "dudunsparce",
                    "receipts": {
                        "specialist_transition": {
                            "failed_at": "2026-07-26T02:15:50+00:00"
                        }
                    },
                },
            }
        },
    )
    _write_json(
        state_root / "post-dudunsparce-core-v6-handoff.json",
        {
            "schema": "poke_bot.post_starmie_core_handoff_state/v1",
            "phase": "starmie_pass_verified",
            "source": {"specialist_id": "dudunsparce"},
        },
    )
    _write_json(
        state_root / "post-dudunsparce-cumulative-core-v6-handoff.json",
        {
            "schema": "poke_bot.post_specialist_core_refresh_handoff/v1",
            "core_refresh": {"max_epochs": 25},
        },
    )
    log_path = log_root / "specialist-transition-graph.log"
    log_path.write_text(
        "[core-refresh] loading protected balanced corpus records=10 "
        "decisions=100 archetypes=2 teachers=6 device=cuda:1\n"
        "pack Blackwell corpus:  40%|####| 4/10 "
        "[00:01<00:01, 4.00game/s]\n",
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.json"
    _write_json(frozen, {"specialists": []})
    monkeypatch.setattr(dashboard_snapshot_module, "ROOT", root)
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "SPECIALIST_TRANSITION_GRAPH_STATE",
        state_root / "specialist-transition-graph.json",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "SPECIALIST_CYCLE_HANDOFF_LOG", log_path
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "FROZEN_SPECIALIST_REGISTRY", frozen
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda name, **_kwargs: {
            "name": name,
            "load_state": "loaded",
            "active": True,
            "active_state": "activating",
            "pid": 123,
        },
    )

    result = dashboard_snapshot_module.post_starmie_specialist_handoff_state()

    assert result["source_specialist_id"] == "dudunsparce"
    assert result["source"].endswith("post-dudunsparce-core-v6-handoff.json")
    assert result["stage"] == "deck_agnostic_cumulative_core_corpus_pack"
    assert result["current"] == 4
    assert result["total"] == 10


def test_cycle_autorestart_reports_v6_sync_instead_of_stale_handoff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent"
    state_root = root / "outputs/state"
    log_root = root / "outputs/logs"
    state_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    transition = state_root / "specialist-transition-graph.json"
    _write_json(
        transition,
        {
            "transitions": {
                "receipt": {
                    "active_specialist": "dragapult-dusknoir",
                    "status": "failed",
                    "receipts": {
                        "specialist_transition": {
                            "status": "failed",
                            "error": (
                                "expanded cumulative-core corpus is not "
                                "atomically promoted"
                            ),
                        }
                    },
                }
            }
        },
    )
    sync = state_root / "expert-latest20-v6-strategic-sync.json"
    _write_json(
        sync,
        {
            "status": "syncing_balanced_core",
            "copied_bytes": 80,
            "source_bytes": 100,
            "percent": 80.0,
            "bandwidth_limit_kib_per_second": 8000,
        },
    )
    frozen = tmp_path / "frozen.json"
    _write_json(
        frozen,
        {
            "specialists": [
                {"specialist_id": "alakazam"},
                {"specialist_id": "hops-trevenant"},
                {"specialist_id": "lucario"},
                {"specialist_id": "starmie"},
            ]
        },
    )
    roster = tmp_path / "roster.json"
    _write_json(roster, {"required_specialist_count": 18})
    prestage = state_root / "next-specialist-prestage-v1.json"
    _write_json(
        prestage,
        {
            "status": "blocked",
            # Current receipts use the compact scalar identity. The dashboard
            # also accepts the older object-shaped identity.
            "selected_specialist": "dudunsparce",
        },
    )
    monkeypatch.setattr(dashboard_snapshot_module, "ROOT", root)
    monkeypatch.setattr(
        dashboard_snapshot_module, "SPECIALIST_TRANSITION_GRAPH_STATE", transition
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "V6_STRATEGIC_SPECIALIST_SYNC_STATE", sync
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "V6_STRATEGIC_SPECIALIST_CURRENT",
        root / "data/bootstrap/current-specialist-latest20-v6-strategic",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "FROZEN_SPECIALIST_REGISTRY", frozen
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "CANONICAL_MATCHUP_ADAPTER_ROSTER", roster
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "NEXT_SPECIALIST_PRESTAGE_STATE", prestage
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "SPECIALIST_CYCLE_HANDOFF_LOG",
        log_root / "specialist-transition-graph.log",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda name, **_kwargs: {
            "name": name,
            "load_state": "loaded",
            "active": False,
            "active_state": "activating",
            "sub_state": "auto-restart",
            "pid": 0,
        },
    )

    result = dashboard_snapshot_module.post_starmie_specialist_handoff_state()

    assert result["active"] is True
    assert result["phase"] == "waiting_for_v6_corpus_sync"
    assert result["stage"] == "atomic_checksum_sync_to_inzi"
    assert result["source_specialist_id"] == "dragapult-dusknoir"
    assert result["completed_specialists_after_starmie"] == 5
    assert result["next_specialist_id"] == "dudunsparce"
    assert result["current"] == 80
    assert result["total"] == 100
    assert result["percent"] == 80.0
    assert "80.0% complete" in result["latest_line"]


def test_inactive_cycle_surfaces_latest_failed_core_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent"
    state_root = root / "outputs/state"
    log_root = root / "outputs/logs"
    state_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    transition = state_root / "specialist-transition-graph.json"
    _write_json(
        transition,
        {
            "transitions": {
                "receipt": {
                    "active_specialist": "marnie-s-grimmsnarl-ex",
                    "status": "failed",
                    "receipts": {
                        "specialist_transition": {
                            "status": "failed",
                            "failed_at": "2026-07-26T16:27:01+00:00",
                            "error": (
                                "refreshed core failed established gameplay "
                                "regression"
                            ),
                        }
                    },
                }
            }
        },
    )
    _write_json(
        state_root / "post-marnie-s-grimmsnarl-ex-core-v7-handoff.json",
        {
            "phase": "core_gameplay_regression_complete",
            "source": {"specialist_id": "marnie-s-grimmsnarl-ex"},
            "core_gameplay_regression": {
                "schema": "poke_bot.multi_teacher_core_gameplay_regression/v1",
                "passed": False,
                "criteria": {
                    "all_reports_valid": True,
                    "aggregate_raw_win_rate": 0.5044642857142858,
                    "aggregate_raw_win_rate_minimum": 0.4,
                    "per_teacher_raw_win_rate_minimum": 0.35,
                },
                "results": [
                    {
                        "specialist_id": "alakazam",
                        "report": {"games": 80, "wr": 0.725},
                    },
                    {
                        "specialist_id": "marnie-s-grimmsnarl-ex",
                        "report": {"games": 80, "wr": 0.325},
                    },
                ],
            },
        },
    )
    _write_json(
        state_root
        / "post-marnie-s-grimmsnarl-ex-cumulative-core-v7-handoff.json",
        {"core_refresh": {"max_epochs": 25}},
    )
    frozen = tmp_path / "frozen.json"
    _write_json(frozen, {"specialists": []})
    roster = tmp_path / "roster.json"
    _write_json(roster, {"required_specialist_count": 18})
    monkeypatch.setattr(dashboard_snapshot_module, "ROOT", root)
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "SPECIALIST_TRANSITION_GRAPH_STATE",
        transition,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "SPECIALIST_CYCLE_HANDOFF_LOG",
        log_root / "specialist-transition-graph.log",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "FROZEN_SPECIALIST_REGISTRY", frozen
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "CANONICAL_MATCHUP_ADAPTER_ROSTER",
        roster,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda name, **_kwargs: {
            "name": name,
            "load_state": "loaded",
            "active": False,
            "active_state": "inactive",
            "sub_state": "dead",
            "pid": 0,
        },
    )

    result = dashboard_snapshot_module.post_starmie_specialist_handoff_state()

    assert result["active"] is False
    assert result["terminal_failure"] is True
    assert result["phase"] == "core_gameplay_regression_failed"
    assert result["stage"] == "deck_agnostic_cumulative_core_gate_failed"
    assert result["current"] == 160
    assert result["total"] == 160
    assert result["percent"] == 100.0
    assert result["core_gameplay_regression"]["failed_teachers"] == [
        {
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "games": 80,
            "win_rate": 0.325,
        }
    ]
    assert "aggregate 50.45%" in result["latest_line"]
    assert "marnie-s-grimmsnarl-ex 32.50%" in result["latest_line"]


def test_authoritative_training_prefers_terminal_handoff_failure() -> None:
    result = dashboard_snapshot_module.authoritative_training_state(
        {
            "run": "stale-run",
            "active": False,
            "progress": {"stage": "measure:research_controls"},
        },
        {},
        {
            "active": False,
            "terminal_failure": True,
            "source": "/state/current-handoff.json",
            "log": "/logs/current-handoff.log",
            "latest_line": "Cumulative core gameplay gate failed closed.",
            "stage": "deck_agnostic_cumulative_core_gate_failed",
            "current": 560,
            "total": 560,
            "percent": 100.0,
            "core_gameplay_regression": {"passed": False},
        },
    )

    assert result["status"] == "failed"
    assert result["mode"] == "specialist_handoff"
    assert result["phase"] == "deck_agnostic_cumulative_core_gate_failed"
    assert result["service"]["active"] is False
    assert result["terminal_failure"] is True
    assert result["core_gameplay_regression"] == {"passed": False}


def test_reconcile_preserves_inactive_terminal_handoff_failure() -> None:
    result = dashboard_snapshot_module.reconcile_current_specialist_handoff(
        {
            "available": True,
            "active": False,
            "terminal_failure": True,
            "phase": "core_gameplay_regression_failed",
            "stage": "deck_agnostic_cumulative_core_gate_failed",
            "latest_line": "Cumulative core gameplay gate failed closed.",
        },
        active_specialist="marnie-s-grimmsnarl-ex",
        program_progress={
            "completed_frozen": 7,
            "remaining_after_active": 11,
        },
        next_specialist="garchomp",
    )

    assert result["terminal_failure"] is True
    assert result["phase"] == "core_gameplay_regression_failed"
    assert result["stage"] == "deck_agnostic_cumulative_core_gate_failed"
    assert result["latest_line"] == (
        "Cumulative core gameplay gate failed closed."
    )


def test_active_handoff_inherits_canonical_next_specialist() -> None:
    result = dashboard_snapshot_module.reconcile_current_specialist_handoff(
        {
            "active": True,
            "completed_specialists_after_starmie": 5,
            "remaining_specialists_after_starmie": 13,
            "next_specialist_id": None,
        },
        active_specialist="dragapult-dusknoir",
        next_specialist="dudunsparce",
    )

    assert result["next_specialist_id"] == "dudunsparce"
    assert result["completed_specialists"] == 5
    assert result["remaining_specialists_after_active"] == 13


def test_active_handoff_supersedes_stale_protocol_next_action() -> None:
    protocol = {
        "available": True,
        "phase": "specialist_baseline_rl",
        "active_specialist": "",
        "canonical_active_specialist": "starmie",
        "shared_core_status": "ready",
        "next_action": "continue Starmie training",
        "program_progress": {
            "required_specialists_total": 18,
            "completed_frozen": 5,
        },
    }
    handoff = {
        "active": True,
        "label": "Starmie frozen → shared core v2 → specialist 4 of 22",
        "stage": "deck_agnostic_core_v2_training",
        "epoch": 12,
        "epochs_target": 25,
    }

    result = dashboard_snapshot_module.reconcile_protocol_with_active_handoff(
        protocol,
        handoff,
    )

    assert result["phase"] == "shared_core_derivation"
    assert result["active_specialist"] == ""
    assert result["shared_core_status"] == "refreshing"
    assert result["handoff_reconciled"] is True
    assert result["canonical_pointer_stale"] is True
    assert result["program_progress"]["remaining_after_active"] == 13
    assert "continue Starmie training" not in result["next_action"]
    assert "Epoch 12/25" in result["next_action"]
    assert "materialize every frozen predecessor" in result["next_action"]


def test_active_handoff_reports_lucario_bootstrap_after_core_validation() -> None:
    result = dashboard_snapshot_module.reconcile_protocol_with_active_handoff(
        {"available": True, "canonical_active_specialist": "starmie"},
        {
            "active": True,
            "label": "Starmie frozen → shared core v2 → specialist 4 of 22",
            "stage": "lucario_expert_bootstrap_training",
            "epoch": 3,
            "epochs_target": 25,
        },
    )

    assert result["phase"] == "specialist_bootstrap"
    assert result["active_specialist"] == "lucario"
    assert result["shared_core_status"] == "validated"
    assert "start Lucario curriculum RL" in result["next_action"]


def test_active_generic_handoff_names_selected_specialist_bootstrap() -> None:
    result = dashboard_snapshot_module.reconcile_protocol_with_active_handoff(
        {"available": True, "canonical_active_specialist": ""},
        {
            "active": True,
            "label": "Lucario frozen → cumulative core v3 → Dudunsparce",
            "stage": "next_specialist_expert_bootstrap_training",
            "next_specialist_id": "dudunsparce",
            "epoch": 3,
            "epochs_target": 25,
        },
    )

    assert result["phase"] == "specialist_bootstrap"
    assert result["active_specialist"] == "dudunsparce"
    assert "start Dudunsparce curriculum RL" in result["next_action"]


def test_dashboard_exposes_rare_route_boundary_preparation() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert "rare_route_preparation" in html
    assert "RARE-ROUTE EXPANSION" in html
    assert "active specialist unchanged" in html
    snapshot_source = (
        Path(__file__).resolve().parents[1] / "scripts/dashboard_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "pokebot-rare-route-assets-v37-import.service" in snapshot_source
    assert "rare-route-assets-v37-ready.json" in snapshot_source


class _DashboardFakeTensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


def _verified_expanded_head_contract() -> dict:
    state_dict = {
        f"{module}.weight": _DashboardFakeTensor(3, 4)
        for module in dashboard_snapshot_module.EXPANDED_HEAD_MODULES.values()
    }
    contract = {
        "schema": dashboard_snapshot_module.EXPANDED_HEAD_CONTRACT_SCHEMA,
        "architecture_present_heads": list(
            dashboard_snapshot_module.EXPANDED_HEAD_MODULES.values()
        ),
        "trained_heads": ["action_q_head", "action_type"],
        "gradient_enabled_heads": ["action_q_head"],
        "runtime_enabled_heads": ["action_q_head"],
        "loss_weights": {"action_q_head": 0.25},
        "heads": {
            "action_q_head": {
                "train_loss": 0.125,
                "validation_loss": 0.25,
                "labeled_rows": 80,
                "masked_rows": 20,
                "total_rows": 100,
            }
        },
        "stage": "action_heads",
        "epoch": 5,
        "epochs_total": 25,
    }
    return dashboard_snapshot_module._expanded_head_checkpoint_contract(
        state_dict,
        {"expanded_head_training": contract},
    )


class _DashboardFakeNonzero:
    def item(self) -> int:
        return 1


class _DashboardFakeFusionTensor(_DashboardFakeTensor):
    def count_nonzero(self) -> _DashboardFakeNonzero:
        return _DashboardFakeNonzero()


def test_decision_fusion_checkpoint_contract_requires_exact_all_head_inventory() -> None:
    required = list(dashboard_snapshot_module.DECISION_FUSION_REQUIRED_HEADS)
    result = dashboard_snapshot_module._decision_fusion_checkpoint_contract(
        {
            "decision_fusion.residual.2.weight": _DashboardFakeFusionTensor(4, 4),
            "decision_fusion.residual.2.bias": _DashboardFakeFusionTensor(4),
        },
        {
            "model_config": {
                "decision_fusion_enabled": True,
                "decision_fusion_runtime_enabled": True,
            },
            "provenance": {
                "decision_fusion": {
                    "schema": dashboard_snapshot_module.DECISION_FUSION_SCHEMA,
                    "required_heads": required,
                    "runtime_enabled": True,
                }
            },
        },
    )

    assert result["verified"] is True
    assert result["phase"] == "runtime_active"
    assert result["serving_eligible"] is True
    assert result["trained_nonzero"] is True
    assert result["required_head_count"] == 17


def test_decision_fusion_checkpoint_contract_fails_closed_if_successor_drops_head() -> None:
    required = list(dashboard_snapshot_module.DECISION_FUSION_REQUIRED_HEADS)
    result = dashboard_snapshot_module._decision_fusion_checkpoint_contract(
        {
            "decision_fusion.residual.2.weight": _DashboardFakeFusionTensor(4, 4),
        },
        {
            "model_config": {
                "decision_fusion_enabled": True,
                "decision_fusion_runtime_enabled": False,
            },
            "provenance": {
                "decision_fusion": {
                    "schema": dashboard_snapshot_module.DECISION_FUSION_SCHEMA,
                    "required_heads": required[:-1],
                    "runtime_enabled": False,
                }
            },
        },
    )

    assert result["verified"] is False
    assert result["phase"] == "contract_mismatch"
    assert result["serving_eligible"] is False


def test_dashboard_renders_checksum_bound_all_head_decision_path() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'id="model-fusion-status"' in html
    assert 'id="model-fusion-heads"' in html
    assert "FUSED POLICY ACTIVE" in html
    assert "TRAINING WARMUP · RUNTIME FLAT" in html
    assert "loop_activation_bound" in html
    assert "successor_activation_bound" in html


def test_live_curriculum_reconciles_stale_specialist_boundary_projection() -> None:
    result = dashboard_snapshot_module.reconcile_protocol_with_live_curriculum(
        {
            "available": True,
            "preparation": {
                "terminal_protocol_active": False,
                "current_premium_gate_games": 1750,
                "current_official_research_games": 1000,
                "current_total_evaluation_games": 2750,
            },
        },
        service={
            "active": True,
            "name": "pokebot-pure-rl-trevenant-staged.service",
            "command": (
                "python train.py --minimum-terminal-iteration 5 "
                "--iterations 16"
            ),
        },
        curriculum={
            "run": "pure_rl_marnie",
            "latest_official_heldout": {"games": 1000},
            "gate_program": {
                "active_gate_id": "frozen-specialists-r6",
                "next_gate": {
                    "evaluation": {"games_total": 2250},
                    "research_measurements": [
                        {"opponent_id": f"official-{index}", "games": 250}
                        for index in range(4)
                    ],
                },
            },
        },
    )

    prep = result["preparation"]
    assert prep["terminal_protocol_active"] is True
    assert prep["terminal_active_gate_id"] == "frozen-specialists-r6"
    assert prep["current_premium_gate_games"] == 2250
    assert prep["current_official_research_games"] == 1000
    assert prep["current_total_evaluation_games"] == 3250
    assert prep["gate_handler_minimum_completed_iteration"] == 5
    assert prep["terminal_iteration_ceiling"] == 15
    assert prep["gate_handler_source"] == "live_service_and_gate_program"


def test_successor_fusion_activation_binds_exact_bootstrap_checkpoint(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    state_root = tmp_path / "state"
    state_root.mkdir()
    receipt_path = (
        state_root
        / "marnie-s-grimmsnarl-ex-specialist-rl-activation-v6.json"
    )
    fusion = {
        "schema": dashboard_snapshot_module.DECISION_FUSION_SCHEMA,
        "runtime_enabled": True,
        "required_heads": list(
            dashboard_snapshot_module.DECISION_FUSION_REQUIRED_HEADS
        ),
    }
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_rl_activation/v2",
                "status": "ready",
                "identity": {
                    "next_specialist_bootstrap": {
                        "specialist_id": "marnie-s-grimmsnarl-ex",
                        "checkpoint_digest": digest,
                        "decision_fusion": fusion,
                    },
                    "runtime_registration": {
                        "specialist_id": "marnie-s-grimmsnarl-ex",
                        "runtime_row": {
                            "initial_checkpoint_sha256": digest,
                            "decision_fusion": {
                                **fusion,
                                "required": True,
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = dashboard_snapshot_module._successor_decision_fusion_activation(
        state_root=state_root,
        specialist_id="marnie-s-grimmsnarl-ex",
        checkpoint_digest=digest,
    )

    assert result["runtime_enabled"] is True
    assert result["training_action_eligible"] is True
    assert result["terminal_serving_eligible"] is False
    assert result["checkpoint_digest"] == digest


def test_successor_fusion_activation_rejects_wrong_checkpoint(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    (
        state_root / "garchomp-specialist-rl-activation-v7.json"
    ).write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_rl_activation/v2",
                "status": "ready",
                "identity": {
                    "next_specialist_bootstrap": {
                        "specialist_id": "garchomp",
                        "checkpoint_digest": "sha256:" + "b" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert not dashboard_snapshot_module._successor_decision_fusion_activation(
        state_root=state_root,
        specialist_id="garchomp",
        checkpoint_digest="sha256:" + "c" * 64,
    )


def test_successor_fusion_activation_accepts_committed_nonchampion_descendant(
    tmp_path: Path,
) -> None:
    seed_digest = "sha256:" + "a" * 64
    learner_digest = "sha256:" + "b" * 64
    fingerprint = "sha256:" + "c" * 64
    state_root = tmp_path / "state"
    state_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    fusion = {
        "schema": dashboard_snapshot_module.DECISION_FUSION_SCHEMA,
        "runtime_enabled": True,
        "required_heads": list(
            dashboard_snapshot_module.DECISION_FUSION_REQUIRED_HEADS
        ),
    }
    (
        state_root / "garchomp-specialist-rl-activation-v7.json"
    ).write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_rl_activation/v2",
                "status": "ready",
                "identity": {
                    "next_specialist_bootstrap": {
                        "specialist_id": "garchomp",
                        "checkpoint_digest": seed_digest,
                        "decision_fusion": fusion,
                    },
                    "runtime_registration": {
                        "specialist_id": "garchomp",
                        "runtime_row": {
                            "initial_checkpoint_sha256": seed_digest,
                            "decision_fusion": {
                                **fusion,
                                "required": True,
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "commits/iter_00000.json").write_text(
        json.dumps(
            {
                "design_fingerprint": fingerprint,
                # Continuous-learner safety carry can publish the exact fused
                # descendant while retaining the prior heldout champion.
                "champion": {"digest": seed_digest},
                "history": [
                    {
                        "completed": True,
                        "candidate": {"digest": learner_digest},
                        "learner_before": {"digest": seed_digest},
                        "learner_after": {"digest": learner_digest},
                        "next_collection_publish": {
                            "digest": learner_digest,
                            "local_ok": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = dashboard_snapshot_module._successor_decision_fusion_activation(
        state_root=state_root,
        specialist_id="garchomp",
        checkpoint_digest=learner_digest,
        run_dir=run_dir,
        design_fingerprint=fingerprint,
        initial_checkpoint_digest=seed_digest,
    )

    assert result["runtime_enabled"] is True
    assert result["checkpoint_digest"] == learner_digest
    assert result["bootstrap_checkpoint_digest"] == seed_digest
    assert result["activation_scope"] == "successor_committed_descendant"
    assert result["lineage_commit"].endswith("iter_00000.json")


def test_successor_initial_learner_digest_precedes_materialized_seed() -> None:
    bootstrap_digest = "sha256:" + "a" * 64
    seed_digest = "sha256:" + "b" * 64

    assert (
        dashboard_snapshot_module._initial_learner_checkpoint_digest(
            {},
            {
                "initial_learner_checkpoint": {
                    "digest": bootstrap_digest,
                },
                "checkpoint_digest": seed_digest,
            },
        )
        == bootstrap_digest
    )


def test_dashboard_runtime_root_follows_canonical_selector(tmp_path: Path) -> None:
    runtime_root = tmp_path / "active-runtime"
    selector = tmp_path / "specialist_runtime.env"
    selector.write_text(
        "\n".join(
            [
                "POKEBOT_ACTIVE_SPECIALIST=rockets-mewtwo",
                f"POKEBOT_SPECIALIST_RUNTIME_ROOT={runtime_root}",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        dashboard_snapshot_module._selected_specialist_runtime_root(
            selector,
            tmp_path / "stale-fallback",
        )
        == runtime_root
    )


def test_successor_fusion_activation_rejects_descendant_design_mismatch(
    tmp_path: Path,
) -> None:
    seed_digest = "sha256:" + "a" * 64
    learner_digest = "sha256:" + "b" * 64
    state_root = tmp_path / "state"
    state_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    fusion = {
        "schema": dashboard_snapshot_module.DECISION_FUSION_SCHEMA,
        "runtime_enabled": True,
        "required_heads": list(
            dashboard_snapshot_module.DECISION_FUSION_REQUIRED_HEADS
        ),
    }
    (state_root / "garchomp-specialist-rl-activation-v7.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_rl_activation/v2",
                "status": "ready",
                "identity": {
                    "next_specialist_bootstrap": {
                        "specialist_id": "garchomp",
                        "checkpoint_digest": seed_digest,
                        "decision_fusion": fusion,
                    },
                    "runtime_registration": {
                        "specialist_id": "garchomp",
                        "runtime_row": {
                            "initial_checkpoint_sha256": seed_digest,
                            "decision_fusion": {**fusion, "required": True},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "commits/iter_00000.json").write_text(
        json.dumps(
            {
                "completed": True,
                "design_fingerprint": "sha256:" + "d" * 64,
                "champion": {"digest": learner_digest},
                "candidate": {"digest": learner_digest},
                "learner_before": {"digest": seed_digest},
                "learner_after": {"digest": learner_digest},
                "next_collection_publish": {
                    "digest": learner_digest,
                    "local_ok": True,
                },
            }
        ),
        encoding="utf-8",
    )

    assert not dashboard_snapshot_module._successor_decision_fusion_activation(
        state_root=state_root,
        specialist_id="garchomp",
        checkpoint_digest=learner_digest,
        run_dir=run_dir,
        design_fingerprint="sha256:" + "c" * 64,
        initial_checkpoint_digest=seed_digest,
    )


def test_expanded_head_checkpoint_contract_inventories_exact_tensors_and_metrics() -> None:
    result = _verified_expanded_head_contract()

    assert result["verified"] is True
    assert result["legacy_v5"] is False
    assert result["stage"] == "action_heads"
    assert len(result["heads"]) == len(
        dashboard_snapshot_module.EXPANDED_HEAD_MODULES
    )
    action_q = next(row for row in result["heads"] if row["id"] == "action_q")
    assert action_q["module"] == "action_q_head"
    assert action_q["present"] is True
    assert action_q["declared_present"] is True
    assert action_q["trained"] is True
    assert action_q["gradient_enabled"] is True
    assert action_q["runtime_enabled"] is True
    assert action_q["parameter_count"] == 12
    assert action_q["loss_weight"] == pytest.approx(0.25)
    assert action_q["train_loss"] == pytest.approx(0.125)
    assert action_q["validation_loss"] == pytest.approx(0.25)
    assert action_q["labeled_rows"] == 80
    assert action_q["masked_rows"] == 20
    assert action_q["total_rows"] == 100
    assert action_q["coverage"] == pytest.approx(0.8)


def test_expanded_head_checkpoint_contract_fails_closed_on_metadata_tensor_drift() -> None:
    result = dashboard_snapshot_module._expanded_head_checkpoint_contract(
        {"action_q_head.weight": _DashboardFakeTensor(2, 2)},
        {
            "expanded_head_training": {
                "schema": (
                    dashboard_snapshot_module.EXPANDED_HEAD_CONTRACT_SCHEMA
                ),
                "architecture_present_heads": ["action_type_head"],
                "heads": {"invented_head": {"trained": True}},
            }
        },
    )

    assert result["verified"] is False
    assert result["missing_tensor_heads"] == ["action_type"]
    assert result["undeclared_tensor_heads"] == ["action_q"]
    assert result["unknown_declared_heads"] == ["invented_head"]
    assert all(row["contract_valid"] is False for row in result["heads"])


def test_expanded_head_checkpoint_contract_keeps_legacy_v5_valid() -> None:
    result = dashboard_snapshot_module._expanded_head_checkpoint_contract(
        {"policy_head.weight": _DashboardFakeTensor(2, 2)},
        {},
    )

    assert result["available"] is False
    assert result["verified"] is True
    assert result["legacy_v5"] is True
    assert result["heads"] == []


def test_live_model_exposes_only_checksum_bound_expanded_heads() -> None:
    checkpoint = "/run/iter_00012.pt"
    digest = "sha256:" + "c" * 64
    expanded = _verified_expanded_head_contract()
    model = learner_model_state(
        {
            "run_name": "expanded-head-test",
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                }
            },
        },
        {"learner": {"path": checkpoint, "digest": digest}},
        checkpoint_structure={
            "verified": True,
            "checkpoint": checkpoint,
            "checkpoint_digest": digest,
            "model_parameters": 2_000_000,
            "expanded_head_training": expanded,
        },
    )

    assert model["expanded_head_training"]["verified"] is True
    assert model["heads"]["action_q"]["expanded"] is True
    assert model["heads"]["action_q"]["scope"] == (
        "active_committed_checkpoint"
    )
    assert model["heads"]["action_q"]["trained"] is True
    assert model["heads"]["action_q"]["runtime_enabled"] is True


def test_live_model_marks_every_fusion_input_as_used_in_decisions() -> None:
    checkpoint = "/run/iter_00012.pt"
    digest = "sha256:" + "c" * 64
    expanded = _verified_expanded_head_contract()
    required = list(dashboard_snapshot_module.DECISION_FUSION_REQUIRED_HEADS)
    model = learner_model_state(
        {
            "run_name": "expanded-head-fusion-test",
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                }
            },
        },
        {"learner": {"path": checkpoint, "digest": digest}},
        checkpoint_structure={
            "verified": True,
            "checkpoint": checkpoint,
            "checkpoint_digest": digest,
            "model_parameters": 2_000_000,
            "expanded_head_training": expanded,
            "decision_fusion": {
                "verified": True,
                "runtime_enabled": True,
                "serving_eligible": True,
                "required_heads": required,
            },
        },
    )
    model["decision_fusion"]["activation_bound"] = True

    # The helper receives activation receipts separately in production.  This
    # unit-level fixture verifies the per-head projection using an exact bound
    # fusion contract on a second call.
    structure = dict(model["checkpoint_structure"])
    structure["decision_fusion"] = {
        **structure["decision_fusion"],
        "activation_bound": True,
    }
    model = learner_model_state(
        {
            "run_name": "expanded-head-fusion-test",
            "design_contract": {
                "learner": {
                    "profile": {"d_model": 96, "temporal_layers": 1},
                }
            },
        },
        {
            "learner": {"path": checkpoint, "digest": digest},
            "decision_fusion_activation": {
                "schema": "poke_bot.causal_decision_fusion_activation/v1",
                "learner_digest": digest,
                "runtime_enabled": True,
            },
        },
        checkpoint_structure=structure,
    )

    assert model["decision_fusion"]["activation_bound"] is True
    assert all(
        model["heads"][head_id]["used_in_decisions"] is True
        for head_id in (
            "value",
            "archetype",
            "opponent_hand",
            "opponent_remainder",
            "lethal_threat",
            "prize_race",
            "action_q",
            "action_type",
            "action_target",
            "action_resource",
            "action_utility",
            "tactical_outcome",
            "opponent_response",
            "resource_forecast",
            "game_phase",
            "outcome_distribution",
            "remaining_turns",
        )
    )


def test_staged_head_schedule_never_becomes_an_active_runtime_claim() -> None:
    result = dashboard_snapshot_module.staged_expanded_head_training_state(
        {},
        {
            "next_specialist": {"id": "dudunsparce"},
            "training": {
                "expanded_head_training": {
                    "schedule_version": "expanded-heads-v1",
                }
            },
        },
    )

    assert result["scope"] == "staged_next_specialist"
    assert result["specialist_id"] == "dudunsparce"
    assert result["available"] is True
    assert result["verified"] is False
    assert result["checkpoint_pending"] is True
    assert result["heads"] == []


def test_transition_heads_prefer_checksum_bound_active_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "core"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoints" / "epoch_03.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    digest = "sha256:" + "a" * 64
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "training",
                "history": [
                    {
                        "epoch": 3,
                        "checkpoint": str(checkpoint),
                        "checkpoint_digest": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "_file_sha256_matches",
        lambda *_args, **_kwargs: True,
    )
    expanded = _verified_expanded_head_contract()
    expanded["schema"] = "poke_bot.expanded_head_training/v1"
    expanded["architecture_present_heads"] = [
        str(value["id"]) for value in expanded["heads"]
    ]
    expanded["heads"] = {
        str(value["id"]): {**value, "present": True}
        for value in expanded["heads"]
    }
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "training",
                "history": [
                    {
                        "epoch": 3,
                        "checkpoint": str(checkpoint),
                        "checkpoint_digest": digest,
                        "expanded_head_training": expanded,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = dashboard_snapshot_module.staged_expanded_head_training_state(
        {},
        {
            "next_specialist": {"id": "dudunsparce"},
            "training": {"expanded_head_training": {"future": True}},
        },
        cumulative_core_contract={
            "core_refresh": {
                "run_dir": str(run_dir),
                "max_epochs": 25,
            }
        },
    )

    assert result["scope"] == "active_cumulative_core_refresh"
    assert result["specialist_id"] == "deck-agnostic-core"
    assert result["verified"] is True
    assert result["checkpoint_pending"] is False
    assert result["checkpoint"] == str(checkpoint)
    assert result["checkpoint_digest"] == digest
    assert result["epoch"] == 3
    assert result["epochs_target"] == 25
    assert result["source"] == str(run_dir / "state.json")


def test_live_specialist_bootstrap_supersedes_completed_core_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_dir = tmp_path / "core"
    specialist_dir = tmp_path / "grimmsnarl"
    core_dir.mkdir()
    specialist_dir.mkdir()
    core_checkpoint = core_dir / "epoch_25.pt"
    specialist_checkpoint = specialist_dir / "epoch_06.pt"
    core_checkpoint.write_bytes(b"core")
    specialist_checkpoint.write_bytes(b"specialist")
    core_digest = "sha256:" + "a" * 64
    specialist_digest = "sha256:" + "b" * 64
    (core_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "history": [
                    {
                        "epoch": 25,
                        "checkpoint": str(core_checkpoint),
                        "checkpoint_digest": core_digest,
                        "expanded_head_training": {
                            "schema": (
                                dashboard_snapshot_module
                                .EXPANDED_HEAD_CONTRACT_SCHEMA
                            ),
                            "architecture_present_heads": ["action_q"],
                            "heads": {
                                "action_q": {
                                    "present": True,
                                    "trained": True,
                                }
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (specialist_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "training",
                "history": [
                    {
                        "epoch": 6,
                        "checkpoint": str(specialist_checkpoint),
                        "checkpoint_digest": specialist_digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "_file_sha256_matches",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "checkpoint_structure_telemetry",
        lambda path, digest, **_kwargs: {
            "expanded_head_training": {
                "schema": (
                    dashboard_snapshot_module.EXPANDED_HEAD_CONTRACT_SCHEMA
                ),
                "verified": True,
                "epoch": 6,
                "heads": {
                    "action_q": {"trained": True},
                    "tactical_outcome": {"trained": True},
                },
            }
        },
    )

    result = dashboard_snapshot_module.staged_expanded_head_training_state(
        {"phase": "next_specialist_selected"},
        {
            "next_specialist": {
                "id": "marnie-s-grimmsnarl-ex",
                "run_dir": str(specialist_dir),
            },
            "training": {"expanded_head_training": {"future": True}},
        },
        cumulative_core_contract={
            "core_refresh": {
                "run_dir": str(core_dir),
                "max_epochs": 25,
            }
        },
    )

    assert result["scope"] == "staged_next_specialist"
    assert result["specialist_id"] == "marnie-s-grimmsnarl-ex"
    assert result["checkpoint"] == str(specialist_checkpoint)
    assert result["checkpoint_digest"] == specialist_digest
    assert result["epoch"] == 6
    assert set(result["heads"]) == {"action_q", "tactical_outcome"}


def test_dashboard_integrity_allows_legacy_v5_and_rejects_expanded_head_drift() -> None:
    checkpoint = "/run/iter_00012.pt"
    digest = "sha256:" + "d" * 64
    payload = {
        "dashboard_sampled_at": time.time(),
        "model": {
            "active_checkpoint": checkpoint,
            "active_checkpoint_digest": digest,
            "checkpoint_structure": {
                "verified": True,
                "checkpoint": checkpoint,
                "checkpoint_digest": digest,
                "expanded_head_training": {
                    "available": False,
                    "verified": True,
                    "legacy_v5": True,
                    "actual_tensor_heads": [],
                },
            },
        },
    }
    SnapshotCache._annotate_source_integrity(payload)
    legacy = payload["source_integrity"]["rows"]["expanded_heads"]
    assert legacy["required"] is False
    assert legacy["current"] is True
    assert (
        payload["source_integrity"]["rows"]["model"]["checks"][
            "legacy_v5_allowed"
        ]
        is True
    )

    payload["model"]["checkpoint_structure"].update(
        {
            "verified": False,
            "expanded_head_training": {
                "available": True,
                "verified": False,
                "legacy_v5": False,
                "actual_tensor_heads": ["action_q"],
                "reason": "metadata/tensor mismatch",
            },
        }
    )
    SnapshotCache._annotate_source_integrity(payload)
    expanded = payload["source_integrity"]["rows"]["expanded_heads"]
    assert expanded["required"] is True
    assert expanded["current"] is False
    assert "expanded_heads" in payload["source_integrity"]["failed"]


def test_dashboard_renders_dynamic_active_and_staged_head_observability() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'id="model-expanded-heads"' in html
    assert 'id="model-staged-expanded-heads"' in html
    assert "model.expanded_head_training||{}" in html
    assert "model.staged_expanded_head_training||{}" in html
    assert "CONTRACT MISMATCH · FAIL CLOSED" in html
    assert "TRAINED · SHADOW" in html
    assert "STAGED ONLY · RUNTIME FLAG RECORDED" in html
    assert "coverage ${coverage}" in html
    assert "masked ${masked===null?'—'" in html


def test_dashboard_renders_owner_pinned_post_spidops_goal_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    yaml = pytest.importorskip("yaml")
    state = yaml.safe_load(
        (root / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    priority = state["training_priority"]
    active_id = state["current"]["active_specialist"]
    strict_ids = priority["strict_post_spidops_prefix"]["ids"]
    ordered_ids = priority["ordered_unfinished_ids_after_active"]
    removed_ids = set(priority["owner_removal"]["specialist_ids"])
    required_ids = {row["id"] for row in state["specialists"]}
    allowed_statuses = set(state["allowed_status_values"]["specialist"])

    assert active_id == "team-rockets-spidops"
    assert all(
        row["status"] in allowed_statuses for row in state["specialists"]
    )
    assert strict_ids == [
        "hammer-pult",
        "teal-mask-ogerpon-ex",
        "archaludon-ex",
    ]
    assert ordered_ids[:3] == strict_ids
    assert removed_ids == {
        "dragapult-blaziken",
        "dragapult-dudunsparce",
    }
    assert removed_ids.isdisjoint(required_ids)
    assert removed_ids.isdisjoint(ordered_ids)

    protocol = specialist_protocol_state(root / "state/specialists.yaml")
    assert protocol["available"] is True, protocol.get("reason")
    assert protocol["training_priority"]["strict_post_spidops_prefix"][
        "ids"
    ] == strict_ids
    assert set(
        protocol["training_priority"]["owner_removal"]["specialist_ids"]
    ) == removed_ids

    display_order = list(
        dict.fromkeys(
            specialist_id
            for specialist_id in [active_id, *strict_ids, *ordered_ids]
            if specialist_id and specialist_id not in removed_ids
        )
    )
    assert display_order[:4] == [
        "team-rockets-spidops",
        "hammer-pult",
        "teal-mask-ogerpon-ex",
        "archaludon-ex",
    ]
    assert removed_ids.isdisjoint(display_order)

    snapshot_source = (
        root / "scripts/dashboard_snapshot.py"
    ).read_text(encoding="utf-8")
    html = (root / "dashboard/lan/index.html").read_text(encoding="utf-8")
    assert (
        '"strict_post_spidops_prefix": strict_prefix_contract'
        in snapshot_source
    )
    assert '"owner_removal": owner_removal_contract' in snapshot_source
    assert 'id="protocol-goals"' in html
    assert 'id="protocol-removed"' in html
    assert (
        "goalOrder=[protocol.active_specialist,...strictGoalIds,"
        "...(protocolPriority.ordered_unfinished_ids_after_active||[])]"
        in html
    )
    assert "!removedGoalSet.has(id)" in html
    assert "STRICT POST-SPIDOPS PREFIX" in html
    assert "REMOVED FROM REQUIRED GOALS" in html
