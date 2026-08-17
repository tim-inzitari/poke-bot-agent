"""Regression coverage for the live LAN training dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import time
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
    canonical_next_prestage_overlay,
    checkpoint_parameter_telemetry,
    committed_official_heldout_state,
    competition_gate_program_state,
    effective_design_contract_for_run,
    expert_rehearsal_state,
    learner_model_state,
    latest_committed_active_gate_result,
    latest_committed_official_heldout_state,
    latest_committed_research_control_result,
    iteration_timing_state,
    matchup_runtime_collection_state,
    parse_curriculum_progress,
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_next_prestage_separates_ready_corpus_from_router_block() -> None:
    overlay = canonical_next_prestage_overlay(
        {
            "current": {
                "next_successor_prestage": {
                    "specialist_id": "teal-mask-ogerpon-ex",
                    "status": "blocked_validated_causal_runtime_route_missing",
                    "representative_ready": True,
                    "pre_stage_ready": False,
                    "pre_stage_receipt": "/state/prestage.json",
                    "blockers": ["protocol_valid_expert_corpus_not_ready"],
                    "corpus": {
                        "status": "ready_checksum_validated_imported",
                        "records": 1135,
                        "decisions": 76226,
                        "guide_rows": 6814,
                        "protected_pointer": "/data/teal/PROTECTED_EXPERT_CORPUS.json",
                    },
                    "runtime_route": {
                        "blocker": "validated_causal_runtime_route_missing"
                    },
                }
            }
        }
    )

    assert overlay["intended_specialist"] == "teal-mask-ogerpon-ex"
    assert overlay["expert_corpus_ready"] is True
    assert overlay["expert_records"] == 1135
    assert overlay["expert_decisions"] == 76226
    assert overlay["representative_ready"] is True
    assert overlay["blocks_v6_handoff"] is True
    assert overlay["blocker"] == "validated_causal_runtime_route_missing"


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
    assert next(
        row for row in runtime_state["specialists"] if row["id"] == "hammer-pult"
    )["active"] is True
    assert next(
        row for row in runtime_state["specialists"] if row["id"] == "alakazam"
    )["active"] is False

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
            }
        )
    _write_json(
        run_dir / "commits/iter_00002.json",
        {"history": history},
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
    assert overlay["premium_holdout"]["games"] == 2250
    assert overlay["official_research"]["games"] == 1000
    assert overlay["checkpoint_digest"] == digest


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


def test_active_gate_is_base_eight_plus_s_plus_and_original_four_are_research_only() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "ops/alakazam_gate_program_v1.json").read_text(encoding="utf-8")
    )

    assert contract["active_gate_id"] == (
        "alakazam-strong-public-roster-lc55-v2+frozen-specialists-r1"
    )
    semantics = contract["active_gate_semantics"]
    assert semantics["gate_roster_size"] == 9
    assert semantics["games_per_opponent"] == 250
    assert semantics["gate_games_total"] == 2250
    assert semantics["frozen_specialist_tier"] == "S+"
    assert semantics["original_four_role"] == "research_control_only"
    assert semantics["original_four_gate_weight"] == 0.0
    active = contract["next_gate"]
    assert active["id"] == contract["active_gate_id"]
    assert len(active["roster"]) == 9
    assert active["evaluation"]["games_total"] == 2250
    assert active["roster"][-1]["tier"] == "S+"
    assert active["roster"][-1]["frozen_specialist"] is True
    assert all(row["gate_weight"] == 0.0 for row in active["research_measurements"])
    assert all(
        row["included_in_gate_pass"] is False
        for row in active["research_measurements"]
    )


def test_dashboard_has_one_committed_holdout_percent_and_specialist_mix_panel() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert 'data-card="outcomes"' in html
    assert "Last iteration's holdout results" in html
    assert 'id="last-holdout-percent"' in html
    assert "latestAttemptAvailable?nextExact:null" in html
    assert "NO COMMITTED RESULT" in html
    assert "immutable active-gate result" in html
    assert 'id="specialist-mix-summary"' in html
    assert 'id="specialist-mix-rows"' in html
    assert "No frozen specialist-model opponent was recorded" in html
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
    assert state["games"] == 18
    assert state["roster_size"] == 9
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
        "games": 90,
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
    assert state["active_gate_semantics"]["gate_games_total"] == 2250
    assert next_gate["status"] == "queued"
    assert len(next_gate["roster"]) == 9
    assert len({row["content_digest"] for row in next_gate["roster"]}) == 9
    assert all(row["archetype_label"] for row in next_gate["roster"])
    assert next_gate["diagnostic"]["available"] is True
    assert next_gate["diagnostic"]["games"] == 90
    assert next_gate["diagnostic"]["roster_coverage"] == 1.0
    assert next_gate["exact_result_available"] is False
    assert next_gate["research_measurements_valid"] is True
    assert len(next_gate["research_measurements"]) == 4
    assert sum(row["games"] for row in next_gate["research_measurements"]) == 1000
    assert all(row["gate_weight"] == 0 for row in next_gate["research_measurements"])
    assert all(row["archetype_label"] for row in next_gate["research_measurements"])

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


def test_active_gate_runtime_keeps_2250_target_during_research_controls(
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
    assert runtime["current"] == 2250
    assert runtime["total"] == 2250
    assert runtime["percent"] == 100.0
    assert runtime["roster_size"] == 9
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
            "total": 2250,
            "percent": 27.7777777778,
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
    assert runtime["total"] == 2250
    assert runtime["percent"] == pytest.approx(27.7777777778)
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
    assert runtime["total"] == 2250
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
    assert state["next_gate"]["exact_result"]["games"] == 2250

    active_only_pass = json.loads(result_path.read_text())
    active_only_pass["passed"] = True
    active_only_pass["checks"] = {
        "audit": True,
        "skill_weighted_win_rate": True,
        "skill_weighted_confidence_lower": True,
        "s_tier_mean_floor": True,
        "individual_opponent_floor": True,
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
    assert (
        "${row.seat0??activeGateSeat0} candidate-first + "
        "${row.seat1??activeGateSeat1} candidate-second"
    ) in html
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
    with (root / ".staging/com.pokebot.remote-worker-8766.plist").open("rb") as fh:
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
        "/home/pokebot/poke-bot-agent/outputs/pure_rl/"
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
        'elmo:8765=192,bert.local:8766=64"'
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
            "schema": "poke_bot.matchup_adapter_rehearsal_authorization/v1",
            "first_eligible_iteration": 6,
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
    assert "OFF · DEPLOYED DORMANT" in html
    assert "ROUTER ACTIVE · ZERO OUTPUT" in html
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
    assert "0 / 20 · 0 / 20" in html
    assert "Current parallel work" in html


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
        "depths={'elmo:8765': 376} "
        "high_water={'elmo:8765': 376} "
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
        "depths={'elmo:8765': 376} "
        "high_water={'elmo:8765': 376} "
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
    compose = (
        root / ".staging/elmo-docker-compose.production.yml"
    ).read_text(encoding="utf-8")
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
        )
    )

    assert result["phase"] == "waiting_for_active_specialist_gate"
    assert result["label"] == "Starmie → next unfinished specialist"
    assert "19 specialists remain" in result["latest_line"]
    assert result["historical_source_suppressed"] is True


def test_live_post_starmie_handoff_reports_remaining_program(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = tmp_path / "post-starmie.json"
    log = tmp_path / "post-starmie.log"
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
    monkeypatch.setattr(
        dashboard_snapshot_module, "POST_STARMIE_HANDOFF_STATE", state
    )
    monkeypatch.setattr(
        dashboard_snapshot_module, "POST_STARMIE_HANDOFF_LOG", log
    )
    monkeypatch.setattr(
        dashboard_snapshot_module,
        "unit_state",
        lambda *_args, **_kwargs: {
            "load_state": "loaded",
            "active": True,
            "active_state": "activating",
            "pid": 123,
            "memory_bytes": 456,
        },
    )

    result = dashboard_snapshot_module.post_starmie_specialist_handoff_state()

    assert result["active"] is True
    assert result["stage"] == "deck_agnostic_core_v2_corpus_pack"
    assert result["current"] == 16547
    assert result["total"] == 33095
    # The training plan is owned by state/specialists.yaml, not by the
    # independently retained matchup-route roster.  Ten specialists are
    # already frozen in the canonical state and seven of the seventeen
    # required targets remain.
    assert result["remaining_specialists_after_starmie"] == 7
    assert result["program_complete"] is False
    assert result["population_transition_ready"] is False


def test_active_handoff_supersedes_stale_protocol_next_action() -> None:
    protocol = {
        "available": True,
        "phase": "specialist_baseline_rl",
        "active_specialist": "",
        "canonical_active_specialist": "starmie",
        "shared_core_status": "ready",
        "next_action": "continue Starmie training",
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
    assert "continue Starmie training" not in result["next_action"]
    assert "Epoch 12/25" in result["next_action"]
    assert "materialize every frozen predecessor" in result["next_action"]


def test_dashboard_exposes_rare_route_boundary_preparation() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")

    assert "rare_route_preparation" in html
    assert "RARE-ROUTE EXPANSION" in html
    assert "live Starmie unchanged" in html
    snapshot_source = (
        Path(__file__).resolve().parents[1] / "scripts/dashboard_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "pokebot-rare-route-assets-v35-import.service" in snapshot_source
    assert "rare-route-assets-v35-ready.json" in snapshot_source


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

    assert active_id == "hammer-pult"
    assert all(
        row["status"] in allowed_statuses for row in state["specialists"]
    )
    assert strict_ids == [
        "hammer-pult",
        "teal-mask-ogerpon-ex",
        "archaludon-ex",
    ]
    assert ordered_ids[:2] == strict_ids[1:]
    assert removed_ids == {
        "dragapult-blaziken",
        "dragapult-dudunsparce",
        "walrein",
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
    assert display_order[:3] == [
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


def test_future_guide_curriculum_and_bounded_head_routes_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(dashboard_snapshot_module, "ROOT", root)
    yaml = pytest.importorskip("yaml")
    projection = json.loads(
        (root / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        "scales_guide_gradient_contribution_to_shared_policy_learning"
        not in json.dumps(projection)
    )
    assert "shadow_unfused" not in json.dumps(projection)
    protocol_source = (root / "config/rl_protocol.yaml").read_text(
        encoding="utf-8"
    )
    assert "shadow_unfused" not in protocol_source
    protocol = yaml.safe_load(protocol_source)

    scope = projection["current_owner_overrides"][
        "future_guide_strategic_branch_scope"
    ]
    assert scope["guide_curriculum_revision"] == 51
    assert scope["strategic_branch_scope_revision"] == 56
    assert scope["head_action_scope_revision"] == 56
    assert scope["prospective_effective_specialist"] == "archaludon-ex"
    assert scope["training_target_mode"] == (
        "bounded_strategic_head_curriculum"
    )
    assert scope["direct_policy_cross_entropy_allowed"] is False
    assert scope["guide_runtime_input_allowed"] is False
    assert scope["guide_action_selection_allowed"] is False
    assert scope["replace_observed_outcome_targets_allowed"] is False
    assert scope["fused_policy_learning_authority"] == (
        "realized_outcomes_and_win_objectives"
    )
    assert scope["all_future_heads_must_influence_actions"] is True
    assert scope["allowed_fusion_roles"] == ["fused_input"]
    assert scope["required_computation_role"] == "independent_head"
    assert scope["required_action_influence"] == (
        "bounded_option_conditioned_route"
    )
    assert scope["decision_fusion_schema"] == "option_conditioned_per_head/v2"
    assert scope["action_route_granularity"] == (
        "one_distinct_route_per_learned_decision_head"
    )
    assert scope["parent_v1_fusion_residual_preserved"] is True
    assert scope["route_aggregation"] == "fixed_mean"
    assert scope["aggregate_route_delta_logit_cap"] == 1.0
    assert scope["route_final_projection_initialization"] == "exact_zero"
    assert scope["existing_learned_decision_source_count"] == 17
    assert scope["canonical_learned_decision_source_count_with_setup"] == 18
    assert scope["guide_is_only_action_route_exception"] is True
    assert (
        scope[
            "independent_means_pre_fusion_computation_not_action_isolation"
        ]
        is True
    )
    assert scope["direct_action_selection_authority"] is False
    assert scope["fusion_selects_action"] is True
    assert scope["materially_influences_fused_logits"] is True
    assert scope["runtime_enabled"] is False
    assert scope["runtime_activation_requirement"] == (
        "receipt_backed_validation"
    )
    projected_setup = scope["setup_board_outcome_head"]
    assert projected_setup["owner_decision_revision"] == 56
    assert projected_setup["computation_role"] == "independent_head"
    assert projected_setup["fusion_role"] == "fused_input"
    assert projected_setup["action_influence"] == (
        "bounded_option_conditioned_route"
    )

    guide = protocol["specialist_training"]["current_deck_guide"]
    modes = guide["training_target_modes"]
    assert modes["legacy_started_runs"] == {
        "mode": "confidence_weighted_policy_cross_entropy",
        "immutable_scope": "completed_frozen_and_already_started_specialists",
        "active_teal_remains_legacy": True,
    }
    future = modes["future_specialists"]
    assert future["owner_decision_revision"] == 51
    assert future["effective_from_specialist"] == "archaludon-ex"
    assert future["mode"] == "bounded_strategic_head_curriculum"
    assert future["direct_policy_cross_entropy_allowed"] is False
    assert future["fused_policy_learning_authority"] == (
        "realized_outcomes_and_win_objectives"
    )
    branch = future["strategic_branch_scope"]
    assert branch["owner_decision_revision"] == 56
    assert branch["allowed_fusion_roles"] == ["fused_input"]
    assert branch["required_computation_role"] == "independent_head"
    assert branch["required_action_influence"] == (
        "bounded_option_conditioned_route"
    )
    assert branch["decision_fusion_schema"] == "option_conditioned_per_head/v2"
    assert branch["action_route_granularity"] == (
        "one_distinct_route_per_learned_decision_head"
    )
    assert branch["parent_v1_fusion_residual_preserved"] is True
    assert branch["route_aggregation"] == "fixed_mean"
    assert branch["aggregate_route_delta_logit_cap"] == 1.0
    assert branch["route_final_projection_initialization"] == "exact_zero"
    assert branch["guide_is_only_action_route_exception"] is True
    assert branch["omission_from_action_score_allowed"] is False
    adaptive = guide["adaptive_annealing"]
    assert adaptive["every_head_has_bounded_option_conditioned_route"] is True
    assert "every_head_has_bounded_decision_fusion_route" not in adaptive
    setup = future["setup_board_outcome_head"]
    assert setup["owner_decision_revision"] == 56
    assert setup["computation_role"] == "independent_head"
    assert setup["fusion_role"] == "fused_input"
    assert setup["action_influence"] == (
        "bounded_option_conditioned_route"
    )
    assert setup["direct_action_selection_authority"] is False
    assert setup["runtime_activation_requires_validation_receipt"] is True

    snapshot_protocol = specialist_protocol_state(
        root / "state/specialists.yaml"
    )
    projected_modes = snapshot_protocol[
        "current_deck_guide_training_modes"
    ]
    assert projected_modes["active_started_lineage"]["mode"] == (
        "confidence_weighted_policy_cross_entropy"
    )
    assert projected_modes["future_lineage"]["mode"] == (
        "bounded_strategic_head_curriculum"
    )
    action_contract = projected_modes["future_head_action_contract"]
    assert action_contract["computation_role"] == "independent_head"
    assert action_contract["fusion_role"] == "fused_input"
    assert action_contract["action_influence"] == (
        "bounded_option_conditioned_route"
    )
    assert action_contract["preserve_v1_additive_residual"] is True
    assert action_contract["route_reduction"] == "fixed_mean"
    assert action_contract["aggregate_absolute_cap"] == 1.0
    assert action_contract["zero_safe_final_projections"] is True
    assert action_contract["direct_action_selection_authority"] is False
    assert action_contract["materially_influences_fused_logits"] is True
    assert action_contract["runtime_enabled"] is False

    html = (root / "dashboard/lan/index.html").read_text(encoding="utf-8")
    assert 'id="protocol-guide-active"' in html
    assert 'id="protocol-guide-future"' in html
    assert 'id="protocol-guide-action"' in html
    assert "bounded_option_conditioned_route" in html
    assert (
        "every routed head must materially influence fused logits"
        in html.lower()
    )
    assert "shadow_unfused" not in html
    assert "direct shared-policy teaching" in html
    assert "ramp scales guide gradients into shared policy learning" not in html
