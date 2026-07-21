"""Regression coverage for the live LAN training dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
import plistlib
import re

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
    _run_name_from_command,
    _select_curriculum_run_dir,
    annotate_expert_optimizer_sps,
    authoritative_training_state,
    checkpoint_parameter_telemetry,
    committed_official_heldout_state,
    competition_gate_program_state,
    learner_model_state,
    iteration_timing_state,
    parse_curriculum_progress,
    replay_window_state,
    scheduler_queue_state,
    strong_public_gate_runtime_state,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_active_gate_is_eight_public_agents_and_original_four_are_research_only() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "ops/alakazam_gate_program_v1.json").read_text(encoding="utf-8")
    )

    assert contract["active_gate_id"] == "alakazam-strong-public-roster-v1"
    semantics = contract["active_gate_semantics"]
    assert semantics["gate_roster_size"] == 8
    assert semantics["games_per_opponent"] == 250
    assert semantics["gate_games_total"] == 2000
    assert semantics["original_four_role"] == "research_control_only"
    assert semantics["original_four_gate_weight"] == 0.0
    active = contract["next_gate"]
    assert active["id"] == contract["active_gate_id"]
    assert len(active["roster"]) == 8
    assert active["evaluation"]["games_total"] == 2000
    assert all(row["gate_weight"] == 0.0 for row in active["research_measurements"])
    assert all(
        row["included_in_gate_pass"] is False
        for row in active["research_measurements"]
    )
    assert COMPETITION_GATE_PROGRAM.parts[-2:] == (
        "ops",
        "alakazam_gate_program_v1.json",
    )


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
    assert "Archived original-four research controls" in html
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
    _write_json(
        registry_path,
        {
            "checkpoint_digest": digest,
            "immutable": True,
            "automatic_pruning_allowed": False,
        },
    )
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
        "games": 80,
        "iteration": 1,
        "matchups": matchups,
    }

    state = competition_gate_program_state(
        official,
        public_mix,
        contract_path=contract_path,
        registry_path=registry_path,
    )

    assert state["accepted_gate"]["accepted"] is True
    assert state["accepted_gate"]["registry_protected"] is True
    submissions = state["accepted_gate"]["submissions"]
    assert len(submissions) == 3
    assert submissions[-1]["ref"] == 54883731
    assert submissions[-1]["authorization"].startswith("unauthorized")
    next_gate = state["next_gate"]
    assert next_gate["available"] is True
    assert state["active_gate_id"] == contract["active_gate_id"]
    assert state["active_gate_semantics"]["gate_games_total"] == 2000
    assert next_gate["status"] == "queued"
    assert len(next_gate["roster"]) == 8
    assert len({row["content_digest"] for row in next_gate["roster"]}) == 8
    assert all(row["archetype_label"] for row in next_gate["roster"])
    assert next_gate["diagnostic"]["available"] is True
    assert next_gate["diagnostic"]["games"] == 80
    assert next_gate["diagnostic"]["roster_coverage"] == 1.0
    assert next_gate["exact_result_available"] is False
    assert next_gate["research_measurements_valid"] is True
    assert len(next_gate["research_measurements"]) == 4
    assert sum(row["games"] for row in next_gate["research_measurements"]) == 1000
    assert all(row["gate_weight"] == 0 for row in next_gate["research_measurements"])
    assert all(row["archetype_label"] for row in next_gate["research_measurements"])

    official["checkpoint_digest"] = "sha256:not-the-protected-model"
    mismatch = competition_gate_program_state(
        official,
        public_mix,
        contract_path=contract_path,
        registry_path=registry_path,
    )
    assert mismatch["accepted_gate"]["accepted"] is False

    invalid_contract = json.loads(contract_path.read_text())
    invalid_contract["active_gate_semantics"]["gate_games_total"] = 1000
    invalid_contract_path = tmp_path / "invalid-gate-program.json"
    _write_json(invalid_contract_path, invalid_contract)
    invalid = competition_gate_program_state(
        official,
        public_mix,
        contract_path=invalid_contract_path,
        registry_path=registry_path,
    )
    assert invalid["next_gate"]["available"] is False
    assert "active gate identity" in invalid["next_gate"]["contract_reason"]


def test_active_gate_runtime_keeps_2000_target_during_research_controls(
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
            "stage": "heldout:original_four_research",
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
    assert runtime["current"] == 2000
    assert runtime["total"] == 2000
    assert runtime["percent"] == 100.0
    assert runtime["roster_size"] == 8
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
            "total": 2000,
            "percent": 31.25,
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
    assert runtime["total"] == 2000
    assert runtime["percent"] == 31.25
    assert runtime["source"] == "main curriculum run-bound progress"


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
    assert state["next_gate"]["exact_result"]["games"] == 2000

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
    assert stale_state["next_gate"]["exact_result_available"] is False
    assert stale_state["next_gate"]["status"] == "queued"


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
    assert "researchArchetypeById" in html
    assert "Archived original-four research controls" in html
    assert "toLocaleString()+' games'" in html
    assert "nextRows=nextGate.exact_result_available?(nextExact.matchups||[]):[]" in html
    assert "exact active-gate games" in html
    assert "'NOT RUN'" in html
    assert "(nextDiag.rows||[])" not in html
    assert "acceptedGate=gateProgram.accepted_gate||{}" in html
    assert "owner decision ACCEPTED · protected registry identity reconciled" in html
    assert "Active holdout sample" in html
    assert "greedy · both seats · fixed seeds" in html
    assert "sampled temperature policy · NON-GATE" not in html
    assert "live public-mix win rate" in html
    assert "fixed disjoint seeds · checkpoint digest pinned" in html
    assert "all matchups complete · no early-stop gate" in html
    assert "excluded from every gate calculation" in html
    assert "alias duplicates excluded" in html
    assert "UNAUTHORIZED RETRY INCIDENT" in html
    assert "active gate held-out win rate" in html
    assert "activeGateAllocation=activeGateTotal.toLocaleString()+' total · '" in html
    assert "original-four research excluded" in html
    assert "requires ops/alakazam_gate_program_v1.json" in html
    assert "Number.isFinite(c.heldout_wr)" not in html
    assert "+' gate + '+researchGames" not in html
    assert "Dashboard UI v14" in html


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
        "source": str(dormant_contract),
    }
    assert model["dormant_modules"][0]["runtime_enabled"] is False


def test_model_panel_labels_current_and_staged_profiles_separately() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "dashboard/lan/index.html"
    ).read_text(encoding="utf-8")
    assert "Current production model" in html
    assert 'id="model-params-source"' in html
    assert 'id="model-training"' in html
    assert 'id="model-optimizer"' in html
    assert 'id="model-active-params"' in html
    assert 'id="model-dormant-params"' in html
    assert 'id="model-staged-params"' in html
    assert 'id="model-adapter-status"' in html
    assert "Dormant matchup adapters" in html
    assert "DORMANT MATCHUP ADAPTERS ONLY (history remains active)" in html


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
    assert "$('batch-sps').textContent=curriculumActive?(rlTraining?num(cp.sps,0):num(collectionSps,0))" in html
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
    assert "displayGateResult=!!(activeGateResultAvailable&&!gatePhaseLive&&!researchPhaseLive)" in html
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
    compose = (
        root / ".staging/elmo-docker-compose.production.yml"
    ).read_text(encoding="utf-8")
    assert 'POKEBOT_REMOTE_MAX_CONNECTIONS: "420"' in compose
    assert re.search(r"^\s*pids_limit:\s*1536\s*$", compose, re.MULTILINE)
    assert re.search(r"^\s*mem_limit:\s*64g\s*$", compose, re.MULTILINE)
    assert 'POKEBOT_REMOTE_TREE_RSS_LIMIT_GB: "30"' in compose
