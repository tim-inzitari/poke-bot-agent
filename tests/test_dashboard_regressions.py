"""Regression coverage for the live LAN training dashboard."""

from __future__ import annotations

import time
from pathlib import Path
import plistlib
import re

from dashboard.lan.server import SnapshotCache
from scripts.dashboard_snapshot import (
    _active_curriculum_services,
    _run_name_from_command,
    _select_curriculum_run_dir,
    committed_official_heldout_state,
    learner_model_state,
    parse_curriculum_progress,
)


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
        "rem=0.5/21, v=0.2]"
    )
    parsed = parse_curriculum_progress("", line, iteration_hint=3)
    assert parsed["stage"] == "train:policy"
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
    assert len(units) == 2
    assert pids == [222]
    assert run_name == "live-specialist"


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
    assert "Official heldout gate" in html
    assert "exact audit" in html
    assert "seats '+(x.seat0" in html


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
    assert payload["fleet_rates"]["ingest_gps"] == 2.90
    assert payload["fleet_rates"]["buffered_results"] == 4370
    assert payload["fleet"]["bert"]["worker"]["allocation_state"] == (
        "RESULTS BUFFERED · 4370 fleet games awaiting ingest"
    )


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
        "design_contract": {
            "learner": {
                "profile": {"d_model": 96, "temporal_layers": 0},
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
    model = learner_model_state(manifest)
    assert model["trainable_parameters"] == 1_489_505
    assert model["profile"]["temporal_layers"] == 0
    assert set(model["heads"]) == {
        "policy", "value", "archetype", "opponent_hand",
        "opponent_remainder", "lethal_threat", "prize_race",
    }
    assert all(row["enabled"] for row in model["heads"].values())
    assert model["training_targets"]["alakazam_guide"]["parameterized_head"] is False
    assert model["seed_checkpoint"] == "/tmp/alakazam.pt"


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


def test_elmo_has_bounded_task_headroom_for_constant_refill() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (
        root / ".staging/elmo-docker-compose.production.yml"
    ).read_text(encoding="utf-8")
    assert 'POKEBOT_REMOTE_MAX_CONNECTIONS: "420"' in compose
    assert re.search(r"^\s*pids_limit:\s*1536\s*$", compose, re.MULTILINE)
    assert re.search(r"^\s*mem_limit:\s*64g\s*$", compose, re.MULTILINE)
    assert 'POKEBOT_REMOTE_TREE_RSS_LIMIT_GB: "30"' in compose
