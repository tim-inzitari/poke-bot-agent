from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_managed_shadow_timer_is_bounded_and_nonblocking() -> None:
    service = (
        ROOT
        / "ops/systemd/pokebot-future-guide-weight-shadow.service"
    ).read_text()
    timer = (
        ROOT / "ops/systemd/pokebot-future-guide-weight-shadow.timer"
    ).read_text()
    assert "process_future_guide_weight_review_queue.py" in service
    assert "CPUQuota=400%" in service
    assert "MemoryMax=112G" in service
    assert "IOSchedulingClass=idle" in service
    assert "--shadow-device cpu" in service
    assert "--production-device cuda:1" in service
    assert "OnUnitInactiveSec=2min" in timer


def test_queue_processor_exactly_bypasses_active_teal(tmp_path: Path) -> None:
    selector = tmp_path / "selector.env"
    selector.write_text("POKEBOT_ACTIVE_SPECIALIST=teal-mask-ogerpon-ex\n")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "runtime_root": str(tmp_path),
                "specialists": {
                    "teal-mask-ogerpon-ex": {
                        "run_name": "teal",
                        "guide_loss_weight": 0.25,
                        "guide_weight_policy": None,
                    }
                },
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/process_future_guide_weight_review_queue.py"),
            "--selector",
            str(selector),
            "--registry",
            str(registry),
            "--baseline-manifest",
            str(tmp_path / "missing-baselines.json"),
            "--output-root",
            str(tmp_path / "output"),
            "--lock",
            str(tmp_path / "queue.lock"),
            "--training-unit",
            "not-used.service",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "ok": True,
        "specialist_id": "teal-mask-ogerpon-ex",
        "status": "not_an_eligible_future_specialist",
    }


def test_queue_processor_bypasses_crustle_persistent_guide_hold(
    tmp_path: Path,
) -> None:
    selector = tmp_path / "selector.env"
    selector.write_text(
        "\n".join(
            (
                "POKEBOT_ACTIVE_SPECIALIST=crustle",
                "POKEBOT_FUTURE_GUIDE_WEIGHT_POLICY_REVISION=44",
                "POKEBOT_GUIDE_LEARNING_SEMANTICS_REVISION=46",
            )
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "runtime_root": str(tmp_path),
                "specialists": {
                    "crustle": {
                        "run_name": "crustle",
                        "guide_loss_weight": 0.05,
                        "guide_weight_policy": {
                            "scope": "crustle_persistent_training_only",
                            "automatic_decay_allowed": False,
                        },
                    }
                },
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/process_future_guide_weight_review_queue.py"),
            "--selector",
            str(selector),
            "--registry",
            str(registry),
            "--baseline-manifest",
            str(tmp_path / "missing-baselines.json"),
            "--output-root",
            str(tmp_path / "output"),
            "--lock",
            str(tmp_path / "queue.lock"),
            "--training-unit",
            "not-used.service",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert json.loads(result.stdout) == {
        "ok": True,
        "specialist_id": "crustle",
        "status": "not_an_eligible_future_specialist",
    }


def test_queue_processor_runs_eligible_future_review_through_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import process_future_guide_weight_review_queue as queue

    selector = tmp_path / "selector.env"
    selector.write_text(
        "\n".join(
            (
                "POKEBOT_ACTIVE_SPECIALIST=archaludon-ex",
                "POKEBOT_FUTURE_GUIDE_WEIGHT_POLICY_REVISION=44",
                "POKEBOT_GUIDE_LEARNING_SEMANTICS_REVISION=46",
                "POKEBOT_GUIDE_CONSECUTIVE_NONPOSITIVE_EVALUATIONS=0",
            )
        )
        + "\n"
    )
    boundary = tmp_path / "apply_boundary.py"
    boundary.write_text("# boundary controller fixture\n")
    run_dir = tmp_path / "outputs/pure_rl/archaludon-run"
    request_dir = run_dir / "guide_weight_reviews"
    request_dir.mkdir(parents=True)
    request = request_dir / "review_00005.request.json"
    request.write_text(json.dumps({"completed_iteration": 5}))
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "runtime_root": str(tmp_path),
                "specialists": {
                    "archaludon-ex": {
                        "run_name": "archaludon-run",
                        "log": str(tmp_path / "archaludon.log"),
                        "guide_weight_policy": {
                            "scope": "future_specialist_training_runs_only",
                            "prospective_scope_revision": 44,
                            "learning_semantics_revision": 46,
                            "boundary_controller": str(boundary),
                        },
                    }
                },
            }
        )
    )

    calls: list[object] = []
    manifest = tmp_path / "shadow.manifest.json"
    training = tmp_path / "shadow.training.json"
    rows = tmp_path / "shadow.rows.json"
    schedule = tmp_path / "guide_weight_schedule.json"

    def _prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return manifest

    def _train(path, *, device):
        calls.append(("train", path, device))
        return training

    def _evaluate(path, receipt, *, workers, game_timeout_seconds):
        calls.append(
            (
                "evaluate",
                path,
                receipt,
                workers,
                game_timeout_seconds,
            )
        )
        return rows

    def _finalize(path, receipt, results):
        calls.append(("finalize", path, receipt, results))
        return tmp_path / "evidence.json", schedule

    boundary_commands: list[list[str]] = []

    def _run(command, *, check):
        assert check is True
        boundary_commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(queue, "prepare_manifest", _prepare)
    monkeypatch.setattr(queue, "run_training", _train)
    monkeypatch.setattr(queue, "run_evaluation", _evaluate)
    monkeypatch.setattr(queue, "finalize", _finalize)
    monkeypatch.setattr(queue.subprocess, "run", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ROOT / "scripts/process_future_guide_weight_review_queue.py"),
            "--selector",
            str(selector),
            "--registry",
            str(registry),
            "--baseline-manifest",
            str(tmp_path / "baselines.json"),
            "--output-root",
            str(tmp_path / "shadow-output"),
            "--lock",
            str(tmp_path / "queue.lock"),
            "--training-unit",
            "pokebot-specialist.service",
            "--workers",
            "3",
            "--game-timeout-seconds",
            "77",
        ],
    )

    assert queue.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "processed"
    assert payload["specialist_id"] == "archaludon-ex"
    assert payload["completed_iteration"] == 5
    assert [call[0] for call in calls] == [
        "prepare",
        "train",
        "evaluate",
        "finalize",
    ]
    assert calls[1] == ("train", manifest, "cpu")
    assert calls[2] == ("evaluate", manifest, training, 3, 77)
    assert calls[3] == ("finalize", manifest, training, rows)
    assert len(boundary_commands) == 1
    command = boundary_commands[0]
    assert command[:2] == [sys.executable, str(boundary.resolve())]
    assert command[command.index("--schedule") + 1] == str(schedule)
    assert command[command.index("--run-dir") + 1] == str(run_dir)
    assert command[command.index("--unit") + 1] == (
        "pokebot-specialist.service"
    )
