"""Unit tests for Pure-RL auto-progress gate → phase helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.pure_rl_auto_progress as ap
import scripts.launch_pure_rl as launcher
import scripts.warm_start_pure_rl_specialist as warm_start


def test_resolve_gate_checkpoint_prefers_gate_field(tmp_path: Path) -> None:
    run = tmp_path / "core"
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    chosen = ckpt_dir / "iter_00007.pt"
    chosen.write_bytes(b"x")
    (ckpt_dir / "iter_00003.pt").write_bytes(b"y")
    gate = {
        "iteration": 7,
        "wr": 0.71,
        "checkpoint": str(chosen),
        "checkpoint_digest": ap._checkpoint_digest(chosen),
    }
    assert ap.resolve_gate_checkpoint(run, gate) == chosen


def test_resolve_gate_checkpoint_refuses_latest_fallback(tmp_path: Path) -> None:
    run = tmp_path / "core"
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    older = ckpt_dir / "iter_00001.pt"
    newer = ckpt_dir / "iter_00002.pt"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    assert ap.resolve_gate_checkpoint(run, {"wr": 0.7}) is None


def test_resolve_gate_checkpoint_refuses_digest_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "core"
    chosen = run / "checkpoints" / "iter_00002.pt"
    chosen.parent.mkdir(parents=True)
    chosen.write_bytes(b"candidate")
    gate = {
        "checkpoint": str(chosen),
        "checkpoint_digest": ap._checkpoint_digest(chosen),
    }
    chosen.write_bytes(b"replaced")
    assert ap.resolve_gate_checkpoint(run, gate) is None


def test_launcher_requires_explicit_specialist_for_auto_progress() -> None:
    assert launcher._parse_args(["--run-name", "safe-default"]).auto_progress is False
    with pytest.raises(SystemExit):
        launcher._parse_args(["--run-name", "unsafe", "--auto-progress"])
    explicit = [
        "--run-name",
        "specialist-explicit",
        "--auto-progress",
        "--specialist-archetype",
        "alakazam",
    ]
    trainer_source = (
        Path(__file__).resolve().parents[1] / "scripts/train_pure_rl.py"
    ).read_text(encoding="utf-8")
    if '"--specialist-archetype"' in trainer_source:
        assert launcher._parse_args(explicit).specialist_archetype == "alakazam"
    else:
        with pytest.raises(SystemExit):
            launcher._parse_args(explicit)


def test_launcher_preflight_isolated_from_runtime_tuning() -> None:
    clean = launcher._preflight_environment(
        {
            "PATH": "/usr/bin",
            "CG_LIB_PATH": "/runtime",
            "PURE_RL_REBALANCE_MAX_WORKERS": "48",
            "POKEBOT_LIVE_POOL_MAX_WORKERS": "48",
            "POKEBOT_REMOTE_REQUIRE_ALL": "1",
        }
    )
    assert clean == {"PATH": "/usr/bin", "CG_LIB_PATH": "/runtime"}


def test_warm_start_requires_explicit_specialist() -> None:
    with pytest.raises(SystemExit):
        warm_start._parse_args(
            ["--run-name", "unsafe", "--core-checkpoint", "seed.pt"]
        )


def test_submit_script_never_selects_latest_checkpoint() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/submit_pure_rl_greedy.sh"
    ).read_text(encoding="utf-8")
    assert "resolve_gate_checkpoint" in script
    assert "ls -1" not in script
    assert "tail -1" not in script
    assert "hammer-pult" not in script


def test_auto_progress_coordination_defaults_are_per_run() -> None:
    source = Path(ap.__file__).read_text(encoding="utf-8")
    assert 'core_dir / "auto_progress" / "state.json"' in source
    assert 'core_dir / "auto_progress" / "auto_progress.lock"' in source


def test_read_gate_and_advance_watching_to_warm_start(tmp_path: Path, monkeypatch) -> None:
    core = tmp_path / "pure_rl_core_test"
    ckpt_dir = core / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "iter_00001.pt"
    ckpt.write_bytes(b"ckpt")
    (core / "CORE_GATE_PASSED").write_text(
        json.dumps(
            {
                "iteration": 1,
                "wr": 0.72,
                "games": 200,
                "checkpoint": str(ckpt),
                "checkpoint_digest": ap._checkpoint_digest(ckpt),
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(ap, "ROOT", tmp_path)
    monkeypatch.setattr(ap, "_ACTIVE_LOG_PATH", tmp_path / "auto.log")
    monkeypatch.setattr(ap, "pure_rl_trainers_alive", lambda run_name=None: [])
    monkeypatch.setattr(
        ap,
        "wait_until_trainers_idle",
        lambda **kwargs: True,
    )

    args = ap._parse_args(
        [
            "--core-run-dir",
            str(core),
            "--state-path",
            str(state_path),
            "--dry-run",
            "--once",
            "--specialist-run-name",
            "pure_rl_hammer_test",
            "--archetype",
            "hammer-pult",
        ]
    )
    # Dry-run warm_start + launch without spawning; drive one advance from watching.
    state = {"phase": "watching_core"}
    # Patch warm_start / launch to no-op paths for dry-run already handled.
    out = ap.advance_once(args, state)
    assert out["phase"] in ("watching_specialist", "launch_specialist", "warm_start", "done")
    # With dry-run, warm_start + launch should complete through watching_specialist
    # (gate not present → stays watching_specialist) or reach watching_specialist.
    assert out.get("core_checkpoint") == str(ckpt)
    assert out.get("specialist_run_name") == "pure_rl_hammer_test"
    assert out["phase"] == "watching_specialist"


def test_advance_specialist_gate_to_done_without_submit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ap, "ROOT", tmp_path)
    monkeypatch.setattr(ap, "_ACTIVE_LOG_PATH", tmp_path / "auto.log")
    monkeypatch.setattr(ap, "pure_rl_trainers_alive", lambda run_name=None: [])
    monkeypatch.setattr(ap, "wait_until_trainers_idle", lambda **kwargs: True)

    core = tmp_path / "core_done"
    core.mkdir()
    spec = tmp_path / "outputs" / "pure_rl" / "hammer_done"
    # advance_once builds specialist path as ROOT/outputs/pure_rl/<name>
    spec.mkdir(parents=True)
    gated = spec / "checkpoints" / "iter_00003.pt"
    gated.parent.mkdir()
    gated.write_bytes(b"promoted")
    (spec / "SPECIALIST_GATE_PASSED").write_text(
        json.dumps(
            {
                "iteration": 3,
                "wr": 0.8,
                "games": 200,
                "checkpoint": str(gated),
                "checkpoint_digest": ap._checkpoint_digest(gated),
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "state2.json"
    args = ap._parse_args(
        [
            "--core-run-dir",
            str(core),
            "--state-path",
            str(state_path),
            "--dry-run",
            "--no-submit",
            "--once",
            "--specialist-run-name",
            "hammer_done",
            "--archetype",
            "hammer-pult",
        ]
    )
    state = {
        "phase": "watching_specialist",
        "specialist_run_name": "hammer_done",
        "core_run_name": core.name,
    }
    out = ap.advance_once(args, state)
    assert out["phase"] == "done"
    assert out["specialist_gate"]["wr"] == 0.8
