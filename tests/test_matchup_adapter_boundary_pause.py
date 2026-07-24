from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import pause_matchup_adapters_at_iter15 as watcher


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary(tmp_path: Path) -> tuple[Path, dict]:
    run = tmp_path / "run"
    (run / "commits").mkdir(parents=True)
    parent = tmp_path / "learner.pt"
    parent.write_bytes(b"exact iteration-15 parent")
    state = {
        "version": 2,
        "run_name": "test",
        "mode": "specialist",
        "last_completed_iteration": 15,
        "next_iteration": 16,
        "learner": {"path": str(parent.resolve()), "digest": _sha(parent)},
        "history": [{"iteration": 15, "stage_gate": {"passed": False}}],
    }
    serialized = json.dumps(state, sort_keys=True) + "\n"
    (run / "commits" / "iter_00015.json").write_text(serialized)
    (run / "loop_state.json").write_text(serialized)
    return run, state


def test_boundary_proof_requires_exact_immutable_ledger_and_parent(
    tmp_path: Path,
) -> None:
    run, state = _boundary(tmp_path)
    proof = watcher.boundary_proof(run)
    assert proof is not None
    assert proof.state == state
    assert proof.parent_checkpoint_digest == state["learner"]["digest"]

    changed = dict(state)
    changed["next_iteration"] = 17
    (run / "loop_state.json").write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="does not exactly match"):
        watcher.boundary_proof(run)


def test_boundary_proof_never_rolls_back_committed_iteration16(tmp_path: Path) -> None:
    run, state = _boundary(tmp_path)
    committed = {**state, "last_completed_iteration": 16, "next_iteration": 17}
    (run / "commits" / "iter_00016.json").write_text(json.dumps(committed))
    with pytest.raises(RuntimeError, match="already immutable"):
        watcher.boundary_proof(run)


def test_boundary_proof_retries_commit_before_loop_pointer_window(
    tmp_path: Path,
) -> None:
    run, committed = _boundary(tmp_path)
    previous = {
        **committed,
        "last_completed_iteration": 14,
        "next_iteration": 15,
        "history": [],
    }
    previous_serialized = json.dumps(previous, sort_keys=True) + "\n"
    (run / "commits" / "iter_00014.json").write_text(previous_serialized)
    (run / "loop_state.json").write_text(previous_serialized)

    assert watcher.boundary_proof(run) is None

    committed_serialized = json.dumps(committed, sort_keys=True) + "\n"
    (run / "loop_state.json").write_text(committed_serialized)
    proof = watcher.boundary_proof(run)
    assert proof is not None
    assert proof.state == committed


def test_pause_stops_then_quarantines_raced_iteration16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, state = _boundary(tmp_path)
    status = tmp_path / "status.json"
    service = {"active": "active", "pid": "123", "refuse": "no"}

    def service_value(_unit: str, key: str) -> str:
        return {
            "ActiveState": service["active"],
            "MainPID": service["pid"],
            "RefuseManualStop": service["refuse"],
        }[key]

    def stop(_unit: str) -> None:
        # Model the real race: iter16 starts after the commit is observed but
        # before systemd finishes stopping the trainer.
        (run / "shards").mkdir()
        (run / "shards" / "iter_00016.jsonl").write_text("partial\n")
        (run / "iteration_runtime.json").write_text(
            json.dumps({"iteration": 16, "phase": "collect"})
        )
        service.update(active="inactive", pid="0")

    recovered_path = run / "quarantine" / "iter_00016" / "failure.json"

    def recover(
        _run: Path,
        recovered_state: dict,
        *,
        preserve_completed_collection: bool,
    ) -> Path:
        assert recovered_state == state
        assert preserve_completed_collection is False
        source = run / "shards" / "iter_00016.jsonl"
        destination = run / "quarantine" / "iter_00016" / "shard.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        recovered_path.write_text("{}")
        return recovered_path

    trainer = SimpleNamespace(
        _load_loop_state=lambda _run: state,
        _recover_interrupted_iteration=recover,
    )
    monkeypatch.setattr(watcher, "_service_value", service_value)
    monkeypatch.setattr(watcher, "_stop_service", stop)

    proof = watcher.pause_at_boundary(
        run_dir=run,
        unit="trainer.service",
        trainer=trainer,
        status_path=status,
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert proof.state == state
    payload = json.loads(status.read_text())
    assert payload["status"] == "paused_clean_15_to_16"
    assert payload["runtime_activation_enabled"] is False
    assert payload["adapter_training_enabled"] is False
    assert not (run / "iteration_runtime.json").exists()
    assert (
        run
        / "quarantine"
        / "iter_00016"
        / "boundary_pause_iteration_runtime.json"
    ).is_file()


def test_pause_refuses_to_arm_while_manual_stop_is_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _state = _boundary(tmp_path)
    monkeypatch.setattr(
        watcher,
        "_service_value",
        lambda _unit, key: "yes" if key == "RefuseManualStop" else "active",
    )
    with pytest.raises(RuntimeError, match="RefuseManualStop=no"):
        watcher.pause_at_boundary(
            run_dir=run,
            unit="trainer.service",
            trainer=SimpleNamespace(),
            status_path=tmp_path / "status.json",
            poll_seconds=0.01,
            timeout_seconds=1.0,
        )


def test_passed_gate_marker_is_not_suppressed_before_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, state = _boundary(tmp_path)
    state["history"][-1]["stage_gate"]["passed"] = True
    serialized = json.dumps(state, sort_keys=True) + "\n"
    (run / "commits" / "iter_00015.json").write_text(serialized)
    (run / "loop_state.json").write_text(serialized)
    service = {"active": "active"}
    observed_marker_at_stop: list[bool] = []

    def service_value(_unit: str, key: str) -> str:
        if key == "RefuseManualStop":
            return "no"
        if key == "MainPID":
            return "123" if service["active"] == "active" else "0"
        return service["active"]

    sleeps = {"count": 0}

    def sleep(_seconds: float) -> None:
        sleeps["count"] += 1
        if sleeps["count"] == 1:
            (run / "SPECIALIST_GATE_PASSED").write_text("passed\n")

    def stop(_unit: str) -> None:
        observed_marker_at_stop.append((run / "SPECIALIST_GATE_PASSED").is_file())
        service["active"] = "inactive"

    trainer = SimpleNamespace(
        _load_loop_state=lambda _run: state,
        _recover_interrupted_iteration=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(watcher, "_service_value", service_value)
    monkeypatch.setattr(watcher, "_stop_service", stop)
    monkeypatch.setattr(watcher.time, "sleep", sleep)

    watcher.pause_at_boundary(
        run_dir=run,
        unit="trainer.service",
        trainer=trainer,
        status_path=tmp_path / "status.json",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )
    assert observed_marker_at_stop == [True]


def test_committed_terminal_pass_uses_formal_gate_not_incumbent_h2h() -> None:
    state = {
        "history": [
            {
                "iteration": 15,
                "active_gate_result": {
                    "passed": True,
                    "pipeline_gate_passed": False,
                    "promotion_passed": False,
                    "checks": {
                        "audit": True,
                        "skill_weighted_win_rate": True,
                        "skill_weighted_confidence_lower": True,
                        "s_tier_mean_floor": True,
                        "individual_opponent_floor": True,
                    },
                },
            }
        ]
    }
    assert watcher._committed_terminal_pass(state)
    state["history"][-1]["active_gate_result"]["checks"][
        "individual_opponent_floor"
    ] = False
    assert not watcher._committed_terminal_pass(state)
