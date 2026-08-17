from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from scripts import pause_at_committed_iteration as watcher


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_boundary_requires_exact_commit_loop_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "iter_00005.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified learner")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = {
        "last_completed_iteration": 5,
        "next_iteration": 6,
        "learner": {"path": str(checkpoint), "digest": digest},
    }
    _write(tmp_path / "commits" / "iter_00005.json", state)
    _write(tmp_path / "loop_state.json", state)

    proof = watcher._boundary(tmp_path, 5)

    assert proof is not None
    assert proof[1] == state
    assert proof[2] == checkpoint.resolve()
    assert proof[3] == digest


def test_boundary_refuses_next_committed_iteration(tmp_path: Path) -> None:
    _write(tmp_path / "commits" / "iter_00006.json", {})

    try:
        watcher._boundary(tmp_path, 5)
    except RuntimeError as exc:
        assert "already committed" in str(exc)
    else:
        raise AssertionError("next committed iteration must fail closed")


def test_pause_explicitly_starts_idempotent_successor(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoints" / "iter_00020.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"terminal learner")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = {
        "last_completed_iteration": 20,
        "next_iteration": 21,
        "learner": {"path": str(checkpoint), "digest": digest},
    }
    _write(tmp_path / "commits" / "iter_00020.json", state)
    _write(tmp_path / "loop_state.json", state)
    service = {"active": True}
    calls: list[tuple[str, ...]] = []

    def fake_value(unit: str, key: str) -> str:
        if key == "RefuseManualStop":
            return "no"
        if key == "LoadState":
            return "loaded"
        if key == "ActiveState":
            return "active" if service["active"] else "inactive"
        if key == "MainPID":
            return "123" if service["active"] else "0"
        raise AssertionError((unit, key))

    def fake_systemctl(*args: str, timeout: float = 90.0):
        del timeout
        calls.append(args)
        if args == ("stop", "trainer.service"):
            service["active"] = False
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(watcher, "_service_value", fake_value)
    monkeypatch.setattr(watcher, "_systemctl", fake_systemctl)
    status = tmp_path / "status.json"
    watcher.pause(
        run_dir=tmp_path,
        unit="trainer.service",
        completed_iteration=20,
        status_path=status,
        poll_seconds=0.01,
        timeout_seconds=1.0,
        next_unit="gate-handler.service",
    )
    assert ("start", "--no-block", "gate-handler.service") in calls
    saved = json.loads(status.read_text(encoding="utf-8"))
    assert saved["status"] == "paused_successor_started"
    assert saved["successor_start_requested"] is True
