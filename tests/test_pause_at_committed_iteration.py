from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
