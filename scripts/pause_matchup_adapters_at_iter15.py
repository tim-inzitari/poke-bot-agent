#!/usr/bin/env python3
"""Pause pure RL at the immutable iteration-15 adapter boundary.

This is deliberately an external watcher.  It does not train, materialize, or
enable matchup adapters.  It waits for the exact append-only iteration-15
commit, stops the owning user service, and quarantines any uncommitted
iteration-16 work that raced the stop.  A committed iteration 16 is never
rolled back.

The service must already expose ``RefuseManualStop=no``.  Requiring that
precondition up front avoids discovering at the boundary that systemd refuses
the stop.  The later dormant-bank migration is a separate, explicit command.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


COMPLETED_ITERATION = 15
NEXT_ITERATION = 16
STATUS_SCHEMA = "poke_bot.matchup_adapter_boundary_pause/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read exact JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish(path: Path, **values: Any) -> None:
    _atomic_json(
        path,
        {
            "schema": STATUS_SCHEMA,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        },
    )


def _systemctl(*args: str, timeout: float = 90.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _service_value(unit: str, key: str) -> str:
    result = _systemctl("show", unit, "-p", key, "--value", timeout=15.0)
    if result.returncode:
        raise RuntimeError(
            f"cannot read systemd {key} for {unit}: {result.stdout.strip()}"
        )
    return result.stdout.strip()


def _stop_service(unit: str) -> None:
    result = _systemctl("stop", unit, timeout=90.0)
    if result.returncode:
        raise RuntimeError(
            f"systemd refused boundary stop for {unit}: {result.stdout.strip()}"
        )
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        active = _service_value(unit, "ActiveState")
        pid = int(_service_value(unit, "MainPID") or 0)
        if active in {"inactive", "failed"} and pid == 0:
            return
        time.sleep(0.05)
    raise RuntimeError(f"service did not stop at adapter boundary: {unit}")


def _load_trainer(root: Path) -> ModuleType:
    source = root / "scripts" / "train_pure_rl.py"
    if not source.is_file():
        raise RuntimeError(f"trainer source is missing: {source}")
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    name = f"matchup_adapter_boundary_trainer_{hashlib.sha256(root_text.encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for required in ("_load_loop_state", "_recover_interrupted_iteration"):
        if not callable(getattr(module, required, None)):
            raise RuntimeError(f"trainer lacks recovery function {required}")
    return module


@dataclass(frozen=True)
class BoundaryProof:
    commit_path: Path
    commit_digest: str
    parent_checkpoint: Path
    parent_checkpoint_digest: str
    state: dict[str, Any]


def _iteration_commit(run_dir: Path, iteration: int) -> Path:
    return run_dir / "commits" / f"iter_{iteration:05d}.json"


def boundary_proof(run_dir: Path) -> BoundaryProof | None:
    """Return the exact 15 -> 16 proof, or ``None`` while still waiting."""

    commit_path = _iteration_commit(run_dir, COMPLETED_ITERATION)
    loop_path = run_dir / "loop_state.json"
    if not commit_path.is_file():
        if _iteration_commit(run_dir, NEXT_ITERATION).exists():
            raise RuntimeError("iteration 16 committed without an iteration-15 boundary")
        return None
    if _iteration_commit(run_dir, NEXT_ITERATION).exists():
        raise RuntimeError("iteration 16 is already immutable; refusing rollback")
    commit = _read_json(commit_path)
    loop_state = _read_json(loop_path)
    if (
        int(commit.get("last_completed_iteration", -1)) != COMPLETED_ITERATION
        or int(commit.get("next_iteration", -1)) != NEXT_ITERATION
    ):
        raise RuntimeError("iteration-15 commit is not the exact 15 -> 16 boundary")
    if loop_state != commit:
        # The append-only commit is published immediately before the mutable
        # loop pointer. Observing the exact prior immutable pointer is a normal
        # atomic-publication window, not corruption. Retry until loop_state is
        # advanced; every other mismatch remains fail-closed.
        previous_path = _iteration_commit(run_dir, COMPLETED_ITERATION - 1)
        previous = _read_json(previous_path) if previous_path.is_file() else None
        if (
            isinstance(previous, dict)
            and loop_state == previous
            and int(previous.get("last_completed_iteration", -1))
            == COMPLETED_ITERATION - 1
            and int(previous.get("next_iteration", -1)) == COMPLETED_ITERATION
        ):
            return None
        raise RuntimeError("loop_state does not exactly match immutable iteration-15 commit")
    learner = commit.get("learner")
    if not isinstance(learner, dict):
        raise RuntimeError("iteration-15 commit has no learner identity")
    parent = Path(str(learner.get("path") or "")).expanduser().resolve()
    digest = str(learner.get("digest") or "").lower()
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise RuntimeError("iteration-15 learner digest is malformed")
    if not parent.is_file() or _sha256(parent) != digest:
        raise RuntimeError("iteration-15 learner path/digest identity mismatch")
    return BoundaryProof(
        commit_path=commit_path.resolve(),
        commit_digest=_sha256(commit_path),
        parent_checkpoint=parent,
        parent_checkpoint_digest=digest,
        state=commit,
    )


def _committed_terminal_pass(state: dict[str, Any]) -> bool:
    history = state.get("history")
    if not isinstance(history, list) or not history:
        return False
    row = history[-1]
    if not isinstance(row, dict) or int(row.get("iteration", -1)) != COMPLETED_ITERATION:
        return False
    active_gate = row.get("active_gate_result")
    checks = (
        active_gate.get("checks")
        if isinstance(active_gate, dict)
        else None
    )
    return bool(
        isinstance(active_gate, dict)
        and active_gate.get("passed") is True
        and isinstance(checks, dict)
        and frozenset(checks)
        in {
            frozenset(
                {
                    "audit",
                    "skill_weighted_win_rate",
                    "skill_weighted_confidence_lower",
                    "s_tier_mean_floor",
                    "individual_opponent_floor",
                }
            ),
            frozenset(
                {
                    "audit",
                    "skill_weighted_win_rate",
                    "skill_weighted_confidence_lower",
                    "s_tier_mean_floor",
                    "individual_opponent_floor",
                    "s_plus_matchup_floor_allowance",
                }
            ),
        }
        and all(value is True for value in checks.values())
    )


def _runtime_started_iteration(run_dir: Path) -> int | None:
    path = run_dir / "iteration_runtime.json"
    if not path.is_file():
        return None
    value = _read_json(path).get("iteration")
    try:
        return int(value)
    except (TypeError, ValueError):
        return NEXT_ITERATION


def _quarantine_iteration_runtime(run_dir: Path) -> Path | None:
    runtime = run_dir / "iteration_runtime.json"
    observed = _runtime_started_iteration(run_dir)
    if observed is None or observed < NEXT_ITERATION:
        return None
    destination = (
        run_dir
        / "quarantine"
        / f"iter_{NEXT_ITERATION:05d}"
        / "boundary_pause_iteration_runtime.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != _sha256(runtime):
            raise RuntimeError("iteration-runtime quarantine destination conflicts")
        runtime.unlink()
    else:
        runtime.replace(destination)
    return destination


def _iter16_started(run_dir: Path) -> bool:
    stem = f"iter_{NEXT_ITERATION:05d}"
    fixed = (
        run_dir / "collection_receipts" / f"{stem}.json",
        run_dir / "shards" / f"{stem}.jsonl",
        run_dir / "checkpoints" / f"{stem}.pt",
        run_dir / "commits" / f"{stem}.json",
        run_dir / "eval" / f"{stem}.json",
        run_dir / "metrics" / f"{stem}.json",
        run_dir / "research_controls" / f"{stem}.json",
    )
    if any(path.exists() for path in fixed):
        return True
    for parent in (run_dir / "shards", run_dir / "checkpoints"):
        if parent.is_dir() and any(parent.glob(f"*{stem}*")):
            return True
    runtime_iteration = _runtime_started_iteration(run_dir)
    if runtime_iteration is not None and runtime_iteration >= NEXT_ITERATION:
        return True
    latest = run_dir / "metrics" / "latest.json"
    if latest.is_file():
        try:
            if int(_read_json(latest).get("iteration", -1)) >= NEXT_ITERATION:
                return True
        except (TypeError, ValueError):
            return True
    return False


def pause_at_boundary(
    *,
    run_dir: Path,
    unit: str,
    trainer: ModuleType,
    status_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
) -> BoundaryProof:
    """Wait, stop, and restore one clean uncommitted 15 -> 16 boundary."""

    if _service_value(unit, "RefuseManualStop").strip().lower() != "no":
        raise RuntimeError(
            "boundary watcher requires RefuseManualStop=no before it is armed"
        )
    if _service_value(unit, "ActiveState") not in {"active", "activating"}:
        raise RuntimeError("production trainer is not active before boundary watch")

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    _publish(
        status_path,
        status="waiting_for_immutable_iter15",
        unit=unit,
        run_dir=str(run_dir.resolve()),
    )
    proof: BoundaryProof | None = None
    while time.monotonic() < deadline:
        proof = boundary_proof(run_dir)
        if proof is not None:
            break
        if _service_value(unit, "ActiveState") not in {"active", "activating"}:
            raise RuntimeError("trainer stopped before immutable iteration 15")
        time.sleep(max(0.01, float(poll_seconds)))
    if proof is None:
        raise RuntimeError("timed out waiting for immutable iteration 15")

    boundary_history = list(proof.state.get("history") or [])
    boundary_row = boundary_history[-1] if boundary_history else {}
    boundary_gate = (
        boundary_row.get("stage_gate") if isinstance(boundary_row, dict) else None
    )
    if isinstance(boundary_gate, dict) and boundary_gate.get("passed") is True:
        # Commit publication precedes the terminal marker.  Suppressing that
        # marker would also suppress the separately guarded freeze/submission
        # handler, so allow the trainer to finish only this terminal action.
        pass_marker = run_dir / "SPECIALIST_GATE_PASSED"
        _publish(
            status_path,
            status="waiting_for_iter15_gate_marker",
            commit=str(proof.commit_path),
            commit_digest=proof.commit_digest,
        )
        marker_deadline = time.monotonic() + 90.0
        while time.monotonic() < marker_deadline and not pass_marker.is_file():
            if _iteration_commit(run_dir, NEXT_ITERATION).exists():
                raise RuntimeError(
                    "iteration 16 committed before the passed-gate marker"
                )
            if _service_value(unit, "ActiveState") == "failed":
                raise RuntimeError(
                    "trainer failed after passing iter15 but before gate marker"
                )
            time.sleep(0.05)
        if not pass_marker.is_file():
            raise RuntimeError("passed iteration 15 did not publish its gate marker")

    # The immutable commit lands just before SPECIALIST_GATE_PASSED. Never
    # stop that short publication window: the exactly-two submission handler
    # owns a terminal pass and is configured to hand off without resuming RL.
    if _committed_terminal_pass(proof.state):
        marker = run_dir / "SPECIALIST_GATE_PASSED"
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and not marker.is_file():
            if _service_value(unit, "ActiveState") == "failed":
                raise RuntimeError(
                    "trainer failed after a passed commit but before its marker"
                )
            time.sleep(0.02)
        if not marker.is_file():
            raise RuntimeError("passed iteration-15 commit did not publish its marker")
        _publish(
            status_path,
            status="superseded_by_exact_gate_pass",
            unit=unit,
            commit=str(proof.commit_path),
            commit_digest=proof.commit_digest,
            pass_marker=str(marker.resolve()),
            parent_checkpoint=str(proof.parent_checkpoint),
            parent_checkpoint_digest=proof.parent_checkpoint_digest,
            runtime_activation_enabled=False,
            adapter_training_enabled=False,
        )
        return proof

    _publish(
        status_path,
        status="stopping_at_iter15",
        commit=str(proof.commit_path),
        commit_digest=proof.commit_digest,
        parent_checkpoint=str(proof.parent_checkpoint),
        parent_checkpoint_digest=proof.parent_checkpoint_digest,
    )
    _stop_service(unit)

    # The stop may race the first milliseconds of iteration 16.  A committed
    # iteration 16 is an immutable terminal error; every uncommitted artifact
    # is quarantined by the trainer's transaction recovery instead of deleted.
    stopped_proof = boundary_proof(run_dir)
    if stopped_proof is None or stopped_proof.commit_digest != proof.commit_digest:
        raise RuntimeError("iteration-15 boundary identity changed during stop")
    state = trainer._load_loop_state(run_dir)
    if state is None or state != proof.state:
        raise RuntimeError("trainer recovery loaded a different boundary ledger")
    recovered = trainer._recover_interrupted_iteration(
        run_dir,
        state,
        preserve_completed_collection=False,
    )
    runtime_quarantine = _quarantine_iteration_runtime(run_dir)
    if _iter16_started(run_dir):
        raise RuntimeError("iteration-16 artifacts remain after boundary recovery")
    if _service_value(unit, "ActiveState") not in {"inactive", "failed"}:
        raise RuntimeError("trainer restarted after the boundary stop")

    _publish(
        status_path,
        status="paused_clean_15_to_16",
        unit=unit,
        commit=str(proof.commit_path),
        commit_digest=proof.commit_digest,
        parent_checkpoint=str(proof.parent_checkpoint),
        parent_checkpoint_digest=proof.parent_checkpoint_digest,
        recovered_iteration16=str(recovered) if recovered is not None else None,
        runtime_pointer_quarantine=(
            str(runtime_quarantine) if runtime_quarantine is not None else None
        ),
        service_active_state=_service_value(unit, "ActiveState"),
        runtime_activation_enabled=False,
        adapter_training_enabled=False,
        gate_pass_marker=(
            str(run_dir / "SPECIALIST_GATE_PASSED")
            if (run_dir / "SPECIALIST_GATE_PASSED").is_file()
            else None
        ),
    )
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    trainer = _load_trainer(args.trainer_root.resolve())
    if args.validate_only:
        if _service_value(args.unit, "RefuseManualStop").strip().lower() != "no":
            raise RuntimeError("RefuseManualStop must be no before arming watcher")
        _publish(
            args.status,
            status="validated",
            unit=args.unit,
            run_dir=str(args.run_dir.resolve()),
            trainer_root=str(args.trainer_root.resolve()),
        )
        return 0
    pause_at_boundary(
        run_dir=args.run_dir.resolve(),
        unit=args.unit,
        trainer=trainer,
        status_path=args.status.resolve(),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
