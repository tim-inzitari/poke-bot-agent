"""Bound disk growth for a long-lived append-only pure-RL lineage.

Large replay shards are needed only while they are inside the configured replay
window.  Once retired, their immutable bytes are replaced by a small receipt
containing the exact digest, size, and training counts.  Metrics, eval rows,
commits, and protected model identities remain append-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


_ITER_CHECKPOINT = re.compile(r"^iter_(\d{5})\.pt$")
_EXPERT_CHECKPOINT = re.compile(r"^expert_before_iter_(\d{5})\.pt$")
_QUARANTINE_ITERATION = re.compile(r"^iter_(\d{5})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"artifact retirement receipt changed: {path}")


def _retire(path: Path, receipt: Path, payload: dict[str, Any]) -> int:
    if not path.is_file():
        return 0
    row = {
        "schema": 1,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        **payload,
    }
    _write_exclusive(receipt, row)
    # Re-read the durable receipt before unlinking the only large copy.
    verified = json.loads(receipt.read_text(encoding="utf-8"))
    if verified != row:
        raise RuntimeError(f"artifact receipt verification failed: {receipt}")
    size = int(row["bytes"])
    path.unlink()
    return size


def protected_checkpoint_paths(state: dict[str, Any]) -> set[Path]:
    protected: set[Path] = set()
    for key in ("champion", "heldout_champion", "learner", "lineage_base"):
        row = dict(state.get(key) or {})
        if row.get("path"):
            protected.add(Path(str(row["path"])).expanduser().resolve())
    for row in state.get("opponent_pool") or []:
        if isinstance(row, dict) and row.get("path"):
            protected.add(Path(str(row["path"])).expanduser().resolve())
    return protected


def _retire_committed_quarantine(
    run_dir: Path, *, completed_iteration: int
) -> tuple[int, list[int]]:
    """Remove failed-attempt bytes only after an immutable commit exists.

    Recovery quarantine is deliberately durable while an iteration is
    uncommitted. Once the clean retry commits, those partial shards and
    checkpoints can never be consumed again. Preserve their plans/failure
    ledgers in a compact receipt before reclaiming the large payloads.
    """
    root = run_dir / "quarantine"
    reclaimed = 0
    retired: list[int] = []
    try:
        candidates = sorted(root.iterdir())
    except OSError:
        return reclaimed, retired
    for candidate in candidates:
        match = _QUARANTINE_ITERATION.fullmatch(candidate.name)
        if not match or not candidate.is_dir():
            continue
        iteration = int(match.group(1))
        commit = run_dir / "commits" / f"iter_{iteration:05d}.json"
        if iteration > int(completed_iteration) or not commit.is_file():
            continue
        failures: list[dict[str, Any]] = []
        for failure in sorted(candidate.glob("attempt_*/failure.json")):
            try:
                payload = json.loads(failure.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # Do not destroy recovery evidence we cannot preserve exactly.
                failures = []
                break
            failures.append(
                {
                    "source": str(failure),
                    "payload": payload,
                }
            )
        if not failures:
            continue
        size = sum(
            path.stat().st_size for path in candidate.rglob("*") if path.is_file()
        )
        receipt = (
            run_dir
            / "artifact_receipts"
            / "quarantine"
            / f"iter_{iteration:05d}.json"
        )
        row = {
            "schema": 1,
            "kind": "committed_iteration_recovery_quarantine",
            "iteration": iteration,
            "retired_after_iteration": int(completed_iteration),
            "bytes": int(size),
            "commit": str(commit),
            "failures": failures,
            "reason": "clean_retry_committed_partial_attempts_unreachable",
        }
        _write_exclusive(receipt, row)
        if json.loads(receipt.read_text(encoding="utf-8")) != row:
            raise RuntimeError(
                f"quarantine retirement receipt verification failed: {receipt}"
            )
        shutil.rmtree(candidate)
        reclaimed += int(size)
        retired.append(iteration)
    try:
        root.rmdir()
    except OSError:
        pass
    return reclaimed, retired


def apply_artifact_retention(
    run_dir: Path,
    state: dict[str, Any],
    *,
    completed_iteration: int,
    replay_window_shards: int,
    history_iterations: int = 5,
) -> dict[str, Any]:
    """Retire only artifacts no future replay/update can consume."""
    run_dir = Path(run_dir).resolve()
    # Frozen reusable models live outside rolling lineages and carry an
    # explicit fail-closed marker.  Refuse even a mistakenly pointed retention
    # invocation rather than relying only on today's directory glob patterns.
    from .model_registry import is_protected_model_path

    if is_protected_model_path(run_dir):
        raise RuntimeError(
            f"artifact retention refuses protected model registry path: {run_dir}"
        )
    completed = int(completed_iteration)
    window = max(1, int(replay_window_shards))
    receipts = run_dir / "artifact_receipts"
    reclaimed_shards = 0
    retired_shards: list[int] = []

    # Training consumes only ``window`` shards, but retain a short five-iteration
    # rollback/debug history as requested.  This is bounded and does not change
    # which samples enter an update.
    history = max(window, max(1, int(history_iterations)))
    retire_through = completed - history
    for iteration in range(0, max(-1, retire_through) + 1):
        shard = run_dir / "shards" / f"iter_{iteration:05d}.jsonl"
        if not shard.is_file():
            continue
        metric_path = run_dir / "metrics" / f"iter_{iteration:05d}.json"
        metric: dict[str, Any] = {}
        if metric_path.is_file():
            try:
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metric = {}
        reclaimed_shards += _retire(
            shard,
            receipts / "shards" / f"iter_{iteration:05d}.json",
            {
                "kind": "replay_shard",
                "iteration": iteration,
                "retired_after_iteration": completed,
                "games": metric.get("games"),
                "decisions": metric.get("decisions"),
                "reason": "outside_bounded_history_and_replay_window",
            },
        )
        retired_shards.append(iteration)

    protected = protected_checkpoint_paths(state)
    reclaimed_checkpoints = 0
    retired_checkpoints: list[int] = []
    keep_from = completed - history + 1
    for candidate in sorted((run_dir / "checkpoints").glob("iter_*.pt")):
        match = _ITER_CHECKPOINT.match(candidate.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if iteration >= keep_from or candidate.resolve() in protected:
            continue
        reclaimed_checkpoints += _retire(
            candidate,
            receipts / "checkpoints" / f"iter_{iteration:05d}.json",
            {
                "kind": "candidate_checkpoint",
                "iteration": iteration,
                "retired_after_iteration": completed,
                "reason": "outside_checkpoint_debug_window_and_unprotected",
            },
        )
        retired_checkpoints.append(iteration)

    retired_expert_checkpoints: list[int] = []
    for candidate in sorted(
        (run_dir / "checkpoints").glob("expert_before_iter_*.pt")
    ):
        match = _EXPERT_CHECKPOINT.match(candidate.name)
        if not match:
            continue
        before_iteration = int(match.group(1))
        if before_iteration >= keep_from or candidate.resolve() in protected:
            continue
        reclaimed_checkpoints += _retire(
            candidate,
            receipts / "checkpoints" / f"expert_before_iter_{before_iteration:05d}.json",
            {
                "kind": "expert_rehearsal_checkpoint",
                "before_iteration": before_iteration,
                "retired_after_iteration": completed,
                "reason": "outside_checkpoint_debug_window_and_unprotected",
            },
        )
        retired_expert_checkpoints.append(before_iteration)

    reclaimed_quarantine, retired_quarantine = _retire_committed_quarantine(
        run_dir, completed_iteration=completed
    )

    return {
        "retire_through_shard": retire_through,
        "retired_shards": retired_shards,
        "retired_checkpoints": retired_checkpoints,
        "retired_expert_checkpoints": retired_expert_checkpoints,
        "retired_quarantine_iterations": retired_quarantine,
        "reclaimed_quarantine_bytes": reclaimed_quarantine,
        "reclaimed_bytes": (
            reclaimed_shards + reclaimed_checkpoints + reclaimed_quarantine
        ),
        "protected_checkpoints": sorted(str(path) for path in protected),
    }
