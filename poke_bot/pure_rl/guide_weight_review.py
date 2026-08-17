"""Emit immutable fleet-wide current-deck guide review requests.

The production learner only declares a review after its iteration commit is
durable.  A separate shadow worker consumes the request and produces matched
guide-on/guide-off checkpoints and evaluation rows.  This module never trains,
evaluates, changes a weight, or touches serving state.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


REQUEST_SCHEMA = "poke_bot.current_deck_guide_weight_review_request/v1"
REVIEW_EVERY_COMPLETED_ITERATIONS = 5
REVIEW_WINDOW_ITERATIONS = 5


def _file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(source),
        "sha256": "sha256:" + digest.hexdigest(),
        "bytes": source.stat().st_size,
    }


def _immutable_json(path: Path, value: dict[str, Any]) -> Path:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != payload:
            raise RuntimeError(f"guide review request identity changed: {target}")
        return target
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def emit_review_request(
    *,
    run_dir: str | Path,
    specialist_id: str,
    completed_iteration: int,
    current_weight: float,
    iteration_commit: str | Path,
    guide_contract: str | Path,
    guide_version: str,
    prospective_policy_revision: int,
    learning_semantics_revision: int,
    consecutive_nonpositive_evaluations: int,
) -> Path | None:
    """Declare one isolated five-iteration guide review, when due."""

    iteration = int(completed_iteration)
    weight = float(current_weight)
    specialist = str(specialist_id).strip().casefold()
    version = str(guide_version).strip()
    if iteration <= 0 or iteration % REVIEW_EVERY_COMPLETED_ITERATIONS:
        return None
    if weight <= 0.0:
        return None
    if (
        not specialist
        or not version
        or not 0.0 < weight <= 0.50
        or int(prospective_policy_revision) != 44
        or int(learning_semantics_revision) != 46
        or int(consecutive_nonpositive_evaluations) < 0
    ):
        raise ValueError("invalid current-deck guide review identity")

    root = Path(run_dir).expanduser().resolve()
    commit = _file_identity(iteration_commit)
    commit_payload = json.loads(Path(commit["path"]).read_text(encoding="utf-8"))
    if (
        int(commit_payload.get("last_completed_iteration", -1)) != iteration
        or int(commit_payload.get("next_iteration", -1)) != iteration + 1
        or str(commit_payload.get("mode") or "") != "specialist"
    ):
        raise RuntimeError("guide review does not follow an immutable specialist commit")

    window_start = iteration - REVIEW_WINDOW_ITERATIONS + 1
    collections: list[dict[str, Any]] = []
    for source_iteration in range(window_start, iteration + 1):
        receipt = _file_identity(
            root
            / "collection_receipts"
            / f"iter_{source_iteration:05d}.json"
        )
        payload = json.loads(
            Path(receipt["path"]).read_text(encoding="utf-8")
        )
        if (
            payload.get("schema") != "poke_bot.completed_collection/v1"
            or int(payload.get("iteration", -1)) != source_iteration
        ):
            raise RuntimeError("guide review collection receipt is invalid")
        collections.append(receipt)

    seed_receipt = json.loads(
        Path(collections[0]["path"]).read_text(encoding="utf-8")
    )
    seed_checkpoint = _file_identity(seed_receipt["checkpoint"])
    if seed_checkpoint["sha256"] != str(seed_receipt["checkpoint_digest"]):
        raise RuntimeError("guide review seed checkpoint identity changed")

    request = {
        "schema": REQUEST_SCHEMA,
        "status": "ready_for_isolated_shadow_pair",
        "owner_decision_revision": 43,
        "prospective_policy_revision": 44,
        "learning_semantics_revision": 46,
        "scope": "future_specialist_training_runs_only",
        "retroactive_application_allowed": False,
        "specialist_id": specialist,
        "completed_iteration": iteration,
        "earliest_activation_boundary_next_iteration": iteration + 1,
        "application_boundary": (
            "first_available_future_five_iteration_hard_pause"
        ),
        "current_weight": weight,
        "consecutive_nonpositive_evaluations": int(
            consecutive_nonpositive_evaluations
        ),
        "guide_contract": _file_identity(guide_contract),
        "guide_version": version,
        "iteration_commit": commit,
        "review_window": {
            "first_iteration": window_start,
            "last_iteration": iteration,
            "collection_receipts": collections,
            "seed_checkpoint": seed_checkpoint,
        },
        "shadow_pair": {
            "guide_on_weight": weight,
            "guide_off_weight": 0.0,
            "same_parent_replay_split_batch_order_and_optimizer_required": True,
            "promotion_allowed": False,
            "serving_allowed": False,
        },
        "evaluation": {
            "evidence_schema": (
                "poke_bot.current_deck_guide_paired_evaluation/v1"
            ),
            "minimum_pairs": 1000,
            "minimum_pairs_per_matchup": 50,
            "matched_opponent_seat_and_requested_seed": True,
            "first_second_balance_required": True,
            "confidence_level": 0.90,
            "training_eligible": False,
            "replay_eligible": False,
            "formal_gate": False,
        },
        "weight_change_allowed_without_compiled_schedule": False,
        "schedule_receipt_schema": (
            "poke_bot.current_deck_guide_weight_schedule/v1"
        ),
    }
    return _immutable_json(
        root
        / "guide_weight_reviews"
        / f"review_iter_{iteration:05d}.request.json",
        request,
    )


__all__ = [
    "REQUEST_SCHEMA",
    "REVIEW_EVERY_COMPLETED_ITERATIONS",
    "REVIEW_WINDOW_ITERATIONS",
    "emit_review_request",
]
