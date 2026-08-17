#!/usr/bin/env python3
"""Freeze and queue one exact gate result under an authorized future threshold.

The original training commit and its startup-pinned pass/fail decision remain
untouched.  This tool validates that exact committed evidence, applies only the
operator-authorized individual-opponent-floor transition, writes a separate
immutable receipt, freezes the unchanged checkpoint, builds one submission
copy, and queues it for asynchronous upload.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from poke_bot.pure_rl.model_registry import freeze_model, sha256, verify_frozen_model
from scripts.handle_passed_gate import (
    HANDLER_SCHEMA,
    _canonical_digest,
    _copy_submission_slot,
    build_submission_bundle,
    materialize_pinned_specialist_deck,
    queue_submission_copies,
)


SCHEMA = "poke_bot.gate_threshold_transition_acceptance/v1"
REQUIRED_CHECKS = {
    "audit",
    "skill_weighted_win_rate",
    "skill_weighted_confidence_lower",
    "s_tier_mean_floor",
    "individual_opponent_floor",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _transition_config(config_path: Path) -> dict[str, Any]:
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        premium = value["research_evaluations"]["premium_competition"]
        criteria = premium["gate_threshold"]
        transition = premium["threshold_transition"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("canonical threshold transition is absent") from exc
    if (
        float(criteria.get("individual_opponent_floor", -1)) != 0.15
        or float(transition.get("future_individual_opponent_floor", -1)) != 0.15
        or int(transition.get("effective_after_completed_iteration", -1)) != 9
        or transition.get("applies_to_lucario_iteration_10") is not True
        or transition.get("completed_iteration_9_decision_unchanged") is not True
    ):
        raise RuntimeError("canonical future-gate threshold transition changed")
    normalized_transition = json.loads(
        json.dumps(dict(transition), default=str)
    )
    return {
        "criteria": dict(criteria),
        "transition": normalized_transition,
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
    }


def _validation_plan(
    *,
    run_dir: Path,
    commit_path: Path,
    metrics_path: Path,
    receipt_path: Path,
    checkpoint: Path,
    checkpoint_digest: str,
    iteration: int,
    original: dict[str, Any],
    accepted: dict[str, Any],
    candidate: dict[str, Any],
    transition: dict[str, Any],
) -> dict[str, Any]:
    matchups = [dict(row) for row in accepted["matchups"]]
    audit = dict(accepted["audit"])
    roster_ids = sorted(str(row["opponent_id"]) for row in matchups)
    validation: dict[str, bool] = {
        "active_gate_passed": accepted.get("passed") is True,
        "all_gate_criteria": all(dict(accepted["checks"]).values()),
        "audit_passed": audit.get("passed") is True,
        "audit_distribution": audit.get("exact_distribution") is True,
        "audit_weights": audit.get("exact_weights") is True,
        "audit_both_seats": audit.get("both_seats") is True,
        "candidate_digest": str(candidate.get("digest") or "")
        == checkpoint_digest,
        "candidate_path": Path(str(candidate.get("path") or "")).resolve()
        == checkpoint,
        "checkpoint_bytes": sha256(checkpoint) == checkpoint_digest,
        "commit_boundary": True,
        "committed": True,
        "games": int(accepted.get("games", -1)) == 250 * len(matchups),
        "gate_criteria_set": set(dict(accepted["checks"])) == REQUIRED_CHECKS,
        "threshold_authorized": float(transition["criteria"][
            "individual_opponent_floor"
        ])
        == 0.15,
        "original_decision_preserved": original.get("passed") is False,
    }
    audited = dict(audit.get("per_opponent") or {})
    for row in matchups:
        opponent_id = str(row["opponent_id"])
        allocation = dict(audited.get(opponent_id) or {})
        validation[f"matchup_allocation:{opponent_id}"] = (
            int(row.get("games", -1)) == 250
            and int(row.get("seat0", -1)) == 125
            and int(row.get("seat1", -1)) == 125
        )
        validation[f"audit_allocation:{opponent_id}"] = (
            int(allocation.get("games", -1)) == 250
            and int(allocation.get("seat0", -1)) == 125
            and int(allocation.get("seat1", -1)) == 125
        )
    if not all(validation.values()):
        failed = sorted(key for key, value in validation.items() if not value)
        raise RuntimeError("threshold-transition validation failed: " + ",".join(failed))

    return {
        "schema": "poke_bot.exact_pass_archive_plan/v1",
        "run_dir": str(run_dir),
        "iteration": iteration,
        "commit_boundary": iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": checkpoint_digest,
        "games": int(accepted["games"]),
        "games_per_opponent": 250,
        "candidate_first_per_opponent": 125,
        "candidate_second_per_opponent": 125,
        "skill_weighted_wr": float(accepted["skill_weighted_wr"]),
        "confidence_lower": float(accepted["confidence_lower"]),
        "gate_id": str(accepted["gate_id"]),
        "base_gate_id": str(original["gate_id"]),
        "effective_contract_digest": _canonical_digest(
            {
                "base_gate_id": original["gate_id"],
                "individual_opponent_floor": 0.15,
                "authorization": transition["transition"],
            }
        ),
        "roster_ids": roster_ids,
        "contract": transition["config"],
        "contract_sha256": transition["config_sha256"],
        "commit": str(commit_path),
        "commit_digest": _canonical_digest(read_json(commit_path)),
        "commit_file_sha256": sha256(commit_path),
        "exact_result_pointer": str(receipt_path),
        "exact_result_pointer_sha256": None,
        "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "marker": None,
        "result": accepted,
        "validation": validation,
    }


def validate_threshold_transition_receipt(
    receipt_path: Path,
    *,
    specialist_id: str,
) -> dict[str, Any]:
    receipt_path = Path(receipt_path).expanduser().resolve()
    receipt = read_json(receipt_path)
    plan = dict(receipt.get("gate_plan") or {})
    original = dict(receipt.get("original_gate_result") or {})
    accepted = dict(plan.get("result") or {})
    validation = dict(plan.get("validation") or {})
    checkpoint = Path(str(plan.get("checkpoint") or "")).expanduser().resolve()
    commit = Path(str(plan.get("commit") or "")).expanduser().resolve()
    metrics = Path(str(plan.get("metrics") or "")).expanduser().resolve()
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("specialist_id") != specialist_id
        or receipt.get("original_commit_rewritten") is not False
        or receipt.get("checkpoint_rewritten") is not False
        or float(receipt.get("prior_individual_opponent_floor", -1)) != 0.25
        or float(receipt.get("accepted_individual_opponent_floor", -1)) != 0.15
        or not validation
        or not all(value is True for value in validation.values())
        or accepted.get("passed") is not True
        or original.get("passed") is not False
        or sha256(checkpoint) != str(plan.get("checkpoint_digest") or "")
        or sha256(commit) != str(plan.get("commit_file_sha256") or "")
        or sha256(metrics) != str(plan.get("metrics_sha256") or "")
        or receipt.get("checkpoint_digest") != plan.get("checkpoint_digest")
    ):
        raise RuntimeError("threshold-transition acceptance receipt changed")
    return plan


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    checkpoint = run_dir / "checkpoints" / f"iter_{args.iteration:05d}.pt"
    commit_path = run_dir / "commits" / f"iter_{args.iteration:05d}.json"
    metrics_path = run_dir / "metrics" / f"iter_{args.iteration:05d}.json"
    commit = read_json(commit_path)
    rows = [
        dict(row)
        for row in (commit.get("history") or [])
        if isinstance(row, dict)
        and int(row.get("iteration", -1)) == int(args.iteration)
    ]
    if (
        int(commit.get("last_completed_iteration", -1)) != args.iteration
        or int(commit.get("next_iteration", -1)) != args.iteration + 1
        or len(rows) != 1
        or rows[0].get("completed") is not True
    ):
        raise RuntimeError("threshold transition requires one durable iteration commit")
    row = rows[0]
    original = dict(row.get("active_gate_result") or {})
    candidate = dict(row.get("candidate") or {})
    checks = dict(original.get("checks") or {})
    matchups = [dict(value) for value in (original.get("matchups") or [])]
    minimum_wr = min(float(value["wr"]) for value in matchups)
    transition = _transition_config(config)
    if (
        original.get("passed") is not False
        or set(checks) != REQUIRED_CHECKS
        or checks.get("individual_opponent_floor") is not False
        or not all(value for key, value in checks.items() if key != "individual_opponent_floor")
        or int(original.get("games", -1)) != 250 * len(matchups)
        or minimum_wr < 0.15
        or minimum_wr >= 0.25
        or dict(original.get("audit") or {}).get("passed") is not True
    ):
        raise RuntimeError("committed result is not eligible for the 0.25-to-0.15 transition")
    checkpoint_digest = sha256(checkpoint)
    if (
        candidate.get("digest") != checkpoint_digest
        or Path(str(candidate.get("path") or "")).resolve() != checkpoint
        or original.get("checkpoint_digest") != checkpoint_digest
    ):
        raise RuntimeError("threshold transition checkpoint identity changed")

    accepted = copy.deepcopy(original)
    accepted["gate_id"] = str(original["gate_id"]) + "+individual-floor15-r1"
    accepted["checks"]["individual_opponent_floor"] = True
    accepted["passed"] = True
    accepted["minimum_opponent_wr"] = minimum_wr
    accepted["threshold_transition"] = {
        "prior_individual_opponent_floor": 0.25,
        "accepted_individual_opponent_floor": 0.15,
        "authorization": transition["transition"],
        "original_gate_id": original["gate_id"],
        "original_result_digest": _canonical_digest(original),
        "training_commit_rewritten": False,
        "checkpoint_rewritten": False,
    }
    plan = _validation_plan(
        run_dir=run_dir,
        commit_path=commit_path,
        metrics_path=metrics_path,
        receipt_path=receipt_path,
        checkpoint=checkpoint,
        checkpoint_digest=checkpoint_digest,
        iteration=args.iteration,
        original=original,
        accepted=accepted,
        candidate=candidate,
        transition=transition,
    )
    receipt = {
        "schema": SCHEMA,
        "specialist_id": args.specialist_id,
        "iteration": args.iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": checkpoint_digest,
        "prior_individual_opponent_floor": 0.25,
        "accepted_individual_opponent_floor": 0.15,
        "minimum_observed_opponent_win_rate": minimum_wr,
        "original_commit_rewritten": False,
        "checkpoint_rewritten": False,
        "original_gate_result": original,
        "gate_plan": plan,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(receipt_path, receipt)
    plan["exact_result_pointer_sha256"] = sha256(receipt_path)
    receipt["gate_plan"] = plan
    atomic_json(receipt_path, receipt)

    frozen_manifest = freeze_model(
        registry_root=args.registry_root,
        family=args.family,
        display_name=args.display_name,
        checkpoint=checkpoint,
        expected_digest=checkpoint_digest,
        provenance={
            "archive_trigger": "operator_authorized_future_gate_threshold_transition",
            "threshold_transition_receipt": str(receipt_path),
            "threshold_transition_receipt_sha256": sha256(receipt_path),
            "run_dir": str(run_dir),
            "iteration": args.iteration,
            "commit": str(commit_path),
            "commit_file_sha256": sha256(commit_path),
        },
        evidence=accepted,
        require_exact_heldout=False,
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(frozen_manifest["model_path"]).parent)
    deck = materialize_pinned_specialist_deck(
        run_dir=run_dir,
        representatives_path=args.representatives,
        archetype=args.specialist_id,
        output_path=args.submission_root / f"pinned-{args.specialist_id}.deck.csv",
    )
    bundle = build_submission_bundle(
        repo_root=args.runtime_root,
        frozen_manifest=frozen,
        deck_receipt=deck,
        output_dir=args.submission_root / "build",
        python=args.python,
        archetype=args.specialist_id,
        matchup_tree=args.matchup_tree,
    )
    copy_row = _copy_submission_slot(bundle, args.submission_root, 1)
    queued = queue_submission_copies(
        queue_path=args.submission_queue,
        copies=[copy_row],
        gate_plan=plan,
        specialist_id=args.specialist_id,
        competition=args.competition,
    )
    handler = {
        "schema": HANDLER_SCHEMA,
        "phase": "submissions_queued",
        "submission_mode": "queue_and_continue",
        "approval_text": "Each solid submit will only submit one copy to Kaggle",
        "approved_submission_count": 1,
        "gate": plan,
        "frozen_model": frozen,
        "submission_bundle": bundle,
        "submission_queue": str(args.submission_queue.resolve()),
        "queued_submissions": queued,
        "successful_submission_count": 0,
        "all_submissions_succeeded": False,
        "automatic_retries": False,
        "threshold_transition_receipt": str(receipt_path),
        "threshold_transition_receipt_sha256": sha256(receipt_path),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(args.handler_state, handler)

    runtime_registry = read_json(args.runtime_registry)
    row_config = dict(
        (runtime_registry.get("specialists") or {}).get(args.specialist_id) or {}
    )
    handler_config = dict(row_config.get("pass_handler") or {})
    if row_config.get("status") != "ready" or not handler_config:
        raise RuntimeError("active specialist runtime row is not ready")
    handler_config["threshold_transition_receipt"] = str(receipt_path)
    handler_config["threshold_transition_receipt_sha256"] = sha256(receipt_path)
    row_config["pass_handler"] = handler_config
    runtime_registry["specialists"][args.specialist_id] = row_config
    atomic_json(args.runtime_registry, runtime_registry)
    return handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--submission-queue", type=Path, required=True)
    parser.add_argument("--handler-state", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    return parser.parse_args()


def main() -> int:
    state = run(parse_args())
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
