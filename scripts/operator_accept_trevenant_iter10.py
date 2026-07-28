#!/usr/bin/env python3
"""Freeze and submit the explicitly owner-accepted Trevenant iteration 10.

This is deliberately a one-shot migration tool.  It validates the durable
zero-indexed iteration-10 commit and the exact formal active-gate result, while
recording that the normal promotion check did not pass.  It never rewrites the
append-only training ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.model_registry import freeze_model, sha256, verify_frozen_model
from scripts.handle_passed_gate import (
    build_submission_bundle,
    materialize_pinned_specialist_deck,
)


SCHEMA = "poke_bot.operator_accepted_specialist_floor_transition/v1"
REQUIRED_GATE_CHECKS = {
    "audit",
    "skill_weighted_win_rate",
    "skill_weighted_confidence_lower",
    "s_tier_mean_floor",
    "individual_opponent_floor",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    commit_path = run_dir / "commits/iter_00010.json"
    metrics_path = run_dir / "metrics/iter_00010.json"
    checkpoint = run_dir / "checkpoints/iter_00010.pt"
    commit = read_json(commit_path)
    metrics = read_json(metrics_path)
    rows = [
        dict(row)
        for row in commit.get("history", [])
        if isinstance(row, dict) and int(row.get("iteration", -1)) == 10
    ]
    if (
        int(commit.get("last_completed_iteration", -1)) != 10
        or int(commit.get("next_iteration", -1)) != 11
        or len(rows) != 1
        or rows[0].get("completed") is not True
    ):
        raise RuntimeError("iteration 10 does not have one durable commit boundary")
    row = rows[0]
    result = dict(row.get("active_gate_result") or {})
    candidate = dict(row.get("candidate") or {})
    audit = dict(result.get("audit") or {})
    checks = dict(result.get("checks") or {})
    digest = sha256(checkpoint)
    if (
        result.get("passed") is not True
        or set(checks) != REQUIRED_GATE_CHECKS
        or not all(checks.values())
        or audit.get("passed") is not True
        or audit.get("exact_distribution") is not True
        or audit.get("exact_weights") is not True
        or audit.get("both_seats") is not True
        or int(result.get("games", -1)) != 2250
        or str(result.get("checkpoint_digest") or "") != digest
        or str(candidate.get("digest") or "") != digest
        or Path(str(candidate.get("path") or "")).resolve() != checkpoint.resolve()
        or result.get("promotion_passed") is not False
    ):
        raise RuntimeError("iteration 10 is not the exact owner-accepted boundary")

    acceptance = {
        "schema": SCHEMA,
        "specialist_id": "hops-trevenant",
        "iteration_numbering": "zero_indexed",
        "completed_iteration_index": 10,
        "completed_iteration_count": 11,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_digest": digest,
        "formal_active_gate_passed": True,
        "pipeline_promotion_passed": False,
        "operator_decision": (
            "Treat completed iteration index 10 as the eleven-iteration floor; "
            "freeze, submit, and hand off without running iteration 11."
        ),
        "commit": str(commit_path.resolve()),
        "commit_sha256": sha256(commit_path),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "active_gate_result": result,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.receipt.resolve(), acceptance)

    frozen_manifest = freeze_model(
        registry_root=args.registry_root,
        family="hops-trevenant-protocol-gate-pass-v1",
        display_name="Hop's Trevenant Owner-Accepted Iteration-10 Champion",
        checkpoint=checkpoint,
        expected_digest=digest,
        provenance={
            "archive_trigger": "explicit_owner_accepted_zero_indexed_floor",
            "operator_acceptance_receipt": str(args.receipt.resolve()),
            "operator_acceptance_receipt_sha256": sha256(args.receipt),
            "iteration": 10,
            "completed_iteration_count": 11,
            "commit": str(commit_path.resolve()),
            "commit_sha256": sha256(commit_path),
        },
        evidence=result,
        require_exact_heldout=False,
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(frozen_manifest["model_path"]).parent)
    deck = materialize_pinned_specialist_deck(
        run_dir=run_dir,
        representatives_path=args.representatives,
        archetype="hops-trevenant",
        output_path=args.submission_root / "pinned-hops-trevenant.deck.csv",
    )
    bundle = build_submission_bundle(
        repo_root=ROOT,
        frozen_manifest=frozen,
        deck_receipt=deck,
        output_dir=args.submission_root / "build",
        python=args.python,
        archetype="hops-trevenant",
        matchup_tree=args.matchup_tree,
    )
    label = f"Trevenant iter10 floor pass copy 1 {digest[7:19]}"
    completed = subprocess.run(
        [
            str(args.kaggle),
            "competitions",
            "submit",
            "-c",
            args.competition,
            "-f",
            bundle["path"],
            "-m",
            label,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    submission = {
        "label": label,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "bundle": bundle,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    acceptance["frozen_model"] = frozen
    acceptance["submission"] = submission
    atomic_json(args.receipt.resolve(), acceptance)
    if completed.returncode:
        raise RuntimeError(
            f"Kaggle submission failed rc={completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    print(json.dumps(submission, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
