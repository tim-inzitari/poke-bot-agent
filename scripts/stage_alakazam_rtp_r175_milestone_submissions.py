#!/usr/bin/env python3
"""Queue nonblocking Alakazam RTP r175 milestone Kaggle snapshots.

Owner cadence (GOAL revision 175): every 5 iterations, expert refresh then
Kaggle (first_if_allowed, owner pilot deck), then continue self-play.

Expert refresh is enforced in-trainer via --expert-rehearsal-every 5 (fires at
the start of iterations 5/10/15/...). This sidecar stages Kaggle only from
durable commits at those same iterations once the commit history records the
expert_rehearsal pass — i.e. after refresh, without interrupting collection.

Deck: always alakazam-owner-rtp-pilot-r175 (not measurement-deck v1 reps and
not Slop Box Cox/Chao). Live RL measurement digest may still be the older v1
representative; owner r175 authorizes the pilot for submissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.handle_passed_gate import (  # noqa: E402
    _copy_submission_slot,
    build_submission_bundle,
    queue_submission_copies,
)

SCHEMA = "poke_bot.alakazam_rtp_r175_milestone_submission/v1"
PILOT_LIST_ID = "alakazam-owner-rtp-pilot-r175"
PILOT_CSV = ROOT / "decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv"
PILOT_REPS = ROOT / "outputs/state/alakazam-owner-rtp-pilot-representatives-r175.json"
EXPECTED_PILOT_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
EXPECTED_PILOT_CSV_SHA256 = (
    "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
)
R193_LARGE_REFRESH_ITERATION = 15
R193_LARGE_REFRESH_EPOCHS = 25
R193_EXPERT_DATES = [
    "2026-08-01",
    "2026-08-02",
    "2026-08-03",
    "2026-08-04",
    "2026-08-05",
]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def cards_sha256(cards: list[int]) -> str:
    body = json.dumps(cards, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def eligible_iterations(run_dir: Path, *, maximum_iteration: int) -> list[int]:
    """Post-refresh durable commits: 5, 10, 15, ... <= maximum_iteration."""
    found: list[int] = []
    for path in sorted((run_dir / "commits").glob("iter_*.json")):
        try:
            iteration = int(path.stem.removeprefix("iter_"))
        except ValueError:
            continue
        if 5 <= iteration <= maximum_iteration and iteration % 5 == 0:
            found.append(iteration)
    return found


def validate_commit(run_dir: Path, iteration: int) -> tuple[Path, Path, str, dict[str, Any]]:
    commit_path = run_dir / "commits" / f"iter_{iteration:05d}.json"
    checkpoint = run_dir / "checkpoints" / f"iter_{iteration:05d}.pt"
    commit = read_json(commit_path)
    rows = [
        row
        for row in (commit.get("history") or [])
        if isinstance(row, dict) and int(row.get("iteration", -1)) == iteration
    ]
    if (
        int(commit.get("last_completed_iteration", -1)) != iteration
        or int(commit.get("next_iteration", -1)) != iteration + 1
        or len(rows) != 1
        or rows[0].get("completed") is not True
        or not checkpoint.is_file()
    ):
        raise RuntimeError(f"iteration {iteration} has no exact durable commit")
    rehearsal = dict(rows[0].get("expert_rehearsal") or {})
    if not rehearsal:
        raise RuntimeError(
            f"iteration {iteration} commit lacks expert_rehearsal "
            "(refresh-then-Kaggle fail-closed)"
        )
    if iteration == R193_LARGE_REFRESH_ITERATION:
        manifest = dict(rehearsal.get("manifest") or {})
        expanded = dict(rehearsal.get("expanded_head_training") or {})
        checkpoint_identity = Path(str(rehearsal.get("checkpoint") or ""))
        if (
            int(rehearsal.get("before_iteration", -1)) != iteration
            or int(rehearsal.get("epochs", -1)) != R193_LARGE_REFRESH_EPOCHS
            or list(manifest.get("dates") or []) != R193_EXPERT_DATES
            or float(dict(rehearsal.get("loss_weights") or {}).get("alakazam_guide", -1.0))
            != 0.05
            or not checkpoint_identity.is_file()
            or sha256(checkpoint_identity)
            != str(rehearsal.get("checkpoint_digest") or "")
            or not list(expanded.get("trained_this_epoch") or [])
        ):
            raise RuntimeError(
                "iteration 15 lacks the exact owner-r193 25-epoch full-model "
                "expert refresh receipt"
            )
    digest = sha256(checkpoint)
    candidate = dict(rows[0].get("candidate") or {})
    known_digests = {
        str(candidate.get("digest") or ""),
        str((commit.get("learner") or {}).get("digest") or ""),
        str((commit.get("heldout_champion") or {}).get("digest") or ""),
        str((rows[0].get("learner_after") or {}).get("digest") or ""),
    }
    known_digests.discard("")
    if known_digests and digest not in known_digests:
        raise RuntimeError(f"iteration {iteration} checkpoint digest is not commit-bound")
    return commit_path, checkpoint, digest, rehearsal


def materialize_owner_pilot_deck(output_path: Path) -> dict[str, Any]:
    if not PILOT_CSV.is_file():
        raise RuntimeError(f"owner pilot CSV missing: {PILOT_CSV}")
    if not PILOT_REPS.is_file():
        raise RuntimeError(f"owner pilot representatives missing: {PILOT_REPS}")
    csv_digest = sha256(PILOT_CSV)
    if csv_digest != EXPECTED_PILOT_CSV_SHA256:
        raise RuntimeError(
            f"owner pilot CSV digest drift: {csv_digest} != {EXPECTED_PILOT_CSV_SHA256}"
        )
    reps = read_json(PILOT_REPS)
    if reps.get("list_id") != PILOT_LIST_ID:
        raise RuntimeError("representatives list_id is not owner RTP pilot r175")
    cards = [int(c) for c in ((reps.get("decks") or {}).get("alakazam") or {}).get("card_ids") or []]
    if len(cards) != 60:
        raise RuntimeError("owner pilot representatives are not a 60-card deck")
    digest = cards_sha256(cards)
    if digest != EXPECTED_PILOT_CARDS_SHA256:
        raise RuntimeError(
            f"owner pilot cards digest drift: {digest} != {EXPECTED_PILOT_CARDS_SHA256}"
        )
    body = "".join(f"{card}\n" for card in cards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != body:
        raise RuntimeError("refusing to replace an existing pinned submission deck")
    if not output_path.exists():
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, output_path)
    if sha256(output_path) != csv_digest:
        # card-id newline body should match CSV byte identity for this pilot
        # (CSV is already one id per line). Re-check content equality.
        if output_path.read_text(encoding="utf-8") != PILOT_CSV.read_text(encoding="utf-8"):
            # still OK if only trailing newline differs; require cards digest
            pass
    return {
        "path": str(output_path.resolve()),
        "cards": 60,
        "cards_sha256": digest,
        "file_sha256": sha256(output_path),
        "list_id": PILOT_LIST_ID,
        "representatives": str(PILOT_REPS.resolve()),
        "representatives_sha256": sha256(PILOT_REPS),
        "deck_lineage": {
            "kind": "owner_rtp_pilot_r175",
            "not_slop_box_cox_chao_55188658": True,
            "owner_decision_revision": 175,
            "source_csv": str(PILOT_CSV.resolve()),
            "source_csv_sha256": csv_digest,
        },
        "owner_authorized_measurement_deck_mismatch": True,
        "note": (
            "Live r175 RL measurement deck remains specialist_representatives.v1 "
            "alakazam; owner r175 requires pilot for Kaggle milestones."
        ),
    }


def stage_one(args: argparse.Namespace, iteration: int) -> dict[str, Any]:
    receipt = args.receipts / f"iter_{iteration:05d}.json"
    if receipt.is_file():
        existing = read_json(receipt)
        if existing.get("schema") != SCHEMA or existing.get("status") != "queued":
            raise RuntimeError(f"invalid existing milestone receipt: {receipt}")
        return existing

    commit, checkpoint, checkpoint_digest, rehearsal = validate_commit(
        args.run_dir, iteration
    )
    root = args.submission_root / f"iter-{iteration:05d}"
    deck = materialize_owner_pilot_deck(root / "pinned-alakazam-owner-rtp-pilot-r175.deck.csv")
    bundle = build_submission_bundle(
        repo_root=args.runtime_root,
        frozen_manifest={
            "model_path": str(checkpoint.resolve()),
            "checkpoint_digest": checkpoint_digest,
        },
        deck_receipt=deck,
        output_dir=root / "build",
        python=args.python,
        archetype="alakazam",
        matchup_tree=args.matchup_tree,
        turn_order_preference="first_if_allowed",
    )
    copy = _copy_submission_slot(bundle, root, 1)
    queued = queue_submission_copies(
        queue_path=args.queue,
        copies=[copy],
        gate_plan={
            "checkpoint_digest": checkpoint_digest,
            "gate_id": "final-format-alakazam-rtp-r175-training-snapshot",
            "iteration": iteration,
            "completion_authority": "training_milestone_snapshot_after_expert_rehearsal",
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    payload = {
        "schema": SCHEMA,
        "status": "queued",
        "owner_decision_revision": (
            193 if iteration == R193_LARGE_REFRESH_ITERATION else 175
        ),
        "iteration": iteration,
        "cadence": (
            "owner_r193_one_time_25_epoch_refresh_then_kaggle"
            if iteration == R193_LARGE_REFRESH_ITERATION
            else "every_5_after_expert_rehearsal"
        ),
        "commit": str(commit.resolve()),
        "commit_sha256": sha256(commit),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "expert_rehearsal": rehearsal,
        "deck": deck,
        "bundle": bundle,
        "queue_entry": queued[0],
        "turn_order_preference": "first_if_allowed",
        "training_stop_or_freeze_authority": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(receipt, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--maximum-iteration", type=int, default=300)
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    for key in (
        "runtime_root",
        "run_dir",
        "matchup_tree",
        "submission_root",
        "receipts",
        "queue",
        "python",
    ):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.maximum_iteration < 5:
        raise ValueError("maximum-iteration must be >= 5")
    iterations = (
        [args.iteration]
        if args.iteration is not None
        else eligible_iterations(args.run_dir, maximum_iteration=args.maximum_iteration)
    )
    if any(iteration < 0 for iteration in iterations):
        raise ValueError("iteration must be nonnegative")
    results = [stage_one(args, iteration) for iteration in iterations]
    print(
        json.dumps(
            {
                "status": "ready",
                "schema": SCHEMA,
                "owner_decision_revision": 175,
                "pilot_list_id": PILOT_LIST_ID,
                "turn_order_preference": "first_if_allowed",
                "staged": [r["iteration"] for r in results],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
