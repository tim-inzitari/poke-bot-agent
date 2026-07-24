#!/usr/bin/env python3
"""Process frozen-specialist Kaggle submissions oldest-first.

Exactly one process owns the queue lock.  Before any upload it reconciles the
queue against Kaggle by the unique, checkpoint-bound label and counts today's
submissions.  A reported daily-limit error leaves the copy pending and blocks
further attempts until the quota date changes; training and specialist
handoffs are never paused.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from typing import Any

from poke_bot.pure_rl.model_registry import sha256
from scripts.handle_passed_gate import SUBMISSION_QUEUE_SCHEMA


QUOTA_ERROR_TOKENS = (
    "daily submission limit",
    "maximum number of submissions",
    "too many submissions",
    "submission quota",
)
TERMINAL_KAGGLE_FAILURES = {"ERROR", "CANCELLED", "FAILED"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"submission queue is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("submission queue is not an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _list_submissions(kaggle: Path, competition: str) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            str(kaggle),
            "competitions",
            "submissions",
            competition,
            "-v",
            "--page-size",
            "200",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(
            "cannot query Kaggle submissions: "
            + (completed.stderr or completed.stdout).strip()
        )
    return [
        {str(key): str(value or "") for key, value in row.items()}
        for row in csv.DictReader(io.StringIO(completed.stdout))
    ]


def _status_name(value: str) -> str:
    return str(value or "").rsplit(".", 1)[-1].upper()


def _score(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reconcile(
    queue: list[dict[str, Any]],
    submissions: list[dict[str, str]],
) -> None:
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in submissions:
        by_label.setdefault(str(row.get("description") or ""), []).append(row)
    for entry in queue:
        matches = by_label.get(str(entry.get("label") or ""), [])
        if len(matches) > 1:
            entry["queue_status"] = "failed"
            entry["failure_reason"] = "duplicate checkpoint-bound Kaggle labels"
            continue
        if not matches:
            continue
        match = matches[0]
        status = _status_name(match.get("status", ""))
        entry["submission_id"] = int(match["ref"]) if match.get("ref") else None
        entry["submitted_at"] = match.get("date") or entry.get("submitted_at")
        entry["returned_score"] = _score(match.get("publicScore", ""))
        entry["kaggle_status"] = status
        if status == "COMPLETE":
            entry["queue_status"] = "accepted"
            entry["failure_reason"] = None
        elif status in TERMINAL_KAGGLE_FAILURES:
            entry["queue_status"] = "failed"
            entry["failure_reason"] = f"Kaggle status {status}"
        else:
            entry["queue_status"] = "submitted"
            entry["failure_reason"] = None


def _quota_error(output: str) -> bool:
    lowered = str(output or "").casefold()
    return any(token in lowered for token in QUOTA_ERROR_TOKENS)


def _save_queue(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at_utc"] = _now().isoformat()
    _atomic_json(path, payload)


def _canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _verify_queued_bundle_identity(entry: dict[str, Any]) -> None:
    """Re-open the exact upload and bind its model and deck before network I/O."""

    file_path = Path(str(entry.get("file") or "")).expanduser().resolve()
    expected = {
        "bundle": str(entry.get("file_sha256") or ""),
        "model": str(entry.get("model_checksum") or ""),
        "checkpoint": str(entry.get("checkpoint_checksum") or ""),
        "deck_file": str(entry.get("deck_file_checksum") or ""),
        "deck_cards": str(entry.get("deck_cards_checksum") or ""),
        "representatives": str(entry.get("representatives_checksum") or ""),
        "matchup_tree": str(entry.get("matchup_tree_checksum") or ""),
    }
    if (
        not file_path.is_file()
        or sha256(file_path) != expected["bundle"]
        or expected["model"] != expected["checkpoint"]
        or not all(
            digest.startswith("sha256:")
            for digest in (
                expected["model"],
                expected["deck_file"],
                expected["deck_cards"],
                expected["representatives"],
                expected["matchup_tree"],
            )
        )
    ):
        raise RuntimeError("queued submission identity is incomplete")
    with tarfile.open(file_path, "r:gz") as archive:
        members = {
            member.name.removeprefix("./"): member
            for member in archive.getmembers()
            if member.isfile()
        }

        def member_bytes(name: str) -> bytes:
            member = members.get(name)
            stream = archive.extractfile(member) if member is not None else None
            if stream is None:
                raise RuntimeError(f"queued submission is missing {name}")
            return stream.read()

        model_bytes = member_bytes("model.pt")
        deck_bytes = member_bytes("deck.csv")
        member_bytes("main.py")
        member_bytes("cg/api.py")
        matchup_tree_bytes = member_bytes("matchup_tree.json")
    actual_model = "sha256:" + hashlib.sha256(model_bytes).hexdigest()
    actual_deck_file = "sha256:" + hashlib.sha256(deck_bytes).hexdigest()
    try:
        cards = [
            int(line.split(",", 1)[0])
            for raw in deck_bytes.decode("utf-8").splitlines()
            if (line := raw.strip()) and not line.startswith("#")
        ]
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("queued submission deck is invalid") from exc
    if len(cards) != 60:
        raise RuntimeError(
            f"queued submission deck must contain 60 cards, got {len(cards)}"
        )
    checks = {
        "model": actual_model == expected["model"],
        "checkpoint": actual_model == expected["checkpoint"],
        "deck_file": actual_deck_file == expected["deck_file"],
        "deck_cards": _canonical_digest(cards) == expected["deck_cards"],
        "matchup_tree": (
            "sha256:" + hashlib.sha256(matchup_tree_bytes).hexdigest()
            == expected["matchup_tree"]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "queued submission contains the wrong specialist payload: "
            + ",".join(failed)
        )


def process_once(
    *,
    queue_path: Path,
    kaggle: Path,
    default_competition: str,
) -> dict[str, Any]:
    queue_path = queue_path.expanduser().resolve()
    process_lock_path = queue_path.with_suffix(queue_path.suffix + ".processor.lock")
    process_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with process_lock_path.open("a+", encoding="utf-8") as process_lock:
        try:
            fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "another_processor_active"}
        queue_lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
        with queue_lock_path.open("a+", encoding="utf-8") as queue_lock:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX)
            payload = _read_json(queue_path)
            if not payload:
                return {"status": "queue_absent"}
            if payload.get("schema") != SUBMISSION_QUEUE_SCHEMA:
                raise RuntimeError("submission queue schema changed")
            queue = [dict(row) for row in (payload.get("queue") or [])]
            limit = int(payload.get("daily_submission_limit", 5))
            if limit != 5:
                raise RuntimeError("Kaggle daily submission limit changed")
            competition = str(
                next(
                    (
                        row.get("competition")
                        for row in queue
                        if row.get("competition")
                    ),
                    default_competition,
                )
            )
            submissions = _list_submissions(kaggle, competition)
            now = _now()
            quota_date = now.date().isoformat()
            used = sum(
                1
                for row in submissions
                if str(row.get("date") or "").startswith(quota_date)
            )
            _reconcile(queue, submissions)
            quota = dict(payload.get("quota") or {})
            prior_quota_date = str(quota.get("quota_date") or "")
            if prior_quota_date != quota_date:
                quota["last_quota_error"] = None
            exhausted = used >= limit or (
                prior_quota_date == quota_date
                and bool(quota.get("last_quota_error"))
            )
            known_used = limit if exhausted and used < limit else used
            quota.update(
                {
                    "daily_submission_limit": limit,
                    "known_submissions_used_today": known_used,
                    "quota_date": quota_date,
                    "next_reset_time": None,
                    "quota_exhausted": exhausted,
                    "checked_at_utc": now.isoformat(),
                }
            )
            payload["quota"] = quota
            payload["queue"] = queue
            pending = sorted(
                (
                    row
                    for row in queue
                    if row.get("queue_status") == "pending"
                ),
                key=lambda row: (
                    str(row.get("queued_at") or ""),
                    str(row.get("specialist_id") or ""),
                    int(row.get("copy_number", -1)),
                ),
            )
            if not pending:
                _save_queue(queue_path, payload)
                return {"status": "idle", "used": used, "limit": limit}
            if exhausted:
                _save_queue(queue_path, payload)
                return {"status": "quota_exhausted", "used": used, "limit": limit}
            entry = pending[0]
            file_path = Path(str(entry.get("file") or "")).expanduser().resolve()
            try:
                _verify_queued_bundle_identity(entry)
            except (OSError, RuntimeError, tarfile.TarError):
                entry["queue_status"] = "failed"
                entry["failure_reason"] = (
                    "queued submission model/deck identity validation failed"
                )
                _save_queue(queue_path, payload)
                return {"status": "failed_identity"}
            if (
                entry.get("attempt_started_at")
                and not entry.get("submission_id")
            ):
                entry["queue_status"] = "failed"
                entry["failure_reason"] = (
                    "prior upload outcome is unknown; automatic retry refused"
                )
                _save_queue(queue_path, payload)
                return {"status": "failed_unknown_prior_attempt"}
            entry["attempt_started_at"] = now.isoformat()
            entry["attempt_quota_date"] = quota_date
            entry["attempt_count"] = int(entry.get("attempt_count") or 0) + 1
            _save_queue(queue_path, payload)
            completed = subprocess.run(
                [
                    str(kaggle),
                    "competitions",
                    "submit",
                    "-c",
                    str(entry.get("competition") or competition),
                    "-f",
                    str(file_path),
                    "-m",
                    str(entry["label"]),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
            output = "\n".join((completed.stdout, completed.stderr)).strip()
            if completed.returncode == 0:
                entry["queue_status"] = "submitted"
                entry["submitted_at"] = _now().isoformat()
                entry["failure_reason"] = None
                quota["known_submissions_used_today"] = min(limit, used + 1)
                quota["quota_exhausted"] = used + 1 >= limit
            elif _quota_error(output):
                entry["attempt_started_at"] = None
                entry["attempt_quota_date"] = quota_date
                entry["retry_count"] = int(entry.get("retry_count") or 0) + 1
                entry["failure_reason"] = None
                quota["quota_exhausted"] = True
                quota["known_submissions_used_today"] = limit
                quota["last_quota_error"] = output[-2000:]
            else:
                entry["queue_status"] = "failed"
                entry["failure_reason"] = output[-2000:] or (
                    f"Kaggle CLI exited {completed.returncode}"
                )
            payload["quota"] = quota
            _save_queue(queue_path, payload)
            return {
                "status": str(entry["queue_status"]),
                "specialist_id": entry.get("specialist_id"),
                "copy_number": entry.get("copy_number"),
                "returncode": completed.returncode,
            }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/kaggle-submission-queue.json"
        ),
    )
    parser.add_argument(
        "--kaggle",
        type=Path,
        default=Path("/home/inzi/miniconda3/envs/poke-bot-agent/bin/kaggle"),
    )
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    while True:
        try:
            result = process_once(
                queue_path=args.queue,
                kaggle=args.kaggle,
                default_competition=args.competition,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "processor_error", "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(max(15.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
