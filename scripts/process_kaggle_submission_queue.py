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
from datetime import datetime, timedelta, timezone
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
MINIMUM_SUBMISSION_SPACING_HOURS = 4
AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"
DEFAULT_AUTHORIZATION = Path(
    "/home/inzi/.config/pokebot/kaggle-submission-authorization.json"
)
STANDING_OWNER_DECISION = "GOAL.md#/decision-ledger/revision-18"


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


def _submission_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _submission_times(
    submissions: list[dict[str, str]],
    queue: list[dict[str, Any]],
) -> list[datetime]:
    """Return newest-first times for distinct logical submissions.

    Kaggle rows and local queue rows describe the same upload after
    reconciliation.  Keying by submission id or checkpoint-bound label keeps
    that upload from occupying both the newest and second-newest positions.
    """

    logical: dict[tuple[str, str], datetime] = {}
    for index, row in enumerate(submissions):
        timestamp = _submission_time(row.get("date"))
        if timestamp is None:
            continue
        reference = str(row.get("ref") or "").strip()
        label = str(row.get("description") or "").strip()
        key = (
            ("id", reference)
            if reference
            else ("label", label)
            if label
            else ("remote", str(index))
        )
        logical[key] = max(timestamp, logical.get(key, timestamp))
    for index, row in enumerate(queue):
        timestamp = _submission_time(row.get("submitted_at"))
        if timestamp is None:
            continue
        reference = str(row.get("submission_id") or "").strip()
        label = str(row.get("label") or "").strip()
        key = (
            ("id", reference)
            if reference
            else ("label", label)
            if label
            else ("queue", str(index))
        )
        logical[key] = max(timestamp, logical.get(key, timestamp))
    return sorted(logical.values(), reverse=True)


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
        "search_config": str(entry.get("search_config_checksum") or ""),
        "belief_decks": str(entry.get("belief_decks_checksum") or ""),
        "turn_order_preference": str(
            entry.get("turn_order_preference") or "first_if_allowed"
        ),
    }
    search_assets_packaged = bool(entry.get("search_assets_packaged", True))
    expected_search_assets = (
        expected["search_config"].startswith("sha256:")
        and expected["belief_decks"].startswith("sha256:")
        if search_assets_packaged
        else expected["search_config"] == "" and expected["belief_decks"] == ""
    )
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
        or not expected_search_assets
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
        search_config_bytes = (
            member_bytes("search_config.json") if search_assets_packaged else None
        )
        belief_decks_bytes = (
            member_bytes("belief_decks.json") if search_assets_packaged else None
        )
        if not search_assets_packaged and (
            "search_config.json" in members or "belief_decks.json" in members
        ):
            raise RuntimeError("direct-policy queue bundle contains search assets")
        turn_order_member = members.get("turn_order_profile.json")
        turn_order_bytes = (
            archive.extractfile(turn_order_member).read()
            if turn_order_member is not None
            else b""
        )
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
    try:
        search_config = (
            json.loads(search_config_bytes or b"") if search_assets_packaged else {}
        )
        belief_decks = (
            json.loads(belief_decks_bytes or b"") if search_assets_packaged else {}
        )
        turn_order_profile = (
            json.loads(turn_order_bytes)
            if turn_order_bytes
            else {
                "schema": "legacy_default_first",
                "turn_order_preference": "first_if_allowed",
            }
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("queued belief-MCTS assets are invalid JSON") from exc
    hypotheses = belief_decks.get("deck_lists") or []
    search_contract_valid = not search_assets_packaged or (
        search_config.get("schema")
        == "poke_bot.submission_search_config/v1"
        and search_config.get("enabled") is False
        and search_config.get("algorithm")
        == "public_history_root_sampled_belief_mcts"
        and search_config.get("leaf_evaluator")
        == "trained_checkpoint_policy_value_head"
        and search_config.get("leaf_evaluator_checkpoint")
        == "submission_model_pt"
        and search_config.get("require_trained_state_evaluator") is True
        and float(search_config.get("hard_cap_s") or 0) == 600.0
        and float(search_config.get("internal_deadline_s") or 0) == 540.0
        and float(search_config.get("final_greedy_reserve_s") or 0) == 20.0
        and float(search_config.get("total_search_budget_s") or 0) == 400.0
        and float(search_config.get("baseline_call_s") or 0) == 0.2
        and int(search_config.get("maximum_calls") or 0) == 340
        and int(search_config.get("expected_search_decisions") or 0) == 64
        and float(search_config.get("maximum_move_s") or 0) == 4.0
        and float(search_config.get("minimum_move_s") or 0) == 0.5
        and int(search_config.get("minimum_sims") or 0) == 50
        and int(search_config.get("maximum_sims") or 0) == 50
        and search_config.get("search_failure_behavior")
        == "greedy_current_decision_then_retry"
        and search_config.get("game_wide_greedy_only_for_time_budget") is True
        and float(search_config.get("safety_factor") or 0) == 0.8
        and search_config.get("fallback") == "frozen_model_greedy_policy"
        and search_config.get("oracle_inputs_allowed") is False
        and belief_decks.get("schema")
        == "poke_bot.submission_belief_decks/v1"
        and belief_decks.get("anonymous") is True
        and belief_decks.get("contains_opponent_identity") is False
        and belief_decks.get("deck_count") == len(hypotheses)
        and len(hypotheses) >= 8
        and all(
            len(deck) == 60 and all(int(card) > 0 for card in deck)
            for deck in hypotheses
        )
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
        "search_config": (
            not search_assets_packaged
            or "sha256:" + hashlib.sha256(search_config_bytes or b"").hexdigest()
            == expected["search_config"]
        ),
        "belief_decks": (
            not search_assets_packaged
            or "sha256:" + hashlib.sha256(belief_decks_bytes or b"").hexdigest()
            == expected["belief_decks"]
        ),
        "belief_mcts_contract": search_contract_valid,
        "turn_order_preference": (
            turn_order_profile.get("turn_order_preference")
            == expected["turn_order_preference"]
            and expected["turn_order_preference"]
            in {"first_if_allowed", "second_if_allowed"}
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "queued submission contains the wrong specialist payload: "
            + ",".join(failed)
        )


def _ensure_automatic_one_shot_authorization(
    entry: dict[str, Any],
    authorization_path: Path,
    *,
    owner_decision_source: str = STANDING_OWNER_DECISION,
) -> dict[str, Any]:
    """Materialize one exact standing-authorized grant immediately before upload."""

    authorization_path = authorization_path.expanduser().resolve()
    now = _now()
    nonce = (
        f"{entry['specialist_id']}-iter{int(entry['iteration'])}-"
        f"copy{int(entry['copy_number'])}-automatic-"
        f"{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    authorization = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "approval_text": (
            "When the training cycle completes on a deck always submit "
            "with a one shot."
        ),
        "standing_owner_decision_source": str(owner_decision_source),
        "remaining_uses": 1,
        "nonce": nonce,
        "expires_at_epoch": time.time() + 3600.0,
        "competition": str(entry.get("competition") or ""),
        "file_sha256": str(entry.get("file_sha256") or ""),
        "message": str(entry.get("label") or ""),
        "specialist_id": str(entry.get("specialist_id") or ""),
        "frozen_checkpoint_checksum": str(
            entry.get("checkpoint_checksum") or ""
        ),
        "submission_file_checksum": str(entry.get("file_sha256") or ""),
        "turn_order_preference": str(
            entry.get("turn_order_preference") or "first_if_allowed"
        ),
    }
    identity_fields = (
        "schema",
        "explicit_user_approval",
        "remaining_uses",
        "competition",
        "file_sha256",
        "message",
        "specialist_id",
        "frozen_checkpoint_checksum",
        "submission_file_checksum",
        "turn_order_preference",
    )
    if authorization_path.exists():
        existing = _read_json(authorization_path)
        legacy_exact_repair = (
            existing.get("schema") == AUTH_SCHEMA
            and existing.get("explicit_user_approval") is True
            and int(existing.get("remaining_uses") or 0) == 1
            and existing.get("competition") == authorization["competition"]
            and existing.get("file_sha256") == authorization["file_sha256"]
            and existing.get("message") == authorization["message"]
            and existing.get("conditional_checkpoint_digest")
            == authorization["frozen_checkpoint_checksum"]
            and float(existing.get("expires_at_epoch") or 0.0) >= time.time()
        )
        if legacy_exact_repair:
            return existing
        if (
            any(existing.get(key) != authorization.get(key) for key in identity_fields)
            or float(existing.get("expires_at_epoch") or 0.0) < time.time()
        ):
            raise RuntimeError("another Kaggle one-shot authorization is active")
        return existing
    _atomic_json(authorization_path, authorization)
    return authorization


def process_once(
    *,
    queue_path: Path,
    kaggle: Path,
    default_competition: str,
    authorization_path: Path = DEFAULT_AUTHORIZATION,
    required_owner_decision_source: str = STANDING_OWNER_DECISION,
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
            minimum_spacing_hours = int(
                payload.get(
                    "minimum_hours_between_submissions",
                    MINIMUM_SUBMISSION_SPACING_HOURS,
                )
            )
            if minimum_spacing_hours != MINIMUM_SUBMISSION_SPACING_HOURS:
                raise RuntimeError("Kaggle submission spacing policy changed")
            payload["minimum_hours_between_submissions"] = minimum_spacing_hours
            automatic_authorization = bool(
                payload.get(
                    "automatic_one_shot_authorization_on_training_complete",
                    False,
                )
            )
            if automatic_authorization:
                if int(payload.get("one_shot_authorization_uses", -1)) != 1:
                    raise RuntimeError("Kaggle one-shot authorization count changed")
                if (
                    str(payload.get("standing_owner_decision_source") or "")
                    != str(required_owner_decision_source)
                ):
                    raise RuntimeError(
                        "Kaggle standing owner authorization source changed"
                    )
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
            submission_times = _submission_times(submissions, queue)
            last_submission_at = (
                submission_times[0] if submission_times else None
            )
            spacing_anchor_submission_at = (
                submission_times[1] if len(submission_times) >= 2 else None
            )
            next_submission_eligible_at = (
                spacing_anchor_submission_at
                + timedelta(hours=minimum_spacing_hours)
                if spacing_anchor_submission_at is not None
                else None
            )
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
                    "minimum_hours_between_submissions": minimum_spacing_hours,
                    "known_submissions_used_today": known_used,
                    "quota_date": quota_date,
                    "next_reset_time": None,
                    "last_submission_at": (
                        last_submission_at.isoformat()
                        if last_submission_at is not None
                        else None
                    ),
                    "spacing_anchor_policy": (
                        "second_most_recent_logical_submission"
                    ),
                    "spacing_anchor_submission_at": (
                        spacing_anchor_submission_at.isoformat()
                        if spacing_anchor_submission_at is not None
                        else None
                    ),
                    "next_submission_eligible_at": (
                        next_submission_eligible_at.isoformat()
                        if next_submission_eligible_at is not None
                        else None
                    ),
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
            if (
                next_submission_eligible_at is not None
                and now < next_submission_eligible_at
            ):
                _save_queue(queue_path, payload)
                return {
                    "status": "spacing_wait",
                    "next_submission_eligible_at": (
                        next_submission_eligible_at.isoformat()
                    ),
                    "remaining_seconds": int(
                        (next_submission_eligible_at - now).total_seconds()
                    ),
                }
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
            if automatic_authorization:
                try:
                    authorization = _ensure_automatic_one_shot_authorization(
                        entry,
                        authorization_path,
                        owner_decision_source=required_owner_decision_source,
                    )
                except RuntimeError as exc:
                    entry["attempt_started_at"] = None
                    entry["attempt_quota_date"] = None
                    _save_queue(queue_path, payload)
                    return {
                        "status": "authorization_wait",
                        "reason": str(exc),
                    }
                entry["one_shot_authorization_nonce"] = authorization["nonce"]
                entry["one_shot_authorization_created_at"] = now.isoformat()
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
                submitted_at = _now()
                entry["queue_status"] = "submitted"
                entry["submitted_at"] = submitted_at.isoformat()
                entry["failure_reason"] = None
                quota["known_submissions_used_today"] = min(limit, used + 1)
                quota["quota_exhausted"] = used + 1 >= limit
                quota["last_submission_at"] = submitted_at.isoformat()
                quota["spacing_anchor_policy"] = (
                    "second_most_recent_logical_submission"
                )
                quota["spacing_anchor_submission_at"] = (
                    last_submission_at.isoformat()
                    if last_submission_at is not None
                    else None
                )
                quota["next_submission_eligible_at"] = (
                    (
                        last_submission_at
                        + timedelta(hours=minimum_spacing_hours)
                    ).isoformat()
                    if last_submission_at is not None
                    else None
                )
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
    parser.add_argument(
        "--authorization",
        type=Path,
        default=DEFAULT_AUTHORIZATION,
    )
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument(
        "--required-owner-decision-source",
        default=STANDING_OWNER_DECISION,
    )
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
                authorization_path=args.authorization,
                required_owner_decision_source=args.required_owner_decision_source,
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
