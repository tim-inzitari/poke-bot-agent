"""Compile fail-closed paired realized-win evidence for guide-weight reviews."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any, Iterable

from poke_bot.pure_rl.deck_guide_schedule import (
    GuideWeightState,
    update_after_evaluation,
)


EVIDENCE_SCHEMA = "poke_bot.current_deck_guide_paired_evaluation/v1"
SCHEDULE_SCHEMA = "poke_bot.current_deck_guide_weight_schedule/v1"
MINIMUM_PAIRS = 1000
MINIMUM_MATCHUP_PAIRS = 50
CONFIDENCE_LEVEL = 0.90


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _checkpoint_identity(value: Any, *, variant: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{variant} checkpoint identity is missing")
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    expected = str(value.get("sha256") or "")
    if not path.is_file() or len(expected) != 71 or not expected.startswith("sha256:"):
        raise ValueError(f"{variant} checkpoint identity is invalid")
    actual = file_digest(path)
    if actual != expected:
        raise ValueError(f"{variant} checkpoint checksum mismatch")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _score(value: Any) -> float:
    score = float(value)
    if score not in {0.0, 0.5, 1.0}:
        raise ValueError(f"score must be 0, 0.5, or 1; got {score}")
    return score


def _one_sided_hoeffding_lower(deltas: list[float], confidence: float) -> float:
    if not deltas or not 0.0 < confidence < 1.0:
        raise ValueError("invalid confidence-bound inputs")
    mean = sum(deltas) / len(deltas)
    alpha = 1.0 - confidence
    # Paired score deltas are bounded in [-1, 1], a range of two.
    radius = math.sqrt(2.0 * math.log(1.0 / alpha) / len(deltas))
    return max(-1.0, mean - radius)


def _summarize_pairs(
    pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    *,
    confidence: float,
) -> dict[str, Any]:
    rows = list(pairs)
    deltas = [
        _score(on["score"]) - _score(off["score"])
        for on, off in rows
    ]
    on_scores = [_score(on["score"]) for on, _ in rows]
    off_scores = [_score(off["score"]) for _, off in rows]
    seat_counts = {
        "0": sum(int(on["candidate_seat"]) == 0 for on, _ in rows),
        "1": sum(int(on["candidate_seat"]) == 1 for on, _ in rows),
    }
    return {
        "pairs": len(rows),
        "guide_on_win_rate": sum(on_scores) / len(on_scores),
        "guide_off_win_rate": sum(off_scores) / len(off_scores),
        "realized_win_rate_delta": sum(deltas) / len(deltas),
        "realized_win_rate_delta_lower_confidence_bound": (
            _one_sided_hoeffding_lower(deltas, confidence)
        ),
        "confidence_level": confidence,
        "confidence_method": (
            "one_sided_hoeffding_for_matched_draw_aware_score_deltas"
        ),
        "candidate_seat_counts": seat_counts,
        "first_second_balanced": seat_counts["0"] == seat_counts["1"],
    }


def compile_schedule(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate one immutable paired study and calculate the next fleet weight."""

    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("not a current-deck guide paired-evaluation receipt")
    specialist_id = str(evidence.get("specialist_id") or "").strip().casefold()
    if not specialist_id:
        raise ValueError("paired guide evidence has no specialist identity")
    completed_iteration = int(evidence.get("completed_iteration", -1))
    if completed_iteration < 0 or completed_iteration % 5:
        raise ValueError("guide review must follow a completed five-iteration boundary")
    if (
        evidence.get("training_eligible") is not False
        or evidence.get("replay_eligible") is not False
        or evidence.get("formal_gate") is not False
        or evidence.get("serving_allowed") is not False
        or evidence.get("promotion_allowed") is not False
    ):
        raise ValueError("guide contribution evidence is not isolated")
    guide_on = _checkpoint_identity(
        evidence.get("guide_on_checkpoint"), variant="guide_on"
    )
    guide_off = _checkpoint_identity(
        evidence.get("guide_off_checkpoint"), variant="guide_off"
    )
    if guide_on["sha256"] == guide_off["sha256"]:
        raise ValueError("guide-on and guide-off checkpoints must differ")
    rows = list(evidence.get("rows") or ())
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("paired guide evaluation row is malformed")
        variant = str(raw.get("variant") or "")
        schedule_id = str(raw.get("schedule_id") or "")
        opponent_id = str(raw.get("opponent_id") or "")
        if (
            variant not in {"guide_on", "guide_off"}
            or not schedule_id
            or not opponent_id
            or int(raw.get("candidate_seat", -1)) not in {0, 1}
            or int(raw.get("requested_seed", -1)) < 0
            or raw.get("training_eligible") is not False
            or raw.get("replay_eligible") is not False
            or raw.get("formal_gate") is not False
            or bool(raw.get("invalid"))
            or raw.get("error")
        ):
            raise ValueError("paired guide evaluation row violates isolation")
        expected_digest = guide_on["sha256"] if variant == "guide_on" else guide_off["sha256"]
        if str(raw.get("checkpoint_sha256") or "") != expected_digest:
            raise ValueError("paired guide row checkpoint identity mismatch")
        _score(raw.get("score"))
        variants = grouped.setdefault(schedule_id, {})
        if variant in variants:
            raise ValueError("duplicate paired guide evaluation row")
        variants[variant] = dict(raw)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    by_matchup: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for schedule_id, variants in grouped.items():
        if set(variants) != {"guide_on", "guide_off"}:
            raise ValueError(f"incomplete guide evaluation pair: {schedule_id}")
        on, off = variants["guide_on"], variants["guide_off"]
        exact = ("opponent_id", "candidate_seat", "requested_seed")
        if any(on.get(key) != off.get(key) for key in exact):
            raise ValueError(f"guide pair schedule mismatch: {schedule_id}")
        pair = (on, off)
        pairs.append(pair)
        by_matchup.setdefault(str(on["opponent_id"]), []).append(pair)
    if len(pairs) < MINIMUM_PAIRS:
        raise ValueError(
            f"paired guide review requires at least {MINIMUM_PAIRS} pairs"
        )
    if any(len(matchup_pairs) < MINIMUM_MATCHUP_PAIRS for matchup_pairs in by_matchup.values()):
        raise ValueError("paired guide review has insufficient matchup support")
    overall = _summarize_pairs(pairs, confidence=CONFIDENCE_LEVEL)
    if overall["first_second_balanced"] is not True:
        raise ValueError("paired guide review is not first/second balanced")
    matchup_summary = {
        opponent_id: _summarize_pairs(
            matchup_pairs, confidence=CONFIDENCE_LEVEL
        )
        for opponent_id, matchup_pairs in sorted(by_matchup.items())
    }
    if any(row["first_second_balanced"] is not True for row in matchup_summary.values()):
        raise ValueError("paired guide matchup evidence is not seat-balanced")
    previous = GuideWeightState(
        weight=float(evidence.get("current_weight", -1.0)),
        consecutive_nonpositive_evaluations=int(
            evidence.get("consecutive_nonpositive_evaluations", 0)
        ),
    )
    next_state = update_after_evaluation(
        previous,
        realized_win_rate_delta_lower_confidence_bound=overall[
            "realized_win_rate_delta_lower_confidence_bound"
        ],
    )
    evidence_digest = _digest_bytes(_canonical_bytes(evidence))
    return {
        "schema": SCHEDULE_SCHEMA,
        "status": "ready_for_clean_boundary"
        if next_state.weight != previous.weight
        else "hold",
        "specialist_id": specialist_id,
        "completed_iteration": completed_iteration,
        "earliest_activation_boundary_next_iteration": (
            completed_iteration + 1
        ),
        "application_boundary": (
            "first_available_future_five_iteration_hard_pause"
        ),
        "evidence_sha256": evidence_digest,
        "guide_on_checkpoint": guide_on,
        "guide_off_checkpoint": {
            **guide_off,
            "shadow_only": True,
            "serving_allowed": False,
            "promotion_allowed": False,
        },
        "overall": overall,
        "per_matchup": matchup_summary,
        "previous_state": {
            "weight": previous.weight,
            "consecutive_nonpositive_evaluations": (
                previous.consecutive_nonpositive_evaluations
            ),
        },
        "next_state": {
            "weight": next_state.weight,
            "consecutive_nonpositive_evaluations": (
                next_state.consecutive_nonpositive_evaluations
            ),
        },
        "changed": next_state.weight != previous.weight,
        "clean_boundary_receipt_required": True,
        "training_eligible": False,
        "replay_eligible": False,
        "formal_gate": False,
        "serving_allowed": False,
        "promotion_allowed": False,
        "compiled_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }


def immutable_json(path: str | Path, value: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
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


__all__ = [
    "CONFIDENCE_LEVEL",
    "EVIDENCE_SCHEMA",
    "MINIMUM_MATCHUP_PAIRS",
    "MINIMUM_PAIRS",
    "SCHEDULE_SCHEMA",
    "compile_schedule",
    "file_digest",
    "immutable_json",
]
