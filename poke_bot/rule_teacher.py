"""Public-information rule-teacher demonstrations for neural distillation.

The final competition agent remains neural-only.  This module executes a
strong public rule policy in the competition simulator, records only the
teacher seat's public observations and chosen legal actions, and writes the
same game-bounded JSONL schema consumed by :mod:`poke_bot.dataset`.

Collection is deliberately crash resumable and memory bounded.  Worker
results are committed through :class:`poke_bot.replay_writer.OrderedReplayWriter`
so a process restart cannot duplicate a game or leave a partial JSON record.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

RULE_TEACHER_SCHEMA = "poke_bot.rule_teacher_corpus/v1"


def deck_digest(cards: Iterable[int]) -> str:
    payload = ",".join(str(int(card_id)) for card_id in cards).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_rule_teacher_jobs(
    *,
    games: int,
    seed: int,
    job_offset: int,
    teacher_spec: dict[str, Any],
    opponent_spec: dict[str, Any],
    teacher_deck: list[int],
    archetype: str,
    outcome_filter: str,
    timeout_s: int,
) -> list[dict[str, Any]]:
    """Build a deterministic, exactly seat-balanced teacher schedule."""
    if int(games) < 2 or int(games) % 2:
        raise ValueError("rule-teacher games must be an even integer >= 2")
    if len(teacher_deck) != 60:
        raise ValueError("rule-teacher deck must contain exactly 60 cards")
    if outcome_filter not in {"wins", "all"}:
        raise ValueError("outcome_filter must be 'wins' or 'all'")
    if int(timeout_s) < 1:
        raise ValueError("timeout_s must be positive")

    digest = deck_digest(teacher_deck)
    jobs: list[dict[str, Any]] = []
    for local_index in range(int(games)):
        global_id = int(job_offset) + local_index
        jobs.append(
            {
                "job_index": local_index,
                "job_id": global_id,
                "seed": int(seed) + global_id,
                "teacher_seat": global_id % 2,
                "teacher_spec": dict(teacher_spec),
                "opponent_spec": dict(opponent_spec),
                "teacher_deck": [int(card_id) for card_id in teacher_deck],
                "teacher_deck_digest": digest,
                "archetype": str(archetype),
                "outcome_filter": str(outcome_filter),
                "timeout_s": int(timeout_s),
            }
        )
    return jobs


def _legal_action(obs: dict[str, Any], action: list[int]) -> bool:
    select = obs.get("select") if isinstance(obs, dict) else None
    if not isinstance(select, dict):
        return False
    options = list(select.get("option") or [])
    lo = max(0, int(select.get("minCount", 0) or 0))
    hi = min(len(options), int(select.get("maxCount", len(options)) or 0))
    return (
        lo <= len(action) <= hi
        and len(action) == len(set(action))
        and all(0 <= int(index) < len(options) for index in action)
    )


def _worker_rule_teacher_game(job: dict[str, Any]) -> dict[str, Any]:
    """Play one game and return one public teacher-seat trajectory."""
    from .agent import install_quiet_stdout, play_game
    from .baselines_runtime import (
        load_baseline_agent,
        resolve_baseline_spec_payload,
    )
    from .blackwell_heads import attach_blackwell_strategy_labels
    from .replay_import import _strip_opp_private

    install_quiet_stdout(False)
    job_index = int(job["job_index"])
    job_id = int(job["job_id"])
    teacher_seat = int(job["teacher_seat"])
    started = time.perf_counter()
    base = {
        "job_index": job_index,
        "job_id": job_id,
        "teacher_seat": teacher_seat,
        "ok": False,
        "record_written": False,
        "teacher_won": False,
        "winner": 2,
        "steps": 0,
        "decisions": 0,
        "wall_s": 0.0,
        "error": None,
        "record_json": None,
    }
    try:
        teacher_spec = resolve_baseline_spec_payload(
            dict(job["teacher_spec"]), require_content_identity=True
        )
        opponent_spec = resolve_baseline_spec_payload(
            dict(job["opponent_spec"]), require_content_identity=True
        )
        teacher_fn, _teacher_native_deck = load_baseline_agent(teacher_spec)
        opponent_fn, opponent_deck = load_baseline_agent(opponent_spec)
        teacher_deck = [int(card_id) for card_id in job["teacher_deck"]]
        if deck_digest(teacher_deck) != str(job["teacher_deck_digest"]):
            raise ValueError("teacher deck content identity changed")

        captured: list[dict[str, Any]] = []

        def teacher(obs: dict[str, Any]) -> list[int]:
            masked, aux, report = _strip_opp_private(obs)
            if not report.ok:
                raise RuntimeError(
                    "hidden-state guard violation: "
                    + "; ".join(report.violations)
                )
            action = [int(index) for index in teacher_fn(obs)]
            if not _legal_action(masked, action):
                raise RuntimeError("teacher returned an illegal action")
            aux_clean = {key: value for key, value in aux.items() if value is not None}
            aux_clean["opp_archetype"] = str(opponent_spec.id)
            aux_clean["opp_agent"] = str(opponent_spec.id)
            captured.append(
                {
                    "observation": masked,
                    "action": action,
                    "env_step": len(captured),
                    "aux_labels": aux_clean,
                }
            )
            return action

        def on_timeout(_signum, _frame):
            raise TimeoutError(
                f"rule-teacher game exceeded {int(job['timeout_s'])}s"
            )

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, on_timeout)
            signal.alarm(int(job["timeout_s"]))
        try:
            if teacher_seat == 0:
                outcome = play_game(teacher, opponent_fn, teacher_deck, opponent_deck)
            else:
                outcome = play_game(opponent_fn, teacher, opponent_deck, teacher_deck)
        finally:
            if had_alarm:
                signal.alarm(0)

        wall_s = time.perf_counter() - started
        if outcome.get("failed_seat") is not None:
            failed_seat = int(outcome["failed_seat"])
            attribution = "teacher" if failed_seat == teacher_seat else "opponent"
            return {
                **base,
                "wall_s": wall_s,
                "steps": int(outcome.get("steps", 0)),
                "error": f"{attribution} failure: {outcome.get('error')}",
            }
        if outcome.get("incomplete"):
            return {
                **base,
                "wall_s": wall_s,
                "steps": int(outcome.get("steps", 0)),
                "error": "incomplete game",
            }
        if not captured:
            return {**base, "wall_s": wall_s, "error": "no teacher decisions"}

        attach_blackwell_strategy_labels(captured)
        winner = int(outcome["winner"])
        value = 0.0 if winner == 2 else (1.0 if winner == teacher_seat else -1.0)
        teacher_won = value > 0.0
        write_record = str(job["outcome_filter"]) == "all" or teacher_won
        provenance = {
            "schema": RULE_TEACHER_SCHEMA,
            "trusted": True,
            "target_source": "observed_public_rule_teacher_action",
            "search_mode": "none",
            "teacher_id": str(teacher_spec.id),
            "teacher_content_digest": str(job["teacher_spec"]["content_digest"]),
            "opponent_id": str(opponent_spec.id),
            "opponent_content_digest": str(job["opponent_spec"]["content_digest"]),
            "teacher_deck_digest": str(job["teacher_deck_digest"]),
            "job_id": job_id,
            "teacher_seat": teacher_seat,
            "simulator": "competition_libcg",
            "engine_seedable": False,
            "neural_inference_only": True,
        }
        record = {
            "episode_id": f"rule-teacher-{job_id:08d}-seat{teacher_seat}",
            "seat": teacher_seat,
            "archetype": str(job["archetype"]),
            "opp_archetype": str(opponent_spec.id),
            "deck": teacher_deck,
            "value": value,
            "steps": captured,
            "policy_targets": [None] * len(captured),
            "info_set_ok": True,
            "source": "rule_teacher_public_behavior",
            "target_provenance": provenance,
        }
        return {
            **base,
            "ok": True,
            "record_written": write_record,
            "teacher_won": teacher_won,
            "winner": winner,
            "steps": int(outcome.get("steps", 0)),
            "decisions": len(captured),
            "wall_s": wall_s,
            "record_json": (
                json.dumps(record, separators=(",", ":")) if write_record else None
            ),
        }
    except BaseException as exc:  # every scheduled job remains accounted for
        return {
            **base,
            "wall_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def result_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Return the compact journal row, excluding the potentially huge replay."""
    return {
        key: value
        for key, value in result.items()
        if key not in {"record_json", "job_index"}
    }


def summarize_journal(path: Path) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    seats: dict[int, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    wall_s = 0.0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result = dict(row.get("result") or {})
            totals["jobs"] += 1
            seat = int(result.get("teacher_seat", -1))
            ok = bool(result.get("ok"))
            if ok:
                totals["valid_games"] += 1
                seats[seat]["games"] += 1
                totals["decisions"] += int(result.get("decisions") or 0)
                if result.get("teacher_won"):
                    totals["wins"] += 1
                    seats[seat]["wins"] += 1
                elif int(result.get("winner", 2)) == 2:
                    totals["draws"] += 1
                    seats[seat]["draws"] += 1
                else:
                    totals["losses"] += 1
                    seats[seat]["losses"] += 1
                if row.get("record_written"):
                    totals["records_written"] += 1
            else:
                totals["failed_games"] += 1
                errors[str(result.get("error") or "unknown")] += 1
            wall_s += float(result.get("wall_s") or 0.0)
    valid = int(totals["valid_games"])
    return {
        **{key: int(value) for key, value in totals.items()},
        "win_rate": float(totals["wins"] / valid) if valid else None,
        "seat": {
            str(seat): {
                **{key: int(value) for key, value in counter.items()},
                "win_rate": (
                    float(counter["wins"] / counter["games"])
                    if counter["games"]
                    else None
                ),
            }
            for seat, counter in sorted(seats.items())
        },
        "worker_wall_seconds": wall_s,
        "errors": dict(errors.most_common(20)),
    }


def validate_rule_teacher_corpus(
    path: Path,
    *,
    expected_deck_digest: str,
    expected_teacher: str,
    expected_opponent: str,
    require_wins: bool = True,
) -> dict[str, Any]:
    """Stream-validate public-state, action, identity, and conversion contracts."""
    from .dataset import convert_record

    records = decisions = 0
    drops: Counter[str] = Counter()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            provenance = dict(record.get("target_provenance") or {})
            if provenance.get("schema") != RULE_TEACHER_SCHEMA:
                raise ValueError("rule-teacher record has an unknown schema")
            if provenance.get("teacher_id") != expected_teacher:
                raise ValueError("rule-teacher identity mismatch")
            if provenance.get("opponent_id") != expected_opponent:
                raise ValueError("rule-teacher opponent mismatch")
            if deck_digest(record.get("deck") or []) != expected_deck_digest:
                raise ValueError("rule-teacher deck mismatch")
            if require_wins and float(record.get("value", 0.0)) <= 0.0:
                raise ValueError("win-filtered teacher corpus contains a non-win")
            sequence, reason, _details = convert_record(
                record, max_context=320, verify_info_set=True
            )
            if sequence is None:
                drops[str(reason or "unknown")] += 1
                continue
            records += 1
            decisions += len(sequence.decisions)
    if drops:
        raise ValueError(f"rule-teacher corpus conversion drops: {dict(drops)}")
    if records < 1 or decisions < 1:
        raise ValueError("rule-teacher corpus is empty")
    return {
        "records": records,
        "decisions": decisions,
        "info_set_ok": True,
        "conversion_drops": {},
    }


def resolve_protected_teacher_corpus(
    report_path: Path, *, corpus_override: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Resolve and cryptographically verify a protected teacher artifact."""
    report_file = Path(report_path).expanduser().resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("schema") != RULE_TEACHER_SCHEMA:
        raise ValueError("unknown rule-teacher report schema")
    if report.get("protected") is not True or report.get("prune_policy") != "never":
        raise ValueError("rule-teacher corpus is not marked immutable/protected")
    configuration = dict(report.get("configuration") or {})
    if configuration.get("final_agent_runtime") != "neural_only":
        raise ValueError("rule-teacher artifact lacks the neural-only runtime contract")
    corpus_row = dict(report.get("corpus") or {})
    corpus = (
        Path(corpus_override).expanduser().resolve()
        if corpus_override is not None
        else Path(str(corpus_row.get("path") or "")).expanduser().resolve()
    )
    if not corpus.is_file():
        raise FileNotFoundError(f"missing rule-teacher corpus: {corpus}")
    expected = str(corpus_row.get("digest") or "")
    actual = file_digest(corpus)
    if not expected.startswith("sha256:") or actual != expected:
        raise ValueError(
            f"rule-teacher corpus digest mismatch: expected={expected} actual={actual}"
        )
    validation = dict(report.get("validation") or {})
    if (
        validation.get("info_set_ok") is not True
        or int(validation.get("records") or 0) < 1
        or int(validation.get("decisions") or 0) < 1
        or dict(validation.get("conversion_drops") or {})
    ):
        raise ValueError("rule-teacher report did not pass its data-quality gate")
    return corpus, report


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, target)
