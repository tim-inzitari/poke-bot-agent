#!/usr/bin/env python3
"""Run the established policy non-regression gate against frozen teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_bot import checkpoint
from poke_bot.promotion import CheckpointIdentity
from scripts.train_pure_rl import _our_decks, _promotion_eval


SCHEMA = "poke_bot.multi_teacher_core_gameplay_regression/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _teacher_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list((contract.get("core_refresh") or {}).get("teachers") or [])
    if len(rows) < 2:
        raise RuntimeError("core regression requires at least two frozen teachers")
    runtime = dict(contract.get("runtime") or {})
    registry_raw = str(runtime.get("frozen_specialist_registry") or "").strip()
    registry_path = (
        Path(registry_raw).expanduser().resolve() if registry_raw else None
    )
    registry_rows: dict[str, dict[str, Any]] = {}
    if registry_path is not None:
        if not registry_path.is_file():
            raise RuntimeError("frozen specialist inference registry is missing")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("schema") != "poke_bot.frozen_specialist_registry/v1":
            raise RuntimeError("frozen specialist inference registry is invalid")
        registry_rows = {
            str(value.get("specialist_id") or ""): dict(value)
            for value in list(registry.get("specialists") or [])
            if isinstance(value, dict)
        }
    resolved: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        checkpoint_path = Path(str(row.get("checkpoint") or "")).resolve()
        expected = str(row.get("checksum") or "").strip()
        actual = (
            checkpoint.checkpoint_digest(checkpoint_path)
            if checkpoint_path.is_file()
            else ""
        )
        if (
            row.get("mode") != "frozen_inference_only"
            or not str(row.get("specialist_id") or "")
            or not checkpoint_path.is_file()
            or (expected and actual != expected)
        ):
            raise RuntimeError("frozen teacher identity changed")
        # A newly passing specialist cannot have a checksum in the staged
        # pre-pass contract. Resolve it once from the immutable frozen
        # checkpoint and carry that exact identity through the regression
        # receipt; never accept a missing file or a mismatched pinned digest.
        row["checkpoint"] = str(checkpoint_path)
        row["checksum"] = actual
        row["evaluation_checkpoint"] = str(checkpoint_path)
        row["evaluation_checksum"] = actual

        # Historical frozen checkpoints remain byte-for-byte authoritative,
        # but pre-roster18 adapter banks cannot be loaded by the current
        # runtime without an explicit compatibility derivative. The frozen
        # registry binds such a derivative back to the exact source passing
        # checksum and proves that retained rows are byte-identical. Use that
        # inference-only representation for gameplay while preserving the
        # original teacher identity above.
        specialist_id = str(row["specialist_id"])
        registered = registry_rows.get(specialist_id)
        if registered is not None:
            if (
                registered.get("frozen") is not True
                or registered.get("public_mix_eligible") is not True
                or registered.get("kaggle_submission_eligible") is not False
            ):
                raise RuntimeError(
                    "frozen teacher inference registry policy changed"
                )
            receipt_raw = str(
                registered.get("v5_derivative_receipt") or ""
            ).strip()
            registered_source = str(
                registered.get("source_passing_checkpoint_digest") or ""
            ).strip()
            registered_checkpoint = str(
                registered.get("checkpoint_digest") or ""
            ).strip()
            if receipt_raw:
                if registered_source != actual:
                    raise RuntimeError(
                        "frozen teacher inference derivative identity changed"
                    )
                receipt_path = Path(receipt_raw).expanduser().resolve()
                if not receipt_path.is_file():
                    raise RuntimeError(
                        "frozen teacher inference derivative receipt is missing"
                    )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                evaluation_path = Path(
                    str(receipt.get("derived_checkpoint") or "")
                ).expanduser().resolve()
                evaluation_expected = str(
                    receipt.get("derived_checkpoint_digest") or ""
                )
                evaluation_actual = (
                    checkpoint.checkpoint_digest(evaluation_path)
                    if evaluation_path.is_file()
                    else ""
                )
                if (
                    receipt.get("schema")
                    != "poke_bot.frozen_specialist_roster_v5_derivative/v1"
                    or receipt.get("specialist_id") != specialist_id
                    or receipt.get("source_passing_checkpoint_digest") != actual
                    or Path(
                        str(receipt.get("source_passing_checkpoint") or "")
                    ).expanduser().resolve()
                    != checkpoint_path
                    or receipt.get("inference_only") is not True
                    or receipt.get("kaggle_submission_eligible") is not False
                    or receipt.get("retained_rows_byte_identical") is not True
                    or evaluation_expected != registered_checkpoint
                    or evaluation_actual != evaluation_expected
                ):
                    raise RuntimeError(
                        "frozen teacher inference derivative receipt changed"
                    )
                row["evaluation_checkpoint"] = str(evaluation_path)
                row["evaluation_checksum"] = evaluation_actual
                row["evaluation_derivative_receipt"] = str(receipt_path)
            elif (
                registered_checkpoint != actual
                or (registered_source and registered_source != actual)
            ):
                # Current-format frozen specialists are registered directly.
                # They intentionally have no compatibility-derivative receipt;
                # their registered checkpoint must be the exact immutable
                # teacher checkpoint already bound by the handoff contract.
                raise RuntimeError("frozen teacher direct identity changed")
        resolved.append(row)
    return resolved


def _same_evaluation_identity(
    existing: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    """Compare only fields that can change the regression evaluation.

    The full handoff contract also contains downstream specialist-selection
    configuration.  Adding or reordering those fields changes the file digest
    but cannot change already-completed candidate-vs-teacher games.  Preserve
    the original receipt and accept that digest-only drift only when every
    explicit evaluation identity field remains identical.
    """

    return all(
        existing.get(key) == value
        for key, value in expected.items()
        if key != "contract_digest"
    )


def run(
    *,
    contract_path: Path,
    candidate: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    acceptance = dict(contract.get("acceptance") or {})
    threshold = acceptance.get("gate_threshold")
    aggregate_threshold = acceptance.get("aggregate_gate_threshold")
    games = acceptance.get("games_per_teacher")
    confidence = acceptance.get("confidence")
    source = str(acceptance.get("gate_thresholds_authoritative_source") or "")
    if (
        contract.get("schema")
        != "poke_bot.post_specialist_core_refresh_handoff/v1"
        or threshold is None
        or aggregate_threshold is None
        or games is None
        or confidence is None
        or not source
        or int(games) != 80
        or float(threshold) != 0.35
        or float(aggregate_threshold) != 0.40
        or float(confidence) != 0.90
    ):
        raise RuntimeError("core gameplay regression contract is unresolved or drifted")

    candidate = candidate.expanduser().resolve()
    candidate_identity = CheckpointIdentity.from_path(candidate)
    teachers = _teacher_rows(contract)
    expected_identity = {
        "candidate": candidate_identity.as_dict(),
        "contract": str(contract_path),
        "contract_digest": _sha256(contract_path),
        "teacher_checkpoint_digests": [row["checksum"] for row in teachers],
        "teacher_evaluation_checkpoint_digests": [
            row["evaluation_checksum"] for row in teachers
        ],
        "games_per_teacher": int(games),
        "threshold": float(threshold),
        "aggregate_threshold": float(aggregate_threshold),
        "confidence": float(confidence),
        "gate_thresholds_authoritative_source": source,
    }
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != SCHEMA
            or not _same_evaluation_identity(
                dict(existing.get("identity") or {}),
                expected_identity,
            )
        ):
            raise RuntimeError("existing core regression result identity changed")
        return existing

    results: list[dict[str, Any]] = []
    for index, teacher in enumerate(teachers):
        specialist_id = str(teacher["specialist_id"])
        decks = _our_decks("specialist", specialist_id)
        matching = [row for row in decks if row[0] == specialist_id]
        if len(matching) != 1:
            raise RuntimeError(
                f"teacher deck is not uniquely registered: {specialist_id}"
            )
        report, _rows = _promotion_eval(
            candidate=candidate_identity,
            incumbent=CheckpointIdentity.from_path(
                teacher["evaluation_checkpoint"]
            ),
            decks=matching,
            n_games=int(games),
            n_workers=max(1, int(workers)),
            threshold=float(threshold),
            confidence=float(confidence),
            bootstrap_resamples=4000,
            seed=27_000_000 + index * 100_000,
            game_timeout_s=600,
            model_generation=0,
        )
        results.append(
            {
                "specialist_id": specialist_id,
                "teacher_checkpoint": teacher["checkpoint"],
                "teacher_checkpoint_digest": teacher["checksum"],
                "teacher_evaluation_checkpoint": teacher[
                    "evaluation_checkpoint"
                ],
                "teacher_evaluation_checkpoint_digest": teacher[
                    "evaluation_checksum"
                ],
                "teacher_evaluation_derivative_receipt": teacher.get(
                    "evaluation_derivative_receipt"
                ),
                "report": report,
            }
        )

    valid_reports = [
        row["report"]
        for row in results
        if row["report"].get("valid") is True
    ]
    per_teacher_raw_win_rates = [
        float(report.get("wr") or 0.0) for report in valid_reports
    ]
    aggregate_raw_win_rate = (
        sum(per_teacher_raw_win_rates) / len(per_teacher_raw_win_rates)
        if per_teacher_raw_win_rates
        else 0.0
    )
    all_reports_valid = len(valid_reports) == len(results)
    per_teacher_raw_floor_passed = all(
        value >= float(threshold) for value in per_teacher_raw_win_rates
    )
    aggregate_raw_floor_passed = (
        aggregate_raw_win_rate >= float(aggregate_threshold)
    )
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": expected_identity,
        "results": results,
        "criteria": {
            "aggregate_raw_win_rate": aggregate_raw_win_rate,
            "aggregate_raw_win_rate_minimum": float(aggregate_threshold),
            "aggregate_raw_floor_passed": aggregate_raw_floor_passed,
            "per_teacher_raw_win_rate_minimum": float(threshold),
            "per_teacher_raw_floor_passed": per_teacher_raw_floor_passed,
            "all_reports_valid": all_reports_valid,
            "confidence_intervals_diagnostic_only": True,
        },
        "passed": (
            all_reports_valid
            and per_teacher_raw_floor_passed
            and aggregate_raw_floor_passed
        ),
        "training_eligible": False,
        "replay_eligible": False,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    result = run(
        contract_path=args.contract,
        candidate=args.candidate,
        output=args.output,
        workers=int(args.workers),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
