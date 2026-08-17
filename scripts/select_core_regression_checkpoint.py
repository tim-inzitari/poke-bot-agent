#!/usr/bin/env python3
"""Select the best validation checkpoint that passes all core regressions.

The supervised core refresh keeps one checkpoint per epoch.  Validation loss
orders the candidates, but gameplay regression remains the acceptance gate.
This helper evaluates each preserved checkpoint in validation-loss order,
reuses checksum-bound regression receipts, and stops at the first exact pass.
It does not freeze, replace, or publish any model.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_bot import checkpoint
from scripts.run_core_teacher_regression import run as run_regression


SCHEMA = "poke_bot.core_regression_checkpoint_selection/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def select(
    *,
    contract: Path,
    training_state: Path,
    output_root: Path,
    receipt: Path,
    workers: int,
    result_prefix: str = "core",
) -> dict[str, Any]:
    contract = contract.expanduser().resolve()
    training_state = training_state.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    state = _read(training_state)
    history = [dict(row) for row in state.get("history") or []]
    if (
        state.get("schema") != "poke_bot.multi_teacher_core_refresh_state/v1"
        or not history
    ):
        raise RuntimeError("multi-teacher core training state is incomplete")

    ordered = sorted(
        history,
        key=lambda row: (
            float(row["validation_loss"]),
            int(row["epoch"]),
        ),
    )
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    output_root.mkdir(parents=True, exist_ok=True)
    if (
        not result_prefix
        or result_prefix.startswith(".")
        or "/" in result_prefix
    ):
        raise ValueError("result prefix must be one safe filename prefix")
    for row in ordered:
        epoch = int(row["epoch"])
        candidate = Path(str(row["checkpoint"])).expanduser().resolve()
        expected = str(row["checkpoint_digest"])
        actual = checkpoint.checkpoint_digest(candidate)
        if actual != expected:
            raise RuntimeError(
                f"core epoch {epoch} checkpoint identity changed"
            )
        result_path = output_root / (
            f"{result_prefix}-epoch-{epoch:02d}-teacher-regression.json"
        )
        result = run_regression(
            contract_path=contract,
            candidate=candidate,
            output=result_path,
            workers=int(workers),
        )
        attempt = {
            "epoch": epoch,
            "validation_loss": float(row["validation_loss"]),
            "checkpoint": str(candidate),
            "checkpoint_digest": expected,
            "regression_result": str(result_path),
            "passed": result.get("passed") is True,
        }
        attempts.append(attempt)
        if attempt["passed"]:
            selected = attempt
            break

    value = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract),
        "training_state": str(training_state),
        "ordering": "validation_loss_ascending_then_epoch_ascending",
        "thresholds_changed": False,
        "attempts": attempts,
        "selected": selected,
        "passed": selected is not None,
    }
    if receipt.is_file():
        existing = _read(receipt)
        stable_existing = dict(existing)
        stable_value = dict(value)
        stable_existing.pop("created_at_utc", None)
        stable_value.pop("created_at_utc", None)
        if stable_existing != stable_value:
            raise RuntimeError("core regression selection receipt changed")
        return existing
    _atomic(receipt, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--result-prefix", default="core")
    args = parser.parse_args()
    result = select(
        contract=args.contract,
        training_state=args.training_state,
        output_root=args.output_root,
        receipt=args.receipt,
        workers=max(1, int(args.workers)),
        result_prefix=str(args.result_prefix),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
