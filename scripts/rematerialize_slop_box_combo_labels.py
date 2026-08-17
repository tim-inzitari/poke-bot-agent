#!/usr/bin/env python3
"""Attach Slop Box combo_state labels onto an expert trajectory shard.

Reads a trajectory JSONL whose rows use either ``steps`` or ``decisions`` with
masked observations, attaches ``poke_bot.slop_box_combo_state_targets/v1`` in
place on a labeled output shard, and writes a checksum-bound coverage receipt.

Does not restart trainers. Feature shards that already dropped observations must
be rebuilt through convert_record (which now attaches Slop Box labels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.combo_state_contract import (  # noqa: E402
    COMBO_STATE_KEY,
    SLOP_BOX_COMBO_STATE_TARGET_SCHEMA,
    validate_combo_state_labels,
)
from poke_bot.slop_box_combo_targets import (  # noqa: E402
    attach_slop_box_combo_state_labels,
    is_slop_box_combo_deck,
)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _acting_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    steps = list(record.get("steps") or [])
    if steps:
        return steps
    return list(record.get("decisions") or [])


def rematerialize(
    *,
    input_path: Path,
    output_path: Path,
    receipt_path: Path,
    max_games: int = 0,
) -> dict[str, Any]:
    skipped_non_slop_box = 0
    games_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("trajectory row must be an object")
            deck = [int(card) for card in (record.get("deck") or [])]
            rows = _acting_rows(record)
            if not is_slop_box_combo_deck(deck):
                skipped_non_slop_box += 1
                dst.write(json.dumps(record, separators=(",", ":")) + "\n")
                continue
            coverage = attach_slop_box_combo_state_labels(rows, deck=deck)
            if record.get("steps"):
                record["steps"] = rows
            else:
                record["decisions"] = rows
            provenance = dict(record.get("target_provenance") or {})
            provenance["slop_box_combo_state_targets"] = coverage
            provenance["slop_box_combo_state_schema"] = (
                SLOP_BOX_COMBO_STATE_TARGET_SCHEMA
            )
            record["target_provenance"] = provenance
            if rows:
                validate_combo_state_labels(
                    dict(rows[0].get("aux_labels") or {}).get(COMBO_STATE_KEY)
                )
            dst.write(json.dumps(record, separators=(",", ":")) + "\n")
            games_written += 1
            if max_games and games_written >= max_games:
                break

    coverage_totals = {
        "games": 0,
        "decisions": 0,
        "top_deck": 0,
        "seek_source": 0,
        "vector_cells": 0,
        "seek_teal_dance": 0,
        "seek_crispin": 0,
        "seek_glass_trumpet": 0,
        "seek_energy_switch": 0,
        "seek_area_zero": 0,
        "seek_recovery": 0,
        "seek_other_engine": 0,
        "combo_state_rows": 0,
        "rows_with_any_seek_or_top": 0,
        "rows_with_vector_cells": 0,
    }
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            deck = [int(card) for card in (record.get("deck") or [])]
            if not is_slop_box_combo_deck(deck):
                continue
            coverage_totals["games"] += 1
            for row in _acting_rows(record):
                raw = dict(row.get("aux_labels") or {}).get(COMBO_STATE_KEY)
                if raw is None:
                    raise ValueError("labeled shard missing combo_state")
                clean = validate_combo_state_labels(raw)
                coverage_totals["decisions"] += 1
                coverage_totals["combo_state_rows"] += 1
                coverage_totals["top_deck"] += int(clean["top_deck_mask"])
                coverage_totals["seek_source"] += int(clean["seek_source_mask"])
                vector_true = sum(clean["vector_mask"])
                coverage_totals["vector_cells"] += vector_true
                if clean["top_deck_mask"] or clean["seek_source_mask"]:
                    coverage_totals["rows_with_any_seek_or_top"] += 1
                if vector_true:
                    coverage_totals["rows_with_vector_cells"] += 1
                if clean["seek_source_mask"]:
                    seek = int(clean["seek_source_target"])
                    key = {
                        0: "seek_teal_dance",
                        1: "seek_crispin",
                        2: "seek_glass_trumpet",
                        3: "seek_energy_switch",
                        4: "seek_area_zero",
                        5: "seek_recovery",
                        6: "seek_other_engine",
                    }.get(seek)
                    if key:
                        coverage_totals[key] += 1

    nonzero_ready = (
        coverage_totals["decisions"] > 0
        and coverage_totals["vector_cells"] > 0
        and (
            coverage_totals["seek_source"] > 0
            or coverage_totals["top_deck"] > 0
        )
    )
    receipt = {
        "schema": "poke_bot.slop_box_combo_state_rematerialization/v1",
        "goal_revision": 173,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specialist_id": "teal-mask-ogerpon-ex",
        "display_name": "Slop Box",
        "target_schema": SLOP_BOX_COMBO_STATE_TARGET_SCHEMA,
        "head_width": 32,
        "width_remap": False,
        "slot_mapping": {
            "top_deck[5]": "Crispin/Glass Trumpet selected Basic Energy class",
            "seek_source[7]": (
                "Teal Dance / Crispin / Glass Trumpet / Energy Switch / "
                "Area Zero / Night Stretcher / other engine"
            ),
            "vector[0:6]": "engine legality (Teal Dance…any engine)",
            "vector[6:11]": "visible combo pieces",
            "vector[11:16]": "energy-route readiness",
            "vector[16:20]": "engine continuity (Teal/Bolt active/bench)",
        },
        "input_path": str(input_path),
        "input_sha256": _digest_file(input_path),
        "output_path": str(output_path),
        "output_sha256": _digest_file(output_path),
        "skipped_non_slop_box": int(skipped_non_slop_box),
        "coverage": coverage_totals,
        "combo_loss_can_be_nonzero": bool(nonzero_ready),
        "status": "ready" if nonzero_ready else "failed",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["receipt_sha256"] = _digest_file(receipt_path)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=0)
    args = parser.parse_args(argv)
    receipt = rematerialize(
        input_path=args.input.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        receipt_path=args.receipt.expanduser().resolve(),
        max_games=int(args.max_games),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
