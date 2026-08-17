#!/usr/bin/env python3
"""Validate the immutable causal Slowking combo-label corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.combo_state_contract import (
    COMBO_STATE_KEY,
    VECTOR_WIDTH,
    validate_combo_state_labels,
)
from poke_bot.authoritative_visual_trace import TARGET_CONSUMER_CONTRACT
from poke_bot.feature_shards import iter_feature_shard
from poke_bot.slowking_combo_targets import SLOWKING_DECK

GROUPS = {
    "copied_attack_legality": slice(0, 6),
    "visible_combo_piece_availability": slice(6, 11),
    "energy_route_readiness": slice(11, 16),
    "bench_continuity": slice(16, 20),
}
assert max(group.stop for group in GROUPS.values()) == VECTOR_WIDTH


def _digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{value}"


def validate(root: Path) -> dict[str, Any]:
    ready_path = root / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") != "ready" or ready.get("specialist_id") != "slowking":
        raise ValueError("Slowking corpus ready receipt is invalid")
    decisions = 0
    games = 0
    top = 0
    seek = 0
    group_cells = Counter()
    contexts = Counter()
    metadata_combo_rows = 0
    current_consumer_contract = True
    for row in ready.get("daily_shards") or []:
        day = str(row["date"])
        shard = root / f"slowking-{day}.features"
        if not shard.is_file():
            raise FileNotFoundError(shard)
        metadata = json.loads(
            shard.with_suffix(shard.suffix + ".json").read_text(
                encoding="utf-8"
            )
        )
        if int(metadata.get("dataset_schema") or 0) != 8:
            raise ValueError(f"Slowking day {day} is not dataset schema 8")
        if metadata.get("target_consumer_contract") != TARGET_CONSUMER_CONTRACT:
            current_consumer_contract = False
        metadata_combo_rows += int(
            ((metadata.get("stats") or {}).get("target_coverage") or {}).get(
                "combo_state_rows"
            )
            or 0
        )
        for sequence in iter_feature_shard(shard):
            if Counter(int(card) for card in sequence.deck) != SLOWKING_DECK:
                raise ValueError("feature shard contains a substituted Slowking deck")
            games += 1
            for decision in sequence.decisions:
                raw = dict(decision.aux_labels or {}).get(COMBO_STATE_KEY)
                if raw is None:
                    raise ValueError("Slowking decision lacks combo-state target")
                clean = validate_combo_state_labels(raw)
                decisions += 1
                top += int(clean["top_deck_mask"])
                seek += int(clean["seek_source_mask"])
                for name, group in GROUPS.items():
                    group_cells[name] += sum(clean["vector_mask"][group])
                for stage in decision.policy_stages:
                    contexts[int(getattr(stage, "select_context", -1))] += 1
    expected_games = int(ready.get("records") or 0)
    expected_decisions = int(ready.get("decisions") or 0)
    checks = {
        "exact_deck_binding": games == expected_games == 311,
        "all_decisions_labeled": decisions == expected_decisions and decisions > 0,
        "top_deck_construction_coverage": top > 0,
        "seek_source_coverage": seek > 0,
        "copied_attack_legality_coverage": group_cells[
            "copied_attack_legality"
        ] > 0,
        "visible_combo_piece_coverage": group_cells[
            "visible_combo_piece_availability"
        ] > 0,
        "energy_route_coverage": group_cells["energy_route_readiness"] > 0,
        "bench_continuity_coverage": group_cells["bench_continuity"] > 0,
        "slowking_copy_context_coverage": contexts[35] > 0,
        "slowking_target_context_coverage": contexts[15] > 0,
        "current_non_imitation_consumer_contract": current_consumer_contract,
        "metadata_combo_coverage_matches_decisions": (
            metadata_combo_rows == decisions
        ),
    }
    return {
        "schema": "poke_bot.slowking_combo_corpus_validation/v1",
        "specialist_id": "slowking",
        "status": "validated" if all(checks.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(root),
        "ready_receipt": str(ready_path),
        "ready_receipt_sha256": _digest(ready_path),
        "dataset_schema": 8,
        "games": games,
        "decisions": decisions,
        "coverage": {
            "top_deck_labels": top,
            "seek_source_labels": seek,
            "vector_cells": dict(group_cells),
            "select_context_rows": {
                str(name): count for name, count in sorted(contexts.items())
            },
            "metadata_combo_rows": metadata_combo_rows,
        },
        "checks": checks,
        "source_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _digest(path),
            }
            for path in (
                ROOT / "poke_bot/combo_state_contract.py",
                ROOT / "poke_bot/slowking_combo_targets.py",
                ROOT / "poke_bot/authoritative_visual_trace.py",
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.corpus_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
