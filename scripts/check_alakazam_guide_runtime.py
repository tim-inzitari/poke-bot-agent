#!/usr/bin/env python3
"""Exercise Alakazam guide labels on real replay observations without caching."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.alakazam_heuristics import GUIDE_VERSION, enabled, is_alakazam_deck
from poke_bot.dataset import convert_record
from poke_bot.pure_rl.model_registry import sha256


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_canary(
    jsonl: Path,
    *,
    max_records: int = 32,
    min_guide_rows: int = 25,
) -> dict[str, Any]:
    """Return a bounded real-runtime coverage report or fail closed."""
    if not enabled():
        raise RuntimeError("POKEBOT_ALAKAZAM_GUIDE_TARGETS must be enabled")
    source = Path(jsonl).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    started = time.monotonic()
    records_seen = 0
    records_converted = 0
    decisions = 0
    policy_stages = 0
    guide_rows = 0
    positive_confidence = 0.0
    target_indices: set[int] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if records_seen >= int(max_records):
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("archetype") or "").casefold() != "alakazam":
                continue
            deck = [int(card) for card in (record.get("deck") or [])]
            if not is_alakazam_deck(deck):
                continue
            records_seen += 1
            sequence, reason, _details = convert_record(
                record,
                verify_info_set=True,
            )
            if sequence is None:
                raise RuntimeError(f"real Alakazam canary record was rejected: {reason}")
            records_converted += 1
            decisions += len(sequence.decisions)
            for decision in sequence.decisions:
                for stage in decision.policy_stages:
                    policy_stages += 1
                    if int(stage.guide_target_index) >= 0:
                        guide_rows += 1
                        positive_confidence += float(stage.guide_confidence)
                        target_indices.add(int(stage.guide_target_index))
    if records_converted <= 0:
        raise RuntimeError("real runtime canary found no exact Alakazam records")
    if guide_rows < int(min_guide_rows):
        raise RuntimeError(
            f"real runtime canary produced {guide_rows} guide rows; "
            f"required {int(min_guide_rows)}"
        )
    elapsed = time.monotonic() - started
    return {
        "schema": "poke_bot.alakazam_guide_runtime_canary/v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "guide_version": GUIDE_VERSION,
        "guide_source_sha256": sha256(ROOT / "poke_bot/alakazam_heuristics.py"),
        "source": str(source),
        "records_seen": records_seen,
        "records_converted": records_converted,
        "decisions": decisions,
        "policy_stages": policy_stages,
        "guide_rows": guide_rows,
        "guide_row_fraction": guide_rows / max(1, policy_stages),
        "mean_guide_confidence": positive_confidence / max(1, guide_rows),
        "distinct_target_indices": len(target_indices),
        "elapsed_s": elapsed,
        "stages_per_second": policy_stages / max(elapsed, 1e-9),
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--min-guide-rows", type=int, default=25)
    args = parser.parse_args()
    report = run_canary(
        args.jsonl,
        max_records=int(args.max_records),
        min_guide_rows=int(args.min_guide_rows),
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
