#!/usr/bin/env python3
"""Audit sparse current-deck-guide coverage, abstention, and action agreement.

Feature shards retain both the recorded public action target and the collapsed
guide target for every factorized legal stage.  This audit validates index and
confidence integrity, reconciles labeled-stage counts with shard metadata, and
reports exact selected-action agreement as an observational precision proxy.

Agreement is diagnostic only: public actions are not assumed to be a perfect
strategy oracle, and this receipt never authorizes serving, gating, or a
specialist transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.dataset import GameSequence
from poke_bot.feature_shards import iter_feature_shard


SCHEMA = "poke_bot.current_deck_guide_label_audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _empty_counts() -> Counter[str]:
    return Counter(
        {
            "records": 0,
            "decisions": 0,
            "policy_stages": 0,
            "labeled_stages": 0,
            "abstained_stages": 0,
            "expert_action_agreements": 0,
            "expert_action_disagreements": 0,
            "compacted_candidate_stages": 0,
            "invalid_target_indices": 0,
            "invalid_confidences": 0,
        }
    )


COUNT_FIELDS = tuple(_empty_counts())


def _metadata_guide_rows(stats: dict[str, Any]) -> int:
    """Return the materialized guide-row count, including legacy empty shards."""
    coverage = dict(stats.get("target_coverage") or {})
    if "guide_rows" in coverage:
        return int(coverage["guide_rows"])
    if (
        int(stats.get("records_kept", -1)) == 0
        and int(stats.get("decisions_kept", -1)) == 0
    ):
        return 0
    return -1


def _merge_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    counts = _empty_counts()
    confidence_sum = 0.0
    agreement_confidence_sum = 0.0
    buckets: Counter[str] = Counter()
    for row in values:
        for key in COUNT_FIELDS:
            counts[key] += int(row.get(key) or 0)
        labeled = int(row.get("labeled_stages") or 0)
        mean = row.get("mean_label_confidence")
        if labeled and mean is not None:
            row_confidence = float(mean) * labeled
            confidence_sum += row_confidence
            weighted = row.get(
                "confidence_weighted_expert_action_agreement_rate"
            )
            if weighted is not None:
                agreement_confidence_sum += float(weighted) * row_confidence
        buckets.update(
            {
                str(key): int(value)
                for key, value in dict(
                    row.get("confidence_buckets") or {}
                ).items()
            }
        )
    stages = counts["policy_stages"]
    labeled = counts["labeled_stages"]
    agreements = counts["expert_action_agreements"]
    return {
        **dict(counts),
        "label_coverage_rate": labeled / stages if stages else None,
        "abstention_rate": counts["abstained_stages"] / stages if stages else None,
        "expert_action_agreement_rate": agreements / labeled if labeled else None,
        "confidence_weighted_expert_action_agreement_rate": (
            agreement_confidence_sum / confidence_sum
            if confidence_sum > 0.0
            else None
        ),
        "mean_label_confidence": (
            confidence_sum / labeled if labeled else None
        ),
        "confidence_buckets": dict(sorted(buckets.items())),
    }


def audit_sequences(sequences: Iterable[GameSequence]) -> dict[str, Any]:
    counts = _empty_counts()
    confidence_sum = 0.0
    agreement_confidence_sum = 0.0
    confidence_buckets: Counter[str] = Counter()
    for sequence in sequences:
        counts["records"] += 1
        for decision in sequence.decisions:
            counts["decisions"] += 1
            for stage in decision.policy_stages:
                counts["policy_stages"] += 1
                options = len(stage.action_combos)
                target = int(stage.target_index)
                guide_target = int(stage.guide_target_index)
                confidence = float(stage.guide_confidence)
                if not options:
                    # Current temporal-expert shards may compact the candidate
                    # list after target indices are validated at materialization.
                    # The retained expert/guide indices still support exact
                    # agreement accounting, but cannot be upper-bound checked
                    # a second time from this shard alone.
                    counts["compacted_candidate_stages"] += 1
                if target < 0 or (options and target >= options):
                    counts["invalid_target_indices"] += 1
                if guide_target < 0:
                    counts["abstained_stages"] += 1
                    if confidence != 0.0:
                        counts["invalid_confidences"] += 1
                    continue
                counts["labeled_stages"] += 1
                if options and guide_target >= options:
                    counts["invalid_target_indices"] += 1
                if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
                    counts["invalid_confidences"] += 1
                else:
                    confidence_sum += confidence
                    confidence_buckets[
                        "0.00-0.25"
                        if confidence <= 0.25
                        else "0.25-0.50"
                        if confidence <= 0.50
                        else "0.50-0.75"
                        if confidence <= 0.75
                        else "0.75-1.00"
                    ] += 1
                if target == guide_target:
                    counts["expert_action_agreements"] += 1
                    if math.isfinite(confidence):
                        agreement_confidence_sum += confidence
                else:
                    counts["expert_action_disagreements"] += 1
    stages = counts["policy_stages"]
    labeled = counts["labeled_stages"]
    agreements = counts["expert_action_agreements"]
    return {
        **dict(counts),
        "label_coverage_rate": labeled / stages if stages else None,
        "abstention_rate": counts["abstained_stages"] / stages if stages else None,
        "expert_action_agreement_rate": agreements / labeled if labeled else None,
        "confidence_weighted_expert_action_agreement_rate": (
            agreement_confidence_sum / confidence_sum
            if confidence_sum > 0.0
            else None
        ),
        "mean_label_confidence": (
            confidence_sum / labeled if labeled else None
        ),
        "confidence_buckets": dict(sorted(confidence_buckets.items())),
    }


def build_audit(
    *,
    shard_paths: Iterable[Path],
    specialist_id: str,
    guide_version: str,
    corpus_ready_receipt: Path | None = None,
) -> dict[str, Any]:
    ordered = sorted(Path(path).resolve() for path in shard_paths)
    if not ordered:
        raise ValueError("at least one guide feature shard is required")
    daily_metrics: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    metadata_guide_rows = 0
    source_dates: set[str] = set()
    for path in ordered:
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(path if not path.is_file() else metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("required_archetype") != specialist_id
            or metadata.get("guide_version") != guide_version
            or metadata.get("guide_id") != specialist_id
        ):
            raise RuntimeError(f"guide shard identity mismatch: {path}")
        sequences = list(iter_feature_shard(path))
        row = audit_sequences(sequences)
        stats = dict(metadata.get("stats") or {})
        expected_guide_rows = _metadata_guide_rows(stats)
        if (
            int(stats.get("records_kept", -1)) != row["records"]
            or int(stats.get("decisions_kept", -1)) != row["decisions"]
            or expected_guide_rows != row["labeled_stages"]
        ):
            raise RuntimeError(f"guide shard accounting mismatch: {path}")
        dates = [str(value) for value in metadata.get("source_dates") or ()]
        if len(dates) != 1:
            raise RuntimeError(f"guide shard must bind exactly one day: {path}")
        source_dates.update(dates)
        metadata_guide_rows += expected_guide_rows
        daily_metrics.append(row)
        shard_rows.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "source_date": dates[0],
                "metrics": row,
            }
        )
    aggregate = _merge_metrics(daily_metrics)
    if (
        aggregate["records"] <= 0
        or aggregate["decisions"] <= 0
        or aggregate["policy_stages"] <= 0
        or aggregate["labeled_stages"] <= 0
        or aggregate["abstained_stages"] <= 0
        or aggregate["labeled_stages"] != metadata_guide_rows
        or aggregate["invalid_target_indices"] != 0
        or aggregate["invalid_confidences"] != 0
    ):
        raise RuntimeError("guide label audit failed structural validation")
    ready_row: dict[str, Any] | None = None
    if corpus_ready_receipt is not None:
        ready = corpus_ready_receipt.resolve()
        if not ready.is_file():
            raise FileNotFoundError(ready)
        ready_row = {"path": str(ready), "sha256": _sha256(ready)}
    return {
        "schema": SCHEMA,
        "status": "passed_structural_and_observational_validation",
        "specialist_id": specialist_id,
        "guide_version": guide_version,
        "source_dates": sorted(source_dates),
        "shard_count": len(shard_rows),
        "shards_sha256": _canonical_digest(
            [(row["path"], row["sha256"]) for row in shard_rows]
        ),
        "corpus_ready_receipt": ready_row,
        "metrics": aggregate,
        "daily": shard_rows,
        "interpretation": {
            "precision_proxy": "exact_recorded_action_agreement_given_a_sparse_label",
            "candidate_index_bounds": (
                "revalidated_when_candidate_lists_are_retained; otherwise_"
                "trusted_only_after_shard_materialization_and_footer_validation"
            ),
            "recorded_action_is_perfect_strategy_oracle": False,
            "agreement_is_activation_threshold": False,
            "abstention_means_mask_not_zero": True,
            "training_eligible": False,
            "formal_gate_eligible": False,
            "serving_authority": "none",
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--guide-version", required=True)
    parser.add_argument("--corpus-ready-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_audit(
        shard_paths=args.shard,
        specialist_id=str(args.specialist_id).strip().casefold(),
        guide_version=str(args.guide_version).strip(),
        corpus_ready_receipt=args.corpus_ready_receipt,
    )
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"immutable guide label audit exists: {output}")
    _atomic_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
