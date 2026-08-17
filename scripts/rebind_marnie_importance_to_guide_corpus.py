#!/usr/bin/env python3
"""Bind revision-138 pilot weights to a guide-only derivative corpus.

The derivative must contain the exact same ordered games, causal features,
actions, auxiliary labels, and strategic targets as the pinned source corpus.
Only PolicyStage guide target/confidence fields may differ.  The script streams
both manifests and fails closed before publishing a new importance index.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Iterator

from poke_bot.expert_pilot_importance import (
    OWNER_DECISION_REVISION,
    SCHEMA,
    canonical_digest,
    file_digest,
)
from poke_bot.feature_shards import iter_feature_shard


RECEIPT_SCHEMA = "poke_bot.marnie_guide_corpus_importance_rebind/v1"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable output changed: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    partial.write_text(body, encoding="utf-8")
    os.link(partial, path)
    partial.unlink(missing_ok=True)


def sparse(value: Any) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    return (
        tuple(int(item) for item in value.index),
        tuple(float(item) for item in value.value),
        tuple(int(item) for item in value.offset),
    )


def sequence_contract(sequence: Any) -> bytes:
    decisions = []
    for decision in sequence.decisions:
        stages = [
            (
                sparse(stage.options),
                tuple(tuple(int(item) for item in row) for row in stage.action_combos),
                int(stage.target_index),
                int(stage.select_context),
                bool(stage.selected_is_stop),
            )
            for stage in decision.policy_stages
        ]
        decisions.append(
            (
                sparse(decision.board),
                sparse(decision.options),
                tuple(int(item) for item in decision.action),
                int(decision.action_combo_index),
                tuple(
                    tuple(int(item) for item in row)
                    for row in decision.action_combos
                ),
                int(decision.env_step),
                pickle.dumps(decision.aux_labels, protocol=4),
                sparse(decision.action_token),
                tuple(stages),
                int(decision.matchup_adapter_oracle_route),
                int(decision.matchup_adapter_public_route),
            )
        )
    return pickle.dumps(
        (
            str(sequence.episode_id),
            int(sequence.seat),
            str(sequence.archetype),
            str(sequence.opp_archetype),
            tuple(int(item) for item in sequence.deck),
            float(sequence.value),
            bool(sequence.info_set_ok),
            pickle.dumps(sequence.policy_targets, protocol=4),
            pickle.dumps(sequence.factorized_policy_targets, protocol=4),
            pickle.dumps(sequence.matchup_adapter_training_ticket, protocol=4),
            tuple(decisions),
        ),
        protocol=4,
    )


def manifest_shards(path: Path, payload: dict[str, Any]) -> Iterator[tuple[str, Path, dict[str, Any]]]:
    for row in list(payload.get("shards") or ()):
        source_dates = list(row.get("source_dates") or ())
        if len(source_dates) != 1:
            raise RuntimeError("guide rebind requires one exact date per shard")
        yield str(source_dates[0]), (path.parent / str(row["path"])).resolve(), row


def verify_shard_pair(
    source_day: str,
    source_shard: Path,
    guide_shard: Path,
) -> tuple[str, int, int, int, str]:
    identity = hashlib.sha256()
    games = decisions = guide_labels = 0
    source_iter = iter_feature_shard(source_shard)
    guide_iter = iter_feature_shard(guide_shard)
    while True:
        source_game = next(source_iter, None)
        guide_game = next(guide_iter, None)
        if source_game is None or guide_game is None:
            if source_game is not None or guide_game is not None:
                raise RuntimeError("guide derivative changed game count")
            break
        source_contract = sequence_contract(source_game)
        guide_contract = sequence_contract(guide_game)
        if source_contract != guide_contract:
            raise RuntimeError(
                "guide derivative changed a non-guide causal row: "
                f"day={source_day} episode={source_game.episode_id} "
                f"seat={source_game.seat}"
            )
        identity.update(source_contract)
        games += 1
        decisions += len(source_game.decisions)
        guide_labels += sum(
            int(stage.guide_target_index >= 0 and stage.guide_confidence > 0.0)
            for decision in guide_game.decisions
            for stage in decision.policy_stages
        )
    return source_day, games, decisions, guide_labels, identity.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--guide-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    guide_path = args.guide_manifest.resolve()
    source = read_object(source_path)
    guide = read_object(guide_path)
    source_total = dict(source.get("totals") or {})
    guide_total = dict(guide.get("totals") or {})
    for field in ("records_kept", "decisions_kept"):
        if int(source_total.get(field, -1)) != int(guide_total.get(field, -2)):
            raise RuntimeError(f"guide derivative changed {field}")
    if list(source.get("dates") or ()) != list(guide.get("dates") or ()):
        raise RuntimeError("guide derivative changed date order")
    if source.get("expanded_strategic_targets") != guide.get(
        "expanded_strategic_targets"
    ):
        raise RuntimeError("guide derivative changed strategic target contract")

    source_rows = list(manifest_shards(source_path, source))
    guide_rows = list(manifest_shards(guide_path, guide))
    if len(source_rows) != len(guide_rows):
        raise RuntimeError("guide derivative changed shard count")
    jobs: list[tuple[str, Path, Path]] = []
    for source_row, guide_row in zip(source_rows, guide_rows, strict=True):
        source_day, source_shard, source_meta = source_row
        guide_day, guide_shard, guide_meta = guide_row
        if source_day != guide_day:
            raise RuntimeError("guide derivative changed shard date")
        for field in ("records_kept", "decisions_kept"):
            source_value = int((source_meta.get("stats") or {}).get(field, -1))
            guide_value = int((guide_meta.get("stats") or {}).get(field, -2))
            if source_value != guide_value:
                raise RuntimeError(f"guide derivative changed {source_day} {field}")
        jobs.append((source_day, source_shard, guide_shard))
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.workers == 1:
        results = [verify_shard_pair(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            results = list(pool.map(verify_shard_pair, *zip(*jobs, strict=True)))
    identity = hashlib.sha256()
    games = decisions = guide_labels = 0
    for day, shard_games, shard_decisions, shard_guides, shard_digest in results:
        identity.update(f"{day}:{shard_digest}\n".encode("ascii"))
        games += shard_games
        decisions += shard_decisions
        guide_labels += shard_guides
    if games != 73082 or decisions != 6828373 or guide_labels <= 0:
        raise RuntimeError("guide derivative did not preserve the exact corpus")

    source_index_path = args.source_index.resolve()
    index = read_object(source_index_path)
    if (
        index.get("schema") != SCHEMA
        or index.get("status") != "ready"
        or int(index.get("owner_decision_revision", -1))
        != OWNER_DECISION_REVISION
        or index.get("corpus_manifest_sha256") != file_digest(source_path)
        or canonical_digest(index.get("train_game_weights") or ())
        != index.get("train_game_weights_sha256")
    ):
        raise RuntimeError("source pilot-importance index changed")

    receipt_path = args.receipt.resolve()
    rebound = {
        **index,
        "corpus_manifest_sha256": file_digest(guide_path),
        "guide_derivative_rebind": {
            "schema": RECEIPT_SCHEMA,
            "source_manifest_sha256": file_digest(source_path),
            "guide_manifest_sha256": file_digest(guide_path),
            "non_guide_identity_sha256": "sha256:" + identity.hexdigest(),
            "guide_labeled_rows": guide_labels,
            "receipt": str(receipt_path),
        },
    }
    write_once(args.output_index.resolve(), rebound)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "validated_exact_games_actions_and_labels_guide_only_derivative",
        "source_manifest": str(source_path),
        "source_manifest_sha256": file_digest(source_path),
        "guide_manifest": str(guide_path),
        "guide_manifest_sha256": file_digest(guide_path),
        "source_importance_index": str(source_index_path),
        "source_importance_index_sha256": file_digest(source_index_path),
        "rebound_importance_index": str(args.output_index.resolve()),
        "rebound_importance_index_sha256": file_digest(args.output_index.resolve()),
        "non_guide_identity_sha256": "sha256:" + identity.hexdigest(),
        "games": games,
        "decisions": decisions,
        "guide_labeled_rows": guide_labels,
        "only_guide_stage_target_and_confidence_may_differ": True,
        "actions_and_non_guide_labels_unchanged": True,
        "validation_split_and_weights_unchanged": True,
    }
    write_once(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
