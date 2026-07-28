#!/usr/bin/env python3
"""Repair a completed compact shard whose routing metadata used a superset tree.

This is deliberately narrow.  It permits no gameplay, target, outcome, or
route-decision rewrite.  A row is repairable only when:

* the canonical and source artifacts have the same executable decision tree;
* canonical accepted routes are a subset with identical route numbers and
  confidence thresholds;
* no row ever selected or transitioned through a removed route; and
* every row used the expected behavior checkpoint.

The output and an audit receipt are written exclusively.  The normal trainer
recovery path must still scan the repaired shard, build/verify its replay
cache, enforce the canonical runtime contract, and commit a completed-
collection receipt before training can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return f"sha256:{value.hexdigest()}"


def _tree(path: Path, expected_digest: str) -> tuple[dict[str, Any], str]:
    actual = _digest(path)
    if actual != expected_digest:
        raise RuntimeError(
            f"tree digest mismatch for {path}: expected={expected_digest} actual={actual}"
        )
    return json.loads(path.read_text(encoding="utf-8")), actual


def _routing_equivalence(
    canonical: dict[str, Any],
    source: dict[str, Any],
) -> tuple[list[str], list[str]]:
    # These fields contain the executable classifier and its feature/target
    # interpretation.  Metadata-only repair is forbidden if any differs.
    for key in (
        "schema",
        "input_contract",
        "prediction_contract",
        "runtime_calibration",
        "targets",
        "tree",
        "unknown_class",
    ):
        if canonical.get(key) != source.get(key):
            raise RuntimeError(f"routing artifacts differ in executable field {key!r}")
    canonical_runtime = dict(canonical.get("runtime_contract") or {})
    source_runtime = dict(source.get("runtime_contract") or {})
    for key in (
        "schema",
        "consecutive_required",
        "min_leaf_confidence",
        "min_validation_precision",
        "one_route_per_decision",
        "oracle_or_package_identity_forbidden",
        "source_tree_digest",
        "unknown_route_exact_bypass",
    ):
        if canonical_runtime.get(key) != source_runtime.get(key):
            raise RuntimeError(f"routing runtime differs in behavioral field {key!r}")
    canonical_ids = [
        str(value) for value in canonical_runtime.get("accepted_archetype_ids") or ()
    ]
    source_ids = [
        str(value) for value in source_runtime.get("accepted_archetype_ids") or ()
    ]
    if not canonical_ids or not set(canonical_ids).issubset(source_ids):
        raise RuntimeError("canonical accepted routes are not a nonempty source subset")
    canonical_confidence = dict(
        canonical_runtime.get("per_archetype_min_leaf_confidence") or {}
    )
    source_confidence = dict(
        source_runtime.get("per_archetype_min_leaf_confidence") or {}
    )
    for archetype_id in canonical_ids:
        if canonical_confidence.get(archetype_id) != source_confidence.get(
            archetype_id
        ):
            raise RuntimeError(
                "shared route confidence differs for "
                f"{archetype_id!r}"
            )
    removed = sorted(set(source_ids) - set(canonical_ids))
    return sorted(canonical_ids), removed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--canonical-tree", type=Path, required=True)
    parser.add_argument("--canonical-tree-digest", required=True)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    receipt_path = args.receipt.resolve()
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("repair output and receipt must not already exist")
    canonical, canonical_digest = _tree(
        args.canonical_tree.resolve(), args.canonical_tree_digest
    )
    source, source_digest = _tree(
        args.source_tree.resolve(), args.source_tree_digest
    )
    canonical_ids, removed_ids = _routing_equivalence(canonical, source)

    canonical_routes: dict[str, int] | None = None
    games = 0
    canonical_rows = 0
    repair_rows = 0
    removed_route_values: set[int] = set()
    seen_episodes: set[str] = set()
    with source_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            games += 1
            episode = str(row.get("episode_id") or "")
            if not episode or episode in seen_episodes:
                raise RuntimeError(
                    f"missing/duplicate episode at source line {line_number}"
                )
            seen_episodes.add(episode)
            provenance = dict(row.get("target_provenance") or {})
            if (
                str(provenance.get("behavior_checkpoint_digest") or "")
                != args.checkpoint_digest
            ):
                raise RuntimeError(
                    f"behavior checkpoint mismatch at source line {line_number}"
                )
            audit = dict(provenance.get("matchup_runtime_audit") or {})
            digest = str(audit.get("tree_digest") or "")
            routes = {
                str(key): int(value)
                for key, value in dict(audit.get("accepted_routes") or {}).items()
            }
            if digest == canonical_digest:
                canonical_rows += 1
                if canonical_routes is None:
                    canonical_routes = routes
                elif routes != canonical_routes:
                    raise RuntimeError("canonical rows disagree on route mapping")
                continue
            if digest != source_digest:
                raise RuntimeError(
                    f"unexpected tree digest at source line {line_number}: {digest}"
                )
            repair_rows += 1
            if set(routes) != set(canonical_ids) | set(removed_ids):
                raise RuntimeError("source row accepted-route roster is unexpected")
            if canonical_routes is not None and any(
                routes.get(key) != canonical_routes.get(key) for key in canonical_ids
            ):
                raise RuntimeError("source row changed a shared route number")
            removed_route_values.update(
                int(routes[key]) for key in removed_ids if key in routes
            )
            active_id = audit.get("active_archetype_id")
            route_values = {
                int(audit.get("model_route", -1)),
                int(audit.get("initial_model_route", -1)),
            }
            for transition in audit.get("route_transitions") or ():
                route_values.add(int(transition.get("from_route", -1)))
                route_values.add(int(transition.get("to_route", -1)))
            for key in dict(audit.get("per_route") or {}):
                try:
                    route_values.add(int(key))
                except (TypeError, ValueError):
                    pass
            if active_id in removed_ids or (route_values - {-1}) & removed_route_values:
                raise RuntimeError(
                    f"removed route was active at source line {line_number}"
                )
    if games != int(args.expected_games) or canonical_routes is None:
        raise RuntimeError(
            f"source game/canonical proof mismatch: games={games} "
            f"expected={args.expected_games} canonical_rows={canonical_rows}"
        )
    # Recheck shared route numbers now that the canonical map is known.
    source_runtime = dict(source.get("runtime_contract") or {})
    source_ids = set(
        str(value) for value in source_runtime.get("accepted_archetype_ids") or ()
    )
    if set(canonical_routes) != set(canonical_ids) or not set(canonical_ids).issubset(
        source_ids
    ):
        raise RuntimeError("canonical route map disagrees with canonical tree roster")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.repairing.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary repair output already exists: {temporary}")
    patched = 0
    output_hash = hashlib.sha256()
    try:
        with source_path.open("r", encoding="utf-8") as source_stream, temporary.open(
            "xb"
        ) as output_stream:
            for line in source_stream:
                row = json.loads(line)
                provenance = row["target_provenance"]
                audit = provenance["matchup_runtime_audit"]
                if str(audit.get("tree_digest") or "") == source_digest:
                    audit["tree_digest"] = canonical_digest
                    audit["accepted_archetype_ids"] = list(canonical_ids)
                    audit["accepted_routes"] = dict(canonical_routes)
                    patched += 1
                encoded = (
                    json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
                ).encode("utf-8")
                output_stream.write(encoded)
                output_hash.update(encoded)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if patched != repair_rows:
            raise RuntimeError(
                f"repair row count changed: planned={repair_rows} patched={patched}"
            )
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    receipt = {
        "schema": "poke_bot.matchup_runtime_collection_metadata_repair/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source_path),
        "input_sha256": _digest(source_path),
        "output": str(output_path),
        "output_sha256": f"sha256:{output_hash.hexdigest()}",
        "games": games,
        "canonical_rows_unchanged": canonical_rows,
        "rows_metadata_repaired": patched,
        "behavior_checkpoint_digest": args.checkpoint_digest,
        "canonical_tree": str(args.canonical_tree.resolve()),
        "canonical_tree_digest": canonical_digest,
        "source_tree": str(args.source_tree.resolve()),
        "source_tree_digest": source_digest,
        "canonical_accepted_archetype_ids": canonical_ids,
        "removed_source_only_archetype_ids": removed_ids,
        "removed_routes_observed": False,
        "gameplay_or_target_fields_modified": False,
        "repair_fields": [
            "target_provenance.matchup_runtime_audit.tree_digest",
            "target_provenance.matchup_runtime_audit.accepted_archetype_ids",
            "target_provenance.matchup_runtime_audit.accepted_routes",
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
