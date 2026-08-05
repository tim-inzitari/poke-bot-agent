#!/usr/bin/env python3
"""Create an exact-corpus derivative by copying only guide labels.

The authoritative source shard owns every game, feature, action, auxiliary
label, and strategic target.  A separately materialized shard is used only as
the source of ``PolicyStage.guide_target_index`` and ``guide_confidence``.
Candidate and source public option tensors must align exactly at every stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Iterator

from poke_bot.dataset import GameSequence
from poke_bot.feature_shards import SHARD_FORMAT, SHARD_FORMAT_VERSION


GUIDE_ID = "marnie-s-grimmsnarl-ex"
GUIDE_VERSION = "marnie-grimmsnarl-north-star-v1"
DERIVATIVE_SCHEMA = "poke_bot.exact_corpus_guide_label_derivative/v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def sparse(value: Any) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    return (
        tuple(int(item) for item in value.index),
        tuple(float(item) for item in value.value),
        tuple(int(item) for item in value.offset),
    )


def stream(path: Path) -> Iterator[tuple[str, Any]]:
    with path.open("rb") as handle:
        header = pickle.load(handle)
        if not isinstance(header, dict) or header.get("format") != SHARD_FORMAT:
            raise RuntimeError(f"invalid feature shard: {path}")
        yield "header", header
        while True:
            try:
                item = pickle.load(handle)
            except EOFError as exc:
                raise RuntimeError(f"missing feature footer: {path}") from exc
            if isinstance(item, dict) and item.get("format") == SHARD_FORMAT + "-footer":
                if handle.read(1):
                    raise RuntimeError(f"trailing feature bytes: {path}")
                yield "footer", item
                return
            if not isinstance(item, GameSequence):
                raise RuntimeError(f"unexpected shard row: {type(item)!r}")
            yield "sequence", item


def candidate_by_date(candidate_dir: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for sidecar in candidate_dir.glob("*.features.json"):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        dates = list(payload.get("source_dates") or ())
        if len(dates) != 1:
            continue
        shard = sidecar.with_suffix("")
        if shard.is_file():
            values[str(dates[0])] = shard
    return values


def merge_shard(source: Path, candidate: Path, output: Path) -> tuple[int, int, int]:
    source_rows = stream(source)
    candidate_rows = stream(candidate)
    source_kind, source_header = next(source_rows)
    candidate_kind, _candidate_header = next(candidate_rows)
    if source_kind != candidate_kind or source_kind != "header":
        raise RuntimeError("feature header alignment failed")
    header = copy.deepcopy(source_header)
    header["guide_id"] = GUIDE_ID
    header["guide_version"] = GUIDE_VERSION
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    games = decisions = guide_rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("xb") as handle:
            pickle.dump(header, handle, protocol=pickle.HIGHEST_PROTOCOL)
            while True:
                source_kind, source_item = next(source_rows)
                candidate_kind, candidate_item = next(candidate_rows)
                if source_kind != candidate_kind:
                    raise RuntimeError("candidate/source stream lengths differ")
                if source_kind == "footer":
                    footer = copy.deepcopy(source_item)
                    stats = footer.setdefault("stats", {})
                    stats.setdefault("target_coverage", {})["guide_rows"] = guide_rows
                    pickle.dump(footer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    break
                source_key = (str(source_item.episode_id), int(source_item.seat))
                candidate_key = (str(candidate_item.episode_id), int(candidate_item.seat))
                if source_key != candidate_key:
                    raise RuntimeError(
                        f"candidate/source game order differs: {source_key} != {candidate_key}"
                    )
                if len(source_item.decisions) != len(candidate_item.decisions):
                    raise RuntimeError(f"decision count differs: {source_key}")
                for source_decision, candidate_decision in zip(
                    source_item.decisions, candidate_item.decisions, strict=True
                ):
                    if len(source_decision.policy_stages) != len(candidate_decision.policy_stages):
                        raise RuntimeError(f"policy-stage count differs: {source_key}")
                    for source_stage, candidate_stage in zip(
                        source_decision.policy_stages,
                        candidate_decision.policy_stages,
                        strict=True,
                    ):
                        if sparse(source_stage.options) != sparse(candidate_stage.options):
                            raise RuntimeError(f"legal option tensor differs: {source_key}")
                        target = int(candidate_stage.guide_target_index)
                        confidence = float(candidate_stage.guide_confidence)
                        if target < -1 or not 0.0 <= confidence <= 1.0:
                            raise RuntimeError(f"invalid guide label: {source_key}")
                        source_stage.guide_target_index = target
                        source_stage.guide_confidence = confidence
                        guide_rows += int(target >= 0 and confidence > 0.0)
                    decisions += 1
                pickle.dump(source_item, handle, protocol=pickle.HIGHEST_PROTOCOL)
                games += 1
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return games, decisions, guide_rows


def write_once(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-candidate-days",
        action="store_true",
        help=(
            "Retain the source shard unchanged for days without a candidate. "
            "Guide loss is masked on those rows; all other objectives remain active."
        ),
    )
    args = parser.parse_args()
    source_path = args.source_manifest.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidates = candidate_by_date(args.candidate_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(source)
    manifest["guide_derivative"] = {
        "schema": DERIVATIVE_SCHEMA,
        "source_manifest_sha256": digest(source_path),
        "guide_id": GUIDE_ID,
        "guide_version": GUIDE_VERSION,
        "only_policy_stage_guide_fields_changed": True,
    }
    new_rows = []
    total_bytes = total_games = total_decisions = total_guides = 0
    for row in source.get("shards") or ():
        dates = list(row.get("source_dates") or ())
        if len(dates) != 1:
            raise RuntimeError(f"source shard must contain exactly one date: {dates}")
        day = str(dates[0])
        source_shard = (source_path.parent / str(row["path"])).resolve()
        output_shard = output_dir / f"marnie-exact-guide-{day}.features"
        candidate = candidates.get(day)
        if candidate is None:
            if not args.allow_missing_candidate_days:
                raise RuntimeError(f"missing candidate guide shard for {dates}")
            if not output_shard.exists():
                os.link(source_shard, output_shard)
            games = int((row.get("stats") or {}).get("records_kept", -1))
            decisions = int((row.get("stats") or {}).get("decisions_kept", -1))
            guide_rows = int(
                ((row.get("stats") or {}).get("target_coverage") or {}).get(
                    "guide_rows", 0
                )
            )
        else:
            games, decisions, guide_rows = merge_shard(
                source_shard, candidate, output_shard
            )
        new_row = copy.deepcopy(row)
        new_row["path"] = output_shard.name
        new_row["bytes"] = output_shard.stat().st_size
        new_row["sha256"] = digest(output_shard)
        new_row.setdefault("stats", {}).setdefault("target_coverage", {})[
            "guide_rows"
        ] = guide_rows
        new_rows.append(new_row)
        total_bytes += output_shard.stat().st_size
        total_games += games
        total_decisions += decisions
        total_guides += guide_rows
    expected = dict(source.get("totals") or {})
    if total_games != int(expected.get("records_kept", -1)) or total_decisions != int(
        expected.get("decisions_kept", -1)
    ):
        raise RuntimeError("exact source corpus totals changed")
    if total_guides <= 0:
        raise RuntimeError("guide derivative contains no usable guide rows")
    manifest["shards"] = new_rows
    manifest.setdefault("totals", {})["bytes"] = total_bytes
    manifest["totals"].setdefault("target_coverage", {})["guide_rows"] = total_guides
    manifest_path = output_dir / "manifest.json"
    write_once(manifest_path, manifest)
    pointer = {
        "schema": "poke_bot.pinned_expert_corpus/v1",
        "protected": True,
        "manifest": manifest_path.name,
        "manifest_sha256": digest(manifest_path),
        "selection": copy.deepcopy(source.get("selection")),
        "totals": {
            "bytes": total_bytes,
            "records_kept": total_games,
            "decisions_kept": total_decisions,
            "guide_rows": total_guides,
        },
        "guide_derivative_schema": DERIVATIVE_SCHEMA,
    }
    write_once(output_dir / "PROTECTED_EXPERT_CORPUS.json", pointer)
    print(json.dumps(pointer, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
