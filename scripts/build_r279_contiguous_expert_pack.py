#!/usr/bin/env python3
"""Build the immutable tensor-only revision-279 Alakazam expert pack on Inzi."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import time
from typing import Any

import torch

from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.feature_shards import iter_feature_shard
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus
from poke_bot.r241_own_deck_successor import load_r260_owner_contract
from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex
from poke_bot.r274_bootstrap_handoff import validate_full_expert_manifest
from poke_bot.r279_contiguous_expert_pack import (
    R279_FRAGMENT_SCHEMA,
    R279_PACK_SCHEMA,
    build_side_tensors,
    load_fragment,
    merge_core_corpora,
    merge_side_tensors,
    sha256_file,
    validate_r279_pack,
    write_pack_atomic,
)
from poke_bot.own_deck_rollout_store import iter_daily_sidecar_rows
from poke_bot.tactical_sequence_materialization import attach_tactical_target_overlay


_DATE = re.compile(r"20\d{2}-\d{2}-\d{2}")
EXPECTED_GAMES = 26_704
EXPECTED_DECISIONS = 2_040_911
EXPECTED_DAYS = 20


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with partial.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _write_fragment(
    path: Path,
    *,
    core: DeviceResidentBootstrapCorpus,
    side: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    payload = {
        "schema": R279_FRAGMENT_SCHEMA,
        "metadata": metadata,
        "core_scalars": core.scalar_state(),
        "core_tensors": core.tensor_state(),
        "side_tensors": side,
    }
    try:
        with partial.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _build_fragment(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        order,
        source_day,
        feature_path_text,
        feature_sha256,
        expected_games,
        expected_decisions,
        sidecar_index_text,
        sidecar_root_text,
        daily_meta_sha256,
        source_manifest_sha256,
        all_daily_meta_sha256s,
        tactical_overlay_text,
        tactical_sha256,
        card_vocab,
        fragment_path_text,
        contract,
    ) = task
    feature_path = Path(feature_path_text)
    fragment_path = Path(fragment_path_text)
    receipt_path = fragment_path.with_suffix(".json")
    if fragment_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("contract") == contract
            and receipt.get("sha256") == sha256_file(fragment_path)
            and int(receipt.get("games", -1)) == int(expected_games)
            and int(receipt.get("decisions", -1)) == int(expected_decisions)
        ):
            loaded = load_fragment(fragment_path)
            validate_cpu_corpus(loaded.core)
            validate_r279_pack(
                loaded.core,
                loaded.side,
                expected_games=int(expected_games),
                expected_decisions=int(expected_decisions),
            )
            return {**receipt, "order": int(order), "cache_hit": True}

    if sha256_file(feature_path) != feature_sha256:
        raise RuntimeError(f"feature digest mismatch: {feature_path}")
    rows = list(iter_feature_shard(feature_path))
    decisions = sum(len(row.decisions) for row in rows)
    if len(rows) != int(expected_games) or decisions != int(expected_decisions):
        raise RuntimeError(
            f"feature count mismatch day={source_day} games={len(rows)}/{expected_games} "
            f"decisions={decisions}/{expected_decisions}"
        )
    index = R260InziSidecarIndex(
        Path(sidecar_index_text),
        source_manifest_sha256=str(source_manifest_sha256),
        daily_meta_sha256s=dict(all_daily_meta_sha256s),
    )
    attachment = index.attach_available_rows(
        rows,
        iter_daily_sidecar_rows(
            Path(sidecar_root_text),
            str(source_day),
            expected_meta_sha256=str(daily_meta_sha256),
        ),
    )
    tactical = attach_tactical_target_overlay(
        rows, Path(tactical_overlay_text), require_all=False
    )
    core = DeviceResidentBootstrapCorpus.from_splits(
        rows,
        [],
        device=torch.device("cpu"),
        exact_card_vocab=int(card_vocab),
        force_expanded_strategic=True,
    )
    side, side_counts = build_side_tensors(rows, card_vocab=int(card_vocab))
    validate_cpu_corpus(core)
    counts = validate_r279_pack(
        core,
        side,
        expected_games=int(expected_games),
        expected_decisions=int(expected_decisions),
    )
    if counts != side_counts:
        raise RuntimeError(f"core/side fragment count mismatch day={source_day}")
    metadata = {
        "contract": contract,
        "source_day": source_day,
        "feature_path": str(feature_path),
        "feature_sha256": feature_sha256,
        "daily_meta_sha256": daily_meta_sha256,
        "tactical_overlay_sha256": tactical_sha256,
        "counts": counts,
        "attachment": attachment,
        "tactical": tactical,
    }
    identity = _write_fragment(
        fragment_path, core=core, side=side, metadata=metadata
    )
    receipt = {
        "schema": "poke_bot.r279_contiguous_expert_fragment_receipt/v1",
        "contract": contract,
        "source_day": source_day,
        "games": counts["games"],
        "decisions": counts["decisions"],
        "samples": counts["samples"],
        "options": counts["options"],
        "attachment": attachment,
        "tactical": tactical,
        **identity,
    }
    _write_json_atomic(receipt_path, receipt)
    return {**receipt, "order": int(order), "cache_hit": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sidecar-binding", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tactical-overlay", required=True, type=Path)
    parser.add_argument("--fragment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--card-vocab", required=True, type=int)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--validation-days", type=int, default=2)
    args = parser.parse_args()

    if not 0 < int(args.card_vocab) < 2**15:
        raise ValueError("card vocabulary is outside int16 range")
    manifest_identity = validate_full_expert_manifest(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shards = list(manifest["shards"])
    if len(shards) != EXPECTED_DAYS:
        raise RuntimeError(f"r279 requires exactly {EXPECTED_DAYS} feature shards")
    totals = dict(manifest["totals"])
    if (
        int(totals["records_kept"]) != EXPECTED_GAMES
        or int(totals["decisions_kept"]) != EXPECTED_DECISIONS
    ):
        raise RuntimeError("immutable feature manifest has the wrong exact totals")

    owner = load_r260_owner_contract()
    binding = json.loads(args.sidecar_binding.read_text(encoding="utf-8"))
    daily: dict[str, str] = {}
    meta_paths: dict[str, Path] = {}
    for day, row in dict(binding["daily_sidecar_meta_receipts"]).items():
        path = Path(str(row["path"]))
        meta = json.loads(path.read_text(encoding="utf-8"))
        daily[str(day)] = str(meta["meta_sha256"])
        meta_paths[str(day)] = path
    if len(daily) != EXPECTED_DAYS:
        raise RuntimeError("r279 sidecar binding does not contain twenty days")
    sidecar_root = next(iter(meta_paths.values())).parents[2]
    tactical_sha256 = sha256_file(args.tactical_overlay)
    manifest_sha256 = sha256_file(args.manifest)
    binding_sha256 = sha256_file(args.sidecar_binding)
    validation_days = int(args.validation_days)
    if not 0 < validation_days < EXPECTED_DAYS:
        raise ValueError("validation days must leave nonempty train and validation partitions")

    tasks = []
    day_rows: list[dict[str, Any]] = []
    for order, row in enumerate(shards):
        feature_path = (args.manifest.parent / str(row["path"])).resolve()
        match = _DATE.search(feature_path.name)
        if match is None:
            raise RuntimeError(f"feature shard has no source day: {feature_path}")
        source_day = match.group(0)
        if source_day not in daily:
            raise RuntimeError(f"feature shard is absent from sidecar binding: {source_day}")
        stats = dict(row["stats"])
        contract = {
            "schema": R279_FRAGMENT_SCHEMA,
            "source_day": source_day,
            "feature_sha256": str(row["sha256"]),
            "daily_meta_sha256": daily[source_day],
            "source_manifest_sha256": owner.source_manifest_sha256,
            "tactical_overlay_sha256": tactical_sha256,
            "card_vocab": int(args.card_vocab),
            "join": "one_daily_linear_three_key_rekey_bulk_join_v2",
        }
        digest = _canonical_digest(contract).split(":", 1)[1]
        fragment_path = args.fragment_root / f"{order:02d}-{source_day}-{digest}.pt"
        tasks.append(
            (
                order,
                source_day,
                str(feature_path),
                str(row["sha256"]),
                int(stats["records_kept"]),
                int(stats["decisions_kept"]),
                str(args.index.resolve()),
                str(sidecar_root.resolve()),
                daily[source_day],
                owner.source_manifest_sha256,
                dict(daily),
                str(args.tactical_overlay.resolve()),
                tactical_sha256,
                int(args.card_vocab),
                str(fragment_path.resolve()),
                contract,
            )
        )
        day_rows.append({"source_day": source_day, "fragment": str(fragment_path.resolve())})

    context = multiprocessing.get_context("spawn")
    receipts: list[dict[str, Any] | None] = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=min(max(1, int(args.workers)), len(tasks)),
        mp_context=context,
    ) as pool:
        for result in pool.map(_build_fragment, tasks):
            receipts[int(result["order"])] = result
            print(
                f"[r279-pack] fragment day={result['source_day']} "
                f"games={result['games']} decisions={result['decisions']} "
                f"cache_hit={result['cache_hit']}",
                flush=True,
            )
    if any(value is None for value in receipts):
        raise RuntimeError("r279 fragment worker result is missing")

    split = len(day_rows) - validation_days
    ordered_rows = day_rows[:split] + day_rows[split:]
    loaded = [load_fragment(Path(row["fragment"])) for row in ordered_rows]
    core = merge_core_corpora(
        [value.core for value in loaded], train_fragment_count=split
    )
    side = merge_side_tensors([value.side for value in loaded])
    validate_cpu_corpus(core)
    counts = validate_r279_pack(
        core,
        side,
        expected_games=EXPECTED_GAMES,
        expected_decisions=EXPECTED_DECISIONS,
    )
    train_days = [row["source_day"] for row in day_rows[:split]]
    val_days = [row["source_day"] for row in day_rows[split:]]
    if set(train_days) & set(val_days):
        raise RuntimeError("r279 train/validation source days overlap")
    contract = {
        "schema": R279_PACK_SCHEMA,
        "owner_revision": 279,
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": owner.source_manifest_sha256,
        "sidecar_binding_sha256": binding_sha256,
        "tactical_overlay_sha256": tactical_sha256,
        "card_vocab": int(args.card_vocab),
        "train_days": train_days,
        "validation_days": val_days,
        "exact_games": EXPECTED_GAMES,
        "exact_decisions": EXPECTED_DECISIONS,
        "resident_python_objects_during_epoch_training": False,
    }
    metadata = {
        "manifest_identity": manifest_identity,
        "counts": counts,
        "fragment_receipts": [
            {key: value[key] for key in ("source_day", "sha256", "games", "decisions", "samples", "options")}
            for value in receipts
            if value is not None
        ],
        "join_totals": {
            key: sum(int(value["attachment"].get(key, 0)) for value in receipts if value is not None)
            for key in (
                "joined_decision_count",
                "masked_unjoinable_decision_count",
                "missing_sidecar_decision_count",
                "incompatible_sidecar_decision_count",
            )
        },
        "tactical_roots": sum(int(value["tactical"].get("roots", 0)) for value in receipts if value is not None),
    }
    if int(metadata["join_totals"]["joined_decision_count"]) <= 0:
        raise RuntimeError("r279 bulk OwnDeck join produced zero attached rows")
    overlay_roots = int(
        json.loads(args.tactical_overlay.read_text(encoding="utf-8"))["roots"]
    )
    if int(metadata["tactical_roots"]) != overlay_roots or overlay_roots < 1024:
        raise RuntimeError(
            "r279 tactical overlay did not attach every committed root: "
            f"attached={metadata['tactical_roots']} expected={overlay_roots}"
        )
    identity = write_pack_atomic(
        args.output,
        core=core,
        side=side,
        contract=contract,
        metadata=metadata,
    )
    receipt = {
        "schema": "poke_bot.r279_contiguous_expert_pack_receipt/v1",
        "owner_revision": 279,
        "contract": contract,
        "metadata": metadata,
        "pack": identity,
        "validated": True,
    }
    _write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
