#!/usr/bin/env python3
"""Run the exact r274 bootstrap from one RAM-resident joined expert corpus."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import re
from typing import Any

import torch

from poke_bot import checkpoint
from poke_bot.dataset import BootstrapDataset
from poke_bot.feature_shards import iter_feature_shard
from poke_bot.r241_own_deck_successor import load_r260_owner_contract
from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex
from poke_bot.r274_bootstrap_handoff import validate_full_expert_manifest
from poke_bot.tactical_sequence_materialization import attach_tactical_target_overlay
from poke_bot.train import TrainConfig, supervised_r260_host_rehearsal_step
from poke_bot.own_deck_rollout_store import iter_daily_sidecar_rows


JOINED_SHARD_CACHE_SCHEMA = "poke_bot.r274_joined_shard_cache/v1"
_DATE = re.compile(r"2026-(?:07|08)-\d{2}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_shard(
    task: tuple[Any, ...],
) -> tuple[int, str, str, int, int, dict[str, Any], dict[str, Any], bool]:
    (
        index,
        raw_path,
        expected_digest,
        expected_records,
        sidecar_index_path,
        sidecar_root,
        source_day,
        daily_meta_sha256,
        source_manifest_sha256,
        daily_meta_sha256s,
        tactical_overlay_path,
        cache_path_text,
        cache_contract,
    ) = task
    path = Path(raw_path)
    cache_path = Path(cache_path_text)
    receipt_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if cache_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") == JOINED_SHARD_CACHE_SCHEMA
            and receipt.get("contract") == cache_contract
            and int(receipt.get("records", -1)) == int(expected_records)
            and sha256_file(cache_path) == receipt.get("sha256")
        ):
            return (
                index,
                str(cache_path),
                str(receipt["sha256"]),
                int(receipt["records"]),
                int(receipt["decisions"]),
                dict(receipt["attachment"]),
                dict(receipt["tactical"]),
                True,
            )
    if sha256_file(path) != expected_digest:
        raise RuntimeError(f"resident shard digest mismatch: {path}")
    rows = list(iter_feature_shard(path))
    if len(rows) != int(expected_records):
        raise RuntimeError(f"resident shard record mismatch: {path}")
    sidecar = R260InziSidecarIndex(
        Path(sidecar_index_path),
        source_manifest_sha256=str(source_manifest_sha256),
        daily_meta_sha256s=dict(daily_meta_sha256s),
    )
    attachment = sidecar.attach_available_rows(
        rows,
        iter_daily_sidecar_rows(
            Path(sidecar_root),
            str(source_day),
            expected_meta_sha256=str(daily_meta_sha256),
        ),
    )
    tactical = attach_tactical_target_overlay(
        rows, Path(tactical_overlay_path), require_all=False
    )
    decisions = sum(len(row.decisions) for row in rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            pickle.dump(
                {
                    "schema": JOINED_SHARD_CACHE_SCHEMA,
                    "contract": cache_contract,
                    "rows": rows,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    cache_sha256 = sha256_file(cache_path)
    receipt = {
        "schema": JOINED_SHARD_CACHE_SCHEMA,
        "contract": cache_contract,
        "path": str(cache_path),
        "sha256": cache_sha256,
        "size_bytes": cache_path.stat().st_size,
        "records": len(rows),
        "decisions": decisions,
        "attachment": attachment,
        "tactical": tactical,
    }
    receipt_bytes = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    receipt_temporary = receipt_path.with_name(
        f".{receipt_path.name}.partial.{os.getpid()}"
    )
    try:
        with receipt_temporary.open("xb") as stream:
            stream.write(receipt_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(receipt_temporary, 0o444)
        os.link(receipt_temporary, receipt_path)
    finally:
        receipt_temporary.unlink(missing_ok=True)
    return (
        index,
        str(cache_path),
        cache_sha256,
        len(rows),
        decisions,
        attachment,
        tactical,
        False,
    )


def _parallel_load_manifest(
    path: Path,
    workers: int,
    *,
    sidecar_index_path: Path,
    source_manifest_sha256: str,
    daily_meta_sha256s: dict[str, str],
    tactical_overlay_path: Path,
    sidecar_root: Path,
    cache_root: Path,
) -> tuple[BootstrapDataset, dict[str, int], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tactical_sha256 = sha256_file(tactical_overlay_path)
    tasks = []
    for index, row in enumerate(payload["shards"]):
        shard = (path.parent / str(row["path"])).resolve()
        match = _DATE.search(shard.name)
        if match is None:
            raise RuntimeError(f"resident shard has no source date: {shard}")
        source_day = match.group(0)
        if source_day not in daily_meta_sha256s:
            raise RuntimeError(f"resident shard day is absent from sidecar binding: {source_day}")
        cache_contract = {
            "feature_sha256": str(row["sha256"]),
            "source_day": source_day,
            "source_manifest_sha256": str(source_manifest_sha256),
            "daily_meta_sha256": str(daily_meta_sha256s[source_day]),
            "tactical_overlay_sha256": tactical_sha256,
            "join_mode": "daily_linear_sidecar_stream_v1",
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cache_path = cache_root / f"{index:02d}-{source_day}-{cache_key}.pickle"
        tasks.append(
            (
                index,
                str(shard),
                str(row["sha256"]),
                int(row["stats"]["records_kept"]),
                str(sidecar_index_path),
                str(sidecar_root),
                source_day,
                str(daily_meta_sha256s[source_day]),
                str(source_manifest_sha256),
                dict(daily_meta_sha256s),
                str(tactical_overlay_path),
                str(cache_path),
                cache_contract,
            )
        )
    prepared: list[tuple[str, str, int, int, dict[str, Any], dict[str, Any]] | None] = [None] * len(tasks)
    decisions = 0
    attachment_totals = {
        "joined_decision_count": 0,
        "masked_unjoinable_decision_count": 0,
        "missing_sidecar_decision_count": 0,
        "incompatible_sidecar_decision_count": 0,
    }
    tactical_totals = {"roots": 0}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(max(1, int(workers)), len(tasks)),
        mp_context=context,
    ) as pool:
        for index, cache_path, cache_sha256, records, shard_decisions, attachment, tactical, cache_hit in pool.map(
            _load_shard, tasks
        ):
            prepared[index] = (
                cache_path,
                cache_sha256,
                int(records),
                int(shard_decisions),
                attachment,
                tactical,
            )
            decisions += int(shard_decisions)
            for name in attachment_totals:
                attachment_totals[name] += int(attachment[name])
            tactical_totals["roots"] += int(tactical["roots"])
            print(
                f"[r274-resident] prepared shard={index + 1}/{len(tasks)} "
                f"games={records} decisions={shard_decisions} cache_hit={cache_hit}",
                flush=True,
            )
    sequences: list[Any] = []
    for index, item in enumerate(prepared):
        if item is None:
            raise RuntimeError(f"resident joined shard was not prepared: {index}")
        cache_path_text, cache_sha256, records, shard_decisions, _attachment, _tactical = item
        cache_path = Path(cache_path_text)
        if sha256_file(cache_path) != cache_sha256:
            raise RuntimeError(f"resident joined shard cache digest changed: {cache_path}")
        with cache_path.open("rb") as stream:
            cached = pickle.load(stream)
            if stream.read(1):
                raise RuntimeError(f"resident joined shard cache has trailing bytes: {cache_path}")
        if (
            not isinstance(cached, dict)
            or cached.get("schema") != JOINED_SHARD_CACHE_SCHEMA
            or cached.get("contract") != tasks[index][-1]
            or not isinstance(cached.get("rows"), list)
            or len(cached["rows"]) != records
            or sum(len(row.decisions) for row in cached["rows"]) != shard_decisions
        ):
            raise RuntimeError(f"resident joined shard cache is invalid: {cache_path}")
        sequences.extend(cached["rows"])
        print(
            f"[r274-resident] loaded joined cache={index + 1}/{len(tasks)} "
            f"games={len(sequences)} decisions={sum(len(row.decisions) for row in sequences)}",
            flush=True,
        )
    totals = dict(payload["totals"])
    if (
        len(sequences) != int(totals["records_kept"])
        or decisions != int(totals["decisions_kept"])
    ):
        raise RuntimeError("resident corpus totals differ from immutable manifest")
    return (
        BootstrapDataset(sequences=sequences, conversion_stats=totals),
        attachment_totals,
        tactical_totals,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sidecar-binding", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--joined-cache-root", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--tactical-overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--loader-workers", type=int, default=20)
    parser.add_argument("--batch-games", type=int, default=16)
    parser.add_argument("--max-decisions-per-batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=274)
    args = parser.parse_args()

    manifest_identity = validate_full_expert_manifest(args.manifest)
    owner = load_r260_owner_contract(
        expected_sha256=None,
        expected_clarification_revision=None,
    )
    binding = json.loads(args.sidecar_binding.read_text(encoding="utf-8"))
    daily: dict[str, str] = {}
    for day, row in dict(binding["daily_sidecar_meta_receipts"]).items():
        meta = json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))
        daily[str(day)] = str(meta["meta_sha256"])
    first_meta = Path(str(next(iter(binding["daily_sidecar_meta_receipts"].values()))["path"]))
    sidecar_root = first_meta.parents[2]
    index = R260InziSidecarIndex(
        args.index,
        source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    index.assert_verified(
        expected_source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    dataset, attachment, tactical = _parallel_load_manifest(
        args.manifest.resolve(),
        args.loader_workers,
        sidecar_index_path=args.index.resolve(),
        source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
        tactical_overlay_path=args.tactical_overlay.resolve(),
        sidecar_root=sidecar_root.resolve(),
        cache_root=args.joined_cache_root.resolve(),
    )
    dataset.conversion_stats["r260_sidecar_provenance"] = {
        "schema": "poke_bot.r274_resident_sidecar_join/v1",
        "index": str(args.index.resolve()),
        "source_manifest_sha256": owner.source_manifest_sha256,
        **attachment,
    }
    print(
        "[r274-resident] RAM corpus ready "
        f"games={len(dataset.sequences)} decisions={dataset.n_decisions} "
        f"sidecar_joined={attachment['joined_decision_count']} "
        f"tactical_roots={tactical['roots']}",
        flush=True,
    )

    parent = checkpoint.load_checkpoint(args.base_checkpoint, map_location="cpu")
    expanded = dict(
        dict(parent.get("extra") or {})
        .get("expanded_head_training", {})
        .get("loss_weights", {})
    )
    cfg = TrainConfig(
        lr=1e-5,
        weight_decay=0.0,
        epochs=int(args.epochs),
        games_per_batch=int(args.batch_games),
        max_decisions_per_batch=int(args.max_decisions_per_batch),
        split_by_episode=True,
        value_loss_weight=1.0,
        aux_loss_weight=0.05,
        opp_hand_loss_weight=0.05,
        opp_remainder_loss_weight=0.05,
        lethal_threat_loss_weight=0.025,
        prize_race_loss_weight=0.025,
        setup_board_outcome_loss_weight=0.025,
        combo_state_loss_weight=0.0,
        visible_tutor_completion_loss_weight=0.025,
        terminal_conversion_loss_weight=0.025,
        tactical_sequence_outcome_loss_weight=0.025,
        expanded_head_loss_weights=expanded,
        collect_own_deck_promotion_metrics=True,
        pure_rl=False,
        entropy_bonus=0.0,
        amp=True,
        seed=int(args.seed),
    )
    result = supervised_r260_host_rehearsal_step(
        dataset,
        base_ckpt=args.base_checkpoint,
        output_path=args.output,
        archetype_id="alakazam",
        epochs=int(args.epochs),
        cfg=cfg,
        seed=int(args.seed),
        training_provenance={
            "schema": "poke_bot.r274_resident_expert_bootstrap/v1",
            "owner_revision_10_full_model_bootstrap": True,
            "optimizer_scope": "all_r195_to_r274_derivative_non_combo_weights",
            "combo_state_role": "dormant_checkpoint_pack_abi_only",
            "manifest": manifest_identity,
            "loader_workers": int(args.loader_workers),
            "resident_in_host_ram": True,
            "sidecar": attachment,
            "tactical": tactical,
        },
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
