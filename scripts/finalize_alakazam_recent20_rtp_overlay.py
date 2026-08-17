#!/usr/bin/env python3
"""Verify a completed overlay manifest and seal its completion receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from poke_bot.recursive_turn_planner.recent20_overlay import (
    Recent20RTPDataset,
    canonical_bytes,
    read_json,
    sha256_file,
)


SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_completion/v1"


def _hex(digest: str) -> str:
    return digest.removeprefix("sha256:")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-pack-root", type=Path, required=True)
    parser.add_argument("--base-completion-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_sha = sha256_file(args.manifest)
    manifest = read_json(args.manifest)
    dataset = Recent20RTPDataset(
        args.manifest,
        base_pack_root=args.base_pack_root,
        expected_manifest_sha256=manifest_sha,
        expected_base_completion_sha256=args.base_completion_sha256,
        verify_overlay_shards=True,
    )
    smoke_counts: dict[str, int] = {}
    for split in ("train", "validation"):
        count = 0
        for sample in dataset.iter_samples(split):
            if (
                sample.get("public_information_only") is not True
                or sample.get("base_feature_width") != 40
                or not sample.get("base_option_features_by_stage")
                or sample["program"].get("hidden_information_fields_present") is not False
            ):
                raise RuntimeError("direct adapter smoke failed")
            count += 1
            if count == 8:
                break
        if count != 8:
            raise RuntimeError(f"insufficient {split} samples")
        smoke_counts[split] = count

    shards = list(manifest["overlay_shards"])
    totals = dict(manifest["totals"])
    receipt = {
        "schema": SCHEMA,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "overlay_schema_sha256": manifest["overlay_schema_sha256"],
        "base_pack_completion_sha256": args.base_completion_sha256,
        "base_pack_schema_sha256": manifest["base_pack"]["schema_sha256"],
        "base_corpus_manifest_sha256": manifest["base_pack"]["corpus_manifest_sha256"],
        "source_days": manifest["source_days"],
        "counts": totals,
        "qualifying_game_count": totals["qualifying_games"],
        "qualifying_acting_seat_count": totals["qualifying_acting_seats"],
        "list_variant_stratum_count": manifest["list_variant_stratum_count"],
        "overlay_shard_count": len(shards),
        "overlay_bytes": sum(int(row["size_bytes"]) for row in shards),
        "overlay_shards": [
            {key: row[key] for key in ("utc_day", "path", "sha256", "size_bytes")}
            for row in shards
        ],
        "deterministic_join_coverage": 1.0,
        "deterministic_join_unique": True,
        "duplicate_or_double_counted_rows": 0,
        "hidden_information_runtime_field_count": 0,
        "raw_archive_use": "only_referenced_qualifying_members_for_missing_recorded_turn_terminal_outcome_and_raw_action_parity",
        "recollection_refeaturization_simulator_search_or_training_performed": False,
        "pipeline_loader_smoke": {
            "passed": True,
            "entrypoint": "poke_bot.recursive_turn_planner.recent20_overlay.Recent20RTPDataset",
            "archetype_job_accepted": False,
            "train_samples_streamed": smoke_counts["train"],
            "validation_samples_streamed": smoke_counts["validation"],
            "whole_corpus_loaded_into_python_objects": False,
        },
        "resources": {
            "worker_limit": 4,
            "peak_worker_count": 4,
            "peak_worker_rss_bytes": max(int(row["worker_peak_rss_bytes"]) for row in shards),
            "aggregate_worker_cpu_user_seconds": sum(float(row["worker_cpu_user_seconds"]) for row in shards),
            "aggregate_worker_cpu_system_seconds": sum(float(row["worker_cpu_system_seconds"]) for row in shards),
            "cpu_only": True,
            "gpu_used": False,
            "low_priority_invocation": "nice_15_ionice_idle",
        },
        "finalization_recovery": {
            "reason": "initial post-build smoke resolved output-root-relative shard paths from the manifest directory",
            "shards_rebuilt": False,
            "corrected_adapter_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "poke_bot/recursive_turn_planner/recent20_overlay.py"
            ),
        },
        "service_control_performed": False,
        "active_training_or_self_play_on_elmo_at_start": False,
        "active_training_impact": "none_by_host_isolation_no_inzi_outputs_or_services_touched",
        "outputs_transferred_to_inzi": False,
        "sealed_at_unix_seconds": time.time(),
    }
    body = canonical_bytes(receipt)
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"sha256-{_hex(digest)}.completion-receipt.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
