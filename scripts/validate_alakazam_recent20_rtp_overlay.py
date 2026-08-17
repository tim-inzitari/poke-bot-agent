#!/usr/bin/env python3
"""Run the existing ArchetypeRTPJob pipeline loader smoke on an overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.recursive_turn_planner.pipeline import (  # noqa: E402
    ArchetypeRTPJob,
    iter_dataset_samples_for_job,
)
from poke_bot.recursive_turn_planner.recent20_overlay import (  # noqa: E402
    canonical_bytes,
    read_json,
    sha256_file,
)


SCHEMA = "poke_bot.alakazam_recent20_rtp_pipeline_loader_validation/v1"


def _hex(value: str) -> str:
    raw = str(value).removeprefix("sha256:")
    if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
        raise ValueError("invalid SHA-256")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-pack-root", type=Path, required=True)
    parser.add_argument("--base-completion-sha256", required=True)
    parser.add_argument("--builder-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.manifest) != args.manifest_sha256:
        raise RuntimeError("manifest digest mismatch")
    builder_receipt_sha = sha256_file(args.builder_receipt)
    builder = read_json(args.builder_receipt)
    if builder.get("manifest_sha256") != args.manifest_sha256:
        raise RuntimeError("builder receipt is not bound to manifest")
    job = ArchetypeRTPJob(
        specialist_id="alakazam",
        recent20_rtp_overlay_manifest=str(args.manifest),
        recent20_rtp_overlay_manifest_digest=args.manifest_sha256,
        recent20_rtp_base_pack_root=str(args.base_pack_root),
        recent20_rtp_base_pack_completion_digest=args.base_completion_sha256,
        device="cpu",
    )
    if not job.ready_for_recent20_dataset or job.ready_for_host_train:
        raise RuntimeError("ArchetypeRTPJob overlay readiness boundary failed")
    counts = {}
    first_programs = {}
    for split in ("train", "validation"):
        count = 0
        for sample in iter_dataset_samples_for_job(
            job, split=split, verify_overlay_shards=True
        ):
            program = sample["program"]
            if (
                sample.get("public_information_only") is not True
                or sample.get("base_feature_width") != 40
                or program.get("hidden_information_fields_present") is not False
                or not program.get("complete_action_program_reconstructed")
                or len(sample.get("base_option_features_by_stage") or ())
                != len(program.get("stages") or ())
            ):
                raise RuntimeError("joined pipeline sample validation failed")
            if count == 0:
                first_programs[split] = program["program_identity"]
            count += 1
            if count >= 16:
                break
        if count != 16:
            raise RuntimeError(f"insufficient {split} pipeline samples")
        counts[split] = count
    receipt = {
        "schema": SCHEMA,
        "manifest_path": str(args.manifest),
        "manifest_sha256": args.manifest_sha256,
        "builder_completion_receipt_path": str(args.builder_receipt),
        "builder_completion_receipt_sha256": builder_receipt_sha,
        "base_pack_root": str(args.base_pack_root),
        "base_pack_completion_sha256": args.base_completion_sha256,
        "archetype_rtp_job_accepted": True,
        "pipeline_entrypoint": (
            "poke_bot.recursive_turn_planner.pipeline.iter_dataset_samples_for_job"
        ),
        "train_samples_streamed": counts["train"],
        "validation_samples_streamed": counts["validation"],
        "first_program_identity_by_split": first_programs,
        "deterministic_join_passed": True,
        "complete_recorded_action_program_reconstruction_passed": True,
        "whole_corpus_loaded_into_python_objects": False,
        "public_information_boundary_and_masks_passed": True,
        "parent_checkpoint_loaded": False,
        "rtp_training_search_mcts_or_simulator_execution_performed": False,
        "device": "cpu",
        "pipeline_source_sha256": sha256_file(
            ROOT / "poke_bot/recursive_turn_planner/pipeline.py"
        ),
        "overlay_adapter_source_sha256": sha256_file(
            ROOT / "poke_bot/recursive_turn_planner/recent20_overlay.py"
        ),
        "validated_at_unix_seconds": time.time(),
    }
    body = canonical_bytes(receipt)
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"sha256-{_hex(digest)}.pipeline-loader-validation.json"
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
