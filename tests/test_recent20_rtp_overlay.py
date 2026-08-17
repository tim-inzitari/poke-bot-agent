from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from poke_bot.recursive_turn_planner.pipeline import (
    ArchetypeRTPJob,
    iter_dataset_samples_for_job,
)
from poke_bot.recursive_turn_planner.recent20_overlay import (
    BASE_COMPLETION_SCHEMA,
    BASE_SCHEMA,
    MANIFEST_SCHEMA,
    OVERLAY_SCHEMA,
    base_schema_descriptor,
    canonical_bytes,
    canonical_sha256,
    overlay_schema_document,
    sha256_file,
)
from poke_bot.recursive_turn_planner.recent20_shadow import Recent20SemanticAdapter


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = value if isinstance(value, bytes) else canonical_bytes(value)
    path.write_bytes(body)
    return sha256_file(path)


def test_recent20_overlay_job_streams_base_join_without_checkpoint(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    day_root = base_root / "2026-07-23"
    day_root.mkdir(parents=True)
    features = struct.pack("<80f", *[float(index) for index in range(80)])
    files = {}
    for role, name, body in (
        ("features_f32", "features.f32", features),
        ("decision_offsets_u64", "decision_offsets.u64", struct.pack("<2Q", 0, 2)),
        ("selected_option_u32", "selected_option.u32", struct.pack("<I", 1)),
        ("decision_key_sha256", "decision_keys.sha256", b"k" * 32),
    ):
        path = day_root / name
        digest = _write(path, body)
        files[role] = {"path": str(path), "sha256": digest, "size_bytes": len(body)}
    receipt_path = day_root / "receipt.json"
    receipt_sha = _write(receipt_path, {"schema": BASE_SCHEMA})
    pack = {
        "schema": BASE_SCHEMA,
        "source_path": "/source/2026-07-23/source.jsonl",
        "source_sha256": "sha256:" + "1" * 64,
        "receipt_path": "/source/2026-07-23/receipt.json",
        "receipt_sha256": receipt_sha,
        "feature_width": 40,
        "feature_dtype": "float32_le",
        "option_occurrence_count": 2,
        "decision_occurrence_count": 1,
        "deduplicated": False,
        "files": files,
    }
    completion = {
        "schema": BASE_COMPLETION_SCHEMA,
        "corpus_manifest_sha256": "sha256:" + "2" * 64,
        "source_shard_count": 1,
        "option_occurrence_count": 2,
        "decision_occurrence_count": 1,
        "deduplicated": False,
        "packs": [pack],
    }
    completion_path = tmp_path / "completion.json"
    completion_sha = _write(completion_path, completion)
    schema_sha = canonical_sha256(base_schema_descriptor(completion))
    overlay_row = {
        "schema": OVERLAY_SCHEMA,
        "hidden_information_fields_present": False,
        "stages": [
            {
                "base_ref": {
                    "option_start": 0,
                    "option_count": 2,
                }
            }
        ],
    }
    shard_path = tmp_path / "objects" / "row.jsonl"
    shard_sha = _write(shard_path, overlay_row)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_pack": {
            "completion_path": str(completion_path),
            "completion_sha256": completion_sha,
            "schema_sha256": schema_sha,
        },
        "overlay_shards": [
            {
                "path": str(shard_path.relative_to(tmp_path)),
                "sha256": shard_sha,
                "size_bytes": shard_path.stat().st_size,
                "utc_day": "2026-07-23",
                "split": "train",
                "base_source_shard_sha256": pack["source_sha256"],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_sha = _write(manifest_path, manifest)
    job = ArchetypeRTPJob(
        specialist_id="alakazam",
        recent20_rtp_overlay_manifest=str(manifest_path),
        recent20_rtp_overlay_manifest_digest=manifest_sha,
        recent20_rtp_base_pack_root=str(base_root),
        recent20_rtp_base_pack_completion_digest=completion_sha,
    )
    assert job.ready_for_recent20_dataset is True
    assert job.ready_for_host_train is False
    sample = next(iter(iter_dataset_samples_for_job(job, split="train")))
    assert sample["public_information_only"] is True
    assert sample["base_feature_width"] == 40
    assert len(sample["base_option_features_by_stage"][0]) == 2
    assert sample["base_option_features_by_stage"][0][1][0] == 40.0


def test_overlay_schema_forbids_hidden_opponent_and_copies_no_tensors() -> None:
    schema = overlay_schema_document()
    assert schema["base_reference"]["copied_feature_tensors"] is False
    assert "opponent_deck_multiset_sha256" in schema["forbidden_fields"]
    assert schema["counterfactual_targets"] is False


def test_recent20_semantic_adapter_state_is_option_permutation_invariant() -> None:
    torch.manual_seed(7)
    adapter = Recent20SemanticAdapter(d_model=96, hidden_width=32)
    features = torch.randn(5, 40)
    state, options = adapter.encode(features)
    order = torch.tensor([3, 1, 4, 0, 2])
    reordered_state, reordered_options = adapter.encode(features[order])
    assert torch.allclose(state, reordered_state, atol=1e-6, rtol=1e-6)
    assert torch.allclose(options[order], reordered_options, atol=1e-6, rtol=1e-6)
