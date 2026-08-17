from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS
from scripts.sync_latest20_specialist_corpora_from_elmo import (
    atomic_ready_json,
    atomic_symlink,
    tree_bytes,
    validate_balanced_core,
    validate_corpora,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_ready_receipt_is_checksum_stable_across_noop_runs(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "sync.json"
    first = {
        "schema": "poke_bot.latest20_specialist_sync/v1",
        "status": "ready",
        "destination": "/immutable/corpus",
        "completed_at_utc": "2026-08-03T05:00:00+00:00",
    }
    atomic_ready_json(receipt, first)
    first_bytes = receipt.read_bytes()

    rerun = dict(first)
    rerun["completed_at_utc"] = "2026-08-03T06:00:00+00:00"
    atomic_ready_json(receipt, rerun)

    assert receipt.read_bytes() == first_bytes


def test_ready_receipt_updates_content_but_preserves_completion_time(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "sync.json"
    first = {
        "schema": "poke_bot.latest20_specialist_sync/v1",
        "status": "ready",
        "destination": "/immutable/corpus",
        "copied_bytes": 10,
        "completed_at_utc": "2026-08-03T05:00:00+00:00",
    }
    atomic_ready_json(receipt, first)
    corrected = dict(first)
    corrected["copied_bytes"] = 11
    corrected["completed_at_utc"] = "2026-08-03T06:00:00+00:00"
    atomic_ready_json(receipt, corrected)

    published = json.loads(receipt.read_text(encoding="utf-8"))
    assert published["copied_bytes"] == 11
    assert published["completed_at_utc"] == first["completed_at_utc"]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    ids = [f"specialist-{index:02d}" for index in range(18)]
    roster = tmp_path / "roster.json"
    _write_json(
        roster,
        {
            "schema": "poke_bot.matchup_adapter_roster/v1",
            "required_specialist_count": 18,
            "expert_ids": ids,
        },
    )
    root = tmp_path / "corpora"
    results = []
    for index, specialist_id in enumerate(ids):
        directory = root / specialist_id
        directory.mkdir(parents=True)
        if index == 0:
            manifest = directory / "manifest.json"
            _write_json(manifest, {"specialist": specialist_id})
            digest = _sha256(manifest)
            _write_json(
                directory / "PROTECTED_EXPERT_CORPUS.json",
                {
                    "schema": "poke_bot.pinned_expert_corpus/v1",
                    "protected": True,
                    "manifest": manifest.name,
                    "manifest_sha256": digest,
                },
            )
            results.append(
                {
                    "archetype": specialist_id,
                    "status": "ready",
                    "manifest_sha256": digest,
                }
            )
        else:
            _write_json(
                directory / "UNAVAILABLE_EXPERT_CORPUS.json",
                {"archetype": specialist_id},
            )
            results.append(
                {"archetype": specialist_id, "status": "unavailable"}
            )
    _write_json(
        root / "SPECIALIST_CORPORA_READY.json",
        {
            "schema": "poke_bot.specialist_expert_corpora_ready/v1",
            "results": results,
        },
    )
    return root, roster, root / ids[0] / "manifest.json"


def test_validate_corpora_checks_all_18_specialists(tmp_path: Path) -> None:
    root, roster, _manifest = _fixture(tmp_path)
    receipt = validate_corpora(root, roster)
    assert len(receipt["results"]) == 18


def test_validate_corpora_rejects_manifest_drift(tmp_path: Path) -> None:
    root, roster, manifest = _fixture(tmp_path)
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="protected corpus identity failed"):
        validate_corpora(root, roster)


def test_validate_corpora_requires_checksum_bound_expanded_targets(
    tmp_path: Path,
) -> None:
    root, roster, manifest = _fixture(tmp_path)
    schema = "poke_bot.expanded_strategic_targets/v2"
    digest = "sha256:" + "a" * 64
    expanded = {
        "schema": schema,
        "digest": digest,
        "decisions": 1,
        "head_coverage": {
            head_id: {
                "labeled_rows": 1,
                "masked_rows": 0,
                "total_rows": 1,
            }
            for head_id in EXPANDED_HEAD_IDS
        },
    }
    manifest_payload = {
        "specialist": manifest.parent.name,
        "totals": {"decisions_kept": 1},
        "expanded_strategic_targets": expanded,
    }
    _write_json(manifest, manifest_payload)
    manifest_digest = _sha256(manifest)
    pointer = manifest.parent / "PROTECTED_EXPERT_CORPUS.json"
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    pointer_payload["manifest_sha256"] = manifest_digest
    pointer_payload["expanded_strategic_targets"] = expanded
    _write_json(pointer, pointer_payload)
    ready = root / "SPECIALIST_CORPORA_READY.json"
    ready_payload = json.loads(ready.read_text(encoding="utf-8"))
    ready_payload["results"][0]["manifest_sha256"] = manifest_digest
    _write_json(ready, ready_payload)

    validate_corpora(
        root,
        roster,
        required_expanded_target_schema=schema,
        required_expanded_target_digest=digest,
    )

    # A causal target may be genuinely unavailable for every row in a small
    # specialist shard. Exact masking is valid; fabricating one positive label
    # to satisfy the transfer validator would violate the training contract.
    head_id = next(iter(EXPANDED_HEAD_IDS))
    expanded["head_coverage"][head_id] = {
        "labeled_rows": 0,
        "masked_rows": 1,
        "total_rows": 1,
    }
    _write_json(manifest, manifest_payload)
    manifest_digest = _sha256(manifest)
    pointer_payload["manifest_sha256"] = manifest_digest
    pointer_payload["expanded_strategic_targets"] = expanded
    _write_json(pointer, pointer_payload)
    ready_payload["results"][0]["manifest_sha256"] = manifest_digest
    _write_json(ready, ready_payload)
    validate_corpora(
        root,
        roster,
        required_expanded_target_schema=schema,
        required_expanded_target_digest=digest,
    )

    pointer_payload["expanded_strategic_targets"]["decisions"] = 2
    _write_json(pointer, pointer_payload)
    with pytest.raises(
        RuntimeError,
        match="expanded strategic corpus identity failed",
    ):
        validate_corpora(
            root,
            roster,
            required_expanded_target_schema=schema,
            required_expanded_target_digest=digest,
        )


def test_validate_balanced_core_requires_exact_expanded_head_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "core-balanced-v6"
    root.mkdir()
    schema = "poke_bot.expanded_strategic_targets/v2"
    digest = "sha256:" + "b" * 64
    expanded = {
        "schema": schema,
        "digest": digest,
        "decisions": 11,
        "head_coverage": {
            head_id: {
                "labeled_rows": 7,
                "masked_rows": 4,
                "total_rows": 11,
            }
            for head_id in EXPANDED_HEAD_IDS
        },
    }
    manifest = root / "manifest.json"
    _write_json(
        manifest,
        {
            "totals": {"decisions_kept": 11},
            "expanded_strategic_targets": expanded,
        },
    )
    pointer = root / "PROTECTED_CORE_CORPUS.json"
    _write_json(
        pointer,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": manifest.name,
            "manifest_sha256": _sha256(manifest),
            "expanded_strategic_targets": expanded,
        },
    )

    result = validate_balanced_core(
        root,
        required_expanded_target_schema=schema,
        required_expanded_target_digest=digest,
    )
    assert result["decisions"] == 11
    wrong = json.loads(manifest.read_text(encoding="utf-8"))
    wrong["expanded_strategic_targets"]["head_coverage"].pop(
        EXPANDED_HEAD_IDS[-1]
    )
    wrong["expanded_strategic_targets"]["head_coverage"]["wrong-head"] = {
        "labeled_rows": 7,
        "masked_rows": 4,
        "total_rows": 11,
    }
    _write_json(manifest, wrong)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    pointer_payload["manifest_sha256"] = _sha256(manifest)
    pointer_payload["expanded_strategic_targets"] = wrong[
        "expanded_strategic_targets"
    ]
    _write_json(pointer, pointer_payload)
    with pytest.raises(
        RuntimeError, match="expanded balanced-core corpus identity failed"
    ):
        validate_balanced_core(
            root,
            required_expanded_target_schema=schema,
            required_expanded_target_digest=digest,
        )


def test_atomic_symlink_replaces_prior_pointer(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    current = tmp_path / "current"
    atomic_symlink(first, current)
    assert current.resolve() == first
    atomic_symlink(second, current)
    assert current.resolve() == second


def test_tree_bytes_reports_resumable_partial_payload(tmp_path: Path) -> None:
    staging = tmp_path / ".corpus.partial"
    (staging / "a").mkdir(parents=True)
    (staging / "a" / "one.bin").write_bytes(b"1234")
    (staging / "two.bin").write_bytes(b"567")
    assert tree_bytes(staging) == 7
