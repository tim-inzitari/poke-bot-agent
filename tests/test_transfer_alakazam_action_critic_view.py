"""Focused safety coverage for the compact Alakazam critic-view transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import transfer_alakazam_action_critic_view as transfer


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _sha(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return _sha(body)


def _content_path(root: Path, directory: str, body: bytes, suffix: str) -> tuple[Path, str]:
    digest = _sha(body)
    path = root / directory / f"sha256-{digest.removeprefix('sha256:')}{suffix}"
    _write(path, body)
    return path, digest


def _fixture(tmp_path: Path) -> dict[str, object]:
    """Build a small internally sealed pack/overlay using real manifest rules."""

    base = tmp_path / "elmo-base"
    overlay = tmp_path / "elmo-overlay"
    destination = tmp_path / "bert-view"
    days = list(transfer.WINDOW_DAYS[:4])
    packs: list[dict[str, object]] = []
    for index, day in enumerate(days):
        files: dict[str, dict[str, object]] = {}
        for role, name in transfer.BASE_FILE_NAMES.items():
            body = (f"{day}:{role}:{index}\n").encode("utf-8")
            path = base / day / name
            digest = _write(path, body)
            files[role] = {
                "path": f"/sealed/base/{day}/{name}",
                "sha256": digest,
                "size_bytes": len(body),
            }
        receipt_body = _canonical({"schema": "fixture_base_receipt/v1", "day": day})
        receipt_sha = _write(base / day / "receipt.json", receipt_body)
        packs.append(
            {
                "files": files,
                "receipt_path": f"/sealed/base/{day}/receipt.json",
                "receipt_sha256": receipt_sha,
                "schema": transfer.BASE_PACK_SCHEMA,
                "source_path": f"/sealed/source/{day}.jsonl",
                "source_sha256": _sha(f"source:{day}".encode("utf-8")),
                "feature_width": 40,
                "feature_dtype": "float32_le",
            }
        )
    completion_body = _canonical(
        {
            "schema": transfer.BASE_COMPLETION_SCHEMA,
            "source_shard_count": len(packs),
            "packs": packs,
        }
    )
    completion_sha = _write(base / "COMPLETE.json", completion_body)

    corpus_body = _canonical({"schema": "fixture_corpus/v1", "days": days})
    corpus_path, corpus_sha = _content_path(
        overlay, "bindings", corpus_body, ".corpus-manifest.json"
    )
    binding_path, binding_sha = _content_path(
        overlay, "bindings", completion_body, ".base-completion.json"
    )
    assert binding_sha == completion_sha
    base_schema_body = _canonical({"schema": transfer.BASE_PACK_SCHEMA, "feature_width": 40})
    base_schema_path, base_schema_sha = _content_path(
        overlay, "schemas", base_schema_body, ".base-schema.json"
    )
    overlay_schema_body = _canonical({"schema": "fixture_complete_action_overlay/v1"})
    overlay_schema_path, overlay_schema_sha = _content_path(
        overlay, "schemas", overlay_schema_body, ".overlay-schema.json"
    )

    shards: list[dict[str, object]] = []
    for index, (day, pack) in enumerate(zip(days, packs, strict=True)):
        shard_body = _canonical({"schema": "fixture_overlay_row/v1", "day": day, "index": index})
        shard_path, shard_sha = _content_path(overlay, "objects", shard_body, ".rtp-overlay.jsonl")
        shards.append(
            {
                "utc_day": day,
                "split": "train",
                "path": str(shard_path.relative_to(overlay)),
                "sha256": shard_sha,
                "size_bytes": len(shard_body),
                "base_source_shard_sha256": pack["source_sha256"],
            }
        )
    manifest_body = _canonical(
        {
            "schema": transfer.OVERLAY_MANIFEST_SCHEMA,
            "overlay_schema_sha256": overlay_schema_sha,
            "source_days": days,
            "base_pack": {
                "completion_path": f"/sealed/overlay/{binding_path.name}",
                "completion_sha256": completion_sha,
                "corpus_manifest_path": f"/sealed/overlay/{corpus_path.name}",
                "corpus_manifest_sha256": corpus_sha,
                "schema_path": f"/sealed/overlay/{base_schema_path.name}",
                "schema_sha256": base_schema_sha,
                "feature_tensors_copied_into_overlay": False,
                "schema": {
                    "schema": transfer.BASE_PACK_SCHEMA,
                    "feature_width": 40,
                    "feature_dtype": "float32_le",
                },
            },
            "overlay_shards": shards,
        }
    )
    manifest_path, manifest_sha = _content_path(
        overlay, "manifests", manifest_body, ".overlay-manifest.json"
    )
    completion_receipt_body = _canonical(
        {
            "schema": transfer.OVERLAY_COMPLETION_SCHEMA,
            "manifest_sha256": manifest_sha,
            "base_pack_completion_sha256": completion_sha,
        }
    )
    completion_receipt_path, _ = _content_path(
        overlay, "receipts", completion_receipt_body, ".completion-receipt.json"
    )
    validation_body = _canonical(
        {"schema": "fixture_validation/v1", "manifest_sha256": manifest_sha}
    )
    validation_path, _ = _content_path(
        overlay, "validation-receipts", validation_body, ".pipeline-loader-validation.json"
    )
    return {
        "base": base,
        "overlay": overlay,
        "destination": destination,
        "manifest": manifest_path,
        "completion_receipt": completion_receipt_path,
        "validation": validation_path,
        "days": days,
    }


def _plan(fixture: dict[str, object]) -> tuple[dict[str, object], str]:
    return transfer.build_transfer_plan(
        source=transfer.LocalSourceReader(),
        base_root=fixture["base"],
        overlay_root=fixture["overlay"],
        destination_root=fixture["destination"],
        disk_floor_bytes=64,
        strict=False,
        overlay_manifest_path=fixture["manifest"],
        overlay_completion_receipt_path=fixture["completion_receipt"],
        overlay_validation_receipt_path=fixture["validation"],
    )


def test_builds_a_manifest_bound_four_lane_lpt_plan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)

    assert transfer.validate_transfer_plan(plan) == plan_sha
    assert plan["parallel_lanes_exact"] == 4
    assert len(plan["lanes"]) == 4
    assert len(plan["entries"]) == 32  # 4 * (four tensors + receipt), base complete, four overlays, seven metadata objects.
    assert sum(int(row["total_size_bytes"]) for row in plan["lanes"]) == plan["total_size_bytes"]
    assert {entry["lane_id"] for entry in plan["entries"]} == {0, 1, 2, 3}
    assert plan["source"]["read_only"] is True
    assert plan["source"]["overlay_manifest_sha256"].startswith("sha256:")


def test_plan_rejects_late_base_payload_corruption(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    corrupted = Path(fixture["base"]) / str(fixture["days"][-1]) / "features.f32"
    corrupted.write_bytes(b"tampered")

    with pytest.raises(transfer.CriticViewTransferError, match="SHA-256 mismatch"):
        _plan(fixture)


def test_execute_promotes_only_verified_partials_and_is_idempotent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    source = transfer.LocalSourceReader()
    free = lambda _path: 10**9

    first = transfer.execute_transfer_plan(
        plan,
        source=source,
        plan_sha256=plan_sha,
        free_bytes=free,
    )
    second = transfer.execute_transfer_plan(
        plan,
        source=source,
        plan_sha256=plan_sha,
        free_bytes=free,
    )

    destination = Path(fixture["destination"])
    assert first["completion_sha256"] == second["completion_sha256"]
    assert (destination / "base" / "COMPLETE.json").read_bytes() == (
        Path(fixture["base"]) / "COMPLETE.json"
    ).read_bytes()
    assert (destination / "overlay" / "manifests" / Path(fixture["manifest"]).name).read_bytes() == Path(
        fixture["manifest"]
    ).read_bytes()
    assert len(list((destination / "transfer" / "receipts").glob("*.transfer-receipt.json"))) == len(
        plan["entries"]
    )
    assert Path(first["completion_path"]).is_file()
    assert Path(first["training_view_path"]).is_file()
    assert not list((destination / "transfer" / "staging").rglob("*.part"))


def test_resume_refuses_a_partial_with_a_nonmatching_prefix(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    destination = Path(fixture["destination"])
    destination.mkdir()
    entry = next(row for row in plan["entries"] if row["role"] == "complete_action_overlay_shard")
    part = transfer._part_path(destination, plan_sha, entry)
    part.parent.mkdir(parents=True)
    part.write_bytes(b"x")

    with pytest.raises(transfer.CriticViewTransferError, match="prefix"):
        transfer._copy_into_part(transfer.LocalSourceReader(), entry, part, 1)


def test_disk_floor_rejects_execution_before_copying_any_object(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    needed = int(plan["total_size_bytes"])

    with pytest.raises(transfer.CriticViewTransferError, match="free space"):
        transfer.execute_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: 64 + needed - 1,
        )
    assert not Path(fixture["destination"]).exists()


def test_existing_final_conflict_is_never_overwritten(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    destination = Path(fixture["destination"])
    conflict = destination / "base" / "COMPLETE.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"wrong final bytes")

    with pytest.raises(transfer.CriticViewTransferError, match="conflicts"):
        transfer.execute_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: 10**9,
        )
    assert conflict.read_bytes() == b"wrong final bytes"
