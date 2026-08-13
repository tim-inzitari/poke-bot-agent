from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.alakazam_rule_derivative_shard_limit_rev8 import (
    MAXIMUM_FINAL_SHARD_BYTES,
    Rev8ShardLimitError,
    build_revision_8_shard_limit_migration_receipt,
    load_revision_8_contract,
    sha256_file,
    validate_revision_7_half_run_under_revision_8,
)


WINDOW_DAYS = tuple(f"2026-07-{day:02d}" for day in range(13, 32)) + tuple(
    f"2026-08-{day:02d}" for day in range(1, 12)
)


def _write_half(root: Path, host_index: int, *, payload: bytes = b'{"row":1}\n') -> Path:
    shard_sha = "sha256:" + hashlib.sha256(payload).hexdigest()
    filename = f"sha256-{shard_sha[7:]}.refeaturization-census.shard"
    shards = root / "refeatured-records/shards"
    shards.mkdir(parents=True)
    (shards / filename).write_bytes(payload)
    receipt = {
        "schema": "poke_bot.alakazam_fast_distributed_refeature/v1",
        "status": "complete",
        "host_index": host_index,
        "day_count": 15,
        "days": list(WINDOW_DAYS[host_index::2]),
        "card_id_filter": 743,
        "acting_seat_only": True,
        "record_count": 1,
        "maximum_final_shard_bytes": MAXIMUM_FINAL_SHARD_BYTES,
        "shard_manifest": {
            "maximum_shard_size_bytes": MAXIMUM_FINAL_SHARD_BYTES,
            "record_scope": "acting_seat_card_743_materialized_rows",
            "all_shards_individually_schema_sha256_size_validated": True,
            "record_count": 1,
            "shard_count": 1,
            "shards": [
                {
                    "logical_shard_id": "public-token-bucket-00000",
                    "filename": filename,
                    "sha256": shard_sha,
                    "size_bytes": len(payload),
                    "record_count": 1,
                    "schema": "poke_bot.alakazam_collision_census_r298_refeatured_record_shard/v1",
                }
            ],
        },
    }
    path = root / "COMPLETE.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


def test_current_revision_8_contract_loads() -> None:
    contract = load_revision_8_contract()
    assert contract["goal_revision"] == 8
    assert (
        contract["revision_8_one_gib_content_addressed_shards"][
            "maximum_final_content_addressed_shard_bytes"
        ]
        == MAXIMUM_FINAL_SHARD_BYTES
    )


def test_two_revision_7_halves_seal_under_revision_8(tmp_path: Path) -> None:
    first = _write_half(tmp_path / "first", 0)
    second = _write_half(tmp_path / "second", 1)
    receipt = build_revision_8_shard_limit_migration_receipt(
        half_run_receipt_paths=[second, first],
        raw_manifest_sha256="sha256:" + "1" * 64,
        sealed_at_utc="2026-08-13T04:00:00Z",
    )
    assert receipt["goal_revision"] == 8
    assert receipt["all_final_objects_maximum_size_bytes"] == MAXIMUM_FINAL_SHARD_BYTES
    assert receipt["revision_7_run_receipt_sha256s"] == [sha256_file(first), sha256_file(second)]
    assert receipt["restart_or_recompute_due_only_to_revision_8"] is False


def test_non_content_addressed_shard_is_rejected(tmp_path: Path) -> None:
    complete = _write_half(tmp_path / "first", 0)
    value = json.loads(complete.read_text())
    value["shard_manifest"]["shards"][0]["filename"] = "not-content-addressed.refeaturization-census.shard"
    complete.write_text(json.dumps(value) + "\n")
    with pytest.raises(Rev8ShardLimitError, match="not content addressed"):
        validate_revision_7_half_run_under_revision_8(complete)


def test_wrong_day_partition_is_rejected(tmp_path: Path) -> None:
    complete = _write_half(tmp_path / "first", 0)
    value = json.loads(complete.read_text())
    value["days"][0] = "2026-07-14"
    complete.write_text(json.dumps(value) + "\n")
    with pytest.raises(Rev8ShardLimitError, match="exact 15-day partition"):
        validate_revision_7_half_run_under_revision_8(complete)
