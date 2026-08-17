from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest

from scripts.finalize_split_latest20_feature_days import (
    collect,
    validate_completed_shards,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_collect_requires_every_exact_completed_day(tmp_path: Path) -> None:
    days = ["2026-08-01", "2026-08-02"]
    _write(
        tmp_path / "status" / "2026-08-01.json",
        {
            "state": "complete",
            "completed": [{"date": "2026-08-01", "decisions": 10}],
        },
    )
    assert collect(tmp_path, days) is None
    _write(
        tmp_path / "status" / "2026-08-02.json",
        {
            "state": "complete",
            "completed": [{"date": "2026-08-02", "decisions": 20}],
        },
    )
    assert [row["date"] for row in collect(tmp_path, days) or []] == days


def test_collect_fails_closed_on_failed_day(tmp_path: Path) -> None:
    _write(
        tmp_path / "status" / "2026-08-02.json",
        {"state": "failed", "completed": []},
    )
    with pytest.raises(RuntimeError, match="daily feature job failed"):
        collect(tmp_path, ["2026-08-02"])


def test_strategic_schema_guard_rejects_stale_worker_shard(tmp_path: Path) -> None:
    shard = tmp_path / "all-recognized-2026-08-02.features"
    with shard.open("wb") as stream:
        pickle.dump(
            {
                "format": "pokebot-bootstrap-feature-shard",
                "dataset_schema": 5,
                "feature_schema": 5,
            },
            stream,
        )
    rows = [{"date": "2026-08-02", "output": f"/output/{shard.name}"}]
    with pytest.raises(RuntimeError, match="stale or incompatible strategic schema"):
        validate_completed_shards(
            tmp_path,
            rows,
            required_dataset_schema=6,
            required_expanded_target_schema="poke_bot.expanded_strategic_targets/v2",
            required_expanded_target_digest="sha256:target",
        )


def test_strategic_schema_guard_accepts_exact_current_shard(tmp_path: Path) -> None:
    shard = tmp_path / "all-recognized-2026-08-02.features"
    with shard.open("wb") as stream:
        pickle.dump(
            {
                "format": "pokebot-bootstrap-feature-shard",
                "dataset_schema": 6,
                "feature_schema": 5,
                "expanded_strategic_targets": {
                    "schema": "poke_bot.expanded_strategic_targets/v2",
                    "digest": "sha256:target",
                },
            },
            stream,
        )
    rows = [{"date": "2026-08-02", "output": f"/output/{shard.name}"}]
    validate_completed_shards(
        tmp_path,
        rows,
        required_dataset_schema=6,
        required_expanded_target_schema="poke_bot.expanded_strategic_targets/v2",
        required_expanded_target_digest="sha256:target",
    )
