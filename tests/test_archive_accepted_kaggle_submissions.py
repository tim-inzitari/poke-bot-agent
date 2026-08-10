from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.archive_accepted_kaggle_submissions_to_elmo import accepted_entries


def _row(tmp_path: Path, submission_id: int, status: str = "accepted") -> dict:
    bundle = tmp_path / f"{submission_id}.tar.gz"
    bundle.write_bytes(str(submission_id).encode())
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    return {
        "submission_id": submission_id,
        "queue_status": status,
        "file": str(bundle),
        "file_sha256": digest,
        "checkpoint_checksum": "sha256:" + "a" * 64,
        "model_checksum": "sha256:" + "a" * 64,
        "matchup_tree_checksum": "sha256:" + "b" * 64,
    }


def test_selects_every_exact_accepted_submission_in_range(tmp_path: Path) -> None:
    payload = {
        "schema": "poke_bot.kaggle_submission_queue/v1",
        "queue": [
            _row(tmp_path, 55200000),
            _row(tmp_path, 55345107),
            _row(tmp_path, 55359777),
            _row(tmp_path, 55378477),
            _row(tmp_path, 55390000),
            _row(tmp_path, 55360000, "failed"),
        ],
    }
    selected = accepted_entries(payload, minimum_id=55345107, maximum_id=55378477)
    assert [row["submission_id"] for row in selected] == [
        55345107,
        55359777,
        55378477,
    ]


def test_rejects_changed_bundle_bytes(tmp_path: Path) -> None:
    row = _row(tmp_path, 55359777)
    Path(row["file"]).write_bytes(b"changed")
    payload = {"schema": "poke_bot.kaggle_submission_queue/v1", "queue": [row]}
    with pytest.raises(ValueError, match="bundle digest mismatch"):
        accepted_entries(payload, minimum_id=1, maximum_id=None)


def test_rejects_duplicate_submission_identity(tmp_path: Path) -> None:
    row = _row(tmp_path, 55359777)
    payload = {
        "schema": "poke_bot.kaggle_submission_queue/v1",
        "queue": [row, json.loads(json.dumps(row))],
    }
    with pytest.raises(ValueError, match="duplicate accepted submission id"):
        accepted_entries(payload, minimum_id=1, maximum_id=None)
