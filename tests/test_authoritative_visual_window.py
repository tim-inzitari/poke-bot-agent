from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_authoritative_alakazam_window import _atomic_json, _dates


def test_date_window_is_inclusive_and_rejects_reverse_order() -> None:
    assert _dates("2026-07-18", "2026-07-20") == [
        "2026-07-18",
        "2026-07-19",
        "2026-07-20",
    ]
    with pytest.raises(ValueError, match="precede"):
        _dates("2026-07-20", "2026-07-18")


def test_status_json_replaces_atomically_without_partial_files(tmp_path: Path) -> None:
    target = tmp_path / "window.status.json"
    _atomic_json(target, {"state": "running", "completed": []})
    _atomic_json(target, {"state": "complete", "completed": ["2026-07-20"]})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "state": "complete",
        "completed": ["2026-07-20"],
    }
    assert not [path for path in tmp_path.iterdir() if ".partial." in path.name]
