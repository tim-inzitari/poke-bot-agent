from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.wait_for_feature_window import _available_dates


def _publish(root: Path, name: str, day: str) -> None:
    (root / f"{name}.features").write_bytes(b"feature")
    (root / f"{name}.features.json").write_text(
        json.dumps({"path": f"{name}.features", "source_dates": [day]}),
        encoding="utf-8",
    )


def test_available_dates_requires_complete_payload(tmp_path: Path) -> None:
    _publish(tmp_path, "first", "2026-07-02")
    assert _available_dates(tmp_path) == {"2026-07-02"}


def test_available_dates_rejects_overlap(tmp_path: Path) -> None:
    _publish(tmp_path, "first", "2026-07-02")
    _publish(tmp_path, "second", "2026-07-02")
    with pytest.raises(RuntimeError, match="overlapping feature date"):
        _available_dates(tmp_path)
