from __future__ import annotations

import json
from pathlib import Path

from scripts.sync_completed_feature_shards import _complete_local_shards


def test_complete_local_shards_requires_sidecar_and_payload(tmp_path: Path) -> None:
    sidecar = tmp_path / "day.features.json"
    sidecar.write_text(json.dumps({"path": "day.features"}), encoding="utf-8")
    assert _complete_local_shards(tmp_path) == 0
    (tmp_path / "day.features").write_bytes(b"feature")
    assert _complete_local_shards(tmp_path) == 1
