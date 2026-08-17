from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.set_specialist_iteration_window import update


def _registry(path: Path, *, override: bool = False) -> Path:
    row = {"status": "ready"}
    if override:
        row["iteration_ceiling"] = 10
    path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_runtime_registry/v1",
                "minimum_terminal_iteration": 5,
                "iteration_ceiling": 10,
                "specialists": {"example": row},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_sets_global_floor_and_ceiling_atomically(tmp_path: Path) -> None:
    path = _registry(tmp_path / "registry.json")

    receipt = update(path, floor=5, ceiling=15)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["minimum_terminal_iteration"] == 5
    assert payload["iteration_ceiling"] == 15
    assert receipt["prior"] == {
        "minimum_terminal_iteration": 5,
        "iteration_ceiling": 10,
    }
    assert receipt["per_specialist_overrides"] == 0


def test_rejects_per_specialist_window_override(tmp_path: Path) -> None:
    path = _registry(tmp_path / "registry.json", override=True)

    with pytest.raises(RuntimeError, match="overrides are forbidden"):
        update(path, floor=5, ceiling=15)
