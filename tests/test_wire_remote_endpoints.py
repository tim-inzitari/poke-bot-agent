"""Static guards for --remote-worker-endpoints trainer wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_RR = ROOT / "scripts" / "train_round_robin.py"
TRAIN_CORE = ROOT / "scripts" / "train_core_pipeline.py"


@pytest.mark.unit
def test_train_round_robin_exposes_remote_worker_endpoints_flag() -> None:
    source = TRAIN_RR.read_text(encoding="utf-8")
    assert "--remote-worker-endpoints" in source
    assert "RemoteWorkerFarm" in source
    assert "_map_game_batch" in source
    assert "iter_additive_results" in source
    assert "remote_farm.reload_all" in source


@pytest.mark.unit
def test_train_core_pipeline_passes_remote_endpoints_to_hammer_rl() -> None:
    source = TRAIN_CORE.read_text(encoding="utf-8")
    assert "--remote-worker-endpoints" in source
    assert "_map_game_batch" in source
    assert 'command.extend' in source or "remote_worker_endpoints" in source
