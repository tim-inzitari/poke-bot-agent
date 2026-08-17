from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts import finalize_matchup_adapter_fleet as finalizer
from scripts import merge_matchup_adapter_route_workers as merger


def _worker_payload(*, manifest_digest: str) -> dict:
    return {
        "schema": merger.WORKER_SCHEMA,
        "source_checkpoint_digest": "sha256:source",
        "staged_manifest_digest": manifest_digest,
        "canonical_config": {"epochs": 25},
        "complete": True,
    }


def test_worker_loader_rejects_a_manifest_not_pinned_by_source(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "worker.pt"
    torch.save(_worker_payload(manifest_digest="sha256:wrong"), artifact)

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        merger._load_worker(
            artifact,
            "sha256:source",
            {"epochs": 25},
            "sha256:expected",
        )


def test_worker_loader_accepts_only_the_exact_pinned_contract(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "worker.pt"
    expected = _worker_payload(manifest_digest="sha256:expected")
    torch.save(expected, artifact)

    assert (
        merger._load_worker(
            artifact,
            "sha256:source",
            {"epochs": 25},
            "sha256:expected",
        )
        == expected
    )


def test_finalizer_detects_an_active_canonical_recovery(monkeypatch) -> None:
    class Result:
        returncode = 0

    monkeypatch.setattr(finalizer.subprocess, "run", lambda *args, **kwargs: Result())
    assert finalizer._service_active("pokebot-matchup-adapter-v31-recovery.service")


def test_fleet_publication_is_checksum_guarded_and_preserves_backups() -> None:
    source = Path(finalizer.__file__).read_text(encoding="utf-8")
    assert "checkpoint.checkpoint_digest(SOURCE) == expected_source_digest" in source
    assert "refusing fleet publication while the canonical recovery fit is active" in source
    assert 'f"{name}.before-fleet-{backup_suffix}"' in source
    worker = (
        Path(finalizer.__file__).resolve().parent
        / "train_matchup_adapter_route_worker.py"
    ).read_text(encoding="utf-8")
    assert "fleet worker staged manifest differs from the checksum-pinned source" in worker
    assert worker.count("pack_temporal_games=True") == 2
    assert '"active_epoch": epoch + 1' in worker
    assert '"active_route": route_id' in worker


def test_fleet_merge_always_materializes_a_best_checkpoint() -> None:
    source = Path(merger.__file__).read_text(encoding="utf-8")
    assert "best_payload: dict[str, Any] = copy.deepcopy(source)" in source
    assert 'checkpoint.atomic_torch_save(best_payload, output / "best.pt")' in source


def test_published_fleet_finalizer_requires_a_digest_verified_epoch_25(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source"
    fleet_dir = tmp_path / "fleet"
    source_dir.mkdir()
    fleet_dir.mkdir()
    final_path = source_dir / "final.pt"
    payload = {
        "epoch": 25,
        "extra": {
            "matchup_adapter_fit_complete": True,
            "streaming_matchup_adapter_state": {
                "epoch": 25,
                "complete": True,
            },
        },
    }
    torch.save(payload, final_path)
    status = fleet_dir / "fleet-finalizer.json"
    status.write_text(
        json.dumps(
            {
                "phase": "published",
                "final_checkpoint": str(final_path),
                "final_checkpoint_digest": finalizer.checkpoint.checkpoint_digest(
                    final_path
                ),
            }
        )
    )
    monkeypatch.setattr(finalizer, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(finalizer, "FLEET_DIR", fleet_dir)
    monkeypatch.setattr(finalizer, "STATUS", status)
    assert finalizer._published_checkpoint_is_valid()

    payload["epoch"] = 24
    torch.save(payload, final_path)
    assert not finalizer._published_checkpoint_is_valid()
