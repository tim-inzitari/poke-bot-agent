"""Fail-closed replay-byte provenance tests for dynamic model inspection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from replay_inspector.catalog import scan_archive
from replay_inspector.config import InspectorConfig
from replay_inspector.provenance import PROVENANCE_SCHEMA, load_provenance_manifest
from replay_inspector.server import InspectorApplication


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[InspectorConfig, Path, int, int]:
    submission_id = 901
    episode_id = 42
    archive = tmp_path / "archive"
    replay_path = archive / str(submission_id) / f"episode-{episode_id}-replay.json"
    observation = {
        "current": {
            "turn": 1,
            "yourIndex": 0,
            "players": [{"hand": []}, {"hand": None}],
        },
        "select": {
            "context": "Hand",
            "option": [{}, {}],
            "minCount": 1,
            "maxCount": 1,
        },
    }
    _write(
        replay_path,
        json.dumps(
            {
                "steps": [
                    [
                        {"status": "ACTIVE", "observation": observation},
                        {"status": "INACTIVE", "observation": {}},
                    ],
                    [
                        {
                            "status": "ACTIVE",
                            "observation": observation,
                            "action": [1],
                        },
                        {"status": "INACTIVE", "observation": {}, "action": []},
                    ],
                ]
            }
        ),
    )
    _write(
        replay_path.parent / "episodes.json",
        json.dumps(
            {
                "submission_id": submission_id,
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_id": episode_id,
                        "own_agent": {"index": 0, "submission_id": submission_id},
                    }
                ],
            }
        ),
    )
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "model.pt", b"checkpoint")
    bundle = _write(assets / "submission.tar.gz", b"bundle")
    manifest = _write(
        tmp_path / "provenance.json",
        json.dumps(
            {
                "schema": PROVENANCE_SCHEMA,
                "version": 1,
                "records": [
                    {
                        "submission_id": submission_id,
                        "status": "verified",
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": _digest(checkpoint),
                        },
                        "bundle": {"path": str(bundle), "sha256": _digest(bundle)},
                        # This explicit null form keeps the fixture focused on
                        # the replay-byte gate rather than a tree requirement.
                        "runtime": {
                            "matchup_tree_path": None,
                            "matchup_tree_sha256": None,
                        },
                        "replay": {
                            "games": [
                                {
                                    "episode_id": episode_id,
                                    "replay_path": str(replay_path),
                                    "replay_sha256": _digest(replay_path),
                                }
                            ]
                        },
                    }
                ],
            }
        ),
    )
    return (
        InspectorConfig(
            replay_root=archive,
            rollout_root=tmp_path / "rollouts",
            provenance_manifest=manifest,
            artifact_roots=(assets,),
            web_root=Path(__file__).resolve().parents[1] / "replay_inspector" / "web",
        ),
        replay_path,
        submission_id,
        episode_id,
    )


def test_catalog_requires_one_matching_replay_digest_but_keeps_replay_browseable(
    tmp_path: Path,
) -> None:
    config, replay_path, submission_id, episode_id = _fixture(tmp_path)
    provenance = load_provenance_manifest(
        config.provenance_manifest, source_roots=config.source_roots
    )
    catalog = scan_archive(config.replay_root, provenance=provenance)
    entry = catalog.replay(submission_id, episode_id)

    assert entry is not None and entry.replay_available
    assert entry.model_analysis_available
    assert entry.expected_replay_sha256 == _digest(replay_path)
    assert entry.actual_replay_sha256 == _digest(replay_path)

    # The replacement is still valid replay JSON, so timeline browsing stays
    # useful.  It is no longer permitted to drive exact model analysis.
    _write(replay_path, replay_path.read_text(encoding="utf-8") + "\n")
    stale_catalog = scan_archive(config.replay_root, provenance=provenance)
    stale = stale_catalog.replay(submission_id, episode_id)
    assert stale is not None and stale.replay_available
    assert not stale.model_analysis_available
    assert "replay_sha256_mismatch" in stale.model_analysis_availability_reasons


def test_trace_rechecks_replay_bytes_before_any_checkpoint_load(tmp_path: Path) -> None:
    config, replay_path, submission_id, episode_id = _fixture(tmp_path)
    application = InspectorApplication(config)

    class NeverLoadCache:
        def load(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError(
                "stale replay bytes must be rejected before model load"
            )

    # Cataloguing saw the original bytes.  Change only whitespace after the
    # app has started to prove the request-time hash gate, not just scan-time
    # eligibility, protects the torch-load boundary.
    _write(replay_path, replay_path.read_text(encoding="utf-8") + "\n")
    application._model_cache = NeverLoadCache()  # type: ignore[assignment]
    trace = application.trace_payload(submission_id, episode_id, 0, 0)

    assert trace["recorded_action"]["action"] == [1]
    assert trace["model"]["availability"] == {
        "available": False,
        "reason": "replay_sha256_mismatch",
    }
    assert trace["heads"] == []
