"""Focused contract tests for the read-only replay inspector catalogue."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_replay_inspector_provenance.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_inspector.provenance import load_provenance_manifest


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _cache(root: Path, submission_id: int, episode_id: int) -> Path:
    cache = root / "cache" / str(submission_id)
    replay = cache / f"episode-{episode_id}-replay.json"
    _write_json(replay, {"configuration": {"seed": 7}, "steps": [[{}], [{}]]})
    _write_json(
        cache / "episodes.json",
        {
            "submission_id": submission_id,
            "episodes": [
                {
                    "episode_id": episode_id,
                    "submission_id": submission_id,
                    "state": "COMPLETED",
                    "type": "PUBLIC",
                    "own_agent": {
                        "index": 0,
                        "submission_id": submission_id,
                        "reward": 1,
                    },
                    "agents": [
                        {"index": 0, "submission_id": submission_id, "reward": 1},
                        {"index": 1, "submission_id": 998, "reward": 0},
                    ],
                }
            ],
        },
    )
    return root / "cache"


def test_builds_checksum_bound_catalog_from_separate_evidence_and_artifacts(
    tmp_path: Path,
) -> None:
    submission_id = 55400001
    replay_root = _cache(tmp_path, submission_id, 42)
    artifacts = tmp_path / "artifact-store"
    checkpoint = artifacts / "opaque" / "alpha.bin"
    bundle = artifacts / "opaque" / "submission.blob"
    tree = artifacts / "routes" / "tree.dat"
    runtime = artifacts / "runtime" / "config.json"
    checkpoint.parent.mkdir(parents=True)
    tree.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable checkpoint bytes")
    bundle.write_bytes(b"immutable bundle bytes")
    tree.write_bytes(b"immutable matchup tree")
    _write_json(runtime, {"schema": "runtime/v1", "frozen": True})
    parity_receipt = artifacts / "runtime" / "parity-receipt.json"
    _write_json(
        parity_receipt,
        {
            "schema": "poke_bot.replay_model_inspector_runtime_parity_receipt/v1",
            "version": 1,
            "status": "verified",
            "submission_id": submission_id,
            "checkpoint_sha256": _digest(checkpoint),
            "bundle_sha256": _digest(bundle),
            "runtime_package_sha256": _digest(bundle),
            "runtime_source_tree_sha256": "sha256:" + "1" * 64,
            "verification": {
                "method": "independent_exact_runtime_parity",
                "verified_by": "test-fixture",
                "verified_at_utc": "2026-08-07T00:00:00Z",
            },
        },
    )

    queue = tmp_path / "queue.json"
    frozen = tmp_path / "frozen.json"
    _write_json(
        queue,
        {
            "queue": [
                {
                    "submission_id": submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "checkpoint_path": str(checkpoint),
                    "bundle_checksum": _digest(bundle),
                    "bundle_path": str(bundle),
                    "label": "final checkpoint 55400001",
                }
            ]
        },
    )
    _write_json(
        frozen,
        {
            "specialists": [
                {
                    "specialist_id": "alakazam",
                    "source_passing_checkpoint_digest": _digest(checkpoint),
                    "matchup_tree_checksum": _digest(tree),
                    "matchup_tree_path": str(tree),
                    "runtime_registry": str(runtime),
                    "runtime_registry_checksum": _digest(runtime),
                }
            ]
        },
    )
    output = tmp_path / "inspector-catalog.json"
    source_before = {
        path: path.read_bytes()
        for path in (queue, frozen, checkpoint, bundle, tree, parity_receipt)
    }

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(queue),
        "--evidence",
        str(frozen),
        "--artifact-root",
        str(artifacts),
        "--runtime-parity-receipt",
        f"{submission_id}={parity_receipt}",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    catalog = json.loads(output.read_text(encoding="utf-8"))
    assert catalog["schema"] == "poke_bot.replay_model_inspector_provenance/v1"
    assert (
        catalog["selection_authority"]
        == "submission_id_plus_checksum_bound_evidence_only"
    )
    assert catalog["version"] == 1
    assert catalog["summary"]["verified_submission_count"] == 1
    submission = catalog["records"]
    assert len(submission) == 1
    item = submission[0]
    assert item["submission_id"] == submission_id
    assert item["status"] == "verified"
    assert item["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "sha256": _digest(checkpoint),
    }
    assert item["bundle"] == {"path": str(bundle.resolve()), "sha256": _digest(bundle)}
    assert item["matchup_tree"] == {
        "path": str(tree.resolve()),
        "sha256": _digest(tree),
    }
    assert item["runtime_package"] == {
        "path": str(bundle.resolve()),
        "sha256": _digest(bundle),
    }
    assert item["runtime_parity_receipt"] == {
        "path": str(parity_receipt.resolve()),
        "sha256": _digest(parity_receipt),
    }
    assert item["identity"]["checkpoint"]["checksum"] == _digest(checkpoint)
    assert item["identity"]["checkpoint"]["verified_paths"] == [
        str(checkpoint.resolve())
    ]
    assert item["identity"]["bundle"]["checksum"] == _digest(bundle)
    assert item["identity"]["bundle"]["verified_paths"] == [str(bundle.resolve())]
    assert item["identity"]["model"]["checksum"] == _digest(checkpoint)
    assert item["identity"]["model"]["derived_from_checkpoint_checksum"] is True
    assert item["identity"]["matchup_tree"]["verified_paths"] == [str(tree.resolve())]
    assert item["label"] == "final checkpoint 55400001"
    assert item["identity"]["label"] == {
        "value": "final checkpoint 55400001",
        "evidence": [
            {
                "path": str(queue.resolve()),
                "sha256": _digest(queue),
                "pointer": "/queue/0",
                "role": "evidence",
                "key": "label",
            }
        ],
        "availability": "available",
        "unavailable_reason": None,
    }
    assert item["identity"]["specialist_id"]["value"] == "alakazam"
    assert item["identity"]["runtime_config"]["declared_paths"] == [str(runtime)]
    game = item["replay"]["games"][0]
    assert game["episode_id"] == 42
    assert game["replay_path"] == str(
        (replay_root / str(submission_id) / "episode-42-replay.json").resolve()
    )
    assert game["replay_sha256"] == _digest(
        replay_root / str(submission_id) / "episode-42-replay.json"
    )
    loaded = load_provenance_manifest(output, source_roots=[artifacts])
    loaded_submission = loaded.resolve_submission(submission_id)
    assert loaded_submission is not None
    assert loaded_submission.submission_label.available
    assert loaded_submission.submission_label.text == "final checkpoint 55400001"
    assert source_before == {path: path.read_bytes() for path in source_before}
    assert not list(tmp_path.glob(".inspector-catalog.json.*.tmp"))


def test_conflicting_checkpoint_claims_stay_unresolved_without_a_guess(
    tmp_path: Path,
) -> None:
    submission_id = 55400002
    replay_root = _cache(tmp_path, submission_id, 43)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    bundle = "sha256:" + "c" * 64
    _write_json(
        first,
        {
            "submission_id": submission_id,
            "checkpoint_checksum": digest_a,
            "bundle_checksum": bundle,
        },
    )
    _write_json(
        second,
        {
            "submission_id": submission_id,
            "checkpoint_checksum": digest_b,
            "bundle_checksum": bundle,
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(first),
        "--evidence",
        str(second),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    item = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    assert item["status"] == "unresolved"
    assert item["identity"]["checkpoint"]["checksum"] is None
    assert any(
        "conflicting checkpoint checksum" in reason
        for reason in item["unresolved_reasons"]
    )
    assert item["replay"]["game_count"] == 1


def test_safe_yaml_fallback_and_source_root_write_protection(tmp_path: Path) -> None:
    submission_id = 55400003
    replay_root = _cache(tmp_path, submission_id, 44)
    artifacts = tmp_path / "artifacts"
    checkpoint = artifacts / "weights.pt"
    bundle = artifacts / "packed.bin"
    artifacts.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    bundle.write_bytes(b"bundle")
    evidence = tmp_path / "specialists.yaml"
    evidence.write_text(
        "submissions:\n"
        f"  - submission_id: {submission_id}\n"
        f"    checkpoint_checksum: {_digest(checkpoint)}\n"
        f"    checkpoint_path: {checkpoint}\n"
        f"    bundle_checksum: {_digest(bundle)}\n"
        f"    bundle_path: {bundle}\n"
        "    label: safe yaml receipt\n",
        encoding="utf-8",
    )
    output = tmp_path / "yaml-catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(evidence),
        "--artifact-root",
        str(artifacts),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    item = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    assert item["status"] == "verified"
    assert item["identity"]["label"]["value"] == "safe yaml receipt"

    protected_output = replay_root / "must-not-write.json"
    rejected = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(evidence),
        "--artifact-root",
        str(artifacts),
        "--output",
        str(protected_output),
    )
    assert rejected.returncode != 0
    assert not protected_output.exists()


def test_builder_preserves_exact_submission_label_text(tmp_path: Path) -> None:
    submission_id = 55400004
    replay_root = _cache(tmp_path, submission_id, 45)
    evidence = tmp_path / "queue.json"
    exact_label = "  exact queue display text  "
    _write_json(
        evidence,
        {"queue": [{"submission_id": submission_id, "label": exact_label}]},
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(evidence),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    item = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    assert item["label"] == exact_label
    assert item["identity"]["label"]["value"] == exact_label


def test_shared_checkpoint_keeps_each_direct_submission_identity(
    tmp_path: Path,
) -> None:
    first_submission_id = 55400005
    second_submission_id = 55400006
    replay_root = _cache(tmp_path, first_submission_id, 46)
    _cache(tmp_path, second_submission_id, 47)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    checkpoint = artifacts / "checkpoint.bin"
    bundle = artifacts / "submission.bin"
    checkpoint.write_bytes(b"shared immutable checkpoint")
    bundle.write_bytes(b"shared immutable bundle")
    queue = tmp_path / "queue.json"
    _write_json(
        queue,
        {
            "queue": [
                {
                    "submission_id": first_submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "checkpoint_path": str(checkpoint),
                    "bundle_checksum": _digest(bundle),
                    "bundle_path": str(bundle),
                    "label": "direct label for first submission",
                    "specialist_id": "alakazam-first",
                },
                {
                    "submission_id": second_submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "checkpoint_path": str(checkpoint),
                    "bundle_checksum": _digest(bundle),
                    "bundle_path": str(bundle),
                    "label": "direct label for second submission",
                    "specialist_id": "alakazam-second",
                },
            ]
        },
    )
    # This metadata is legitimately checkpoint-linked to both submissions,
    # but it belongs to the shared artifact rather than either submission.
    _write_json(
        artifacts / "checkpoint-metadata.json",
        {
            "historical_package": {
                "source_checkpoint_checksum": _digest(checkpoint),
                "label": "shared artifact metadata label",
                "specialist_id": "shared-artifact-specialist",
            }
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(queue),
        "--artifact-root",
        str(artifacts),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    records = {
        item["submission_id"]: item
        for item in json.loads(output.read_text(encoding="utf-8"))["records"]
    }
    first = records[first_submission_id]
    second = records[second_submission_id]
    assert first["status"] == "verified"
    assert second["status"] == "verified"
    assert first["label"] == "direct label for first submission"
    assert second["label"] == "direct label for second submission"
    assert first["specialist_id"] == "alakazam-first"
    assert second["specialist_id"] == "alakazam-second"
    assert first["identity"]["label"]["evidence"] == [
        {
            "path": str(queue.resolve()),
            "sha256": _digest(queue),
            "pointer": "/queue/0",
            "role": "evidence",
            "key": "label",
        }
    ]
    assert second["identity"]["specialist_id"]["evidence"] == [
        {
            "path": str(queue.resolve()),
            "sha256": _digest(queue),
            "pointer": "/queue/1",
            "role": "evidence",
            "key": "specialist_id",
        }
    ]
    assert not any(
        "conflicting submission label" in reason
        for item in records.values()
        for reason in item["unresolved_reasons"]
    )


def test_shared_checkpoint_keeps_each_submission_package_identity(
    tmp_path: Path,
) -> None:
    first_submission_id = 55400015
    second_submission_id = 55400016
    replay_root = _cache(tmp_path, first_submission_id, 56)
    _cache(tmp_path, second_submission_id, 57)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    checkpoint = artifacts / "checkpoint.bin"
    first_bundle = artifacts / "first-submission.bin"
    second_bundle = artifacts / "second-submission.bin"
    checkpoint.write_bytes(b"shared model bytes")
    first_bundle.write_bytes(b"first exact package")
    second_bundle.write_bytes(b"second exact package")
    queue = tmp_path / "queue.json"
    _write_json(
        queue,
        {
            "queue": [
                {
                    "submission_id": first_submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "checkpoint_path": str(checkpoint),
                    "bundle_checksum": _digest(first_bundle),
                    "bundle_path": str(first_bundle),
                },
                {
                    "submission_id": second_submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "checkpoint_path": str(checkpoint),
                    "bundle_checksum": _digest(second_bundle),
                    "bundle_path": str(second_bundle),
                },
            ]
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(queue),
        "--artifact-root",
        str(artifacts),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    records = {
        item["submission_id"]: item
        for item in json.loads(output.read_text(encoding="utf-8"))["records"]
    }
    assert records[first_submission_id]["status"] == "verified"
    assert records[second_submission_id]["status"] == "verified"
    assert records[first_submission_id]["bundle"]["sha256"] == _digest(first_bundle)
    assert records[second_submission_id]["bundle"]["sha256"] == _digest(second_bundle)


def test_builder_emits_only_replay_and_explicit_evidence_submission_ids(
    tmp_path: Path,
) -> None:
    replay_submission_id = 55400007
    evidence_submission_id = 55400008
    nested_artifact_submission_id = 55400999
    replay_root = _cache(tmp_path, replay_submission_id, 48)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    checkpoint = artifacts / "checkpoint.bin"
    bundle = artifacts / "submission.bin"
    checkpoint.write_bytes(b"evidence-only checkpoint")
    bundle.write_bytes(b"evidence-only bundle")
    evidence = tmp_path / "queue.json"
    _write_json(
        evidence,
        {
            "queue": [
                {
                    "submission_id": evidence_submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "checkpoint_path": str(checkpoint),
                    "bundle_checksum": _digest(bundle),
                    "bundle_path": str(bundle),
                }
            ]
        },
    )
    # Artifact metadata may describe historical/embedded IDs, but that is not
    # authority to add a submission to the inspector catalogue.
    _write_json(
        artifacts / "historical-metadata.json",
        {
            "archive": {
                "nested_submission": {
                    "submission_id": nested_artifact_submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "bundle_checksum": _digest(bundle),
                }
            }
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(evidence),
        "--artifact-root",
        str(artifacts),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    emitted_ids = {
        item["submission_id"]
        for item in json.loads(output.read_text(encoding="utf-8"))["records"]
    }
    assert emitted_ids == {replay_submission_id, evidence_submission_id}
    assert nested_artifact_submission_id not in emitted_ids


def test_canonical_manifest_seeds_only_top_level_records_not_replay_agents(
    tmp_path: Path,
) -> None:
    owner_submission_ids = (55400009, 55400010, 55400011)
    nested_agent_ids = (55400901, 55400902, 55400903)
    replay_root = tmp_path / "cache"
    replay_root.mkdir()
    manifest = tmp_path / "live-provenance-manifest.json"
    _write_json(
        manifest,
        {
            "schema": "poke_bot.replay_model_inspector_provenance/v1",
            "version": 1,
            "records": [
                {
                    "submission_id": owner_submission_id,
                    "status": "unresolved",
                    "replay": {
                        "games": [
                            {
                                "episode_id": 100 + index,
                                "submission_id": owner_submission_id,
                                "own_agent": {
                                    "submission_id": owner_submission_id,
                                },
                                "agents": [
                                    {"submission_id": owner_submission_id},
                                    {"submission_id": nested_agent_ids[index]},
                                ],
                                "evidence_records": [
                                    {
                                        "submission_id": nested_agent_ids[index],
                                        "pointer": "/agents/1",
                                    }
                                ],
                            }
                        ]
                    },
                }
                for index, owner_submission_id in enumerate(owner_submission_ids)
            ],
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(manifest),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    emitted_ids = {
        item["submission_id"]
        for item in json.loads(output.read_text(encoding="utf-8"))["records"]
    }
    assert emitted_ids == set(owner_submission_ids)
    assert emitted_ids.isdisjoint(nested_agent_ids)


def test_generic_evidence_record_does_not_seed_nested_replay_agent_ids(
    tmp_path: Path,
) -> None:
    owner_submission_id = 55400012
    opponent_submission_id = 55400904
    replay_root = tmp_path / "cache"
    replay_root.mkdir()
    evidence = tmp_path / "queue.json"
    _write_json(
        evidence,
        {
            "queue": [
                {
                    "submission_id": owner_submission_id,
                    "label": "direct queue submission",
                    "replay": {
                        "games": [
                            {
                                "episode_id": 112,
                                "submission_id": owner_submission_id,
                                "own_agent": {
                                    "submission_id": owner_submission_id,
                                },
                                "agents": [
                                    {"submission_id": owner_submission_id},
                                    {"submission_id": opponent_submission_id},
                                ],
                            }
                        ]
                    },
                }
            ]
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(evidence),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert [
        item["submission_id"]
        for item in json.loads(output.read_text(encoding="utf-8"))["records"]
    ] == [owner_submission_id]


def test_declared_verified_artifact_path_beats_generic_digest_discovery(
    tmp_path: Path,
) -> None:
    submission_id = 55400013
    replay_root = _cache(tmp_path, submission_id, 49)
    generic_root = tmp_path / "a-generic-artifacts"
    declared_root = tmp_path / "z-declared-artifacts"
    generic_root.mkdir()
    declared_root.mkdir()
    generic_tree = generic_root / "matchup_tree.json"
    declared_tree = declared_root / "submitted-matchup-tree.json"
    generic_tree.write_bytes(b"same immutable matchup tree")
    declared_tree.write_bytes(b"same immutable matchup tree")
    _write_json(
        generic_root / "stale-catalogue-metadata.json",
        {
            "submission_id": submission_id,
            "matchup_tree_checksum": _digest(generic_tree),
            "matchup_tree_path": str(generic_tree),
        },
    )
    checkpoint = declared_root / "checkpoint.bin"
    bundle = declared_root / "bundle.bin"
    checkpoint.write_bytes(b"checkpoint for path preference test")
    bundle.write_bytes(b"bundle for path preference test")
    evidence = tmp_path / "queue.json"
    _write_json(
        evidence,
        {
            "queue": [
                {
                    "submission_id": submission_id,
                    "checkpoint_checksum": _digest(checkpoint),
                    "checkpoint_path": str(checkpoint),
                    "bundle_checksum": _digest(bundle),
                    "bundle_path": str(bundle),
                    "matchup_tree_checksum": _digest(declared_tree),
                    "matchup_tree_path": str(declared_tree),
                }
            ]
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(evidence),
        "--artifact-root",
        str(generic_root),
        "--artifact-root",
        str(declared_root),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    item = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    assert item["status"] == "verified"
    assert item["matchup_tree"] == {
        "path": str(declared_tree.resolve()),
        "sha256": _digest(declared_tree),
    }
    identity = item["identity"]["matchup_tree"]
    assert identity["preferred_path"] == str(declared_tree.resolve())
    assert identity["verified_paths"] == sorted(
        [str(generic_tree.resolve()), str(declared_tree.resolve())]
    )


def test_builder_does_not_parse_raw_replay_or_agent_log_payloads(
    tmp_path: Path,
) -> None:
    submission_id = 55362452
    replay_root = _cache(tmp_path, submission_id, 52)
    submission_root = replay_root / str(submission_id)
    # These payload files are artifacts referenced by episodes.json, not
    # metadata documents to ingest recursively.
    (submission_root / "episode-52-replay.json").write_text(
        "{raw replay payload", encoding="utf-8"
    )
    (submission_root / "episode-52-agent-0-logs.json").write_text(
        "{raw agent logs", encoding="utf-8"
    )
    evidence = tmp_path / "evidence.json"
    _write_json(
        evidence,
        {
            "submission_id": submission_id,
            "checkpoint_checksum": "sha256:" + "a" * 64,
            "bundle_checksum": "sha256:" + "b" * 64,
        },
    )
    output = tmp_path / "catalog.json"

    result = _run(
        "--replay-root",
        str(replay_root),
        "--evidence",
        str(evidence),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["replay_game_count"] == 1
    assert not any(
        "unreadable replay metadata" in warning
        for warning in payload["catalog_warnings"]
    )
