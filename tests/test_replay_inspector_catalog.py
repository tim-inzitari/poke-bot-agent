from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from replay_inspector.catalog import scan_archive
from replay_inspector.provenance import (
    PROVENANCE_SCHEMA,
    ProvenanceError,
    load_provenance_manifest,
    sha256_source_tree,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def _artifact_entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _digest(path)}


def _write_manifest(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    canonical_rows = [{"status": "verified", **row} for row in rows]
    _write(
        path,
        json.dumps(
            {
                "schema": PROVENANCE_SCHEMA,
                "version": 1,
                "records": canonical_rows,
            }
        ),
    )
    return path


def _write_archive(
    root: Path,
    submission_id: int,
    episode_ids: list[int],
) -> Path:
    directory = root / str(submission_id)
    directory.mkdir(parents=True)
    episodes = []
    for episode_id in episode_ids:
        _write(
            directory / f"episode-{episode_id}-replay.json",
            json.dumps({"id": episode_id, "steps": []}),
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "submission_id": submission_id,
                "own_agent": {"index": 0, "submission_id": submission_id},
            }
        )
    _write(
        directory / "episodes.json",
        json.dumps(
            {
                "submission_id": submission_id,
                "episode_count": len(episodes),
                "episodes": episodes,
            }
        ),
    )
    return directory


def test_catalog_joins_verified_archive_and_submission_provenance(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    _write_archive(archive_root, 55300685, [1001, 1002])
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "checkpoints" / "model.pt", b"checkpoint")
    bundle = _write(assets / "bundles" / "submission.tar.gz", b"bundle")
    tree = _write(assets / "routes" / "matchup_tree.json", "{}")
    manifest = _write_manifest(
        tmp_path / "control" / "provenance.json",
        [
            {
                "submission_id": 55300685,
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
                "replay": {
                    "games": [
                        {
                            "episode_id": episode_id,
                            "replay_sha256": _digest(
                                archive_root
                                / "55300685"
                                / f"episode-{episode_id}-replay.json"
                            ),
                        }
                        for episode_id in (1001, 1002)
                    ]
                },
            }
        ],
    )

    provenance = load_provenance_manifest(manifest, source_roots=[assets])
    catalog = scan_archive(archive_root, provenance=provenance)

    assert catalog.available
    assert [row.submission_id for row in catalog.submissions] == [55300685]
    submission = catalog.submission(55300685)
    assert submission is not None and submission.available
    assert submission.provenance is not None
    assert submission.provenance.available
    assert submission.replay_available
    assert submission.model_analysis_available
    assert submission.model_analysis_provenance is submission.provenance
    # Missing presentation provenance is deliberately narrower than missing
    # model provenance: label display fails closed without hiding a fully
    # checksum-verified replay/model mapping.
    assert not submission.submission_label.available
    assert (
        "submission_label_not_declared"
        in submission.submission_label.availability_reasons
    )
    assert submission.provenance.checkpoint.actual_sha256 == _digest(checkpoint)
    # These indexed games have no direct own-agent reward.  The outcome
    # summary remains visible but cannot manufacture a zero/win/loss result.
    assert not submission.cached_replay_outcomes.available
    assert submission.cached_replay_outcomes.games_total == 2
    assert submission.cached_replay_outcomes.games_with_outcome == 0
    assert submission.cached_replay_outcomes.win_rate is None
    assert (
        "cached_replay_outcomes_no_source_backed_own_agent_rewards"
        in submission.cached_replay_outcomes.availability_reasons
    )
    replay = catalog.replay(55300685, 1001)
    assert replay is not None and replay.available
    assert replay.replay_available
    assert replay.model_analysis_available
    assert replay.metadata is not None
    assert replay.metadata["own_agent"]["index"] == 0
    payload = catalog.to_dict()
    assert payload["submissions"][0]["provenance"]["bundle"]["available"] is True
    assert payload["submissions"][0]["label"] is None
    assert payload["submissions"][0]["cached_replay_outcomes"]["wins"] == 0


def test_runtime_trace_requires_checksum_bound_package_and_parity_receipt(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "model.pt", b"checkpoint")
    bundle = _write(assets / "submission.tar.gz", b"bundle")
    tree = _write(assets / "matchup_tree.json", "{}")
    runtime_package = _write(assets / "submitted-runtime.tar.gz", b"runtime")
    runtime_tree = assets / "extracted" / "poke_bot"
    _write(runtime_tree / "__init__.py", "VERSION = 1\n")
    receipt = _write(
        assets / "runtime-parity.json",
        json.dumps(
            {
                "schema": "poke_bot.replay_model_inspector_runtime_parity_receipt/v1",
                "version": 1,
                "status": "verified",
                "submission_id": 87,
                "checkpoint_sha256": _digest(checkpoint),
                "bundle_sha256": _digest(bundle),
                "runtime_package_sha256": _digest(runtime_package),
                "runtime_source_tree_sha256": sha256_source_tree(runtime_tree),
                "verification": {
                    "method": "independent_exact_runtime_parity",
                    "verified_by": "test-fixture",
                    "verified_at_utc": "2026-08-07T00:00:00Z",
                },
            }
        ),
    )
    manifest = _write_manifest(
        tmp_path / "control" / "provenance.json",
        [
            {
                "submission_id": 87,
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
                "runtime_package": _artifact_entry(runtime_package),
                "runtime_parity_receipt": _artifact_entry(receipt),
            }
        ],
    )

    entry = load_provenance_manifest(
        manifest, source_roots=[assets]
    ).resolve_submission(87)

    assert entry is not None and entry.available
    assert entry.trace_available
    assert entry.runtime_parity_receipt is not None
    assert entry.runtime_parity_receipt.available


def test_checkpoint_parameters_remain_available_without_runtime_parity_receipt(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "model.pt", b"checkpoint")
    bundle = _write(assets / "submission.tar.gz", b"bundle")
    tree = _write(assets / "matchup_tree.json", "{}")
    manifest = _write_manifest(
        tmp_path / "control" / "provenance.json",
        [
            {
                "submission_id": 88,
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
            }
        ],
    )

    entry = load_provenance_manifest(
        manifest, source_roots=[assets]
    ).resolve_submission(88)

    assert entry is not None and entry.available
    assert not entry.trace_available
    assert "runtime_package_artifact_missing" in entry.trace_availability_reasons


def test_digest_mismatch_ambiguity_and_unresolved_submissions_remain_listable(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    _write_archive(archive_root, 10, [1])
    _write_archive(archive_root, 11, [2])
    _write_archive(archive_root, 12, [3])
    _write_archive(archive_root, 13, [4])
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "model.pt", b"checkpoint")
    bundle = _write(assets / "submission.tar.gz", b"bundle")
    tree = _write(assets / "matchup_tree.json", "{}")
    bad_checkpoint = {"path": str(checkpoint), "sha256": "sha256:" + "0" * 64}
    manifest = _write_manifest(
        tmp_path / "provenance.json",
        [
            {
                "submission_id": 10,
                "checkpoint": bad_checkpoint,
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
            },
            {
                "submission_id": 10,
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
            },
            {
                "submission_id": 12,
                "checkpoint": bad_checkpoint,
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
            },
            {
                "submission_id": 13,
                "status": "unresolved",
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
            },
        ],
    )

    provenance = load_provenance_manifest(manifest, source_roots=[assets])
    catalog = scan_archive(archive_root, provenance=provenance)

    ambiguous = catalog.submission(10)
    # The replay still belongs to the locally readable archive.  Only model
    # analysis is fail-closed because two candidate bundles exist.
    assert ambiguous is not None and ambiguous.replay_available
    assert not ambiguous.model_analysis_available
    assert "provenance_ambiguous" in ambiguous.model_analysis_availability_reasons
    assert len(ambiguous.provenance_candidates) == 2
    assert any(
        "checkpoint_sha256_mismatch" in reason
        for row in ambiguous.provenance_candidates
        for reason in row.availability_reasons
    )
    ambiguous_replay = ambiguous.replay(1)
    assert ambiguous_replay is not None and ambiguous_replay.replay_available
    assert not ambiguous_replay.model_analysis_available
    assert (
        "provenance_ambiguous" in ambiguous_replay.model_analysis_availability_reasons
    )

    unresolved = catalog.submission(11)
    assert unresolved is not None and unresolved.replay_available
    assert not unresolved.model_analysis_available
    assert "provenance_unresolved" in unresolved.model_analysis_availability_reasons
    unresolved_replay = unresolved.replay(2)
    assert unresolved_replay is not None and unresolved_replay.replay_available
    assert not unresolved_replay.model_analysis_available

    digest_mismatch = catalog.submission(12)
    assert digest_mismatch is not None and digest_mismatch.replay_available
    assert not digest_mismatch.model_analysis_available
    assert (
        "provenance_unavailable" in digest_mismatch.model_analysis_availability_reasons
    )
    assert (
        "provenance:checkpoint_sha256_mismatch"
        in digest_mismatch.model_analysis_availability_reasons
    )

    status_unresolved = catalog.submission(13)
    assert status_unresolved is not None and status_unresolved.replay_available
    assert not status_unresolved.model_analysis_available
    assert (
        "provenance:provenance_status_not_verified"
        in status_unresolved.model_analysis_availability_reasons
    )


def test_configured_root_and_symlink_escapes_fail_closed_without_hiding_rows(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    _write_archive(archive_root, 22, [3])
    assets = tmp_path / "assets"
    assets.mkdir()
    outside = tmp_path / "outside"
    checkpoint = _write(outside / "model.pt", b"outside checkpoint")
    bundle = _write(assets / "bundle.tar.gz", b"bundle")
    tree = _write(assets / "tree.json", "{}")
    escaped = assets / "escaped-model.pt"
    escaped.symlink_to(checkpoint)
    manifest = _write_manifest(
        tmp_path / "provenance.json",
        [
            {
                "submission_id": 22,
                "checkpoint": {"path": str(escaped), "sha256": _digest(checkpoint)},
                "bundle": _artifact_entry(bundle),
                "matchup_tree": _artifact_entry(tree),
            }
        ],
    )
    provenance = load_provenance_manifest(manifest, source_roots=[assets])
    record = provenance.candidates_for_submission(22)[0]
    assert not record.available
    assert "checkpoint_path_outside_configured_roots" in record.availability_reasons

    outside_archive = tmp_path / "outside-archive"
    _write_archive(outside_archive, 99, [9])
    (archive_root / "99").symlink_to(outside_archive / "99", target_is_directory=True)
    catalog = scan_archive(archive_root, provenance=provenance)
    escaped_submission = catalog.submission(99)
    assert escaped_submission is not None
    assert not escaped_submission.available
    assert (
        "archive_submission_directory_outside_configured_root"
        in escaped_submission.availability_reasons
    )


def test_metadata_only_episode_and_bad_manifest_id_block_analysis_not_browsing(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    directory = _write_archive(archive_root, 30, [4])
    _write(
        directory / "episodes.json",
        json.dumps(
            {
                "submission_id": 31,
                "episode_count": 2,
                "episodes": [
                    {"episode_id": 4},
                    {"episode_id": 5},
                ],
            }
        ),
    )
    catalog = scan_archive(archive_root)
    submission = catalog.submission(30)
    assert submission is not None
    assert "episodes_manifest_submission_id_mismatch" in submission.availability_reasons
    replay = submission.replay(4)
    assert replay is not None and replay.replay_available
    assert not replay.model_analysis_available
    assert (
        "episodes_manifest_submission_id_mismatch"
        in replay.model_analysis_availability_reasons
    )
    missing_replay = submission.replay(5)
    assert missing_replay is not None
    assert "replay_file_missing" in missing_replay.availability_reasons
    # No configured provenance is also visible rather than accidentally
    # selecting a same-named checkpoint.
    assert (
        "provenance_not_configured"
        in missing_replay.model_analysis_availability_reasons
    )


def test_invalid_replay_json_is_listed_but_not_browseable_or_analyzable(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    directory = _write_archive(archive_root, 40, [7])
    _write(directory / "episode-7-replay.json", "not json")

    catalog = scan_archive(archive_root)
    replay = catalog.replay(40, 7)

    assert replay is not None
    assert not replay.replay_available
    assert not replay.model_analysis_available
    assert "replay_file_invalid_json" in replay.replay_availability_reasons
    assert "provenance_not_configured" in replay.model_analysis_availability_reasons


def test_manifest_requires_v1_status_and_accepts_only_explicit_null_tree(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "model.pt", b"checkpoint")
    bundle = _write(assets / "submission.tar.gz", b"bundle")
    missing_version = _write(
        tmp_path / "missing-version.json",
        json.dumps({"schema": PROVENANCE_SCHEMA, "records": []}),
    )

    with pytest.raises(ProvenanceError, match="version"):
        load_provenance_manifest(missing_version, source_roots=[assets])

    manifest = _write_manifest(
        tmp_path / "provenance.json",
        [
            {
                "submission_id": 50,
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "runtime": {
                    "matchup_tree_path": None,
                    "matchup_tree_sha256": None,
                },
            },
            {
                "submission_id": 51,
                "status": "",
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "runtime": {
                    "matchup_tree_path": None,
                    "matchup_tree_sha256": None,
                },
            },
        ],
    )

    provenance = load_provenance_manifest(manifest, source_roots=[assets])
    no_tree = provenance.resolve_submission(50)
    assert no_tree is not None and no_tree.matchup_tree is None
    assert no_tree.available

    missing_status = provenance.candidates_for_submission(51)[0]
    assert not missing_status.available
    assert "provenance_status_missing_or_invalid" in missing_status.availability_reasons


def test_catalog_exposes_only_checksum_bound_submission_label_and_two_players(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    directory = _write_archive(archive_root, 61, [9])
    _write(
        directory / "episodes.json",
        json.dumps(
            {
                "submission_id": 61,
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_id": 9,
                        "submission_id": 61,
                        "agents": [
                            {
                                "index": 0,
                                "submission_id": 61,
                                "team_name": "Ada exact",
                                "rank": 4,
                                "reward": 1,
                            },
                            {
                                "index": 1,
                                "submission_id": 700,
                                "team_name": "Bert exact",
                                "rank": 19,
                                "reward": -1,
                            },
                        ],
                        "own_agent": {
                            "index": 0,
                            "submission_id": 61,
                            "team_name": "Ada exact",
                            "rank": 4,
                            "reward": 1,
                        },
                    }
                ],
            }
        ),
    )
    replay_path = directory / "episode-9-replay.json"
    _write(
        replay_path,
        json.dumps(
            {
                "steps": [],
                "info": {
                    "TeamNames": ["Ada exact", "Bert exact"],
                    "TeamRanks": [4, 19],
                    "Agents": [
                        {"Name": "Ada exact", "Rank": 4},
                        {"Name": "Bert exact", "Rank": 19},
                    ],
                },
            }
        ),
    )
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "model.pt", b"checkpoint")
    bundle = _write(assets / "submission.tar.gz", b"bundle")
    exact_label = "final checkpoint 61 — queue receipt"
    manifest = _write_manifest(
        tmp_path / "provenance.json",
        [
            {
                "submission_id": 61,
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "runtime": {
                    "matchup_tree_path": None,
                    "matchup_tree_sha256": None,
                },
                "label": exact_label,
                "identity": {
                    "label": {
                        "value": exact_label,
                        "availability": "available",
                        "evidence": [
                            {
                                "path": "/receipts/kaggle-submission-queue.json",
                                "sha256": "sha256:" + "a" * 64,
                                "pointer": "/queue/30/label",
                                "role": "submission_queue_receipt",
                                "key": "label",
                            }
                        ],
                    }
                },
                "replay": {
                    "games": [
                        {
                            "episode_id": 9,
                            "replay_sha256": _digest(replay_path),
                        }
                    ]
                },
            }
        ],
    )

    provenance = load_provenance_manifest(manifest, source_roots=[assets])
    catalog = scan_archive(archive_root, provenance=provenance)
    submission = catalog.submission(61)
    replay = catalog.replay(61, 9)

    assert submission is not None
    assert submission.submission_label.available
    assert submission.submission_label.text == exact_label
    assert submission.to_dict()["label"] == exact_label
    assert submission.to_dict()["submission_label"]["evidence"] == [
        {
            "path": "/receipts/kaggle-submission-queue.json",
            "sha256": "sha256:" + "a" * 64,
            "pointer": "/queue/30/label",
            "role": "submission_queue_receipt",
            "key": "label",
        }
    ]
    assert replay is not None
    assert [player.to_dict() for player in replay.players] == [
        {
            "seat": 0,
            "name": "Ada exact",
            "name_available": True,
            "name_availability_reasons": [],
            "name_sources": [
                "archived_episode_metadata.agents.team_name",
                "archived_episode_metadata.own_agent.team_name",
                "authoritative_replay_metadata.info.TeamNames",
                "authoritative_replay_metadata.info.Agents.Name",
            ],
            "rank": 4,
            "rank_available": True,
            "rank_availability_reasons": [],
            "rank_sources": [
                "archived_episode_metadata.agents.rank",
                "archived_episode_metadata.own_agent.rank",
                "authoritative_replay_metadata.info.TeamRanks",
                "authoritative_replay_metadata.info.Agents.Rank",
            ],
        },
        {
            "seat": 1,
            "name": "Bert exact",
            "name_available": True,
            "name_availability_reasons": [],
            "name_sources": [
                "archived_episode_metadata.agents.team_name",
                "authoritative_replay_metadata.info.TeamNames",
                "authoritative_replay_metadata.info.Agents.Name",
            ],
            "rank": 19,
            "rank_available": True,
            "rank_availability_reasons": [],
            "rank_sources": [
                "archived_episode_metadata.agents.rank",
                "authoritative_replay_metadata.info.TeamRanks",
                "authoritative_replay_metadata.info.Agents.Rank",
            ],
        },
    ]
    assert replay.metadata is not None
    assert replay.metadata["agents"][1]["reward"] == -1


def test_catalog_fails_closed_for_unbound_label_and_never_infers_rank(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    directory = _write_archive(archive_root, 62, [10])
    _write(
        directory / "episodes.json",
        json.dumps(
            {
                "submission_id": 62,
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_id": 10,
                        "agents": [
                            {
                                "index": 0,
                                "team_name": "Known own player",
                                "reward": 100,
                            },
                            {
                                "index": 1,
                                "team_name": "Known opponent",
                                "reward": -100,
                            },
                        ],
                    }
                ],
            }
        ),
    )
    replay_path = directory / "episode-10-replay.json"
    _write(
        replay_path,
        json.dumps(
            {
                "steps": [],
                "info": {"TeamNames": ["Known own player", "Known opponent"]},
            }
        ),
    )
    assets = tmp_path / "assets"
    checkpoint = _write(assets / "model.pt", b"checkpoint")
    bundle = _write(assets / "submission.tar.gz", b"bundle")
    manifest = _write_manifest(
        tmp_path / "provenance.json",
        [
            {
                "submission_id": 62,
                "checkpoint": _artifact_entry(checkpoint),
                "bundle": _artifact_entry(bundle),
                "runtime": {
                    "matchup_tree_path": None,
                    "matchup_tree_sha256": None,
                },
                # A display string without its checksum-bearing source record
                # is intentionally not a label the UI may show.
                "label": "do not trust this bare label",
                "replay": {
                    "games": [
                        {
                            "episode_id": 10,
                            "replay_sha256": _digest(replay_path),
                        }
                    ]
                },
            }
        ],
    )

    provenance = load_provenance_manifest(manifest, source_roots=[assets])
    catalog = scan_archive(archive_root, provenance=provenance)
    submission = catalog.submission(62)
    replay = catalog.replay(62, 10)

    assert submission is not None
    assert not submission.submission_label.available
    assert submission.to_dict()["label"] is None
    assert (
        "submission_label_identity_missing"
        in submission.submission_label.availability_reasons
    )
    assert replay is not None
    assert [player.name for player in replay.players] == [
        "Known own player",
        "Known opponent",
    ]
    assert [player.rank for player in replay.players] == [None, None]
    assert not any(player.rank_available for player in replay.players)
    assert all(
        player.rank_availability_reasons
        == ("player_rank_unavailable_not_supplied_by_archived_metadata_or_replay",)
        for player in replay.players
    )


def test_cached_outcomes_use_only_direct_own_agent_rewards(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    directory = _write_archive(archive_root, 63, [1, 2, 3, 4, 5, 6])
    _write(
        directory / "episodes.json",
        json.dumps(
            {
                "submission_id": 63,
                "episode_count": 6,
                "episodes": [
                    {"episode_id": 1, "own_agent": {"index": 0, "reward": 5}},
                    {"episode_id": 2, "own_agent": {"index": 0, "reward": 0}},
                    {"episode_id": 3, "own_agent": {"index": 0, "reward": -2.5}},
                    {"episode_id": 4, "own_agent": {"index": 0}},
                    {
                        "episode_id": 5,
                        # An agent-row reward is not an own-agent receipt.
                        # The summary must not select it by submission id.
                        "agents": [{"index": 0, "submission_id": 63, "reward": 999}],
                    },
                    {
                        "episode_id": 6,
                        # Numeric-looking text is still not a direct numeric
                        # reward observation.
                        "own_agent": {"index": 0, "reward": "9"},
                    },
                ],
            }
        ),
    )

    submission = scan_archive(archive_root).submission(63)

    assert submission is not None
    summary = submission.cached_replay_outcomes
    assert summary.available
    assert not summary.complete
    assert summary.games_total == 6
    assert summary.games_with_outcome == 3
    assert summary.games_without_outcome == 3
    assert summary.wins == 1
    assert summary.nonwins == 2
    assert summary.win_rate == pytest.approx(1 / 3)
    assert (
        "cached_replay_outcomes_partial_source_coverage" in summary.availability_reasons
    )
    assert (
        "cached_replay_outcome_own_agent_reward_missing" in summary.availability_reasons
    )
    assert (
        "cached_replay_outcome_own_agent_missing_or_invalid"
        in summary.availability_reasons
    )
    assert (
        "cached_replay_outcome_own_agent_reward_invalid" in summary.availability_reasons
    )
    payload = summary.to_dict()
    assert payload["draws"] is None
    assert payload["draws_available"] is False
    assert payload["nonwins_definition"] == "source_backed_own_agent_reward_lte_zero"
    assert payload["source"] == "archived_episode_metadata.own_agent.reward"
