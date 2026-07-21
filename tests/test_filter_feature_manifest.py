from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION, GameSequence
from poke_bot.feature_shards import (
    COMPACT_MODE,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)
from poke_bot.features import FEATURE_SCHEMA_VERSION
from poke_bot.pure_rl.expert_rehearsal import resolve_expert_manifest
from scripts.filter_feature_manifest import (
    PINNED_CORPUS_NAME,
    filter_manifest,
    seal_filtered_manifest,
    sha256,
)
from scripts.run_alakazam_expert_bootstrap import resolve_filtered_manifest


def _sequence(episode: str, seat: int, archetype: str) -> GameSequence:
    return GameSequence(
        episode_id=episode,
        seat=seat,
        archetype=archetype,
        opp_archetype="other",
        deck=list(range(60)),
        value=1.0,
        decisions=[],
    )


def _write_source(path: Path) -> dict:
    rows = [
        _sequence("a", 0, "alakazam"),
        _sequence("a", 1, "lucario"),
        _sequence("b", 1, "AlAkAzAm"),
    ]
    # Give selected rows at least one decision-equivalent item for the filter's
    # nonempty-decision gate without constructing full feature tensors.
    for row in rows:
        row.decisions.append(object())  # pickleable and never inspected here
    stats = {
        "records_total": 3,
        "records_kept": 3,
        "records_dropped": 0,
        "decisions_kept": 3,
    }
    with path.open("wb") as stream:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "compact_mode": COMPACT_MODE,
                "source_dates": ["2026-07-18"],
                "max_context": 320,
            },
            stream,
        )
        for row in rows:
            pickle.dump(row, stream)
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            },
            stream,
        )
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "source_dates": ["2026-07-18"],
        "stats": stats,
    }


def test_filter_keeps_only_the_acting_alakazam_seats(tmp_path: Path) -> None:
    source = tmp_path / "all.features"
    shard = _write_source(source)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "date_start": "2026-07-18",
                "date_end": "2026-07-18",
                "dates": ["2026-07-18"],
                "shards": [shard],
            }
        )
    )
    manifest_path = filter_manifest(
        source_manifest,
        tmp_path / "alakazam",
        archetype="alakazam",
    )
    manifest = json.loads(manifest_path.read_text())
    selected_path = manifest_path.parent / manifest["shards"][0]["path"]
    selected = list(iter_feature_shard(selected_path))

    assert [(row.episode_id, row.seat) for row in selected] == [("a", 0), ("b", 1)]
    assert all(row.archetype.lower() == "alakazam" for row in selected)
    assert manifest["selection"]["seat_semantics"] == "acting_seat_only"
    assert manifest["totals"]["records_kept"] == 2
    assert manifest["shards"][0]["stats"]["source_records_scanned"] == 3
    pointer = json.loads((manifest_path.parent / PINNED_CORPUS_NAME).read_text())
    assert pointer["protected"] is True
    assert pointer["manifest"] == "manifest.json"
    assert pointer["manifest_sha256"] == sha256(manifest_path)
    assert pointer["selection"]["value"] == "alakazam"
    identity = resolve_expert_manifest(
        manifest_path.parent / PINNED_CORPUS_NAME, min_decisions=1
    )
    assert identity.path == str(manifest_path.resolve())
    assert identity.decisions == 2
    resolved_path, resolved = resolve_filtered_manifest(
        manifest_path.parent / PINNED_CORPUS_NAME, min_decisions=1
    )
    assert resolved_path == manifest_path.resolve()
    assert resolved["selection"]["seat_semantics"] == "acting_seat_only"


def test_filter_is_idempotent_only_for_identical_source_and_selection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "all.features"
    shard = _write_source(source)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "shards": [shard],
            }
        )
    )
    first = filter_manifest(source_manifest, tmp_path / "out", archetype="alakazam")
    second = filter_manifest(source_manifest, tmp_path / "out", archetype="alakazam")
    assert first == second
    assert seal_filtered_manifest(second) == second.parent / PINNED_CORPUS_NAME

    original = second.read_text()
    second.write_text(original + "\n")
    with pytest.raises(ValueError, match="pointer digest mismatch"):
        resolve_expert_manifest(second.parent / PINNED_CORPUS_NAME, min_decisions=1)
