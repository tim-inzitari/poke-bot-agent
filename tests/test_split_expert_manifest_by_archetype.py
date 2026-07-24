from __future__ import annotations

import json
from pathlib import Path

from poke_bot.feature_shards import (
    COMPACT_MODE,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    iter_feature_shard,
)
from poke_bot.pure_rl.expert_rehearsal import resolve_expert_manifest
from scripts.split_expert_manifest_by_archetype import split_manifest
from scripts.validate_specialist_corpora import validate_corpora
from tests.test_filter_feature_manifest import _write_source


def test_many_specialist_corpora_are_built_in_one_source_scan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "all.features"
    shard = _write_source(source)
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "date_start": "2026-07-18",
                "date_end": "2026-07-18",
                "dates": ["2026-07-18"],
                "max_context": 320,
                "shards": [shard],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "specialists"
    first = split_manifest(
        manifest,
        root,
        archetypes=("alakazam", "lucario", "starmie"),
        minimum_decisions=1,
    )
    second = split_manifest(
        manifest,
        root,
        archetypes=("alakazam", "lucario", "starmie"),
        minimum_decisions=1,
    )
    assert first == second
    ready = json.loads(first.read_text(encoding="utf-8"))
    by_id = {row["archetype"]: row for row in ready["results"]}
    assert by_id["alakazam"]["records"] == 2
    assert by_id["lucario"]["records"] == 1
    assert by_id["starmie"]["status"] == "unavailable"
    validation = validate_corpora(
        first,
        required_archetypes=("alakazam", "lucario", "starmie"),
        required_compact_mode=COMPACT_MODE,
        required_target_coverage=(),
    )
    assert validation["counts"] == {
        "ready": 2,
        "insufficient_decisions": 0,
        "unavailable": 1,
    }
    for target, expected in (("alakazam", 2), ("lucario", 1)):
        assert (
            by_id[target]["protected_corpus"]
            == f"{target}/PROTECTED_EXPERT_CORPUS.json"
        )
        pointer = root / target / "PROTECTED_EXPERT_CORPUS.json"
        identity = resolve_expert_manifest(pointer, min_decisions=1)
        assert identity.records == expected
        manifest_payload = json.loads(
            (root / target / "manifest.json").read_text(encoding="utf-8")
        )
        rows = list(
            iter_feature_shard(
                root / target / manifest_payload["shards"][0]["path"]
            )
        )
        assert len(rows) == expected
        assert all(row.archetype.casefold() == target for row in rows)


def test_nonempty_corpus_below_bootstrap_floor_is_not_marked_ready(
    tmp_path: Path,
) -> None:
    source = tmp_path / "all.features"
    shard = _write_source(source)
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "date_start": "2026-07-18",
                "date_end": "2026-07-18",
                "dates": ["2026-07-18"],
                "max_context": 320,
                "shards": [shard],
            }
        ),
        encoding="utf-8",
    )
    ready_path = split_manifest(
        manifest,
        tmp_path / "specialists",
        archetypes=("lucario",),
        minimum_decisions=2,
    )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    assert len(ready["results"]) == 1
    result = ready["results"][0]
    assert result["archetype"] == "lucario"
    assert result["status"] == "insufficient_decisions"
    assert result["records"] == 1
    assert result["decisions"] == 1
    assert result["minimum_decisions"] == 2
    assert result["protected_corpus"] == (
        "lucario/PROTECTED_EXPERT_CORPUS.json"
    )
    assert result["manifest_sha256"].startswith("sha256:")
