from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from poke_bot.feature_shards import (
    COMPACT_MODE,
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)
from poke_bot.strategic_heads import (
    EXPANDED_STRATEGIC_SCHEMA,
    TARGET_SCHEMA_DIGEST,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER = ROOT / "scripts/assemble_feature_manifest.py"


def _write_shard(staging: Path, day: str) -> None:
    shard = staging / f"top_ladder_all_{day}.features"
    shard.write_bytes((f"compact-{day}-" * 31).encode())
    digest = "sha256:" + hashlib.sha256(shard.read_bytes()).hexdigest()
    sidecar = shard.with_suffix(shard.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "compact_mode": COMPACT_MODE,
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": digest,
                "source_dates": [day],
                "stats": {
                    "records_total": 100,
                    "records_kept": 99,
                    "decisions_kept": 500,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _assemble(staging: Path, out: Path, dates: list[str], *extra: str) -> None:
    command = [
        sys.executable,
        str(ASSEMBLER),
        "--staging-dir",
        str(staging),
        "--out",
        str(out),
        "--min-free-gib",
        "0",
    ]
    for day in dates:
        command.extend(["--expected-date", day])
    command.extend(extra)
    subprocess.run(command, check=True, capture_output=True, text=True)


def test_per_day_verification_cache_feeds_final_manifest(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    verified = staging / ".verified"
    fragments = staging / "partial-manifests"
    staging.mkdir()
    days = ["2026-07-09", "2026-07-10"]

    for day in days:
        _write_shard(staging, day)
        _assemble(
            staging,
            fragments / f"{day}.json",
            [day],
            "--only-date",
            day,
            "--verified-dir",
            str(verified),
        )

    assert len(list(verified.glob("*.verified.json"))) == 2
    final = staging / "manifest.json"
    _assemble(staging, final, days, "--verified-dir", str(verified))
    payload = json.loads(final.read_text(encoding="utf-8"))
    assert payload["format"] == MANIFEST_FORMAT
    assert payload["dates"] == days
    assert len(payload["shards"]) == 2
    assert payload["totals"]["records_kept"] == 198
    assert payload["totals"]["decisions_kept"] == 1000
    expanded = payload["expanded_strategic_targets"]
    assert expanded["schema"] == EXPANDED_STRATEGIC_SCHEMA
    assert expanded["digest"] == TARGET_SCHEMA_DIGEST
    assert expanded["decisions"] == 1000
    assert set(expanded["head_coverage"]) == set(EXPANDED_HEAD_IDS)
    assert all(
        row == {
            "labeled_rows": 0,
            "masked_rows": 1000,
            "total_rows": 1000,
        }
        for row in expanded["head_coverage"].values()
    )


def test_expanded_target_coverage_is_validated_and_aggregated(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    days = ["2026-07-21", "2026-07-22"]
    for day in days:
        _write_shard(staging, day)
        sidecar = staging / f"top_ladder_all_{day}.features.json"
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["stats"]["expanded_strategic_targets"] = {
            "schema": EXPANDED_STRATEGIC_SCHEMA,
            "digest": TARGET_SCHEMA_DIGEST,
            "decisions": 500,
            "head_coverage": {
                head_id: {
                    "labeled_rows": 400,
                    "masked_rows": 100,
                    "total_rows": 500,
                }
                for head_id in EXPANDED_HEAD_IDS
            },
        }
        sidecar.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    manifest = staging / "manifest.json"
    _assemble(staging, manifest, days)
    expanded = json.loads(manifest.read_text(encoding="utf-8"))[
        "expanded_strategic_targets"
    ]
    assert expanded == {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "digest": TARGET_SCHEMA_DIGEST,
        "decisions": 1000,
        "head_coverage": {
            head_id: {
                "labeled_rows": 800,
                "masked_rows": 200,
                "total_rows": 1000,
            }
            for head_id in EXPANDED_HEAD_IDS
        },
    }

    sidecar = staging / f"top_ladder_all_{days[0]}.features.json"
    malformed = json.loads(sidecar.read_text(encoding="utf-8"))
    malformed["stats"]["expanded_strategic_targets"]["head_coverage"][
        "action_q"
    ]["masked_rows"] = 99
    sidecar.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        _assemble(staging, tmp_path / "invalid.json", days)


def test_authoritative_manifest_is_protected_only_with_complete_targets(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "authoritative"
    staging.mkdir()
    day = "2026-07-20"
    _write_shard(staging, day)
    sidecar = next(staging.glob("*.features.json"))
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.update(
        {
            "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
            "required_archetype": "alakazam",
            "max_context": 320,
        }
    )
    decisions = int(metadata["stats"]["decisions_kept"])
    required_targets = (
        "temporal_action_rows",
        "opponent_hand_rows",
        "opponent_remainder_rows",
        "opponent_private_prize_rows",
        "lethal_threat_rows",
        "prize_race_rows",
    )
    metadata["stats"]["target_coverage"] = {
        name: decisions for name in required_targets
    }
    sidecar.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    manifest = staging / "manifest.json"
    extra = [
        "--compact-mode",
        COMPACT_MODE_TEMPORAL_EXPERT,
        "--required-archetype",
        "alakazam",
        "--expected-max-context",
        "320",
        "--seal-protected",
    ]
    for target in required_targets:
        extra.extend(["--require-target-coverage", target])
    _assemble(staging, manifest, [day], *extra)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    pointer = json.loads(
        (staging / "PROTECTED_EXPERT_CORPUS.json").read_text(encoding="utf-8")
    )
    assert payload["selection"]["value"] == "alakazam"
    assert payload["max_context"] == 320
    assert payload["quality_gates"]["required_target_rows_complete"] is True
    assert pointer["protected"] is True
    assert pointer["manifest"] == "manifest.json"
    assert pointer["manifest_sha256"] == "sha256:" + hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    _write_shard(incomplete, day)
    bad_sidecar = next(incomplete.glob("*.features.json"))
    bad = json.loads(bad_sidecar.read_text(encoding="utf-8"))
    bad.update(
        {
            "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
            "required_archetype": "alakazam",
            "max_context": 320,
        }
    )
    bad["stats"]["target_coverage"] = {
        name: decisions - int(name == "opponent_hand_rows")
        for name in required_targets
    }
    bad_sidecar.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        _assemble(incomplete, incomplete / "manifest.json", [day], *extra)
    assert not (incomplete / "PROTECTED_EXPERT_CORPUS.json").exists()
