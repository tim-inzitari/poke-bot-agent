from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from poke_bot.feature_shards import (
    COMPACT_MODE,
    MANIFEST_FORMAT,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)


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
