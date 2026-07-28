from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/promote_current_deck_guide_corpus.py"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source(root: Path) -> Path:
    specialist = "example-specialist"
    root.mkdir()
    rows = []
    dates = [f"2026-07-{day:02d}" for day in range(1, 21)]
    for date in dates:
        shard = root / f"{specialist}-{date}.features"
        shard.write_bytes(f"{date}\n".encode())
        rows.append(
            {
                "date": date,
                "records": 1,
                "decisions": 2,
                "guide_rows": 1,
                "sha256": _sha256(shard),
            }
        )
    manifest = root / "manifest.json"
    _json(
        manifest,
        {
            "totals": {
                "decisions_kept": 40,
                "target_coverage": {"guide_rows": 20},
            }
        },
    )
    pointer = root / "PROTECTED_EXPERT_CORPUS.json"
    _json(
        pointer,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": "manifest.json",
            "manifest_sha256": _sha256(manifest),
            "selection": {"value": specialist},
        },
    )
    _json(
        root / "CURRENT_DECK_GUIDE_CORPUS_READY.json",
        {
            "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
            "status": "ready",
            "specialist_id": specialist,
            "guide_version": "example-v1",
            "manifest_sha256": _sha256(manifest),
            "protected_pointer_sha256": _sha256(pointer),
            "decisions": 40,
            "guide_rows": 20,
            "days": 20,
            "dates": dates,
            "daily_shards": rows,
        },
    )
    return root


def test_promotion_is_checksum_gated_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source")
    destination = tmp_path / "corpora" / "example-specialist"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("ordinary corpus\n", encoding="utf-8")
    state = tmp_path / "state.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--specialist-id",
        "example-specialist",
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--state",
        str(state),
        "--bwlimit-kib",
        "100000",
    ]
    subprocess.run(command, check=True)
    first = json.loads(state.read_text())
    assert first["status"] == "ready"
    assert first["identity"]["guide_rows"] == 20
    assert Path(first["rollback_directory"], "old.txt").is_file()
    assert not (destination / "old.txt").exists()

    subprocess.run(command, check=True)
    second = json.loads(state.read_text())
    assert second["status"] == "ready"
    assert second["already_promoted"] is True


def test_promotion_rejects_a_tampered_landed_shard(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    destination = tmp_path / "corpora" / "example-specialist"
    destination.mkdir(parents=True)
    (source / "example-specialist-2026-07-20.features").write_text(
        "tampered\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--specialist-id",
            "example-specialist",
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--state",
            str(tmp_path / "state.json"),
            "--bwlimit-kib",
            "100000",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "guide feature checksum failed: 2026-07-20" in result.stderr
    assert not (destination / "PROTECTED_EXPERT_CORPUS.json").exists()
