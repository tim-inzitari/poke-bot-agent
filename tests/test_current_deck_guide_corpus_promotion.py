from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/promote_current_deck_guide_corpus.py"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source(root: Path, *, days: int = 20) -> Path:
    specialist = "example-specialist"
    root.mkdir()
    rows = []
    start = date(2026, 6, 26)
    dates = [
        (start + timedelta(days=offset)).isoformat()
        for offset in range(days)
    ]
    for source_date in dates:
        shard = root / f"{specialist}-{source_date}.features"
        shard.write_bytes(f"{source_date}\n".encode())
        rows.append(
            {
                "date": source_date,
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
                "decisions_kept": days * 2,
                "target_coverage": {"guide_rows": days},
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
            "decisions": days * 2,
            "guide_rows": days,
            "days": days,
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
    tampered_date = "2026-07-15"
    (source / f"example-specialist-{tampered_date}.features").write_text(
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
    assert f"guide feature checksum failed: {tampered_date}" in result.stderr
    assert not (destination / "PROTECTED_EXPERT_CORPUS.json").exists()


def test_promotion_accepts_an_explicit_full_history_window(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source", days=32)
    destination = tmp_path / "corpora" / "example-specialist"
    state = tmp_path / "state.json"

    subprocess.run(
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
            str(state),
            "--expected-days",
            "32",
            "--bwlimit-kib",
            "100000",
        ],
        check=True,
    )

    receipt = json.loads(state.read_text())
    assert receipt["status"] == "ready"
    assert receipt["identity"]["days"] == 32
    assert len(receipt["identity"]["dates"]) == 32
    assert receipt["rollback_directory"] is None
    assert receipt["replaced_existing_destination"] is False
    assert (destination / "PROTECTED_EXPERT_CORPUS.json").is_file()
