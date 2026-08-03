from __future__ import annotations

import csv
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

import scripts.finalize_expert_latest20_elmo as expert_finalizer
from scripts.finalize_expert_latest20_elmo import commit, prepare


def _index(path: Path, *, days: int = 20) -> list[str]:
    values = [
        (date(2026, 7, 4) + timedelta(days=offset)).isoformat()
        for offset in range(days)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("date", "daily_dataset_slug", "episode_count"),
        )
        writer.writeheader()
        for value in values:
            writer.writerow(
                {
                    "date": value,
                    "daily_dataset_slug": (
                        f"pokemon-tcg-ai-battle-episodes-{value}"
                    ),
                    "episode_count": 1,
                }
            )
    return values


def _archive(path: Path, day: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{day}/episode.json", "{}")


def test_prepare_reuses_valid_archives_and_reports_only_missing(tmp_path: Path):
    index = tmp_path / "manifest.csv"
    days = _index(index)
    reuse = tmp_path / "reuse"
    archive_root = tmp_path / "archive"
    for day in days[:-1]:
        _archive(
            reuse / f"pokemon-tcg-ai-battle-episodes-{day}.zip", day
        )

    result = prepare(
        index, archive_root=archive_root, reuse_roots=[reuse]
    )

    assert [row["date"] for row in result["missing"]] == [days[-1]]
    assert len(list(archive_root.glob("*.zip"))) == 19


def test_commit_writes_exact_atomic_latest20_receipts(tmp_path: Path):
    index = tmp_path / "manifest.csv"
    days = _index(index)
    archive_root = tmp_path / "archive"
    for day in days:
        _archive(
            archive_root / f"pokemon-tcg-ai-battle-episodes-{day}.zip",
            day,
        )
    receipt_root = tmp_path / "receipts"

    result = commit(
        index, archive_root=archive_root, receipt_root=receipt_root
    )

    current = json.loads(
        (receipt_root / "current.json").read_text(encoding="utf-8")
    )
    assert result["window_start"] == days[0]
    assert result["window_end"] == days[-1]
    assert result["days"] == 20
    assert len(result["archives"]) == 20
    assert all(row["validated"] for row in result["archives"])
    assert current["status"] == "ready"
    assert Path(current["versioned_receipt"]).is_file()


def test_prepare_rejects_nonconsecutive_calendar_window(tmp_path: Path):
    index = tmp_path / "manifest.csv"
    _index(index)
    rows = list(csv.DictReader(index.open(encoding="utf-8")))
    rows[-1]["date"] = "2026-07-25"
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RuntimeError, match="consecutive calendar"):
        prepare(index, archive_root=tmp_path / "archive", reuse_roots=[])


def test_known_publisher_omission_is_exactly_checksum_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    archive = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-24.zip"
    archive.write_bytes(b"published-archive-placeholder")
    exception = expert_finalizer.KNOWN_SOURCE_DISCREPANCIES["2026-07-24"]
    monkeypatch.setattr(
        expert_finalizer,
        "_archive_count",
        lambda _path: exception["validated_episode_count"],
    )
    monkeypatch.setattr(
        expert_finalizer,
        "_sha256",
        lambda _path: exception["archive_sha256"],
    )

    result = expert_finalizer._validate(
        archive,
        exception["index_episode_count"],
        day="2026-07-24",
    )

    assert result["index_episode_count"] == 4_445
    assert result["validated_episode_count"] == 4_444
    assert result["source_discrepancy"] == exception


def test_known_publisher_omission_rejects_wrong_archive_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    archive = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-24.zip"
    archive.write_bytes(b"different-archive")
    exception = expert_finalizer.KNOWN_SOURCE_DISCREPANCIES["2026-07-24"]
    monkeypatch.setattr(
        expert_finalizer,
        "_archive_count",
        lambda _path: exception["validated_episode_count"],
    )
    monkeypatch.setattr(
        expert_finalizer,
        "_sha256",
        lambda _path: "sha256:" + "0" * 64,
    )

    with pytest.raises(RuntimeError, match="episode-count mismatch"):
        expert_finalizer._validate(
            archive,
            exception["index_episode_count"],
            day="2026-07-24",
        )


def test_elmo_feature_builder_has_bounded_preemptible_cpu_shares() -> None:
    script = Path("ops/elmo/run_missing_latest20_feature_days.sh").read_text()

    assert 'CPU_SHARES="${POKEBOT_CPU_SHARES:-256}"' in script
    assert '--cpu-shares "$CPU_SHARES"' in script
    assert '"$CPU_SHARES"' in script
    assert '${XDG_RUNTIME_DIR:-/tmp}/${NAME}.lock' in script
    assert 'READY_RECEIPT_NAME="${POKEBOT_READY_RECEIPT_NAME:-MISSING_DAYS_READY.json}"' in script
    assert '-e POKEBOT_READY_RECEIPT_NAME="$READY_RECEIPT_NAME"' in script
    assert 'SOURCE="${POKEBOT_SOURCE:-/home/admin/pokebot-expert-src-v6-strategic}"' in script
    assert 'REQUIRED_DATASET_SCHEMA="${POKEBOT_REQUIRED_DATASET_SCHEMA:-6}"' in script
    assert "from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION" in script
    assert "EXPANDED_STRATEGIC_SCHEMA_DIGEST" in script
