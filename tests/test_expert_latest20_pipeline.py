from __future__ import annotations

import csv
import json
import subprocess
import zipfile
from argparse import Namespace
from datetime import date, timedelta
from pathlib import Path

import pytest

import scripts.finalize_expert_latest20_elmo as expert_finalizer
import scripts.refresh_expert_latest20_bert as expert_refresher
from scripts.finalize_expert_latest20_elmo import commit, prepare


def _index(
    path: Path,
    *,
    days: int = 20,
    start: date = date(2026, 7, 4),
) -> list[str]:
    values = [
        (start + timedelta(days=offset)).isoformat()
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
    assert current["window_policy"] == "latest_20_consecutive_calendar_days"
    assert Path(current["versioned_receipt"]).is_file()


def test_pinned_window_selects_the_exact_requested_dates_in_both_stages(
    tmp_path: Path,
):
    index = tmp_path / "manifest.csv"
    all_days = _index(index, days=22, start=date(2026, 7, 20))
    pinned_days = all_days[2:]
    window_start, window_end = pinned_days[0], pinned_days[-1]
    archive_root = tmp_path / "archive"
    for day in pinned_days:
        _archive(
            archive_root / f"pokemon-tcg-ai-battle-episodes-{day}.zip",
            day,
        )

    finalizer_rows = expert_finalizer._window(
        index,
        window_start=window_start,
        window_end=window_end,
    )
    refresher_rows = expert_refresher._window(
        index,
        window_start=window_start,
        window_end=window_end,
    )
    result = commit(
        index,
        archive_root=archive_root,
        receipt_root=tmp_path / "receipts",
        window_start=window_start,
        window_end=window_end,
    )

    assert [row["date"] for row in finalizer_rows] == pinned_days
    assert [row["date"] for row in refresher_rows] == pinned_days
    assert result["window_start"] == "2026-07-22"
    assert result["window_end"] == "2026-08-10"
    assert result["window_policy"] == "exact_20_consecutive_calendar_days"
    assert [row["date"] for row in result["archives"]] == pinned_days


@pytest.mark.parametrize(
    "window",
    [expert_finalizer._window, expert_refresher._window],
)
def test_pinned_window_refuses_to_substitute_an_older_twentieth_day(
    tmp_path: Path,
    window,
):
    index = tmp_path / "manifest.csv"
    _index(index, days=20, start=date(2026, 7, 21))

    with pytest.raises(RuntimeError, match="does not contain exactly 20 days"):
        window(
            index,
            window_start="2026-07-22",
            window_end="2026-08-10",
        )


@pytest.mark.parametrize(
    "window",
    [expert_finalizer._window, expert_refresher._window],
)
def test_pinned_window_requires_both_bounds(tmp_path: Path, window):
    index = tmp_path / "manifest.csv"
    _index(index)

    with pytest.raises(RuntimeError, match="must be supplied together"):
        window(index, window_start="2026-07-22")


def test_refresher_forwards_pinned_window_to_every_elmo_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index = tmp_path / "manifest.csv"
    days = _index(index, days=22, start=date(2026, 7, 20))[2:]
    window_start, window_end = days[0], days[-1]
    download_calls: list[tuple[str | None, str | None]] = []
    remote_commands: list[list[str]] = []
    receipt = {
        "status": "ready",
        "window_start": window_start,
        "window_end": window_end,
        "window_policy": "exact_20_consecutive_calendar_days",
        "days": 20,
        "archives": [
            {"date": day, "sha256": f"sha256:{offset:064x}"}
            for offset, day in enumerate(days)
        ],
    }

    def fake_download_index(
        _root: Path,
        *,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> Path:
        download_calls.append((window_start, window_end))
        return index

    def fake_remote_json(
        _host: str,
        _source_address: str,
        command: list[str],
    ) -> dict[str, object]:
        remote_commands.append(command)
        if "prepare" in command:
            return {"missing": []}
        if "commit" in command:
            return receipt
        raise AssertionError(f"unexpected remote command: {command}")

    monkeypatch.setattr(expert_refresher, "_default_interface", lambda: "en1")
    monkeypatch.setattr(expert_refresher, "_download_index", fake_download_index)
    monkeypatch.setattr(expert_refresher, "_remote_json", fake_remote_json)
    monkeypatch.setattr(
        expert_refresher,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )

    result = expert_refresher.refresh(
        Namespace(
            root=tmp_path / "bert",
            wifi_interface="en1",
            ethernet_source="bert",
            elmo="admin@elmo",
            remote_stage="/tmp/expert-refresh",
            elmo_finalizer="/tmp/finalize_expert_latest20_elmo.py",
            elmo_archive_root="/tmp/archive",
            elmo_receipt_root="/tmp/receipts",
            inzi="trainer@example.test",
            inzi_receipt="/tmp/inzi/expert-latest20-current.json",
            elmo_reuse_root=[],
            window_start=window_start,
            window_end=window_end,
        )
    )

    assert download_calls == [(window_start, window_end)]
    assert result["window_policy"] == "exact_20_consecutive_calendar_days"
    assert len(remote_commands) == 2
    for command in remote_commands:
        assert command[
            command.index("--window-start") + 1
        ] == window_start
        assert command[command.index("--window-end") + 1] == window_end


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


@pytest.mark.parametrize(
    ("day", "indexed", "validated", "checksum"),
    [
        (
            "2026-08-06",
            4_633,
            4_631,
            "sha256:46f3a95ba0456027870b504a64424fd3f9afcf3aabe5d0453803d7c5145631a4",
        ),
        (
            "2026-08-07",
            4_645,
            4_639,
            "sha256:c9325476fde8bf6e3a9021520867e9dbfeaf3ec09124b01f742cb07fa5877e63",
        ),
    ],
)
def test_august_publisher_discrepancies_are_exactly_checksum_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    day: str,
    indexed: int,
    validated: int,
    checksum: str,
):
    archive = tmp_path / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
    archive.write_bytes(b"published-archive-placeholder")
    expected = {
        "index_episode_count": indexed,
        "validated_episode_count": validated,
        "archive_sha256": checksum,
    }
    finalizer_exception = expert_finalizer.KNOWN_SOURCE_DISCREPANCIES[day]
    assert {
        key: finalizer_exception[key]
        for key in expected
    } == expected
    assert expert_refresher.KNOWN_SOURCE_DISCREPANCIES[day] == expected

    monkeypatch.setattr(expert_finalizer, "_archive_count", lambda _path: validated)
    monkeypatch.setattr(expert_finalizer, "_sha256", lambda _path: checksum)
    assert expert_finalizer._validate(
        archive,
        indexed,
        day=day,
    )["source_discrepancy"] == finalizer_exception

    class SyntheticArchive:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def namelist(self):
            return ["episode.json"] * validated

    monkeypatch.setattr(
        expert_refresher.zipfile,
        "ZipFile",
        lambda _path: SyntheticArchive(),
    )
    monkeypatch.setattr(expert_refresher, "_sha256", lambda _path: checksum)
    assert expert_refresher._validate(archive, indexed, day=day) == checksum

    bad_checksum = "sha256:" + "0" * 64
    monkeypatch.setattr(expert_finalizer, "_sha256", lambda _path: bad_checksum)
    with pytest.raises(RuntimeError, match="episode-count mismatch"):
        expert_finalizer._validate(archive, indexed, day=day)
    monkeypatch.setattr(expert_refresher, "_sha256", lambda _path: bad_checksum)
    with pytest.raises(RuntimeError, match="episode-count mismatch"):
        expert_refresher._validate(archive, indexed, day=day)


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
