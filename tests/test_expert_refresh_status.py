from __future__ import annotations

import csv
from pathlib import Path

from scripts.expert_refresh_status import build_status


def test_expert_refresh_reports_dynamic_window_and_download_progress(
    tmp_path: Path, monkeypatch
) -> None:
    index = tmp_path / "kaggle/input/pokemon-tcg-ai-battle-episodes-index"
    index.mkdir(parents=True)
    with (index / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "daily_dataset_slug", "daily_dataset_url", "episode_count", "total_bytes"],
        )
        writer.writeheader()
        for day in range(11, 21):
            writer.writerow(
                {
                    "date": f"2026-07-{day:02d}",
                    "daily_dataset_slug": f"episodes-{day}",
                    "daily_dataset_url": "https://example.invalid",
                    "episode_count": 5000,
                    "total_bytes": 1,
                }
            )
    (tmp_path / "data/episodes/raw").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "refresh-status.tsv").write_text(
        "stage\tdownloading\nday\t2026-07-13\ncompleted\t2\ntotal\t10\n"
        "window_start\t2026-07-11\nwindow_end\t2026-07-20\n"
        "detail\tdownloading\nupdated_epoch\t123\n",
        encoding="utf-8",
    )
    for day in (11, 12):
        (tmp_path / "data/episodes/raw" / f"pokemon-tcg-ai-battle-episodes-2026-07-{day}.zip").write_bytes(b"ok")
    (tmp_path / "data/episodes/raw/pokemon-tcg-ai-battle-episodes-2026-07-13.zip").write_bytes(b"partial")
    (tmp_path / "logs/refresh.log").write_text(
        " 39%|███▉| 274M/704M [01:07<01:53, 3.98MB/s]\r",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.expert_refresh_status._unit_state",
        lambda _name: {"active": True, "active_state": "active", "sub_state": "running", "pid": 42},
    )

    result = build_status(tmp_path, host="Elmo", unit="refresh.service")

    assert result["available"] is True
    assert result["window_start"] == "2026-07-11"
    assert result["window_end"] == "2026-07-20"
    assert result["archive_ready_days"] == 2
    assert result["day_percent"] == 39.0
    assert result["percent"] == 11.95
    assert result["current"] == 274 * 1024**2
    assert result["total"] == 704 * 1024**2
    assert result["rate"] == 3.98 * 1024**2
    assert result["days"][2]["stage"] == "downloading"
