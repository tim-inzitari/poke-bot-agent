from __future__ import annotations

import json
from pathlib import Path

from scripts.current_deck_guide_prestage_snapshot import snapshot


def test_snapshot_prefers_running_window_and_reports_exact_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = tmp_path / "old/status"
    old.mkdir(parents=True)
    (old / "window.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "updated_at": 10,
                "date_window": {
                    "start": "2026-07-01",
                    "end": "2026-07-20",
                    "days": 20,
                },
                "completed": [{"date": "2026-07-01"}] * 20,
                "totals": {"guide_rows": 100},
            }
        ),
        encoding="utf-8",
    )
    active = tmp_path / "garchomp/status"
    active.mkdir(parents=True)
    (active / "window.json").write_text(
        json.dumps(
            {
                "state": "running",
                "updated_at": 20,
                "date_window": {
                    "start": "2026-07-04",
                    "end": "2026-07-23",
                    "days": 20,
                },
                "completed": [
                    {"date": "2026-07-04"},
                    {"date": "2026-07-05"},
                ],
                "current_dates": ["2026-07-06", "2026-07-07"],
                "totals": {
                    "records": 20,
                    "decisions": 200,
                    "guide_rows": 30,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.current_deck_guide_prestage_snapshot._unit_state",
        lambda specialist_id: {
            "name": f"pokebot-{specialist_id}-guide-window-v1.service",
            "active": specialist_id == "garchomp",
            "pid": 123 if specialist_id == "garchomp" else 0,
        },
    )

    result = snapshot(tmp_path)

    assert result["available"] is True
    assert result["active"]["specialist_id"] == "garchomp"
    assert result["active"]["completed_days"] == 2
    assert result["active"]["expected_days"] == 20
    assert result["active"]["percent"] == 10.0
    assert result["active"]["guide_rows"] == 30
