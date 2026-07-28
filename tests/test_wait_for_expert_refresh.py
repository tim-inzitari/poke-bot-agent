from __future__ import annotations

from scripts.wait_for_expert_refresh import _dates


def test_dates_are_inclusive() -> None:
    assert _dates("2026-07-02", "2026-07-04") == [
        "2026-07-02",
        "2026-07-03",
        "2026-07-04",
    ]
