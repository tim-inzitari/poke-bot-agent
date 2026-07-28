from pathlib import Path

import pytest

from scripts.run_live_dashboard_snapshot import resolve_runtime_snapshot


def test_dashboard_snapshot_follows_selector_runtime_root(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    snapshot = runtime / "scripts" / "dashboard_snapshot.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("# snapshot\n", encoding="utf-8")
    selector = tmp_path / "specialist_runtime.env"
    selector.write_text(
        f"POKEBOT_ACTIVE_SPECIALIST=thwackey\n"
        f"POKEBOT_SPECIALIST_RUNTIME_ROOT={runtime}\n",
        encoding="utf-8",
    )

    assert resolve_runtime_snapshot(selector) == snapshot


def test_dashboard_snapshot_rejects_duplicate_runtime_roots(
    tmp_path: Path,
) -> None:
    selector = tmp_path / "specialist_runtime.env"
    selector.write_text(
        "POKEBOT_SPECIALIST_RUNTIME_ROOT=/one\n"
        "POKEBOT_SPECIALIST_RUNTIME_ROOT=/two\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="absent or duplicated"):
        resolve_runtime_snapshot(selector)
