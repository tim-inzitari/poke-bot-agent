from __future__ import annotations

import json
from pathlib import Path

from scripts.current_deck_guide_prestage_snapshot import _unit_state, snapshot


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
        lambda specialist_id, **_kwargs: {
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


def test_snapshot_includes_checksum_ready_additional_full_history_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    teal = tmp_path / "teal-mask-ogerpon-ex-guide-corpus-full-v3"
    status = teal / "status"
    status.mkdir(parents=True)
    (status / "window.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "updated_at": 30,
                "date_window": {
                    "start": "2026-06-26",
                    "end": "2026-07-27",
                    "days": 32,
                },
                "completed": [{"date": str(index)} for index in range(32)],
                "current_dates": [],
                "totals": {
                    "records": 1135,
                    "decisions": 76226,
                    "guide_rows": 6814,
                },
            }
        ),
        encoding="utf-8",
    )
    (teal / "CURRENT_DECK_GUIDE_CORPUS_READY.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
                "status": "ready",
                "specialist_id": "teal-mask-ogerpon-ex",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.current_deck_guide_prestage_snapshot._unit_state",
        lambda _specialist_id, **_kwargs: {"active": False, "pid": 0},
    )

    result = snapshot(primary, additional_roots=(teal,))

    assert result["active"]["specialist_id"] == "teal-mask-ogerpon-ex"
    assert result["active"]["ready"] is True
    assert result["active"]["completed_days"] == 32
    assert result["active"]["records"] == 1135


def test_snapshot_includes_ready_archaludon_additional_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    arch = (
        tmp_path / "archaludon-ex-guide-full-public-schema7-r56-v1"
    )
    status = arch / "status"
    status.mkdir(parents=True)
    (status / "window.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "updated_at": 40,
                "date_window": {
                    "start": "2026-06-16",
                    "end": "2026-07-29",
                    "days": 44,
                },
                "completed": [{"date": str(index)} for index in range(44)],
                "current_dates": [],
                "totals": {
                    "records": 20000,
                    "decisions": 900000,
                    "guide_rows": 18000,
                },
            }
        ),
        encoding="utf-8",
    )
    (arch / "CURRENT_DECK_GUIDE_CORPUS_READY.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
                "status": "ready",
                "specialist_id": "archaludon-ex",
            }
        ),
        encoding="utf-8",
    )
    (arch / "ARCHALUDON_EX_GUIDE_CORPUS_READY.json").write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.archaludon_ex_guide_corpus_validation/v2"
                ),
                "status": "ready_checksum_validated",
                "records": 20000,
                "dataset_schema": 7,
                "feature_schema": 5,
                "schema6_feature_reuse_allowed": False,
                "source_window": {
                    "start": "2026-06-16",
                    "end": "2026-07-29",
                    "days": 44,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.current_deck_guide_prestage_snapshot._unit_state",
        lambda _specialist_id, **_kwargs: {"active": False, "pid": 0},
    )

    result = snapshot(primary, additional_roots=(arch,))

    assert result["active"]["specialist_id"] == "archaludon-ex"
    assert result["active"]["ready"] is True
    assert result["active"]["completed_days"] == 44
    assert result["active"]["records"] == 20000
    assert result["active"]["decisions"] == 900000


def test_snapshot_rejects_archaludon_without_schema7_final_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    arch = (
        tmp_path / "archaludon-ex-guide-full-public-schema7-r56-v1"
    )
    status = arch / "status"
    status.mkdir(parents=True)
    (status / "window.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "updated_at": 40,
                "date_window": {
                    "start": "2026-06-16",
                    "end": "2026-07-29",
                    "days": 44,
                },
                "completed": [{"date": str(index)} for index in range(44)],
                "totals": {
                    "records": 20000,
                    "decisions": 900000,
                    "guide_rows": 18000,
                },
            }
        ),
        encoding="utf-8",
    )
    (arch / "CURRENT_DECK_GUIDE_CORPUS_READY.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
                "status": "ready",
                "specialist_id": "archaludon-ex",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.current_deck_guide_prestage_snapshot._unit_state",
        lambda _specialist_id, **_kwargs: {"active": False, "pid": 0},
    )

    result = snapshot(primary, additional_roots=(arch,))

    assert result["active"]["specialist_id"] == "archaludon-ex"
    assert result["active"]["ready"] is False
    assert result["active"]["final_validation_receipt"] is None
    assert result["active"]["guide_rows"] == 18000


def test_full33_slop_box_window_uses_v4_managed_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    class Result:
        stdout = (
            "ActiveState=active\nSubState=running\n"
            "Result=success\nMainPID=123\n"
        )

    def fake_run(command, **_kwargs):
        captured.append(command)
        return Result()

    monkeypatch.setattr(
        "scripts.current_deck_guide_prestage_snapshot.subprocess.run",
        fake_run,
    )
    corpus = tmp_path / (
        "teal-mask-ogerpon-ex-guide-corpus-full-v4-slop-box"
    )

    result = _unit_state(
        "teal-mask-ogerpon-ex",
        corpus_root=corpus,
    )

    assert captured[0][2] == (
        "pokebot-teal-mask-slop-box-full33-guide-v4.service"
    )
    assert result["name"] == captured[0][2]
    assert result["active"] is True


def test_running_full33_window_infers_stable_teal_specialist_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    teal = tmp_path / (
        "teal-mask-ogerpon-ex-guide-corpus-full-v4-slop-box"
    )
    status = teal / "status"
    status.mkdir(parents=True)
    (status / "window.json").write_text(
        json.dumps(
            {
                "state": "running",
                "updated_at": 50,
                "date_window": {
                    "start": "2026-06-26",
                    "end": "2026-07-28",
                    "days": 33,
                },
                "completed": [{"date": str(index)} for index in range(27)],
                "current_dates": ["2026-07-24"],
                "totals": {"records": 307},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.current_deck_guide_prestage_snapshot._unit_state",
        lambda specialist_id, **_kwargs: {
            "name": "pokebot-teal-mask-slop-box-full33-guide-v4.service",
            "active": specialist_id == "teal-mask-ogerpon-ex",
            "pid": 123,
        },
    )

    result = snapshot(primary, additional_roots=(teal,))

    assert result["active"]["specialist_id"] == "teal-mask-ogerpon-ex"
    assert result["active"]["service"]["active"] is True
