from __future__ import annotations

import json
from pathlib import Path

from scripts import materialize_authoritative_guide_window_parallel as window
from scripts.materialize_authoritative_alakazam_day import (
    additive_classifier_ids,
    logical_aliases,
)


def test_parallel_window_dates_are_complete_and_ordered() -> None:
    assert window._dates("2026-07-04", "2026-07-07") == [
        "2026-07-04",
        "2026-07-05",
        "2026-07-06",
        "2026-07-07",
    ]


def test_daily_guide_classifier_keeps_requested_specialist_additive() -> None:
    assert additive_classifier_ids("rockets-mewtwo", []) == (
        "rockets-mewtwo",
    )
    assert additive_classifier_ids(
        "rockets-mewtwo", ["garchomp", "rockets-mewtwo"]
    ) == ("rockets-mewtwo", "garchomp")


def test_daily_guide_classifier_parses_one_logical_alias() -> None:
    assert logical_aliases(["festival-lead=thwackey"]) == {
        "festival-lead": "thwackey"
    }


def test_elmo_guide_launcher_preserves_native_runtime_and_parallelism() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "ops" / "elmo" / "run_current_deck_guide_window.sh"
    ).read_text(encoding="utf-8")
    assert '--setenv="CG_LIB_PATH=$cg_runtime"' in source
    assert 'test -f "$cg_runtime/cg/__init__.py"' in source
    assert '--day-parallelism "$day_parallelism"' in source
    assert '--workers-per-day "$workers_per_day"' in source
    assert 'roster.get("logical_aliases")' in source
    assert '"${logical_alias_args[@]}"' in source
    assert 'systemctl is-active --quiet "$unit"' in source


def test_teal_full32_builder_requires_exact_catalog_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    builder = (
        root / "ops" / "elmo" / "build_teal_mask_ogerpon_full32_corpus.sh"
    ).read_text(encoding="utf-8")
    parallel = (
        root / "scripts" / "materialize_authoritative_guide_window_parallel.py"
    ).read_text(encoding="utf-8")
    daily = (
        root / "scripts" / "materialize_authoritative_alakazam_day.py"
    ).read_text(encoding="utf-8")

    assert "teal-mask-ogerpon-ex-guide-corpus-full-v2" in builder
    assert (
        "--authoritative-only-archetype teal-mask-ogerpon-ex" in builder
    )
    assert "--guide-version teal-mask-ogerpon-ex-north-star-v2" in builder
    assert "actual_by_day != expected_by_day" in builder
    assert '"--authoritative-only-archetype"' in parallel
    assert '"--authoritative-only-archetype"' in daily


def test_completed_row_requires_exact_specialist_guide_and_accepts_zero_day(
    tmp_path: Path,
) -> None:
    output = tmp_path / "garchomp-2026-07-04.features"
    output.write_bytes(b"feature")
    receipt = {
        "format": "pokebot-authoritative-visual-day-receipt",
        "format_version": 1,
        "source_date": "2026-07-04",
        "selection": {
            "acting_seat_archetype": "garchomp",
            "current_deck_guide": "garchomp",
        },
        "stats": {
            "records_kept": 12,
            "decisions_kept": 345,
            "target_coverage": {"guide_rows": 67},
        },
        "output": {"sha256": "sha256:" + "a" * 64},
    }
    output.with_name(output.name + ".receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    row = window._completed_row(
        day="2026-07-04",
        output=output,
        specialist_id="garchomp",
        guide_id="garchomp",
    )

    assert row is not None
    assert row["records"] == 12
    assert row["decisions"] == 345
    assert row["guide_rows"] == 67

    receipt["stats"]["target_coverage"]["guide_rows"] = 0
    output.with_name(output.name + ".receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    zero_day = window._completed_row(
        day="2026-07-04",
        output=output,
        specialist_id="garchomp",
        guide_id="garchomp",
    )
    assert zero_day is not None
    assert zero_day["guide_rows"] == 0
    assert zero_day["zero_guide_rows"] is True

    receipt["stats"]["records_kept"] = 0
    receipt["stats"]["decisions_kept"] = 0
    receipt["stats"]["target_coverage"] = {}
    output.with_name(output.name + ".receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    empty_day = window._completed_row(
        day="2026-07-04",
        output=output,
        specialist_id="garchomp",
        guide_id="garchomp",
    )
    assert empty_day is not None
    assert empty_day["guide_rows"] == 0

    receipt["stats"]["records_kept"] = 1
    receipt["stats"]["decisions_kept"] = 1
    output.with_name(output.name + ".receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    assert (
        window._completed_row(
            day="2026-07-04",
            output=output,
            specialist_id="garchomp",
            guide_id="garchomp",
        )
        is None
    )
