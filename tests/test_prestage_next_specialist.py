from __future__ import annotations

import json
from pathlib import Path

from scripts import prestage_next_specialist as prestage


def test_prestage_contract_cannot_control_live_training() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/prestage_next_specialist.py"
    ).read_text(encoding="utf-8")
    assert "systemctl" not in source
    assert "subprocess" not in source
    assert "_atomic_selector" not in source
    assert "register_specialist_runtime" not in source
    assert '"live_training_modified": False' in source


def test_cycle_contract_pins_read_only_prestage() -> None:
    root = Path(__file__).parents[1]
    contract = json.loads(
        (root / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    stage = contract["prestage"]
    assert stage["live_training_modification_allowed"] is False
    assert stage["selector_update_allowed"] is False
    assert stage["service_control_allowed"] is False
    assert stage["required_target_coverage"] == list(prestage.TARGETS)
    assert stage["ladder_representatives"].endswith(
        "top_ladder_representatives.v1.json"
    )


def test_prestage_service_is_resource_bounded_and_periodic() -> None:
    root = Path(__file__).parents[1]
    service = (
        root / "deploy/systemd/pokebot-next-specialist-prestage.service"
    ).read_text(encoding="utf-8")
    timer = (
        root / "deploy/systemd/pokebot-next-specialist-prestage.timer"
    ).read_text(encoding="utf-8")
    assert "--build-cpu-pack" in service
    assert "MemoryMax=28G" in service
    assert "CPUQuota=400%" in service
    assert "Nice=15" in service
    assert "OnActiveSec=1min" in timer
    assert "OnUnitActiveSec=30min" in timer


def test_missing_representative_is_an_explicit_blocker(tmp_path: Path) -> None:
    registry = tmp_path / "representatives.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_deck_representatives/v1",
                "decks": {},
            }
        ),
        encoding="utf-8",
    )
    result = prestage._representative(registry, "future-specialist")
    assert result["ready"] is False
    assert result["reason"] == "exact_60_card_representative_missing"


def test_exact_representative_is_checksum_bound(tmp_path: Path) -> None:
    registry = tmp_path / "representatives.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_deck_representatives/v1",
                "decks": {"future-specialist": {"card_ids": list(range(60))}},
            }
        ),
        encoding="utf-8",
    )
    result = prestage._representative(registry, "future-specialist")
    assert result["ready"] is True
    assert result["card_count"] == 60
    assert result["cards_sha256"].startswith("sha256:")
