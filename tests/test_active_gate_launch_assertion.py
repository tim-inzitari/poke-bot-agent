from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from poke_bot.pure_rl.strong_public_gate import load_active_gate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ops" / "alakazam_gate_program_v1.json"
STAGED_TRAINER = ROOT / "deploy" / "staging" / "train_pure_rl_v11.py"
if not STAGED_TRAINER.is_file():
    STAGED_TRAINER = ROOT / "scripts" / "train_pure_rl.py"
PRODUCTION_DROPIN = (
    ROOT
    / "deploy/systemd/pokebot-pure-rl-alakazam.service.d"
    / "99-active-gate-v13.conf"
)


def test_launch_assertion_reports_active_contract_2000(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "assert_active_gate_launch.py"),
            "--contract",
            str(CONTRACT),
            "--receipt",
            str(receipt),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(receipt.read_text())
    assert "ACTIVE_GATE_LAUNCH_ASSERT PASS" in completed.stdout
    assert payload["opponents"] == 8
    assert payload["games_total"] == 2000
    assert payload["games_per_opponent"] == 250
    assert payload["seat0_games_per_opponent"] == 125
    assert payload["seat1_games_per_opponent"] == 125


def test_contract_can_advance_but_legacy_original_four_cannot_return(
    tmp_path: Path,
) -> None:
    future = json.loads(CONTRACT.read_text())
    future["active_gate_id"] = "future-nine-agent-gate"
    future["next_gate"]["id"] = "future-nine-agent-gate"
    ninth = dict(future["next_gate"]["roster"][0])
    ninth.update(
        opponent_id="future-opponent",
        content_digest="sha256:" + "f" * 64,
        tier="A",
        weight=1.0,
    )
    future["next_gate"]["roster"].append(ninth)
    future["next_gate"]["evaluation"]["games_total"] = 2250
    future["active_gate_semantics"].update(
        gate_roster_size=9,
        gate_games_total=2250,
    )
    path = tmp_path / "future.json"
    path.write_text(json.dumps(future))
    assert load_active_gate_contract(path)["active_gate_id"] == (
        "future-nine-agent-gate"
    )

    legacy = json.loads(CONTRACT.read_text())
    legacy["next_gate"]["roster"] = [
        {
            "opponent_id": row["opponent_id"],
            "content_digest": "sha256:" + str(index + 1) * 64,
            "tier": "A",
            "weight": 1.0,
        }
        for index, row in enumerate(legacy["next_gate"]["research_measurements"])
    ]
    legacy["next_gate"]["evaluation"].update(
        games_total=1000,
        games_per_opponent=250,
        seat0_games_per_opponent=125,
        seat1_games_per_opponent=125,
    )
    legacy["active_gate_semantics"].update(
        gate_roster_size=4,
        games_per_opponent=250,
        gate_games_total=1000,
    )
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="cannot be the active gate"):
        load_active_gate_contract(path)


def test_production_trainer_has_contract_driven_strong_gate_wiring() -> None:
    source = STAGED_TRAINER.read_text()
    assert "production specialist launch requires --active-gate-contract" in source
    assert 'stage_label=(\n                        "heldout:strong_public_gate"' in source
    assert "official_specs=heldout_specs" in source
    assert "opponent_ids=active_gate_ids" in source
    assert "build_active_gate_result(" in source
    assert (
        "heldout_ids = set(OFFICIAL_BASELINE_IDS) | set(active_gate_ids)"
        in source
    )
    assert "active_gate_content_digests = set(installed_gate_digests.values())" in source
    assert "excluded_gate_digest_aliases" in source
    assert "ACTIVE_GATE_TRAINING_DISJOINT" in source
    assert '"opponents.collect",' in source


def test_steady_production_launch_is_contract_bound_and_has_no_migration_key() -> None:
    source = PRODUCTION_DROPIN.read_text()
    assert "scripts/assert_active_gate_launch.py --contract" in source
    assert "--active-gate-contract" in source
    assert "--heldout-games" not in source
    assert "--allow-clean-boundary-design-migration" not in source
    assert "--boundary-design-migration-reason" not in source
