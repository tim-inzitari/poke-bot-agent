from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from poke_bot.pure_rl.strong_public_gate import load_active_gate_contract
from scripts.apply_archetype_label_integrity_at_boundary import (
    _ALLOWED_MIGRATION_PREFIXES,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ops" / "alakazam_gate_program_v1.json"
# The executable source is authoritative.  A historical deploy/staging copy
# must never mask a regression in the trainer that a future deployment uses.
STAGED_TRAINER = ROOT / "scripts" / "train_pure_rl.py"
STAGED_DROPIN = (
    ROOT
    / "deploy/systemd/pokebot-pure-rl-alakazam.service.d"
    / "99-active-gate-v15-strong-practice.conf"
)
V17_MIGRATION_DROPIN = (
    ROOT / "deploy/staging/zz-archetype-label-integrity-v16-migration.conf"
)
V17_STEADY_DROPIN = (
    ROOT / "deploy/staging/zz-archetype-label-integrity-v16-steady.conf"
)
V17_ROOT = "/home/pokebot/poke-bot-agent-deployments/pure-rl-resident-v17-research-controls"


def test_launch_assertion_reports_active_contract_1750(tmp_path: Path) -> None:
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
    assert payload["opponents"] == 7
    assert payload["games_total"] == 1750
    assert payload["games_per_opponent"] == 250
    assert payload["seat0_games_per_opponent"] == 125
    assert payload["seat1_games_per_opponent"] == 125


def test_contract_can_advance_but_legacy_original_four_cannot_return(
    tmp_path: Path,
) -> None:
    future = json.loads(CONTRACT.read_text())
    future["active_gate_id"] = "future-eight-agent-gate"
    future["next_gate"]["id"] = "future-eight-agent-gate"
    # The staged Alakazam LC50 fallback is intentionally bound to the current
    # LC55 gate identity. A future gate revision must define its own fallback
    # explicitly rather than inheriting a stale transition.
    future.pop("fallback_transition", None)
    ninth = dict(future["next_gate"]["roster"][0])
    ninth.update(
        opponent_id="future-opponent",
        content_digest="sha256:" + "f" * 64,
        tier="A",
        weight=1.0,
    )
    future["next_gate"]["roster"].append(ninth)
    future["next_gate"]["evaluation"]["games_total"] = 2000
    future["active_gate_semantics"].update(
        gate_roster_size=8,
        gate_games_total=2000,
    )
    path = tmp_path / "future.json"
    path.write_text(json.dumps(future))
    assert load_active_gate_contract(path)["active_gate_id"] == (
        "future-eight-agent-gate"
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
    assert "production full-loop launch requires --active-gate-contract" in source
    assert "research controls are diagnostic-only" in source
    assert 'stage_label=(\n                        "heldout:strong_public_gate"' in source
    assert "official_specs=heldout_specs" in source
    assert "opponent_ids=active_gate_ids" in source
    assert "build_active_gate_result(" in source
    assert "heldout_ids = set(research_ids) | set(active_gate_ids)" in source
    assert "active_gate_content_digests = set(installed_gate_digests.values())" in source
    assert "excluded_gate_digest_aliases" in source
    assert "excluded_research_digest_aliases" in source
    assert "active_gate_digests=" in source
    assert "ACTIVE_GATE_PRACTICE_SEED_DISJOINT" in source
    assert "RESEARCH_CONTROL_GROUP" in source
    assert "measure:research_controls" in source
    assert "_assert_strong_public_practice_jobs" in source
    assert "_assert_research_control_jobs" in source
    assert '"opponents.collect",' in source


def test_staged_v15_launch_preserves_practice_and_fixed_research_quota() -> None:
    source = STAGED_DROPIN.read_text()
    assert "scripts/assert_active_gate_launch.py --contract" in source
    assert "--active-gate-contract" in source
    assert "--heldout-games" not in source
    assert "--games-per-iter 8192" in source
    assert "--official-collect-frac 0.50" in source
    assert "--research-control-games-per-iter 1000" in source
    assert "--official-adaptive-targeting" in source
    assert "--strong-public-practice-target-wr 0.55" in source
    assert "--strong-public-practice-temperature 0.35" in source
    assert "--allow-clean-boundary-design-migration" in source
    assert "--boundary-design-migration-reason additive-research-controls-v1" in source

    exec_line = next(
        line for line in source.splitlines() if line.startswith("ExecStart=") and line != "ExecStart="
    )
    argv = shlex.split(exec_line.split("=", 1)[1])
    trainer_argv = argv[argv.index("--") + 1 :]
    assert trainer_argv[trainer_argv.index("--games-per-iter") + 1] == "8192"
    assert trainer_argv[
        trainer_argv.index("--research-control-games-per-iter") + 1
    ] == "1000"
    assert trainer_argv[trainer_argv.index("--official-collect-frac") + 1] == "0.50"
    assert "--allow-clean-boundary-design-migration" in trainer_argv
    assert trainer_argv[
        trainer_argv.index("--boundary-design-migration-reason") + 1
    ] == "additive-research-controls-v1"


def test_boundary_watcher_allows_expected_contract_identity_changes() -> None:
    assert "expert_rehearsal.loss_weights" in _ALLOWED_MIGRATION_PREFIXES
    assert "gates.active_contract" in _ALLOWED_MIGRATION_PREFIXES


def test_boundary_watcher_pins_the_complete_runtime_identity() -> None:
    source = (
        ROOT / "scripts/apply_archetype_label_integrity_at_boundary.py"
    ).read_text(encoding="utf-8")
    for flag in (
        "--source-sha256",
        "--source-tree-sha256",
        "--gate-contract-sha256",
        "--migration-dropin-sha256",
        "--steady-dropin-sha256",
        "--research-module-sha256",
        "--research-registry-sha256",
    ):
        assert f'parser.add_argument("{flag}", required=True)' in source
    assert 'module._source_snapshot(staged_root)' in source
    assert '"active gate contract"' in source
    assert '"migration drop-in"' in source
    assert '"steady drop-in"' in source


def test_v17_migration_and_steady_dropins_differ_only_by_one_time_authority() -> None:
    migration = V17_MIGRATION_DROPIN.read_text(encoding="utf-8")
    steady = V17_STEADY_DROPIN.read_text(encoding="utf-8")
    for payload, stop_lock in ((migration, "no"), (steady, "yes")):
        assert f"RefuseManualStop={stop_lock}" in payload
        assert f"WorkingDirectory={V17_ROOT}" in payload
        assert f"--active-gate-contract {V17_ROOT}/ops/alakazam_gate_program_v1.json" in payload
        assert f"--research-control-registry {V17_ROOT}/ops/research_control_registry_v1.json" in payload
        assert "--research-control-games-per-iter 1000" in payload
        assert "--games-per-iter 8192" in payload
        assert "--resume auto" in payload
        assert "pure-rl-resident-v15-strong-practice" not in payload
    assert "--allow-clean-boundary-design-migration" in migration
    assert (
        "--boundary-design-migration-reason "
        "canonical-archetype-labels-and-research-controls-v17"
    ) in migration
    assert "--allow-clean-boundary-design-migration" not in steady
    assert "--boundary-design-migration-reason" not in steady

    def trainer_argv(payload: str) -> list[str]:
        line = next(
            value for value in payload.splitlines() if value.startswith("ExecStart=")
            and value != "ExecStart="
        )
        return shlex.split(line.split("=", 1)[1])

    migration_argv = trainer_argv(migration)
    steady_argv = trainer_argv(steady)
    marker = migration_argv.index("--allow-clean-boundary-design-migration")
    del migration_argv[marker]
    reason = migration_argv.index("--boundary-design-migration-reason")
    del migration_argv[reason : reason + 2]
    assert migration_argv == steady_argv
