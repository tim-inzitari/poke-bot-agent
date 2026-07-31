import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.model_registry import sha256
from scripts import watch_deck_agnostic_core_transition as watcher


ROOT = Path(__file__).resolve().parents[1]


def test_alakazam_specialist_unit_pins_volume_fleet_and_rehearsal_contract() -> None:
    unit = (ROOT / "deploy/systemd/pokebot-pure-rl-alakazam.service").read_text()
    assert "Environment=PURE_RL_SELF_PLAY_FRAC=0.125" in unit
    assert "Environment=PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0" in unit
    assert "Environment=PURE_RL_REMOTE_DISPATCH_CHUNK=256" in unit
    assert "Environment=POKEBOT_REMOTE_SOCKET_PREFETCH=1" in unit
    assert "Environment=POKEBOT_REMOTE_SOCKET_PREFETCH_MAX=2" in unit
    assert (
        'Environment="POKEBOT_REMOTE_ENDPOINT_CHUNKS='
        '192.168.1.143:8765=336,bert.local:8766=176"'
    ) in unit
    assert "Environment=POKEBOT_REMOTE_QUEUE_LOW_WATER_FRAC=0.50" in unit
    assert "Environment=POKEBOT_REMOTE_QUEUE_PROBE_S=1.0" in unit
    assert "Environment=PURE_RL_REPLAY_WINDOW_SHARDS=2" in unit
    assert "--mode specialist" in unit
    assert "--specialist-archetype alakazam" in unit
    assert "--official-collect-frac 0.50" in unit
    assert "Environment=POKEBOT_ALAKAZAM_GUIDE_TARGETS=1" in unit
    assert "--alakazam-guide-loss-weight 0.05" in unit
    assert "--archetype-aux-loss-weight 0.05" in unit
    assert "--opp-hand-loss-weight 0.05" in unit
    assert "--opp-remainder-loss-weight 0.05" in unit
    assert "--lethal-threat-loss-weight 0.025" in unit
    assert "--prize-race-loss-weight 0.025" in unit
    assert "libcg_hidden_inzi_v1.so" in unit
    assert "--games-per-iter 8192" in unit
    assert "--measurement-decks alakazam" in unit
    assert "--expert-rehearsal-every 5" in unit
    assert "--expert-rehearsal-epochs 5" in unit
    assert "alakazam-latest10-20260709-20260718/PROTECTED_EXPERT_CORPUS.json" in unit
    assert (
        "--initial-learner-checkpoint /home/inzi/poke-bot-agent/outputs/"
        "pure_rl/_handoff/alakazam_temporal1_expert_v15_seed.pt"
    ) in unit
    assert "MemoryMax=112G" in unit


def test_transition_watcher_is_exact_40_percent_or_patience_ten() -> None:
    unit = (ROOT / "deploy/systemd/pokebot-deck-agnostic-transition.service").read_text()
    assert "--threshold-wr 0.40" in unit
    assert "--plateau-patience 10" in unit
    assert "--force-after-iteration 3" in unit
    assert "--required-heldout-games 1000" in unit
    assert "--start-iteration 0" in unit
    assert (
        "--specialist-deployment-root /home/inzi/poke-bot-agent-deployments/"
        "pure-rl-continuous-rehearsal-v1"
    ) in unit
    assert (
        "--handoff-ready /home/inzi/poke-bot-agent/outputs/state/"
        "alakazam-specialist-build-ready.json"
    ) in unit
    assert "pokebot-pure-rl-alakazam-bootstrap.service" in unit
    assert "alakazam-latest10-20260709-20260718/PROTECTED_EXPERT_CORPUS.json" in unit


def test_user_force_boundary_waits_for_exact_iteration_three() -> None:
    watching = {
        "latest_iteration": 2,
        "triggered": False,
        "reason": "watching",
        "best": {"iteration": 1, "win_rate": 0.301},
    }
    assert watcher.apply_forced_iteration_boundary(watching, 3) is watching
    completed = {**watching, "latest_iteration": 3}
    forced = watcher.apply_forced_iteration_boundary(completed, 3)
    assert forced["triggered"] is True
    assert forced["reason"] == "user_forced_after_exact_iteration_3"
    assert forced["forced_after_iteration"] == 3
    assert forced["best"] == completed["best"]


def test_transition_holds_core_until_tested_guide_digest_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    for relative in watcher.SPECIALIST_BUILD_ARTIFACTS:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# tested {relative}\n", encoding="utf-8")
    guide_source = tmp_path / "poke_bot" / "alakazam_heuristics.py"
    unit_dir = tmp_path / "installed-units"
    unit_dir.mkdir()
    deployment_root = tmp_path / "specialist-deployment"
    deployment_artifacts: dict[str, str] = {}
    for relative in watcher.SPECIALIST_DEPLOYMENT_ARTIFACTS:
        source = tmp_path / relative
        target = deployment_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        deployment_artifacts[relative] = sha256(source)
    installed: dict[str, str] = {}
    for name in (
        "pokebot-pure-rl-alakazam-bootstrap.service",
        "pokebot-pure-rl-alakazam.service",
        "pokebot-deck-agnostic-transition.service",
    ):
        source = tmp_path / "deploy/systemd" / name
        target = unit_dir / name
        target.write_bytes(source.read_bytes())
        installed[name] = sha256(source)
    marker = tmp_path / "ready.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "poke_bot.alakazam_specialist_build_ready/v1",
                "status": "ready",
                "guide": {
                    "version": watcher.GUIDE_VERSION,
                    "source_sha256": sha256(guide_source),
                },
                "artifacts": {
                    relative: sha256(tmp_path / relative)
                    for relative in watcher.SPECIALIST_BUILD_ARTIFACTS
                },
                "specialist_deployment": {
                    "status": "validated",
                    "root": str(deployment_root),
                    "artifacts": deployment_artifacts,
                },
                "tests": {"status": "passed", "passed": 1},
                "systemd_verify": {
                    "status": "passed",
                    "installed_unit_digests": installed,
                },
                "runtime_canary": {
                    "status": "passed",
                    "guide_rows": 1,
                    "guide_version": watcher.GUIDE_VERSION,
                    "guide_source_sha256": sha256(guide_source),
                },
                "runtime_preflight": {"status": "passed"},
                "handoff_contract": {
                    "core_continues_during_bootstrap": False,
                    "bootstrap_physical_gpu": "RTX PRO 5000 Blackwell",
                    "device_resident_bootstrap": True,
                    "stop_core_at_exact_gate": True,
                },
                "expert_corpus": {"status": "validated"},
            }
        ),
        encoding="utf-8",
    )
    assert watcher.validate_handoff_ready(
        marker,
        installed_unit_dir=unit_dir,
        specialist_deployment_root=deployment_root,
    )["status"] == "ready"

    guide_source.write_text("# changed after test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest is stale"):
        watcher.validate_handoff_ready(
            marker,
            installed_unit_dir=unit_dir,
            specialist_deployment_root=deployment_root,
        )


def test_transition_stops_core_at_gate_before_blackwell_bootstrap() -> None:
    source = (ROOT / "scripts/watch_deck_agnostic_core_transition.py").read_text()
    bootstrap = 'systemctl("start", args.bootstrap_unit, timeout=14400)'
    stop_core = 'systemctl("stop", args.source_unit, timeout=90)'
    assert source.index(stop_core) < source.index(bootstrap)
    assert "stopping_core_at_exact_gate_for_blackwell_bootstrap" in source
    assert "training_alakazam_expert_bootstrap_blackwell_device_resident" in source


def test_bootstrap_uses_the_same_checksum_pinned_alakazam_corpus() -> None:
    unit = (
        ROOT / "deploy/systemd/pokebot-pure-rl-alakazam-bootstrap.service"
    ).read_text()
    assert "--epochs 25" in unit
    assert "--patience 5" in unit
    assert "--device-resident" in unit
    assert "Environment=CUDA_VISIBLE_DEVICES=1" in unit
    assert "MemoryMax=96G" in unit
    assert "alakazam-latest10-20260709-20260718/PROTECTED_EXPERT_CORPUS.json" in unit
