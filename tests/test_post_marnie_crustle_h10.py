from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_crustle_chain_is_exactly_after_marnie_and_before_population() -> None:
    state = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())
    phase = state["post_fleet_refresh"]
    assert phase["ordered_specialist_ids"] == [
        "alakazam",
        "marnie-s-grimmsnarl-ex",
        "crustle",
    ]
    crustle = next(row for row in state["specialists"] if row["id"] == "crustle")
    assert crustle["required_specialist"] is False
    assert crustle["post_fleet_specialist_required"] is True
    assert crustle["model_contract"]["capacity_profile"] == "H10-I/v1"
    assert crustle["model_contract"]["decision_fusion_schema"] == (
        "poke_bot.causal_decision_fusion/v3"
    )
    assert crustle["public_practice_gate_opponent"]["inference_only"] is True


def test_crustle_v2_corpus_is_fail_closed_before_checksum_reaudit() -> None:
    guide = yaml.safe_load((ROOT / "config/deck_guides/crustle.yaml").read_text())
    state = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())
    crustle = next(row for row in state["specialists"] if row["id"] == "crustle")
    rebind = crustle["heads"]["current_deck_guide"][
        "public_guide_corpus_pipeline"
    ]["v2_rebind"]
    assert guide["guide_version"] == "crustle-north-star-v2"
    assert rebind["guide_version"] == "crustle-north-star-v2"
    assert rebind["training_authority_before_validation"] is False
    assert rebind["output_root"].endswith("crustle-guide-corpus-family-full33-v2")


def test_managed_units_preserve_96_workers_and_terminal_chain() -> None:
    units = ROOT / "deploy/systemd"
    marnie_completion = (
        units / "pokebot-final-format-marnie-r104-completion.service"
    ).read_text()
    latest20_override = (
        units
        / "pokebot-final-format-marnie-r104-h10-rl.service.d"
        / "zz-latest20-r109.conf"
    ).read_text()
    bootstrap = (units / "pokebot-final-format-crustle-r113-h10-bootstrap.service").read_text()
    register = (units / "pokebot-final-format-crustle-r113-h10-register.service").read_text()
    trainer = (units / "pokebot-final-format-crustle-r113-h10-rl.service").read_text()
    completion = (units / "pokebot-final-format-crustle-r113-completion.service").read_text()
    capacity = (units / "pokebot-post-refresh-capacity-boundary-r104.service").read_text()
    assert "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service" in marnie_completion
    assert "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service" in latest20_override
    assert "--next-service pokebot-post-refresh-capacity-boundary-r104.service" not in latest20_override
    assert "ConditionPathExists=/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-completion-v1.json" in bootstrap
    assert "OnSuccess=pokebot-final-format-crustle-r113-h10-register.service" in bootstrap
    assert "OnSuccess=pokebot-final-format-crustle-r113-h10-rl.service" in register
    assert "specialist_runtime_registry_h10_r113_iter20_v3.json" in register
    assert "PURE_RL_SIM_WORKERS=96" in trainer
    assert "PURE_RL_GAMES_IN_FLIGHT=96" in trainer
    assert "POKEBOT_LIVE_POOL_MAX_WORKERS=96" in trainer
    assert "OnSuccess=pokebot-final-format-crustle-r113-h10-gate-handler.service" in trainer
    assert "pokebot-post-refresh-capacity-boundary-r104.service" in completion
    assert "--crustle-completion" in capacity


def test_marnie_iteration20_boundary_cannot_terminate_at_iteration5() -> None:
    stage = (ROOT / "scripts/stage_marnie_iteration20_r113.py").read_text()
    watcher = (
        ROOT
        / "deploy/systemd/pokebot-marnie-opponent-tiers-r111-activate.service"
    ).read_text()
    trainer = (
        ROOT
        / "deploy/systemd/pokebot-final-format-marnie-r104-h10-rl.service"
    ).read_text()
    assert 'candidate["minimum_terminal_iteration"] = 20' in stage
    assert (
        'candidate["specialists"][SPECIALIST_ID]'
        '["minimum_terminal_iteration"] = 20'
    ) in stage
    assert "iteration20-stage-r113-v3.json" in watcher
    assert "registry_h10_r113_iter20_v3.json" in watcher
    assert 'isolated["self_play_games_per_iteration"] = 1024' in stage
    assert 'isolated["public_opponent_games_per_iteration"] = 7168' in stage
    assert 'isolated["premium_skill_weighted_win_rate"] = 0.80' in stage
    assert 'isolated["premium_skill_weighted_confidence_lower"] = 0.50' in stage
    assert "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service" in trainer


def test_bootstrap_retains_parent_h10_combo_route_without_slowking_authority() -> None:
    source = (ROOT / "scripts/run_post_marnie_crustle_h10_bootstrap.py").read_text()
    assert '"--allow-h10-specialist-parent"' in source
    assert '"--retain-inherited-h10-combo-state-head"' in source
    assert "include_combo_state=True" in source
    assert "--combo-state-implementation-receipt" not in source
