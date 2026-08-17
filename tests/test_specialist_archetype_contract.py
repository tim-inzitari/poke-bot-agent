from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from scripts.train_pure_rl import (
    LADDER_DECK_REPRESENTATIVES_PATH,
    _our_decks,
    _parse_args,
)
from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT,
    SLOT_CAPACITY,
    load_slot_registry,
    registry_digest,
)


def test_production_alakazam_unit_is_temporal_8k() -> None:
    unit = (
        Path(__file__).resolve().parents[1]
        / "deploy/systemd/pokebot-pure-rl-alakazam.service"
    ).read_text(encoding="utf-8")
    assert (
        "WorkingDirectory=/home/pokebot/poke-bot-agent-deployments/pure-rl-resident-v7"
        in unit
    )
    assert "Environment=PURE_RL_DECISION_CONTEXT=history" in unit
    assert "Environment=PURE_RL_TEMPORAL_LAYERS=1" in unit
    assert unit.count("--games-per-iter 8192") == 1
    assert unit.count("--train-max-decisions-per-batch 8192") == 1
    assert "--official-exploit-frac 1.00" in unit
    assert "Environment=PURE_RL_OFFICIAL_ADAPTIVE_TARGETING=1" in unit
    assert "Environment=PURE_RL_OFFICIAL_ADAPTIVE_MIN_SHARE=0.05" in unit
    assert "Environment=PURE_RL_OFFICIAL_ADAPTIVE_GAP_POWER=2.0" in unit
    assert "--initial-learner-checkpoint" in unit
    assert "--base-checkpoint" not in unit
    assert "--train-device-resident" not in unit
    assert "--allow-clean-boundary-design-migration" not in unit
    assert "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE" not in unit


def test_specialist_requires_an_explicit_archetype() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--run-name", "unsafe", "--mode", "specialist"])
    parsed = _parse_args(
        [
            "--run-name",
            "alakazam-safe",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "AlAkAzAm",
            "--official-collect-frac",
            "0.50",
        ]
    )
    assert parsed.specialist_archetype == "alakazam"
    assert parsed.official_collect_frac == 0.50
    assert parsed.archetype_aux_loss_weight > 0.0
    assert parsed.opp_hand_loss_weight > 0.0
    assert parsed.opp_remainder_loss_weight > 0.0
    assert parsed.lethal_threat_loss_weight > 0.0
    assert parsed.prize_race_loss_weight > 0.0


def test_router_format_6_checkpoint_registers_teal_for_adapter_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = load_slot_registry()
    checkpoint = tmp_path / "router-format-6.pt"
    torch.save(
        {
            "extra": {
                "matchup_adapter_config": {
                    "format": ADAPTER_CHECKPOINT_FORMAT,
                    "slot_capacity": SLOT_CAPACITY,
                    "slot_registry_digest": registry_digest(registry),
                    "slot_registry": registry,
                }
            }
        },
        checkpoint,
    )

    parsed = _parse_args(
        [
            "--run-name",
            "teal-router-format-6",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "teal-mask-ogerpon-ex",
            "--initial-learner-checkpoint",
            str(checkpoint),
            "--dormant-matchup-adapter-epochs",
            "1",
            "--dormant-matchup-adapter-activation-receipt",
            str(tmp_path / "authorization.json"),
            "--official-collect-frac",
            "0.50",
        ]
    )

    assert parsed.specialist_archetype == "teal-mask-ogerpon-ex"

    registry_path = (
        Path(__file__).resolve().parents[1]
        / "state/matchup_adapter_roster.json"
    )
    monkeypatch.setenv(
        "POKEBOT_MATCHUP_ADAPTER_FORMAT",
        "poke-bot-matchup-adapter-bank-v6",
    )
    monkeypatch.setenv(
        "POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH",
        str(registry_path),
    )
    resumed = _parse_args(
        [
            "--run-name",
            "teal-router-format-6-resume",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "teal-mask-ogerpon-ex",
            "--dormant-matchup-adapter-epochs",
            "1",
            "--dormant-matchup-adapter-activation-receipt",
            str(tmp_path / "authorization.json"),
            "--official-collect-frac",
            "0.50",
        ]
    )
    assert resumed.initial_learner_checkpoint is None


def test_specialist_accepts_audited_official_exploit_mix() -> None:
    parsed = _parse_args(
        [
            "--run-name",
            "iono-exploit",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "alakazam",
            "--official-collect-frac",
            "0.50",
            "--official-exploit-opponents",
            "iono",
            "--official-exploit-frac",
            "0.50",
            "--official-exploit-temperature",
            "0.35",
        ]
    )
    assert parsed.official_exploit_opponents == ("iono",)
    assert parsed.official_exploit_frac == pytest.approx(0.50)
    assert parsed.official_exploit_temperature == pytest.approx(0.35)


def test_specialist_accepts_adaptive_official_targeting_only_with_training_mix() -> None:
    parsed = _parse_args(
        [
            "--run-name",
            "adaptive-official",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "alakazam",
            "--official-collect-frac",
            "0.50",
            "--official-adaptive-targeting",
            "--official-adaptive-min-share",
            "0.05",
            "--official-adaptive-gap-power",
            "2.0",
        ]
    )
    assert parsed.official_adaptive_targeting is True
    assert parsed.official_adaptive_min_share == pytest.approx(0.05)
    assert parsed.official_adaptive_gap_power == pytest.approx(2.0)
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "adaptive-without-training",
                "--mode",
                "specialist",
                "--specialist-archetype",
                "alakazam",
                "--official-adaptive-targeting",
            ]
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--official-exploit-opponents", "not-official", "--official-exploit-frac", "0.5"],
        ["--official-exploit-opponents", "iono", "--official-exploit-frac", "0"],
        ["--official-exploit-frac", "0.5"],
    ],
)
def test_official_exploit_rejects_ambiguous_or_invalid_configs(
    extra: list[str],
) -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "bad-exploit",
                "--mode",
                "specialist",
                "--specialist-archetype",
                "alakazam",
                "--official-collect-frac",
                "0.5",
                *extra,
            ]
        )


def test_core_curriculum_defaults_train_every_shared_head() -> None:
    parsed = _parse_args(["--run-name", "all-head-core", "--mode", "core"])
    assert parsed.archetype_aux_loss_weight == pytest.approx(0.05)
    assert parsed.opp_hand_loss_weight == pytest.approx(0.05)
    assert parsed.opp_remainder_loss_weight == pytest.approx(0.05)
    assert parsed.lethal_threat_loss_weight == pytest.approx(0.025)
    assert parsed.prize_race_loss_weight == pytest.approx(0.025)


def test_expert_rehearsal_threads_every_cli_head_weight() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/train_pure_rl.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "supervised_rehearsal_step"
    ]
    assert len(calls) == 1
    keyword_sources = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }
    assert keyword_sources == {
        **keyword_sources,
        "aux_loss_weight": "float(args.archetype_aux_loss_weight)",
        "opp_hand_loss_weight": "float(args.opp_hand_loss_weight)",
        "opp_remainder_loss_weight": "float(args.opp_remainder_loss_weight)",
        "lethal_threat_loss_weight": "float(args.lethal_threat_loss_weight)",
        "prize_race_loss_weight": "float(args.prize_race_loss_weight)",
        "alakazam_guide_loss_weight": "float(args.alakazam_guide_loss_weight)",
    }
    prepare_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "expert_cache"
        and node.func.attr == "prepare"
    ]
    assert len(prepare_calls) == 1
    prepare_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in prepare_calls[0].keywords
        if keyword.arg is not None
    }
    assert (
        prepare_keywords.get("belief_card_vocab")
        == "rehearsal_belief_card_vocab"
    )
    source_text = (Path(__file__).resolve().parents[1] / "scripts/train_pure_rl.py").read_text(
        encoding="utf-8"
    )
    assert "checkpoint_mod.load_checkpoint(" in source_text
    assert 'parent_checkpoint.get("model_state_dict")' in source_text
    assert 'trusted_parent.get("model_state_dict")' not in source_text


def test_curriculum_rejects_negative_head_weight() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "invalid-head",
                "--opp-hand-loss-weight",
                "-0.1",
            ]
        )


def test_alakazam_specialist_uses_the_exact_pinned_ladder_list() -> None:
    artifact = json.loads(LADDER_DECK_REPRESENTATIVES_PATH.read_text())
    expected = artifact["decks"]["alakazam"]["card_ids"]
    assert _our_decks("specialist", "alakazam") == [("alakazam", expected)]


def test_core_rejects_specialist_archetype_argument() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "unsafe-core",
                "--mode",
                "core",
                "--specialist-archetype",
                "alakazam",
            ]
        )


def test_core_cannot_train_on_its_formal_holdout_policies() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "unsafe-core-targeting",
                "--mode",
                "core",
                "--official-collect-frac",
                "0.5",
            ]
        )


def test_alakazam_guide_requires_specialist_and_explicit_target_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", raising=False)
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "guide-without-targets",
                "--mode",
                "specialist",
                "--specialist-archetype",
                "alakazam",
                "--alakazam-guide-loss-weight",
                "0.05",
            ]
        )
    monkeypatch.setenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "1")
    parsed = _parse_args(
        [
            "--run-name",
            "guided-alakazam",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "alakazam",
            "--alakazam-guide-loss-weight",
            "0.05",
        ]
    )
    assert parsed.alakazam_guide_loss_weight == pytest.approx(0.05)
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "guided-core",
                "--alakazam-guide-loss-weight",
                "0.05",
            ]
        )
