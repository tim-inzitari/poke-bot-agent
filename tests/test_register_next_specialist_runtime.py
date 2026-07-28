from __future__ import annotations

import json
from pathlib import Path

from poke_bot.matchup_adapters import EXPERT_IDS
from scripts import register_next_specialist_runtime as register


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_registration_is_single_source_and_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "family"
    model = _json(family / "model.pt", {"model": True})
    _json(family / "manifest.json", {"checkpoint": str(model)})
    expert = _json(
        tmp_path / "expert.json",
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "totals": {"decisions_kept": 173_490},
        },
    )
    targets = list(EXPERT_IDS)
    tree = _json(
        tmp_path / "tree.json",
        {
            "runtime_enabled": True,
            "targets": targets,
            "runtime_contract": {
                "accepted_archetype_ids": targets,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    registry = _json(
        tmp_path / "registry.json",
        {
            "schema": "poke_bot.specialist_runtime_registry/v1",
            "runtime_root": str(tmp_path),
            "specialists": {},
        },
    )
    selector = tmp_path / "specialist.env"
    selector.write_text(
        "# canonical\nPOKEBOT_ACTIVE_SPECIALIST=starmie\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        register,
        "verify_frozen_model",
        lambda _family: {
            "model_path": str(model),
            "checkpoint_digest": "sha256:" + "a" * 64,
        },
    )

    kwargs = {
        "specialist_id": "lucario",
        "family": family,
        "expert": expert,
        "runtime_tree": tree,
        "runtime_registry": registry,
        "selector_env": selector,
        "state_root": tmp_path / "state",
        "run_name": "pure_rl_lucario_temporal1_8k_v1",
        "handoff_service": "pokebot-specialist-cycle-handoff.service",
    }
    first = register.register(**kwargs)
    second = register.register(**kwargs)

    assert first["identity_sha256"] == second["identity_sha256"]
    assert selector.read_text(encoding="utf-8").count(
        "POKEBOT_ACTIVE_SPECIALIST="
    ) == 1
    assert "POKEBOT_ACTIVE_SPECIALIST=lucario" in selector.read_text(
        encoding="utf-8"
    )
    assert f"POKEBOT_SPECIALIST_RUNTIME_ROOT={tmp_path}" in selector.read_text(
        encoding="utf-8"
    )
    assert f"PYTHONPATH={tmp_path}" in selector.read_text(encoding="utf-8")
    assert "POKEBOT_EXPANDED_HEADS_ENABLED=1" in selector.read_text(
        encoding="utf-8"
    )
    assert "POKEBOT_DECISION_FUSION_ENABLED=1" in selector.read_text(
        encoding="utf-8"
    )
    assert "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=1" in selector.read_text(
        encoding="utf-8"
    )
    row = json.loads(registry.read_text(encoding="utf-8"))["specialists"][
        "lucario"
    ]
    assert row["status"] == "ready"
    assert row["measurement_decks"] == "lucario"
    assert row["decision_fusion"]["required"] is True
    assert row["decision_fusion"]["runtime_enabled"] is True
    assert len(row["decision_fusion"]["required_heads"]) == 17
    assert row["pass_handler"]["handoff_service"] == (
        "pokebot-specialist-cycle-handoff.service"
    )
    authorization = json.loads(
        Path(row["matchup_adapter_authorization"]).read_text(encoding="utf-8")
    )
    assert authorization["optimizer_scope"] == "matchup_adapter_bank_only"
    assert authorization["runtime_enabled"] is False

    corrected = register.register(
        **{
            **kwargs,
            "run_name": "pure_rl_lucario_temporal1_8k_corrected_v2",
            "authorization_name": (
                "lucario-matchup-adapter-bootstrap-corrected-v2.json"
            ),
            "replace_unpassed": True,
        }
    )
    corrected_row = corrected["runtime_row"]
    assert corrected_row["run_name"].endswith("corrected_v2")
    assert corrected_row["matchup_adapter_authorization"].endswith(
        "corrected-v2.json"
    )


def test_registration_honors_specialist_specific_corpus_floor(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "family"
    model = _json(family / "model.pt", {"model": True})
    _json(family / "manifest.json", {"checkpoint": str(model)})
    expert = _json(
        tmp_path / "expert.json",
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "totals": {"decisions_kept": 10_946},
        },
    )
    targets = list(EXPERT_IDS)
    tree = _json(
        tmp_path / "tree.json",
        {
            "runtime_enabled": True,
            "targets": targets,
            "runtime_contract": {
                "accepted_archetype_ids": targets,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    registry = _json(
        tmp_path / "registry.json",
        {
            "schema": "poke_bot.specialist_runtime_registry/v1",
            "runtime_root": str(tmp_path),
            "specialists": {},
        },
    )
    selector = tmp_path / "specialist.env"
    selector.write_text(
        "POKEBOT_ACTIVE_SPECIALIST=lucario\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        register,
        "verify_frozen_model",
        lambda _family: {
            "model_path": str(model),
            "checkpoint_digest": "sha256:" + "b" * 64,
        },
    )

    receipt = register.register(
        specialist_id="dragapult-dusknoir",
        family=family,
        expert=expert,
        runtime_tree=tree,
        runtime_registry=registry,
        selector_env=selector,
        state_root=tmp_path / "state",
        run_name="pure_rl_dragapult-dusknoir_temporal1_8k_v1",
        handoff_service="pokebot-specialist-cycle-handoff.service",
        minimum_decisions=10_000,
        required_target_coverage=(
            "temporal_action_rows",
            "opponent_remainder_rows",
            "lethal_threat_rows",
            "prize_race_rows",
        ),
    )

    assert receipt["specialist_id"] == "dragapult-dusknoir"
    assert receipt["runtime_row"]["expert_minimum_decisions"] == 10_000
    assert receipt["runtime_row"]["expert_required_target_coverage"] == [
        "temporal_action_rows",
        "opponent_remainder_rows",
        "lethal_threat_rows",
        "prize_race_rows",
    ]


def test_registration_binds_successor_guide_to_runtime_row(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "family"
    model = _json(family / "model.pt", {"model": True})
    _json(family / "manifest.json", {"checkpoint": str(model)})
    expert = _json(
        tmp_path / "expert.json",
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "totals": {"decisions_kept": 20_000},
        },
    )
    targets = list(EXPERT_IDS)
    tree = _json(
        tmp_path / "tree.json",
        {
            "runtime_enabled": True,
            "targets": targets,
            "runtime_contract": {
                "accepted_archetype_ids": targets,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    registry = _json(
        tmp_path / "registry.json",
        {
            "schema": "poke_bot.specialist_runtime_registry/v1",
            "runtime_root": str(tmp_path),
            "specialists": {},
        },
    )
    selector = tmp_path / "specialist.env"
    selector.write_text("", encoding="utf-8")
    guide = tmp_path / "grimmsnarl.yaml"
    guide.write_text("schema: guide\n", encoding="utf-8")
    monkeypatch.setattr(
        register,
        "verify_frozen_model",
        lambda _family: {
            "model_path": str(model),
            "checkpoint_digest": "sha256:" + "c" * 64,
        },
    )

    receipt = register.register(
        specialist_id="marnie-s-grimmsnarl-ex",
        family=family,
        expert=expert,
        runtime_tree=tree,
        runtime_registry=registry,
        selector_env=selector,
        state_root=tmp_path / "state",
        run_name="pure_rl_grimmsnarl",
        handoff_service="pokebot-specialist-cycle-handoff.service",
        guide_id="marnie-s-grimmsnarl-ex",
        guide_loss_weight=0.05,
        guide_contract=guide,
        guide_contract_sha256=register.sha256(guide),
        guide_version="marnie-grimmsnarl-north-star-v1",
    )

    row = receipt["runtime_row"]
    assert row["guide_id"] == "marnie-s-grimmsnarl-ex"
    assert row["guide_loss_weight"] == 0.05
    assert row["guide_contract"] == str(guide.resolve())
    assert row["guide_contract_sha256"] == register.sha256(guide).removeprefix(
        "sha256:"
    )
