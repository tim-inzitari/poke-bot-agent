from __future__ import annotations

import json
from pathlib import Path

from scripts.run_specialist_cycle_handoff import (
    _compatible_prior_cumulative_contract,
    _cumulative_core_contract,
    _generated,
    _required_specialist_ids,
    population_transition_ready,
)
from scripts.run_post_starmie_core_handoff import (
    _reusable_core_candidate,
    _reusable_core_regression,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cycle_contract_keeps_exact_protocol_for_later_specialists() -> None:
    contract = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = {
        "id": "lucario",
        "run_dir": "/runs/lucario",
        "training_service": "pokebot-pure-rl-trevenant-staged.service",
        "gate_contract": "/runtime/ops/gate.json",
        "gate_marker_name": "SPECIALIST_GATE_PASSED.lucario-splus-v1",
        "minimum_completed_iteration": 5,
        "matchup_runtime_tree": "/state/tree.json",
        "passed_family": "/models/lucario",
        "handler_state": "/state/lucario-handler.json",
    }
    result = _generated(
        contract=contract,
        source=source,
        selected={
            "specialist_id": "dragapult",
            "pointer": "/corpora/dragapult/PROTECTED_EXPERT_CORPUS.json",
        },
        core_digest="sha256:" + "b" * 64,
    )

    assert result["source_specialist"] == source
    assert result["next_specialist"]["id"] == "dragapult"
    assert result["training"]["supervised_epochs"] == 25
    assert result["training"]["minimum_decisions"] == 20000
    assert result["submission_policy"]["completion_blocks_handoff"] is False
    assert result["runtime_registration"]["handoff_service"] == (
        "pokebot-specialist-cycle-handoff.service"
    )
    assert result["runtime_registration"]["gate_handler_service"] == (
        "pokebot-specialist-passed-gate-handler.service"
    )
    assert result["paths"]["state"].endswith(
        "post-lucario-dragapult-handoff-v1.json"
    )


def test_cycle_contract_has_explicit_population_terminal_handoff() -> None:
    contract = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["runtime"]["population_handoff_service"] == (
        "pokebot-population-round-robin-handoff.service"
    )
    assert contract["runtime"]["population_training_service"] == (
        "pokebot-population-round-robin.service"
    )
    assert contract["selection"]["corpus_root"].endswith(
        "expert-evidence28-20260626-20260723/specialist-corpora-v3-full-head"
    )
    assert contract["runtime"]["inactive_tree_candidate"].endswith(
        "public-matchup-tree-calibration-v37.inactive.json"
    )
    assert contract["runtime"]["candidate_audit"].endswith(
        "public-matchup-tree-calibration-v37.audit.json"
    )
    assert contract["selection"]["minimum_decisions_by_specialist"] == {
        "dragapult-dusknoir": 10000,
        "dudunsparce": 2000,
    }
    assert contract["selection"]["strict_priority_prefix"] == [
        "dragapult-dusknoir",
        "dudunsparce",
    ]


def test_lucario_runtime_row_exposes_exact_threshold_transition() -> None:
    registry = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    lucario = registry["specialists"]["lucario"]

    assert lucario["status"] == "ready"
    assert lucario["minimum_terminal_iteration"] == 10
    assert lucario["pass_handler"]["threshold_transition_receipt"] == (
        "/home/inzi/poke-bot-agent/outputs/state/"
        "lucario-gate-floor15-transition-v1.json"
    )


def test_accepted_core_is_reused_without_rematerialization(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "models" / "core-v4"
    family.mkdir(parents=True)
    ready_path = tmp_path / "core-v4-ready.json"
    digest = "sha256:" + "a" * 64
    teacher_digests = ["sha256:" + "b" * 64, "sha256:" + "c" * 64]
    ready_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.multi_teacher_core_ready/v1",
                "status": "ready",
                "gameplay_regression_passed": True,
                "checkpoint_digest": digest,
                "teacher_checkpoint_digests": teacher_digests,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.run_post_starmie_core_handoff.verify_frozen_model",
        lambda path: {
            "family": Path(path).name,
            "model_path": str(Path(path) / "model.pt"),
            "checkpoint_digest": digest,
        },
    )
    result = _reusable_core_candidate(
        contract={
            "core_refresh": {
                "family": str(family),
                "ready_receipt": str(ready_path),
                "teachers": [
                    {"checksum": teacher_digests[0]},
                    {"checksum": teacher_digests[1]},
                ],
            }
        }
    )
    assert result is not None
    assert result[0]["status"] == "ready"
    assert result[1]["checkpoint_digest"] == digest


def test_prior_cumulative_contract_accepts_only_additive_selection_controls() -> None:
    current = {
        "trigger": {
            "specialist_id": "lucario",
            "threshold_transition_receipt": "/state/transition.json",
        },
        "next_specialist": {
            "minimum_decisions": 20_000,
            "minimum_decisions_by_specialist": {
                "dragapult-dusknoir": 10_000
            },
            "strict_priority_prefix": ["dragapult-dusknoir"],
        },
    }
    prior = json.loads(json.dumps(current))
    prior["trigger"].pop("threshold_transition_receipt")
    prior["next_specialist"].pop("minimum_decisions_by_specialist")
    prior["next_specialist"].pop("strict_priority_prefix")
    assert _compatible_prior_cumulative_contract(prior, current)
    prior["next_specialist"]["minimum_decisions"] = 9_999
    assert not _compatible_prior_cumulative_contract(prior, current)


def test_accepted_core_regression_survives_controller_only_change(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "d" * 64
    teachers = ["sha256:" + "e" * 64]
    path = tmp_path / "regression.json"
    path.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.multi_teacher_core_gameplay_regression/v1"
                ),
                "passed": True,
                "identity": {
                    "contract_digest": "sha256:" + "0" * 64,
                    "candidate": {"digest": digest},
                    "teacher_checkpoint_digests": teachers,
                },
            }
        ),
        encoding="utf-8",
    )
    result = _reusable_core_regression(
        path=path,
        candidate_digest=digest,
        teacher_digests=teachers,
    )
    assert result is not None
    assert result["passed"] is True


def test_each_future_specialist_gets_a_new_cumulative_core(
    tmp_path: Path, monkeypatch
) -> None:
    template = json.loads(
        (ROOT / "ops/post_starmie_core_v2_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cycle = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cycle["runtime"]["registry_root"] = str(tmp_path / "models")
    cycle["runtime"]["state_root"] = str(tmp_path / "state")
    cycle["selection"]["state"] = str(tmp_path / "specialists.yaml")
    cycle["selection"]["corpus_root"] = str(tmp_path / "corpora")
    specialists = {
        specialist_id: {
            "pass_handler": {"family": f"{specialist_id}-pass-v1"}
        }
        for specialist_id in ("alakazam", "hops-trevenant", "starmie", "lucario")
    }
    specialists["alakazam"] = {}

    def frozen(path: Path) -> dict[str, str]:
        family = Path(path).name
        return {
            "family": family,
            "model_path": str(Path(path) / "model.pt"),
            "checkpoint_digest": "sha256:" + family.encode().hex().ljust(64, "0")[:64],
        }

    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff.verify_frozen_model", frozen
    )
    result = _cumulative_core_contract(
        template=template,
        cycle=cycle,
        source={
            "id": "lucario",
            "run_dir": "/runs/lucario",
            "training_service": "trainer.service",
            "gate_contract": "/runtime/gate.json",
            "gate_marker_name": "SPECIALIST_GATE_PASSED.lucario",
            "minimum_completed_iteration": 5,
            "passed_family": "/models/lucario-pass-v1",
            "handler_state": "/state/lucario-handler.json",
        },
        completed_ids={"alakazam", "hops-trevenant", "starmie", "lucario"},
        runtime_registry={"specialists": specialists},
        current_core={
            "model_path": "/models/deck-core-v2/model.pt",
            "checkpoint_digest": "sha256:" + "c" * 64,
        },
    )
    assert result["core_refresh"]["version"] == 4
    assert [
        row["specialist_id"] for row in result["core_refresh"]["teachers"]
    ] == ["alakazam", "hops-trevenant", "lucario", "starmie"]
    assert result["core_refresh"]["direct_checkpoint_tensor_sources_exclude"] == []
    assert result["core_refresh"]["initialization"]["checkpoint"] == (
        "/models/deck-core-v2/model.pt"
    )
    assert result["next_specialist"]["hot_start_core_version"] == 4
    assert result["trigger"]["minimum_completed_iteration"] == 5


def test_required_specialist_ids_is_exact_canonical_roster() -> None:
    identifiers = _required_specialist_ids(ROOT / "state/specialists.yaml")
    assert len(identifiers) == 22
    assert "starmie" in identifiers
    assert "hops-trevenant" in identifiers


def test_starmie_pass_still_leaves_nineteen_before_population() -> None:
    required = _required_specialist_ids(ROOT / "state/specialists.yaml")
    completed_after_starmie = {
        "alakazam",
        "hops-trevenant",
        "starmie",
    }
    assert len(required - completed_after_starmie) == 19
    assert not population_transition_ready(completed_after_starmie, required)


def test_population_requires_all_twenty_two_specialists() -> None:
    required = _required_specialist_ids(ROOT / "state/specialists.yaml")
    assert population_transition_ready(set(required), required)
    assert not population_transition_ready(
        set(sorted(required)[:-1]),
        required,
    )
