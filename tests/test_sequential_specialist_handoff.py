from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_sequential_specialist_handoff as handoff
from scripts.run_starmie_expert_bootstrap import (
    decision_fusion_handoff_contract,
    expanded_handoff_training_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ops/post_trevenant_starmie_handoff_v1.json"


def test_contract_locks_exact_protocol_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handoff, "path_value", lambda *args: ROOT / "README.md")
    contract, digest = handoff.load_contract(CONTRACT)
    assert digest.startswith("sha256:")
    assert contract["source_specialist"]["id"] == "hops-trevenant"
    assert contract["source_specialist"]["minimum_completed_iteration"] == 5
    assert contract["next_specialist"]["id"] == "starmie"
    assert contract["training"] == {
        "supervised_epochs": 25,
        "patience_diagnostic_only": 5,
        "minimum_decisions": 100000,
        "requested_decisions_per_batch": 12288,
    }
    assert contract["submission_policy"] == {
        "required_copies": 1,
        "completion_blocks_handoff": False,
        "queue_order": "oldest_first",
    }


def test_contract_rejects_non_exact_bootstrap_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["training"]["supervised_epochs"] = 24
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(handoff, "path_value", lambda *args: ROOT / "README.md")
    with pytest.raises(RuntimeError, match="contract changed"):
        handoff.load_contract(changed)


def test_expanded_handoff_passes_checksum_pinned_schedule_to_bootstrap() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    expanded = expanded_handoff_training_contract()
    contract["training"]["expanded_heads"] = expanded
    contract["training"]["decision_fusion"] = decision_fusion_handoff_contract()

    command = handoff.bootstrap_command(contract)

    assert "--expanded-heads" in command
    assert command[
        command.index("--expected-expanded-schedule-digest") + 1
    ] == expanded["schedule_digest"]
    assert command[
        command.index("--expected-expanded-target-digest") + 1
    ] == expanded["target_schema_digest"]


def test_successor_with_expanded_heads_requires_decision_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["training"]["expanded_heads"] = expanded_handoff_training_contract()
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(handoff, "path_value", lambda *args: ROOT / "README.md")

    with pytest.raises(RuntimeError, match="successor lacks"):
        handoff.load_contract(changed)


def test_contract_rejects_weakened_source_iteration_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["source_specialist"]["minimum_completed_iteration"] = 2
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(handoff, "path_value", lambda *args: ROOT / "README.md")
    with pytest.raises(RuntimeError, match="contract changed"):
        handoff.load_contract(changed)


def test_runtime_tree_target_count_derives_from_canonical_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handoff, "EXPERT_IDS", ("one", "two", "three"))
    contract = {"runtime_registration": {"matchup_target_ids": ["one", "two", "three"]}}
    assert handoff.canonical_matchup_target_ids(contract) == (
        "one",
        "two",
        "three",
    )

    contract["runtime_registration"]["matchup_target_ids"] = ["one", "two"]
    with pytest.raises(RuntimeError, match="differs from the canonical"):
        handoff.canonical_matchup_target_ids(contract)

    source = (
        ROOT / "scripts/run_sequential_specialist_handoff.py"
    ).read_text(encoding="utf-8")
    assert 'audit.get("target_count") or 0) != 22' not in source
    assert "len(canonical_matchup_target_ids(contract))" in source


def test_runtime_tree_target_count_uses_the_v6_slot_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = (*handoff.EXPERT_IDS, "teal-mask-ogerpon-ex")
    monkeypatch.setattr(
        handoff,
        "load_slot_registry",
        lambda path: {"active_expert_ids": list(targets)},
    )
    contract = {
        "runtime_registration": {
            "matchup_target_ids": list(targets),
            "matchup_v6": {
                "enabled": True,
                "registry": "/absolute/state/matchup_adapter_roster.json",
            },
        }
    }

    assert handoff.canonical_matchup_target_ids(contract) == targets
    contract["runtime_registration"]["matchup_target_ids"] = list(
        handoff.EXPERT_IDS
    )
    with pytest.raises(RuntimeError, match="canonical adapter registry"):
        handoff.canonical_matchup_target_ids(contract)


def test_frozen_predecessor_registry_preserves_source_gate_s_plus_rows() -> None:
    evidence = handoff.validate_frozen_predecessor_registry(
        {
            "gate": {
                "roster_ids": [
                    "public-one",
                    "specialist-alakazam-owner-accepted-iter39",
                    "specialist-hops-trevenant-gate-iter10-462f201f8de6",
                ]
            }
        },
        {
            "schema": "poke_bot.frozen_specialist_registry/v1",
            "specialists": [
                {
                    "specialist_id": "alakazam",
                    "opponent_id": "specialist-alakazam-owner-accepted-iter39",
                    "frozen": True,
                },
                {
                    "specialist_id": "hops-trevenant",
                    "opponent_id": (
                        "specialist-hops-trevenant-gate-iter10-462f201f8de6"
                    ),
                    "frozen": True,
                },
            ],
        },
    )
    assert evidence["required_count"] == 2


def test_frozen_predecessor_registry_rejects_dropped_source_gate_row() -> None:
    with pytest.raises(RuntimeError, match="dropped source-gate S\\+"):
        handoff.validate_frozen_predecessor_registry(
            {
                "gate": {
                    "roster_ids": [
                        "public-one",
                        "specialist-alakazam-owner-accepted-iter39",
                        "specialist-hops-trevenant-gate-iter10-462f201f8de6",
                    ]
                }
            },
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "specialist_id": "hops-trevenant",
                        "opponent_id": (
                            "specialist-hops-trevenant-gate-iter10-462f201f8de6"
                        ),
                        "frozen": True,
                    }
                ],
            },
        )


def test_saved_preflight_source_survives_later_gate_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = tmp_path / "frozen-source"
    family.mkdir()
    manifest = family / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    handler_path = tmp_path / "handler.json"
    state_path = tmp_path / "handoff-state.json"
    digest = "sha256:" + "4" * 64
    frozen = {
        "family": "source",
        "model_path": str(family / "model.pt"),
        "checkpoint_digest": digest,
    }
    plan = {
        "commit_boundary": 10,
        "checkpoint_digest": digest,
        "validation": {"committed": True, "audit_passed": True},
    }
    queued = {
        "copy_number": 1,
        "label": "source copy 1",
        "checkpoint_checksum": digest,
        "queued_at": "2026-07-24T00:00:00+00:00",
    }
    handler_path.write_text(
        json.dumps(
            {
                "schema": handoff.HANDLER_SCHEMA,
                "phase": "submissions_queued",
                "submission_mode": "queue_and_continue",
                "gate": plan,
                "frozen_model": frozen,
                "queued_submissions": [queued],
            }
        ),
        encoding="utf-8",
    )
    saved = {
        "specialist_id": "starmie",
        "gate": plan,
        "frozen_family": str(family.resolve()),
        "frozen_manifest_sha256": handoff.sha256(manifest),
        "checkpoint_digest": digest,
        "queued_submission_copies": [queued],
    }
    contract_digest = "sha256:" + "5" * 64
    state_path.write_text(
        json.dumps(
            {
                "schema": handoff.STATE_SCHEMA,
                "contract_sha256": contract_digest,
                "phase": "preflight_verified",
                "source_specialist": saved,
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "source_specialist": {
            "id": "starmie",
            "passed_family": str(family),
            "handler_state": str(handler_path),
            "minimum_completed_iteration": 5,
        },
        "paths": {"state": str(state_path)},
    }
    monkeypatch.setattr(handoff, "verify_frozen_model", lambda _path: frozen)
    assert (
        handoff.validate_saved_preflight_source(contract, contract_digest)
        == saved
    )


def test_ceiling_plan_allows_only_checksum_identical_contract_alias(
    tmp_path: Path,
) -> None:
    first = tmp_path / "runtime-v5" / "gate.json"
    second = tmp_path / "safe-boundary" / "gate.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('{"gate":"same"}\n', encoding="utf-8")
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    digest = handoff.sha256(first)
    base = {
        "schema": "poke_bot.ceiling_acceptance_archive_plan/v1",
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "contract_sha256": digest,
        "checkpoint_digest": "sha256:" + "a" * 64,
        "commit_boundary": 15,
    }
    saved = {**base, "contract": str(first)}
    current = {**base, "contract": str(second)}

    assert handoff.compatible_ceiling_acceptance_plan(saved, current)

    second.write_text('{"gate":"changed"}\n', encoding="utf-8")
    assert not handoff.compatible_ceiling_acceptance_plan(saved, current)


def test_source_ceiling_uses_exact_fused_child_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "sha256:" + "a" * 64
    exact_receipt = tmp_path / "runtime-exact-gate.json"
    exact_receipt.write_text("{}\n", encoding="utf-8")
    plan = {
        "schema": "poke_bot.exact_pass_archive_plan/v1",
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "checkpoint_digest": digest,
        "commit_boundary": 15,
        "exact_result_pointer": str(exact_receipt),
    }
    frozen = {"checkpoint_digest": digest}
    handler_state = {
        "schema": handoff.HANDLER_SCHEMA,
        "phase": "submissions_queued",
        "submission_mode": "queue_and_continue",
        "gate": plan,
        "frozen_model": frozen,
        "queued_submissions": [
            {
                "copy_number": 1,
                "label": "dudunsparce exact fused copy 1",
                "checkpoint_checksum": digest,
                "queued_at": "2026-07-25T00:00:00+00:00",
            }
        ],
    }
    paths = {
        "handler_state": tmp_path / "handler.json",
        "run_dir": tmp_path / "run",
        "gate_contract": tmp_path / "gate.json",
        "passed_family": tmp_path / "frozen",
    }
    monkeypatch.setattr(
        handoff,
        "path_value",
        lambda _contract, _group, key: paths[key],
    )
    monkeypatch.setattr(handoff, "read_json", lambda _path: handler_state)
    monkeypatch.setattr(
        handoff,
        "validate_exact_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("no flat pass marker")
        ),
    )
    seen: list[tuple[Path, int]] = []

    def validate_runtime(
        _run: Path,
        _contract: Path,
        receipt: Path,
        *,
        accept_ceiling: bool,
        ceiling_iteration: int,
    ) -> dict:
        assert accept_ceiling is True
        seen.append((receipt, ceiling_iteration))
        return plan

    monkeypatch.setattr(handoff, "validate_runtime_exact_gate", validate_runtime)
    monkeypatch.setattr(handoff, "verify_frozen_model", lambda _path: frozen)
    monkeypatch.setattr(handoff, "sha256", lambda _path: digest)

    evidence = handoff.validate_source(
        {
            "source_specialist": {
                "id": "dudunsparce",
                "minimum_completed_iteration": 5,
                "gate_marker_name": "unused-flat-marker",
            }
        }
    )

    assert seen == [(exact_receipt, 15)]
    assert evidence["gate"] == plan


def test_next_specialist_gate_rejects_stale_source_checkpoint(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    gate = tmp_path / "gate.json"
    stale = "sha256:" + "1" * 64
    passing = "sha256:" + "2" * 64
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "specialist_id": "hops-trevenant",
                        "opponent_id": "specialist-trevenant",
                        "checkpoint_digest": stale,
                        "frozen": True,
                        "public_mix_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    gate.write_text(
        json.dumps(
            {
                "next_gate": {
                    "evaluation": {
                        "games_total": 250,
                        "games_per_opponent": 250,
                        "seat0_games_per_opponent": 125,
                        "seat1_games_per_opponent": 125,
                    },
                    "roster": [
                        {
                            "opponent_id": "specialist-trevenant",
                            "tier": "S+",
                            "frozen_specialist": True,
                            "frozen_checkpoint_digest": stale,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "next_specialist": {
            "gate_contract": str(gate),
            "frozen_specialist_registry": str(registry),
        }
    }
    with pytest.raises(RuntimeError, match="S\\+ gate identity"):
        handoff.validate_next_specialist_gate(
            contract,
            {
                "specialist_id": "hops-trevenant",
                "checkpoint_digest": passing,
            },
        )


def test_next_specialist_gate_requires_every_frozen_predecessor_at_s_plus(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    gate = tmp_path / "gate.json"
    alakazam = "sha256:" + "1" * 64
    trevenant = "sha256:" + "2" * 64
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "specialist_id": "alakazam",
                        "opponent_id": "specialist-alakazam",
                        "checkpoint_digest": alakazam,
                        "frozen": True,
                        "public_mix_eligible": True,
                    },
                    {
                        "specialist_id": "hops-trevenant",
                        "opponent_id": "specialist-trevenant",
                        "checkpoint_digest": trevenant,
                        "frozen": True,
                        "public_mix_eligible": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    gate.write_text(
        json.dumps(
            {
                "next_gate": {
                    "evaluation": {
                        "games_total": 250,
                        "games_per_opponent": 250,
                        "seat0_games_per_opponent": 125,
                        "seat1_games_per_opponent": 125,
                    },
                    "roster": [
                        {
                            "opponent_id": "specialist-trevenant",
                            "tier": "S+",
                            "frozen_specialist": True,
                            "frozen_checkpoint_digest": trevenant,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "next_specialist": {
            "gate_contract": str(gate),
            "frozen_specialist_registry": str(registry),
        }
    }
    with pytest.raises(
        RuntimeError,
        match=(
            "frozen specialist missing or not S\\+ in next gate: "
            "specialist-alakazam"
        ),
    ):
        handoff.validate_next_specialist_gate(
            contract,
            {
                "specialist_id": "hops-trevenant",
                "checkpoint_digest": trevenant,
            },
        )


def test_handoff_refuses_source_target_overlap() -> None:
    source = (ROOT / "scripts/run_sequential_specialist_handoff.py").read_text(
        encoding="utf-8"
    )
    bootstrap = source.index("run_checked(bootstrap_command(contract))")
    source_guard = source.index(
        'if service_active(source_service):', source.index("def run(")
    )
    activation = source.index("write_or_validate_activation", bootstrap)
    start = source.index(
        'run_checked(["/usr/bin/systemctl", "--user", "start", target_service])'
    )
    assert source_guard < bootstrap < activation < start


def test_candidate_router_is_bound_after_bootstrap_before_registration() -> None:
    source = (
        ROOT / "scripts/run_sequential_specialist_handoff.py"
    ).read_text(encoding="utf-8")
    bootstrap = source.index("run_checked(bootstrap_command(contract))")
    bind = source.index("runtime_tree = prepare_runtime_tree", bootstrap)
    register = source.index("registration = register_specialist_runtime", bind)
    start = source.index(
        'run_checked(["/usr/bin/systemctl", "--user", "start", target_service])',
        register,
    )
    assert bootstrap < bind < register < start
    assert "minimum_validation_precision" in source
    assert "!= 0.93" in source
    assert "minimum_validation_weighted_support" in source
    assert "!= 10_000" in source


def test_verified_preflight_is_resumable_after_gate_materialization() -> None:
    source = (ROOT / "scripts/run_sequential_specialist_handoff.py").read_text(
        encoding="utf-8"
    )
    assert "resumable_preflight" in source
    assert '"preflight_verified"' in source
    assert 'previous["source_specialist"]' in source
    assert "source = validate_source(contract)" in source


def test_activation_verify_uses_checksum_bound_recorded_preflight() -> None:
    source = (ROOT / "scripts/run_sequential_specialist_handoff.py").read_text(
        encoding="utf-8"
    )
    assert "if recorded:" in source
    assert 'state.get("contract_sha256") == digest' in source
    assert 'next_gate = dict(state["next_specialist_splus_gate"])' in source


def test_activation_identity_ignores_only_registration_timestamp(
    tmp_path: Path,
) -> None:
    path = tmp_path / "activation.json"
    first = {
        "checkpoint": "sha256:model",
        "runtime_registration": {
            "created_at_utc": "first",
            "identity_sha256": "sha256:registration",
        },
    }
    second = {
        **first,
        "runtime_registration": {
            **first["runtime_registration"],
            "created_at_utc": "second",
        },
    }

    original = handoff.write_or_validate_activation(path, first)
    resumed = handoff.write_or_validate_activation(path, second)

    assert resumed == original


def test_handoff_defers_gate_handler_to_trainer_on_success() -> None:
    source = (ROOT / "scripts/run_sequential_specialist_handoff.py").read_text(
        encoding="utf-8"
    )
    start = source.index(
        'run_checked(["/usr/bin/systemctl", "--user", "start", target_service])'
    )
    target_active = source.index(
        '["/usr/bin/systemctl", "--user", "is-active", "--quiet", target_service]',
        start,
    )
    trigger = source.index('"trainer_on_success"', target_active)
    assert start < target_active < trigger
    assert '"restart",\n                    gate_handler_service' not in source


def test_starmie_unit_uses_exact_games_and_research_totals() -> None:
    unit = (
        ROOT / "deploy/systemd/pokebot-pure-rl-starmie-staged.service"
    ).read_text(encoding="utf-8")
    # The checked-in staged unit is updated before production activation.
    assert "--games-per-iter 8192" in unit


def test_starmie_is_fail_closed_until_verified_trevenant_handoff() -> None:
    drop_in = (
        ROOT
        / "deploy/systemd/pokebot-pure-rl-starmie-staged.service.d"
        / "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-predecessor-gate.conf"
    ).read_text(encoding="utf-8")
    assert "run_sequential_specialist_handoff.py verify" in drop_in
    assert "post_trevenant_starmie_handoff_v1.json" in drop_in
    assert "Restart=no" in drop_in
