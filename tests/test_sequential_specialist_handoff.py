from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_sequential_specialist_handoff as handoff


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


def test_handoff_rebinds_gate_handler_after_target_is_active() -> None:
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
    restart_handler = source.index('"restart",', target_active)
    handler_active = source.index(
        '"is-active",\n                    "--quiet",\n                    gate_handler_service',
        restart_handler,
    )
    assert start < target_active < restart_handler < handler_active


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
