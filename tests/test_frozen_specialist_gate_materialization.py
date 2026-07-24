from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

from scripts.materialize_frozen_specialist_gate import materialize_from_contract


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bundle(path: Path, model: bytes) -> str:
    matchup_tree = json.dumps(
        {
            "runtime_enabled": True,
            "runtime_contract": {
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        }
    ).encode("utf-8")
    members = {
        "model.pt": model,
        "deck.csv": b"1,60\n",
        "main.py": b"def main(state): return [0]\n",
        "cg/api.py": b"# portable engine\n",
        "matchup_tree.json": matchup_tree,
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return _digest(path.read_bytes())


def test_exact_passing_specialist_is_added_to_every_future_s_plus_gate(
    tmp_path: Path,
) -> None:
    model = b"exact-trevenant-checkpoint"
    checkpoint = _digest(model)
    bundle = tmp_path / "submission.tar.gz"
    bundle_digest = _bundle(bundle, model)
    handler = tmp_path / "handler.json"
    _write_json(
        handler,
        {
            "schema": "poke_bot.passed_gate_handler/v1",
            "phase": "complete_handoff_started",
            "updated_at_utc": "2026-07-23T18:00:00+00:00",
            "frozen_model": {"checkpoint_digest": checkpoint},
            "submission_bundle": {
                "path": str(bundle),
                "sha256": bundle_digest,
                "contents": {"model_sha256": checkpoint},
            },
            "queued_submissions": [
                {
                    "copy_number": 1,
                    "checkpoint_checksum": checkpoint,
                }
            ],
        },
    )
    base_registry = tmp_path / "base-registry.json"
    _write_json(
        base_registry,
        {
            "schema": "poke_bot.frozen_specialist_registry/v1",
            "version": 1,
            "policy": {},
            "specialists": [
                {
                    "specialist_id": "alakazam",
                    "opponent_id": "specialist-alakazam",
                    "archetype_id": "alakazam",
                    "archetype_label": "Frozen Alakazam specialist",
                    "source": "exact Alakazam",
                    "baseline_dir": "alakazam",
                    "baseline_group": "specialists",
                    "checkpoint_digest": _digest(b"alakazam-model"),
                    "content_digest": _digest(b"alakazam-package"),
                    "frozen": True,
                    "public_mix_eligible": True,
                }
            ],
        },
    )
    public_roster = [
        {
            "opponent_id": f"public-{index}",
            "archetype_id": f"public-{index}",
            "archetype_label": f"Public {index}",
            "source": "fixture",
            "tier": "S" if index < 3 else "A",
            "weight": 2.0 if index < 3 else 1.0,
            "content_digest": _digest(f"public-{index}".encode()),
        }
        for index in range(8)
    ]
    base_gate = tmp_path / "base-gate.json"
    _write_json(
        base_gate,
        {
            "schema": "poke_bot.competition_gate_program/v1",
            "active_gate_id": "strong+frozen-specialists-r1",
            "active_gate_semantics": {},
            "fallback_transition": {
                "id": "fallback+frozen-specialists-r1",
                "prior_gate_id": "strong+frozen-specialists-r1",
            },
            "next_gate": {
                "id": "strong+frozen-specialists-r1",
                "exact_result_pointer": str(
                    tmp_path / "specialist_strong_public_plus_frozen_r1_result.json"
                ),
                "evaluation": {},
                "roster": public_roster
                + [
                    {
                        "opponent_id": "specialist-alakazam",
                        "frozen_specialist": True,
                    }
                ],
            },
        },
    )
    baseline_root = tmp_path / "baselines"
    baseline_manifest = baseline_root / "manifest.json"
    _write_json(
        baseline_manifest,
        {
            "agents": [
                {
                    "id": "specialist-alakazam",
                    "dir": "alakazam",
                    "group": "specialists",
                },
                {
                    "id": "specialist-hops-trevenant-gate-iter2",
                    "dir": "stale",
                    "group": "specialists",
                },
            ],
            "field_notes": {},
        },
    )
    target_registry = tmp_path / "target-registry.json"
    _write_json(
        target_registry,
        {
            "schema": "poke_bot.frozen_specialist_registry/v1",
            "specialists": [
                {
                    "specialist_id": "hops-trevenant",
                    "opponent_id": "specialist-hops-trevenant-gate-iter2",
                }
            ],
        },
    )
    target_gate = tmp_path / "target-gate.json"
    receipt = tmp_path / "receipt.json"
    contract = {
        "source_specialist": {
            "handler_state": str(handler),
            "minimum_completed_iteration": 25,
        },
        "next_specialist": {
            "gate_contract": str(target_gate),
            "frozen_specialist_registry": str(target_registry),
        },
        "gate_materialization": {
            "archetype_label": "Hop's Trevenant",
            "base_gate_contract": str(base_gate),
            "base_frozen_specialist_registry": str(base_registry),
            "baseline_root": str(baseline_root),
            "baseline_manifest": str(baseline_manifest),
            "receipt": str(receipt),
            "fleet_sync": [],
        },
    }
    source = {
        "specialist_id": "hops-trevenant",
        "checkpoint_digest": checkpoint,
        "gate": {
            "iteration": 25,
            "checkpoint_digest": checkpoint,
        },
    }

    first = materialize_from_contract(contract, source)
    second = materialize_from_contract(contract, source)

    assert second == first
    registry = json.loads(target_registry.read_text(encoding="utf-8"))
    gate = json.loads(target_gate.read_text(encoding="utf-8"))
    manifest = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    assert [row["specialist_id"] for row in registry["specialists"]] == [
        "alakazam",
        "hops-trevenant",
    ]
    frozen = [
        row for row in gate["next_gate"]["roster"] if row.get("frozen_specialist")
    ]
    assert len(gate["next_gate"]["roster"]) == 10
    assert gate["next_gate"]["evaluation"] == {
        "games_total": 2500,
        "games_per_opponent": 250,
        "minimum_games_per_opponent": 250,
        "seat0_games_per_opponent": 125,
        "seat1_games_per_opponent": 125,
    }
    assert {row["tier"] for row in frozen} == {"S+"}
    assert {row["frozen_checkpoint_digest"] for row in frozen} == {
        registry["specialists"][0]["checkpoint_digest"],
        checkpoint,
    }
    assert not any(
        row["id"] == "specialist-hops-trevenant-gate-iter2"
        for row in manifest["agents"]
    )
    assert any(row["id"] == first["opponent_id"] for row in manifest["agents"])
    assert registry["specialists"][-1]["matchup_tree_checksum"].startswith(
        "sha256:"
    )
