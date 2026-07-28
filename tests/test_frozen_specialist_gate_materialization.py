from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

from scripts.materialize_frozen_specialist_gate import (
    _build_gate,
    _sync_one_remote,
    materialize_from_contract,
)


def test_production_handoffs_materialize_into_live_runtime_baselines() -> None:
    root = Path(__file__).resolve().parents[1]
    live_runtime = Path(
        "/home/inzi/poke-bot-agent-deployments/"
        "pure-rl-resident-v41-specialist-matchup-runtime"
    )
    for relative, section in (
        ("ops/specialist_cycle_handoff_v1.json", "gate_materialization"),
        ("ops/post_starmie_core_v2_handoff_v1.json", "gate_materialization"),
        ("ops/post_trevenant_starmie_handoff_v1.json", "gate_materialization"),
        ("ops/population_round_robin_v1.json", "paths"),
    ):
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        configured = payload[section]
        expected_baseline_root = (
            root / "baselines"
            if relative == "ops/specialist_cycle_handoff_v1.json"
            else live_runtime / "baselines"
        )
        assert Path(configured["baseline_root"]) == expected_baseline_root
        assert Path(configured["baseline_manifest"]) == (
            expected_baseline_root / "manifest.json"
        )


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
            "minimum_completed_iteration": 5,
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
            "iteration": 5,
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
    assert registry["specialists"][-1]["kaggle_submission_eligible"] is False


def test_frozen_lucario_supersedes_only_external_lucario_holdouts() -> None:
    public = [
        {
            "opponent_id": f"lucario-{index}",
            "archetype_id": "lucario",
        }
        for index in range(5)
    ] + [
        {
            "opponent_id": f"other-{index}",
            "archetype_id": f"other-{index}",
        }
        for index in range(3)
    ]
    base = {
        "active_gate_id": "strong+frozen-specialists-r1",
        "active_gate_semantics": {},
        "fallback_transition": {
            "id": "fallback+frozen-specialists-r1",
        },
        "next_gate": {
            "id": "strong+frozen-specialists-r1",
            "evaluation": {},
            "roster": public,
            "exact_result_pointer": "frozen_r1_result.json",
        },
    }
    frozen = {
        "policy": {
            "external_premium_archetype_supersession": {
                "enabled": True,
                "scope": "premium_holdout_external_opponents",
                "preserve_historical_results": True,
                "keep_triggering_frozen_specialist": True,
                "rules": [
                    {
                        "trigger_specialist_id": "lucario",
                        "external_archetype_id": "lucario",
                        "remove_external_opponents": True,
                    }
                ],
            }
        },
        "specialists": [
            {
                "specialist_id": "lucario",
                "opponent_id": "specialist-lucario",
                "archetype_id": "lucario",
                "archetype_label": "Frozen Lucario specialist",
                "source": "exact frozen Lucario",
                "checkpoint_digest": _digest(b"lucario-model"),
                "content_digest": _digest(b"lucario-package"),
                "frozen": True,
            }
        ],
    }

    gate = _build_gate(
        base=base,
        registry=frozen,
        timestamp="2026-07-23T00:00:00Z",
    )
    roster = gate["next_gate"]["roster"]
    assert [row["opponent_id"] for row in roster] == [
        "other-0",
        "other-1",
        "other-2",
        "specialist-lucario",
    ]
    assert gate["next_gate"]["evaluation"]["games_total"] == 1000
    assert gate["active_gate_semantics"][
        "superseded_external_premium_archetypes"
    ] == ["lucario"]
    assert gate["active_gate_semantics"][
        "superseded_external_premium_opponent_ids"
    ] == [f"lucario-{index}" for index in range(5)]
    assert roster[-1]["frozen_specialist"] is True


def test_container_sync_updates_read_only_bind_sources(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "dudunsparce-gate-iter15"
    package.mkdir()
    (package / "model.pt").write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"agents": []})
    commands: list[list[str]] = []

    def fake_checked(argv: list[str]) -> None:
        commands.append(argv)

    def fake_capture(argv: list[str]) -> str:
        commands.append(argv)
        return json.dumps(
            [
                {
                    "Mounts": [
                        {
                            "Destination": "/workspace/baselines/manifest.json",
                            "Source": (
                                "/mnt/Main/Elmo/baseline-sync/manifest.json"
                            ),
                        },
                        {
                            "Destination": "/workspace/baselines/specialists",
                            "Source": (
                                "/mnt/Main/Elmo/baseline-sync/specialists"
                            ),
                        },
                    ]
                }
            ]
        )

    monkeypatch.setattr(
        "scripts.materialize_frozen_specialist_gate._run_checked", fake_checked
    )
    monkeypatch.setattr(
        "scripts.materialize_frozen_specialist_gate._run_capture", fake_capture
    )
    waited: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "scripts.materialize_frozen_specialist_gate._wait_tcp_endpoint",
        lambda host, port: waited.append((host, port)),
    )

    receipt = _sync_one_remote(
        host="elmo",
        remote_root="/mnt/Main/archive/baselines",
        package=package,
        manifest=manifest,
        container="poke-bot-truenas-worker",
    )

    flattened = [" ".join(command) for command in commands]
    assert not any(" docker cp " in f" {command} " for command in flattened)
    assert not any(
        "docker exec poke-bot-truenas-worker mkdir" in command
        for command in flattened
    )
    assert any(
        "sudo -n rsync -a --delete "
        "/mnt/Main/archive/baselines/specialists/dudunsparce-gate-iter15/ "
        "/mnt/Main/Elmo/baseline-sync/specialists/dudunsparce-gate-iter15/"
        in command
        for command in flattened
    )
    assert any(
        "sudo -n install -m 0644 "
        "/mnt/Main/archive/baselines/manifest.json "
        "/mnt/Main/Elmo/baseline-sync/manifest.json"
        in command
        for command in flattened
    )
    assert any(
        "sudo -n docker restart poke-bot-truenas-worker" in command
        for command in flattened
    )
    assert waited == [("elmo", 8765)]
    assert receipt["container_manifest_reloaded"] is True
    assert receipt["container_mounts"] == {
        "/workspace/baselines/manifest.json": (
            "/mnt/Main/Elmo/baseline-sync/manifest.json"
        ),
        "/workspace/baselines/specialists": (
            "/mnt/Main/Elmo/baseline-sync/specialists"
        ),
    }
