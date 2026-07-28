from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.quarantine_unpassed_specialist_gate import quarantine


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _gate(*opponents: tuple[str, str]) -> dict:
    public = [{"opponent_id": f"public-{index}"} for index in range(8)]
    frozen = [
        {
            "opponent_id": opponent,
            "tier": "S+",
            "frozen_specialist": True,
            "frozen_checkpoint_digest": digest,
        }
        for opponent, digest in opponents
    ]
    return {
        "schema": "poke_bot.competition_gate_program/v1",
        "next_gate": {
            "evaluation": {
                "games_total": 250 * (8 + len(frozen)),
                "games_per_opponent": 250,
                "seat0_games_per_opponent": 125,
                "seat1_games_per_opponent": 125,
            },
            "roster": public + frozen,
        },
    }


def _registry(*rows: tuple[str, str, str]) -> dict:
    return {
        "schema": "poke_bot.frozen_specialist_registry/v1",
        "specialists": [
            {
                "specialist_id": specialist,
                "opponent_id": opponent,
                "checkpoint_digest": digest,
            }
            for specialist, opponent, digest in rows
        ],
    }


def test_exact_unpassed_extension_is_quarantined_idempotently(
    tmp_path: Path,
) -> None:
    alakazam = "sha256:" + "a" * 64
    trevenant = "sha256:" + "b" * 64
    source_gate = tmp_path / "source-gate.json"
    source_registry = tmp_path / "source-registry.json"
    target_gate = tmp_path / "target-gate.json"
    target_registry = tmp_path / "target-registry.json"
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    _write(source_gate, _gate(("specialist-alakazam", alakazam)))
    _write(
        source_registry,
        _registry(("alakazam", "specialist-alakazam", alakazam)),
    )
    _write(
        target_gate,
        _gate(
            ("specialist-alakazam", alakazam),
            ("specialist-trevenant-iter2", trevenant),
        ),
    )
    _write(
        target_registry,
        _registry(
            ("alakazam", "specialist-alakazam", alakazam),
            ("hops-trevenant", "specialist-trevenant-iter2", trevenant),
        ),
    )
    _write(
        manifest,
        {
            "agents": [
                {"id": "specialist-alakazam"},
                {"id": "specialist-trevenant-iter2"},
            ]
        },
    )
    args = SimpleNamespace(
        source_gate=source_gate,
        source_registry=source_registry,
        target_gate=target_gate,
        target_registry=target_registry,
        baseline_manifest=manifest,
        specialist_id="hops-trevenant",
        opponent_id="specialist-trevenant-iter2",
        checkpoint_digest=trevenant,
        receipt=receipt,
    )

    first = quarantine(args)
    second = quarantine(args)

    assert second == first
    assert json.loads(target_gate.read_text()) == json.loads(source_gate.read_text())
    assert json.loads(target_registry.read_text()) == json.loads(
        source_registry.read_text()
    )
    assert [row["id"] for row in json.loads(manifest.read_text())["agents"]] == [
        "specialist-alakazam"
    ]
    assert first["checkpoint_artifacts_deleted"] is False
