from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

from scripts.materialize_population_opponents import (
    build_opponent_registry,
    materialize_current_version,
    materialize_refresh_bundle,
)
from scripts.population_round_robin_state import initialize_state


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_materialize_current_population_checkpoint(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baselines"
    source = baseline_root / "specialists" / "starmie-frozen"
    source.mkdir(parents=True)
    for name, data in {
        "model.pt": b"old model",
        "main.py": b"print('ok')\n",
        "deck.csv": b"1\n",
        "matchup_tree.json": b"{}\n",
    }.items():
        (source / name).write_bytes(data)
    manifest = baseline_root / "manifest.json"
    manifest.write_text(
        json.dumps({"agents": [], "field_notes": {}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "new.pt"
    checkpoint.write_bytes(b"new population model")
    result = materialize_current_version(
        specialist_id="starmie",
        population_cycle=2,
        checkpoint=checkpoint,
        checkpoint_digest=_digest(checkpoint.read_bytes()),
        source_package=source,
        baseline_root=baseline_root,
        baseline_manifest=manifest,
    )
    assert Path(result["checkpoint"]).read_bytes() == checkpoint.read_bytes()
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert [row["id"] for row in saved["agents"]] == [
        result["opponent_id"]
    ]


def test_materialize_final_format_refresh_bundle(tmp_path: Path) -> None:
    source = tmp_path / "bundle-source"
    source.mkdir()
    for name, data in {
        "model.pt": b"h10 refresh",
        "main.py": b"print('h10')\n",
        "deck.csv": b"1\n",
        "matchup_tree.json": b"{}\n",
    }.items():
        (source / name).write_bytes(data)
    bundle = tmp_path / "submission.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in source.iterdir():
            archive.add(path, arcname=path.name)
    baseline_root = tmp_path / "baselines"
    baseline_root.mkdir()
    manifest = baseline_root / "manifest.json"
    manifest.write_text(
        json.dumps({"agents": [], "field_notes": {}}), encoding="utf-8"
    )
    result = materialize_refresh_bundle(
        specialist_id="alakazam",
        checkpoint_digest=_digest((source / "model.pt").read_bytes()),
        bundle=bundle,
        bundle_digest=_digest(bundle.read_bytes()),
        baseline_root=baseline_root,
        baseline_manifest=manifest,
    )
    assert Path(result["checkpoint"]).read_bytes() == b"h10 refresh"
    assert result["baseline_group"] == "population-refresh"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["agents"][0]["id"] == result["opponent_id"]


def test_build_registry_uses_all_14_current_own_models(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baselines"
    members = []
    for index in range(14):
        specialist_id = f"specialist-{index:02d}"
        package = (
            baseline_root / "specialists" / f"{specialist_id}-frozen"
        )
        package.mkdir(parents=True)
        model = package / "model.pt"
        model.write_bytes(f"model-{index}".encode())
        for name in ("main.py", "deck.csv", "matchup_tree.json"):
            (package / name).write_text(name, encoding="utf-8")
        from poke_bot.baselines_runtime import baseline_content_digest

        members.append(
            {
                "specialist_id": specialist_id,
                "checkpoint": str(model),
                "checkpoint_digest": _digest(model.read_bytes()),
                "content_digest": baseline_content_digest(package),
                "opponent_id": f"specialist-{specialist_id}-frozen",
                "baseline_group": "specialists",
                "baseline_dir": f"{specialist_id}-frozen",
                "baseline_package": str(package),
                "expert_manifest": f"/expert/{specialist_id}.json",
                "expert_manifest_digest": "sha256:" + f"{index:064x}",
                "trainable_in_population": True,
                "external_agent": False,
            }
        )
    readiness = {
        "schema": "poke_bot.population_round_robin_ready/v1",
        "status": "ready",
        "member_count": 14,
        "members": members,
        "training_opponent_scope": "own_models_only",
        "external_agents_training_eligible": False,
        "rl_epochs_per_cycle": 5,
        "expert_rehearsal_epochs_per_cycle": 5,
    }
    state = initialize_state(readiness)
    registry = build_opponent_registry(
        state=state,
        baseline_root=baseline_root,
        output=tmp_path / "population-opponents.json",
    )
    assert registry["member_count"] == 14
    assert len(registry["opponent_ids"]) == 14
    assert all(
        row["external_agent"] is False for row in registry["opponents"]
    )
