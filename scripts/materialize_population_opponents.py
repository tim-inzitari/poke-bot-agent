#!/usr/bin/env python3
"""Materialize checksum-bound current/history population opponent packages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from poke_bot.baselines_runtime import baseline_content_digest
from poke_bot.pure_rl.model_registry import sha256
from scripts.population_round_robin_state import (
    MEMBER_COUNT,
    STATE_SCHEMA,
    eligible_own_opponents,
)


REGISTRY_SCHEMA = "poke_bot.population_opponent_registry/v1"
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def materialize_current_version(
    *,
    specialist_id: str,
    population_cycle: int,
    checkpoint: Path,
    checkpoint_digest: str,
    source_package: Path,
    baseline_root: Path,
    baseline_manifest: Path,
) -> dict[str, Any]:
    specialist_id = str(specialist_id)
    if (
        not SAFE_ID.fullmatch(specialist_id)
        or int(population_cycle) < 0
        or not str(checkpoint_digest).startswith("sha256:")
    ):
        raise ValueError("unsafe population package identity")
    checkpoint = checkpoint.expanduser().resolve()
    source_package = source_package.expanduser().resolve()
    baseline_root = baseline_root.expanduser().resolve()
    baseline_manifest = baseline_manifest.expanduser().resolve()
    if (
        not checkpoint.is_file()
        or sha256(checkpoint) != checkpoint_digest
        or not source_package.is_dir()
        or not (source_package / "model.pt").is_file()
        or not baseline_manifest.is_file()
    ):
        raise RuntimeError("population package inputs are missing or changed")
    suffix = checkpoint_digest.removeprefix("sha256:")[:12]
    baseline_dir = f"{specialist_id}-cycle-{population_cycle:04d}-{suffix}"
    opponent_id = f"population-{baseline_dir}"
    group_root = baseline_root / "population"
    destination = group_root / baseline_dir
    group_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{baseline_dir}.", dir=str(group_root))
    )
    try:
        shutil.rmtree(temporary)
        shutil.copytree(source_package, temporary, symlinks=False)
        model = temporary / "model.pt"
        staged_model = temporary / ".model.pt.population.tmp"
        shutil.copy2(checkpoint, staged_model)
        os.replace(staged_model, model)
        if sha256(model) != checkpoint_digest:
            raise RuntimeError("materialized population model digest changed")
        for required in ("main.py", "deck.csv", "matchup_tree.json"):
            if not (temporary / required).is_file():
                raise RuntimeError(
                    f"population package missing required file: {required}"
                )
        content_digest = baseline_content_digest(temporary)
        if destination.exists():
            if baseline_content_digest(destination) != content_digest:
                raise RuntimeError(
                    "refusing to replace a different population package"
                )
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    manifest = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    agents = [
        dict(row)
        for row in (manifest.get("agents") or [])
        if str(row.get("id") or "") != opponent_id
    ]
    agents.append(
        {
            "id": opponent_id,
            "name": (
                f"{specialist_id} population cycle {population_cycle:04d}"
            ),
            "group": "population",
            "dir": baseline_dir,
            "source": (
                "local checksum-bound population current version "
                f"{checkpoint_digest}"
            ),
        }
    )
    if len({str(row.get("id") or "") for row in agents}) != len(agents):
        raise RuntimeError("population baseline manifest has duplicate ids")
    manifest["agents"] = agents
    notes = dict(manifest.get("field_notes") or {})
    notes["total"] = len(agents)
    notes["population_versions"] = sum(
        str(row.get("group") or "") == "population" for row in agents
    )
    manifest["field_notes"] = notes
    _atomic_json(baseline_manifest, manifest)
    return {
        "specialist_id": specialist_id,
        "population_cycle": int(population_cycle),
        "opponent_id": opponent_id,
        "checkpoint": str(destination / "model.pt"),
        "checkpoint_digest": checkpoint_digest,
        "content_digest": content_digest,
        "baseline_group": "population",
        "baseline_dir": baseline_dir,
        "baseline_package": str(destination),
    }


def build_opponent_registry(
    *,
    state: dict[str, Any],
    baseline_root: Path,
    output: Path,
) -> dict[str, Any]:
    if (
        state.get("schema") != STATE_SCHEMA
        or int(state.get("member_count") or 0) != MEMBER_COUNT
    ):
        raise RuntimeError("population state changed")
    active = str(state.get("active_specialist_id") or "")
    rows = eligible_own_opponents(
        state,
        active_specialist_id=active,
    )
    baseline_root = baseline_root.expanduser().resolve()
    for row in rows:
        package = Path(str(row["baseline_package"])).expanduser().resolve()
        checkpoint = Path(str(row["checkpoint"])).expanduser().resolve()
        if (
            not package.is_dir()
            or not checkpoint.is_file()
            or sha256(checkpoint) != row["checkpoint_digest"]
            or baseline_content_digest(package) != row["content_digest"]
            or baseline_root not in package.parents
        ):
            raise RuntimeError(
                "population opponent package identity failed: "
                f"{row['opponent_id']}"
            )
    registry = {
        "schema": REGISTRY_SCHEMA,
        "population_cycle": int(state["population_cycle"]),
        "active_specialist_id": active,
        "member_count": MEMBER_COUNT,
        "specialist_ids": [
            str(row["specialist_id"]) for row in state["members"]
        ],
        "opponents": rows,
        "opponent_ids": [str(row["opponent_id"]) for row in rows],
        "external_agents_training_eligible": False,
        "official_agents_role": "research_only",
        "premium_agents_role": "research_only",
    }
    registry["identity"] = _canonical_digest(registry)
    _atomic_json(output.expanduser().resolve(), registry)
    return registry
