#!/usr/bin/env python3
"""Materialize one exact passing specialist into the next additive S+ gate.

The passed-gate handler first freezes the exact checkpoint and builds the
checksum-bound Kaggle submission archive.  This module reuses that verified
archive as an inference-only baseline package, updates the next specialist's
frozen registry and gate atomically, and distributes the package to the
simulation fleet.  It never changes the frozen model or any training data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from typing import Any

from poke_bot.baselines_runtime import baseline_content_digest
from poke_bot.pure_rl.holdout_supersession import (
    apply_external_holdout_supersession,
    superseded_external_archetypes,
)
from poke_bot.pure_rl.model_registry import sha256


RECEIPT_SCHEMA = "poke_bot.frozen_specialist_gate_materialization/v1"
REGISTRY_SCHEMA = "poke_bot.frozen_specialist_registry/v1"
HANDLER_SCHEMA = "poke_bot.passed_gate_handler/v1"
ALLOWED_HANDLER_PHASES = {
    "submissions_queued",
    "waiting_for_terminal_trainer_before_handoff",
    "complete_handoff_started",
}
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
REVISION_SUFFIX = re.compile(r"(?P<prefix>.*\+frozen-specialists-r)\d+$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing/corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
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


def _versioned_gate_id(identifier: str, frozen_count: int) -> str:
    matched = REVISION_SUFFIX.fullmatch(str(identifier))
    if matched is None:
        raise RuntimeError(f"gate id lacks frozen-specialist revision: {identifier}")
    return f"{matched.group('prefix')}{frozen_count}"


def _safe_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise RuntimeError("submission archive is empty")
    for member in members:
        normalized = member.name.removeprefix("./")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            raise RuntimeError(f"unsafe submission archive member: {member.name}")
    return members


def _materialize_package(
    *,
    bundle_path: Path,
    bundle_digest: str,
    checkpoint_digest: str,
    baseline_root: Path,
    baseline_dir: str,
) -> tuple[Path, str, str]:
    if not bundle_path.is_file() or sha256(bundle_path) != bundle_digest:
        raise RuntimeError("passing specialist submission bundle identity changed")
    specialists = baseline_root / "specialists"
    specialists.mkdir(parents=True, exist_ok=True)
    destination = specialists / baseline_dir
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{baseline_dir}.", dir=str(specialists))
    )
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            members = _safe_archive_members(archive)
            for member in members:
                relative = PurePosixPath(member.name.removeprefix("./"))
                target = temporary.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                with source, target.open("xb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
                target.chmod(0o755 if relative.name == "main.py" else 0o644)
        required = (
            "model.pt",
            "deck.csv",
            "main.py",
            "cg/api.py",
            "matchup_tree.json",
        )
        if any(not (temporary / name).is_file() for name in required):
            raise RuntimeError("passing specialist package is incomplete")
        if sha256(temporary / "model.pt") != checkpoint_digest:
            raise RuntimeError("passing specialist package contains another checkpoint")
        tree = _read_json(temporary / "matchup_tree.json")
        runtime = dict(tree.get("runtime_contract") or {})
        if (
            tree.get("runtime_enabled") is not True
            or runtime.get("one_route_per_decision") is not True
            or runtime.get("unknown_route_exact_bypass") is not True
        ):
            raise RuntimeError(
                "passing specialist package lacks a validated causal router"
            )
        matchup_tree_digest = sha256(temporary / "matchup_tree.json")
        content_digest = baseline_content_digest(temporary)
        if destination.exists():
            if baseline_content_digest(destination) != content_digest:
                raise RuntimeError(
                    "refusing to replace a different specialist baseline package"
                )
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return destination, content_digest, matchup_tree_digest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _frozen_gate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "opponent_id": row["opponent_id"],
        "archetype_id": row["archetype_id"],
        "archetype_label": row["archetype_label"],
        "source": row["source"],
        "tier": "S+",
        "weight": 2.0,
        "frozen_specialist": True,
        "frozen_checkpoint_digest": row["checkpoint_digest"],
        "content_digest": row["content_digest"],
    }


def _build_registry(
    *,
    base: dict[str, Any],
    specialist_row: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    if base.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("base frozen specialist registry schema changed")
    source_id = str(specialist_row["specialist_id"])
    rows = [
        copy.deepcopy(row)
        for row in (base.get("specialists") or [])
        if str(row.get("specialist_id") or "") != source_id
    ]
    rows.append(copy.deepcopy(specialist_row))
    ids = [str(row.get("specialist_id") or "") for row in rows]
    opponents = [str(row.get("opponent_id") or "") for row in rows]
    if (
        len(ids) != len(set(ids))
        or len(opponents) != len(set(opponents))
        or any(
            row.get("frozen") is not True
            or row.get("public_mix_eligible") is not True
            or not str(row.get("checkpoint_digest") or "").startswith("sha256:")
            or not str(row.get("content_digest") or "").startswith("sha256:")
            for row in rows
        )
    ):
        raise RuntimeError("materialized frozen specialist registry is invalid")
    result = copy.deepcopy(base)
    result["specialists"] = rows
    result["version"] = len(rows)
    result["updated_at_utc"] = timestamp
    return result


def _build_gate(
    *,
    base: dict[str, Any],
    registry: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    gate = dict(result.get("next_gate") or {})
    unfiltered_public_rows = [
        copy.deepcopy(row)
        for row in (gate.get("roster") or [])
        if row.get("frozen_specialist") is not True
    ]
    public_rows, newly_retired = apply_external_holdout_supersession(
        unfiltered_public_rows,
        registry,
    )
    frozen_rows = [
        _frozen_gate_row(dict(row)) for row in registry["specialists"]
    ]
    roster = public_rows + frozen_rows
    superseded_archetypes = superseded_external_archetypes(registry)
    prior_semantics = dict(result.get("active_gate_semantics") or {})
    prior_retired_ids = {
        str(value)
        for value in prior_semantics.get(
            "superseded_external_premium_opponent_ids"
        )
        or []
    }
    retired_ids = sorted(
        prior_retired_ids
        | {str(row.get("opponent_id") or "") for row in newly_retired}
    )
    if not superseded_archetypes and len(public_rows) != 8:
        raise RuntimeError("established eight-agent premium roster changed")
    if superseded_archetypes and any(
        str(row.get("archetype_id") or "") in superseded_archetypes
        for row in public_rows
    ):
        raise RuntimeError("superseded external premium archetype remains")
    frozen_count = len(frozen_rows)
    gate_id = _versioned_gate_id(str(gate.get("id") or ""), frozen_count)
    evaluation = dict(gate.get("evaluation") or {})
    evaluation.update(
        {
            "games_total": 250 * len(roster),
            "games_per_opponent": 250,
            "minimum_games_per_opponent": 250,
            "seat0_games_per_opponent": 125,
            "seat1_games_per_opponent": 125,
        }
    )
    pass_criteria = dict(gate.get("pass_criteria") or {})
    pass_criteria["individual_opponent_floor"] = 0.15
    pass_criteria["s_plus_individual_floor"] = 0.30
    pass_criteria["s_plus_below_floor_allowance"] = 2
    pointer = str(gate.get("exact_result_pointer") or "")
    pointer = re.sub(
        r"frozen_r\d+_result\.json$",
        f"frozen_r{frozen_count}_result.json",
        pointer,
    )
    gate.update(
        {
            "id": gate_id,
            "roster": roster,
            "evaluation": evaluation,
            "pass_criteria": pass_criteria,
            "exact_result_pointer": pointer,
            "status": "queued",
            "threshold_transition": {
                "individual_opponent_floor": 0.15,
                "s_plus_individual_floor": 0.30,
                "s_plus_below_floor_allowance": 2,
                "effective_after_completed_specialist": "lucario",
                "effective_after_completed_iteration": 9,
                "applies_to_lucario_iteration_10": True,
            },
        }
    )
    fallback = dict(result.get("fallback_transition") or {})
    fallback_id = _versioned_gate_id(
        str(fallback.get("id") or ""), frozen_count
    )
    fallback.update(
        {
            "id": fallback_id,
            "prior_gate_id": gate_id,
            "unchanged_requirements": [
                (
                    f"{len(public_rows)} active external premium agents after "
                    "configured archetype supersession plus all frozen S+ "
                    "specialists"
                ),
                f"{250 * len(roster)} exact greedy games with "
                f"{frozen_count} frozen specialist(s)",
                "250 games per opponent",
                "125 games in each seat per opponent",
                "skill-weighted win rate >= 0.50",
                "S-tier mean >= 0.40",
                "every opponent >= 0.15",
                "accepted official holdout non-regression >= 0.50",
                "audit must pass",
            ],
        }
    )
    semantics = prior_semantics
    semantics.update(
        {
            "base_premium_agents": len(public_rows),
            "original_base_premium_agents": 8,
            "frozen_specialist_agents": frozen_count,
            "frozen_specialist_tier": "S+",
            "superseded_external_premium_archetypes": list(
                superseded_archetypes
            ),
            "superseded_external_premium_opponent_ids": retired_ids,
            "historical_superseded_results_preserved": True,
            "gate_roster_size": len(roster),
            "gate_games_total": 250 * len(roster),
            "games_per_opponent": 250,
        }
    )
    result.update(
        {
            "active_gate_id": gate_id,
            "active_gate_semantics": semantics,
            "next_gate": gate,
            "fallback_transition": fallback,
            "updated_at_utc": timestamp,
        }
    )
    return result


def _build_baseline_manifest(
    *,
    current: dict[str, Any],
    stale_opponent_ids: set[str],
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    agents = [
        copy.deepcopy(row)
        for row in (current.get("agents") or [])
        if str(row.get("id") or "") not in stale_opponent_ids
    ]
    agents.append(copy.deepcopy(manifest_row))
    ids = [str(row.get("id") or "") for row in agents]
    if len(ids) != len(set(ids)):
        raise RuntimeError("baseline manifest would contain duplicate agent ids")
    result = copy.deepcopy(current)
    result["agents"] = agents
    notes = dict(result.get("field_notes") or {})
    notes["total"] = len(agents)
    notes["frozen_specialists"] = sum(
        1 for row in agents if str(row.get("group") or "") == "specialists"
    )
    result["field_notes"] = notes
    return result


def _run_checked(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed rc={completed.returncode}: {' '.join(argv)}")


def _run_capture(argv: list[str]) -> str:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}{suffix}"
        )
    return completed.stdout


def _wait_tcp_endpoint(
    host: str,
    port: int,
    *,
    attempts: int = 90,
    interval_s: float = 1.0,
) -> None:
    """Wait until a restarted managed worker is accepting new sessions."""

    ssh_host = str(host).rsplit("@", 1)[-1]
    # Fleet entries are SSH aliases (for example ``elmo``), not necessarily
    # names that the socket resolver can use.  Resolve the alias through the
    # same SSH configuration used for the restart before probing its worker
    # data plane.  This also avoids accidentally selecting an unusable IPv6
    # address when SSH itself is pinned to the LAN IPv4 address.
    network_host = ssh_host
    try:
        ssh_config = _run_capture(["ssh", "-G", ssh_host])
        for line in ssh_config.splitlines():
            key, _, value = line.partition(" ")
            if key.lower() == "hostname" and value.strip():
                network_host = value.strip()
                break
    except RuntimeError:
        # Preserve the previous direct-host behavior when an SSH
        # implementation does not support configuration expansion.
        pass
    last_error: OSError | None = None
    for _attempt in range(max(1, int(attempts))):
        try:
            with socket.create_connection(
                (network_host, int(port)),
                timeout=min(2.0, max(0.2, float(interval_s))),
            ):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(max(0.0, float(interval_s)))
    raise RuntimeError(
        f"managed worker did not resume at {network_host}:{int(port)}"
    ) from last_error


def _container_baseline_mounts(*, host: str, container: str) -> dict[str, str]:
    output = _run_capture(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "sudo",
            "-n",
            "docker",
            "inspect",
            container,
        ]
    )
    try:
        rows = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"container {container} returned invalid mount metadata"
        ) from exc
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], dict)
    ):
        raise RuntimeError(
            f"container {container} returned invalid mount metadata"
        )
    rows = rows[0].get("Mounts")
    if not isinstance(rows, list):
        raise RuntimeError(
            f"container {container} returned invalid mount metadata"
        )
    mounts = {
        str(row.get("Destination") or ""): str(row.get("Source") or "")
        for row in rows
        if isinstance(row, dict)
        and str(row.get("Destination") or "")
        and str(row.get("Source") or "")
    }
    required = {
        "/workspace/baselines/specialists",
        "/workspace/baselines/manifest.json",
    }
    missing = sorted(required - mounts.keys())
    if missing:
        raise RuntimeError(
            f"container {container} is missing baseline bind mounts: {missing}"
        )
    for destination in required:
        if not PurePosixPath(mounts[destination]).is_absolute():
            raise RuntimeError(
                f"container {container} has unsafe bind source for {destination}"
            )
    return mounts


def _sync_one_remote(
    *,
    host: str,
    remote_root: str,
    package: Path,
    manifest: Path,
    container: str | None,
    group: str = "specialists",
) -> dict[str, Any]:
    if not SAFE_ID.fullmatch(group):
        raise ValueError("unsafe baseline group")
    remote_group = f"{remote_root.rstrip('/')}/{group}"
    remote_package = f"{remote_group}/{package.name}"
    _run_checked(["ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", remote_group])
    _run_checked(
        [
            "rsync",
            "-a",
            "--delete",
            f"{package}/",
            f"{host}:{remote_package}/",
        ]
    )
    temporary_manifest = f"{remote_root.rstrip('/')}/.manifest.json.pokebot.tmp"
    _run_checked(["scp", "-q", str(manifest), f"{host}:{temporary_manifest}"])
    _run_checked(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "mv",
            temporary_manifest,
            f"{remote_root.rstrip('/')}/manifest.json",
        ]
    )
    if container:
        mounts = _container_baseline_mounts(host=host, container=container)
        container_group_source = mounts[f"/workspace/baselines/{group}"]
        container_package_source = (
            f"{container_group_source.rstrip('/')}/{package.name}"
        )
        container_manifest_source = mounts["/workspace/baselines/manifest.json"]
        _run_checked(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "sudo",
                "-n",
                "mkdir",
                "-p",
                container_package_source,
            ]
        )
        _run_checked(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "sudo",
                "-n",
                "rsync",
                "-a",
                "--delete",
                f"{remote_package}/",
                f"{container_package_source}/",
            ]
        )
        _run_checked(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "sudo",
                "-n",
                "install",
                "-m",
                "0644",
                f"{remote_root.rstrip('/')}/manifest.json",
                container_manifest_source,
            ]
        )
        _run_checked(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "sudo",
                "-n",
                "docker",
                "exec",
                container,
                "test",
                "-f",
                f"/workspace/baselines/{group}/{package.name}/model.pt",
            ]
        )
        _run_checked(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "sudo",
                "-n",
                "docker",
                "exec",
                container,
                "test",
                "-f",
                "/workspace/baselines/manifest.json",
            ]
        )
        # The remote worker resolves its baseline manifest once at startup.
        # Updating the read-only bind source alone leaves its in-memory
        # registry stale. Rotate only the declared managed container, then
        # fail closed until its data-plane endpoint accepts new sessions.
        _run_checked(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "sudo",
                "-n",
                "docker",
                "restart",
                container,
            ]
        )
        _wait_tcp_endpoint(host, 8765)
    return {
        "host": host,
        "baseline_root": remote_root,
        "package": remote_package,
        "container": container,
        "container_mounts": mounts if container else None,
        "container_manifest_reloaded": bool(container),
    }


def materialize_from_contract(
    contract: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    source_contract = dict(contract.get("source_specialist") or {})
    target = dict(contract.get("next_specialist") or {})
    materialization = dict(contract.get("gate_materialization") or {})
    source_id = str(source.get("specialist_id") or "")
    if not SAFE_ID.fullmatch(source_id):
        raise RuntimeError("source specialist id is invalid")
    handler_path = Path(str(source_contract["handler_state"])).expanduser().resolve()
    handler = _read_json(handler_path)
    bundle = dict(handler.get("submission_bundle") or {})
    gate = dict(source.get("gate") or {})
    frozen = dict(handler.get("frozen_model") or {})
    queued = [dict(row) for row in (handler.get("queued_submissions") or [])]
    checkpoint_digest = str(source.get("checkpoint_digest") or "")
    if (
        handler.get("schema") != HANDLER_SCHEMA
        or handler.get("phase") not in ALLOWED_HANDLER_PHASES
        or frozen.get("checkpoint_digest") != checkpoint_digest
        or bundle.get("contents", {}).get("model_sha256") != checkpoint_digest
        or len(queued) != 1
        or queued[0].get("checkpoint_checksum") != checkpoint_digest
        or gate.get("checkpoint_digest") != checkpoint_digest
    ):
        raise RuntimeError("source gate handler cannot materialize an S+ package")
    iteration = int(gate.get("iteration", -1))
    if iteration < int(source_contract.get("minimum_completed_iteration", -1)):
        raise RuntimeError("source checkpoint predates specialist iteration floor")
    digest_short = checkpoint_digest.removeprefix("sha256:")[:12]
    timestamp = str(handler.get("updated_at_utc") or "")
    if not timestamp:
        raise RuntimeError("source handler lacks a stable materialization timestamp")
    baseline_dir = f"{source_id}-gate-iter{iteration}-{digest_short}"
    opponent_id = f"specialist-{baseline_dir}"
    baseline_root = Path(str(materialization["baseline_root"])).expanduser().resolve()
    baseline_manifest_path = Path(
        str(materialization["baseline_manifest"])
    ).expanduser().resolve()
    package, content_digest, matchup_tree_digest = _materialize_package(
        bundle_path=Path(str(bundle["path"])).expanduser().resolve(),
        bundle_digest=str(bundle["sha256"]),
        checkpoint_digest=checkpoint_digest,
        baseline_root=baseline_root,
        baseline_dir=baseline_dir,
    )
    archetype_label = str(materialization["archetype_label"])
    if gate.get("completion_authority") == "explicit_owner_ceiling_acceptance":
        source_label = (
            f"owner-accepted ceiling {archetype_label} iteration {iteration} "
            f"checkpoint {digest_short}; measured gate result preserved"
        )
    else:
        source_label = (
            f"exact protocol-gate passing {archetype_label} iteration "
            f"{iteration} checkpoint {digest_short}"
        )
    specialist_row = {
        "specialist_id": source_id,
        "opponent_id": opponent_id,
        "archetype_id": source_id,
        "archetype_label": f"Frozen {archetype_label} specialist",
        "baseline_dir": baseline_dir,
        "baseline_group": "specialists",
        "checkpoint_digest": checkpoint_digest,
        "content_digest": content_digest,
        "matchup_tree_checksum": matchup_tree_digest,
        "frozen": True,
        "public_mix_eligible": True,
        "research_eligible": False,
        # This registry describes inference opponents. Kaggle eligibility
        # belongs to the separately checksum-bound passing checkpoint and
        # persistent submission queue, never to this materialized copy.
        "kaggle_submission_eligible": False,
        "registered_at_utc": timestamp,
        "source": source_label,
    }
    base_registry_path = Path(
        str(materialization["base_frozen_specialist_registry"])
    ).expanduser().resolve()
    target_registry_path = Path(
        str(target["frozen_specialist_registry"])
    ).expanduser().resolve()
    old_target_registry = (
        _read_json(target_registry_path) if target_registry_path.is_file() else {}
    )
    registry = _build_registry(
        base=_read_json(base_registry_path),
        specialist_row=specialist_row,
        timestamp=timestamp,
    )
    stale_ids = {
        str(row.get("opponent_id") or "")
        for row in (old_target_registry.get("specialists") or [])
        if str(row.get("specialist_id") or "") == source_id
    }
    stale_ids.add(opponent_id)
    manifest = _build_baseline_manifest(
        current=_read_json(baseline_manifest_path),
        stale_opponent_ids=stale_ids,
        manifest_row={
            "id": opponent_id,
            "name": f"Frozen {archetype_label} Specialist · Iteration {iteration}",
            "dir": baseline_dir,
            "group": "specialists",
            "source": source_label,
        },
    )
    base_gate_path = Path(
        str(materialization["base_gate_contract"])
    ).expanduser().resolve()
    target_gate_path = Path(str(target["gate_contract"])).expanduser().resolve()
    next_gate = _build_gate(
        base=_read_json(base_gate_path),
        registry=registry,
        timestamp=timestamp,
    )
    _atomic_json(baseline_manifest_path, manifest)
    _atomic_json(target_registry_path, registry)
    _atomic_json(target_gate_path, next_gate)
    fleet = [
        _sync_one_remote(
            host=str(row["host"]),
            remote_root=str(row["baseline_root"]),
            package=package,
            manifest=baseline_manifest_path,
            container=(
                str(row["container"]) if row.get("container") is not None else None
            ),
        )
        for row in (materialization.get("fleet_sync") or [])
    ]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "specialist_id": source_id,
        "checkpoint_digest": checkpoint_digest,
        "iteration": iteration,
        "opponent_id": opponent_id,
        "baseline_dir": baseline_dir,
        "baseline_package": str(package),
        "baseline_content_digest": content_digest,
        "matchup_tree_checksum": matchup_tree_digest,
        "baseline_manifest": str(baseline_manifest_path),
        "baseline_manifest_sha256": sha256(baseline_manifest_path),
        "frozen_specialist_registry": str(target_registry_path),
        "frozen_specialist_registry_sha256": sha256(target_registry_path),
        "gate_contract": str(target_gate_path),
        "gate_contract_sha256": sha256(target_gate_path),
        "gate_id": next_gate["next_gate"]["id"],
        "gate_games_total": next_gate["next_gate"]["evaluation"]["games_total"],
        "frozen_specialist_ids": [
            row["specialist_id"] for row in registry["specialists"]
        ],
        "fleet_sync": fleet,
        "created_at_utc": timestamp,
    }
    receipt["identity_sha256"] = _canonical_digest(
        {key: value for key, value in receipt.items() if key != "created_at_utc"}
    )
    receipt_path = Path(str(materialization["receipt"])).expanduser().resolve()
    if receipt_path.is_file():
        existing = _read_json(receipt_path)
        if existing.get("identity_sha256") != receipt["identity_sha256"]:
            raise RuntimeError("existing S+ materialization receipt changed")
        return existing
    _atomic_json(receipt_path, receipt)
    return receipt
