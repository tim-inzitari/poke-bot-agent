#!/usr/bin/env python3
"""Stage completed Marnie H10 as Crustle practice and formal-holdout expert."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any

from poke_bot.baselines_runtime import baseline_content_digest
from poke_bot.pure_rl.model_registry import sha256


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_bundle(
    bundle: Path,
    *,
    bundle_digest: str,
    checkpoint_digest: str,
    destination: Path,
) -> str:
    if not bundle.is_file() or sha256(bundle) != bundle_digest:
        raise RuntimeError("Marnie H10 submission bundle is missing or changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = PurePosixPath(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise RuntimeError("unsafe member in Marnie H10 bundle")
            archive.extractall(temporary)
        for name in ("main.py", "model.pt", "deck.csv", "matchup_tree.json"):
            if not (temporary / name).is_file():
                raise RuntimeError(f"Marnie H10 package missing {name}")
        if sha256(temporary / "model.pt") != checkpoint_digest:
            raise RuntimeError("Marnie H10 bundle checkpoint changed")
        content_digest = baseline_content_digest(temporary)
        if destination.exists():
            if baseline_content_digest(destination) != content_digest:
                raise RuntimeError("existing Marnie H10 baseline differs")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return content_digest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marnie-completion", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--marnie-training-freeze", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    completion_path = args.marnie_completion.resolve()
    completion = read_json(completion_path)
    refresh = read_json(args.refresh_registry.resolve())
    marnie = next(
        (
            dict(row)
            for row in refresh.get("refreshes") or ()
            if row.get("specialist_id") == SPECIALIST_ID
        ),
        None,
    )
    training_freeze = (
        read_json(args.marnie_training_freeze.resolve())
        if args.marnie_training_freeze is not None
        else None
    )
    source_registry_path = Path(str(completion.get("runtime_registry") or "")).resolve()
    checkpoint_digest = str(
        (training_freeze or {}).get("checkpoint_sha256")
        or completion.get("refresh_checkpoint_checksum")
        or ""
    )
    if (
        completion.get("schema")
        != "poke_bot.post_fleet_specialist_refresh_completion/v1"
        or completion.get("specialist_id") != SPECIALIST_ID
        or completion.get("frozen") is not True
        or completion.get("registered") is not True
        or marnie is None
        or (
            training_freeze is None
            and marnie.get("checkpoint_checksum") != checkpoint_digest
        )
        or not source_registry_path.is_file()
        or sha256(source_registry_path) != completion.get("runtime_registry_sha256")
    ):
        raise RuntimeError("completed Marnie H10 identity is invalid")
    if training_freeze is not None:
        if (
            training_freeze.get("schema") != "poke_bot.marnie_training_freeze/v1"
            or training_freeze.get("status") != "owner_selected_training_freeze"
            or training_freeze.get("training_use_only") is not True
            or training_freeze.get("kaggle_upload_requested") is not False
            or training_freeze.get("specialist_id") != SPECIALIST_ID
            or training_freeze.get("checkpoint_sha256") != checkpoint_digest
            or int(training_freeze.get("iteration", -1)) != 9
        ):
            raise RuntimeError("Marnie training freeze receipt is invalid")
    source_registry = read_json(source_registry_path)
    runtime_root = Path(str(source_registry.get("runtime_root") or "")).resolve()
    frozen_source = runtime_root / str(source_registry["frozen_specialist_registry"])
    gate_source = runtime_root / str(source_registry["active_gate_contract"])
    baseline_root = runtime_root / "baselines"
    baseline_manifest = baseline_root / "manifest.json"
    for path in (frozen_source, gate_source, baseline_manifest):
        if not path.is_file():
            raise RuntimeError(f"source runtime artifact missing: {path}")

    short = checkpoint_digest.removeprefix("sha256:")[:12]
    opponent_id = f"specialist-marnie-final-format-h10-{short}"
    baseline_dir = f"marnie-final-format-h10-{short}"
    bundle = Path(
        str(
            (training_freeze or {}).get("submission_bundle")
            or marnie.get("submission_bundle")
            or ""
        )
    ).resolve()
    bundle_digest = str(
        (training_freeze or {}).get("submission_bundle_sha256")
        or marnie.get("submission_bundle_sha256")
        or ""
    )
    package = baseline_root / "specialists" / baseline_dir
    content_digest = extract_bundle(
        bundle,
        bundle_digest=bundle_digest,
        checkpoint_digest=checkpoint_digest,
        destination=package,
    )

    manifest = read_json(baseline_manifest)
    agents = [
        copy.deepcopy(row)
        for row in manifest.get("agents") or ()
        if row.get("id") != opponent_id
    ]
    agents.append(
        {
            "id": opponent_id,
            "name": "Frozen final-format H10 Marnie's Grimmsnarl ex refresh",
            "group": "specialists",
            "dir": baseline_dir,
            "source": f"checksum-bound Marnie completion {sha256(completion_path)}",
        }
    )
    if len({str(row.get("id") or "") for row in agents}) != len(agents):
        raise RuntimeError("baseline manifest contains duplicate ids")
    manifest["agents"] = agents
    manifest.setdefault("field_notes", {})["total"] = len(agents)
    atomic_json(baseline_manifest, manifest)

    frozen = read_json(frozen_source)
    old_rows = [
        copy.deepcopy(row)
        for row in frozen.get("specialists") or ()
        if row.get("specialist_id") == SPECIALIST_ID
    ]
    if len(old_rows) != 1:
        raise RuntimeError("expected one historical Marnie frozen row")
    old = old_rows[0]
    new_row = {
        **old,
        "archetype_label": "Frozen final-format H10 Marnie's Grimmsnarl ex refresh",
        "baseline_dir": baseline_dir,
        "checkpoint_digest": checkpoint_digest,
        "content_digest": content_digest,
        "deck_file_checksum": sha256(package / "deck.csv"),
        "matchup_tree_checksum": sha256(package / "matchup_tree.json"),
        "opponent_id": opponent_id,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "roster_version": 6,
        "source": (
            "owner-selected Marnie iter-9 training freeze"
            if training_freeze is not None
            else "completed final-format Marnie H10 iteration-20 refresh"
        ),
        "source_passing_checkpoint_digest": checkpoint_digest,
        "refresh_completion_receipt": str(completion_path),
        "refresh_completion_receipt_sha256": sha256(completion_path),
        "supersedes_for_crustle_opponent_id": old["opponent_id"],
        "final_format_h10_refresh": True,
    }
    frozen_rows = [
        copy.deepcopy(row)
        for row in frozen.get("specialists") or ()
        if row.get("specialist_id") != SPECIALIST_ID
    ] + [new_row]
    if len(frozen_rows) != len(frozen.get("specialists") or ()):
        raise RuntimeError("Crustle scoped frozen roster size changed")
    frozen["specialists"] = frozen_rows
    frozen["owner_decision_revision"] = 163 if training_freeze is not None else 160
    frozen["scope"] = "post-marnie-crustle-r163" if training_freeze is not None else "post-marnie-crustle-r160"
    frozen["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen_output_rel = (
        "ops/frozen_specialist_registry_crustle_r163_h10_marnie.json"
        if training_freeze is not None
        else "ops/frozen_specialist_registry_crustle_r160_h10_marnie.json"
    )
    frozen_output = runtime_root / frozen_output_rel
    atomic_json(frozen_output, frozen)

    gate = read_json(gate_source)
    next_gate = dict(gate.get("next_gate") or {})
    roster = [copy.deepcopy(row) for row in next_gate.get("roster") or ()]
    matched = 0
    for row in roster:
        if row.get("opponent_id") != old["opponent_id"]:
            continue
        matched += 1
        row["opponent_id"] = opponent_id
        row["archetype_label"] = new_row["archetype_label"]
        row["frozen_checkpoint_digest"] = checkpoint_digest
        row["tier"] = "S"
        row["weight"] = 2.0
    if matched != 1:
        raise RuntimeError("historical Marnie gate row is not singular")
    gate_revision = 163 if training_freeze is not None else 160
    new_gate_id = str(next_gate.get("id") or "") + f"+marnie-h10-r{gate_revision}"
    next_gate["id"] = new_gate_id
    next_gate["roster"] = roster
    gate["next_gate"] = next_gate
    gate["active_gate_id"] = new_gate_id
    gate["owner_decision_revision"] = gate_revision
    gate_output_rel = (
        "runtime/final_format_crustle_gate_r163_h10_marnie.json"
        if training_freeze is not None
        else "runtime/final_format_crustle_gate_r160_h10_marnie.json"
    )
    gate_output = runtime_root / gate_output_rel
    atomic_json(gate_output, gate)

    receipt = {
        "schema": "poke_bot.crustle_marnie_expert_practice_stage/v1",
        "status": "ready_for_post_marnie_crustle_bootstrap",
        "owner_decision_revision": 163 if training_freeze is not None else 160,
        "marnie_completion": str(completion_path),
        "marnie_completion_sha256": sha256(completion_path),
        **(
            {
                "marnie_training_freeze": str(args.marnie_training_freeze.resolve()),
                "marnie_training_freeze_sha256": sha256(args.marnie_training_freeze.resolve()),
                "training_use_only": True,
            }
            if args.marnie_training_freeze is not None
            else {}
        ),
        "checkpoint_digest": checkpoint_digest,
        "submission_bundle": str(bundle),
        "submission_bundle_sha256": bundle_digest,
        "opponent_id": opponent_id,
        "baseline_dir": baseline_dir,
        "baseline_content_digest": content_digest,
        "frozen_registry": str(frozen_output),
        "frozen_registry_relative": frozen_output_rel,
        "frozen_registry_sha256": sha256(frozen_output),
        "active_gate": str(gate_output),
        "active_gate_relative": gate_output_rel,
        "active_gate_sha256": sha256(gate_output),
        "active_gate_id": new_gate_id,
        "practice_public_mix_eligible": True,
        "formal_holdout_eligible": True,
        "actions_relabelled_as_crustle_expert_targets": False,
    }
    atomic_json(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
