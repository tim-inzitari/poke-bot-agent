#!/usr/bin/env python3
"""Build and queue the exact r195 Alakazam NO-RTP/RTP submission pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.handle_passed_gate import (
    _copy_submission_slot,
    build_submission_bundle,
    queue_submission_copies,
)
from scripts.stage_alakazam_rtp_r175_milestone_submissions import (
    materialize_owner_pilot_deck,
)
from poke_bot import checkpoint as checkpoint_mod


SCHEMA = "poke_bot.alakazam_terminal_no_rtp_submission_r195/v1"
PARENT_SHA256 = "sha256:87caf05bdeda3a798268905a5670841125b1797f31b9a823343c393d7f0ced65"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable r195 submission receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def audit_no_rtp_bundle(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        names = {
            member.name.removeprefix("./")
            for member in archive.getmembers()
            if member.isfile()
        }
        profile_member = next(
            (member for member in archive.getmembers() if member.name.removeprefix("./") == "runtime_profile.json"),
            None,
        )
        main_member = next(
            (member for member in archive.getmembers() if member.name.removeprefix("./") == "main.py"),
            None,
        )
        if profile_member is None or main_member is None:
            raise RuntimeError("NO RTP bundle lacks runtime profile or entrypoint")
        profile_stream = archive.extractfile(profile_member)
        main_stream = archive.extractfile(main_member)
        if profile_stream is None or main_stream is None:
            raise RuntimeError("NO RTP bundle members are unreadable")
        profile = json.loads(profile_stream.read())
        main_source = main_stream.read().decode("utf-8")
    forbidden_sidecars = sorted(
        name
        for name in names
        if name.endswith("rtp_shadow_planner.pt") or name.endswith("rtp_checkpoint.pt")
    )
    checks = {
        "profile_schema": profile.get("schema") == "poke_bot.submission_runtime_profile/v1",
        "profile_disabled": profile.get("recursive_turn_planner") == "disabled",
        "profile_display": profile.get("display") == "NO RTP",
        "profile_no_sidecar": profile.get("rtp_sidecar_packaged") is False,
        "no_sidecar_member": not forbidden_sidecars,
        "entrypoint_forces_off": 'os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "0"' in main_source,
        "entrypoint_removes_checkpoint": '"POKEBOT_RTP_CHECKPOINT"' in main_source and "os.environ.pop" in main_source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("NO RTP bundle audit failed: " + ",".join(failed))
    return {"passed": True, "checks": checks, "forbidden_sidecars": forbidden_sidecars}


def audit_rtp_bundle(path: Path, expected_sidecar_sha256: str) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = {
            member.name.removeprefix("./"): member
            for member in archive.getmembers()
            if member.isfile()
        }
        profile_member = members.get("runtime_profile.json")
        sidecar_member = members.get("rtp_shadow_planner.pt")
        main_member = members.get("main.py")
        if profile_member is None or sidecar_member is None or main_member is None:
            raise RuntimeError("RTP bundle lacks profile, sidecar, or entrypoint")
        profile_stream = archive.extractfile(profile_member)
        sidecar_stream = archive.extractfile(sidecar_member)
        main_stream = archive.extractfile(main_member)
        if profile_stream is None or sidecar_stream is None or main_stream is None:
            raise RuntimeError("RTP bundle members are unreadable")
        profile = json.loads(profile_stream.read())
        sidecar_digest = "sha256:" + hashlib.sha256(sidecar_stream.read()).hexdigest()
        main_source = main_stream.read().decode("utf-8")
    checks = {
        "profile_schema": profile.get("schema") == "poke_bot.submission_runtime_profile/v1",
        "profile_enabled": profile.get("recursive_turn_planner") == "enabled",
        "profile_display": profile.get("display") == "RTP",
        "profile_sidecar": profile.get("rtp_sidecar_packaged") is True,
        "profile_digest": profile.get("rtp_checkpoint_sha256") == expected_sidecar_sha256,
        "sidecar_digest": sidecar_digest == expected_sidecar_sha256,
        "entrypoint_forces_on": 'os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "1"' in main_source,
        "entrypoint_binds_sidecar": 'os.environ["POKEBOT_RTP_CHECKPOINT"] = str(sidecar)' in main_source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("RTP bundle audit failed: " + ",".join(failed))
    return {"passed": True, "checks": checks, "sidecar_sha256": sidecar_digest}


def materialize_checkpoint_compatible_matchup_tree(
    *, source: Path, checkpoint: Path, output: Path
) -> tuple[Path, dict[str, Any]]:
    """Fail closed to bypass for routes with an exact-zero output projection."""
    tree = read_object(source)
    payload = checkpoint_mod.load_checkpoint(checkpoint, map_location="cpu")
    state = dict(payload.get("model_state_dict") or {})
    runtime = dict(tree.get("runtime_contract") or {})
    targets = [str(value) for value in runtime.get("route_target_ids") or []]
    slots = [int(value) for value in runtime.get("route_physical_slots") or []]
    accepted = [str(value) for value in runtime.get("accepted_archetype_ids") or []]
    if len(targets) != len(slots) or not targets or "alakazam" not in accepted:
        raise RuntimeError("source matchup tree route registry changed")
    slot_by_target = dict(zip(targets, slots, strict=True))
    zero_output: list[str] = []
    verified_nonzero: list[str] = []
    for target in accepted:
        slot = slot_by_target[target]
        tensors = [
            state.get(f"matchup_adapter_bank.experts.{slot}.up.weight"),
            state.get(f"matchup_adapter_bank.experts.{slot}.up.bias"),
        ]
        if any(tensor is None for tensor in tensors):
            raise RuntimeError(f"matchup adapter output tensor missing: {target}@{slot}")
        if any(int(tensor.detach().count_nonzero().item()) > 0 for tensor in tensors):
            verified_nonzero.append(target)
        else:
            zero_output.append(target)
    if "alakazam" not in verified_nonzero:
        raise RuntimeError("Alakazam matchup adapter route is not trained")
    runtime["accepted_archetype_ids"] = verified_nonzero
    confidence = dict(runtime.get("per_archetype_min_leaf_confidence") or {})
    runtime["per_archetype_min_leaf_confidence"] = {
        target: confidence[target] for target in verified_nonzero
    }
    rejected = dict(runtime.get("rejected_archetype_ids") or {})
    for target in zero_output:
        rejected[target] = "checkpoint_exact_zero_output_projection_bypass"
    runtime["rejected_archetype_ids"] = rejected
    runtime["checkpoint"] = str(checkpoint)
    runtime["checkpoint_digest"] = sha256(checkpoint)
    runtime["zero_materialized_adapters_allowed"] = False
    tree["runtime_contract"] = runtime
    body = json.dumps(tree, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != body:
        raise RuntimeError("checkpoint-compatible matchup tree changed")
    if not output.exists():
        output.write_text(body, encoding="utf-8")
    receipt = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "checkpoint_sha256": sha256(checkpoint),
        "accepted_nonzero_routes": verified_nonzero,
        "exact_zero_bypass_routes": zero_output,
        "alakazam_runtime_enabled": "alakazam" in verified_nonzero,
    }
    return output, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "runtime_root", "completion", "contract", "matchup_tree",
        "submission_root", "queue", "python", "receipt",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())

    completion = read_object(args.completion)
    contract = read_object(args.contract)
    if (
        completion.get("schema")
        != "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195_completion/v1"
        or completion.get("status")
        != "completed_25_of_25_ready_for_no_rtp_package"
        or completion.get("parent_checkpoint_sha256") != PARENT_SHA256
        or int(completion.get("epochs_completed") or 0) != 25
        or contract.get("schema")
        != "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195/v1"
        or int(contract.get("owner_decision_revision") or 0) != 195
        or int(contract.get("submission", {}).get("exact_count") or 0) != 2
    ):
        raise RuntimeError("r195 completion/contract is not package-authoritative")
    profiles = list(contract.get("submission", {}).get("profiles") or [])
    if (
        len(profiles) != 2
        or profiles[0].get("copy_number") != 1
        or profiles[0].get("rtp_enabled") is not False
        or profiles[0].get("label_required_literal") != "NO RTP"
        or profiles[1].get("copy_number") != 2
        or profiles[1].get("rtp_enabled") is not True
        or profiles[1].get("label_required_literal") != "RTP"
    ):
        raise RuntimeError("r195 runtime comparison profiles changed")
    checkpoint_path = Path(str(completion.get("checkpoint") or "")).resolve()
    checkpoint_digest = str(completion.get("checkpoint_sha256") or "")
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint_digest:
        raise RuntimeError("r195 trained checkpoint identity changed")

    root = args.submission_root
    deck = materialize_owner_pilot_deck(root / "pinned-alakazam-owner-rtp-pilot-r175.deck.csv")
    compatible_tree, matchup_tree_receipt = materialize_checkpoint_compatible_matchup_tree(
        source=args.matchup_tree,
        checkpoint=checkpoint_path,
        output=root / "pinned-checkpoint-compatible-matchup-tree-v2.json",
    )
    no_rtp_bundle = build_submission_bundle(
        repo_root=args.runtime_root,
        frozen_manifest={
            "model_path": str(checkpoint_path),
            "checkpoint_digest": checkpoint_digest,
        },
        deck_receipt=deck,
        output_dir=root / "build-no-rtp",
        python=args.python,
        archetype="alakazam",
        matchup_tree=compatible_tree,
        turn_order_preference="first_if_allowed",
        rtp_mode="disabled",
    )
    no_rtp_audit = audit_no_rtp_bundle(Path(no_rtp_bundle["path"]))
    rtp_checkpoint = Path(str(profiles[1].get("rtp_checkpoint") or "")).resolve()
    expected_rtp_digest = str(profiles[1].get("rtp_checkpoint_sha256") or "")
    if (
        not rtp_checkpoint.is_file()
        or sha256(rtp_checkpoint) != expected_rtp_digest
        or rtp_checkpoint.stat().st_size != int(profiles[1].get("rtp_checkpoint_bytes") or -1)
    ):
        raise RuntimeError("r195 canonical RTP sidecar identity changed")
    previous_rtp = os.environ.get("POKEBOT_SUBMISSION_RTP_CHECKPOINT")
    os.environ["POKEBOT_SUBMISSION_RTP_CHECKPOINT"] = str(rtp_checkpoint)
    try:
        rtp_bundle = build_submission_bundle(
            repo_root=args.runtime_root,
            frozen_manifest={
                "model_path": str(checkpoint_path),
                "checkpoint_digest": checkpoint_digest,
            },
            deck_receipt=deck,
            output_dir=root / "build-rtp",
            python=args.python,
            archetype="alakazam",
            matchup_tree=compatible_tree,
            turn_order_preference="first_if_allowed",
            rtp_mode="enabled",
        )
    finally:
        if previous_rtp is None:
            os.environ.pop("POKEBOT_SUBMISSION_RTP_CHECKPOINT", None)
        else:
            os.environ["POKEBOT_SUBMISSION_RTP_CHECKPOINT"] = previous_rtp
    rtp_audit = audit_rtp_bundle(Path(rtp_bundle["path"]), expected_rtp_digest)
    copy_no_rtp = _copy_submission_slot(no_rtp_bundle, root, 1)
    copy_no_rtp["label_suffix"] = "NO RTP"
    copy_rtp = _copy_submission_slot(rtp_bundle, root, 2)
    copy_rtp["label_suffix"] = "RTP"
    queued = queue_submission_copies(
        queue_path=args.queue,
        copies=[copy_no_rtp, copy_rtp],
        gate_plan={
            "checkpoint_digest": checkpoint_digest,
            "gate_id": "alakazam-terminal-expert-bootstrap-r195-runtime-comparison",
            "iteration": 21,
            "completion_authority": "training_milestone_snapshot",
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    if (
        len(queued) != 2
        or not str(queued[0].get("label") or "").endswith("NO RTP")
        or not str(queued[1].get("label") or "").endswith("RTP")
        or str(queued[1].get("label") or "").endswith("NO RTP")
    ):
        raise RuntimeError("queued r195 submission messages do not show NO RTP/RTP")
    receipt = {
        "schema": SCHEMA,
        "status": "queued",
        "owner_decision_revision": 195,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": str(args.completion),
        "completion_sha256": sha256(args.completion),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "deck": deck,
        "matchup_tree": matchup_tree_receipt,
        "bundles": {"no_rtp": no_rtp_bundle, "rtp": rtp_bundle},
        "no_rtp_audit": no_rtp_audit,
        "rtp_audit": rtp_audit,
        "queue_entries": queued,
        "submission_messages": [row["label"] for row in queued],
        "submitted_runtime_profiles": ["NO RTP", "RTP"],
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
