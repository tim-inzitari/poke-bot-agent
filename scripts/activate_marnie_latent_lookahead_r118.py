#!/usr/bin/env python3
"""Activate the trained Marnie latent policy aid at an owner-requested boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.train import load_model_from_checkpoint  # noqa: E402

SCHEMA = "poke_bot.marnie_latent_lookahead_owner_activation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def activate(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
    parent = args.parent.expanduser().resolve()
    loop_state_path = args.loop_state.expanduser().resolve()
    output = args.output.expanduser().resolve()
    receipt = args.receipt.expanduser().resolve()
    backup = args.loop_state_backup.expanduser().resolve()
    for path in (source, parent, loop_state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or receipt.exists() or backup.exists():
        raise FileExistsError("revision-118 activation outputs are immutable")

    source_payload = checkpoint.load_checkpoint(source, map_location="cpu")
    source_config = dict(source_payload.get("model_config") or {})
    if source_config.get("latent_lookahead_enabled") is not True:
        raise RuntimeError("trained source has no latent lookahead")
    if source_config.get("latent_lookahead_action_authority_enabled") is not False:
        raise RuntimeError("trained source is not the authority-off shadow")
    if float(source_config.get("latent_lookahead_policy_aid_cap", -1.0)) != 0.25:
        raise RuntimeError("latent policy-aid cap changed")

    parent_payload = checkpoint.load_checkpoint(parent, map_location="cpu")
    source_state = source_payload["model_state_dict"]
    parent_state = parent_payload["model_state_dict"]
    for name, tensor in parent_state.items():
        if name not in source_state or not torch.equal(tensor, source_state[name]):
            raise RuntimeError(f"protected parent tensor changed: {name}")
    latent_keys = sorted(name for name in source_state if name.startswith("latent_lookahead."))
    if len(latent_keys) != 12:
        raise RuntimeError(f"expected 12 latent tensors, got {len(latent_keys)}")

    loop_state = json.loads(loop_state_path.read_text())
    if int(loop_state.get("last_completed_iteration", -1)) != 5 or int(loop_state.get("next_iteration", -1)) != 6:
        raise RuntimeError("activation requires the exact committed iteration-5 to 6 boundary")
    learner = dict(loop_state.get("learner") or {})
    if learner.get("digest") != _sha256(parent) or Path(str(learner.get("path") or "")).resolve() != parent:
        raise RuntimeError("loop learner is not the checksum-exact protected parent")

    payload = copy.deepcopy(source_payload)
    payload["model_config"] = source_config | {
        "latent_lookahead_action_authority_enabled": True,
    }
    payload.pop("optimizer_state_dict", None)
    payload.pop("scaler_state_dict", None)
    extra = dict(payload.get("extra") or {})
    extra["latent_lookahead_owner_activation"] = {
        "schema": SCHEMA,
        "owner_decision_revision": 118,
        "accepted_policy_generation": 15,
        "boundary": {"last_completed_iteration": 5, "restarted_iteration": 6},
        "source_shadow": str(source),
        "source_shadow_digest": _sha256(source),
        "protected_parent": str(parent),
        "protected_parent_digest": _sha256(parent),
        "action_authority_enabled": True,
        "policy_aid_cap": 0.25,
        "terminal_strength_gate_unchanged": 0.80,
        "rollback_parent_preserved": True,
        "post_activation_nonregression_monitor_required": True,
        "mcts_allowed": False,
        "beam_search_allowed": False,
        "competition_time_simulator_search_allowed": False,
    }
    payload["extra"] = extra
    provenance = dict(payload.get("provenance") or {})
    provenance["accepted_policy_generation"] = 15
    provenance["owner_decision_revision"] = 118
    payload["provenance"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.immutable_torch_save(payload, output)

    loaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    inventory = loaded.latent_lookahead_inventory()
    if inventory.get("enabled") is not True or inventory.get("action_authority_enabled") is not True:
        raise RuntimeError("published checkpoint did not reconstruct with latent authority")
    output_payload = checkpoint.load_checkpoint(output, map_location="cpu")
    for name, tensor in source_state.items():
        if not torch.equal(tensor, output_payload["model_state_dict"][name]):
            raise RuntimeError(f"activation changed tensor bytes: {name}")

    _exclusive_json(backup, loop_state)
    digest = _sha256(output)
    updated = copy.deepcopy(loop_state)
    updated["learner"] = {"path": str(output), "digest": digest}
    updated["accepted_policy_generation"] = 15
    updated["owner_boundary_activation"] = {
        "schema": SCHEMA,
        "owner_decision_revision": 118,
        "receipt": str(receipt),
        "checkpoint": str(output),
        "checkpoint_digest": digest,
    }
    updated["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(loop_state_path, updated)

    report = {
        "schema": SCHEMA,
        "status": "activated_for_restarted_iteration_6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_decision_revision": 118,
        "accepted_policy_generation": 15,
        "boundary": {"last_completed_iteration": 5, "restarted_iteration": 6},
        "source_shadow": {"path": str(source), "digest": _sha256(source)},
        "protected_parent": {"path": str(parent), "digest": _sha256(parent)},
        "activated_checkpoint": {"path": str(output), "digest": digest},
        "loop_state_backup": {"path": str(backup), "digest": _sha256(backup)},
        "proof": {
            "all_source_tensors_bit_identical": True,
            "all_protected_parent_tensors_bit_identical": True,
            "only_model_config_authority_flag_changed": True,
            "optimizer_state_reset": True,
            "policy_aid_cap": 0.25,
            "latent_tensor_count": len(latent_keys),
            "model_parameters": sum(t.numel() for t in output_payload["model_state_dict"].values()),
            "rollback_parent_preserved": True,
        },
        "architecture": inventory,
    }
    _exclusive_json(receipt, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--loop-state-backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(activate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
