#!/usr/bin/env python3
"""Issue the post-refresh capacity barrier, then release population handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from poke_bot import checkpoint
from scripts.register_final_format_marnie_h10_rl import _validate_checkpoint


SCHEMA = "poke_bot.post_refresh_sequence_complete_for_capacity/v2"
COMPLETION_SCHEMA = "poke_bot.post_fleet_specialist_refresh_completion/v1"
REGISTRY_SCHEMA = "poke_bot.post_fleet_refresh_registry/v1"
CORE_BOUNDARY_SCHEMA = "poke_bot.post_alakazam_core_refresh_boundary/v1"
ORIGINALS = {
    "alakazam": "sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f32534f99af74aece2ea",
    "marnie-s-grimmsnarl-ex": "sha256:52a5207e4c98dce80b49b6403cbb17f14d6fc4d2ac5b625532020a1a25f233ac",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    body = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"capacity boundary receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _service_state(service: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", service, "-p", "ActiveState", "--value"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect managed service {service}")
    return result.stdout.strip()


def _static(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "original_alakazam": args.original_alakazam.expanduser().resolve(),
        "original_marnie": args.original_marnie.expanduser().resolve(),
        "next_unit": args.next_unit.expanduser().resolve(),
        "protocol": args.protocol.expanduser().resolve(),
        "state": args.specialist_state.expanduser().resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("post-refresh boundary is missing: " + ",".join(missing))
    if _sha256(paths["original_alakazam"]) != ORIGINALS["alakazam"]:
        raise RuntimeError("historical Alakazam checkpoint changed")
    if _sha256(paths["original_marnie"]) != ORIGINALS["marnie-s-grimmsnarl-ex"]:
        raise RuntimeError("historical Marnie checkpoint changed")
    return {name: _sha256(path) for name, path in paths.items()}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    static = _static(args)
    completion_paths = {
        "alakazam": args.alakazam_completion.expanduser().resolve(),
        "marnie-s-grimmsnarl-ex": args.marnie_completion.expanduser().resolve(),
        "crustle": args.crustle_completion.expanduser().resolve(),
    }
    completions = {name: _read(path) for name, path in completion_paths.items()}
    registry_path = args.refresh_registry.expanduser().resolve()
    registry = _read(registry_path)
    boundary_path = args.core_boundary.expanduser().resolve()
    boundary = _read(boundary_path)
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("historical_frozen_registry_modified") is not False
        or registry.get("ordered_refresh_ids")
        != ["alakazam", "marnie-s-grimmsnarl-ex", "crustle"]
        or boundary.get("schema") != CORE_BOUNDARY_SCHEMA
        or boundary.get("status") != "selected_core_ready_for_marnie_h10"
        or boundary.get("normal_core_refresh_attempted") is not True
        or boundary.get("rejected_candidate_blocks_production") is not False
    ):
        raise RuntimeError("post-refresh registry/core disposition is invalid")
    rows = [dict(row) for row in registry.get("refreshes") or []]
    if [row.get("specialist_id") for row in rows] != [
        "alakazam",
        "marnie-s-grimmsnarl-ex",
        "crustle",
    ]:
        raise RuntimeError("both ordered refreshes are not registered")

    inventories: dict[str, Any] = {}
    for row in rows:
        specialist_id = str(row["specialist_id"])
        completion = completions[specialist_id]
        model = Path(str(row.get("checkpoint") or "")).resolve()
        digest = str(row.get("checkpoint_checksum") or "")
        manifest = Path(str(row.get("frozen_manifest") or "")).resolve()
        if (
            completion.get("schema") != COMPLETION_SCHEMA
            or completion.get("specialist_id") != specialist_id
            or completion.get("frozen") is not True
            or completion.get("registered") is not True
            or completion.get("refresh_checkpoint_checksum") != digest
            or (
                specialist_id in ORIGINALS
                and completion.get("original_checkpoint_checksum")
                != ORIGINALS[specialist_id]
            )
            or completion.get("completion_authority")
            not in {"measured_gate_pass", "explicit_owner_ceiling_acceptance"}
            or not model.is_file()
            or checkpoint.checkpoint_digest(model) != digest
            or not manifest.is_file()
            or _sha256(manifest) != row.get("frozen_manifest_sha256")
        ):
            raise RuntimeError(f"refresh completion identity failed: {specialist_id}")
        _validate_checkpoint(model, digest)
        payload = checkpoint.load_checkpoint(model, map_location="cpu")
        config = dict(payload.get("model_config") or {})
        fusion = dict((payload.get("provenance") or {}).get("decision_fusion") or {})
        inventories[specialist_id] = {
            "checkpoint": str(model),
            "checkpoint_sha256": digest,
            "completion_receipt": str(completion_paths[specialist_id]),
            "completion_receipt_sha256": _sha256(completion_paths[specialist_id]),
            "frozen_manifest": str(manifest),
            "frozen_manifest_sha256": _sha256(manifest),
            "capacity_profile": "H10-I/v1",
            "architecture": {
                "spatial_layers": int(config["spatial_layers"]),
                "temporal_layers": int(config["temporal_layers"]),
                "option_decoder_layers": int(config["option_decoder_layers"]),
                "feed_forward_width": int(config["ff_dim"]),
                "strategic_head_residual_width": int(
                    config["h10_head_residual_width"]
                ),
            },
            "decision_fusion_schema": fusion.get("schema"),
            "learned_head_count": len(fusion.get("required_heads") or []),
            "learned_route_count": len(
                dict(fusion.get("dedicated_routes") or {}).get("route_names") or []
            ),
            "guide_runtime_route_count": 0,
        }

    state_path = args.specialist_state.expanduser().resolve()
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    phase = dict(state.get("post_fleet_refresh") or {})
    if (
        phase.get("status") != "refresh_sequence_complete_population_handoff_pending"
        or phase.get("completed_refresh_specialist_ids")
        != ["alakazam", "marnie-s-grimmsnarl-ex", "crustle"]
        or list(phase.get("pending_specialist_ids") or [])
        or phase.get("active_refresh_specialist_id") is not None
        or phase.get("next_refresh_specialist_id") is not None
    ):
        raise RuntimeError("post-fleet refresh state is not terminal")
    active_training = {
        name: _service_state(name)
        for name in (
            "pokebot-final-format-alakazam-r79-h10.service",
            "pokebot-final-format-marnie-r104-h10-rl.service",
            "pokebot-final-format-crustle-r113-h10-rl.service",
        )
    }
    if any(value not in {"inactive", "failed"} for value in active_training.values()):
        raise RuntimeError("a refresh trainer is still active")

    receipt = {
        "schema": SCHEMA,
        "status": "post_refresh_sequence_complete_for_capacity_v2",
        "authorization_scope": "post_refresh_capacity_and_population_handoff",
        "ordered_refresh_ids": [
            "alakazam",
            "marnie-s-grimmsnarl-ex",
            "crustle",
        ],
        "refresh_registry": str(registry_path),
        "refresh_registry_sha256": _sha256(registry_path),
        "historical_checkpoints_preserved": {
            "alakazam": {
                "checkpoint": str(args.original_alakazam.expanduser().resolve()),
                "checkpoint_sha256": ORIGINALS["alakazam"],
            },
            "marnie-s-grimmsnarl-ex": {
                "checkpoint": str(args.original_marnie.expanduser().resolve()),
                "checkpoint_sha256": ORIGINALS["marnie-s-grimmsnarl-ex"],
            },
            "crustle": {
                "historical_public_baseline_preserved": True,
                "public_baseline_id": "pilkwang-meta-20260708",
                "public_baseline_substituted_for_new_h10_specialist": False,
            },
        },
        "refresh_inventories": inventories,
        "intervening_core_disposition": {
            "receipt": str(boundary_path),
            "receipt_sha256": _sha256(boundary_path),
            "attempted_core_generation": boundary.get("attempted_core_generation"),
            "attempt_accepted": boundary.get("attempt_accepted"),
            "selected_core_generation": boundary.get("selected_core_generation"),
            "selected_core_checkpoint_sha256": boundary.get(
                "selected_core_checkpoint_sha256"
            ),
            "terminally_resolved": True,
        },
        "active_training_services": active_training,
        "pending_specialist_refresh_ids": [],
        "active_specialist_refresh_id": None,
        "pending_cumulative_core_transition": False,
        "guide_runtime_route_count": 0,
        "pending_kaggle_submissions_block": False,
        "training_contract": {
            "canonical_source": "config/rl_protocol.yaml",
            "sha256": static["protocol"],
        },
        "training_authority": False,
        "selector_authority": False,
        "next_service": args.next_service,
    }
    receipt_path = args.receipt.expanduser().resolve()
    _write_once(receipt_path, receipt)
    started = subprocess.run(
        ["systemctl", "--user", "start", args.next_service],
        check=False,
        capture_output=True,
        text=True,
    )
    if started.returncode:
        raise RuntimeError(
            "could not start population handoff: "
            + started.stdout.strip()
            + started.stderr.strip()
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--alakazam-completion", type=Path, required=True)
    parser.add_argument("--marnie-completion", type=Path, required=True)
    parser.add_argument("--crustle-completion", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--core-boundary", type=Path, required=True)
    parser.add_argument("--original-alakazam", type=Path, required=True)
    parser.add_argument("--original-marnie", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--specialist-state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--next-service", required=True)
    parser.add_argument("--next-unit", type=Path, required=True)
    args = parser.parse_args()
    static = _static(args)
    if args.check:
        print("POST_REFRESH_CAPACITY_BOUNDARY_OK " + json.dumps(static, sort_keys=True))
        return 0
    print(json.dumps(finalize(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
