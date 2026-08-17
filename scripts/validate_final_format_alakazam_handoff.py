#!/usr/bin/env python3
"""Fail-closed preflight for the post-Slowking final-format Alakazam bridge.

Static validation is safe before the Slowking terminal boundary: it reads contracts and
checksums only.  ``--require-release-boundary`` additionally proves that the
canonical 15-slot fleet is terminally resolved as 14 frozen specialists plus
the explicit revision-79 failed-experiment Slowking exception, and that the
managed specialist trainer is inactive. It still does not load a checkpoint, instantiate a model, scan
replays, or write a release receipt; those actions belong to the separately
managed refresh state machine after G0/G1 authority exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.model_registry import sha256  # noqa: E402
from scripts.run_specialist_cycle_handoff import (  # noqa: E402
    _required_specialist_ids,
    _validated_post_fleet_refresh_progress,
)


STATIC_SCHEMA = "poke_bot.final_format_alakazam_static_preparation/v1"
VALIDATION_SCHEMA = (
    "poke_bot.final_format_alakazam_static_preparation_validation/v12"
)
FROZEN_REGISTRY_SCHEMA = "poke_bot.frozen_specialist_registry/v1"
EXPECTED_PARENT = (
    "sha256:270b5156781b0a95f703abe3e8fe13866"
    "d2fbb4c85a8f32534f99af74aece2ea"
)
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "sha256:f7a0923fcf2f936821b97790bf1644520"
    "5051b5a22b4029e225b64dfc3f151d0"
)
EXPECTED_SOURCE_RUN_MANIFEST_SHA256 = (
    "sha256:0aa2205f65cd63ccaa9bee74f440084b7"
    "c045ea1a37a686df92e9a26cb22b87f"
)
EXPECTED_PARENT_MODEL_CONFIG_SHA256 = (
    "sha256:cdad51d0bebf836979aeef2f12963abd9"
    "5b5c4dcbd5df7ba0f59989bb5c2c5a7"
)
PARENT_MANIFEST_SCHEMA = "poke_bot.frozen_model/v1"
PARENT_MODEL_CONFIG_PROJECTION_SCHEMA = (
    "poke_bot.parent_model_profile_projection/v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing or corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value or "")).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_preferred_parent_metadata(manifest_path: Path) -> dict[str, Any]:
    """Bind the historical parent config without loading model tensors.

    The frozen-model manifest points back to the source run, whose immutable
    run manifest records the exact model profile used to instantiate the
    accepted checkpoint.  Hashing that profile's canonical JSON is a metadata
    operation; it neither imports torch nor opens ``model.pt``.
    """

    manifest = _read_json(manifest_path)
    manifest_digest = sha256(manifest_path)
    if (
        manifest.get("schema") != PARENT_MANIFEST_SCHEMA
        or manifest.get("immutable") is not True
        or manifest.get("checkpoint_digest") != EXPECTED_PARENT
        or manifest.get("family") != "alakazam-owner-accepted-iter39-v1"
        or manifest_digest != EXPECTED_PARENT_MANIFEST_SHA256
    ):
        raise RuntimeError("preferred Alakazam frozen manifest changed")
    source_run = Path(
        str((manifest.get("provenance") or {}).get("source_run") or "")
    ).expanduser()
    source_manifest_path = source_run / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    source_manifest_digest = sha256(source_manifest_path)
    model_profile = source_manifest.get("model_profile")
    model_config_digest = _canonical_json_sha256(model_profile)
    if (
        source_manifest.get("run_name")
        != "pure_rl_alakazam_temporal1_8k_teacher_v16_20260721"
        or not isinstance(model_profile, dict)
        or not model_profile
        or model_profile
        != (source_manifest.get("design_contract") or {})
        .get("learner", {})
        .get("profile")
        or source_manifest_digest != EXPECTED_SOURCE_RUN_MANIFEST_SHA256
        or model_config_digest != EXPECTED_PARENT_MODEL_CONFIG_SHA256
    ):
        raise RuntimeError("preferred Alakazam source model profile changed")
    return {
        "checkpoint_sha256": EXPECTED_PARENT,
        "frozen_manifest_sha256": manifest_digest,
        "source_run_manifest_sha256": source_manifest_digest,
        "model_config_projection_schema": (
            PARENT_MODEL_CONFIG_PROJECTION_SCHEMA
        ),
        "model_config_sha256": model_config_digest,
        "checkpoint_loaded": False,
        "model_instantiated": False,
    }


def validate_static(
    *,
    root: Path,
    static_contract_path: Path,
    validation_receipt_path: Path,
    cycle_contract_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    static = _read_json(static_contract_path)
    validation = _read_json(validation_receipt_path)
    cycle = _read_json(cycle_contract_path)
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError("canonical specialist state is not an object")
    phase = dict(state.get("post_fleet_refresh") or {})
    bridge = dict(static.get("bridge") or {})
    authority = dict(static.get("authority") or {})
    if (
        static.get("schema") != STATIC_SCHEMA
        or static.get("status")
        != "static_preparation_complete_slowking_failed_experiment_boundary_reached_waiting_for_g0_g1"
        or validation.get("schema") != VALIDATION_SCHEMA
        or validation.get("status")
        != "revalidated_against_revision105_h10_source_snapshot"
        or validation.get("static_contract_sha256")
        != sha256(static_contract_path)
        or bridge.get("specialist_id") != "alakazam"
        or bridge.get("model_format") != "final_submission_format"
        or bridge.get("start_after_terminal_disposition")
        != "slowking_failed_experiment"
        or int(bridge.get("required_fleet_count") or 0) != 15
        or int(bridge.get("required_frozen_specialist_count") or 0) != 14
        or int(bridge.get("terminal_failed_experiment_exception_count") or 0)
        != 1
        or bridge.get("preferred_parent_checkpoint_sha256") != EXPECTED_PARENT
        or bridge.get("training_seat_split")
        != {"first": 0.5, "second": 0.5}
        or bridge.get("package_preference") != "first_if_allowed"
        or bridge.get("second_focus_1_to_7_allowed") is not False
        or bridge.get("guide_runtime_route_count") != 0
        or authority.get("static_implementation") is not True
        or any(
            authority.get(key) is not False
            for key in (
                "model_computation",
                "checkpoint_loading",
                "model_instantiation",
                "replay_scanning",
                "corpus_materialization",
                "benchmarking",
                "training",
                "evaluation",
                "package_construction",
                "submission_construction",
            )
        )
        or phase.get("static_preparation", {}).get("contract_checksum")
        != sha256(static_contract_path)
        or phase.get("static_preparation", {}).get(
            "validation_receipt_checksum"
        )
        != sha256(validation_receipt_path)
    ):
        raise RuntimeError("final-format Alakazam static contract changed")

    checked: dict[str, str] = {}
    for section in ("specifications", "templates"):
        rows = static.get(section)
        if not isinstance(rows, dict) or not rows:
            raise RuntimeError(f"static contract has no {section}")
        for artifact_id, raw in rows.items():
            row = dict(raw or {})
            path = _resolve(root, row.get("path"))
            digest = str(row.get("sha256") or "")
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError(
                    f"final-format static artifact changed: {artifact_id}"
                )
            checked[str(artifact_id)] = digest

    source_compatibility = _read_json(
        _resolve(
            root,
            static["specifications"][
                "final_format_alakazam_source_compatibility_v1"
            ]["path"],
        )
    )
    for relative, expected in dict(
        source_compatibility.get("source_files") or {}
    ).items():
        source = _resolve(root, relative)
        if not source.is_file() or sha256(source) != expected:
            raise RuntimeError(
                f"final-format source compatibility changed: {relative}"
            )

    order, completed = _validated_post_fleet_refresh_progress(
        state_path=state_path,
        cycle_contract=cycle,
    )
    if order != ["alakazam", "marnie-s-grimmsnarl-ex", "crustle"]:
        raise RuntimeError("post-fleet refresh order changed")
    return {
        "static_contract_sha256": sha256(static_contract_path),
        "validation_receipt_sha256": sha256(validation_receipt_path),
        "checked_artifact_count": len(checked),
        "refresh_order": order,
        "completed_refresh_specialist_ids": completed,
        "model_or_replay_computation_performed": False,
    }


def _service_active(service: str) -> bool:
    completed = subprocess.run(
        ["/usr/bin/systemctl", "--user", "is-active", "--quiet", service],
        check=False,
    )
    return completed.returncode == 0


def validate_release_boundary(
    *,
    state_path: Path,
    frozen_registry_path: Path,
    failed_experiment_receipt_path: Path,
    training_service: str,
) -> dict[str, Any]:
    if _service_active(training_service):
        raise RuntimeError("Slowking managed trainer is still active")
    required = _required_specialist_ids(state_path)
    registry = _read_json(frozen_registry_path)
    if registry.get("schema") != FROZEN_REGISTRY_SCHEMA:
        raise RuntimeError("frozen specialist registry schema changed")
    rows = list(registry.get("specialists") or [])
    frozen = {
        str(row.get("specialist_id") or ""): dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("frozen") is True
    }
    failed = _read_json(failed_experiment_receipt_path)
    expected_frozen = required - {"slowking"}
    disposition = dict(failed.get("terminal_disposition") or {})
    if (
        len(required) != 15
        or set(frozen) != expected_frozen
        or len(frozen) != 14
        or failed.get("schema")
        != "poke_bot.specialist_failed_experiment_transition/v1"
        or failed.get("status") != "activated"
        or failed.get("goal_revision") != 79
        or failed.get("specialist_id") != "slowking"
        or disposition.get("classification") != "failed_experiment"
        or disposition.get("passing_status_granted") is not False
        or disposition.get("completion_credit_granted") is not False
        or disposition.get("frozen_registry_contains_slowking") is not False
    ):
        raise RuntimeError("revision-79 terminal fleet disposition is invalid")
    return {
        "required_specialist_count": len(required),
        "frozen_specialist_count": len(frozen),
        "terminal_failed_experiment_exception_count": 1,
        "slowking_terminal_disposition": "failed_experiment",
        "slowking_failed_experiment_receipt_sha256": sha256(
            failed_experiment_receipt_path
        ),
        "frozen_registry_sha256": sha256(frozen_registry_path),
        "training_service_inactive": True,
        "checkpoint_loaded": False,
        "model_instantiated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT
    )
    parser.add_argument(
        "--static-contract",
        type=Path,
        default=Path(
            "ops/final_submission/final_format_alakazam_static_preparation_v1.json"
        ),
    )
    parser.add_argument(
        "--validation-receipt",
        type=Path,
        default=Path("state/final_format_alakazam_static_preparation_validation_v11.json"),
    )
    parser.add_argument(
        "--cycle-contract",
        type=Path,
        default=Path("ops/specialist_cycle_handoff_v1.json"),
    )
    parser.add_argument(
        "--state", type=Path, default=Path("state/specialists.yaml")
    )
    parser.add_argument(
        "--frozen-registry",
        type=Path,
        default=Path("ops/frozen_specialist_registry_v1.json"),
    )
    parser.add_argument(
        "--training-service",
        default="pokebot-pure-rl-trevenant-staged.service",
    )
    parser.add_argument(
        "--failed-experiment-receipt",
        type=Path,
        default=Path("state/slowking_failed_experiment_transition_r79.json"),
    )
    parser.add_argument(
        "--preferred-parent-manifest",
        type=Path,
        default=Path(
            "/home/pokebot/poke-bot-agent/outputs/pure_rl/_protected/models/"
            "alakazam-owner-accepted-iter39-v1/manifest.json"
        ),
    )
    parser.add_argument("--require-release-boundary", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    resolve = lambda value: _resolve(root, value)
    result = validate_static(
        root=root,
        static_contract_path=resolve(args.static_contract),
        validation_receipt_path=resolve(args.validation_receipt),
        cycle_contract_path=resolve(args.cycle_contract),
        state_path=resolve(args.state),
    )
    if args.require_release_boundary:
        result["release_boundary"] = validate_release_boundary(
            state_path=resolve(args.state),
            frozen_registry_path=resolve(args.frozen_registry),
            failed_experiment_receipt_path=resolve(
                args.failed_experiment_receipt
            ),
            training_service=str(args.training_service),
        )
        result["preferred_parent_metadata"] = (
            validate_preferred_parent_metadata(
                resolve(args.preferred_parent_manifest)
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
