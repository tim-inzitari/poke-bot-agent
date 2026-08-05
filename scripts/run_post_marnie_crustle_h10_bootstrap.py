#!/usr/bin/env python3
"""Bootstrap the Revision-113 H10 Crustle specialist after Marnie completes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from poke_bot import checkpoint
from scripts.run_starmie_expert_bootstrap import (
    TARGETS,
    atomic_json,
    load_expanded_head_contract,
    main as bootstrap_main,
)
from scripts.validate_future_specialist_strategic_curriculum import materialize


SPECIALIST_ID = "crustle"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marnie-completion", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--marnie-opponent-stage", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--guide-ready", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--pilot-importance-index", type=Path, required=True)
    parser.add_argument("--curriculum-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    completion = read_json(args.marnie_completion.resolve())
    opponent_stage = read_json(args.marnie_opponent_stage.resolve())
    training_freeze = read_json(
        Path(str(opponent_stage.get("marnie_training_freeze") or "")).resolve()
    ) if opponent_stage.get("marnie_training_freeze") else None
    registry = read_json(args.refresh_registry.resolve())
    rows = [dict(row) for row in registry.get("refreshes") or []]
    marnie = next(
        (row for row in rows if row.get("specialist_id") == "marnie-s-grimmsnarl-ex"),
        None,
    )
    if (
        completion.get("schema")
        != "poke_bot.post_fleet_specialist_refresh_completion/v1"
        or completion.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or completion.get("frozen") is not True
        or completion.get("registered") is not True
        or int(dict(completion.get("training_contract") or {}).get("terminal_iteration", 20))
        != 20
        or marnie is None
        or marnie.get("checkpoint_checksum")
        != completion.get("refresh_checkpoint_checksum")
        or opponent_stage.get("schema")
        != "poke_bot.crustle_marnie_expert_practice_stage/v1"
        or opponent_stage.get("status")
        != "ready_for_post_marnie_crustle_bootstrap"
        or (
            opponent_stage.get("checkpoint_digest")
            != completion.get("refresh_checkpoint_checksum")
            and training_freeze is None
        )
        or opponent_stage.get("marnie_completion_sha256")
        != sha256(args.marnie_completion.resolve())
    ):
        raise RuntimeError("Marnie iteration-20 completion is not authoritative")
    parent = Path(
        str((training_freeze or {}).get("checkpoint") or marnie.get("checkpoint") or "")
    ).resolve()
    parent_digest = str(
        (training_freeze or {}).get("checkpoint_sha256")
        or marnie.get("checkpoint_checksum")
        or ""
    )
    parent_family = (
        args.registry_root.resolve() / "marnie-iter9-training-freeze-r163"
        if training_freeze is not None
        else (parent.parent.parent if parent.parent.name == "checkpoints" else parent.parent)
    )
    if not parent.is_file() or checkpoint.checkpoint_digest(parent) != parent_digest:
        raise RuntimeError("Marnie H10 parent checkpoint changed")

    guide = args.guide.resolve()
    guide_ready = args.guide_ready.resolve()
    expert = args.expert.resolve()
    for path in (
        guide,
        guide_ready,
        expert,
        args.pilot_importance_index.resolve(),
        args.protocol.resolve(),
    ):
        if not path.is_file():
            raise RuntimeError(f"Crustle bootstrap input is missing: {path}")
    guide_payload = __import__("yaml").safe_load(guide.read_text(encoding="utf-8"))
    if (
        guide_payload.get("specialist_id") != SPECIALIST_ID
        or guide_payload.get("guide_version") != "crustle-north-star-v2"
    ):
        raise RuntimeError("Crustle guide identity changed")

    curriculum = materialize(
        specialist_id=SPECIALIST_ID,
        guide_contract=guide,
        guide_ready_receipt=guide_ready,
        output_root=args.curriculum_root.resolve(),
        training_implementation=(Path(__file__).resolve().parents[1] / "poke_bot/train.py"),
        training_mode="strategic_directional_v2",
        include_combo_state=True,
    )
    expanded_raw, expanded = load_expanded_head_contract(args.protocol.resolve())
    del expanded_raw
    argv = [
        "--expert-corpus", str(expert),
        "--pilot-importance-index", str(args.pilot_importance_index.resolve()),
        "--archetype", SPECIALIST_ID,
        "--family", args.family,
        "--display-name", "Crustle Final-Format H10 Specialist",
        "--core-family", str(parent_family),
        "--registry-root", str(args.registry_root.resolve()),
        "--ready", str(args.ready.resolve()),
        "--run-name", "final_format_crustle_r113_h10_bootstrap",
        "--run-dir", str(args.run_dir.resolve()),
        "--epochs", "35",
        "--owner-crustle-guide-active-epochs", "35",
        "--owner-crustle-guide-free-refresh-epochs", "0",
        "--batch-size", str(args.batch_size),
        "--min-decisions", "100000",
        "--cpu-pack-root", str(args.cpu_pack_root.resolve()),
        "--expanded-heads",
        "--decision-fusion",
        "--allow-h10-specialist-parent",
        "--retain-inherited-h10-combo-state-head",
        "--rl-protocol", str(args.protocol.resolve()),
        "--expected-expanded-schedule-digest", str(expanded["schedule_digest"]),
        "--expected-expanded-target-digest", str(expanded["target_schema_digest"]),
        "--current-deck-guide-contract", str(guide),
        "--expected-current-deck-guide-sha256", sha256(guide),
        "--current-deck-guide-version", "crustle-north-star-v2",
        "--current-deck-guide-corpus-ready", str(guide_ready),
        "--expected-current-deck-guide-corpus-ready-sha256", sha256(guide_ready),
        "--strategic-curriculum-spec", str(curriculum["curriculum_spec"]),
        "--strategic-curriculum-spec-sha256", sha256(curriculum["curriculum_spec"]),
        "--strategic-head-role-map", str(curriculum["head_role_map"]),
        "--strategic-head-role-map-sha256", sha256(curriculum["head_role_map"]),
        "--strategic-validation-receipt", str(curriculum["validation_receipt"]),
        "--strategic-validation-receipt-sha256", sha256(curriculum["validation_receipt"]),
    ]
    for target in TARGETS:
        argv.extend(("--required-target", target))
    result = bootstrap_main(argv)
    if result != 0:
        return result
    ready = read_json(args.ready.resolve())
    anchor = {
        "schema": "poke_bot.crustle_marnie_expert_practice_anchor/v1",
        "owner_decision_revision": 163 if training_freeze is not None else 162,
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "checkpoint": str(parent),
        "checkpoint_digest": parent_digest,
        "completion_receipt": str(args.marnie_completion.resolve()),
        "completion_receipt_sha256": sha256(args.marnie_completion.resolve()),
        "submission_bundle": (training_freeze or {}).get("submission_bundle") or marnie.get("submission_bundle"),
        "submission_bundle_sha256": (training_freeze or {}).get("submission_bundle_sha256") or marnie.get("submission_bundle_sha256"),
        "roles": [
            "checksum_bound_h10_predecessor",
            "required_crustle_public_practice_opponent",
            "required_crustle_formal_holdout_opponent",
        ],
        "actions_relabelled_as_crustle_expert_targets": False,
        "opponent_stage_receipt": str(args.marnie_opponent_stage.resolve()),
        "opponent_stage_receipt_sha256": sha256(
            args.marnie_opponent_stage.resolve()
        ),
        "opponent_id": opponent_stage.get("opponent_id"),
        "frozen_registry_sha256": opponent_stage.get("frozen_registry_sha256"),
        "active_gate_sha256": opponent_stage.get("active_gate_sha256"),
    }
    if (
        ready.get("schema")
        != "poke_bot.specialist_expert_bootstrap_ready/v1"
        or int(ready.get("epochs_completed") or 0) != 35
        or ready.get("hot_start_checkpoint_digest") != parent_digest
        or not Path(str(anchor["submission_bundle"] or "")).is_file()
        or sha256(Path(str(anchor["submission_bundle"])).resolve())
        != anchor["submission_bundle_sha256"]
    ):
        raise RuntimeError("Crustle Marnie expert/practice anchor is invalid")
    ready["marnie_expert_practice_anchor"] = anchor
    atomic_json(args.ready.resolve(), ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
