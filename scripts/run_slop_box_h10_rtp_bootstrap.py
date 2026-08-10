#!/usr/bin/env python3
"""Launch Slop Box H10 expert bootstrap and the initial RTP cut.

Owner revision 170:
  - Full teal-mask-ogerpon-ex / Slop Box acting-seat expert corpus for training
  - Cox/Chao ≥90% held policy_acc gate (documented in identity receipt)
  - Dual pipeline: guide RL-only strategic_directional_v2; RTP neural-only
  - Expert-bootstrap phase MUST produce initial rtp_shadow_planner.pt bound to
    the Slop Box H10 parent (do not wait for later pure-RL cotrain only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.expert_pilot_importance import (  # noqa: E402
    load_training_weights_for_corpus,
)
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_registry import verify_frozen_model  # noqa: E402
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS  # noqa: E402
from scripts.run_starmie_expert_bootstrap import (  # noqa: E402
    TARGETS,
    _specialist_hot_start_from_core,
    atomic_json,
    current_deck_guide_handoff_contract,
    load_expanded_head_contract,
    main as bootstrap_main,
)
from scripts.validate_future_specialist_strategic_curriculum import (  # noqa: E402
    materialize,
)

SPECIALIST_ID = "teal-mask-ogerpon-ex"
IDENTITY_SCHEMA = "poke_bot.slop_box_h10_rtp_prestage_identity/v1"
RTP_CUT_SCHEMA = "poke_bot.slop_box_h10_rtp_bootstrap_cut/v1"
MARNIE_DIGEST = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
ALAKAZAM_DIGEST = (
    "sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _publish_live(src: Path, live: Path) -> None:
    live.parent.mkdir(parents=True, exist_ok=True)
    tmp = live.with_suffix(live.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, live)
    receipt = src.with_name(src.name + ".receipt.json")
    if receipt.is_file():
        shutil.copy2(receipt, live.with_name(live.name + ".receipt.json"))


def _rtp_registry_text(
    *,
    parent: Path,
    parent_digest: str,
    shard: Path,
    device: str,
    epochs: int,
    max_games: int,
) -> str:
    return "\n".join(
        [
            "schema: poke_bot.recursive_turn_planner.archetype_registry/v1",
            "archetypes:",
            f"  - specialist_id: {SPECIALIST_ID}",
            '    display_name: "Slop Box H10 RTP"',
            "    enabled: true",
            "    profile: pure_rl",
            "    d_model: 96",
            f"    epochs: {epochs}",
            f"    max_games: {max_games}",
            "    lr: 0.001",
            "    seed: 0",
            f"    device: {device}",
            "    also_poke_rlm: false",
            f"    parent_checkpoint: {parent}",
            f"    parent_digest: {parent_digest}",
            f"    training_shard: {shard}",
            "",
        ]
    )


def _rtp_pipeline_cmd(
    *,
    parent: Path,
    shard: Path,
    out_dir: Path,
    registry: Path,
    device: str,
    epochs: int,
    max_games: int,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_rtp_archetype_pipeline.py"),
        "--registry",
        str(registry),
        "--out-dir",
        str(out_dir),
        "--specialist",
        SPECIALIST_ID,
        "--checkpoint",
        str(parent),
        "--shard",
        str(shard),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--max-games",
        str(max_games),
        "--profile",
        "pure_rl",
    ]


def _rtp_cut_satisfies(
    receipt_path: Path,
    *,
    parent_digest: str,
    shard: Path,
    epochs: int,
    max_games: int,
) -> dict[str, Any] | None:
    """Return an existing full-cut receipt when it already matches intensify knobs."""
    if not receipt_path.is_file():
        return None
    try:
        cut = read_json(receipt_path)
    except Exception:
        return None
    if cut.get("schema") != RTP_CUT_SCHEMA:
        return None
    if cut.get("status") != "bootstrap_cut_ready":
        return None
    if str(cut.get("parent_digest") or "") != str(parent_digest):
        return None
    if Path(str(cut.get("training_shard") or "")).resolve() != shard.resolve():
        return None
    if int(cut.get("rtp_epochs") or 0) != int(epochs):
        return None
    if int(cut.get("rtp_max_games") or 0) != int(max_games):
        return None
    rtp_ckpt = Path(str(cut.get("rtp_checkpoint") or ""))
    if not rtp_ckpt.is_file():
        return None
    if str(cut.get("rtp_checkpoint_sha256") or "") != sha256(rtp_ckpt):
        return None
    return cut


def _run_rtp_cut(
    *,
    parent: Path,
    parent_digest: str,
    shard: Path,
    out_dir: Path,
    live_checkpoint: Path,
    device: str,
    epochs: int,
    max_games: int,
) -> dict[str, Any]:
    # One-job registry so the archetype pipeline binds Slop Box parent+shard.
    registry = out_dir / "_bootstrap_registry_r170.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        _rtp_registry_text(
            parent=parent,
            parent_digest=parent_digest,
            shard=shard,
            device=device,
            epochs=epochs,
            max_games=max_games,
        ),
        encoding="utf-8",
    )
    cmd = _rtp_pipeline_cmd(
        parent=parent,
        shard=shard,
        out_dir=out_dir,
        registry=registry,
        device=device,
        epochs=epochs,
        max_games=max_games,
    )
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "RTP bootstrap cut failed:\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    rtp_ckpt = out_dir / SPECIALIST_ID / "rtp" / "rtp_shadow_planner.pt"
    if not rtp_ckpt.is_file():
        raise RuntimeError(f"RTP cut missing checkpoint: {rtp_ckpt}")
    _publish_live(rtp_ckpt, live_checkpoint)
    return {
        "schema": RTP_CUT_SCHEMA,
        "status": "bootstrap_cut_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specialist_id": SPECIALIST_ID,
        "parent_checkpoint": str(parent),
        "parent_digest": parent_digest,
        "training_shard": str(shard),
        "rtp_epochs": int(epochs),
        "rtp_max_games": int(max_games),
        "rtp_device": str(device),
        "rtp_checkpoint": str(rtp_ckpt.resolve()),
        "rtp_checkpoint_sha256": sha256(rtp_ckpt),
        "live_checkpoint": str(live_checkpoint.resolve()),
        "guide_fed_to_rtp": False,
        "neural_only": True,
        "serving_eligible": False,
        "pipeline_stdout_tail": proc.stdout[-2000:],
    }


def _validate_preflight_receipt(
    path: Path | None,
    *,
    expected_schema: str,
    expected_status: set[str],
) -> tuple[Path, dict[str, Any]]:
    if path is None:
        raise RuntimeError(f"preflight requires {expected_schema} receipt")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"preflight receipt is absent: {resolved}")
    payload = read_json(resolved)
    if (
        payload.get("schema") != expected_schema
        or str(payload.get("status") or "") not in expected_status
    ):
        raise RuntimeError(f"preflight receipt contract changed: {resolved}")
    return resolved, payload


def _run_preflight(
    *,
    args: argparse.Namespace,
    identity: dict[str, Any],
    parent: Path,
    parent_digest: str,
    guide: Path,
    guide_ready: Path,
    expert: Path,
    rtp_shard: Path,
) -> dict[str, Any]:
    """Exercise launch validators without touching durable training state."""

    if (
        int(args.epochs) != 40
        or int(args.owner_slop_box_ce_intensify_epochs) != 40
        or int(args.batch_size) != 2048
        or int(args.rtp_epochs) != 8
        or int(args.rtp_max_games) != 42453
    ):
        raise RuntimeError("Slop Box r174 preflight schedule must be 40/2048/8/42453")

    protocol = args.protocol.expanduser().resolve()
    registry_root = args.registry_root.expanduser().resolve()
    parent_family = registry_root / "marnie-iter9-training-freeze-r163"
    frozen = verify_frozen_model(parent_family)
    frozen_source = Path(str(frozen.get("source_checkpoint") or "")).resolve()
    if (
        str(frozen.get("checkpoint_digest") or "") != parent_digest
        or frozen_source != parent
    ):
        raise RuntimeError("frozen Marnie parent registry no longer binds --parent")

    expanded_raw, expanded = load_expanded_head_contract(protocol)
    del expanded_raw
    manifest_identity = resolve_expert_manifest(
        expert,
        min_decisions=90000,
        require_protected=True,
        required_archetype=SPECIALIST_ID,
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=TARGETS,
        required_expanded_target_schema=str(expanded["target_schema"]),
        required_expanded_target_digest=str(expanded["target_schema_digest"]),
        required_expanded_heads=tuple(EXPANDED_HEAD_IDS),
    )

    with tempfile.TemporaryDirectory(prefix="slop-box-preflight-") as raw_tmp:
        hot_start, hot_start_digest, hot_start_expansion = (
            _specialist_hot_start_from_core(
                Path(str(frozen["model_path"])).resolve(),
                run_dir=Path(raw_tmp) / "hot-start",
                archetype=SPECIALIST_ID,
                enable_expanded_heads=True,
                expanded_identity=expanded,
                enable_decision_fusion=True,
                enable_strategic_curriculum=True,
                enable_combo_state_head=True,
                allow_h10_specialist_parent=True,
            )
        )
        if not hot_start.is_file() or checkpoint.checkpoint_digest(hot_start) != hot_start_digest:
            raise RuntimeError("ephemeral H10 successor hot-start validation failed")
        curriculum = materialize(
            specialist_id=SPECIALIST_ID,
            guide_contract=guide,
            guide_ready_receipt=guide_ready,
            output_root=Path(raw_tmp),
            training_implementation=(ROOT / "poke_bot" / "train.py"),
            training_mode="strategic_directional_v2",
            include_combo_state=True,
        )
        guide_contract = current_deck_guide_handoff_contract(
            specialist_id=SPECIALIST_ID,
            contract_path=guide,
            expected_contract_sha256=sha256(guide),
            guide_version="teal-mask-ogerpon-ex-slop-box-north-star-v3",
            corpus_ready_receipt=guide_ready,
            expected_corpus_ready_sha256=sha256(guide_ready),
            strategic_curriculum_spec=Path(curriculum["curriculum_spec"]),
            expected_strategic_curriculum_spec_sha256=sha256(
                Path(curriculum["curriculum_spec"])
            ),
            strategic_head_role_map=Path(curriculum["head_role_map"]),
            expected_strategic_head_role_map_sha256=sha256(
                Path(curriculum["head_role_map"])
            ),
            strategic_validation_receipt=Path(curriculum["validation_receipt"]),
            expected_strategic_validation_receipt_sha256=sha256(
                Path(curriculum["validation_receipt"])
            ),
            protocol_path=protocol,
        )

    guide_payload = read_json(guide_ready)
    manifest_payload = read_json(Path(manifest_identity.path))
    guide_rows = int(
        ((manifest_payload.get("totals") or {}).get("target_coverage") or {}).get(
            "guide_rows"
        )
        or 0
    )
    if (
        guide_payload.get("manifest_sha256") != manifest_identity.digest
        or guide_payload.get("protected_pointer_sha256") != sha256(expert)
        or int(guide_payload.get("decisions") or 0) != manifest_identity.decisions
        or int(guide_payload.get("guide_rows") or 0) != guide_rows
        or guide_rows <= 0
    ):
        raise RuntimeError("current-deck guide corpus binding changed")

    if args.pilot_importance_index is None:
        raise RuntimeError("Slop Box full-window preflight requires pilot importance")
    importance_path = args.pilot_importance_index.expanduser().resolve()
    importance_payload = read_json(importance_path)
    train_games = int(importance_payload.get("train_games") or 0)
    validation_games = int(importance_payload.get("validation_games") or 0)
    if (
        train_games + validation_games != manifest_identity.records
        or (train_games, validation_games) != (2430, 269)
    ):
        raise RuntimeError("pilot-importance split no longer matches full corpus")
    weights, importance_contract = load_training_weights_for_corpus(
        importance_path,
        expected_manifest_digest=manifest_identity.digest,
        split_seed=20260722,
        validation_fraction=0.10,
        max_context=320,
        train_games=train_games,
        validation_games=validation_games,
    )
    if (
        len(weights) != 2430
        or importance_contract.get("actions_and_labels_unchanged") is not True
        or importance_contract.get("validation_unweighted") is not True
        or importance_contract.get("kaggle_evaluation_replays_excluded") is not True
    ):
        raise RuntimeError("pilot-importance contract is not training-safe")

    cpu_ready_path, cpu_ready = _validate_preflight_receipt(
        args.cpu_pack_ready_receipt,
        expected_schema="poke_bot.slop_box_schema8_jul24_aug5_cpu_pack_ready_r175/v1",
        expected_status={"ready_bound"},
    )
    if (
        Path(str(cpu_ready.get("protected_expert_corpus") or "")).resolve()
        != expert
        or Path(str((cpu_ready.get("cpu_pack") or {}).get("root") or "")).resolve()
        != args.cpu_pack_root.expanduser().resolve()
        or int(cpu_ready.get("records") or 0) != manifest_identity.records
        or int(cpu_ready.get("decisions") or 0) != manifest_identity.decisions
        or cpu_ready.get("combo_state_targets") is not True
        or cpu_ready.get("setup_board_labels_present") is not True
    ):
        raise RuntimeError("full-window CPU pack receipt no longer binds launch")

    markup_path, markup = _validate_preflight_receipt(
        args.matchup_markup_receipt,
        expected_schema="poke_bot.slop_box_h10_rtp_expert_matchup_markup_r171/v1",
        expected_status={"ready_for_wipe_and_fresh_adapter_bootstrap"},
    )
    if (
        int(markup.get("source_games") or 0) != manifest_identity.records
        or int(markup.get("source_decisions") or 0) != manifest_identity.decisions
        or int(markup.get("marked_games") or 0) != 2515
        or int(markup.get("marked_decisions") or 0) != 171243
    ):
        raise RuntimeError("matchup markup receipt no longer binds full corpus")

    rtp_source_path, rtp_source = _validate_preflight_receipt(
        args.rtp_source_receipt,
        expected_schema="poke_bot.slop_box_combo_state_rematerialization/v1",
        expected_status={"ready"},
    )
    if (
        Path(str(rtp_source.get("output_path") or "")).resolve() != rtp_shard
        or str(rtp_source.get("output_sha256") or "") != sha256(rtp_shard)
        or int((rtp_source.get("coverage") or {}).get("games") or 0) != 42453
        or int((rtp_source.get("coverage") or {}).get("combo_state_rows") or 0)
        != 2979468
        or rtp_source.get("combo_loss_can_be_nonzero") is not True
    ):
        raise RuntimeError("r173 combo-labeled RTP shard binding changed")

    result = {
        "schema": "poke_bot.slop_box_h10_rtp_bootstrap_preflight/v1",
        "status": "passed_activation_still_held",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "side_effect_free": True,
        "activation_allowed": False,
        "owner_authority_required": True,
        "specialist_id": SPECIALIST_ID,
        "identity": str(args.identity.expanduser().resolve()),
        "identity_sha256": sha256(args.identity.expanduser().resolve()),
        "parent": str(parent),
        "parent_digest": parent_digest,
        "frozen_parent_family": str(parent_family),
        "expert": str(expert),
        "expert_pointer_sha256": sha256(expert),
        "expert_manifest": manifest_identity.path,
        "expert_manifest_sha256": manifest_identity.digest,
        "records": manifest_identity.records,
        "decisions": manifest_identity.decisions,
        "expanded_schedule_digest": expanded["schedule_digest"],
        "expanded_target_digest": expanded["target_schema_digest"],
        "ephemeral_hot_start_digest": hot_start_digest,
        "ephemeral_hot_start_expansion": hot_start_expansion,
        "ephemeral_hot_start_removed_after_validation": True,
        "guide": str(guide),
        "guide_sha256": sha256(guide),
        "guide_ready": str(guide_ready),
        "guide_ready_sha256": sha256(guide_ready),
        "guide_training_mode": guide_contract.get("training_mode"),
        "guide_rows": guide_rows,
        "pilot_importance": str(importance_path),
        "pilot_importance_sha256": sha256(importance_path),
        "pilot_train_games": train_games,
        "pilot_validation_games": validation_games,
        "cpu_pack_root": str(args.cpu_pack_root.expanduser().resolve()),
        "cpu_pack_ready_receipt": str(cpu_ready_path),
        "cpu_pack_ready_receipt_sha256": sha256(cpu_ready_path),
        "matchup_markup_receipt": str(markup_path),
        "matchup_markup_receipt_sha256": sha256(markup_path),
        "rtp_shard": str(rtp_shard),
        "rtp_shard_sha256": sha256(rtp_shard),
        "rtp_source_receipt": str(rtp_source_path),
        "rtp_source_receipt_sha256": sha256(rtp_source_path),
        "schedule": {
            "outer_epochs": 40,
            "batch_size": 2048,
            "rtp_epochs": 8,
            "rtp_max_games": 42453,
        },
        "identity_goal_revision": int(identity.get("goal_revision") or 0),
        "remaining_blocker": "revision_175_explicit_ce_recovery_hold_requires_owner_lift",
    }
    if args.preflight_receipt is not None:
        atomic_json(args.preflight_receipt.expanduser().resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity",
        type=Path,
        default=ROOT / "state" / "slop_box_h10_rtp_prestage_identity_r170.json",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/pure_rl/"
            "final_format_marnie_r104_h10_i_v6_8k/checkpoints/iter_00007.pt"
        ),
    )
    parser.add_argument("--parent-digest", default=MARNIE_DIGEST)
    parser.add_argument(
        "--alakazam-teacher",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/pure_rl/"
            "final_format_alakazam_r79_h10_i_v6_8k/checkpoints/iter_00020.pt"
        ),
    )
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--guide-ready", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    parser.add_argument("--curriculum-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    # H10 CE fit-test peaks ~34 GiB / 47 GiB at 2048 on Blackwell; do not raise
    # without a fresh device fit-test receipt.
    parser.add_argument("--batch-size", type=int, default=2048)
    # Owner r170: Slop Box CE intensify continues expert policy CE through 40
    # then 60 epochs (locked 25-epoch path plus owner-scoped extension).
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--owner-slop-box-ce-intensify-epochs",
        type=int,
        default=60,
        help="Must match --epochs for the owner-scoped 40/300-epoch deep CE intensify path",
    )
    parser.add_argument(
        "--pilot-importance-index",
        type=Path,
        default=None,
        help=(
            "Optional checksum-bound Cox/Chao train upweight index "
            "(full-archetype corpus retained; validation unweighted)"
        ),
    )
    parser.add_argument(
        "--rtp-shard",
        type=Path,
        required=True,
        help="Expert-derived or bootstrap-trajectory jsonl shard for RTP cut",
    )
    parser.add_argument(
        "--rtp-out-dir",
        type=Path,
        default=Path("/home/inzi/poke-bot-agent/outputs/rtp_fleet/slop-box-h10"),
    )
    parser.add_argument(
        "--rtp-live-checkpoint",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/rtp_fleet/"
            "slop-box-h10.live/rtp_shadow_planner.pt"
        ),
    )
    parser.add_argument(
        "--rtp-cut-receipt",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-h10-rtp-bootstrap-cut-r170.json"
        ),
    )
    parser.add_argument("--rtp-device", default="cuda:0")
    # Owner r170: use the full expert trajectory shard (42453 games), not a
    # 512-game under-sample.
    parser.add_argument("--rtp-epochs", type=int, default=8)
    parser.add_argument("--rtp-max-games", type=int, default=42453)
    parser.add_argument(
        "--rtp-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlap full-shard RTP cut on --rtp-device (cuda:0 / 3080 Ti) with "
            "H10 CE on Blackwell (cuda:1). Default on for CE intensify."
        ),
    )
    parser.add_argument(
        "--skip-h10-bootstrap",
        action="store_true",
        help="Only run RTP cut against an existing H10 parent/ready path",
    )
    parser.add_argument(
        "--rtp-only",
        action="store_true",
        help=(
            "Run/refresh the RTP cut only; do not touch H10 CE, ready, or the "
            "Cox/Chao gate (prep overlap path)."
        ),
    )
    parser.add_argument(
        "--h10-parent-for-rtp",
        type=Path,
        default=None,
        help="Override parent for RTP cut (defaults to bootstrap hot-start parent)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the exact launch contract and exit before durable writes",
    )
    parser.add_argument("--preflight-receipt", type=Path, default=None)
    parser.add_argument("--cpu-pack-ready-receipt", type=Path, default=None)
    parser.add_argument("--matchup-markup-receipt", type=Path, default=None)
    parser.add_argument("--rtp-source-receipt", type=Path, default=None)
    args = parser.parse_args()

    identity = read_json(args.identity.resolve())
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise RuntimeError("Slop Box identity schema mismatch")
    if int(identity.get("goal_revision") or 0) < 170:
        raise RuntimeError("Slop Box identity requires goal revision >= 170")

    parent = args.parent.expanduser().resolve()
    parent_digest = str(args.parent_digest)
    if not parent.is_file() or checkpoint.checkpoint_digest(parent) != parent_digest:
        raise RuntimeError("primary warm-start parent digest mismatch")
    alakazam = args.alakazam_teacher.expanduser().resolve()
    if alakazam.is_file() and checkpoint.checkpoint_digest(alakazam) != ALAKAZAM_DIGEST:
        raise RuntimeError("Alakazam teacher digest mismatch")

    guide = args.guide.resolve()
    guide_ready = args.guide_ready.resolve()
    expert = args.expert.resolve()
    for path in (guide, guide_ready, expert, args.protocol.resolve()):
        if not path.is_file():
            raise RuntimeError(f"missing bootstrap input: {path}")

    rtp_shard = args.rtp_shard.expanduser().resolve()
    if not rtp_shard.is_file():
        raise RuntimeError(
            "RTP bootstrap cut requires an expert-derived/bootstrap jsonl shard: "
            f"{rtp_shard}"
        )

    rtp_parent = (
        args.h10_parent_for_rtp.expanduser().resolve()
        if args.h10_parent_for_rtp is not None
        else parent
    )
    rtp_parent_digest = checkpoint.checkpoint_digest(rtp_parent)
    rtp_out_dir = args.rtp_out_dir.expanduser().resolve()
    rtp_live = args.rtp_live_checkpoint.expanduser().resolve()
    rtp_receipt = args.rtp_cut_receipt.expanduser().resolve()
    rtp_epochs = int(args.rtp_epochs)
    rtp_max_games = int(args.rtp_max_games)
    rtp_device = str(args.rtp_device)

    if args.preflight_only:
        result = _run_preflight(
            args=args,
            identity=identity,
            parent=parent,
            parent_digest=parent_digest,
            guide=guide,
            guide_ready=guide_ready,
            expert=expert,
            rtp_shard=rtp_shard,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.rtp_only:
        cut = _rtp_cut_satisfies(
            rtp_receipt,
            parent_digest=rtp_parent_digest,
            shard=rtp_shard,
            epochs=rtp_epochs,
            max_games=rtp_max_games,
        )
        if cut is None:
            cut = _run_rtp_cut(
                parent=rtp_parent,
                parent_digest=rtp_parent_digest,
                shard=rtp_shard,
                out_dir=rtp_out_dir,
                live_checkpoint=rtp_live,
                device=rtp_device,
                epochs=rtp_epochs,
                max_games=rtp_max_games,
            )
            cut["identity_sha256"] = sha256(args.identity.resolve())
            cut["cox_chao_policy_accuracy_gate"] = dict(
                (identity.get("expert_bootstrap") or {}).get(
                    "cox_chao_policy_accuracy_gate"
                )
                or {}
            )
            cut["rtp_only_prep"] = True
            atomic_json(rtp_receipt, cut)
        else:
            cut = {
                **cut,
                "reused_existing_full_cut": True,
                "rtp_only_prep": True,
            }
            atomic_json(rtp_receipt, cut)
        print(json.dumps(cut, indent=2, sort_keys=True))
        return 0

    rtp_proc: subprocess.Popen[str] | None = None
    rtp_log_path: Path | None = None
    existing_cut = _rtp_cut_satisfies(
        rtp_receipt,
        parent_digest=rtp_parent_digest,
        shard=rtp_shard,
        epochs=rtp_epochs,
        max_games=rtp_max_games,
    )

    if (
        existing_cut is None
        and not args.skip_h10_bootstrap
        and bool(args.rtp_parallel)
    ):
        # Overlap full-shard RTP on 3080 Ti with H10 CE on Blackwell.
        rtp_out_dir.mkdir(parents=True, exist_ok=True)
        registry = rtp_out_dir / "_bootstrap_registry_r170.yaml"
        registry.write_text(
            _rtp_registry_text(
                parent=rtp_parent,
                parent_digest=rtp_parent_digest,
                shard=rtp_shard,
                device=rtp_device,
                epochs=rtp_epochs,
                max_games=rtp_max_games,
            ),
            encoding="utf-8",
        )
        rtp_log_path = (
            ROOT
            / "outputs"
            / "final_format_slop_box_h10_rtp"
            / "logs"
            / "rtp_fullcut_parallel.log"
        )
        rtp_log_path.parent.mkdir(parents=True, exist_ok=True)
        rtp_log = rtp_log_path.open("a", encoding="utf-8")
        cmd = _rtp_pipeline_cmd(
            parent=rtp_parent,
            shard=rtp_shard,
            out_dir=rtp_out_dir,
            registry=registry,
            device=rtp_device,
            epochs=rtp_epochs,
            max_games=rtp_max_games,
        )
        print(
            "[slop-box-bootstrap] starting parallel RTP full-cut on "
            f"{rtp_device} while H10 CE runs",
            flush=True,
        )
        rtp_proc = subprocess.Popen(
            cmd,
            stdout=rtp_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if not args.skip_h10_bootstrap:
        curriculum = materialize(
            specialist_id=SPECIALIST_ID,
            guide_contract=guide,
            guide_ready_receipt=guide_ready,
            output_root=args.curriculum_root.resolve(),
            training_implementation=(ROOT / "poke_bot" / "train.py"),
            training_mode="strategic_directional_v2",
            include_combo_state=True,
        )
        _expanded_raw, expanded = load_expanded_head_contract(
            args.protocol.resolve()
        )
        del _expanded_raw
        parent_family = (
            args.registry_root.resolve() / "marnie-iter9-training-freeze-r163"
        )
        argv = [
            "--expert-corpus",
            str(expert),
            "--archetype",
            SPECIALIST_ID,
            "--family",
            args.family,
            "--display-name",
            "Slop Box Final-Format H10 RTP Specialist",
            "--core-family",
            str(parent_family),
            "--registry-root",
            str(args.registry_root.resolve()),
            "--ready",
            str(args.ready.resolve()),
            "--run-name",
            "final_format_slop_box_h10_rtp_bootstrap",
            "--run-dir",
            str(args.run_dir.resolve()),
            "--epochs",
            str(args.epochs),
            "--owner-slop-box-ce-intensify-epochs",
            str(args.owner_slop_box_ce_intensify_epochs),
            *(
                (
                    "--pilot-importance-index",
                    str(args.pilot_importance_index.resolve()),
                )
                if args.pilot_importance_index is not None
                else ()
            ),
            "--batch-size",
            str(args.batch_size),
            "--min-decisions",
            "90000",
            "--cpu-pack-root",
            str(args.cpu_pack_root.resolve()),
            "--expanded-heads",
            "--decision-fusion",
            "--allow-h10-specialist-parent",
            "--retain-inherited-h10-combo-state-head",
            "--rl-protocol",
            str(args.protocol.resolve()),
            "--expected-expanded-schedule-digest",
            str(expanded["schedule_digest"]),
            "--expected-expanded-target-digest",
            str(expanded["target_schema_digest"]),
            "--current-deck-guide-contract",
            str(guide),
            "--expected-current-deck-guide-sha256",
            sha256(guide),
            "--current-deck-guide-version",
            # Must match CURRENT_DECK_GUIDE_CORPUS_READY.guide_version.
            "teal-mask-ogerpon-ex-slop-box-north-star-v3",
            "--current-deck-guide-corpus-ready",
            str(guide_ready),
            "--expected-current-deck-guide-corpus-ready-sha256",
            sha256(guide_ready),
            "--strategic-curriculum-spec",
            str(curriculum["curriculum_spec"]),
            "--strategic-curriculum-spec-sha256",
            sha256(curriculum["curriculum_spec"]),
            "--strategic-head-role-map",
            str(curriculum["head_role_map"]),
            "--strategic-head-role-map-sha256",
            sha256(curriculum["head_role_map"]),
            "--strategic-validation-receipt",
            str(curriculum["validation_receipt"]),
            "--strategic-validation-receipt-sha256",
            sha256(curriculum["validation_receipt"]),
        ]
        for target in TARGETS:
            argv.extend(("--required-target", target))
        # Point the H10 parent reuse at the checksum-bound Marnie freeze.
        # run_starmie_expert_bootstrap resolves --core-family; also stage a
        # seed symlink under run-dir for RTP binding.
        args.run_dir.mkdir(parents=True, exist_ok=True)
        seed = args.run_dir / "checkpoints" / "seed.pt"
        seed.parent.mkdir(parents=True, exist_ok=True)
        if not seed.exists():
            os.symlink(parent, seed)
        result = bootstrap_main(argv)
        if result != 0:
            if rtp_proc is not None:
                rtp_proc.terminate()
                rtp_proc.wait(timeout=30)
            return result

    if existing_cut is not None:
        cut = {
            **existing_cut,
            "reused_existing_full_cut": True,
        }
        print(
            "[slop-box-bootstrap] reusing checksum-bound full-shard RTP cut",
            flush=True,
        )
    elif rtp_proc is not None:
        print(
            "[slop-box-bootstrap] waiting for parallel RTP full-cut to finish",
            flush=True,
        )
        rtp_rc = int(rtp_proc.wait())
        if rtp_log_path is not None:
            stdout_tail = rtp_log_path.read_text(encoding="utf-8")[-2000:]
        else:
            stdout_tail = ""
        if rtp_rc != 0:
            raise RuntimeError(
                "parallel RTP bootstrap cut failed "
                f"(rc={rtp_rc}); see {rtp_log_path}"
            )
        rtp_ckpt = rtp_out_dir / SPECIALIST_ID / "rtp" / "rtp_shadow_planner.pt"
        if not rtp_ckpt.is_file():
            raise RuntimeError(f"RTP cut missing checkpoint: {rtp_ckpt}")
        _publish_live(rtp_ckpt, rtp_live)
        cut = {
            "schema": RTP_CUT_SCHEMA,
            "status": "bootstrap_cut_ready",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "specialist_id": SPECIALIST_ID,
            "parent_checkpoint": str(rtp_parent),
            "parent_digest": rtp_parent_digest,
            "training_shard": str(rtp_shard),
            "rtp_epochs": rtp_epochs,
            "rtp_max_games": rtp_max_games,
            "rtp_device": rtp_device,
            "rtp_checkpoint": str(rtp_ckpt.resolve()),
            "rtp_checkpoint_sha256": sha256(rtp_ckpt),
            "live_checkpoint": str(rtp_live.resolve()),
            "guide_fed_to_rtp": False,
            "neural_only": True,
            "serving_eligible": False,
            "parallel_with_h10_ce": True,
            "pipeline_stdout_tail": stdout_tail,
        }
    else:
        cut = _run_rtp_cut(
            parent=rtp_parent,
            parent_digest=rtp_parent_digest,
            shard=rtp_shard,
            out_dir=rtp_out_dir,
            live_checkpoint=rtp_live,
            device=rtp_device,
            epochs=rtp_epochs,
            max_games=rtp_max_games,
        )
    cut["identity_sha256"] = sha256(args.identity.resolve())
    cut["cox_chao_policy_accuracy_gate"] = dict(
        (identity.get("expert_bootstrap") or {}).get("cox_chao_policy_accuracy_gate")
        or {}
    )
    atomic_json(rtp_receipt, cut)

    if args.ready.expanduser().resolve().is_file():
        ready = read_json(args.ready.expanduser().resolve())
        # Never promote a failed/quarantined ready merely because RTP refreshed.
        ready["slop_box_rtp_bootstrap_cut"] = {
            "receipt": str(rtp_receipt),
            "receipt_sha256": sha256(rtp_receipt),
            "live_checkpoint": cut["live_checkpoint"],
            "rtp_checkpoint_sha256": cut["rtp_checkpoint_sha256"],
            "guide_fed_to_rtp": False,
        }
        ready["cox_chao_policy_accuracy_gate"] = cut["cox_chao_policy_accuracy_gate"]
        atomic_json(args.ready.expanduser().resolve(), ready)

    # Fail-closed completion gate: ready must not remain production-ready
    # unless Cox/Chao held policy_acc ≥ 0.90.
    gate_cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "evaluate_slop_box_cox_chao_held_policy_acc_gate_r170.py"),
        "--ready",
        str(args.ready.expanduser().resolve()),
        "--split-seed",
        "20260722",
        "--threshold",
        "0.95",
    ]
    gate_proc = subprocess.run(gate_cmd, check=False)
    cut["cox_chao_held_policy_acc_gate_rc"] = int(gate_proc.returncode)
    atomic_json(rtp_receipt, cut)
    print(json.dumps(cut, indent=2, sort_keys=True))
    if gate_proc.returncode != 0:
        raise RuntimeError(
            "Cox/Chao held policy_acc gate failed closed "
            "(required ≥ 0.95); bootstrap ready must not proceed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
