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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from scripts.run_starmie_expert_bootstrap import (  # noqa: E402
    TARGETS,
    atomic_json,
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
        "\n".join(
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
        ),
        encoding="utf-8",
    )
    cmd = [
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
        "rtp_checkpoint": str(rtp_ckpt.resolve()),
        "rtp_checkpoint_sha256": sha256(rtp_ckpt),
        "live_checkpoint": str(live_checkpoint.resolve()),
        "guide_fed_to_rtp": False,
        "neural_only": True,
        "serving_eligible": False,
        "pipeline_stdout_tail": proc.stdout[-2000:],
    }


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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=35)
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
    parser.add_argument("--rtp-epochs", type=int, default=4)
    parser.add_argument("--rtp-max-games", type=int, default=512)
    parser.add_argument(
        "--skip-h10-bootstrap",
        action="store_true",
        help="Only run RTP cut against an existing H10 parent/ready path",
    )
    parser.add_argument(
        "--h10-parent-for-rtp",
        type=Path,
        default=None,
        help="Override parent for RTP cut (defaults to bootstrap hot-start parent)",
    )
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
            "slop-box-h10-rtp-north-star-v1",
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
            return result

    rtp_parent = (
        args.h10_parent_for_rtp.expanduser().resolve()
        if args.h10_parent_for_rtp is not None
        else parent
    )
    rtp_parent_digest = checkpoint.checkpoint_digest(rtp_parent)
    cut = _run_rtp_cut(
        parent=rtp_parent,
        parent_digest=rtp_parent_digest,
        shard=rtp_shard,
        out_dir=args.rtp_out_dir.expanduser().resolve(),
        live_checkpoint=args.rtp_live_checkpoint.expanduser().resolve(),
        device=str(args.rtp_device),
        epochs=int(args.rtp_epochs),
        max_games=int(args.rtp_max_games),
    )
    cut["identity_sha256"] = sha256(args.identity.resolve())
    cut["cox_chao_policy_accuracy_gate"] = dict(
        (identity.get("expert_bootstrap") or {}).get("cox_chao_policy_accuracy_gate")
        or {}
    )
    atomic_json(args.rtp_cut_receipt.expanduser().resolve(), cut)

    if args.ready.expanduser().resolve().is_file():
        ready = read_json(args.ready.expanduser().resolve())
        ready["slop_box_rtp_bootstrap_cut"] = {
            "receipt": str(args.rtp_cut_receipt.expanduser().resolve()),
            "receipt_sha256": sha256(args.rtp_cut_receipt.expanduser().resolve()),
            "live_checkpoint": cut["live_checkpoint"],
            "rtp_checkpoint_sha256": cut["rtp_checkpoint_sha256"],
            "guide_fed_to_rtp": False,
        }
        ready["cox_chao_policy_accuracy_gate"] = cut["cox_chao_policy_accuracy_gate"]
        atomic_json(args.ready.expanduser().resolve(), ready)
    print(json.dumps(cut, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
