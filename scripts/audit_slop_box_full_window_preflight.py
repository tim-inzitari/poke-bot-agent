#!/usr/bin/env python3
"""Write a fail-closed Slop Box full-window bootstrap preflight receipt."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/pokebot/poke-bot-agent")
STATE = ROOT / "outputs/state"
OUTPUT = STATE / "slop-box-full-window-preflight-held-post-alakazam-terminal-v5.json"
BOOTSTRAP_UNIT = "pokebot-final-format-slop-box-h10-rtp-bootstrap.service"
RL_UNIT = "pokebot-final-format-slop-box-h10-rtp-rl.service"

FULL_EXPERT = (
    ROOT
    / "data/bootstrap/expert-slop-box-schema8-jul24-aug5-r175/"
    "teal-mask-ogerpon-ex/PROTECTED_EXPERT_CORPUS.json"
)
FULL_GUIDE = FULL_EXPERT.parent / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
FULL_MANIFEST = FULL_EXPERT.parent / "manifest.json"
FULL_CPU_PACK = (
    ROOT
    / "outputs/bootstrap/cpu-packs/"
    "final_format_slop_box_h10_rtp_schema8_jul24_aug5_r175"
)
FULL_MARKUP = (
    ROOT
    / "outputs/bootstrap/slop-box-h10-rtp/"
    "expert-matchup-marked-selfplay-equiv-jul24-aug5-r175"
)
MARKUP_RECEIPT = (
    STATE / "slop-box-h10-rtp-expert-matchup-markup-jul24-aug5-r175.json"
)
SEAL = STATE / "slop-box-schema8-jul24-aug5-sealed-r175.json"
CPU_READY = STATE / "slop-box-schema8-jul24-aug5-cpu-pack-ready-r175.json"
OLD_PLAN = STATE / "slop-box-fullcorpus-bootstrap-r175-plan.json"
OLD_RTP_SHARD = (
    ROOT / "outputs/bootstrap/slop-box-h10-rtp/expert_trajectory_shard.combo_v1.jsonl"
)
RTP_RECEIPT = ROOT / "state/slop-box-combo-state-rematerialization-r173.json"
ADAPTER_AUTH = STATE / "slop-box-h10-rtp-matchup-adapter-bootstrap-r171.json"
FULL_TARGETS = (
    STATE / "slop-box-cox-chao-held-targets-schema8-jul24-aug5-r175.json"
)
FULL_PILOT = (
    STATE / "slop-box-cox-chao-held-pilot-map-schema8-jul24-aug5-r175.json"
)
FULL_HELD = (
    STATE / "slop-box-cox-chao-held-split-pilot-map-schema8-jul24-aug5-r175.json"
)
FULL_UPWEIGHT = (
    STATE
    / "slop-box-cox-chao-train-upweight-importance-schema8-jul24-aug5-r175.json"
)
PRESTAGE_DROPIN = (
    Path.home()
    / ".config/systemd/user/"
    "pokebot-final-format-slop-box-h10-rtp-bootstrap.service.d/"
    "50-jul24-aug5-full-window-r175-prestage.conf"
)
LAUNCH_PREFLIGHT = STATE / "slop-box-full-window-bootstrap-launch-preflight-r194-v2.json"
GUIDE_READY_REPAIR = STATE / "slop-box-full-window-guide-ready-repair-r194.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def unit_property(unit: str, name: str) -> str:
    return subprocess.check_output(
        ["systemctl", "--user", "show", unit, f"-p{name}", "--value"],
        text=True,
    ).strip()


def unit_text(unit: str) -> str:
    return subprocess.check_output(
        ["systemctl", "--user", "cat", unit, "--no-pager"], text=True
    )


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    required_files = (
        FULL_EXPERT,
        FULL_GUIDE,
        FULL_MANIFEST,
        FULL_MARKUP / "manifest.json",
        FULL_MARKUP / "MARKUP_READY.json",
        MARKUP_RECEIPT,
        SEAL,
        CPU_READY,
        RTP_RECEIPT,
        FULL_TARGETS,
        FULL_PILOT,
        FULL_HELD,
        FULL_UPWEIGHT,
        PRESTAGE_DROPIN,
        LAUNCH_PREFLIGHT,
        GUIDE_READY_REPAIR,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if not FULL_CPU_PACK.is_dir():
        missing.append(str(FULL_CPU_PACK))
    if missing:
        raise RuntimeError(f"full-window prerequisites missing: {missing}")

    seal = json.loads(SEAL.read_text())
    cpu_ready = json.loads(CPU_READY.read_text())
    expert = json.loads(FULL_EXPERT.read_text())
    guide = json.loads(FULL_GUIDE.read_text())
    markup = json.loads(MARKUP_RECEIPT.read_text())
    manifest = json.loads((FULL_MARKUP / "manifest.json").read_text())
    rtp_receipt = json.loads(RTP_RECEIPT.read_text())
    targets = json.loads(FULL_TARGETS.read_text())
    pilot = json.loads(FULL_PILOT.read_text())
    held = json.loads(FULL_HELD.read_text())
    upweight = json.loads(FULL_UPWEIGHT.read_text())
    launch_preflight = json.loads(LAUNCH_PREFLIGHT.read_text())
    guide_repair = json.loads(GUIDE_READY_REPAIR.read_text())
    if (
        seal.get("totals", {}).get("records"),
        seal.get("totals", {}).get("decisions"),
        cpu_ready.get("records"),
        cpu_ready.get("decisions"),
        expert.get("records_kept"),
        expert.get("decisions_kept"),
        guide.get("records_kept"),
        guide.get("decisions_kept"),
    ) != (2699, 182951, 2699, 182951, 2699, 182951, 2699, 182951):
        raise RuntimeError("full-window expert/guide/CPU-pack totals disagree")
    if (
        markup.get("source_games"),
        markup.get("source_decisions"),
        markup.get("marked_games"),
        markup.get("marked_decisions"),
    ) != (2699, 182951, 2515, 171243):
        raise RuntimeError("promoted markup totals disagree")
    if not manifest.get("offline_oracle_only"):
        raise RuntimeError("promoted markup lost offline-only contract")
    if manifest.get("runtime_routes_enabled") is not False:
        raise RuntimeError("promoted markup unexpectedly enables runtime routes")
    if len(targets.get("train_rows", [])) != 2430 or len(
        targets.get("validation_rows", [])
    ) != 269:
        raise RuntimeError("full-window importance split totals disagree")
    if (
        pilot.get("requested_rows"),
        pilot.get("resolved_rows"),
        pilot.get("unverifiable_rows"),
    ) != (2699, 2699, 0):
        raise RuntimeError("full-window raw-archive pilot map is incomplete")
    if (
        held.get("cox_chao_train_games"),
        held.get("cox_chao_held_validation_games"),
    ) != (1304, 135):
        raise RuntimeError("full-window Cox/Chao held split totals disagree")
    if upweight.get("tier_counts") != {"10x": 1304, "1x": 1126}:
        raise RuntimeError("full-window Cox/Chao upweight tiers disagree")
    if upweight.get("effective_training_weight_mass") != 14166.0:
        raise RuntimeError("full-window Cox/Chao training mass disagrees")
    if (
        guide.get("manifest_sha256")
        != "sha256:a7a71d4af9de38f454412e00a9a198ac85a070a183e1ae6555230c7b3d801371"
        or guide.get("protected_pointer_sha256") != sha256(FULL_EXPERT)
        or int(guide.get("guide_rows") or 0) != 16580
        or int(guide.get("decisions") or 0) != 182951
        or guide_repair.get("status") != "repaired_while_activation_held"
        or guide_repair.get("postimage_sha256") != sha256(FULL_GUIDE)
    ):
        raise RuntimeError("full-window guide-ready trainer projection is invalid")
    if (
        launch_preflight.get("schema")
        != "poke_bot.slop_box_h10_rtp_bootstrap_preflight/v1"
        or launch_preflight.get("status") != "passed_activation_still_held"
        or launch_preflight.get("side_effect_free") is not True
        or launch_preflight.get("activation_allowed") is not False
        or not str(launch_preflight.get("ephemeral_hot_start_digest") or "").startswith(
            "sha256:"
        )
        or launch_preflight.get("ephemeral_hot_start_removed_after_validation")
        is not True
        or launch_preflight.get("parent_digest")
        != "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
        or launch_preflight.get("expert_manifest_sha256")
        != "sha256:a7a71d4af9de38f454412e00a9a198ac85a070a183e1ae6555230c7b3d801371"
        or launch_preflight.get("guide_ready_sha256") != sha256(FULL_GUIDE)
        or launch_preflight.get("pilot_importance_sha256") != sha256(FULL_UPWEIGHT)
        or launch_preflight.get("rtp_shard_sha256")
        != "sha256:a22b1ac07e04e9c063397fc06afc6e9017ac4e0ab2b2fcd105dff550aa1ab56e"
        or launch_preflight.get("schedule")
        != {
            "batch_size": 2048,
            "outer_epochs": 40,
            "rtp_epochs": 8,
            "rtp_max_games": 42453,
        }
    ):
        raise RuntimeError("exact Slop Box bootstrap launch preflight is invalid")

    bootstrap_text = unit_text(BOOTSTRAP_UNIT)
    rl_text = unit_text(RL_UNIT)
    bootstrap_held = (
        unit_property(BOOTSTRAP_UNIT, "ActiveState") == "inactive"
        and "ExecStart=/bin/false" in bootstrap_text
    )
    rl_held = (
        unit_property(RL_UNIT, "ActiveState") == "inactive"
        and "ExecStart=/bin/false" in rl_text
    )
    if not bootstrap_held or not rl_held:
        raise RuntimeError("Slop Box bootstrap/RL holds are not both effective")

    prestage_text = PRESTAGE_DROPIN.read_text()
    staged_bindings = {
        "guide_ready": str(FULL_GUIDE) in prestage_text,
        "expert": str(FULL_EXPERT) in prestage_text,
        "cpu_pack": str(FULL_CPU_PACK) in prestage_text,
        "pilot_importance": str(FULL_UPWEIGHT) in prestage_text,
        "markup_receipt": str(MARKUP_RECEIPT) in prestage_text,
        "rtp_shard": str(OLD_RTP_SHARD) in prestage_text,
        "epochs_40": "--epochs 40" in prestage_text,
    }
    plan = json.loads(OLD_PLAN.read_text()) if OLD_PLAN.is_file() else {}
    rtp_receipt_valid = (
        OLD_RTP_SHARD.is_file()
        and rtp_receipt.get("status") == "ready"
        and rtp_receipt.get("output_path") == str(OLD_RTP_SHARD)
        and rtp_receipt.get("output_sha256")
        == "sha256:a22b1ac07e04e9c063397fc06afc6e9017ac4e0ab2b2fcd105dff550aa1ab56e"
        and rtp_receipt.get("coverage", {}).get("games") == 42453
        and rtp_receipt.get("coverage", {}).get("combo_state_rows") == 2979468
    )
    blockers = []
    if not all(staged_bindings.values()):
        blockers.append("full_window_bootstrap_dropin_incomplete")
    if not rtp_receipt_valid:
        blockers.append("r173_combo_labeled_rtp_shard_receipt_invalid")
    if not ADAPTER_AUTH.is_file():
        blockers.append("adapter_authorization_requires_future_ce_parent")
    blockers.append("revision_175_explicit_ce_recovery_hold_requires_owner_lift")

    payload = {
        "schema": "poke_bot.slop_box_full_window_preflight_held/v5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "offline_inputs_ready_activation_held",
        "activation_allowed": False,
        "owner_authority_required": True,
        "bootstrap_held": bootstrap_held,
        "rl_held": rl_held,
        "window": {"start": "2026-07-24", "end": "2026-08-05"},
        "inputs": {
            "expert": str(FULL_EXPERT),
            "expert_sha256": sha256(FULL_EXPERT),
            "guide_ready": str(FULL_GUIDE),
            "guide_ready_sha256": sha256(FULL_GUIDE),
            "manifest": str(FULL_MANIFEST),
            "manifest_sha256": sha256(FULL_MANIFEST),
            "cpu_pack": str(FULL_CPU_PACK),
            "cpu_pack_ready": str(CPU_READY),
            "cpu_pack_ready_sha256": sha256(CPU_READY),
            "markup": str(FULL_MARKUP),
            "markup_receipt": str(MARKUP_RECEIPT),
            "markup_receipt_sha256": sha256(MARKUP_RECEIPT),
            "markup_manifest_sha256": sha256(FULL_MARKUP / "manifest.json"),
            "targets": str(FULL_TARGETS),
            "targets_sha256": sha256(FULL_TARGETS),
            "pilot_map": str(FULL_PILOT),
            "pilot_map_sha256": sha256(FULL_PILOT),
            "held_split": str(FULL_HELD),
            "held_split_sha256": sha256(FULL_HELD),
            "pilot_importance": str(FULL_UPWEIGHT),
            "pilot_importance_sha256": sha256(FULL_UPWEIGHT),
        },
        "totals": {
            "source_games": 2699,
            "source_decisions": 182951,
            "marked_games": 2515,
            "marked_decisions": 171243,
            "excluded_unsupported_or_unroutable": 184,
        },
        "bootstrap_prestage": {
            "dropin": str(PRESTAGE_DROPIN),
            "dropin_sha256": sha256(PRESTAGE_DROPIN),
            "staged_bindings": staged_bindings,
            "effective_exec_start": "/bin/false",
            "hold_wins_after_prestage_dropin": True,
        },
        "exact_launch_preflight": {
            "path": str(LAUNCH_PREFLIGHT),
            "sha256": sha256(LAUNCH_PREFLIGHT),
            "status": launch_preflight["status"],
            "side_effect_free": True,
            "trainer_parent_and_structure_validated": True,
            "ephemeral_h10_successor_migration_validated": True,
            "expert_manifest_and_all_heads_validated": True,
            "strategic_directional_guide_and_routes_validated": True,
            "pilot_importance_validated": True,
            "cpu_pack_binding_validated": True,
            "matchup_markup_validated": True,
            "rtp_shard_validated": True,
        },
        "guide_ready_repair": {
            "path": str(GUIDE_READY_REPAIR),
            "sha256": sha256(GUIDE_READY_REPAIR),
            "preimage_sha256": guide_repair["preimage_sha256"],
            "postimage_sha256": guide_repair["postimage_sha256"],
            "guide_rows": 16580,
            "activation_held_during_repair": True,
        },
        "superseded_historical_plan": {
            "path": str(OLD_PLAN),
            "status": plan.get("status"),
            "train_corpus": plan.get("train_corpus"),
            "cpu_pack": plan.get("cpu_pack"),
        },
        "adapter": {
            "marked_corpus_ready": True,
            "runtime_routes_enabled_in_markup": False,
            "authorization_exists": ADAPTER_AUTH.is_file(),
            "authorization_timing": "after_ce_parent_before_rl_start",
        },
        "rtp": {
            "current_shard": str(OLD_RTP_SHARD),
            "current_shard_exists": OLD_RTP_SHARD.is_file(),
            "r173_receipt": str(RTP_RECEIPT),
            "r173_receipt_sha256": sha256(RTP_RECEIPT),
            "r173_receipt_valid": rtp_receipt_valid,
            "games": rtp_receipt.get("coverage", {}).get("games"),
            "combo_state_rows": rtp_receipt.get("coverage", {}).get(
                "combo_state_rows"
            ),
            "fresh_full_window_cut_required_before_activation": False,
            "note": (
                "The r173 RTP cut is a separately sealed 42,453-game, "
                "combo-labeled shard; Jul24-Aug5 expands CE/rehearsal inputs "
                "and does not invalidate that cut."
            ),
        },
        "blockers": blockers,
        "ce_activation_blockers": [
            "revision_175_explicit_ce_recovery_hold_requires_owner_lift"
        ],
        "rl_activation_blockers": [
            "complete_fresh_ce_and_bind_checksum_exact_parent",
            "materialize_parent_bound_adapter_authorization",
            "explicit_owner_authority_to_lift_rl_hold",
        ],
        "next_authorized_actions": [
            "retain_passed_exact_launch_preflight",
            "retain_repaired_checksum_bound_guide_ready_projection",
            "do_not_lift_bootstrap_or_rl_hold_without_explicit_owner_authority",
        ],
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
