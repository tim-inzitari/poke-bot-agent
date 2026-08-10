#!/usr/bin/env python3
"""Poll markup/schema8/combo gates; start fresh Slop Box H10+RTP bootstrap when ready."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/inzi/poke-bot-agent")
STATE = ROOT / "outputs" / "state"
WAIT = STATE / "slop-box-fresh-bootstrap-wait-gates-r171.json"
MARKUP = STATE / "slop-box-h10-rtp-expert-matchup-markup-r171.json"
MARKUP_ALT = STATE / "slop-box-expert-self-play-equivalent-markup-r171.json"
COMBO = ROOT / "state" / "slop-box-combo-state-rematerialization-r173.json"
COMBO_SMOKE = ROOT / "state" / "slop-box-combo-state-rematerialization-r173.smoke500.json"
COMBO_TARGETS = ROOT / "state" / "slop-box-combo-state-targets-r173.json"
COMBO_PACK = STATE / "slop-box-combo-cpu-pack-included-r173.json"
COMBO_PACK_ALT = STATE / "slop-box-expert-cpu-pack-combo-r173.json"
SCHEMA8_PROGRESS = STATE / "slop-box-schema8-setup-board-activation-progress-r170.json"
SCHEMA8_CPU = STATE / "slop-box-schema8-cpu-pack-ready-r170.json"
SCHEMA8_SEAL = STATE / "slop-box-schema8-setup-board-sealed-r170.json"
SCHEMA8_PREBUILD = STATE / "slop-box-schema8-cpu-pack-prebuild-r170.json"
WIPE = STATE / "slop-box-learned-weights-wipe-r171.json"
BOOTSTRAP_UNIT = "pokebot-final-format-slop-box-h10-rtp-bootstrap.service"
POLL_SEC = 30
COMBO_TRAJECTORY = (
    ROOT / "outputs/bootstrap/slop-box-h10-rtp/expert_trajectory_shard.combo_v1.jsonl"
)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _markup_ready() -> tuple[bool, dict[str, Any]]:
    for path in (MARKUP, MARKUP_ALT):
        d = _read(path)
        if not d:
            continue
        status = str(d.get("status") or "").lower()
        tickets = 0
        for key in (
            "ticketed_games",
            "marked_games",
            "source_games",
            "games",
            "records",
            "marked_decisions",
            "source_decisions",
        ):
            if key in d and isinstance(d[key], (int, float)):
                tickets = max(tickets, int(d[key]))
        em = d.get("expert_feature_markup") or d.get("totals") or {}
        if isinstance(em, dict):
            for key in (
                "ticketed_games",
                "marked_games",
                "games",
                "records_kept",
                "records",
                "marked_decisions",
            ):
                if key in em and isinstance(em[key], (int, float)):
                    tickets = max(tickets, int(em[key]))
        ready = tickets > 0 and (
            "complete" in status
            or "ready" in status
            or bool(d.get("ready_for_fresh_bootstrap"))
            or bool(d.get("markup_complete"))
        )
        return ready, {
            "path": str(path),
            "status": status,
            "ticketed_games": tickets,
            "marked_games": d.get("marked_games"),
            "marked_decisions": d.get("marked_decisions"),
            "receipt_digest": d.get("receipt_digest"),
        }
    return False, {"path": None, "status": "absent"}


def _combo_remat_ready() -> tuple[bool, dict[str, Any]]:
    d = _read(COMBO)
    if not d:
        smoke = _read(COMBO_SMOKE)
        return False, {
            "path": str(COMBO) if not smoke else str(COMBO_SMOKE),
            "status": "smoke_only" if smoke else "absent",
            "full_corpus_attach_ready": False,
            "pack_included": False,
        }
    cov = d.get("coverage") or {}
    rows = int(cov.get("combo_state_rows") or cov.get("decisions") or 0)
    nonzero = bool(d.get("combo_loss_can_be_nonzero"))
    games = int(cov.get("games") or 0)
    remat_ok = nonzero and rows > 0 and games >= 1000
    return remat_ok, {
        "path": str(COMBO),
        "games": games,
        "combo_state_rows": rows,
        "combo_loss_can_be_nonzero": nonzero,
        "status": d.get("status"),
        "remat_ready": remat_ok,
    }


def _combo_pack_included() -> tuple[bool, dict[str, Any]]:
    """Expert CE needs combo inside rebuilt CPU packs — remat alone is not enough."""
    for path in (COMBO_PACK, COMBO_PACK_ALT):
        d = _read(path)
        if not d:
            continue
        status = str(d.get("status") or "").lower()
        nonzero = bool(
            d.get("combo_masks_nonzero")
            or d.get("combo_state_targets")
            or d.get("combo_loss_can_be_nonzero")
            or d.get("pack_includes_combo_state")
            or ((d.get("coverage") or {}).get("combo_state_rows") or 0) > 0
        )
        ready = nonzero and (
            "ready" in status
            or "bound" in status
            or "complete" in status
            or bool(d.get("pack_included"))
        )
        if ready:
            return True, {
                "path": str(path),
                "status": status,
                "pack_included": True,
                "combo_masks_nonzero": nonzero,
            }

    targets = _read(COMBO_TARGETS) or {}
    loss = targets.get("loss_nonzero_now") or {}
    if bool(loss.get("existing_expert_cpu_packs")):
        return True, {
            "path": str(COMBO_TARGETS),
            "status": targets.get("status"),
            "pack_included": True,
            "via": "loss_nonzero_now.existing_expert_cpu_packs",
        }

    for path in (SCHEMA8_CPU, SCHEMA8_PREBUILD):
        d = _read(path)
        if not d:
            continue
        cpu = d.get("cpu_pack") if isinstance(d.get("cpu_pack"), dict) else {}
        combo_ok = bool(
            d.get("combo_masks_nonzero")
            or d.get("combo_state_targets")
            or d.get("pack_includes_combo_state")
            or d.get("combo_loss_can_be_nonzero")
            or cpu.get("combo_masks_nonzero")
            or cpu.get("combo_state_targets")
            or ((d.get("combo_coverage") or {}).get("combo_state_rows") or 0) > 0
        )
        status = str(d.get("status") or "").lower()
        if combo_ok and (
            "ready" in status or "bound" in status or "complete" in status or path == SCHEMA8_PREBUILD
        ):
            return True, {
                "path": str(path),
                "status": status,
                "pack_included": True,
                "via": "schema8_cpu_pack_combo_fields",
            }

    return False, {
        "path": None,
        "status": "remat_ready_pack_rebuild_pending"
        if (_read(COMBO) or {}).get("status")
        else "absent",
        "pack_included": False,
        "note": "need expert CPU pack rebuilt with nonzero combo masks before CE",
    }


def _combo_ready() -> tuple[bool, dict[str, Any]]:
    remat_ok, remat_info = _combo_remat_ready()
    pack_ok, pack_info = _combo_pack_included()
    info = {
        **remat_info,
        "pack_included": pack_ok,
        "pack": pack_info,
        "status": (
            "ready_pack_included"
            if (remat_ok and pack_ok)
            else (
                "remat_ready_pack_rebuild_pending"
                if remat_ok
                else remat_info.get("status")
            )
        ),
    }
    return remat_ok and pack_ok, info


def _cpu_pack_has_combo(d: dict[str, Any]) -> bool:
    cpu = d.get("cpu_pack") if isinstance(d.get("cpu_pack"), dict) else {}
    pre = _read(SCHEMA8_PREBUILD) or {}
    return bool(
        d.get("combo_state_targets")
        or d.get("require_combo_state")
        or d.get("combo_masks_nonzero")
        or d.get("pack_includes_combo_state")
        or d.get("combo_loss_can_be_nonzero")
        or cpu.get("combo_state_targets")
        or cpu.get("require_combo_state")
        or pre.get("combo_state_targets")
        or ((d.get("combo_coverage") or {}).get("combo_state_rows") or 0) > 0
    )


def _schema8_ready() -> tuple[bool, dict[str, Any]]:
    """Require rebuilt schema-8 CPU pack with combo (--require-combo-state). Seal alone is not enough."""
    d = _read(SCHEMA8_CPU)
    if d:
        status = str(d.get("status") or "").lower()
        labels = bool(
            d.get("setup_board_labels_present")
            or d.get("any_setup_labels")
            or d.get("setup_labels_present")
        )
        packing = d.get("packing_schema") or (d.get("cpu_pack") or {}).get(
            "packing_schema"
        )
        records = int(
            d.get("records")
            or d.get("records_kept")
            or ((d.get("totals") or {}).get("records") if isinstance(d.get("totals"), dict) else 0)
            or 0
        )
        combo_ok = _cpu_pack_has_combo(d)
        ready = (
            labels
            and records > 0
            and combo_ok
            and packing in (8, "8")
            and (
                "ready" in status
                or "bound" in status
                or "complete" in status
            )
        )
        info = {
            "path": str(SCHEMA8_CPU),
            "status": status,
            "records": records,
            "packing_schema": packing,
            "setup_board_labels_present": labels,
            "require_combo_state": combo_ok,
            "seal_present": SCHEMA8_SEAL.is_file(),
        }
        if ready:
            return True, info
        return False, {
            **info,
            "note": "need schema8 CPU pack ready_bound with packing_schema=8 + combo_state_targets",
        }
    seal = _read(SCHEMA8_SEAL) or {}
    prog = _read(SCHEMA8_PROGRESS) or {}
    remat = prog.get("schema8_remat") or {}
    return False, {
        "path": str(SCHEMA8_PROGRESS) if prog else (str(SCHEMA8_SEAL) if seal else None),
        "status": remat.get("status") or prog.get("status") or seal.get("status") or "absent",
        "setup_board_labels_yet": remat.get("setup_board_labels_yet")
        or seal.get("setup_board_labels_present"),
        "require_combo_state": False,
        "note": "need schema-8 CPU pack rebuilt with --require-combo-state (seal alone insufficient)",
    }


def _wipe_confirmed() -> tuple[bool, dict[str, Any]]:
    d = _read(WIPE)
    if not d:
        return False, {"path": str(WIPE), "status": "absent"}
    status = str(d.get("status") or "").lower()
    ready = any(
        token in status
        for token in ("quarantine", "wiped", "complete", "done", "awaiting")
    ) or bool(d.get("quarantine_root") or d.get("quarantine"))
    return ready, {
        "path": str(WIPE),
        "status": status,
        "quarantine_root": d.get("quarantine_root") or d.get("quarantine"),
    }


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_bootstrap_unit() -> Path:
    unit = Path.home() / ".config/systemd/user" / BOOTSTRAP_UNIT
    unit.parent.mkdir(parents=True, exist_ok=True)
    expert = (
        ROOT
        / "data/bootstrap/expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex"
    )
    # Prefer schema8-bound expert pointer if sibling published one.
    schema8_expert = _read(SCHEMA8_CPU) or {}
    expert_ptr = Path(
        str(
            schema8_expert.get("protected_expert_corpus")
            or schema8_expert.get("expert")
            or (expert / "PROTECTED_EXPERT_CORPUS.json")
        )
    )
    guide_ready = Path(
        str(
            schema8_expert.get("guide_ready")
            or (expert / "CURRENT_DECK_GUIDE_CORPUS_READY.json")
        )
    )
    run_dir = ROOT / "outputs/bootstrap/final_format_slop_box_h10_rtp_fresh_r171"
    cpu_pack = ROOT / "outputs/bootstrap/cpu-packs/final_format_slop_box_h10_rtp_fresh_r171"
    schema8_cpu = _read(SCHEMA8_CPU) or {}
    schema8_root = ((schema8_cpu.get("cpu_pack") or {}) if isinstance(schema8_cpu.get("cpu_pack"), dict) else {}).get("root")
    if schema8_root:
        cpu_pack = Path(str(schema8_root))
    rtp_shard = COMBO_TRAJECTORY if COMBO_TRAJECTORY.is_file() else (
        ROOT / "outputs/bootstrap/slop-box-h10-rtp/expert_trajectory_shard.jsonl"
    )
    curriculum = ROOT / "outputs/state/final_format_slop_box_curriculum_fresh_r171"
    family = "final-format-slop-box-h10-rtp-expert-bootstrap-fresh-r171"
    py = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
    h10_env = ROOT / "config/final_format_marnie_h10_runtime.env"
    text = f"""[Unit]
Description=Fresh Bootstrap Slop Box H10 expert + RTP cut (r171 wipe restart)
After=network-online.target
Wants=network-online.target
ConditionPathExists={ROOT}/state/slop_box_h10_rtp_prestage_identity_r170.json
ConditionPathExists={ROOT}/outputs/state/crustle-owner-abandon-r170.json
ConditionPathExists={ROOT}/outputs/bootstrap/slop-box-h10-rtp/expert_trajectory_shard.jsonl
ConditionPathExists={WAIT}

[Service]
Type=oneshot
WorkingDirectory={ROOT}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH={ROOT}
Environment=ALLOW_TF32=1
Environment=NVIDIA_TF32_OVERRIDE=1
EnvironmentFile=-{h10_env}
Environment=POKEBOT_COMBO_STATE_HEAD_ENABLED=1
Environment=POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=1
Environment=PURE_RL_SPATIAL_LAYERS=7
Environment=PURE_RL_TEMPORAL_LAYERS=3
Environment=PURE_RL_OPTION_DECODER_LAYERS=7
Environment=PURE_RL_FF_DIM=2496
Environment=POKEBOT_H10_CAPACITY_ENABLED=1
Environment=POKEBOT_H10_HEAD_RESIDUAL_WIDTH=512
ExecStart={py} -u {ROOT}/scripts/run_slop_box_h10_rtp_bootstrap.py \\
  --identity {ROOT}/state/slop_box_h10_rtp_prestage_identity_r170.json \\
  --parent {ROOT}/outputs/pure_rl/final_format_marnie_r104_h10_i_v6_8k/checkpoints/iter_00007.pt \\
  --parent-digest sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381 \\
  --guide {ROOT}/config/deck_guides/slop-box-h10-rtp-north-star-v1.yaml \\
  --guide-ready {guide_ready} \\
  --expert {expert_ptr} \\
  --protocol {ROOT}/config/rl_protocol.yaml \\
  --registry-root {ROOT}/outputs/pure_rl/_protected/models \\
  --ready {STATE}/final-format-slop-box-h10-rtp-bootstrap-ready.json \\
  --run-dir {run_dir} \\
  --cpu-pack-root {cpu_pack} \\
  --curriculum-root {curriculum} \\
  --family {family} \\
  --batch-size 2048 \\
  --epochs 300 \\
  --owner-slop-box-ce-intensify-epochs 300 \\
  --pilot-importance-index {STATE}/slop-box-cox-chao-train-upweight-importance-schema8-fresh-r171.json \\
  --rtp-shard {rtp_shard} \\
  --rtp-out-dir {ROOT}/outputs/rtp_fleet/slop-box-h10 \\
  --rtp-live-checkpoint {ROOT}/outputs/rtp_fleet/slop-box-h10.live/rtp_shadow_planner.pt \\
  --rtp-cut-receipt {STATE}/slop-box-h10-rtp-bootstrap-cut-r170.json \\
  --rtp-device cuda:0 \\
  --rtp-epochs 8 \\
  --rtp-max-games 42453 \\
  --rtp-parallel
Restart=no
TimeoutStartSec=infinity
MemoryHigh=96G
MemoryMax=112G
MemorySwapMax=0
StandardOutput=append:{ROOT}/outputs/final_format_slop_box_h10_rtp/logs/bootstrap-fresh-r171.log
StandardError=append:{ROOT}/outputs/final_format_slop_box_h10_rtp/logs/bootstrap-fresh-r171.log

[Install]
WantedBy=default.target
"""
    unit.write_text(text, encoding="utf-8")
    return unit


def _start_bootstrap() -> dict[str, Any]:
    unit_path = _write_bootstrap_unit()
    # Ensure log dir exists
    (ROOT / "outputs/final_format_slop_box_h10_rtp/logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/bootstrap/final_format_slop_box_h10_rtp_fresh_r171").mkdir(
        parents=True, exist_ok=True
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(
        ["systemctl", "--user", "reset-failed", BOOTSTRAP_UNIT], check=False
    )
    # Keep RL held
    subprocess.run(
        ["systemctl", "--user", "stop", "pokebot-final-format-slop-box-h10-rtp-rl.service"],
        check=False,
    )
    subprocess.run(["systemctl", "--user", "start", BOOTSTRAP_UNIT], check=False)
    time.sleep(3)
    show = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            BOOTSTRAP_UNIT,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "Result",
            "--no-pager",
        ],
        text=True,
    )
    props = dict(
        line.split("=", 1) for line in show.splitlines() if "=" in line
    )
    return {
        "unit": BOOTSTRAP_UNIT,
        "unit_path": str(unit_path),
        "ActiveState": props.get("ActiveState"),
        "SubState": props.get("SubState"),
        "MainPID": int(props.get("MainPID") or 0),
        "Result": props.get("Result"),
    }


def main() -> int:
    started = False
    while True:
        m_ok, m_info = _markup_ready()
        s_ok, s_info = _schema8_ready()
        c_ok, c_info = _combo_ready()
        w_ok, w_info = _wipe_confirmed()
        now = datetime.now(timezone.utc).isoformat()
        all_ok = m_ok and s_ok and c_ok and w_ok
        wait = _read(WAIT) or {
            "schema": "poke_bot.slop_box_fresh_bootstrap_wait_gates_r171/v1"
        }
        wait.update(
            {
                "updated_at_utc": now,
                "agent": "7bc394c0",
                "status": (
                    "starting_fresh_bootstrap"
                    if (all_ok and not started)
                    else (
                        "fresh_bootstrap_started"
                        if started
                        else "waiting_all_gates_before_fresh_bootstrap"
                    )
                ),
                "fresh_bootstrap_started": started,
                "owner_order": (
                    "WAIT for markup + schema8 CPU pack with --require-combo-state + "
                    "confirmed wipe; then fresh H10+RTP bootstrap; no schema-6"
                ),
                "gates": {
                    "1_self_play_equivalent_markup": {
                        "ready": m_ok,
                        **m_info,
                        "owner": "b90dbf9e",
                    },
                    "2_schema8_cpu_pack_with_combo": {
                        "ready": s_ok,
                        **s_info,
                        "owner": "ed6464fd",
                        "requirement": "schema-8 CPU pack rebuilt with --require-combo-state",
                    },
                    "3_combo_pack_included": {
                        "ready": c_ok,
                        **c_info,
                        "owner": "ed6464fd",
                        "remat_owner": "f93685e2",
                        "requirement": "full remat + expert CPU pack combo_state_targets=true",
                    },
                    "4_wipe_confirmed": {
                        "ready": w_ok,
                        **w_info,
                        "owner": "7bc394c0",
                    },
                },
                "rl_held": True,
                "no_crustle_deletes": True,
            }
        )
        _atomic(WAIT, wait)
        print(
            f"[watch] {now} markup={m_ok} schema8_combo_pack={s_ok} "
            f"combo={c_ok} wipe={w_ok}",
            flush=True,
        )
        if all_ok and not started:
            print("[watch] ALL GATES READY — starting fresh bootstrap", flush=True)
            status = _start_bootstrap()
            started = True
            receipt = {
                "schema": "poke_bot.slop_box_fresh_bootstrap_started_r171/v1",
                "status": "started",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "agent": "7bc394c0",
                "warm_start_parent": (
                    str(
                        ROOT
                        / "outputs/pure_rl/final_format_marnie_r104_h10_i_v6_8k/checkpoints/iter_00007.pt"
                    )
                ),
                "warm_start_digest": (
                    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
                ),
                "gates": wait["gates"],
                "wipe": w_info,
                "systemd": status,
                "rl_still_held": True,
                "no_crustle_deletes": True,
            }
            out = STATE / "slop-box-fresh-bootstrap-started-r171.json"
            _atomic(out, receipt)
            wait["status"] = "fresh_bootstrap_started"
            wait["fresh_bootstrap_started"] = True
            wait["start_receipt"] = str(out)
            wait["MainPID"] = status.get("MainPID")
            _atomic(WAIT, wait)
            print(
                "FRESH_BOOTSTRAP_STARTED",
                json.dumps(status, indent=2, sort_keys=True),
                flush=True,
            )
            # Keep watching until MainPID is live (oneshot may take a moment)
            for _ in range(40):
                show = subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        BOOTSTRAP_UNIT,
                        "-p",
                        "ActiveState",
                        "-p",
                        "SubState",
                        "-p",
                        "MainPID",
                        "--no-pager",
                    ],
                    text=True,
                )
                props = dict(
                    line.split("=", 1) for line in show.splitlines() if "=" in line
                )
                main_pid = int(props.get("MainPID") or 0)
                print(
                    f"[watch] bootstrap ActiveState={props.get('ActiveState')} "
                    f"SubState={props.get('SubState')} MainPID={main_pid}",
                    flush=True,
                )
                if main_pid > 0 and props.get("ActiveState") in {
                    "activating",
                    "active",
                }:
                    receipt["systemd"] = {
                        **status,
                        "ActiveState": props.get("ActiveState"),
                        "SubState": props.get("SubState"),
                        "MainPID": main_pid,
                    }
                    receipt["status"] = "mainpid_live"
                    _atomic(out, receipt)
                    print(f"MAINPID_LIVE {main_pid}", flush=True)
                    return 0
                if props.get("ActiveState") == "failed":
                    receipt["status"] = "failed"
                    receipt["systemd"] = props
                    _atomic(out, receipt)
                    return 2
                time.sleep(3)
            return 0
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
