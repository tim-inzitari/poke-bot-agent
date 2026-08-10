#!/usr/bin/env python3
"""Owner-ceiling proceed for Slop Box when Chao-hard overfits (r171).

Signal (owner clarification):
  - train ≫ held and held stuck ≪ 0.90 (e.g. train>=0.85 and held<=0.80
    across multiple measured epochs), OR
  - Chao-hard finished with best held << 0.90 on the same plateau.
  Do not keep spinning for a measured 0.90 pass.

Actions (systemd-only for training control):
  1) stop Chao-hard oneshot via systemctl
  2) Chao-held select across available epochs
  3) freeze best checkpoint as protected bootstrap family
  4) publish ready receipt under explicit owner ceiling (never a measured pass)
  5) queue nonblocking first_if_allowed Kaggle milestone (Cox/Chao deck)
  6) register + start Slop Box H10 RL with expert rehearsal every 5 iters
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.model_registry import freeze_model, sha256  # noqa: E402
from scripts.handle_passed_gate import (  # noqa: E402
    _copy_submission_slot,
    build_submission_bundle,
    queue_submission_copies,
)

SPECIALIST = "teal-mask-ogerpon-ex"
CHAO_UNIT = "pokebot-slop-box-chao-hard-ce-r170.service"
RL_UNIT = "pokebot-final-format-slop-box-h10-rtp-rl.service"
REGISTER_UNIT = "pokebot-final-format-slop-box-h10-rtp-register.service"
RUN_DIR = Path(
    "/home/inzi/poke-bot-agent/outputs/bootstrap/final_format_slop_box_chao_hard_r170"
)
STATE = Path("/home/inzi/poke-bot-agent/outputs/state")
PROTECTED = Path("/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models")
FAMILY = "final-format-slop-box-h10-rtp-expert-bootstrap-v1"
DECK_CSV = Path(
    "/home/inzi/poke-bot-agent/outputs/submissions/"
    "slop-box-h10-rtp-cox-chao-55188658/pinned-teal-mask-ogerpon-ex.deck.csv"
)
MATCHUP = Path(
    "/home/inzi/poke-bot-agent/outputs/state/slowking-public-matchup-tree-v33.json"
)
EXPERT = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex/"
    "PROTECTED_EXPERT_CORPUS.json"
)
GUIDE = Path(
    "/home/inzi/poke-bot-agent/config/deck_guides/slop-box-h10-rtp-north-star-v1.yaml"
)
CURRICULUM = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final_format_slop_box_chao_hard_curriculum_r170"
)
RTP_LIVE = Path(
    "/home/inzi/poke-bot-agent/outputs/rtp_fleet/slop-box-h10.live/"
    "rtp_shadow_planner.pt"
)
PY = Path("/home/inzi/miniconda3/envs/poke-bot-agent/bin/python")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _unit_state(unit: str) -> dict[str, str]:
    out = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            unit,
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
    data: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k] = v
    return data


def latest_train_acc() -> tuple[int | None, float | None]:
    log = Path(
        "/home/inzi/poke-bot-agent/outputs/final_format_slop_box_chao_hard_r170/"
        "logs/chao_hard.log"
    )
    if not log.is_file():
        return None, None
    pat = re.compile(r"expert rehearsal before iter(\d+) ep5/5:.*?acc=([0-9.]+)%")
    by: dict[int, float] = {}
    for ln in re.split(r"[\r\n]+", log.read_text(errors="ignore")):
        m = pat.search(ln)
        if m:
            by[int(m.group(1))] = float(m.group(2)) / 100.0
    if not by:
        return None, None
    ep = max(by)
    return ep, by[ep]


def epochs_done() -> int:
    cks = sorted(
        (RUN_DIR / "checkpoints").glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return int(cks[-1].stem.split("_")[1]) if cks else 0


def overfit_signal(held_select: dict[str, Any] | None) -> dict[str, Any]:
    us = _unit_state(CHAO_UNIT)
    finished = us.get("ActiveState") == "inactive" and us.get("SubState") == "dead"
    train_ep, train_acc = latest_train_acc()
    held_rows = []
    if held_select:
        for key in ("candidates", "evaluations", "results"):
            rows = held_select.get(key)
            if isinstance(rows, list) and rows:
                held_rows = rows
                break
    held_vals = [
        float(r["cox_chao_held_policy_acc"])
        for r in held_rows
        if r.get("cox_chao_held_policy_acc") is not None
    ]
    best_held = max(held_vals) if held_vals else None
    flat_low = (
        len(held_vals) >= 3
        and all(v <= 0.80 for v in held_vals[-min(5, len(held_vals)) :])
    ) or (
        best_held is not None
        and best_held <= 0.80
        and epochs_done() >= 8
    )
    # Owner clarification: train ≫ held with held stuck ≪0.90 — do not wait for 0.95.
    train_high = train_acc is not None and train_acc >= 0.85
    gap = (
        (train_acc - best_held)
        if train_acc is not None and best_held is not None
        else None
    )
    overfit_now = bool(
        train_high and flat_low and gap is not None and gap >= 0.10
    )
    natural_ceiling = finished and best_held is not None and best_held < 0.90
    triggered = overfit_now or natural_ceiling
    return {
        "triggered": triggered,
        "finished": finished,
        "train_ep": train_ep,
        "train_acc": train_acc,
        "best_held": best_held,
        "flat_low_held": flat_low,
        "train_high": train_high,
        "train_held_gap": gap,
        "overfit_now": overfit_now,
        "natural_ceiling": natural_ceiling,
        "epochs_done": epochs_done(),
        "unit": us,
    }


def stop_chao_hard() -> None:
    subprocess.run(["systemctl", "--user", "stop", CHAO_UNIT], check=False)


def run_held_select() -> dict[str, Any]:
    out = STATE / "slop-box-chao-hard-held-select-ceiling-r171.json"
    cmd = [
        str(PY),
        "-u",
        str(ROOT / "scripts" / "select_slop_box_chao_held_checkpoint_r170.py"),
        "--checkpoint-dir",
        str(RUN_DIR / "checkpoints"),
        "--epochs",
        "all",
        "--expert-pointer",
        str(EXPERT),
        "--targets",
        str(STATE / "slop-box-cox-chao-held-targets-expanded-r170.json"),
        "--pilot-map",
        str(STATE / "slop-box-cox-chao-held-pilot-map-expanded-r170.json"),
        "--held-split",
        str(STATE / "slop-box-cox-chao-held-split-pilot-map-expanded-r170.json"),
        "--cpu-pack-root",
        "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
        "final_format_slop_box_chao_hard_r170",
        "--output",
        str(out),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    print("[ceiling] held select", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    return json.loads(out.read_text(encoding="utf-8"))


def select_checkpoint(held: dict[str, Any]) -> Path:
    rec = held.get("hot_start_recommendation") or {}
    path = Path(str(rec.get("path") or ""))
    if path.is_file():
        return path.resolve()
    # fallback: latest fusion-valid epoch
    cks = sorted(
        (RUN_DIR / "checkpoints").glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not cks:
        raise RuntimeError("no Chao-hard checkpoints")
    return cks[-1].resolve()


def freeze_family(checkpoint: Path, held: dict[str, Any]) -> dict[str, Any]:
    digest = sha256(checkpoint)
    provenance = {
        "acting_seat_archetype": SPECIALIST,
        "owner_decision_revision": 171,
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "chao_held_select": held.get("hot_start_recommendation"),
        "source_run_dir": str(RUN_DIR),
        "measured_gate_never_pass": True,
    }
    return freeze_model(
        registry_root=PROTECTED,
        family=FAMILY,
        display_name="Slop Box H10 RTP Expert Bootstrap (owner ceiling r171)",
        checkpoint=checkpoint,
        expected_digest=digest,
        provenance=provenance,
        evidence={
            "kind": "cox_chao_held_policy_acc",
            "measured": (held.get("hot_start_recommendation") or {}).get(
                "cox_chao_held_policy_acc"
            ),
            "threshold": 0.9,
            "passed": False,
            "select_receipt": str(
                STATE / "slop-box-chao-hard-held-select-ceiling-r171.json"
            ),
        },
    )


def build_representatives() -> Path:
    cards: list[str] = []
    with DECK_CSV.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() in {"card_id", "id", "card"}:
                continue
            cards.append(row[0].strip())
    if len(cards) != 60:
        raise RuntimeError(f"Cox/Chao deck must be 60 cards, got {len(cards)}")
    path = STATE / "slop-box-cox-chao-owner-ceiling-representatives-r171.json"
    payload = {
        "schema": "poke_bot.specialist_deck_representatives/v1",
        "specialist_id": SPECIALIST,
        "updated_at_utc": _now(),
        "owner_decision_revision": 171,
        "decks": {SPECIALIST: {"card_ids": cards, "source_csv": str(DECK_CSV)}},
    }
    _atomic_json(path, payload)
    return path


def publish_ready(manifest: dict[str, Any], held: dict[str, Any]) -> Path:
    failed = STATE / (
        "final-format-slop-box-h10-rtp-bootstrap-ready-FAILED-cox-chao-gate-r170.json"
    )
    base = json.loads(failed.read_text(encoding="utf-8")) if failed.is_file() else {}
    measured = (held.get("hot_start_recommendation") or {}).get(
        "cox_chao_held_policy_acc"
    )
    ready = dict(base)
    ready.update(
        {
            "schema": "poke_bot.specialist_expert_bootstrap_ready/v1",
            "status": "ready_owner_ceiling_acceptance",
            "completion_authority": "explicit_owner_ceiling_acceptance",
            "measured_gate_passed": False,
            "family": FAMILY,
            "checkpoint": manifest["model_path"],
            "checkpoint_digest": manifest["checkpoint_digest"],
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "owner_decision_revision": 171,
            "acting_seat_archetype": SPECIALIST,
            "cox_chao_policy_accuracy_gate": {
                "fail_closed": True,
                "measured": measured,
                "threshold": 0.9,
                "passed": False,
                "metric_id": "cox_chao_held_policy_acc",
                "authority": "explicit_owner_ceiling_acceptance",
                "never_relabel_as_measured_pass": True,
                "select_receipt": str(
                    STATE / "slop-box-chao-hard-held-select-ceiling-r171.json"
                ),
                "select_receipt_sha256": sha256(
                    STATE / "slop-box-chao-hard-held-select-ceiling-r171.json"
                ),
            },
        }
    )
    path = STATE / "final-format-slop-box-h10-rtp-bootstrap-ready.json"
    _atomic_json(path, ready)
    return path


def _deck_cards_sha256(deck_csv: Path) -> str:
    import hashlib as _hl

    cards = []
    for line in deck_csv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower() in {"card_id", "id", "card"}:
            continue
        cards.append(int(line.split(",")[0]))
    if len(cards) != 60:
        raise RuntimeError(f"deck must be 60 cards, got {len(cards)}")
    body = json.dumps(
        cards, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + _hl.sha256(body).hexdigest()


def queue_kaggle(manifest: dict[str, Any], reps: Path) -> dict[str, Any]:
    root = Path(
        "/home/inzi/poke-bot-agent/outputs/submissions/"
        "final-format-slop-box-owner-ceiling-r171"
    )
    root.mkdir(parents=True, exist_ok=True)
    pinned = root / "pinned-teal-mask-ogerpon-ex.deck.csv"
    if not pinned.exists():
        shutil.copy2(DECK_CSV, pinned)
    deck_receipt = {
        "path": str(pinned.resolve()),
        "cards": 60,
        "cards_sha256": _deck_cards_sha256(pinned),
        "file_sha256": sha256(pinned),
        "representatives": str(reps.resolve()),
        "representatives_sha256": sha256(reps),
    }
    bundle = build_submission_bundle(
        repo_root=Path("/home/inzi/poke-bot-agent"),
        frozen_manifest={
            "model_path": manifest["model_path"],
            "checkpoint_digest": manifest["checkpoint_digest"],
        },
        deck_receipt=deck_receipt,
        output_dir=root / "build",
        python=PY,
        archetype=SPECIALIST,
        matchup_tree=MATCHUP,
        turn_order_preference="first_if_allowed",
    )
    copy = _copy_submission_slot(bundle, root, 1)
    # Prefer owner-facing label with RTP note via queue identity fields.
    queued = queue_submission_copies(
        queue_path=STATE / "kaggle-submission-queue.json",
        copies=[copy],
        gate_plan={
            "gate_id": "final-format-slop-box-owner-ceiling-bootstrap-r171",
            "iteration": 0,
            "checkpoint_digest": manifest["checkpoint_digest"],
            "completion_authority": "explicit_owner_ceiling_acceptance",
        },
        specialist_id=SPECIALIST,
        competition="pokemon-tcg-ai-battle",
    )
    receipt = {
        "schema": "poke_bot.slop_box_owner_ceiling_kaggle_r171/v1",
        "status": "queued",
        "turn_order_preference": "first_if_allowed",
        "checkpoint_sha256": manifest["checkpoint_digest"],
        "bundle_sha256": bundle["sha256"],
        "rtp_sidecar": str(RTP_LIVE) if RTP_LIVE.is_file() else None,
        "queued": queued,
        "updated_at_utc": _now(),
    }
    _atomic_json(STATE / "slop-box-owner-ceiling-kaggle-queued-r171.json", receipt)
    return receipt


def write_register_and_rl_units() -> None:
    runtime_root = Path(
        "/home/inzi/poke-bot-agent/outputs/final_format_slop_box_h10_rtp/runtime"
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    logs = Path("/home/inzi/poke-bot-agent/outputs/final_format_slop_box_h10_rtp/logs")
    logs.mkdir(parents=True, exist_ok=True)
    registry = runtime_root / "specialist_runtime_registry_h10_r171.json"
    selector = runtime_root / "specialist_runtime_h10_r171.env"
    unit_dir = Path.home() / ".config/systemd/user"

    register_py = ROOT / "scripts" / "register_slop_box_h10_rtp_r171.py"
    reg_unit = unit_dir / REGISTER_UNIT
    reg_unit.write_text(
        f"""[Unit]
Description=Register Slop Box H10 RTP runtime under owner ceiling r171
After=network-online.target
ConditionPathExists=/home/inzi/poke-bot-agent/outputs/state/final-format-slop-box-h10-rtp-bootstrap-ready.json
OnSuccess={RL_UNIT}

[Service]
Type=oneshot
WorkingDirectory=/home/inzi/poke-bot-agent
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/inzi/poke-bot-agent
ExecStart={PY} -u {register_py} --runtime-registry {registry} --selector {selector}
Restart=no
TimeoutStartSec=infinity
MemoryHigh=8G
MemoryMax=12G
MemorySwapMax=0
StandardOutput=append:{logs}/register.log
StandardError=append:{logs}/register.log
""",
        encoding="utf-8",
    )

    rtp_env = ""
    if RTP_LIVE.is_file():
        rtp_env = f"Environment=POKEBOT_RTP_CHECKPOINT={RTP_LIVE}\n"

    rl_unit = unit_dir / RL_UNIT
    rl_unit.write_text(
        f"""[Unit]
Description=Final-format Slop Box H10 RTP RL (owner ceiling r171)
After={REGISTER_UNIT} network-online.target
Wants=network-online.target
ConditionPathExists={registry}
Conflicts={CHAO_UNIT} pokebot-final-format-slop-box-h10-rtp-bootstrap.service
StartLimitIntervalSec=900
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory=/home/inzi/poke-bot-agent
EnvironmentFile={selector}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/inzi/poke-bot-agent
Environment=POKEBOT_CURRENT_DECK_GUIDE={SPECIALIST}
Environment=PURE_RL_SIM_WORKERS=96
Environment=PURE_RL_GAMES_IN_FLIGHT=96
{rtp_env}ExecStartPre=/usr/bin/test -s /home/inzi/poke-bot-agent/outputs/state/TRAINING_ARMED
ExecStartPre={PY} -u /home/inzi/poke-bot-agent/scripts/launch_active_specialist.py --registry {registry} --check
ExecStart={PY} -u /home/inzi/poke-bot-agent/scripts/launch_active_specialist.py --registry {registry}
Restart=on-failure
RestartSec=60
TimeoutStopSec=45
KillMode=control-group
LimitNOFILE=16384
MemoryHigh=96G
MemoryMax=112G
MemorySwapMax=0
OOMPolicy=stop
StandardOutput=append:{logs}/rl.log
StandardError=append:{logs}/rl.log
""",
        encoding="utf-8",
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)


def start_register() -> None:
    subprocess.run(
        ["systemctl", "--user", "reset-failed", REGISTER_UNIT, RL_UNIT],
        check=False,
    )
    subprocess.run(["systemctl", "--user", "start", REGISTER_UNIT], check=True)


def update_dash(line: str, *, phase: str) -> None:
    now = _now()
    proj = {
        "schema": "poke_bot.slop_box_dashboard_bootstrap_projection_r170/v1",
        "specialist": SPECIALIST,
        "display_name": "Slop Box H10 RTP",
        "phase": phase,
        "status": phase,
        "ready": phase.startswith("rl") or "ceiling" in phase,
        "rl": phase.startswith("rl"),
        "cox_chao_gate_threshold": 0.9,
        "cox_chao_gate_passed": False,
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "latest_line": line,
        "updated_at_utc": now,
    }
    sync = {
        "schema": "poke_bot.slop_box_dashboard_active_sync/v1",
        "goal_revision": 171,
        "phase": phase,
        "latest_line": line,
        "cox_chao_gate_passed": False,
        "cox_chao_gate_threshold": 0.9,
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "crustle_data_deleted": False,
        "updated_at_utc": now,
        "created_at_utc": now,
    }
    _atomic_json(STATE / "slop-box-dashboard-bootstrap-projection-r170.json", proj)
    _atomic_json(STATE / "slop-box-dashboard-active-sync-r170.json", sync)


def execute() -> int:
    done = STATE / "slop-box-owner-ceiling-overfit-proceed-r171.json"
    if done.is_file():
        existing = json.loads(done.read_text(encoding="utf-8"))
        if existing.get("status") == "executed":
            print("[ceiling] already executed; refusing duplicate", flush=True)
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return 0
    update_dash("OWNER CEILING: stopping Chao-hard for select/freeze", phase="bootstrap:owner_ceiling")
    stop_chao_hard()
    held = run_held_select()
    ckpt = select_checkpoint(held)
    print(f"[ceiling] selected {ckpt}", flush=True)
    manifest = freeze_family(ckpt, held)
    ready = publish_ready(manifest, held)
    reps = build_representatives()
    kaggle = queue_kaggle(manifest, reps)
    write_register_and_rl_units()
    # register script must exist beside this file
    start_register()
    receipt = {
        "schema": "poke_bot.slop_box_owner_ceiling_overfit_proceed_r171/v1",
        "status": "executed",
        "goal_revision": 171,
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "measured_gate_passed": False,
        "never_relabel_as_measured_pass": True,
        "selected_checkpoint": str(ckpt),
        "selected_checkpoint_sha256": manifest["checkpoint_digest"],
        "held_select": held.get("hot_start_recommendation"),
        "family": FAMILY,
        "ready": str(ready),
        "ready_sha256": sha256(ready),
        "kaggle": kaggle,
        "register_unit": REGISTER_UNIT,
        "rl_unit": RL_UNIT,
        "rtp_live": str(RTP_LIVE) if RTP_LIVE.is_file() else None,
        "updated_at_utc": _now(),
    }
    _atomic_json(STATE / "slop-box-owner-ceiling-overfit-proceed-r171.json", receipt)
    repo_state = ROOT / "state" / "slop-box-owner-ceiling-overfit-proceed-r171.json"
    _atomic_json(repo_state, receipt)
    measured = (held.get("hot_start_recommendation") or {}).get(
        "cox_chao_held_policy_acc"
    )
    update_dash(
        f"OWNER CEILING active · held={measured} <0.90 (not a pass) · "
        f"Kaggle queued · register/RL starting · ckpt {manifest['checkpoint_digest'][:19]}",
        phase="rl:owner_ceiling_starting",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


def watch_until_trigger(*, poll: int, max_wait: int) -> int:
    start = datetime.now(timezone.utc).timestamp()
    held_path = STATE / "slop-box-chao-hard-held-select-midrun-r170.json"
    while True:
        held = (
            json.loads(held_path.read_text(encoding="utf-8"))
            if held_path.is_file()
            else None
        )
        sig = overfit_signal(held)
        print(json.dumps({"watch": sig}, sort_keys=True), flush=True)
        line = (
            f"WATCH ceiling · ep={sig['epochs_done']} train={sig['train_acc']} "
            f"best_held={sig['best_held']} trigger={sig['triggered']}"
        )
        update_dash(line, phase="bootstrap:chao_hard_ce")
        if sig["triggered"]:
            return execute()
        if datetime.now(timezone.utc).timestamp() - start > max_wait:
            print("[ceiling] watch timeout; not triggered", flush=True)
            return 2
        # refresh midrun select every ~10 min on newest epochs
        if sig["epochs_done"] >= 15 and (
            not held_path.is_file()
            or held_path.stat().st_mtime
            < datetime.now().timestamp() - 600
        ):
            try:
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = "0"
                eps = ",".join(
                    str(e)
                    for e in range(max(1, sig["epochs_done"] - 4), sig["epochs_done"] + 1)
                )
                subprocess.run(
                    [
                        str(PY),
                        "-u",
                        str(ROOT / "scripts" / "select_slop_box_chao_held_checkpoint_r170.py"),
                        "--checkpoint-dir",
                        str(RUN_DIR / "checkpoints"),
                        "--epochs",
                        eps,
                        "--expert-pointer",
                        str(EXPERT),
                        "--targets",
                        str(STATE / "slop-box-cox-chao-held-targets-expanded-r170.json"),
                        "--pilot-map",
                        str(STATE / "slop-box-cox-chao-held-pilot-map-expanded-r170.json"),
                        "--held-split",
                        str(
                            STATE
                            / "slop-box-cox-chao-held-split-pilot-map-expanded-r170.json"
                        ),
                        "--cpu-pack-root",
                        "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
                        "final_format_slop_box_chao_hard_r170",
                        "--output",
                        str(held_path),
                    ],
                    check=False,
                    env=env,
                )
            except Exception as exc:
                print(f"[ceiling] midrun refresh failed: {exc}", flush=True)
        import time

        time.sleep(poll)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-now", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=4 * 3600)
    args = parser.parse_args()
    if args.execute_now:
        return execute()
    if args.watch:
        return watch_until_trigger(
            poll=int(args.poll_seconds), max_wait=int(args.max_wait_seconds)
        )
    held_path = STATE / "slop-box-chao-hard-held-select-midrun-r170.json"
    held = (
        json.loads(held_path.read_text(encoding="utf-8"))
        if held_path.is_file()
        else None
    )
    print(json.dumps(overfit_signal(held), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
