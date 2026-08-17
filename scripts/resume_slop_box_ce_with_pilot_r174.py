#!/usr/bin/env python3
"""Canonical r174: at next clean epoch boundary, resume CE with Cox/Chao pilot.

Does not SIGKILL mid-step. Stops via systemctl after epoch_N.pt lands, rewrites
unit with --pilot-importance-index (schema8-fresh), same run-dir, --epochs 40,
then starts. Epoch-40 orchestrator remains owner of stop→Kaggle→RL.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/pokebot/poke-bot-agent")
STATE = ROOT / "outputs" / "state"
RUN = ROOT / "outputs/bootstrap/final_format_slop_box_h10_rtp_fresh_r171"
CKPT_DIR = RUN / "checkpoints"
BOOTSTRAP_UNIT = "pokebot-final-format-slop-box-h10-rtp-bootstrap.service"
PILOT = STATE / "slop-box-cox-chao-train-upweight-importance-schema8-fresh-r171.json"
PROGRESS = STATE / "slop-box-pilot-resume-progress-r174.json"
SYNC = STATE / "slop-box-canonical-sync-ack-r174.json"
POLL_SEC = 15
TARGET_EPOCHS = 40


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _show(unit: str) -> dict[str, str]:
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
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _progress(**kwargs: Any) -> None:
    base = _read(PROGRESS) or {
        "schema": "poke_bot.slop_box_pilot_resume_progress_r174/v1"
    }
    base.update({"updated_at_utc": _now(), **kwargs})
    _atomic(PROGRESS, base)
    print(json.dumps({"t": _now(), **kwargs}, sort_keys=True), flush=True)


def _pilot_valid() -> tuple[bool, dict[str, Any]]:
    d = _read(PILOT)
    if not d:
        return False, {"path": str(PILOT), "status": "absent"}
    prot = _read(
        ROOT
        / "data/bootstrap/expert-slop-box-schema8-jul24-30-r170/teal-mask-ogerpon-ex/PROTECTED_EXPERT_CORPUS.json"
    ) or {}
    ok = (
        d.get("status") == "ready"
        and int(d.get("train_games") or 0) > 0
        and d.get("corpus_manifest_sha256")
        and d.get("corpus_manifest_sha256") == prot.get("manifest_sha256")
        and int(d.get("matched_top_100_train_games") or 0) > 0
    )
    return ok, {
        "path": str(PILOT),
        "status": d.get("status"),
        "train_games": d.get("train_games"),
        "matched_top_100_train_games": d.get("matched_top_100_train_games"),
        "corpus_manifest_sha256": d.get("corpus_manifest_sha256"),
        "live_corpus_sha256": prot.get("manifest_sha256"),
        "corpus_match": d.get("corpus_manifest_sha256") == prot.get("manifest_sha256"),
        "valid": ok,
    }


def _epoch_state() -> dict[str, Any]:
    st = _read(RUN / "state.json") or {}
    hist = st.get("history") or []
    last = int(hist[-1]["epoch"]) if hist else 0
    ckpt = CKPT_DIR / f"epoch_{last:02d}.pt"
    return {
        "history_epochs": len(hist),
        "last_epoch": last,
        "ckpt": str(ckpt) if ckpt.is_file() else None,
        "pilot_sha": st.get("expert_pilot_importance_index_sha256"),
        "status": st.get("status"),
    }


def _rewrite_unit_with_pilot() -> Path:
    unit = Path.home() / ".config/systemd/user" / BOOTSTRAP_UNIT
    text = unit.read_text(encoding="utf-8")
    # ensure epochs 40
    text = text.replace("--epochs 300", f"--epochs {TARGET_EPOCHS}")
    text = text.replace(
        "--owner-slop-box-ce-intensify-epochs 300",
        f"--owner-slop-box-ce-intensify-epochs {TARGET_EPOCHS}",
    )
    # strip existing pilot lines then append
    lines = [ln for ln in text.splitlines() if "pilot-importance-index" not in ln]
    # Insert pilot flag before Restart= or end of ExecStart continuation
    out: list[str] = []
    inserted = False
    for i, ln in enumerate(lines):
        out.append(ln)
        if (not inserted) and ln.strip().startswith("--rtp-parallel"):
            out.append(
                f"  --pilot-importance-index {PILOT} \\"
            )
            inserted = True
    if not inserted:
        # fallback: before Restart=
        rebuilt: list[str] = []
        for ln in out:
            if (not inserted) and ln.startswith("Restart="):
                rebuilt.append(
                    f"  --pilot-importance-index {PILOT}"
                )
                inserted = True
            rebuilt.append(ln)
        out = rebuilt
    body = "\n".join(out) + "\n"
    if f"--epochs {TARGET_EPOCHS}" not in body:
        raise RuntimeError("failed to pin epochs 40 in unit")
    if "pilot-importance-index" not in body:
        raise RuntimeError("failed to bind pilot in unit")
    unit.write_text(body, encoding="utf-8")
    staged = unit.with_suffix(unit.suffix + ".pilot-resume-r174")
    staged.write_text(body, encoding="utf-8")
    return unit


def main() -> int:
    sync = {
        "schema": "poke_bot.slop_box_canonical_sync_ack_r174/v1",
        "status": "ACK_OWNED",
        "created_at_utc": _now(),
        "agent": "7bc394c0",
        "owner": "canonical_training_pipeline",
        "apology": (
            "Bootstrap started because markup+schema8/combo packs+wipe were greenlit "
            "while Cox/Chao pilot upweight was treated as optional/omitted — wrong "
            "vs owner wanting Cox/Chao focus."
        ),
        "plan": {
            "keep_ce_healthy_until_boundary": True,
            "pilot_resume_from_latest_ckpt": True,
            "stop_total_epoch": TARGET_EPOCHS,
            "then": ["kaggle_55188658", "rl_self_play_after_preflight"],
            "orchestrator": "scripts/watch_slop_box_epoch40_submit_rl_r174.py",
            "no_dash_edits": True,
            "no_crustle_deletes": True,
        },
    }
    _atomic(SYNC, sync)

    ok, pilot_info = _pilot_valid()
    ep0 = _epoch_state()
    unit0 = _show(BOOTSTRAP_UNIT)
    _progress(
        phase="armed_wait_boundary",
        pilot=pilot_info,
        epoch=ep0,
        systemd=unit0,
        sync=str(SYNC),
    )
    if not ok:
        _progress(phase="waiting_pilot_index_valid", pilot=pilot_info)
        while True:
            ok, pilot_info = _pilot_valid()
            _progress(phase="waiting_pilot_index_valid", pilot=pilot_info, epoch=_epoch_state())
            if ok:
                break
            time.sleep(POLL_SEC)

    # Wait for next completed epoch beyond arm snapshot (clean boundary).
    arm_last = int(ep0.get("last_epoch") or 0)
    target_boundary = arm_last  # if arm_last ckpt exists and we're mid next, wait for arm_last+1
    if (CKPT_DIR / f"epoch_{arm_last:02d}.pt").is_file():
        target_boundary = arm_last + 1
    _progress(
        phase="waiting_clean_epoch_boundary",
        arm_last=arm_last,
        target_boundary=target_boundary,
        epoch=_epoch_state(),
    )
    while True:
        ep = _epoch_state()
        last = int(ep.get("last_epoch") or 0)
        ckpt = CKPT_DIR / f"epoch_{target_boundary:02d}.pt"
        unit = _show(BOOTSTRAP_UNIT)
        _progress(
            phase="waiting_clean_epoch_boundary",
            epoch=ep,
            target_boundary=target_boundary,
            boundary_ckpt=str(ckpt) if ckpt.is_file() else None,
            systemd=unit,
        )
        if last >= target_boundary and ckpt.is_file():
            break
        mp = int(unit.get("MainPID") or 0)
        if mp == 0 and last < TARGET_EPOCHS:
            _progress(phase="ce_died_before_pilot_resume", epoch=ep, systemd=unit)
            return 2
        if last >= TARGET_EPOCHS:
            _progress(phase="already_at_or_past_40_skip_pilot_resume", epoch=ep)
            return 0
        time.sleep(POLL_SEC)

    # Clean stop — no SIGKILL.
    _progress(phase="stopping_for_pilot_resume", epoch=_epoch_state())
    subprocess.run(["systemctl", "--user", "stop", BOOTSTRAP_UNIT], check=False)
    for _ in range(90):
        props = _show(BOOTSTRAP_UNIT)
        if int(props.get("MainPID") or 0) == 0:
            break
        time.sleep(2)
    props = _show(BOOTSTRAP_UNIT)
    _progress(phase="stopped", systemd=props, epoch=_epoch_state())

    unit_path = _rewrite_unit_with_pilot()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "reset-failed", BOOTSTRAP_UNIT], check=False)
    subprocess.run(["systemctl", "--user", "start", BOOTSTRAP_UNIT], check=False)
    time.sleep(4)
    after = _show(BOOTSTRAP_UNIT)
    receipt = {
        "schema": "poke_bot.slop_box_pilot_resume_started_r174/v1",
        "status": "resumed_with_pilot",
        "created_at_utc": _now(),
        "pilot": pilot_info,
        "boundary_epoch": target_boundary,
        "resume_run_dir": str(RUN),
        "epochs_max": TARGET_EPOCHS,
        "unit": str(unit_path),
        "systemd": after,
        "MainPID": int(after.get("MainPID") or 0),
        "note": "Resumed same run-dir; completed epochs skipped; pilot bound for remaining CE through 40.",
    }
    _atomic(STATE / "slop-box-pilot-resume-started-r174.json", receipt)
    _progress(phase="resumed_with_pilot", receipt=receipt)
    print("PILOT_RESUME_STARTED", json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0 if int(after.get("MainPID") or 0) > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
