#!/usr/bin/env python3
"""Bind dashboard to live Slop Box fresh bootstrap (r171) + tqdm log.

Dashboard-only. Never restarts bootstrap/training units.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PATH = Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py")
RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "slop-box-dashboard-fresh-bootstrap-bind-r171.json"
)
FRESH_RUN = (
    'ROOT / "outputs/bootstrap/final_format_slop_box_h10_rtp_fresh_r171"'
)
FRESH_LOG = (
    'ROOT / "outputs/final_format_slop_box_h10_rtp/logs/'
    'bootstrap-fresh-r171.log"'
)
EARLY_HEALTH = (
    'ROOT / "outputs/state/slop-box-fresh-bootstrap-early-health-r171.json"'
)
STARTED = (
    'ROOT / "outputs/state/slop-box-fresh-bootstrap-started-r171.json"'
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = PATH.with_name(PATH.name + f".pre-fresh-dash-r171-{stamp}")
    shutil.copy2(PATH, bak)

    # 1) Point RUN_DIR at fresh_r171 and add log/health constants.
    old_run = (
        'FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR = (\n'
        '    ROOT / "outputs/bootstrap/final_format_slop_box_h10_rtp"\n'
        ')'
    )
    new_run = (
        'FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR = (\n'
        f'    {FRESH_RUN}\n'
        ')\n'
        'FINAL_FORMAT_SLOP_BOX_H10_FRESH_LOG = (\n'
        f'    {FRESH_LOG}\n'
        ')\n'
        'FINAL_FORMAT_SLOP_BOX_H10_EARLY_HEALTH = (\n'
        f'    {EARLY_HEALTH}\n'
        ')\n'
        'FINAL_FORMAT_SLOP_BOX_H10_FRESH_STARTED = (\n'
        f'    {STARTED}\n'
        ')'
    )
    if "FINAL_FORMAT_SLOP_BOX_H10_FRESH_LOG" not in text:
        if old_run not in text:
            # already rebound to fresh path?
            old_run2 = (
                'FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR = (\n'
                '    ROOT / "outputs/bootstrap/'
                'final_format_slop_box_h10_rtp_fresh_r171"\n'
                ')'
            )
            if old_run2 in text:
                text = text.replace(old_run2, new_run, 1)
            else:
                raise SystemExit("RUN_DIR anchor missing")
        else:
            text = text.replace(old_run, new_run, 1)

    # 2) Inject fresh-bootstrap preference + tqdm enrichment near top of
    #    final_format_slop_box_progress after chao_hard short-circuit.
    marker = (
        '    chao_hard = final_format_slop_box_chao_hard_progress()\n'
        '    if chao_hard is not None:\n'
        '        return chao_hard\n'
    )
    inject = '''    chao_hard = final_format_slop_box_chao_hard_progress()
    if chao_hard is not None:
        # Fresh r171 bootstrap owns the shared unit when its run dir / health
        # receipt is live — do not let stale Chao-hard CE hide tqdm.
        _fresh_state = read_json(
            FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json"
        ) or {}
        _fresh_health = read_json(FINAL_FORMAT_SLOP_BOX_H10_EARLY_HEALTH) or {}
        _fresh_svc = unit_state(
            FINAL_FORMAT_SLOP_BOX_H10_RTP_BOOTSTRAP_SERVICE, user=True
        )
        _fresh_pid = int((_fresh_svc or {}).get("pid") or 0)
        _health_pid = int(_fresh_health.get("MainPID") or 0)
        _fresh_live = bool(
            _fresh_pid > 0
            and (
                bool(_fresh_state.get("history"))
                or (
                    _health_pid > 0
                    and _health_pid == _fresh_pid
                )
                or FINAL_FORMAT_SLOP_BOX_H10_FRESH_STARTED.is_file()
            )
        )
        if not _fresh_live:
            return chao_hard

'''
    if "FINAL_FORMAT_SLOP_BOX_H10_EARLY_HEALTH" in text and marker in text:
        if "_fresh_live = bool(" not in text.split(
            "def final_format_slop_box_progress", 1
        )[-1][:2500]:
            text = text.replace(marker, inject, 1)

    # 3) Disable stale continue60 false-positive when fresh bootstrap is live.
    # Replace continue60_live assignment block with fresh-aware gate.
    old_c60 = '''    continue60_live = bool(
        unit_running
        and (
            c60_active_status
            or (
                c60_from > 0
                and epochs_completed >= max(c60_from - 1, 0)
                and epochs_completed < max(c60_to, epochs_max)
            )
            or (epochs_max > 40 and epochs_completed < epochs_max)
        )
    )'''
    new_c60 = '''    fresh_health = read_json(FINAL_FORMAT_SLOP_BOX_H10_EARLY_HEALTH) or {}
    fresh_started = read_json(FINAL_FORMAT_SLOP_BOX_H10_FRESH_STARTED) or {}
    fresh_run_live = bool(
        unit_running
        and (
            bool(state.get("history"))
            or int(fresh_health.get("MainPID") or 0) == svc_pid
            or bool(fresh_started)
        )
        and str(
            (state.get("current_deck_guide") or {})
            .get("owner_epoch_schedule", {})
            .get("schema")
            or ""
        ).startswith("poke_bot.slop_box_ce_intensify_epochs")
        or (
            unit_running
            and FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR.name.endswith("fresh_r171")
            and (
                bool(state.get("history"))
                or int(fresh_health.get("MainPID") or 0) == svc_pid
            )
        )
    )
    # Stale continue60 receipts must not override the live fresh bootstrap.
    continue60_live = bool(
        unit_running
        and (not fresh_run_live)
        and (
            c60_active_status
            or (
                c60_from > 0
                and epochs_completed >= max(c60_from - 1, 0)
                and epochs_completed < max(c60_to, epochs_max)
            )
            or (epochs_max > 40 and epochs_completed < epochs_max)
        )
    )'''
    if old_c60 not in text:
        raise SystemExit("continue60_live anchor missing")
    if "fresh_run_live = bool(" not in text:
        text = text.replace(old_c60, new_c60, 1)

    # 4) Before return payload, enrich from fresh state tip + tqdm log.
    # Find the percent = ( block inside final_format_slop_box_progress and
    # insert enrichment just before `return {`.
    fn_start = text.find("def final_format_slop_box_progress")
    fn_ret = text.find("\n    return {\n        \"available\": True,", fn_start)
    if fn_ret < 0:
        raise SystemExit("return payload anchor missing")
    enrich = '''
    # --- fresh r171 live tqdm / epoch tip bind ---
    tip = {}
    history = state.get("history") or []
    if history and isinstance(history[-1], dict):
        tip = history[-1]
        try:
            epochs_completed = max(epochs_completed, int(tip.get("epoch") or 0))
        except (TypeError, ValueError):
            pass
        try:
            epochs_max = max(epochs_max, int(state.get("epochs_max") or epochs_max))
        except (TypeError, ValueError):
            pass
    val_loss = None
    if tip.get("validation_loss") is not None:
        try:
            val_loss = float(tip["validation_loss"])
        except (TypeError, ValueError):
            val_loss = None
    if tip.get("validation_accuracy") is not None:
        try:
            val_acc = float(tip["validation_accuracy"])
        except (TypeError, ValueError):
            pass
    health = read_json(FINAL_FORMAT_SLOP_BOX_H10_EARLY_HEALTH) or {}
    hp = dict(health.get("progress") or {})
    if hp:
        try:
            epochs_completed = max(
                epochs_completed, int(hp.get("outer_epoch") or 0)
            )
        except (TypeError, ValueError):
            pass
        try:
            epochs_max = max(epochs_max, int(hp.get("epochs_max") or 0) or epochs_max)
        except (TypeError, ValueError):
            pass
        if val_loss is None and hp.get("val_loss") is not None:
            try:
                val_loss = float(hp["val_loss"])
            except (TypeError, ValueError):
                pass
        if val_acc is None and hp.get("policy_acc") is not None:
            try:
                val_acc = float(hp["policy_acc"])
            except (TypeError, ValueError):
                pass
    tqdm_current = None
    tqdm_total = None
    tqdm_rate = None
    tqdm_rate_unit = None
    tqdm_eta = None
    tqdm_line = None
    tqdm_loss = None
    tqdm_acc = None
    fresh_log = FINAL_FORMAT_SLOP_BOX_H10_FRESH_LOG
    if fresh_log.is_file():
        raw_log = read_tail(fresh_log, 2_000_000)
        clean_log = ANSI_RE.sub("", raw_log).replace("\\r", "\\n")
        # Prefer newest tqdm frame from this fresh bootstrap log.
        frame_re = re.compile(
            r"^(?P<label>.+?):\\s+(?P<pct>\\d+)%\\|.*\\|\\s*"
            r"(?P<cur>\\d+)/(?P<tot>\\d+)\\s+"
            r"\\[(?P<timing>[^\\]]+)\\]"
            r"(?:,\\s*(?P<meta>.*))?$"
        )
        for raw_line in reversed(clean_log.splitlines()):
            line = raw_line.strip()
            if not line or "%|" not in line:
                continue
            m = frame_re.match(line)
            if not m:
                continue
            tqdm_line = line
            try:
                tqdm_current = int(m.group("cur"))
                tqdm_total = int(m.group("tot"))
            except (TypeError, ValueError):
                pass
            timing = m.group("timing") or ""
            rate_m = re.search(
                r"([0-9.]+)\\s*(batch/s|it/s|s/batch|s/it)", timing
            )
            if rate_m:
                try:
                    tqdm_rate = float(rate_m.group(1))
                    tqdm_rate_unit = rate_m.group(2)
                except (TypeError, ValueError):
                    pass
            eta_m = re.search(r"<\\s*([^,\\]]+)", timing)
            if eta_m:
                tqdm_eta = eta_m.group(1).strip()
            meta = m.group("meta") or ""
            loss_m = re.search(r"\\bloss=([0-9.]+)", meta)
            if loss_m:
                try:
                    tqdm_loss = float(loss_m.group(1))
                except (TypeError, ValueError):
                    pass
            acc_m = re.search(r"\\bacc=([0-9.]+)%?", meta)
            if acc_m:
                try:
                    tqdm_acc = float(acc_m.group(1))
                    if tqdm_acc > 1.5:
                        tqdm_acc = tqdm_acc / 100.0
                except (TypeError, ValueError):
                    pass
            break
        if fresh_log.is_file():
            updated_candidates.append(fresh_log.stat().st_mtime)
            updated = max(updated_candidates) if updated_candidates else updated
    if fresh_run_live or (unit_running and (state.get("history") or health)):
        # Prefer live fresh-bootstrap wording over stale CE intensify 0/80.
        loss_txt = (
            f" · val_loss={val_loss:.3f}" if val_loss is not None else ""
        )
        acc_txt = (
            f" · val_acc={val_acc:.3f}" if val_acc is not None else ""
        )
        bar_txt = ""
        if tqdm_current is not None and tqdm_total:
            bar_txt = f" · batch {tqdm_current}/{tqdm_total}"
            if tqdm_rate is not None and tqdm_rate_unit:
                bar_txt += f" · {tqdm_rate:.2f}{tqdm_rate_unit}"
            if tqdm_loss is not None:
                bar_txt += f" · loss={tqdm_loss:.3f}"
        phase = "bootstrap:expert_ce"
        status = "running"
        latest_line = (
            f"Slop Box fresh bootstrap LIVE · epoch "
            f"{epochs_completed}/{epochs_max}{loss_txt}{acc_txt}{bar_txt}"
        )
        continue_rehearsal = True  # keep CE progress surface for owner tqdm
        percent = (
            (float(epochs_completed) / float(epochs_max)) * 100.0
            if epochs_max
            else None
        )
'''
    # Avoid double-inject
    if "fresh r171 live tqdm" not in text:
        text = text[:fn_ret] + enrich + text[fn_ret:]

    # 5) Patch return dict fields for log/mode/tqdm counters.
    # After enrichment vars exist, rewrite key return fields via targeted replace
    # inside the function return only once.
    old_ret_head = '''    return {
        "available": True,
        "authoritative": True,
        "source": str(
            FINAL_FORMAT_SLOP_BOX_H10_READY
            if ready
            else FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json"
        ),
        "log": str(FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json"),
        "latest_line": latest_line,
        "raw_latest_line": latest_line,
        "updated_at": updated,
        "fresh": bool(active or live_healthy or str(phase).startswith("stopped:rehearsal:gate_remeasure")),
        "status": status,
        "mode": (
            "final_format_slop_box_h10_rtp_ce_intensify"
            if continue_rehearsal
            else "final_format_slop_box_h10_rtp_bootstrap"
        ),
        "phase": phase,
        "run": (
            "final_format_slop_box_h10_rtp_ce_intensify"
            if continue_rehearsal
            else "final_format_slop_box_h10_rtp_bootstrap"
        ),
        "specialist_id": specialist_id,
        "display_name": specialist.get("display_name") or "Slop Box",
        "epoch": epochs_completed,
        "epochs_target": epochs_max,
        "iteration": epochs_completed,
        "iterations_target": epochs_max,
        "current": epochs_completed,
        "total": epochs_max,
        "percent": percent,
        "rate": None,
        "rate_unit": None,
        "games_per_second": 0.0,
        "samples_per_second": 0.0,
        "eta": None,
'''
    new_ret_head = '''    return {
        "available": True,
        "authoritative": True,
        "source": str(FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json"),
        "log": str(
            FINAL_FORMAT_SLOP_BOX_H10_FRESH_LOG
            if FINAL_FORMAT_SLOP_BOX_H10_FRESH_LOG.is_file()
            else FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json"
        ),
        "latest_line": latest_line,
        "raw_latest_line": tqdm_line or latest_line,
        "updated_at": updated,
        "fresh": bool(
            active
            or live_healthy
            or fresh_run_live
            or str(phase).startswith("stopped:rehearsal:gate_remeasure")
        ),
        "status": status,
        "mode": (
            "final_format_slop_box_h10_rtp_fresh_bootstrap_r171"
            if fresh_run_live or str(phase).startswith("bootstrap:expert")
            else (
                "final_format_slop_box_h10_rtp_ce_intensify"
                if continue_rehearsal
                else "final_format_slop_box_h10_rtp_bootstrap"
            )
        ),
        "phase": phase,
        "run": FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR.name,
        "specialist_id": specialist_id,
        "display_name": specialist.get("display_name") or "Slop Box",
        "epoch": epochs_completed,
        "epochs_target": epochs_max,
        "iteration": epochs_completed,
        "iterations_target": epochs_max,
        "current": tqdm_current if tqdm_current is not None else epochs_completed,
        "total": tqdm_total if tqdm_total is not None else epochs_max,
        "percent": (
            (float(tqdm_current) / float(tqdm_total)) * 100.0
            if tqdm_current is not None and tqdm_total
            else percent
        ),
        "rate": tqdm_rate,
        "rate_unit": tqdm_rate_unit,
        "games_per_second": 0.0,
        "samples_per_second": (
            float(tqdm_rate)
            if tqdm_rate is not None and tqdm_rate_unit == "batch/s"
            else 0.0
        ),
        "eta": tqdm_eta,
        "val_loss": val_loss,
        "loss": tqdm_loss if tqdm_loss is not None else val_loss,
        "val_acc": val_acc,
'''
    if old_ret_head not in text:
        raise SystemExit("return head anchor missing")
    text = text.replace(old_ret_head, new_ret_head, 1)

    # metrics block: add val_loss / tqdm
    old_metrics = '''        "metrics": {
            "best_metric": ready.get("best_metric") or state.get("best_metric"),
            "expert_complete": expert_complete,
            "rtp_cut_ready": rtp_cut_ready,
            "rtp_live_checkpoint": str(FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE)
            if rtp_live
            else None,
            "cox_chao_gate_failed": bool(gate_failed and not gate_passed),
            "cox_chao_gate_passed": bool(gate_passed),
            "cox_chao_held_policy_acc": gate_measured_f,
            "cox_chao_gate_threshold": 0.9,
            "ce_intensify_active": bool(continue_rehearsal),
            "continue_intent": bool(continue_intent),
            "ready_status": ready.get("status") or None,
        },'''
    new_metrics = '''        "metrics": {
            "best_metric": ready.get("best_metric") or state.get("best_metric"),
            "expert_complete": expert_complete,
            "rtp_cut_ready": rtp_cut_ready,
            "rtp_live_checkpoint": str(FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE)
            if rtp_live
            else None,
            "cox_chao_gate_failed": bool(gate_failed and not gate_passed),
            "cox_chao_gate_passed": bool(gate_passed),
            "cox_chao_held_policy_acc": gate_measured_f,
            "cox_chao_gate_threshold": 0.9,
            "ce_intensify_active": bool(continue_rehearsal),
            "continue_intent": bool(continue_intent),
            "ready_status": ready.get("status") or None,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "tqdm_loss": tqdm_loss,
            "tqdm_acc": tqdm_acc,
            "tqdm_line": tqdm_line,
            "fresh_bootstrap_r171": True,
            "outer_epoch": epochs_completed,
            "epochs_max": epochs_max,
        },'''
    if old_metrics not in text:
        raise SystemExit("metrics anchor missing")
    text = text.replace(old_metrics, new_metrics, 1)

    # Ensure `re` is imported at module level (it should be).
    if not re.search(r"^import re\b", text, flags=re.M):
        text = "import re\n" + text

    PATH.write_text(text, encoding="utf-8")

    # Smoke: import and call progress
    sys.path.insert(0, "/home/inzi/poke-bot-agent")
    # compile check
    compile(text, str(PATH), "exec")

    receipt = {
        "schema": "poke_bot.slop_box_dashboard_fresh_bootstrap_bind_r171/v1",
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "backup": str(bak),
        "backup_sha256": _sha(bak),
        "patched_sha256": _sha(PATH),
        "run_dir": "/home/inzi/poke-bot-agent/outputs/bootstrap/"
        "final_format_slop_box_h10_rtp_fresh_r171",
        "log": "/home/inzi/poke-bot-agent/outputs/final_format_slop_box_h10_rtp/"
        "logs/bootstrap-fresh-r171.log",
        "training_touched": False,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
