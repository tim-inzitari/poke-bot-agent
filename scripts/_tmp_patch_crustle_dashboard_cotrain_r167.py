#!/usr/bin/env python3
"""Attach concurrent RTP cotrain sidecar telemetry to Crustle RL dash projection."""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

PATHS = [
    Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py"),
    Path(
        "/home/inzi/poke-bot-agent-deployments/"
        "final-format-marnie-postupload-r136/scripts/dashboard_snapshot.py"
    ),
]
RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "crustle-dashboard-cotrain-sidecar-projection-r167.json"
)

HELPER = '''
def crustle_rtp_cotrain_sidecar() -> dict[str, Any]:
    """Attach concurrent RTP co-train sidecar facts without stealing RL ownership."""

    receipt = read_json(FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_RECEIPT) or {}
    if not receipt:
        return {}
    service = unit_state(FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_SERVICE, user=True)
    live = read_json(FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_LIVE_RECEIPT) or {}
    metrics = dict(live.get("metrics") or {})
    active = bool(
        service.get("active")
        and (
            int(service.get("pid") or 0) > 0
            or service.get("sub_state") in {"running", "start"}
        )
    )
    return {
        "available": True,
        "active": active,
        "service": service.get("name"),
        "pid": service.get("pid"),
        "mode": receipt.get("mode"),
        "live_checkpoint": receipt.get("live_checkpoint"),
        "rtp_device": receipt.get("rtp_device"),
        "epoch": metrics.get("epoch"),
        "mean_loss": metrics.get("mean_loss"),
        "n_steps": metrics.get("n_steps"),
        "serving_eligible": live.get("serving_eligible"),
        "log": str(FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_LOG),
    }

'''


def patch_text(text: str) -> str:
    if "FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_SERVICE" not in text:
        if 'FINAL_FORMAT_CRUSTLE_RTP_SERVICE = "pokebot-crustle-rtp-r167.service"\n' in text:
            text = text.replace(
                'FINAL_FORMAT_CRUSTLE_RTP_SERVICE = "pokebot-crustle-rtp-r167.service"\n',
                'FINAL_FORMAT_CRUSTLE_RTP_SERVICE = "pokebot-crustle-rtp-r167.service"\n'
                'FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_SERVICE = '
                '"pokebot-crustle-rtp-cotrain-r167.service"\n',
                1,
            )
        else:
            old = (
                "FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE = (\n"
                '    "pokebot-final-format-crustle-r113-h10-rl.service"\n'
                ")\n"
            )
            if old not in text:
                raise SystemExit("RL service const missing")
            text = text.replace(
                old,
                old
                + 'FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_SERVICE = '
                '"pokebot-crustle-rtp-cotrain-r167.service"\n',
                1,
            )

    if "FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_RECEIPT" not in text:
        paths_block = (
            "FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_RECEIPT = (\n"
            '    ROOT / "outputs/state/crustle-rtp-rl-cotrain-r167.json"\n'
            ")\n"
            "FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_LIVE_RECEIPT = (\n"
            '    ROOT / "outputs/rtp_fleet/crustle.live/'
            'rtp_shadow_planner.pt.receipt.json"\n'
            ")\n"
            "FINAL_FORMAT_CRUSTLE_RTP_COTRAIN_LOG = (\n"
            '    ROOT / "outputs/rtp_fleet/crustle_cotrain_r167.log"\n'
            ")\n"
        )
        if "FINAL_FORMAT_CRUSTLE_RTP_SUMMARY" in text:
            anchor = (
                "FINAL_FORMAT_CRUSTLE_RTP_SUMMARY = (\n"
                '    ROOT / "outputs/rtp_fleet/crustle/pipeline_summary.json"\n'
                ")\n"
            )
            if anchor not in text:
                raise SystemExit("RTP summary anchor missing")
            text = text.replace(anchor, anchor + paths_block, 1)
        else:
            ready = (
                "FINAL_FORMAT_CRUSTLE_H10_READY = (\n"
                '    ROOT / "outputs/state/'
                'final-format-crustle-r113-h10-bootstrap-ready.json"\n'
                ")\n"
            )
            if ready not in text:
                raise SystemExit("ready anchor missing")
            text = text.replace(ready, ready + paths_block, 1)

    if "def crustle_rtp_cotrain_sidecar(" not in text:
        for marker in (
            "def final_format_crustle_rtp_progress(",
            "def final_format_crustle_progress(",
        ):
            if marker in text:
                text = text.replace(marker, HELPER + marker, 1)
                break
        else:
            raise SystemExit("function marker missing")

    if 'progress_metrics["rtp_cotrain"]' not in text:
        needle = (
            '        progress_metrics = dict(progress.get("metrics") or {})\n'
            '        progress_metrics.update(drain_projection.get("metrics") or {})\n'
        )
        if needle not in text:
            raise SystemExit("metrics needle missing")
        insert = (
            needle
            + "        cotrain = crustle_rtp_cotrain_sidecar()\n"
            + "        latest_co = \"\"\n"
            + "        if cotrain:\n"
            + '            progress_metrics["rtp_cotrain"] = cotrain\n'
            + '            if cotrain.get("active"):\n'
            + "                latest_co = (\n"
            + '                    f" · rtp_cotrain pid={cotrain.get(\'pid\')} "\n'
            + '                    f"epoch={cotrain.get(\'epoch\')} "\n'
            + '                    f"loss={cotrain.get(\'mean_loss\')}"\n'
            + "                )\n"
            + "            else:\n"
            + '                latest_co = " · rtp_cotrain inactive"\n'
        )
        text = text.replace(needle, insert, 1)

        line_old = (
            "        latest_line = drain_projection.get(\n"
            '            "latest_line", progress.get("line")\n'
            "        )\n"
        )
        line_new = (
            line_old
            + "        if latest_co:\n"
            + '            latest_line = f"{latest_line}{latest_co}"\n'
        )
        if line_old in text:
            text = text.replace(line_old, line_new, 1)
        else:
            compact = (
                "        latest_line = drain_projection.get("
                '"latest_line", progress.get("line"))\n'
            )
            if compact not in text:
                raise SystemExit("latest_line anchor missing")
            text = text.replace(
                compact,
                compact
                + "        if latest_co:\n"
                + '            latest_line = f"{latest_line}{latest_co}"\n',
                1,
            )
    return text


def main() -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digests: dict[str, str] = {}
    for path in PATHS:
        if not path.is_file():
            print(f"skip_missing {path}", file=sys.stderr)
            continue
        bak = path.with_name(path.name + f".pre-crustle-cotrain-dash-{stamp}")
        shutil.copy2(path, bak)
        path.write_text(patch_text(path.read_text(encoding="utf-8")), encoding="utf-8")
        digests[str(path)] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    os.chdir("/home/inzi/poke-bot-agent")
    spec = importlib.util.spec_from_file_location("ds", PATHS[0])
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prog = mod.final_format_crustle_progress()
    cotrain = (prog.get("metrics") or {}).get("rtp_cotrain") or {}
    receipt = {
        "schema": "poke_bot.crustle_dashboard_cotrain_sidecar_projection/v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "paths": digests,
        "projection": {
            "status": prog.get("status"),
            "mode": prog.get("mode"),
            "phase": prog.get("phase"),
            "pid": (prog.get("service") or {}).get("pid"),
            "latest_line": (prog.get("latest_line") or "")[:240],
            "rtp_cotrain": cotrain,
        },
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if prog.get("mode") != "final_format_crustle_h10_rl":
        raise SystemExit(f"unexpected mode {prog.get('mode')}")
    if not cotrain.get("active"):
        raise SystemExit("expected active rtp_cotrain sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
