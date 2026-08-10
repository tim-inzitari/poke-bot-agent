#!/usr/bin/env python3
"""Select the checkpoint with best Cox/Chao held policy_acc (r170).

Non-destructive: evaluates completed epoch_*.pt files, writes a checksum-bound
selection receipt, and never marks ready/RL. Prefer running on a free GPU or
after the live deep-CE oneshot releases the primary device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.expert_pilot_importance import file_digest  # noqa: E402

GATE_SCRIPT = ROOT / "scripts" / "evaluate_slop_box_cox_chao_held_policy_acc_gate_r170.py"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _epoch_key(path: Path) -> int:
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return -1
    return -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/bootstrap/"
            "final_format_slop_box_h10_rtp/checkpoints"
        ),
    )
    parser.add_argument(
        "--epochs",
        default="24,39,40,41,60,80",
        help="Comma-separated epoch numbers, or 'all' for every epoch_*.pt",
    )
    parser.add_argument(
        "--expert-pointer",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/data/bootstrap/"
            "expert-latest20-2026-07-04-2026-07-23-roster18-v6-strategic/"
            "teal-mask-ogerpon-ex/PROTECTED_EXPERT_CORPUS.json"
        ),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-targets-r170.json"
        ),
    )
    parser.add_argument(
        "--pilot-map",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-pilot-map-r170.json"
        ),
    )
    parser.add_argument(
        "--held-split",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-split-pilot-map-r170.json"
        ),
    )
    parser.add_argument(
        "--cpu-pack-root",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
            "final_format_slop_box_h10_rtp"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-chao-held-checkpoint-select-r170.json"
        ),
    )
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-chao-held-select-receipts-r170"
        ),
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir.expanduser().resolve()
    if not ckpt_dir.is_dir():
        raise SystemExit(f"missing checkpoint dir: {ckpt_dir}")

    if str(args.epochs).strip().lower() == "all":
        paths = sorted(ckpt_dir.glob("epoch_*.pt"), key=_epoch_key)
    else:
        wanted = {int(x.strip()) for x in str(args.epochs).split(",") if x.strip()}
        paths = [
            ckpt_dir / f"epoch_{epoch}.pt"
            for epoch in sorted(wanted)
            if (ckpt_dir / f"epoch_{epoch}.pt").is_file()
        ]
    if not paths:
        raise SystemExit("no checkpoints selected")

    receipt_dir = args.receipt_dir.expanduser().resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in paths:
        out = receipt_dir / f"gate-{path.stem}.json"
        cmd = [
            sys.executable,
            str(GATE_SCRIPT),
            "--checkpoint",
            str(path),
            "--expert-pointer",
            str(args.expert_pointer.expanduser().resolve()),
            "--targets",
            str(args.targets.expanduser().resolve()),
            "--pilot-map",
            str(args.pilot_map.expanduser().resolve()),
            "--held-split",
            str(args.held_split.expanduser().resolve()),
            "--cpu-pack-root",
            str(args.cpu_pack_root.expanduser().resolve()),
            "--output",
            str(out),
            "--no-quarantine-ready-on-fail",
            "--ready",
            str(receipt_dir / "ready-unused-by-selector.json"),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if not out.is_file():
            rows.append(
                {
                    "checkpoint": str(path),
                    "epoch": _epoch_key(path),
                    "status": "eval_failed",
                    "returncode": int(proc.returncode),
                    "stderr_tail": (proc.stderr or "")[-500:],
                }
            )
            continue
        gate = json.loads(out.read_text(encoding="utf-8"))
        rows.append(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": gate.get("checkpoint_sha256"),
                "epoch": _epoch_key(path),
                "cox_chao_held_policy_acc": float(gate["cox_chao_held_policy_acc"]),
                "games": int(gate["games"]),
                "decisions": int(gate["decisions"]),
                "gate_receipt": str(out),
                "gate_receipt_sha256": file_digest(out),
                "status": gate.get("status"),
            }
        )

    scored = [
        row
        for row in rows
        if isinstance(row.get("cox_chao_held_policy_acc"), float)
    ]
    scored.sort(
        key=lambda row: (
            float(row["cox_chao_held_policy_acc"]),
            -int(row.get("epoch") or 0),
        ),
        reverse=True,
    )
    best = scored[0] if scored else None
    payload = {
        "schema": "poke_bot.slop_box_chao_held_checkpoint_select_r170/v1",
        "goal_revision": 170,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(ckpt_dir),
        "select_metric": "cox_chao_held_policy_acc",
        "candidates": rows,
        "best": best,
        "hot_start_recommendation": (
            None
            if best is None
            else {
                "path": best["checkpoint"],
                "sha256": best.get("checkpoint_sha256"),
                "cox_chao_held_policy_acc": best["cox_chao_held_policy_acc"],
                "note": (
                    "Prefer this over deep-CE val_loss best_path for Chao-hard "
                    "continue; do not use late memorized epochs if Chao held is worse"
                ),
            }
        ),
        "no_ready": True,
        "no_rl": True,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
