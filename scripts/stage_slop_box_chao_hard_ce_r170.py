#!/usr/bin/env python3
"""Stage Chao-hard CE maps + systemd unit for next-boundary continue (r170).

Train corpus is the FULL available teal-mask / Slop Box acting-seat expert
set (every train row from the expanded protected corpus). Cox/Chao×N and
optional James Cox×M are importance / loss upweights on top of that mix —
never a Chao-only filter. Held exact \"James Cox & Henry Chao\" stays
0-overlap out of train for the 0.90 gate only.

Does not interrupt live deep CE. Writes a dedicated Chao-hard importance
index and optionally arms OnSuccess after the deep CE oneshot exits.
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

from poke_bot.expert_pilot_importance import (  # noqa: E402
    canonical_digest,
    file_digest,
)

COX_CHAO = "James Cox & Henry Chao"
JAMES_COX = "James Cox"
STATE = Path("/home/pokebot/poke-bot-agent/outputs/state")
DEFAULT_EXPERT = Path(
    "/home/pokebot/poke-bot-agent/data/bootstrap/"
    "expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex/"
    "PROTECTED_EXPERT_CORPUS.json"
)
UNIT_NAME = "pokebot-slop-box-chao-hard-ce-r170.service"
BOOTSTRAP_UNIT = "pokebot-final-format-slop-box-h10-rtp-bootstrap.service"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_chao_hard_upweight(
    *,
    targets: Path,
    pilot: Path,
    held: Path,
    out: Path,
    cox_chao_weight: float,
    james_cox_train_weight: float,
) -> dict[str, Any]:
    targets_obj = json.loads(targets.read_text(encoding="utf-8"))
    pilot_obj = json.loads(pilot.read_text(encoding="utf-8"))
    held_obj = json.loads(held.read_text(encoding="utf-8"))
    pilots = {
        (str(row["episode_id"]), int(row["seat"])): str(row["team_name"])
        for row in pilot_obj.get("rows") or ()
    }
    train_rows = list(targets_obj.get("train_rows") or ())
    if not train_rows:
        raise RuntimeError("expanded targets have zero train_rows")
    weights: list[float] = []
    weighted_rows: list[dict[str, Any]] = []
    matched_chao = 0
    matched_james = 0
    matched_other = 0
    for index, row in enumerate(train_rows):
        key = (str(row["episode_id"]), int(row["seat"]))
        team = pilots.get(key)
        if team == COX_CHAO:
            weight = float(cox_chao_weight)
            matched_chao += 1
        elif team == JAMES_COX and james_cox_train_weight > 1.0:
            weight = float(james_cox_train_weight)
            matched_james += 1
        else:
            weight = 1.0
            matched_other += 1
        weights.append(weight)
        weighted_rows.append(
            {
                "train_index": index,
                "episode_id": key[0],
                "seat": key[1],
                "team_name": team,
                "weight": weight,
            }
        )
    if int(pilot_obj.get("unverifiable_rows") or 0) != 0:
        raise RuntimeError("pilot map has unverifiable rows")
    # Fail closed: upweight must cover the full train mix, not a Chao subset.
    if len(weights) != len(train_rows):
        raise RuntimeError("train weight vector length mismatch")
    if matched_chao + matched_james + matched_other != len(train_rows):
        raise RuntimeError("train mix partition does not cover all train_rows")
    if matched_chao >= len(train_rows):
        raise RuntimeError(
            "refusing Chao-only train mix; full teal archetype train required"
        )
    if matched_other < 1:
        raise RuntimeError(
            "refusing train mix with zero non-Chao/non-James-Cox games"
        )
    if min(weights) < 1.0 - 1e-9:
        raise RuntimeError("non-positive or sub-1x weights are not allowed")
    tier_counts: dict[str, int] = {
        f"{float(cox_chao_weight):.0f}x_cox_chao": matched_chao,
        "1x": len(train_rows) - matched_chao - matched_james,
    }
    if matched_james:
        tier_counts[f"{float(james_cox_train_weight):.0f}x_james_cox_train_only"] = (
            matched_james
        )
    payload = {
        "schema": "poke_bot.expert_pilot_importance/v1",
        "owner_decision_revision": 138,
        "status": "ready",
        "goal_revision": 170,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "chao_hard_ce_r170",
        "corpus_manifest": targets_obj.get("corpus_manifest"),
        "corpus_manifest_sha256": targets_obj.get("corpus_manifest_sha256"),
        "split_seed": int(targets_obj.get("split_seed") or 20260722),
        "validation_fraction": float(targets_obj.get("validation_fraction") or 0.10),
        "max_context": 320,
        "support_partition": "training_only",
        "join_key": ["episode_id", "seat", "exact_team_name"],
        "cox_chao_team_name_exact": COX_CHAO,
        "gate_team_name_exact": COX_CHAO,
        "james_cox_train_only_team_name": JAMES_COX,
        "cox_chao_train_weight": float(cox_chao_weight),
        "james_cox_train_weight": float(james_cox_train_weight),
        "non_cox_chao_train_weight": 1.0,
        "full_archetype_train_preserved": True,
        "train_corpus_mode": "full_available_teal_acting_seat_upweight",
        "chao_only_filter": False,
        "actions_and_labels_unchanged": True,
        "validation_unweighted": True,
        "held_out_chao_split_for_gate": True,
        "kaggle_evaluation_replays_excluded": True,
        "targets_sha256": file_digest(targets),
        "pilot_map_sha256": file_digest(pilot),
        "leaderboard_snapshot_sha256": "sha256:" + ("0" * 64),
        "train_identity_sha256": targets_obj.get("train_identity_sha256"),
        "validation_identity_sha256": targets_obj.get("validation_identity_sha256"),
        "train_games": len(train_rows),
        "validation_games": len(targets_obj.get("validation_rows") or ()),
        "matched_top_100_train_games": matched_chao,
        "matched_james_cox_train_only_games": matched_james,
        "matched_other_full_archetype_train_games": matched_other,
        "unmatched_or_unverifiable_train_games": len(train_rows)
        - matched_chao
        - matched_james,
        "unmatched_or_unverifiable_weight": 1.0,
        "effective_training_weight_mass": float(sum(weights)),
        "tier_counts": tier_counts,
        "tiers": [
            {
                "team_name": COX_CHAO,
                "weight": float(cox_chao_weight),
                "gate_identity": True,
            },
            {
                "team_name": JAMES_COX,
                "weight": float(james_cox_train_weight),
                "gate_identity": False,
                "train_only": True,
            },
        ],
        "per_team_support": [
            {
                "team_name": COX_CHAO,
                "train_games": matched_chao,
                "weight": float(cox_chao_weight),
            },
            {
                "team_name": JAMES_COX,
                "train_games": matched_james,
                "weight": float(james_cox_train_weight),
            },
        ],
        "train_game_weights": weights,
        "weighted_train_rows": weighted_rows,
        "train_game_weights_sha256": canonical_digest(weights),
        "owner_note": (
            "Owner clarification: train on ALL available teal acting-seat "
            "expert games; Chao×N and James Cox×M are upweights only — never "
            "drop non-Chao games. Exact Chao held stays 0-overlap for the gate."
        ),
        "held_split_ref": {
            "path": str(held),
            "sha256": file_digest(held),
            "cox_chao_held_validation_games": held_obj.get(
                "cox_chao_held_validation_games"
            ),
            "cox_chao_train_games": held_obj.get("cox_chao_train_games"),
        },
    }
    if out.is_file():
        out.unlink()
    _atomic_json(out, payload)
    return payload


def unit_text(
    *,
    expert: Path,
    guide_ready: Path,
    upweight: Path,
    epochs: int,
) -> str:
    py = "/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python"
    run_dir = "/home/pokebot/poke-bot-agent/outputs/bootstrap/final_format_slop_box_chao_hard_r170"
    ready = "/home/pokebot/poke-bot-agent/outputs/state/final-format-slop-box-chao-hard-ce-ready-r170.json"
    deep_state = (
        "/home/pokebot/poke-bot-agent/outputs/bootstrap/"
        "final_format_slop_box_h10_rtp/state.json"
    )
    return f"""[Unit]
Description=Slop Box Chao-hard CE continue after deep CE (r170)
After=network-online.target {BOOTSTRAP_UNIT}
Wants=network-online.target
ConditionPathExists={expert}
ConditionPathExists={guide_ready}
ConditionPathExists={upweight}
ConditionPathExists={deep_state}
# Do not start while deep CE oneshot is still activating.
Conflicts={BOOTSTRAP_UNIT}

[Service]
Type=oneshot
WorkingDirectory=/home/pokebot/poke-bot-agent
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/pokebot/poke-bot-agent
Environment=ALLOW_TF32=1
Environment=NVIDIA_TF32_OVERRIDE=1
# Hot-start resolves at start time from deep CE fusion-valid checkpoint.
ExecStart={py} -u /home/pokebot/poke-bot-agent/scripts/run_slop_box_chao_hard_ce_r170.py \\
  --expert {expert} \\
  --guide-ready {guide_ready} \\
  --pilot-importance-index {upweight} \\
  --run-dir {run_dir} \\
  --ready {ready} \\
  --epochs {epochs}
Restart=no
TimeoutStartSec=infinity
MemoryHigh=96G
MemoryMax=112G
MemorySwapMax=0
StandardOutput=append:/home/pokebot/poke-bot-agent/outputs/final_format_slop_box_chao_hard_r170/logs/chao_hard.log
StandardError=append:/home/pokebot/poke-bot-agent/outputs/final_format_slop_box_chao_hard_r170/logs/chao_hard.log

[Install]
WantedBy=default.target
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-pointer", type=Path, default=DEFAULT_EXPERT)
    # Moderate hard-mass: ×50 without Chao-held selection just memorizes like ×10.
    parser.add_argument("--cox-chao-train-weight", type=float, default=25.0)
    parser.add_argument("--james-cox-train-weight", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--install-systemd",
        action="store_true",
        help=(
            "Write user unit + OnSuccess drop-in and daemon-reload (no start). "
            "Only use after expanded maps exist; do not arm while corpus expand "
            "or deep CE is still the active boundary."
        ),
    )
    parser.add_argument(
        "--skip-map-rebuild",
        action="store_true",
        help="Reuse existing expanded targets/pilot/held; only rebuild Chao-hard upweight",
    )
    args = parser.parse_args()

    expert = args.expert_pointer.expanduser().resolve()
    guide_ready = expert.parent / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    targets = STATE / "slop-box-cox-chao-held-targets-expanded-r170.json"
    pilot = STATE / "slop-box-cox-chao-held-pilot-map-expanded-r170.json"
    held = STATE / "slop-box-cox-chao-held-split-pilot-map-expanded-r170.json"
    upweight = STATE / "slop-box-chao-hard-train-upweight-importance-r170.json"

    if not args.skip_map_rebuild:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "rebuild_slop_box_cox_chao_maps_r170.py"),
                "--expert-pointer",
                str(expert),
                "--cox-chao-train-weight",
                "10.0",
                "--output-suffix=-expanded-r170",
                "--skip-live-upweight",
            ],
            check=True,
            cwd=str(ROOT),
        )

    for path in (targets, pilot, held, guide_ready, expert):
        if not path.is_file():
            raise SystemExit(f"missing required map/corpus artifact: {path}")

    payload = build_chao_hard_upweight(
        targets=targets,
        pilot=pilot,
        held=held,
        out=upweight,
        cox_chao_weight=float(args.cox_chao_train_weight),
        james_cox_train_weight=float(args.james_cox_train_weight),
    )

    # Stage both full-corpus upweight options for owner audit:
    # preferred = Chao×N + JamesCox×M + all-other×1
    # alt = Chao×N + JamesCox×1 + all-other×1  (still full mix; not Chao-filtered)
    alt = STATE / "slop-box-chao-hard-train-upweight-no-james-cox-boost-r170.json"
    build_chao_hard_upweight(
        targets=targets,
        pilot=pilot,
        held=held,
        out=alt,
        cox_chao_weight=float(args.cox_chao_train_weight),
        james_cox_train_weight=1.0,
    )
    # Keep legacy filename as a hardlink/copy alias so older prep receipts resolve.
    legacy_alt = STATE / "slop-box-chao-hard-train-upweight-cox-chao-only-r170.json"
    if legacy_alt.exists() or legacy_alt.is_symlink():
        legacy_alt.unlink()
    try:
        os.link(alt, legacy_alt)
    except OSError:
        legacy_alt.write_bytes(alt.read_bytes())

    unit_path = Path.home() / ".config/systemd/user" / UNIT_NAME
    dropin_dir = (
        Path.home()
        / ".config/systemd/user"
        / f"{BOOTSTRAP_UNIT}.d"
    )
    dropin = dropin_dir / "10-chao-hard-onsuccess.conf"
    summary = {
        "schema": "poke_bot.slop_box_chao_hard_ce_staged_r170/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expert_pointer": str(expert),
        "guide_ready": str(guide_ready),
        "targets": str(targets),
        "pilot": str(pilot),
        "held": str(held),
        "upweight_preferred_with_james_cox_train": str(upweight),
        "upweight_full_mix_no_james_cox_boost": str(alt),
        "upweight_cox_chao_only_alias_legacy": str(legacy_alt),
        "upweight_sha256": file_digest(upweight),
        "cox_chao_train_weight": float(args.cox_chao_train_weight),
        "james_cox_train_weight": float(args.james_cox_train_weight),
        "matched_cox_chao_train_games": payload["matched_top_100_train_games"],
        "matched_james_cox_train_only_games": payload[
            "matched_james_cox_train_only_games"
        ],
        "matched_other_full_archetype_train_games": payload[
            "matched_other_full_archetype_train_games"
        ],
        "held_cox_chao_games": payload["held_split_ref"][
            "cox_chao_held_validation_games"
        ],
        "effective_training_weight_mass": payload["effective_training_weight_mass"],
        "train_games": payload["train_games"],
        "validation_games": payload["validation_games"],
        "train_corpus_mode": "full_available_teal_acting_seat_upweight",
        "chao_only_filter": False,
        "hot_start": (
            "prefer_select_slop_box_chao_held_checkpoint_r170_then_fusion_valid"
        ),
        "select_metric": "cox_chao_held_policy_acc",
        "epochs": int(args.epochs),
        "unit": UNIT_NAME,
        "armed_via": (
            f"OnSuccess= of {BOOTSTRAP_UNIT}"
            if args.install_systemd
            else "not_armed_manual_or_later_onsuccess"
        ),
        "gate_team": COX_CHAO,
        "anti_overfit": {
            "full_archetype_train_preserved": True,
            "train_all_available_games": True,
            "chao_only_filter": False,
            "validation_unweighted": True,
            "held_out_chao_split_for_gate": True,
            "james_cox_train_only_upweight": True,
            "select_by_chao_held_not_train_acc": True,
            "avoid_naive_x10_without_selection": True,
        },
        "plan_receipt": str(
            ROOT / "state" / "slop-box-chao-held-generalization-plan-r170.json"
        ),
        "status": (
            "staged_systemd_armed_wait_deep_ce"
            if args.install_systemd
            else "staged_maps_ready_wait_boundary"
        ),
    }

    if args.install_systemd:
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        dropin_dir.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(
            unit_text(
                expert=expert,
                guide_ready=guide_ready,
                upweight=upweight,
                epochs=int(args.epochs),
            ),
            encoding="utf-8",
        )
        dropin.write_text(
            f"[Unit]\nOnSuccess={UNIT_NAME}\n",
            encoding="utf-8",
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        # Enable Chao-hard unit so OnSuccess can start it; do not start now.
        subprocess.run(
            ["systemctl", "--user", "enable", UNIT_NAME],
            check=True,
        )
        summary["unit_path"] = str(unit_path)
        summary["dropin_path"] = str(dropin)
        summary["systemd_armed"] = True

    _atomic_json(STATE / "slop-box-chao-hard-ce-staged-r170.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
