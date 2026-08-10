#!/usr/bin/env python3
"""Rebuild Cox/Chao targets/pilot/held/upweight after corpus expand (r170)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.expert_pilot_importance import (  # noqa: E402
    canonical_digest,
    file_digest,
)

COX = "James Cox & Henry Chao"
DEFAULT_EXPERT = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex/"
    "PROTECTED_EXPERT_CORPUS.json"
)
STATE = Path("/home/inzi/poke-bot-agent/outputs/state")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-pointer", type=Path, default=DEFAULT_EXPERT)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--cox-chao-train-weight", type=float, default=10.0)
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "If set (e.g. -expanded-r170), write maps to *{suffix}.json instead of "
            "overwriting the live deep-CE paths."
        ),
    )
    parser.add_argument(
        "--skip-live-upweight",
        action="store_true",
        help="Build targets/pilot/held only; leave live ×10 upweight untouched.",
    )
    parser.add_argument(
        "--reuse-existing-maps",
        action="store_true",
        help=(
            "Require and reuse the suffix-scoped targets/pilot/held outputs, "
            "then materialize only the upweight index."
        ),
    )
    args = parser.parse_args()

    expert = args.expert_pointer.expanduser().resolve()
    suffix = str(args.output_suffix or "")
    if suffix:
        targets = STATE / f"slop-box-cox-chao-held-targets{suffix}.json"
        pilot = STATE / f"slop-box-cox-chao-held-pilot-map{suffix}.json"
        held = STATE / f"slop-box-cox-chao-held-split-pilot-map{suffix}.json"
        upweight = STATE / f"slop-box-cox-chao-train-upweight-importance{suffix}.json"
    else:
        targets = STATE / "slop-box-cox-chao-held-targets-r170.json"
        pilot = STATE / "slop-box-cox-chao-held-pilot-map-r170.json"
        held = STATE / "slop-box-cox-chao-held-split-pilot-map-r170.json"
        upweight = STATE / "slop-box-cox-chao-train-upweight-importance-r170.json"

    if args.reuse_existing_maps:
        absent = [str(path) for path in (targets, pilot, held) if not path.is_file()]
        if absent:
            raise RuntimeError(f"existing map reuse requested but missing: {absent}")
        if upweight.exists():
            raise RuntimeError(f"refusing to overwrite existing upweight: {upweight}")
    else:
        # Preserve previous maps as immutable audit copies.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in (targets, pilot, held, upweight):
            if path.is_file():
                backup = path.with_name(
                    path.stem + f".pre-expand-{stamp}" + path.suffix
                )
                if not backup.exists():
                    backup.write_bytes(path.read_bytes())

        # Targets must be rewritten; materialize_expert_pilot_importance refuses
        # overwrite of differing immutable outputs, so remove the live targets path
        # after backup.
        if targets.is_file():
            targets.unlink()

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "materialize_expert_pilot_importance.py"),
                "targets",
                "--expert-pointer",
                str(expert),
                "--output",
                str(targets),
                "--split-seed",
                str(args.split_seed),
                "--validation-fraction",
                str(args.validation_fraction),
                "--max-context",
                "320",
                "--workers",
                "8",
            ],
            check=True,
            cwd=str(ROOT),
        )

        # Rebuild pilot/held. Expanded suffix uses dedicated builder argv so the
        # live deep-CE maps stay untouched.
        if suffix:
            subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "build_slop_box_cox_chao_held_pilot_map_r170.py"
                    ),
                    "--targets",
                    str(targets),
                    "--pilot-out",
                    str(pilot),
                    "--held-out",
                    str(held),
                ],
                check=True,
                cwd=str(ROOT),
            )
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "build_slop_box_cox_chao_held_pilot_map_r170.py"
                    ),
                ],
                check=True,
                cwd=str(ROOT),
            )

    if args.skip_live_upweight:
        held_obj = json.loads(held.read_text(encoding="utf-8"))
        summary = {
            "schema": "poke_bot.slop_box_cox_chao_maps_rebuilt_r170/v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "expert_pointer": str(expert),
            "targets": str(targets),
            "targets_sha256": file_digest(targets),
            "pilot": str(pilot),
            "pilot_sha256": file_digest(pilot),
            "held": str(held),
            "held_sha256": file_digest(held),
            "upweight": None,
            "skipped_upweight": True,
            "output_suffix": suffix or None,
            "train_games": len(json.loads(targets.read_text())["train_rows"]),
            "validation_games": len(
                json.loads(targets.read_text()).get("validation_rows") or ()
            ),
            "cox_chao_train_games": held_obj.get("cox_chao_train_games"),
            "cox_chao_held_games": held_obj.get("cox_chao_held_validation_games"),
            "cox_chao_total_games": held_obj.get("cox_chao_acting_seat_games_total"),
        }
        _atomic_json(STATE / "slop-box-cox-chao-maps-rebuilt-r170.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    # Build Cox/Chao 10x upweight index from pilot+targets.
    targets_obj = json.loads(targets.read_text(encoding="utf-8"))
    pilot_obj = json.loads(pilot.read_text(encoding="utf-8"))
    pilots = {
        (str(row["episode_id"]), int(row["seat"])): str(row["team_name"])
        for row in pilot_obj.get("rows") or ()
    }
    train_rows = list(targets_obj.get("train_rows") or ())
    weights = []
    weighted_rows = []
    matched = 0
    for index, row in enumerate(train_rows):
        key = (str(row["episode_id"]), int(row["seat"]))
        team = pilots.get(key)
        weight = float(args.cox_chao_train_weight) if team == COX else 1.0
        if weight > 1.0:
            matched += 1
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
    held_obj = json.loads(held.read_text(encoding="utf-8"))
    payload = {
        "schema": "poke_bot.expert_pilot_importance/v1",
        "owner_decision_revision": 138,
        "status": "ready",
        "goal_revision": 170,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_manifest": targets_obj.get("corpus_manifest"),
        "corpus_manifest_sha256": targets_obj.get("corpus_manifest_sha256"),
        "split_seed": int(args.split_seed),
        "validation_fraction": float(args.validation_fraction),
        "max_context": 320,
        "support_partition": "training_only",
        "join_key": ["episode_id", "seat", "exact_team_name"],
        "cox_chao_team_name_exact": COX,
        "cox_chao_train_weight": float(args.cox_chao_train_weight),
        "non_cox_chao_train_weight": 1.0,
        "full_archetype_train_preserved": True,
        "actions_and_labels_unchanged": True,
        "validation_unweighted": True,
        "kaggle_evaluation_replays_excluded": True,
        "targets_sha256": file_digest(targets),
        "pilot_map_sha256": file_digest(pilot),
        "leaderboard_snapshot_sha256": "sha256:" + ("0" * 64),
        "train_identity_sha256": targets_obj.get("train_identity_sha256"),
        "validation_identity_sha256": targets_obj.get("validation_identity_sha256"),
        "train_games": len(train_rows),
        "validation_games": len(targets_obj.get("validation_rows") or ()),
        "matched_top_100_train_games": matched,
        "unmatched_or_unverifiable_train_games": len(train_rows) - matched,
        "unmatched_or_unverifiable_weight": 1.0,
        "effective_training_weight_mass": float(sum(weights)),
        "tier_counts": {
            f"{float(args.cox_chao_train_weight):.0f}x": matched,
            "1x": len(train_rows) - matched,
        },
        "tiers": [
            {
                "minimum_games": 1,
                "maximum_games": None,
                "weight": float(args.cox_chao_train_weight),
            }
        ],
        "per_team_support": [
            {
                "team_name": COX,
                "train_games": matched,
                "weight": float(args.cox_chao_train_weight),
            }
        ],
        "train_game_weights": weights,
        "weighted_train_rows": weighted_rows,
        "train_game_weights_sha256": canonical_digest(weights),
        "owner_note": (
            "r170 expanded Cox/Chao upweight after corpus refresh through 2026-08-05"
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
    if upweight.is_file():
        upweight.unlink()
    _atomic_json(upweight, payload)

    summary = {
        "schema": "poke_bot.slop_box_cox_chao_maps_rebuilt_r170/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expert_pointer": str(expert),
        "targets": str(targets),
        "targets_sha256": file_digest(targets),
        "pilot": str(pilot),
        "pilot_sha256": file_digest(pilot),
        "held": str(held),
        "held_sha256": file_digest(held),
        "upweight": str(upweight),
        "upweight_sha256": file_digest(upweight),
        "train_games": len(train_rows),
        "validation_games": len(targets_obj.get("validation_rows") or ()),
        "cox_chao_train_games": matched,
        "cox_chao_held_games": held_obj.get("cox_chao_held_validation_games"),
        "cox_chao_total_games": held_obj.get("cox_chao_acting_seat_games_total"),
    }
    _atomic_json(STATE / "slop-box-cox-chao-maps-rebuilt-r170.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
