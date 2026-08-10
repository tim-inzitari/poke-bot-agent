#!/usr/bin/env python3
"""Fail-closed Cox/Chao held policy_acc ≥ 0.90 gate for Slop Box H10 RTP.

Evaluates argmax policy accuracy on the held validation split restricted to
acting seats whose TeamNames[seat] is exactly 'James Cox & Henry Chao'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint, device as device_mod  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.train import (  # noqa: E402
    TrainConfig,
    belief_card_vocab_from_state,
    device_temporal_batch_losses,
    load_model_from_checkpoint,
)

COX = "James Cox & Henry Chao"
GATE_SCHEMA = "poke_bot.slop_box_cox_chao_held_policy_acc_gate_r170/v1"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models/"
            "final-format-slop-box-h10-rtp-expert-bootstrap-v1/model.pt"
        ),
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
        "--ready",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "final-format-slop-box-h10-rtp-bootstrap-ready.json"
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
            "slop-box-cox-chao-held-policy-acc-gate-r170.json"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--quarantine-ready-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    ckpt = args.checkpoint.expanduser().resolve()
    targets_path = args.targets.expanduser().resolve()
    pilot_path = args.pilot_map.expanduser().resolve()
    held_path = args.held_split.expanduser().resolve()
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    held = json.loads(held_path.read_text(encoding="utf-8"))

    if int(targets.get("split_seed") or -1) != int(args.split_seed):
        raise RuntimeError(
            f"targets split_seed {targets.get('split_seed')} != "
            f"bootstrap seed {args.split_seed}"
        )
    if int(pilot.get("unverifiable_rows", -1)) != 0:
        raise RuntimeError("pilot map has unverifiable rows; fail closed")
    if held.get("targets_sha256") != file_digest(targets_path):
        raise RuntimeError("held-split targets digest mismatch")
    if held.get("pilot_map_sha256") != file_digest(pilot_path):
        raise RuntimeError("held-split pilot-map digest mismatch")

    pilots = {
        (str(row["episode_id"]), int(row["seat"])): str(row["team_name"])
        for row in pilot.get("rows") or ()
    }
    val_rows = list(targets.get("validation_rows") or ())
    cox_game_offsets = [
        index
        for index, row in enumerate(val_rows)
        if pilots.get((str(row["episode_id"]), int(row["seat"]))) == COX
    ]
    if not cox_game_offsets:
        raise RuntimeError("Cox/Chao held validation set is empty")

    pointer = args.expert_pointer.expanduser().resolve()
    identity = resolve_expert_manifest(
        pointer,
        require_protected=True,
        required_archetype="teal-mask-ogerpon-ex",
        min_decisions=90000,
    )
    if args.device:
        device = torch.device(args.device)
    else:
        device = device_mod.training_device(
            prefer_name="RTX PRO 5000", allow_cpu=False
        )
        # Prefer a free GPU if bootstrap still holds cuda:0 for RTP.
        if str(device).startswith("cuda:"):
            try:
                free, _total = torch.cuda.mem_get_info(int(str(device).split(":")[1]))
                if free < 4 * (1024**3):
                    for idx in range(torch.cuda.device_count()):
                        free_i, _ = torch.cuda.mem_get_info(idx)
                        if free_i >= 8 * (1024**3):
                            device = torch.device(f"cuda:{idx}")
                            break
            except Exception:
                pass

    cache = ResidentExpertCorpusCache(
        cpu_pack_root=args.cpu_pack_root.expanduser().resolve()
    )
    core_payload = checkpoint.load_checkpoint(ckpt, map_location="cpu")
    belief_card_vocab = belief_card_vocab_from_state(
        dict(core_payload.get("model_state_dict") or {})
    )
    corpus = cache.prepare(
        identity,
        device=device,
        seed=int(args.split_seed),
        max_context=320,
        belief_card_vocab=belief_card_vocab,
    )
    if int(corpus.val_games) != len(val_rows):
        raise RuntimeError(
            f"val game count mismatch corpus={corpus.val_games} targets={len(val_rows)}"
        )

    model = load_model_from_checkpoint(ckpt, device=device)
    model.eval()
    cfg = TrainConfig()
    correct = 0.0
    total = 0
    games = 0
    for offset in cox_game_offsets:
        game_id = int(corpus.train_games) + int(offset)
        batch_ids = torch.tensor([game_id], device=corpus.device, dtype=torch.long)
        with torch.no_grad():
            _, metrics = device_temporal_batch_losses(
                model,
                corpus,
                batch_ids,
                value_weight=cfg.value_loss_weight,
                aux_weight=cfg.aux_loss_weight,
                opp_hand_weight=cfg.opp_hand_loss_weight,
                opp_remainder_weight=cfg.opp_remainder_loss_weight,
                lethal_threat_weight=cfg.lethal_threat_loss_weight,
                prize_race_weight=cfg.prize_race_loss_weight,
                alakazam_guide_weight=0.0,
                current_deck_guide_training_mode="strategic_directional_v2",
                setup_board_outcome_loss_weight=cfg.setup_board_outcome_loss_weight,
                combo_state_loss_weight=cfg.combo_state_loss_weight,
                expanded_head_weights=cfg.expanded_head_loss_weights,
                archetype_residual_weights=cfg.archetype_residual_loss_weights,
            )
        n = int(metrics.n_decisions)
        acc = float(metrics.policy_acc)
        if n <= 0 or not math.isfinite(acc):
            raise RuntimeError(f"invalid Cox/Chao game metrics at offset {offset}")
        correct += acc * n
        total += n
        games += 1

    policy_acc = float(correct / total)
    passed = policy_acc >= float(args.threshold)
    receipt = {
        "schema": GATE_SCHEMA,
        "goal_revision": 170,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_id": "cox_chao_held_policy_acc",
        "definition": (
            "mean(argmax(policy_logits) == expert_target_action_index) over "
            "legal-masked Cox/Chao-only held validation decisions"
        ),
        "threshold": float(args.threshold),
        "cox_chao_held_policy_acc": policy_acc,
        "decisions": int(total),
        "games": int(games),
        "split_seed": int(args.split_seed),
        "device": str(device),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": checkpoint.checkpoint_digest(ckpt),
        "held_split": str(held_path),
        "held_split_sha256": file_digest(held_path),
        "pilot_map": str(pilot_path),
        "pilot_map_sha256": file_digest(pilot_path),
        "targets": str(targets_path),
        "targets_sha256": file_digest(targets_path),
        "team_name_exact": COX,
        "status": "passed" if passed else "failed",
        "fail_closed": True,
        "passed": passed,
    }
    out = args.output.expanduser().resolve()
    atomic_json(out, receipt)

    ready_path = args.ready.expanduser().resolve()
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        ready["cox_chao_policy_accuracy_gate"] = {
            "metric_id": "cox_chao_held_policy_acc",
            "threshold": float(args.threshold),
            "measured": policy_acc,
            "passed": passed,
            "fail_closed": True,
            "receipt": str(out),
            "receipt_sha256": file_digest(out),
        }
        if passed:
            atomic_json(ready_path, ready)
        elif args.quarantine_ready_on_fail:
            ready["status"] = "failed_cox_chao_held_policy_acc_gate"
            quarantine = ready_path.with_name(
                ready_path.stem + "-FAILED-cox-chao-gate-r170.json"
            )
            atomic_json(quarantine, ready)
            stub = {
                "schema": "poke_bot.specialist_expert_bootstrap_ready/v1",
                "status": "failed_cox_chao_held_policy_acc_gate",
                "goal_revision": 170,
                "fail_closed": True,
                "cox_chao_held_policy_acc": policy_acc,
                "threshold": float(args.threshold),
                "gate_receipt": str(out),
                "gate_receipt_sha256": file_digest(out),
                "quarantined_ready": str(quarantine),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(ready_path, stub)

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
