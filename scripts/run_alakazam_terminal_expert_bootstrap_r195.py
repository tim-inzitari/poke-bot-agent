#!/usr/bin/env python3
"""Run the receipt-bound r195 Alakazam terminal expert bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint, device as device_mod  # noqa: E402
from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    commit_rehearsal_receipt,
    recover_rehearsal,
    rehearsal_paths,
    resolve_expert_manifest,
)
from poke_bot.train import (  # noqa: E402
    GUIDE_TRAINING_MODE_DIRECTIONAL,
    belief_card_vocab_from_state,
    supervised_rehearsal_step,
)


PARENT_SHA256 = "sha256:87caf05bdeda3a798268905a5670841125b1797f31b9a823343c393d7f0ced65"
POINTER_SHA256 = "sha256:2427c2b51cc93beccc3618085d9c77c83f49fb69cabf0208040608c384a659cd"
REQUIRED_TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)
LOSS_WEIGHTS = {
    "value": 1.0,
    "archetype": 0.05,
    "opponent_hand": 0.05,
    "opponent_hidden_remainder": 0.05,
    "lethal_threat": 0.025,
    "prize_race": 0.025,
    "alakazam_guide": 0.05,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable r195 receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def expanded_contract(payload: dict[str, Any], before_iteration: int) -> dict[str, Any]:
    training = dict((payload.get("extra") or {}).get("expanded_head_training") or {})
    if training.get("schema") != "poke_bot.expanded_head_training/v1":
        raise RuntimeError("terminal parent lacks expanded-head training provenance")
    weights = {
        str(name): float(weight)
        for name, weight in dict(training.get("loss_weights") or {}).items()
        if float(weight) > 0.0
    }
    if len(weights) != 11:
        raise RuntimeError("terminal parent does not expose all 11 expanded-head losses")
    return {
        "schema": "poke_bot.expanded_head_schedule/v1",
        "target_schema": str(training.get("target_schema_version") or ""),
        "target_schema_digest": str(training.get("target_schema_digest") or ""),
        "schedule_digest": str(training.get("schedule_digest") or ""),
        "epoch": 25,
        "stage_index": 5,
        "loss_weights": weights,
        "runtime_enabled_heads": [],
        "rehearsal_iteration": int(before_iteration),
    }


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    parent = args.parent.expanduser().resolve()
    pointer = args.expert_manifest.expanduser().resolve()
    if checkpoint.checkpoint_digest(parent) != PARENT_SHA256:
        raise RuntimeError("r195 terminal Alakazam parent digest changed")
    if sha256(pointer) != POINTER_SHA256:
        raise RuntimeError("r195 protected expert pointer digest changed")
    payload = checkpoint.load_checkpoint(parent, map_location="cpu")
    profile = dict(payload.get("model_config") or {})
    if (
        str(payload.get("archetype_id") or "") != "alakazam"
        or not bool(profile.get("h10_capacity_enabled"))
        or not bool(profile.get("expanded_heads_enabled"))
        or not bool(profile.get("decision_fusion_enabled"))
        or bool(profile.get("combo_state_route_enabled", True))
    ):
        raise RuntimeError("terminal parent is not the expected H10 Fusion-v3 Alakazam model")
    max_context = int(profile.get("max_context") or 0)
    if max_context <= 0:
        raise RuntimeError("terminal parent lost temporal context")
    contract = expanded_contract(payload, int(args.before_iteration))
    identity = resolve_expert_manifest(
        pointer,
        min_decisions=20_000,
        require_protected=True,
        required_archetype="alakazam",
        required_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        required_max_context=max_context,
        required_target_coverage=REQUIRED_TARGETS,
        required_expanded_target_schema=str(contract["target_schema"]),
        required_expanded_target_digest=str(contract["target_schema_digest"]),
        required_expanded_heads=tuple(contract["loss_weights"]),
    )
    seat = read_object(args.seat_split_receipt.expanduser().resolve())
    seat_path = args.seat_split_receipt.expanduser().resolve()
    seat_identity = {
        "schema": str(seat.get("schema") or ""),
        "path": str(seat_path),
        "sha256": sha256(seat_path),
    }
    if seat_identity["schema"] != "poke_bot.alakazam_refresh_rehearsal_seat_split_index/v1":
        raise RuntimeError("r195 exact-seat receipt schema changed")
    return payload, identity, {"contract": contract, "seat": seat_identity}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    parser.add_argument("--curriculum-spec", type=Path, required=True)
    parser.add_argument("--head-role-map", type=Path, required=True)
    parser.add_argument("--curriculum-validation", type=Path, required=True)
    parser.add_argument("--seat-split-receipt", type=Path, required=True)
    parser.add_argument("--before-iteration", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--split-seed", type=int, default=5_000_000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if int(args.epochs) != 25:
        raise RuntimeError("r195 expert bootstrap must run exactly 25 epochs")
    for required in (args.curriculum_spec, args.head_role_map, args.curriculum_validation):
        if not required.expanduser().resolve().is_file():
            raise FileNotFoundError(required)

    payload, identity, evidence = validate_inputs(args)
    parent = args.parent.expanduser().resolve()
    parent_digest = checkpoint.checkpoint_digest(parent)
    profile = dict(payload.get("model_config") or {})
    max_context = int(profile["max_context"])
    state = dict(payload.get("model_state_dict") or {})
    belief_vocab = belief_card_vocab_from_state(state)
    run_dir = args.run_dir.expanduser().resolve()
    output, _receipt = rehearsal_paths(run_dir, int(args.before_iteration))
    summary = {
        "schema": "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195_preflight/v1",
        "status": "ready" if args.check else "running",
        "parent": str(parent),
        "parent_sha256": parent_digest,
        "expert_manifest": identity.as_dict(),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "output": str(output),
        "combo_state_loss_weight": 0.0,
        "combo_state_route_enabled": False,
        "submitted_rtp_enabled": False,
    }
    if args.check:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    recovered = recover_rehearsal(
        run_dir,
        before_iteration=int(args.before_iteration),
        parent_digest=parent_digest,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        manifest_identity=identity,
        loss_weights=LOSS_WEIGHTS,
        corpus_split_seed=int(args.split_seed),
        expanded_head_contract=evidence["contract"],
        training_seat_split_receipt=evidence["seat"],
    )
    if recovered is None:
        device = device_mod.training_device(prefer_name="RTX PRO 5000", allow_cpu=False)
        cache = ResidentExpertCorpusCache(cpu_pack_root=args.cpu_pack_root.expanduser().resolve())
        try:
            corpus = cache.prepare(
                identity,
                device=device,
                seed=int(args.split_seed),
                max_context=max_context,
                belief_card_vocab=belief_vocab,
                require_exact_seat_split=True,
            )
            result = supervised_rehearsal_step(
                corpus,
                base_ckpt=parent,
                output_path=output,
                parent_digest=parent_digest,
                rehearsal_iteration=int(args.before_iteration),
                manifest_identity=identity.as_dict(),
                epochs=25,
                lr=float(args.learning_rate),
                requested_batch_size=int(args.batch_size),
                seed=5_100_000 + int(args.before_iteration),
                corpus_split_seed=int(args.split_seed),
                device=device,
                aux_loss_weight=0.05,
                opp_hand_loss_weight=0.05,
                opp_remainder_loss_weight=0.05,
                lethal_threat_loss_weight=0.025,
                prize_race_loss_weight=0.025,
                alakazam_guide_loss_weight=0.05,
                current_deck_guide_training_mode=GUIDE_TRAINING_MODE_DIRECTIONAL,
                setup_board_outcome_loss_weight=0.025,
                combo_state_loss_weight=0.0,
                current_deck_guide_curriculum_spec=str(args.curriculum_spec.expanduser().resolve()),
                current_deck_guide_head_role_map=str(args.head_role_map.expanduser().resolve()),
                current_deck_guide_curriculum_validation_receipt=str(args.curriculum_validation.expanduser().resolve()),
                expanded_head_loss_weights=dict(evidence["contract"]["loss_weights"]),
                expanded_head_schedule=evidence["contract"],
                output_archetype_id="alakazam",
                output_model_id="alakazam-terminal-expert-bootstrap-r195",
                extra_updates={
                    "owner_r195": {
                        "schema": "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195/v1",
                        "epochs": 25,
                        "submitted_rtp_enabled": False,
                        "submission_label_required_literal": "NO RTP",
                    }
                },
                training_seat_split_receipt=evidence["seat"],
            )
            recovered = commit_rehearsal_receipt(
                run_dir,
                before_iteration=int(args.before_iteration),
                parent_digest=parent_digest,
                manifest=identity,
                epochs=25,
                learning_rate=float(args.learning_rate),
                loss_weights=LOSS_WEIGHTS,
                corpus_split_seed=int(args.split_seed),
                result=result,
                expanded_head_contract=evidence["contract"],
                training_seat_split_receipt=evidence["seat"],
            )
        finally:
            cache.release()

    candidate = Path(str(recovered["checkpoint"])).resolve()
    final_payload = checkpoint.load_checkpoint(candidate, map_location="cpu")
    tensors = dict(final_payload.get("model_state_dict") or {})
    completion = {
        "schema": "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195_completion/v1",
        "status": "completed_25_of_25_ready_for_no_rtp_package",
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": parent_digest,
        "expert_manifest": identity.as_dict(),
        "epochs_requested": 25,
        "epochs_completed": 25,
        "checkpoint": str(candidate),
        "checkpoint_size_bytes": candidate.stat().st_size,
        "checkpoint_sha256": checkpoint.checkpoint_digest(candidate),
        "parameter_count": sum(int(value.numel()) for value in tensors.values()),
        "model_config": dict(final_payload.get("model_config") or {}),
        "loss_weights": LOSS_WEIGHTS,
        "setup_board_outcome_loss_weight": 0.025,
        "combo_state_loss_weight": 0.0,
        "combo_state_route_enabled": False,
        "submitted_rtp_enabled": False,
        "submission_label_required_literal": "NO RTP",
        "rehearsal_receipt": str((run_dir / "rehearsals" / f"before_iter_{int(args.before_iteration):05d}.json").resolve()),
    }
    completion_path = run_dir / "completion.json"
    atomic_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
