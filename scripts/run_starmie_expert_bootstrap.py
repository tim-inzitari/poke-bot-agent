#!/usr/bin/env python3
"""Build and protect one temporal1 specialist from the distilled core.

The historical command-line defaults remain Starmie-compatible.  New
specialists pass ``--archetype`` and ``--expert-corpus``.  The protocol's 25
supervised epochs are exact: validation still selects the frozen checkpoint,
but diagnostic patience never shortens the bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import archetypes, checkpoint, device as device_mod
from poke_bot.pure_rl.expert_rehearsal import (
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_registry import freeze_model, verify_frozen_model
from poke_bot.train import belief_card_vocab_from_state, supervised_rehearsal_step


FAMILY = "starmie_expert_bootstrap_from_distilled_core_v1"
TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)
SPECIALIST_AUX_EXPANSION_SCHEMA = (
    "poke_bot.specialist_aux_archetype_head_expansion/v1"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _specialist_hot_start_from_core(
    core_path: Path,
    *,
    run_dir: Path,
    archetype: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Append registered archetype rows without changing the frozen core."""

    payload = checkpoint.load_checkpoint(core_path, map_location="cpu")
    state = dict(payload.get("model_state_dict") or {})
    weight_key = "aux_head.3.weight"
    bias_key = "aux_head.3.bias"
    old_weight = state.get(weight_key)
    old_bias = state.get(bias_key)
    if not isinstance(old_weight, torch.Tensor) or not isinstance(
        old_bias, torch.Tensor
    ):
        raise RuntimeError("shared core lacks the canonical archetype head")

    target_ids = list(archetypes.archetype_ids())
    target_classes = len(target_ids) + 1
    old_classes = int(old_weight.shape[0])
    parent_digest = checkpoint.checkpoint_digest(core_path)
    if old_classes == target_classes:
        return (
            core_path,
            parent_digest,
            {
                "schema": SPECIALIST_AUX_EXPANSION_SCHEMA,
                "status": "already_current",
                "parent_checkpoint": str(core_path),
                "parent_checkpoint_digest": parent_digest,
                "target_archetype_ids": target_ids,
                "classes": target_classes,
            },
        )

    compatible_orders = (
        archetypes.CUMULATIVE_V4_AUX_ARCHETYPE_IDS,
        archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS,
        archetypes.LEGACY_AUX_ARCHETYPE_IDS,
    )
    old_ids = next(
        (
            list(order)
            for order in compatible_orders
            if old_classes == len(order) + 1
        ),
        None,
    )
    if old_ids is None:
        raise RuntimeError(
            "shared core archetype head cannot be append-expanded safely: "
            f"classes={old_classes}"
        )
    if any(name not in target_ids for name in old_ids):
        raise RuntimeError("shared core archetype order is not append-compatible")

    seed_material = (
        f"{parent_digest}|{archetype}|{SPECIALIST_AUX_EXPANSION_SCHEMA}"
    ).encode("utf-8")
    expansion_seed = int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8], "big"
    ) % (2**31)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(expansion_seed)
        expanded = nn.Linear(
            int(old_weight.shape[1]),
            target_classes,
            bias=True,
            device="cpu",
            dtype=old_weight.dtype,
        )
    with torch.no_grad():
        for old_index, name in enumerate(old_ids):
            new_index = target_ids.index(name)
            expanded.weight[new_index].copy_(old_weight[old_index])
            expanded.bias[new_index].copy_(old_bias[old_index])
        expanded.weight[-1].copy_(old_weight[-1])
        expanded.bias[-1].copy_(old_bias[-1])

    expansion = {
        "schema": SPECIALIST_AUX_EXPANSION_SCHEMA,
        "status": "expanded_append_only",
        "parent_checkpoint": str(core_path),
        "parent_checkpoint_digest": parent_digest,
        "source_archetype_ids": old_ids,
        "target_archetype_ids": target_ids,
        "source_classes": old_classes,
        "target_classes": target_classes,
        "copied_named_rows": old_ids,
        "newly_initialized_rows": [
            name for name in target_ids if name not in old_ids
        ],
        "unknown_row_moved_to_final": True,
        "expansion_seed": expansion_seed,
        "optimizer_state_imported": False,
    }
    state[weight_key] = expanded.weight.detach().clone()
    state[bias_key] = expanded.bias.detach().clone()
    payload["model_state_dict"] = state
    payload["step"] = 0
    payload["epoch"] = 0
    payload["rl_iteration"] = 0
    payload["archetype_id"] = "unknown"
    payload["model_id"] = f"{archetype}.shared_core_hot_start"
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "early_stop_state",
    ):
        payload.pop(key, None)
    payload["extra"] = {
        **dict(payload.get("extra") or {}),
        "specialist_aux_archetype_head_expansion": expansion,
    }

    hot_start = run_dir / "shared_core_hot_start.current_archetypes.pt"
    if hot_start.is_file():
        existing = checkpoint.load_checkpoint(hot_start, map_location="cpu")
        existing_expansion = dict(
            (existing.get("extra") or {}).get(
                "specialist_aux_archetype_head_expansion"
            )
            or {}
        )
        if existing_expansion != expansion:
            raise RuntimeError("existing specialist hot-start identity changed")
    else:
        checkpoint.atomic_torch_save(payload, hot_start)
    hot_digest = checkpoint.checkpoint_digest(hot_start)
    return hot_start, hot_digest, {**expansion, "checkpoint_digest": hot_digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    corpus_group = parser.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument("--expert-corpus", type=Path)
    corpus_group.add_argument("--starmie-corpus", type=Path)
    parser.add_argument("--archetype", default="starmie")
    parser.add_argument("--family", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--core-family", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-decisions", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=12288)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    parser.add_argument("--required-target", action="append", default=[])
    args = parser.parse_args(argv)
    required_targets = tuple(args.required_target or TARGETS)
    if (
        "temporal_action_rows" not in required_targets
        or len(set(required_targets)) != len(required_targets)
        or not set(required_targets).issubset(TARGETS)
    ):
        raise ValueError("invalid specialist target-coverage contract")
    all_auxiliary_heads_trained = set(required_targets) == set(TARGETS)
    archetype = str(args.archetype).strip().casefold()
    if not archetype or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in archetype):
        raise ValueError("specialist archetype must be a non-empty lowercase slug")
    if int(args.epochs) != 25:
        raise ValueError("specialist bootstrap is locked to exactly 25 epochs")
    corpus_argument = args.expert_corpus or args.starmie_corpus
    assert corpus_argument is not None
    family_name = str(args.family).strip() or (
        f"{archetype}_expert_bootstrap_from_distilled_core_v1"
    )
    display_name = str(args.display_name).strip() or (
        f"{archetype.replace('-', ' ').title()} Expert Bootstrap from Distilled Core"
    )
    state_schema = (
        "poke_bot.starmie_expert_bootstrap_state/v1"
        if archetype == "starmie"
        else "poke_bot.specialist_expert_bootstrap_state/v1"
    )
    epoch_schema = (
        "poke_bot.starmie_bootstrap_epoch/v1"
        if archetype == "starmie"
        else "poke_bot.specialist_bootstrap_epoch/v1"
    )
    ready_schema = (
        "poke_bot.starmie_expert_bootstrap_ready/v1"
        if archetype == "starmie"
        else "poke_bot.specialist_expert_bootstrap_ready/v1"
    )

    core = verify_frozen_model(args.core_family.expanduser().resolve())
    core_path = Path(str(core["model_path"])).resolve()
    core_payload = checkpoint.load_checkpoint(core_path, map_location="cpu")
    core_cfg = dict(core_payload.get("model_config") or {})
    if (
        int(core_cfg.get("temporal_layers", -1)) != 1
        or str(core_cfg.get("decision_context") or "") != "history"
        or int(core_cfg.get("max_context", -1)) != 320
        or str(core_payload.get("archetype_id") or "") != "unknown"
    ):
        raise ValueError("protected distilled core has the wrong architecture/role")
    corpus_pointer = corpus_argument.expanduser().resolve()
    identity = resolve_expert_manifest(
        corpus_pointer,
        min_decisions=int(args.min_decisions),
        require_protected=True,
        required_archetype=archetype,
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=required_targets,
    )
    family_dir = args.registry_root.expanduser().resolve() / family_name
    if args.ready.is_file() and family_dir.is_dir():
        ready = json.loads(args.ready.read_text(encoding="utf-8"))
        frozen = verify_frozen_model(family_dir)
        if (
            ready.get("status") == "ready"
            and ready.get("checkpoint_digest") == frozen.get("checkpoint_digest")
            and ready.get("core_checkpoint_digest") == core.get("checkpoint_digest")
            and ready.get("expert_manifest_sha256") == identity.digest
            and ready.get("acting_seat_archetype") == archetype
        ):
            print(json.dumps(ready, indent=2), flush=True)
            return 0
        raise RuntimeError("existing specialist readiness identity changed")

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hot_start, hot_start_digest, hot_start_expansion = (
        _specialist_hot_start_from_core(
            core_path,
            run_dir=run_dir,
            archetype=archetype,
        )
    )
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if state and (
        state.get("core_checkpoint_digest") != core.get("checkpoint_digest")
        or state.get("manifest_digest") != identity.digest
        or state.get("hot_start_checkpoint_digest") != hot_start_digest
    ):
        raise RuntimeError("specialist bootstrap state identity changed")

    device = device_mod.training_device(prefer_name="RTX PRO 5000", allow_cpu=False)
    core_payload = checkpoint.load_checkpoint(core_path, map_location="cpu")
    belief_card_vocab = belief_card_vocab_from_state(
        dict(core_payload.get("model_state_dict") or {})
    )
    cache = ResidentExpertCorpusCache(cpu_pack_root=args.cpu_pack_root)
    print(
        f"[specialist-bootstrap:{archetype}] loading protected acting-seat corpus "
        f"records={identity.records} decisions={identity.decisions} device={device}",
        flush=True,
    )
    corpus = cache.prepare(
        identity,
        device=device,
        seed=int(args.split_seed),
        max_context=320,
        belief_card_vocab=belief_card_vocab,
    )
    if not corpus.has_exact_targets:
        raise RuntimeError(
            "specialist bootstrap corpus lost required all-head targets"
        )
    history = list(state.get("history") or [])
    best_metric = float(state.get("best_metric", math.inf))
    best_path = str(state.get("best_path") or "")
    best_digest = str(state.get("best_digest") or "")
    bad_epochs = int(state.get("bad_epochs", 0))
    parent = hot_start
    parent_digest = hot_start_digest
    start_epoch = 1
    if history:
        last = history[-1]
        parent = Path(str(last["checkpoint"])).resolve()
        parent_digest = str(last["checkpoint_digest"])
        if checkpoint.checkpoint_digest(parent) != parent_digest:
            raise RuntimeError("specialist resume parent digest mismatch")
        start_epoch = int(last["epoch"]) + 1
    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            output = checkpoint_dir / f"epoch_{epoch:02d}.pt"
            if output.is_file():
                saved = checkpoint.load_checkpoint(output, map_location="cpu")
                extra = dict((saved.get("extra") or {}).get("specialist_bootstrap") or {})
                rehearsal = dict((saved.get("extra") or {}).get("expert_rehearsal") or {})
                if (
                    extra.get("schema") != epoch_schema
                    or int(extra.get("epoch", -1)) != epoch
                    or extra.get("core_checkpoint_digest") != core.get("checkpoint_digest")
                    or extra.get("parent_digest") != parent_digest
                    or extra.get("manifest_digest") != identity.digest
                    or extra.get("acting_seat_archetype") != archetype
                ):
                    raise RuntimeError(f"existing specialist epoch identity drift: {output}")
                result = {
                    "candidate_digest": checkpoint.checkpoint_digest(output),
                    "train_metrics": rehearsal.get("train_metrics"),
                    "validation_metrics": rehearsal.get("validation_metrics"),
                    "reused": True,
                }
            else:
                result = supervised_rehearsal_step(
                    corpus,
                    base_ckpt=parent,
                    output_path=output,
                    parent_digest=parent_digest,
                    rehearsal_iteration=epoch,
                    manifest_identity=identity.as_dict(),
                    epochs=1,
                    lr=5e-5,
                    requested_batch_size=int(args.batch_size),
                    seed=20260722 + epoch,
                    corpus_split_seed=int(args.split_seed),
                    device=device,
                    aux_loss_weight=0.05,
                    opp_hand_loss_weight=0.05,
                    opp_remainder_loss_weight=0.05,
                    lethal_threat_loss_weight=0.025,
                    prize_race_loss_weight=0.025,
                    alakazam_guide_loss_weight=0.0,
                    output_archetype_id=archetype,
                    output_model_id=f"{args.run_name}.epoch{epoch:02d}",
                    extra_updates={
                        "specialist_bootstrap": {
                            "schema": epoch_schema,
                            "epoch": epoch,
                            "acting_seat_archetype": archetype,
                            "core_checkpoint_digest": core["checkpoint_digest"],
                            "hot_start_checkpoint_digest": hot_start_digest,
                            "hot_start_expansion": hot_start_expansion,
                            "parent_digest": parent_digest,
                            "manifest_digest": identity.digest,
                            "all_auxiliary_heads_trained": (
                                all_auxiliary_heads_trained
                            ),
                            "trained_target_coverage": list(required_targets),
                            "inherited_target_coverage": [
                                target
                                for target in TARGETS
                                if target not in required_targets
                            ],
                        }
                    },
                )
            metric = float((result.get("validation_metrics") or {}).get("total_loss", math.inf))
            if not math.isfinite(metric):
                raise RuntimeError("specialist validation metric is not finite")
            row = {
                "epoch": epoch,
                "checkpoint": str(output),
                "checkpoint_digest": str(result["candidate_digest"]),
                "parent_digest": parent_digest,
                "validation_loss": metric,
                "validation_accuracy": float(
                    (result.get("validation_metrics") or {}).get("policy_acc", 0.0)
                ),
                "train_metrics": result.get("train_metrics"),
                "validation_metrics": result.get("validation_metrics"),
                "reused": bool(result.get("reused")),
            }
            history.append(row)
            if metric < best_metric - float(args.min_delta):
                best_metric = metric
                best_path = str(output)
                best_digest = str(result["candidate_digest"])
                bad_epochs = 0
            else:
                bad_epochs += 1
            parent = output
            parent_digest = str(result["candidate_digest"])
            state = {
                "schema": state_schema,
                "status": "training",
                "acting_seat_archetype": archetype,
                "core_checkpoint": str(core_path),
                "core_checkpoint_digest": core["checkpoint_digest"],
                "hot_start_checkpoint": str(hot_start),
                "hot_start_checkpoint_digest": hot_start_digest,
                "hot_start_expansion": hot_start_expansion,
                "manifest": identity.as_dict(),
                "manifest_digest": identity.digest,
                "history": history,
                "best_path": best_path,
                "best_digest": best_digest,
                "best_metric": best_metric,
                "bad_epochs": bad_epochs,
                "patience": int(args.patience),
                "epochs_max": int(args.epochs),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(state_path, state)
            print(
                f"[specialist-bootstrap:{archetype}] epoch={epoch}/25 "
                f"val_loss={metric:.6f} "
                f"best={best_metric:.6f} patience={bad_epochs}/{int(args.patience)}",
                flush=True,
            )
    finally:
        cache.release()

    best = Path(best_path).resolve()
    if not best.is_file() or checkpoint.checkpoint_digest(best) != best_digest:
        raise RuntimeError("selected specialist bootstrap identity is invalid")
    best_payload = checkpoint.load_checkpoint(best, map_location="cpu")
    if str(best_payload.get("archetype_id") or "") != archetype:
        raise RuntimeError("selected specialist checkpoint lost archetype metadata")
    frozen = freeze_model(
        registry_root=args.registry_root,
        family=family_name,
        display_name=display_name,
        checkpoint=best,
        expected_digest=best_digest,
        provenance={
            "initialized_from_family": core.get("family"),
            "initialized_from_digest": core["checkpoint_digest"],
            "specialist_hot_start": {
                "checkpoint": str(hot_start),
                "checkpoint_digest": hot_start_digest,
                "aux_archetype_head_expansion": hot_start_expansion,
            },
            "expert_manifest": identity.as_dict(),
            **(
                {"starmie_manifest": identity.as_dict()}
                if archetype == "starmie"
                else {}
            ),
            "acting_seat_archetype": archetype,
            "all_auxiliary_heads_trained": all_auxiliary_heads_trained,
            "trained_target_coverage": list(required_targets),
            "inherited_target_coverage": [
                target
                for target in TARGETS
                if target not in required_targets
            ],
            "epochs_max": int(args.epochs),
            "early_stop_patience": int(args.patience),
            "history": history,
        },
        evidence={
            "kind": "specialist_episode_disjoint_expert_validation",
            "best_metric": best_metric,
            "epochs_completed": len(history),
        },
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(frozen["model_path"]).parent)
    ready = {
        "schema": ready_schema,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "checkpoint": frozen["model_path"],
        "checkpoint_digest": frozen["checkpoint_digest"],
        "core_checkpoint_digest": core["checkpoint_digest"],
        "hot_start_checkpoint": str(hot_start),
        "hot_start_checkpoint_digest": hot_start_digest,
        "acting_seat_archetype": archetype,
        "family": family_name,
        "expert_manifest": identity.path,
        "expert_manifest_sha256": identity.digest,
        **(
            {
                "starmie_manifest": identity.path,
                "starmie_manifest_sha256": identity.digest,
            }
            if archetype == "starmie"
            else {}
        ),
        "records": identity.records,
        "decisions": identity.decisions,
        "epochs_completed": len(history),
        "epochs_max": int(args.epochs),
        "early_stop_patience": int(args.patience),
        "best_metric": best_metric,
        "trained_target_coverage": list(required_targets),
        "inherited_target_coverage": [
            target for target in TARGETS if target not in required_targets
        ],
    }
    atomic_json(args.ready, ready)
    atomic_json(state_path, {**state, "status": "complete", "ready": ready})
    print(json.dumps(ready, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
