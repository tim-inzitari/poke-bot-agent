#!/usr/bin/env python3
"""Generalize a frozen passed Alakazam policy into a protected temporal core.

The exact passed checkpoint initializes the student.  A checksummed,
deck-balanced multi-archetype corpus then trains policy, value, archetype,
opponent-belief, lethal, and prize-race heads in immutable one-epoch steps.
Validation selects the best step with patience-based early stopping.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint, device as device_mod
from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint
from poke_bot.matchup_adapters import (
    ZERO_DORMANT_CHECKPOINT_SCHEMA,
    MatchupAdapterBank,
)
from poke_bot.pure_rl.expert_rehearsal import (
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_registry import freeze_model, sha256, verify_frozen_model
from poke_bot.train import (
    belief_card_vocab_from_state,
    load_model_from_checkpoint,
    supervised_rehearsal_step,
)


FAMILY = "deck_agnostic_core_distilled_from_alakazam_v1"
TRANSFER_INITIALIZATION_SCHEMA = "poke_bot.deck_agnostic_transfer_initialization/v1"
TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
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


def _architecture(parent: Path) -> dict[str, Any]:
    # ``assert_trusted_policy_checkpoint`` returns validated metadata, not a
    # checkpoint path.  Consume that metadata directly so this preflight does
    # not accidentally pass a dict back into ``load_checkpoint``.
    trusted = checkpoint.assert_trusted_policy_checkpoint(parent)
    cfg = dict(trusted.get("model_config") or {})
    required = {
        "d_model": 96,
        "spatial_layers": 4,
        "temporal_layers": 1,
        "option_decoder_layers": 4,
        "n_heads": 8,
        "ff_dim": 384,
        "max_context": 320,
        "decision_context": "history",
    }
    drift = {
        key: {"actual": cfg.get(key), "required": value}
        for key, value in required.items()
        if cfg.get(key) != value
    }
    if drift:
        raise ValueError(f"passed Alakazam is not temporal1 Pure-RL: {drift}")
    return cfg


def _fresh_dormant_bank_state() -> tuple[MatchupAdapterBank, dict[str, Any]]:
    """Return one deterministic, exact-zero-output adapter bank.

    The passed Alakazam policy is allowed to contribute its general encoder,
    temporal state, option decoder, policy/value heads, and deck-agnostic
    auxiliary heads.  Its matchup adapters are specialist state and must not
    become the initialization for every later specialist.
    """

    import torch

    rng = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        bank = MatchupAdapterBank(enabled=False)
    finally:
        torch.random.set_rng_state(rng)
    bank.requires_grad_(False)
    state = {
        f"matchup_adapter_bank.{name}": value.detach().cpu().clone()
        for name, value in bank.state_dict().items()
    }
    unsafe = [
        name
        for name, value in state.items()
        if (name.endswith(".up.weight") or name.endswith(".up.bias"))
        and int(value.count_nonzero().item()) != 0
    ]
    if unsafe:
        raise RuntimeError(f"fresh transfer adapter bank is not dormant: {unsafe[:4]}")
    return bank, state


def _validate_transfer_initialization(
    path: Path,
    *,
    source_checkpoint: Path,
    source_digest: str,
) -> dict[str, Any]:
    """Prove that a transfer parent preserved only reusable model state."""

    import torch

    resolved = Path(path).expanduser().resolve()
    payload = checkpoint.load_checkpoint(resolved, map_location="cpu")
    source = checkpoint.load_checkpoint(source_checkpoint, map_location="cpu")
    record = dict((payload.get("extra") or {}).get("core_transfer_initialization") or {})
    if (
        record.get("schema") != TRANSFER_INITIALIZATION_SCHEMA
        or record.get("source_checkpoint") != str(source_checkpoint)
        or record.get("source_checkpoint_digest") != source_digest
        or str(payload.get("archetype_id") or "") != "unknown"
        or any(
            key in payload
            for key in (
                "optimizer_state_dict",
                "scaler_state_dict",
                "scheduler_state_dict",
                "rng_state",
            )
        )
    ):
        raise RuntimeError("deck-agnostic transfer initialization metadata drifted")

    source_state = dict(source.get("model_state_dict") or {})
    transfer_state = dict(payload.get("model_state_dict") or {})
    source_base = {
        name: value
        for name, value in source_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    transfer_base = {
        name: value
        for name, value in transfer_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    if source_base.keys() != transfer_base.keys():
        raise RuntimeError("transfer initialization changed reusable tensor keys")
    changed = [
        name
        for name in source_base
        if not torch.equal(source_base[name], transfer_base[name])
    ]
    if changed:
        raise RuntimeError(
            f"transfer initialization changed reusable tensors: {changed[:5]}"
        )

    bank, expected_adapter_state = _fresh_dormant_bank_state()
    actual_adapter_state = {
        name: value
        for name, value in transfer_state.items()
        if name.startswith("matchup_adapter_bank.")
    }
    if actual_adapter_state.keys() != expected_adapter_state.keys() or any(
        not torch.equal(actual_adapter_state[name], expected_adapter_state[name])
        for name in expected_adapter_state
    ):
        raise RuntimeError("transfer initialization retained specialist adapter state")
    if (payload.get("extra") or {}).get("matchup_adapter_config") != bank.config_dict():
        raise RuntimeError("transfer initialization has the wrong adapter roster")
    validate_zero_dormant_checkpoint(resolved)
    return {
        "path": str(resolved),
        "digest": checkpoint.checkpoint_digest(resolved),
        "reusable_tensor_count": len(source_base),
        "adapter_expert_count": len(bank.expert_ids),
    }


def _materialize_transfer_initialization(
    *,
    source_checkpoint: Path,
    source_digest: str,
    output_path: Path,
) -> dict[str, Any]:
    """Create a write-once neutral parent for balanced-core distillation."""

    source_checkpoint = Path(source_checkpoint).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if checkpoint.checkpoint_digest(source_checkpoint) != source_digest:
        raise RuntimeError("passed Alakazam source identity changed")
    if output_path.is_file():
        return _validate_transfer_initialization(
            output_path,
            source_checkpoint=source_checkpoint,
            source_digest=source_digest,
        )

    source = checkpoint.load_checkpoint(source_checkpoint, map_location="cpu")
    bank, adapter_state = _fresh_dormant_bank_state()
    source_state = dict(source.get("model_state_dict") or {})
    model_state = {
        name: value
        for name, value in source_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    model_state.update(adapter_state)
    transfer_record = {
        "schema": TRANSFER_INITIALIZATION_SCHEMA,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_digest": source_digest,
        "reusable_state": (
            "card/state embeddings, spatial encoder, temporal encoder, option "
            "decoder, policy/value, and deck-agnostic auxiliary heads"
        ),
        "reset_state": [
            "all matchup-adapter tensors",
            "specialist optimizer/scaler/scheduler moments",
            "specialist RNG state",
            "specialist training counters and extra metadata",
        ],
        "adapter_config": bank.config_dict(),
        "runtime_enabled": False,
    }
    parameter_count = sum(value.numel() for value in adapter_state.values())
    payload = copy.deepcopy(source)
    payload["model_state_dict"] = model_state
    model_config = dict(payload.get("model_config") or {})
    model_config["matchup_adapters_enabled"] = False
    payload["model_config"] = model_config
    payload["archetype_id"] = "unknown"
    payload["model_id"] = "deck_agnostic_core.transfer_initialization"
    payload["step"] = 0
    payload["epoch"] = 0
    payload["rl_iteration"] = 0
    payload["best_metric"] = None
    payload["early_stop_state"] = None
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "rng_state",
    ):
        payload.pop(key, None)
    provenance = dict(payload.get("provenance") or {})
    provenance["core_transfer_initialization"] = copy.deepcopy(transfer_record)
    payload["provenance"] = provenance
    payload["extra"] = {
        "pure_rl": True,
        "core_transfer_initialization": transfer_record,
        "matchup_adapter_config": bank.config_dict(),
        "matchup_adapters_runtime_enabled": False,
        "matchup_adapter_training_enabled": False,
        "matchup_adapter_optimizer_included": False,
        "dormant_matchup_adapter_bank": {
            "schema": ZERO_DORMANT_CHECKPOINT_SCHEMA,
            "materialization": "deck_agnostic_transfer_reset",
            "runtime_enabled": False,
            "training_enabled": False,
            "optimizer_imported": False,
            "optimizer_included": False,
            "frozen": True,
            "zero_output": True,
            "parameter_count": parameter_count,
            "adapter_config": bank.config_dict(),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_digest": source_digest,
        },
    }
    checkpoint.immutable_torch_save(payload, output_path)
    return _validate_transfer_initialization(
        output_path,
        source_checkpoint=source_checkpoint,
        source_digest=source_digest,
    )


def _validate_balanced_manifest(path: Path, *, min_decisions: int) -> dict[str, Any]:
    pointer = json.loads(path.read_text(encoding="utf-8"))
    if (
        pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
    ):
        raise ValueError("core corpus is not protected")
    manifest_raw = Path(str(pointer.get("manifest") or ""))
    manifest_path = (
        manifest_raw.resolve()
        if manifest_raw.is_absolute()
        else (path.parent / manifest_raw).resolve()
    )
    if not manifest_path.is_file() or sha256(manifest_path) != pointer.get(
        "manifest_sha256"
    ):
        raise ValueError("core corpus manifest identity mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = dict(payload.get("selection") or {})
    quality = dict(payload.get("quality_gates") or {})
    totals = dict(payload.get("totals") or {})
    if (
        payload.get("compact_mode") != "temporal-expert-v1"
        or selection.get("operator") != "latest_first_bounded_registered"
        or quality.get("passed") is not True
        or quality.get("checksummed") is not True
        or quality.get("acting_seat_archetype_recognized") is not True
        or quality.get("episode_sequences_unsplit") is not True
        or int(totals.get("decisions_kept", 0)) < int(min_decisions)
        or len(dict(totals.get("records_per_archetype") or {})) < 4
    ):
        raise ValueError("balanced deck-agnostic manifest failed its contract")
    return payload


def _recover_or_train_epoch(
    *,
    epoch: int,
    parent: Path,
    parent_digest: str,
    output: Path,
    corpus: Any,
    manifest_identity: Any,
    device: Any,
    batch_size: int,
    split_seed: int,
    source_digest: str,
    run_name: str,
) -> dict[str, Any]:
    if output.is_file():
        payload = checkpoint.load_checkpoint(output, map_location="cpu")
        record = dict((payload.get("extra") or {}).get("core_distillation") or {})
        if (
            record.get("schema") != "poke_bot.core_distillation_epoch/v1"
            or int(record.get("epoch", -1)) != int(epoch)
            or record.get("source_passed_digest") != source_digest
            or record.get("parent_digest") != parent_digest
            or record.get("manifest_digest") != manifest_identity.digest
        ):
            raise RuntimeError(f"existing epoch checkpoint identity drift: {output}")
        rehearsal = dict((payload.get("extra") or {}).get("expert_rehearsal") or {})
        validation = dict(rehearsal.get("validation_metrics") or {})
        metric = float(validation.get("total_loss", math.inf))
        if not math.isfinite(metric):
            raise RuntimeError(f"existing epoch checkpoint lacks validation: {output}")
        return {
            "candidate_path": str(output),
            "candidate_digest": checkpoint.checkpoint_digest(output),
            "parent_digest": parent_digest,
            "validation_metrics": validation,
            "train_metrics": dict(rehearsal.get("train_metrics") or {}),
            "reused": True,
        }

    return supervised_rehearsal_step(
        corpus,
        base_ckpt=parent,
        output_path=output,
        parent_digest=parent_digest,
        rehearsal_iteration=epoch,
        manifest_identity=manifest_identity.as_dict(),
        epochs=1,
        lr=2e-5,
        requested_batch_size=int(batch_size),
        seed=20260721 + int(epoch),
        corpus_split_seed=int(split_seed),
        device=device,
        aux_loss_weight=0.05,
        opp_hand_loss_weight=0.05,
        opp_remainder_loss_weight=0.05,
        lethal_threat_loss_weight=0.025,
        prize_race_loss_weight=0.025,
        alakazam_guide_loss_weight=0.0,
        output_archetype_id="unknown",
        output_model_id=f"{run_name}.epoch{int(epoch):02d}",
        extra_updates={
            "core_distillation": {
                "schema": "poke_bot.core_distillation_epoch/v1",
                "epoch": int(epoch),
                "source_passed_digest": source_digest,
                "parent_digest": parent_digest,
                "manifest_digest": manifest_identity.digest,
                "objective": "passed_alakazam_to_balanced_deck_agnostic_core",
                "all_auxiliary_heads_trained": True,
            }
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-corpus", type=Path, required=True)
    parser.add_argument("--passed-family", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-decisions", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=12288)
    parser.add_argument("--split-seed", type=int, default=20260721)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    args = parser.parse_args()
    if int(args.epochs) <= 0 or int(args.patience) <= 0:
        raise ValueError("epochs/patience must be positive")

    core_pointer = args.core_corpus.expanduser().resolve()
    core = _validate_balanced_manifest(
        core_pointer, min_decisions=int(args.min_decisions)
    )
    passed = verify_frozen_model(args.passed_family.expanduser().resolve())
    passed_parent = Path(str(passed["model_path"])).resolve()
    model_config = _architecture(passed_parent)
    source_digest = str(passed["checkpoint_digest"])
    manifest_identity = resolve_expert_manifest(
        core_pointer,
        min_decisions=int(args.min_decisions),
        require_protected=True,
        required_compact_mode="temporal-expert-v1",
        required_max_context=int(model_config["max_context"]),
        required_target_coverage=TARGETS,
    )
    run_dir = args.run_dir.expanduser().resolve()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    transfer_initialization = _materialize_transfer_initialization(
        source_checkpoint=passed_parent,
        source_digest=source_digest,
        output_path=run_dir / "deck_agnostic_transfer_initialization.pt",
    )
    source_parent = Path(str(transfer_initialization["path"])).resolve()
    transfer_digest = str(transfer_initialization["digest"])
    transfer_payload = checkpoint.load_checkpoint(source_parent, map_location="cpu")
    belief_card_vocab = belief_card_vocab_from_state(
        dict(transfer_payload.get("model_state_dict") or {})
    )
    family_dir = args.registry_root.expanduser().resolve() / FAMILY
    if args.ready.is_file() and family_dir.is_dir():
        ready = json.loads(args.ready.read_text(encoding="utf-8"))
        frozen = verify_frozen_model(family_dir)
        if (
            ready.get("status") == "ready"
            and ready.get("checkpoint_digest") == frozen.get("checkpoint_digest")
            and ready.get("passed_alakazam_digest") == source_digest
            and ready.get("transfer_initialization_digest") == transfer_digest
            and ready.get("specialist_matchup_adapters_reset") is True
            and ready.get("core_manifest_sha256") == manifest_identity.digest
        ):
            print(json.dumps(ready, indent=2), flush=True)
            return 0
        raise RuntimeError("existing distilled-core readiness identity changed")

    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if state and (
        state.get("source_passed_digest") != source_digest
        or state.get("transfer_initialization_digest") != transfer_digest
        or state.get("manifest_digest") != manifest_identity.digest
    ):
        raise RuntimeError("distillation state identity changed")

    device = device_mod.training_device(prefer_name="RTX PRO 5000", allow_cpu=False)
    cache = ResidentExpertCorpusCache(cpu_pack_root=args.cpu_pack_root)
    print(
        "[core-distill] loading protected balanced corpus "
        f"records={manifest_identity.records} decisions={manifest_identity.decisions} "
        f"archetypes={len(core['totals']['records_per_archetype'])} device={device}",
        flush=True,
    )
    corpus = cache.prepare(
        manifest_identity,
        device=device,
        seed=int(args.split_seed),
        max_context=int(model_config["max_context"]),
        belief_card_vocab=belief_card_vocab,
    )
    if not corpus.has_exact_targets:
        raise RuntimeError(
            "core distillation corpus lost required all-head targets"
        )
    try:
        history = list(state.get("history") or [])
        best_metric = float(state.get("best_metric", math.inf))
        best_path = str(state.get("best_path") or "")
        best_digest = str(state.get("best_digest") or "")
        bad_epochs = int(state.get("bad_epochs", 0))
        parent = source_parent
        parent_digest = transfer_digest
        start_epoch = 1
        if history:
            last = history[-1]
            parent = Path(str(last["checkpoint"])).resolve()
            parent_digest = str(last["checkpoint_digest"])
            if checkpoint.checkpoint_digest(parent) != parent_digest:
                raise RuntimeError("distillation resume parent digest mismatch")
            start_epoch = int(last["epoch"]) + 1

        for epoch in range(start_epoch, int(args.epochs) + 1):
            output = checkpoints / f"epoch_{epoch:02d}.pt"
            result = _recover_or_train_epoch(
                epoch=epoch,
                parent=parent,
                parent_digest=parent_digest,
                output=output,
                corpus=corpus,
                manifest_identity=manifest_identity,
                device=device,
                batch_size=int(args.batch_size),
                split_seed=int(args.split_seed),
                source_digest=source_digest,
                run_name=args.run_name,
            )
            metric = float((result.get("validation_metrics") or {}).get("total_loss", math.inf))
            if not math.isfinite(metric):
                raise RuntimeError("distillation validation metric is not finite")
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
                "schema": "poke_bot.core_distillation_state/v1",
                "status": "training",
                "source_passed_checkpoint": str(passed_parent),
                "source_passed_digest": source_digest,
                "transfer_initialization": str(source_parent),
                "transfer_initialization_digest": transfer_digest,
                "manifest": manifest_identity.as_dict(),
                "manifest_digest": manifest_identity.digest,
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
                f"[core-distill] epoch={epoch} val_loss={metric:.6f} "
                f"best={best_metric:.6f} patience={bad_epochs}/{int(args.patience)}",
                flush=True,
            )
            if bad_epochs >= int(args.patience):
                break
    finally:
        cache.release()

    best = Path(best_path).resolve()
    if not best.is_file() or checkpoint.checkpoint_digest(best) != best_digest:
        raise RuntimeError("selected distilled core identity is invalid")
    model = load_model_from_checkpoint(best, device="cpu")
    validate_zero_dormant_checkpoint(best)
    initialized = checkpoint.load_checkpoint(source_parent, map_location="cpu")
    selected = checkpoint.load_checkpoint(best, map_location="cpu")
    initialized_adapters = {
        name: value
        for name, value in dict(initialized.get("model_state_dict") or {}).items()
        if name.startswith("matchup_adapter_bank.")
    }
    selected_adapters = {
        name: value
        for name, value in dict(selected.get("model_state_dict") or {}).items()
        if name.startswith("matchup_adapter_bank.")
    }
    import torch

    if initialized_adapters.keys() != selected_adapters.keys() or any(
        not torch.equal(initialized_adapters[name], selected_adapters[name])
        for name in initialized_adapters
    ):
        raise RuntimeError("core distillation modified specialist matchup adapters")
    for key, expected in model_config.items():
        if key in {
            "d_model",
            "spatial_layers",
            "temporal_layers",
            "option_decoder_layers",
            "n_heads",
            "ff_dim",
            "max_context",
            "decision_context",
        } and getattr(model.cfg, key) != expected:
            raise RuntimeError(f"distilled core architecture drifted at {key}")
    frozen = freeze_model(
        registry_root=args.registry_root,
        family=FAMILY,
        display_name="Deck-Agnostic Core Distilled from Passed Alakazam",
        checkpoint=best,
        expected_digest=best_digest,
        provenance={
            "distillation_source_family": passed.get("family"),
            "distillation_source_checkpoint": str(passed_parent),
            "distillation_source_digest": source_digest,
            "transfer_initialization_checkpoint": str(source_parent),
            "transfer_initialization_digest": transfer_digest,
            "specialist_matchup_adapters_reset": True,
            "source_gate_evidence": passed.get("evidence"),
            "core_manifest": manifest_identity.as_dict(),
            "objective": "passed_alakazam_to_balanced_deck_agnostic_core",
            "all_auxiliary_heads_trained": True,
            "epochs_max": int(args.epochs),
            "early_stop_patience": int(args.patience),
            "history": history,
        },
        evidence={
            "kind": "episode_disjoint_multideck_expert_validation",
            "best_metric": best_metric,
            "epochs_completed": len(history),
            "archetypes": len(core["totals"]["records_per_archetype"]),
        },
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(frozen["model_path"]).parent)
    ready = {
        "schema": "poke_bot.distilled_deck_agnostic_core_ready/v1",
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "checkpoint": frozen["model_path"],
        "checkpoint_digest": frozen["checkpoint_digest"],
        "passed_alakazam_digest": source_digest,
        "transfer_initialization": str(source_parent),
        "transfer_initialization_digest": transfer_digest,
        "specialist_matchup_adapters_reset": True,
        "core_manifest": manifest_identity.path,
        "core_manifest_sha256": manifest_identity.digest,
        "records": manifest_identity.records,
        "decisions": manifest_identity.decisions,
        "archetypes": len(core["totals"]["records_per_archetype"]),
        "epochs_completed": len(history),
        "epochs_max": int(args.epochs),
        "early_stop_patience": int(args.patience),
        "best_metric": best_metric,
    }
    atomic_json(args.ready, ready)
    atomic_json(state_path, {**state, "status": "complete", "ready": ready})
    print(json.dumps(ready, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
