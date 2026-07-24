#!/usr/bin/env python3
"""Build a protected deck-agnostic core from all frozen specialist teachers.

The selected teacher checkpoints contribute equally to a neutral, parameter-space
student initialization. Specialist matchup adapters are never averaged: a
fresh canonical zero-output bank replaces them before any supervised update.
The balanced expert corpus then trains every shared policy/value/auxiliary
head in immutable one-epoch steps with episode-disjoint validation and early
stopping.

This script only builds and freezes the candidate core. A separate handoff
coordinator must run the established gameplay regression gates against every
teacher before the core can initialize another specialist.
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

from poke_bot import archetypes, checkpoint, device as device_mod  # noqa: E402
from poke_bot.dormant_adapter_compat import (  # noqa: E402
    validate_zero_dormant_checkpoint,
)
from poke_bot.matchup_adapters import ZERO_DORMANT_CHECKPOINT_SCHEMA  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_registry import (  # noqa: E402
    freeze_model,
    verify_frozen_model,
)
from poke_bot.train import (  # noqa: E402
    belief_card_vocab_from_state,
    load_model_from_checkpoint,
)
from scripts.run_passed_alakazam_core_distillation import (  # noqa: E402
    TARGETS,
    _architecture,
    _fresh_dormant_bank_state,
    _recover_or_train_epoch,
    _validate_balanced_manifest,
    atomic_json,
)


DEFAULT_FAMILY = (
    "deck_agnostic_core_distilled_from_hops_trevenant_and_starmie_v2"
)
INITIALIZATION_SCHEMA = "poke_bot.multi_teacher_core_initialization/v1"
STATE_SCHEMA = "poke_bot.multi_teacher_core_refresh_state/v1"
READY_SCHEMA = "poke_bot.multi_teacher_core_ready/v1"


def _teacher_identity(family: Path) -> dict[str, Any]:
    frozen = verify_frozen_model(family.expanduser().resolve())
    model_path = Path(str(frozen["model_path"])).resolve()
    _architecture(model_path)
    return {
        "family": str(frozen["family"]),
        "family_path": str(model_path.parent),
        "checkpoint": str(model_path),
        "checkpoint_digest": str(frozen["checkpoint_digest"]),
        "manifest": str(model_path.parent / "manifest.json"),
    }


def _base_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in dict(payload.get("model_state_dict") or {}).items()
        if not name.startswith("matchup_adapter_bank.")
    }


def _align_teacher_state(
    payload: dict[str, Any],
    *,
    target_state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Align a frozen legacy auxiliary head to the canonical row registry.

    Shared tensors remain exact frozen-teacher tensors. A legacy auxiliary
    head contributes its named historical rows and unknown row; rows for
    archetypes that did not exist in that checkpoint retain the accepted
    initialization-core values instead of receiving random or fabricated
    teacher values.
    """

    import torch

    source = _base_state(payload)
    if source.keys() != target_state.keys():
        raise RuntimeError("teacher reusable tensor keys differ")
    aligned: dict[str, Any] = {}
    adaptations: list[dict[str, Any]] = []
    current_ids = list(archetypes.archetype_ids())
    pinned_ids = list(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS)
    for name, value in source.items():
        target = target_state[name]
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise RuntimeError(f"teacher state is not tensor-valued: {name}")
        if value.shape == target.shape:
            aligned[name] = value
            continue
        if (
            name in {"aux_head.3.weight", "aux_head.3.bias"}
            and int(value.shape[0]) == len(pinned_ids) + 1
            and int(target.shape[0]) == len(current_ids) + 1
            and value.shape[1:] == target.shape[1:]
        ):
            expanded = target.detach().cpu().clone()
            for old_index, archetype_id in enumerate(pinned_ids):
                expanded[current_ids.index(archetype_id)].copy_(value[old_index])
            expanded[-1].copy_(value[-1])
            aligned[name] = expanded
            adaptations.append(
                {
                    "tensor": name,
                    "source_rows": len(pinned_ids) + 1,
                    "target_rows": len(current_ids) + 1,
                    "mapped_rows": pinned_ids + ["unknown"],
                    "retained_initialization_rows": [
                        archetype_id
                        for archetype_id in current_ids
                        if archetype_id not in pinned_ids
                    ],
                }
            )
            continue
        raise RuntimeError(f"teacher tensor contract differs: {name}")
    return aligned, adaptations


def _validate_initialization(
    path: Path,
    *,
    initialization_digest: str,
    teachers: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    extra = dict(payload.get("extra") or {})
    record = dict(extra.get("multi_teacher_core_initialization") or {})
    expected_teachers = [
        {
            "family": row["family"],
            "checkpoint": row["checkpoint"],
            "checkpoint_digest": row["checkpoint_digest"],
            "weight": 1.0 / len(teachers),
        }
        for row in teachers
    ]
    if (
        record.get("schema") != INITIALIZATION_SCHEMA
        or record.get("initialization_core_digest") != initialization_digest
        or record.get("teachers") != expected_teachers
        or record.get("averaging") != "equal_weight_parameter_space_mean"
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
        raise RuntimeError("multi-teacher initialization metadata drifted")

    initialization_path = Path(
        str(record.get("initialization_core_checkpoint") or "")
    ).expanduser().resolve()
    if (
        not initialization_path.is_file()
        or checkpoint.checkpoint_digest(initialization_path)
        != initialization_digest
    ):
        raise RuntimeError("initialization core identity changed")
    target_state = _base_state(
        checkpoint.load_checkpoint(initialization_path, map_location="cpu")
    )
    teacher_payloads = [
        checkpoint.load_checkpoint(row["checkpoint"], map_location="cpu")
        for row in teachers
    ]
    aligned = [
        _align_teacher_state(value, target_state=target_state)
        for value in teacher_payloads
    ]
    states = [value[0] for value in aligned]
    expected_adaptations = [
        {
            "checkpoint_digest": teachers[index]["checkpoint_digest"],
            "tensors": adaptations,
        }
        for index, (_state, adaptations) in enumerate(aligned)
        if adaptations
    ]
    if record.get("architecture_alignment") != expected_adaptations:
        raise RuntimeError("multi-teacher architecture alignment metadata drifted")
    actual = _base_state(payload)
    if not states or any(state.keys() != states[0].keys() for state in states):
        raise RuntimeError("teacher reusable tensor keys differ")
    if actual.keys() != states[0].keys():
        raise RuntimeError("student reusable tensor keys differ from teachers")
    for name in actual:
        values = [state[name] for state in states]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise RuntimeError(f"teacher state is not tensor-valued: {name}")
        if any(value.shape != values[0].shape for value in values):
            raise RuntimeError(f"teacher tensor shape differs: {name}")
        if values[0].is_floating_point() or values[0].is_complex():
            accumulator_dtype = (
                torch.complex128 if values[0].is_complex() else torch.float64
            )
            expected = torch.stack(
                [value.to(accumulator_dtype) for value in values]
            ).mean(dim=0).to(values[0].dtype)
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise RuntimeError(
                    f"non-floating teacher state differs and cannot be averaged: {name}"
                )
            expected = values[0]
        if not torch.equal(actual[name], expected):
            raise RuntimeError(f"student is not the exact teacher mean: {name}")
    validate_zero_dormant_checkpoint(path)
    return {
        "path": str(path),
        "digest": checkpoint.checkpoint_digest(path),
        "reusable_tensor_count": len(actual),
        "teacher_count": len(teachers),
    }


def materialize_initialization(
    *,
    initialization_family: Path,
    teacher_families: list[Path],
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create or validate the exact neutral equal-teacher initialization."""

    import torch

    initialization = verify_frozen_model(initialization_family.expanduser().resolve())
    initialization_path = Path(str(initialization["model_path"])).resolve()
    initialization_digest = str(initialization["checkpoint_digest"])
    _architecture(initialization_path)
    teachers = [_teacher_identity(path) for path in teacher_families]
    if len(teachers) < 2:
        raise ValueError("multi-teacher refresh requires at least two teachers")
    if len({row["checkpoint_digest"] for row in teachers}) != len(teachers):
        raise ValueError("multi-teacher refresh requires distinct teacher checkpoints")
    output = output.expanduser().resolve()
    if output.is_file():
        return (
            _validate_initialization(
                output,
                initialization_digest=initialization_digest,
                teachers=teachers,
            ),
            teachers,
        )

    template = checkpoint.load_checkpoint(initialization_path, map_location="cpu")
    teacher_payloads = [
        checkpoint.load_checkpoint(row["checkpoint"], map_location="cpu")
        for row in teachers
    ]
    target_state = _base_state(template)
    aligned = [
        _align_teacher_state(value, target_state=target_state)
        for value in teacher_payloads
    ]
    states = [value[0] for value in aligned]
    averaged: dict[str, Any] = {}
    for name in states[0]:
        values = [state[name] for state in states]
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != values[0].shape
            for value in values
        ):
            raise RuntimeError(f"teacher tensor contract differs: {name}")
        if values[0].is_floating_point() or values[0].is_complex():
            accumulator_dtype = (
                torch.complex128 if values[0].is_complex() else torch.float64
            )
            averaged[name] = (
                torch.stack([value.to(accumulator_dtype) for value in values])
                .mean(dim=0)
                .to(values[0].dtype)
            )
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise RuntimeError(
                    f"non-floating teacher state differs and cannot be averaged: {name}"
                )
            averaged[name] = values[0].detach().cpu().clone()

    bank, adapter_state = _fresh_dormant_bank_state()
    averaged.update(adapter_state)
    teacher_rows = [
        {
            "family": row["family"],
            "checkpoint": row["checkpoint"],
            "checkpoint_digest": row["checkpoint_digest"],
            "weight": 1.0 / len(teachers),
        }
        for row in teachers
    ]
    record = {
        "schema": INITIALIZATION_SCHEMA,
        "initialization_core_family": str(initialization["family"]),
        "initialization_core_checkpoint": str(initialization_path),
        "initialization_core_digest": initialization_digest,
        "teachers": teacher_rows,
        "averaging": "equal_weight_parameter_space_mean",
        "architecture_alignment": [
            {
                "checkpoint_digest": teachers[index]["checkpoint_digest"],
                "tensors": adaptations,
            }
            for index, (_state, adaptations) in enumerate(aligned)
            if adaptations
        ],
        "teacher_gradients_allowed": False,
        "teacher_optimizer_steps_allowed": False,
        "specialist_matchup_adapters_reset": True,
        "runtime_enabled": False,
    }
    payload = copy.deepcopy(template)
    payload["model_state_dict"] = averaged
    payload["archetype_id"] = "unknown"
    payload["model_id"] = "deck_agnostic_core.multi_teacher_initialization"
    payload["step"] = 0
    payload["epoch"] = 0
    payload["rl_iteration"] = 0
    payload["best_metric"] = None
    payload["early_stop_state"] = None
    model_config = dict(payload.get("model_config") or {})
    model_config["matchup_adapters_enabled"] = False
    payload["model_config"] = model_config
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "rng_state",
    ):
        payload.pop(key, None)
    payload["provenance"] = {
        "multi_teacher_core_initialization": copy.deepcopy(record)
    }
    payload["extra"] = {
        "pure_rl": True,
        "multi_teacher_core_initialization": record,
        "matchup_adapter_config": bank.config_dict(),
        "matchup_adapters_runtime_enabled": False,
        "matchup_adapter_training_enabled": False,
        "matchup_adapter_optimizer_included": False,
        "dormant_matchup_adapter_bank": {
            "schema": ZERO_DORMANT_CHECKPOINT_SCHEMA,
            "materialization": "multi_teacher_core_refresh_reset",
            "runtime_enabled": False,
            "training_enabled": False,
            "optimizer_imported": False,
            "optimizer_included": False,
            "frozen": True,
            "zero_output": True,
            "parameter_count": sum(value.numel() for value in adapter_state.values()),
            "adapter_config": bank.config_dict(),
            "teacher_checkpoint_digests": [
                row["checkpoint_digest"] for row in teachers
            ],
        },
    }
    checkpoint.immutable_torch_save(payload, output)
    return (
        _validate_initialization(
            output,
            initialization_digest=initialization_digest,
            teachers=teachers,
        ),
        teachers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-corpus", type=Path, required=True)
    parser.add_argument("--initialization-family", type=Path, required=True)
    parser.add_argument(
        "--teacher-family", type=Path, action="append", required=True
    )
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--output-family", default=DEFAULT_FAMILY)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-decisions", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=12288)
    parser.add_argument("--split-seed", type=int, default=20260723)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    args = parser.parse_args()
    if int(args.epochs) <= 0 or int(args.patience) <= 0:
        raise ValueError("epochs/patience must be positive")
    family_name = str(args.output_family).strip()
    if not family_name or "/" in family_name or family_name in {".", ".."}:
        raise ValueError("output family must be one safe registry basename")

    core_pointer = args.core_corpus.expanduser().resolve()
    core_payload = _validate_balanced_manifest(
        core_pointer, min_decisions=int(args.min_decisions)
    )
    core_pointer_payload = json.loads(core_pointer.read_text(encoding="utf-8"))
    core_manifest_digest = str(
        core_pointer_payload.get("manifest_sha256") or ""
    )
    run_dir = args.run_dir.expanduser().resolve()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    family_dir = args.registry_root.expanduser().resolve() / family_name
    if args.ready.is_file() and family_dir.is_dir():
        # A completed refresh is an immutable, versioned artifact.  Resume it
        # from its checksums before consulting the current archetype registry:
        # later staged registry additions must not reinterpret or rebuild an
        # already-frozen core under a different auxiliary-head shape.
        ready = json.loads(args.ready.read_text(encoding="utf-8"))
        frozen = verify_frozen_model(family_dir)
        teachers = [_teacher_identity(path) for path in args.teacher_family]
        initialization_path = Path(
            str(ready.get("initialization") or "")
        ).expanduser().resolve()
        status = str(ready.get("status") or "")
        resumable_status = (
            status == "candidate_ready_for_gameplay_regression"
            or (
                status == "ready"
                and ready.get("gameplay_regression_passed") is True
            )
        )
        if (
            ready.get("schema") == READY_SCHEMA
            and resumable_status
            and ready.get("family") == family_name
            and ready.get("run_name") == args.run_name
            and ready.get("checkpoint_digest") == frozen.get("checkpoint_digest")
            and ready.get("checkpoint") == frozen.get("model_path")
            and initialization_path.is_file()
            and ready.get("initialization_digest")
            == checkpoint.checkpoint_digest(initialization_path)
            and ready.get("teacher_checkpoint_digests")
            == [row["checkpoint_digest"] for row in teachers]
            and ready.get("core_manifest_sha256") == core_manifest_digest
            and int(ready.get("epochs_max") or 0) == int(args.epochs)
            and int(ready.get("early_stop_patience") or 0)
            == int(args.patience)
            and int(ready.get("decisions") or 0) >= int(args.min_decisions)
            and ready.get("specialist_matchup_adapters_reset") is True
        ):
            print(json.dumps(ready, indent=2), flush=True)
            return 0
        raise RuntimeError("existing refreshed-core readiness identity changed")

    initialization, teachers = materialize_initialization(
        initialization_family=args.initialization_family,
        teacher_families=list(args.teacher_family),
        output=run_dir / "multi_teacher_initialization.pt",
    )
    source_parent = Path(initialization["path"]).resolve()
    source_digest = str(initialization["digest"])
    model_config = _architecture(source_parent)
    manifest_identity = resolve_expert_manifest(
        core_pointer,
        min_decisions=int(args.min_decisions),
        require_protected=True,
        required_compact_mode="temporal-expert-v1",
        required_max_context=int(model_config["max_context"]),
        required_target_coverage=TARGETS,
    )
    transfer_payload = checkpoint.load_checkpoint(source_parent, map_location="cpu")
    belief_card_vocab = belief_card_vocab_from_state(
        dict(transfer_payload.get("model_state_dict") or {})
    )

    state_path = run_dir / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    teacher_digests = [row["checkpoint_digest"] for row in teachers]
    if state and (
        state.get("initialization_digest") != source_digest
        or state.get("teacher_checkpoint_digests") != teacher_digests
        or state.get("manifest_digest") != manifest_identity.digest
    ):
        raise RuntimeError("multi-teacher core refresh state identity changed")

    device = device_mod.training_device(prefer_name="RTX PRO 5000", allow_cpu=False)
    cache = ResidentExpertCorpusCache(cpu_pack_root=args.cpu_pack_root)
    print(
        "[core-refresh] loading protected balanced corpus "
        f"records={manifest_identity.records} decisions={manifest_identity.decisions} "
        f"archetypes={len(core_payload['totals']['records_per_archetype'])} "
        f"teachers={len(teachers)} device={device}",
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
        raise RuntimeError("core refresh corpus lost required all-head targets")
    try:
        history = list(state.get("history") or [])
        best_metric = float(state.get("best_metric", math.inf))
        best_path = str(state.get("best_path") or "")
        best_digest = str(state.get("best_digest") or "")
        bad_epochs = int(state.get("bad_epochs", 0))
        parent = source_parent
        parent_digest = source_digest
        start_epoch = 1
        if history:
            last = history[-1]
            parent = Path(str(last["checkpoint"])).resolve()
            parent_digest = str(last["checkpoint_digest"])
            if checkpoint.checkpoint_digest(parent) != parent_digest:
                raise RuntimeError("core refresh resume parent digest mismatch")
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
            metric = float(
                (result.get("validation_metrics") or {}).get(
                    "total_loss", math.inf
                )
            )
            if not math.isfinite(metric):
                raise RuntimeError("core refresh validation metric is not finite")
            row = {
                "epoch": epoch,
                "checkpoint": str(output),
                "checkpoint_digest": str(result["candidate_digest"]),
                "parent_digest": parent_digest,
                "validation_loss": metric,
                "validation_accuracy": float(
                    (result.get("validation_metrics") or {}).get(
                        "policy_acc", 0.0
                    )
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
                "schema": STATE_SCHEMA,
                "status": "training",
                "initialization": str(source_parent),
                "initialization_digest": source_digest,
                "teacher_checkpoint_digests": teacher_digests,
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
                f"[core-refresh] epoch={epoch} val_loss={metric:.6f} "
                f"best={best_metric:.6f} patience={bad_epochs}/{int(args.patience)}",
                flush=True,
            )
            if bad_epochs >= int(args.patience):
                break
    finally:
        cache.release()

    best = Path(best_path).resolve()
    if not best.is_file() or checkpoint.checkpoint_digest(best) != best_digest:
        raise RuntimeError("selected refreshed core identity is invalid")
    validate_zero_dormant_checkpoint(best)
    load_model_from_checkpoint(best, device="cpu")
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
        raise RuntimeError("core refresh modified specialist matchup adapters")
    frozen = freeze_model(
        registry_root=args.registry_root,
        family=family_name,
        display_name=(
            f"Deck-Agnostic Cumulative Core from {len(teachers)} Frozen Specialists"
        ),
        checkpoint=best,
        expected_digest=best_digest,
        provenance={
            "initialization_checkpoint": str(source_parent),
            "initialization_digest": source_digest,
            "teachers": teachers,
            "teacher_contribution": "equal_weight_parameter_space_mean",
            "teacher_gradients_allowed": False,
            "teacher_optimizer_steps_allowed": False,
            "specialist_matchup_adapters_reset": True,
            "core_manifest": manifest_identity.as_dict(),
            "objective": "multi_teacher_balanced_deck_agnostic_core_refresh",
            "all_auxiliary_heads_trained": True,
            "epochs_max": int(args.epochs),
            "early_stop_patience": int(args.patience),
            "history": history,
        },
        evidence={
            "kind": "episode_disjoint_multideck_expert_validation",
            "best_metric": best_metric,
            "epochs_completed": len(history),
            "archetypes": len(core_payload["totals"]["records_per_archetype"]),
            "gameplay_regression_status": "required_before_specialist_handoff",
        },
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(frozen["model_path"]).parent)
    ready = {
        "schema": READY_SCHEMA,
        "status": "candidate_ready_for_gameplay_regression",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "family": family_name,
        "checkpoint": frozen["model_path"],
        "checkpoint_digest": frozen["checkpoint_digest"],
        "initialization": str(source_parent),
        "initialization_digest": source_digest,
        "teacher_checkpoint_digests": teacher_digests,
        "teacher_contribution": "equal_weight_parameter_space_mean",
        "specialist_matchup_adapters_reset": True,
        "core_manifest": manifest_identity.path,
        "core_manifest_sha256": manifest_identity.digest,
        "records": manifest_identity.records,
        "decisions": manifest_identity.decisions,
        "archetypes": len(core_payload["totals"]["records_per_archetype"]),
        "epochs_completed": len(history),
        "epochs_max": int(args.epochs),
        "early_stop_patience": int(args.patience),
        "best_metric": best_metric,
        "gameplay_regression_required": True,
        "gameplay_regression_passed": False,
    }
    atomic_json(args.ready, ready)
    atomic_json(state_path, {**state, "status": "candidate_ready", "ready": ready})
    print(json.dumps(ready, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
