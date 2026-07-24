"""Durable, adapter-only rehearsal on the causal expert-route corpus.

This transaction is deliberately separate from ordinary expert rehearsal.
Only the matchup adapter bank receives gradients, the current learner is
preserved as an immutable parent, and a crash can resume from the streaming
trainer cursor without repeating verified optimizer steps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from poke_bot import checkpoint
from poke_bot.matchup_adapter_activation import merge_dormant_adapter_checkpoint
from poke_bot.pure_rl.matchup_adapter_corpus import sha256_file
from poke_bot.pure_rl.matchup_adapter_trainer import (
    StreamingAdapterTrainConfig,
    load_staged_training_contract,
    train_matchup_adapters_streaming,
)


RECEIPT_SCHEMA = "poke_bot.expert_matchup_adapter_rehearsal/v1"


@dataclass(frozen=True)
class ExpertAdapterRehearsalPaths:
    authorization: Path
    fit_dir: Path
    checkpoint: Path
    receipt: Path


def rehearsal_paths(
    run_dir: Path, before_iteration: int
) -> ExpertAdapterRehearsalPaths:
    root = Path(run_dir).expanduser().resolve() / "rehearsals" / "matchup_adapters"
    stem = f"before_iter_{int(before_iteration):05d}"
    return ExpertAdapterRehearsalPaths(
        authorization=root / f"{stem}.authorization.json",
        fit_dir=root / f"{stem}.fit",
        checkpoint=root / f"{stem}.pt",
        receipt=root / f"{stem}.json",
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(resolved, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    before_iteration: int,
    parent_checkpoint: Path,
    parent_digest: str,
    authorization: Path,
    staged_manifest: Path,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    staged = load_staged_training_contract(staged_manifest)
    checkpoint_path = Path(
        str(receipt.get("checkpoint") or "")
    ).expanduser().resolve()
    fit_checkpoint = Path(
        str(receipt.get("fit_checkpoint") or "")
    ).expanduser().resolve()
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or int(receipt.get("before_iteration", -1)) != int(before_iteration)
        or Path(str(receipt.get("parent_checkpoint") or "")).expanduser().resolve()
        != parent_checkpoint
        or str(receipt.get("parent_digest") or "") != str(parent_digest)
        or Path(str(receipt.get("authorization") or "")).expanduser().resolve()
        != authorization
        or str(receipt.get("authorization_digest") or "")
        != sha256_file(authorization)
        or Path(str(receipt.get("staged_manifest") or "")).expanduser().resolve()
        != staged.manifest_path
        or str(receipt.get("staged_manifest_digest") or "")
        != staged.manifest_file_digest
        or int(receipt.get("epochs", -1)) != int(epochs)
        or float(receipt.get("learning_rate", -1.0)) != float(learning_rate)
        or receipt.get("optimizer_scope") != "matchup_adapter_bank_only"
        or receipt.get("base_frozen") is not True
        or receipt.get("runtime_enabled") is not False
        or not checkpoint_path.is_file()
        or not fit_checkpoint.is_file()
        or str(receipt.get("checkpoint_digest") or "")
        != checkpoint.checkpoint_digest(checkpoint_path)
        or str(receipt.get("fit_checkpoint_digest") or "")
        != checkpoint.checkpoint_digest(fit_checkpoint)
    ):
        raise RuntimeError("expert adapter rehearsal receipt contract mismatch")
    payload = checkpoint.load_checkpoint(checkpoint_path, map_location="cpu")
    extra = dict(payload.get("extra") or {})
    fit = dict(extra.get("dormant_matchup_adapter_fit") or {})
    if (
        bool(
            dict(payload.get("model_config") or {}).get(
                "matchup_adapters_enabled", False
            )
        )
        or extra.get("matchup_adapters_runtime_enabled", False) is not False
        or fit.get("base_frozen") is not True
        or fit.get("optimizer_scope") != "matchup_adapter_bank_only"
        or int(fit.get("phase_epochs", -1)) != int(epochs)
        or int(fit.get("phase_rows", 0)) <= 0
        or int(fit.get("phase_steps", 0)) <= 0
        or fit.get("optimizer_included") is not True
        or not isinstance(
            extra.get("dormant_matchup_adapter_optimizer_state"), dict
        )
    ):
        raise RuntimeError("expert adapter rehearsal checkpoint is not isolated")
    fit_summary = {
        "phase_epochs": int(fit["phase_epochs"]),
        "phase_steps": int(fit["phase_steps"]),
        "phase_rows": int(fit["phase_rows"]),
        "cumulative_epochs": int(fit.get("epochs") or 0),
        "cumulative_steps": int(fit.get("steps") or 0),
        "cumulative_rows": int(fit.get("rows") or 0),
        "trained_archetype_ids": list(fit.get("trained_archetype_ids") or []),
        "dormant_no_example_archetype_ids": list(
            fit.get("dormant_no_example_archetype_ids") or []
        ),
    }
    persisted_fit = receipt.get("fit")
    if persisted_fit is not None and dict(persisted_fit) != fit_summary:
        raise RuntimeError("expert adapter rehearsal fit receipt mismatch")
    return {
        **dict(receipt),
        "reused": True,
        "fit": fit_summary,
    }


def run_or_recover_expert_adapter_rehearsal(
    *,
    run_dir: Path,
    before_iteration: int,
    parent_checkpoint: Path,
    parent_digest: str,
    staged_manifest: Path,
    epochs: int,
    learning_rate: float,
    games_per_batch: int,
    max_decisions_per_batch: int,
    seed: int,
    device: torch.device,
    weight_decay: float = 1e-4,
    value_loss_weight: float = 1.0,
    grad_clip: float = 1.0,
    max_process_rss_gib: float = 16.0,
    min_available_ram_gib: float = 16.0,
) -> dict[str, Any]:
    """Resume or commit one exact adapter-only expert rehearsal transaction."""

    parent = Path(parent_checkpoint).expanduser().resolve()
    if checkpoint.checkpoint_digest(parent) != str(parent_digest):
        raise RuntimeError("expert adapter rehearsal parent digest mismatch")
    if int(epochs) <= 0:
        raise ValueError("expert adapter rehearsal epochs must be positive")
    paths = rehearsal_paths(run_dir, before_iteration)
    staged = load_staged_training_contract(staged_manifest)
    if paths.receipt.is_file():
        return _validate_receipt(
            json.loads(paths.receipt.read_text(encoding="utf-8")),
            before_iteration=before_iteration,
            parent_checkpoint=parent,
            parent_digest=parent_digest,
            authorization=paths.authorization,
            staged_manifest=staged.manifest_path,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    if not paths.authorization.is_file():
        raise RuntimeError(
            "expert adapter rehearsal authorization was not staged at the "
            "preceding clean commit boundary"
        )

    cfg = StreamingAdapterTrainConfig(
        epochs=int(epochs),
        games_per_batch=int(games_per_batch),
        max_decisions_per_batch=int(max_decisions_per_batch),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        value_loss_weight=float(value_loss_weight),
        grad_clip=float(grad_clip),
        amp=device.type == "cuda",
        seed=int(seed),
        early_stop_patience=max(1, int(epochs)),
        exact_epochs=True,
        max_process_rss_gib=float(max_process_rss_gib),
        min_available_ram_gib=float(min_available_ram_gib),
    )
    result = train_matchup_adapters_streaming(
        staged_manifest=staged.manifest_path,
        parent_checkpoint=parent,
        activation_receipt=paths.authorization,
        output_dir=paths.fit_dir,
        train_config=cfg,
        device=device,
        resume="auto",
        run_name=f"expert_matchup_adapters_before_iter_{int(before_iteration):05d}",
        permit_post_boundary_use=True,
        restore_parent_optimizer_state=True,
    )
    fit_checkpoint = Path(
        str(result.get("final_path") or result.get("latest_path") or "")
    ).expanduser().resolve()
    if not fit_checkpoint.is_file():
        raise RuntimeError("expert adapter rehearsal did not produce a checkpoint")
    if not paths.checkpoint.is_file():
        merge_dormant_adapter_checkpoint(
            parent_checkpoint=parent,
            adapter_checkpoint=fit_checkpoint,
            activation_receipt=paths.authorization,
            output_path=paths.checkpoint,
            permit_post_boundary_use=True,
            import_optimizer_state=True,
            accumulate_parent_fit=True,
        )
    merged_payload = checkpoint.load_checkpoint(paths.checkpoint, map_location="cpu")
    merged_fit = dict(
        (merged_payload.get("extra") or {}).get(
            "dormant_matchup_adapter_fit"
        )
        or {}
    )
    fit_summary = {
        "phase_epochs": int(merged_fit.get("phase_epochs") or 0),
        "phase_steps": int(merged_fit.get("phase_steps") or 0),
        "phase_rows": int(merged_fit.get("phase_rows") or 0),
        "cumulative_epochs": int(merged_fit.get("epochs") or 0),
        "cumulative_steps": int(merged_fit.get("steps") or 0),
        "cumulative_rows": int(merged_fit.get("rows") or 0),
        "trained_archetype_ids": list(
            merged_fit.get("trained_archetype_ids") or []
        ),
        "dormant_no_example_archetype_ids": list(
            merged_fit.get("dormant_no_example_archetype_ids") or []
        ),
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "before_iteration": int(before_iteration),
        "parent_checkpoint": str(parent),
        "parent_digest": str(parent_digest),
        "authorization": str(paths.authorization),
        "authorization_digest": sha256_file(paths.authorization),
        "staged_manifest": str(staged.manifest_path),
        "staged_manifest_digest": staged.manifest_file_digest,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "games_per_batch": int(games_per_batch),
        "max_decisions_per_batch": int(max_decisions_per_batch),
        "optimizer_scope": "matchup_adapter_bank_only",
        "base_frozen": True,
        "runtime_enabled": False,
        "fit_checkpoint": str(fit_checkpoint),
        "fit_checkpoint_digest": checkpoint.checkpoint_digest(fit_checkpoint),
        "checkpoint": str(paths.checkpoint),
        "checkpoint_digest": checkpoint.checkpoint_digest(paths.checkpoint),
        "fit": fit_summary,
    }
    _write_json_exclusive(paths.receipt, receipt)
    return _validate_receipt(
        receipt,
        before_iteration=before_iteration,
        parent_checkpoint=parent,
        parent_digest=parent_digest,
        authorization=paths.authorization,
        staged_manifest=staged.manifest_path,
        epochs=epochs,
        learning_rate=learning_rate,
    )


__all__ = [
    "ExpertAdapterRehearsalPaths",
    "RECEIPT_SCHEMA",
    "rehearsal_paths",
    "run_or_recover_expert_adapter_rehearsal",
]
