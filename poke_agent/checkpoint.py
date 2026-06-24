from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from poke_agent.dataset import TrainingTensors
from poke_agent.features import STRUCTURED_FEATURE_DIM
from poke_agent.models.temporal_transformer import TemporalTransformer
from poke_agent.training_diversity import assert_checkpoint_has_no_deck


def load_competition_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_torch_save(obj: Any, output_path: Path) -> None:
    """Write a checkpoint to a temp file then atomically replace the target.

    A crash mid-write leaves the previous checkpoint intact instead of a
    half-written (corrupt) file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, output_path)


def training_checkpoint_paths(output_path: Path) -> dict[str, Path]:
    """Sibling paths for periodic/best checkpoints derived from the final path.

    e.g. outputs/checkpoints/temporal_current.pt ->
         {latest: .../temporal_current.latest.pt,
          best:   .../temporal_current.best.pt}
    """
    base = output_path.with_suffix("")
    return {
        "latest": base.with_name(base.name + ".latest.pt"),
        "best": base.with_name(base.name + ".best.pt"),
    }


def resolve_resume_path(output_path: Path, mode: str | bool | None) -> Path | None:
    """Resolve which checkpoint (if any) to resume the bootstrap run from.

    mode: "0"/False = never resume; "auto"/None = resume latest if present;
    "1"/True = require latest (error if missing).
    """
    latest = training_checkpoint_paths(output_path)["latest"]
    if isinstance(mode, str):
        normalized = mode.strip().lower()
    elif mode is None:
        normalized = "auto"
    else:
        normalized = "1" if mode else "0"

    if normalized in {"0", "false", "no", "off", "none"}:
        return None
    if normalized in {"1", "true", "yes", "on", "require", "force"}:
        if not latest.exists():
            raise FileNotFoundError(
                f"TRAIN_RESUME requested but no resume checkpoint at {latest}. "
                "Start a fresh run with TRAIN_RESUME=0."
            )
        return latest
    # "auto" (default): resume only if a latest checkpoint exists.
    return latest if latest.exists() else None


def load_training_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load a checkpoint and return (full_checkpoint, training_state | None)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, checkpoint.get("training_state")


def save_checkpoint(
    *,
    model: TemporalTransformer,
    tensors: TrainingTensors,
    config: dict[str, Any],
    training_report: dict[str, Any],
    output_path: Path,
    training_state: dict[str, Any] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    competition_results_path = config["competition_results_path"]
    competition_results = load_competition_results(competition_results_path)
    latest_competition_result = competition_results[0] if competition_results else None
    training_report = {
        **training_report,
        "competition_results_path": str(competition_results_path),
        "latest_competition_result": latest_competition_result,
        "checkpoint_path": str(output_path),
        "report_path": str(config.get("report_path")) if config.get("report_path") else None,
    }

    model_cfg = config["model"]
    loss_cfg = config["loss"]
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_id": config.get("model_id"),
        "model_description": config.get("model_description"),
        "architecture": config.get("architecture", "transformer_rl"),
        "model_type": config.get("model_type", "temporal_transformer_rl_complex_loss"),
        "input_dim": int(tensors.x_seq.shape[-1]),
        "coarse_feature_dim": int(config.get("coarse_feature_dim", STRUCTURED_FEATURE_DIM)),
        "policy_dim": tensors.transition_classes,
        "model_config": {
            "d_model": model_cfg["d_model"],
            "heads": model_cfg["heads"],
            "layers": model_cfg["layers"],
            "dim_feedforward": model_cfg["ff"],
            "dropout": model_cfg["dropout"],
            "window_size": tensors.window_size,
            "card_vocab_size": int(config.get("card_vocab_size", 2000)),
            "card_embed_dim": int(config.get("card_embed_dim", 32)),
        },
        "feature_mean": tensors.feature_mean.astype("float32").tolist(),
        "feature_std": tensors.feature_std.astype("float32").tolist(),
        "loss_weights": dict(loss_cfg),
        "rewards": dict(config.get("rewards", {})),
        "beam_search": dict(config.get("beam_search", {})),
        "training_report": training_report,
        "device_used": training_report["device"],
        "data_path": training_report["data_path"],
    }
    if training_state is not None:
        checkpoint["training_state"] = training_state
    assert_checkpoint_has_no_deck(checkpoint)

    _atomic_torch_save(checkpoint, output_path)

    report_file = config.get("report_path")
    if write_report and report_file is not None:
        report_file = Path(report_file)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w", encoding="utf-8") as handle:
            json.dump(training_report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    return training_report


def save_training_checkpoint(
    *,
    model: TemporalTransformer,
    tensors: TrainingTensors,
    config: dict[str, Any],
    training_report: dict[str, Any],
    output_path: Path,
    training_state: dict[str, Any],
) -> None:
    """Write an in-progress checkpoint (model + optimizer/loop state) atomically.

    Used for periodic/best saves during ``train_model``; does not overwrite the
    final JSON report.
    """
    save_checkpoint(
        model=model,
        tensors=tensors,
        config=config,
        training_report=training_report,
        output_path=output_path,
        training_state=training_state,
        write_report=False,
    )


def print_training_report(training_report: dict[str, Any], output_path: Path) -> None:
    print("saved checkpoint", output_path)
    report_file = training_report.get("report_path")
    if report_file:
        print("saved report", report_file)
    print("\nFinal training report")
    print("-" * 22)
    print(f"rows: {training_report['dataset_rows']}")
    print(f"window: {training_report['window_size']} batch_games: {training_report['batch_games']}")
    print(f"device: {training_report['device']}")
    print(f"epochs: {training_report['completed_epochs']} / {training_report['requested_epochs']}")
    print(f"early stopped: {training_report['stopped_early']}")
    print(f"best total loss: {training_report['best_total_loss']:.5f} @ epoch {training_report['best_epoch']}")
    for name, value in training_report["last_metrics"].items():
        print(f"{name}: {value:.5f}")

    latest_competition_result = training_report.get("latest_competition_result")
    if latest_competition_result:
        print("\nLatest official Kaggle result")
        print("-" * 29)
        print(f"file: {latest_competition_result['file_name']}")
        print(f"date: {latest_competition_result['date']}")
        print(f"status: {latest_competition_result['status']}")
        print(f"public score: {latest_competition_result['public_score']}")
        print(f"private score: {latest_competition_result['private_score']}")
    else:
        print(f"\nNo official Kaggle results found at {training_report['competition_results_path']}")

    print("\nInterpretation")
    print("- Loss is not winrate.")
    print("- Lower value_loss means better win/loss prediction on rollout states.")
    print("- Official Kaggle score comes from scripts/fetch_competition_results.py after a submission finishes.")
