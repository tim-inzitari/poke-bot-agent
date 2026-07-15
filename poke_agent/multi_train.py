from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from poke_agent.checkpoint import print_training_report, save_checkpoint
from poke_agent.dataset import TrainingTensors
from poke_agent.device import resolve_infer_device, resolve_train_device
from poke_agent.memory import print_vram_estimate
from poke_agent.model_catalog import TRAIN_MODELS
from poke_agent.model_registry import build_model_config, describe_catalog, get_model_spec, validate_catalog
from poke_agent.paths import print_runtime_info, resolve_root
from poke_agent.training import build_model, train_model


def train_catalog_models(
    *,
    root: Path,
    tensors: TrainingTensors,
    device: torch.device,
    model_ids: list[str] | None = None,
    base_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validate_catalog()
    selected = model_ids or list(TRAIN_MODELS)
    reports: list[dict[str, Any]] = []

    for model_id in selected:
        spec = get_model_spec(model_id)
        if spec.get("kind") != "neural":
            print(f"skip {model_id}: only neural models are trainable (kind={spec.get('kind')})")
            continue

        print("\n" + "=" * 72)
        print(f"training catalog model: {model_id}")
        print(spec.get("description", ""))
        print("=" * 72)

        config = build_model_config(root, model_id, base_config)
        model = build_model(config, tensors, device)
        print_vram_estimate(
            model=model,
            param_count=sum(p.numel() for p in model.parameters()),
            tensors=tensors,
            config=config,
            device=device,
        )
        report = train_model(model, tensors, config, device)
        report["model_id"] = model_id
        report["model_kind"] = "neural"
        report = save_checkpoint(
            model=model,
            tensors=tensors,
            config=config,
            training_report=report,
            output_path=config["output_path"],
        )
        print_training_report(report, config["output_path"])
        reports.append(report)

    return reports


def main() -> None:
    root = resolve_root()
    print_runtime_info(root)
    print("torch", torch.__version__)
    print("catalog models:")
    for row in describe_catalog(root):
        flags = []
        if row["active"]:
            flags.append("active")
        if row["train"]:
            flags.append("train")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        print(f"  - {row['id']}: {row['kind']} — {row['description']}{suffix}")

    from poke_agent.config import build_config
    from poke_agent.dataset import prepare_training_tensors

    base_config = build_config(root)
    print("output layout:")
    for name, path in base_config["output_layout"].items():
        print(f"  {name}: {path}")
    device = resolve_train_device(base_config.get("train_device"))
    infer_device = resolve_infer_device(base_config.get("infer_device"), train_device=device)
    print("train_device", device)
    print("infer_device", infer_device)

    tensors = prepare_training_tensors(base_config, device)
    train_catalog_models(root=root, tensors=tensors, device=device, base_config=base_config)


if __name__ == "__main__":
    main()
