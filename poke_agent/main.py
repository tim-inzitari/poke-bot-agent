from __future__ import annotations

import torch

from poke_agent.checkpoint import print_training_report, resolve_resume_path, save_checkpoint
from poke_agent.config import build_config, resolve_generate_games
from poke_agent.dataset import prepare_training_tensors
from poke_agent.deck import read_deck
from poke_agent.device import resolve_infer_device, resolve_train_device
from poke_agent.paths import print_runtime_info, resolve_root
from poke_agent.simulator import load_simulator, print_simulator_status
from poke_agent.training import build_model, train_model
from poke_agent.memory import print_vram_estimate


def main(*, tensors_only: bool = False) -> None:
    root = resolve_root()
    print_runtime_info(root)
    print("torch", torch.__version__)

    config = build_config(root)
    print("output layout:")
    for name, path in config["output_layout"].items():
        print(f"  {name}: {path}")
    device = resolve_train_device(config.get("train_device"))
    infer_device = resolve_infer_device(config.get("infer_device"), train_device=device)
    print("train_device", device)
    print("infer_device", infer_device)
    if config.get("ollama_base_url"):
        print("ollama_base_url", config["ollama_base_url"])

    from poke_agent.tensor_cache import describe_training_tensor_cache

    cache_status = describe_training_tensor_cache(config)
    if cache_status:
        print(cache_status)

    tensors = prepare_training_tensors(config, device)
    if tensors_only:
        print("tensor cache ready; exiting without training")
        return

    simulator = load_simulator(root)
    print_simulator_status(simulator)

    deck, deck_source = read_deck(config, root)
    print("deck cards", len(deck))
    print("deck source", deck_source)

    generate_games = resolve_generate_games(config)
    if generate_games > 0:
        print(
            "skipping mirror inline rollout generation "
            f"({generate_games} games requested). "
            "Use scripts/generate_cabt_data.py --matchups weighted and scripts/merge_rollouts.py "
            "for multi-deck training data."
        )

    model = build_model(config, tensors, device)
    print_vram_estimate(
        model=model,
        param_count=sum(p.numel() for p in model.parameters()),
        tensors=tensors,
        config=config,
        device=device,
    )
    train_cfg = config["training"]
    resume_path = resolve_resume_path(config["output_path"], train_cfg.get("resume", "auto"))
    training_report = train_model(
        model,
        tensors,
        config,
        device,
        checkpoint_path=config["output_path"],
        checkpoint_every_epochs=int(train_cfg.get("checkpoint_every", 0)),
        resume_path=resume_path,
    )
    training_report = save_checkpoint(
        model=model,
        tensors=tensors,
        config=config,
        training_report=training_report,
        output_path=config["output_path"],
    )
    print_training_report(training_report, config["output_path"])


if __name__ == "__main__":
    main()
