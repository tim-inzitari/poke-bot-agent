from __future__ import annotations

import torch

from poke_agent.checkpoint import print_training_report, save_checkpoint
from poke_agent.config import build_config, resolve_generate_games
from poke_agent.dataset import prepare_training_tensors
from poke_agent.deck import read_deck
from poke_agent.device import torch_device
from poke_agent.paths import print_runtime_info, resolve_root
from poke_agent.simulator import load_simulator, print_simulator_status
from poke_agent.training import build_model, train_model
from poke_agent.memory import print_vram_estimate


def main() -> None:
    root = resolve_root()
    print_runtime_info(root)
    print("torch", torch.__version__)

    config = build_config(root)
    print("output layout:")
    for name, path in config["output_layout"].items():
        print(f"  {name}: {path}")
    device = torch_device()
    print("device", device)

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

    tensors = prepare_training_tensors(config, device)
    model = build_model(config, tensors, device)
    print_vram_estimate(
        model=model,
        param_count=sum(p.numel() for p in model.parameters()),
        tensors=tensors,
        config=config,
        device=device,
    )
    training_report = train_model(model, tensors, config, device)
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
