from pathlib import Path

from scripts.run_alakazam_expert_bootstrap import bootstrap_command


def test_alakazam_bootstrap_is_25_epoch_device_resident_and_core_initialized() -> None:
    command = bootstrap_command(
        python=Path("/python"),
        manifest=Path("/data/alakazam/manifest.json"),
        run_name="alakazam-v1",
        init_checkpoint=Path("/registry/deck_agnostic_core/model.pt"),
        resume=False,
        epochs=25,
        patience=5,
        min_decisions=100_000,
    )
    joined = " ".join(command)
    assert "--archetype alakazam" in joined
    assert "--epochs 25" in joined
    assert "--patience 5" in joined
    assert "--device-resident" in command
    assert "--max-decisions-per-batch 12288" in joined
    assert "--init-checkpoint /registry/deck_agnostic_core/model.pt" in joined
    assert "--aux-loss-weight 0" in joined
