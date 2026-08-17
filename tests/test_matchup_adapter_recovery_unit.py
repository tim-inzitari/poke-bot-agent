from pathlib import Path


def test_recovery_unit_resumes_exact_fit_without_starting_after_completion() -> None:
    root = Path(__file__).resolve().parents[1]
    unit = (
        root / ".staging/pokebot-matchup-adapter-v31-recovery.service"
    ).read_text(encoding="utf-8")

    output = (
        "/home/pokebot/poke-bot-agent/outputs/matchup_adapters/"
        "alakazam-iter26-all22-v31"
    )
    assert f"ConditionPathExists={output}/latest.pt" in unit
    assert f"ConditionPathExists=!{output}/final.pt" in unit
    assert "--parent-checkpoint " in unit
    assert "/checkpoints/iter_00026.pt" in unit
    assert "--activation-receipt " in unit
    assert "--resume auto" in unit
    assert "--epochs 25" in unit
    assert "--device cuda:1" in unit
    assert "Restart=on-failure" in unit
    assert "MemoryMax=64G" in unit


def test_finalizer_recognizes_recovery_unit() -> None:
    root = Path(__file__).resolve().parents[1]
    finalizer = (root / "scripts/finalize_matchup_runtime_v31.sh").read_text(
        encoding="utf-8"
    )

    assert "pokebot-matchup-adapter-v31-recovery.service" in finalizer
    assert "pokebot-adapter-fleet-blackwell.service" in finalizer
    assert "pokebot-adapter-fleet-3080.service" in finalizer
    assert "pokebot-adapter-fleet-finalizer.service" in finalizer
    assert "waiting_for_exact_adapter_fleet" in finalizer
