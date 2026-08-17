from pathlib import Path

from scripts.run_test_profile import load_manifest, profile_commands
from scripts import launch_blackwell, launch_core_pipeline
from scripts.baseline_compat_canary import _random_legal


ROOT = Path(__file__).resolve().parents[1]


def test_profile_selection_is_explicit_and_full_is_manual() -> None:
    manifest = load_manifest()
    quick = profile_commands("quick", "python")
    canary = profile_commands("canary", "python")
    full = profile_commands("full", "python")

    assert len(quick) == 1
    assert "unit and not (native or gpu or integration or slow)" in quick[0]
    assert any("test_native_canary.py" in token for token in canary[1])
    assert "--mode" in full[1] and "full" in full[1]
    assert manifest["profiles"]["full"]["manual_only"] is True


def test_invariant_manifest_references_existing_tests_or_scripts() -> None:
    manifest = load_manifest()
    required = {
        "large_action_complete_ordered_support",
        "belief_card_conservation_and_support",
        "deployment_information_set_and_action_aggregation",
        "chance_and_actor_correct_minimax",
        "writer_exactly_once_shutdown_resume",
        "clean_iteration_contract_migration",
        "model_generation_and_reload_barrier",
        "native_gpu0_trusted_targets",
        "native_gpu1_trusted_targets",
        "full_baseline_compatibility",
    }
    assert required <= set(manifest["invariants"])
    for references in manifest["invariants"].values():
        assert references
        for reference in references:
            relative = reference.split("::", 1)[0]
            assert (ROOT / relative).exists(), reference


def test_pytest_markers_and_default_unit_classification_are_registered() -> None:
    config = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    conftest = (ROOT / "tests/conftest.py").read_text(encoding="utf-8")
    for marker in ("unit", "native", "gpu", "integration", "slow"):
        assert f"    {marker}:" in config
    assert "item.add_marker(pytest.mark.unit)" in conftest
    native = (ROOT / "tests/test_native_canary.py").read_text(encoding="utf-8")
    for marker in ("native", "gpu", "integration", "slow"):
        assert f"pytest.mark.{marker}" in native


def test_both_launchers_default_to_canary_preflight() -> None:
    blackwell = launch_blackwell._parse_args(
        ["--run-name", "test", "--", "--archetype", "hammer-pult"]
    )
    core = launch_core_pipeline._args(["--run-name", "test"])
    assert blackwell.preflight_profile == "canary"
    assert core.preflight_profile == "canary"


def test_compatibility_random_agent_preserves_ordered_legality() -> None:
    import random

    observation = {
        "select": {
            "option": [{}, {}, {}, {}],
            "minCount": 2,
            "maxCount": 3,
        }
    }
    action = _random_legal(observation, random.Random(7))
    assert 2 <= len(action) <= 3
    assert len(action) == len(set(action))
    assert all(0 <= index < 4 for index in action)
