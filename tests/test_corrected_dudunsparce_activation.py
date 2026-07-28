from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_corrected_activation_waits_without_stopping_live_trainer() -> None:
    source = (
        ROOT / "ops/inzi/activate_corrected_dudunsparce_v2.sh"
    ).read_text(encoding="utf-8")
    assert 'while systemctl --user --quiet is-active "${unit}"' in source
    assert 'systemctl --user start "${unit}"' in source
    assert "systemctl --user stop" not in source
    assert "systemctl --user restart" not in source
    assert "kill " not in source
    assert "pkill" not in source


def test_corrected_activation_receipt_binds_every_runtime_input() -> None:
    source = (
        ROOT / "ops/inzi/activate_corrected_dudunsparce_v2.sh"
    ).read_text(encoding="utf-8")
    for field in (
        "active_selector_sha256",
        "runtime_registry_sha256",
        "bootstrap_ready_sha256",
        "gate_contract_sha256",
        "frozen_specialist_registry_sha256",
        "matchup_runtime_tree_sha256",
        "corrected_checkpoint_only",
    ):
        assert field in source
