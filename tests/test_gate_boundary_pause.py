from argparse import Namespace

from scripts.train_pure_rl import (
    _effective_boundary_design_migration_reason,
    _gate_boundary_pause_seconds,
    _parse_args,
)


def test_specialist_gate_pause_defaults_to_thirty_seconds(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PURE_RL_GATE_BOUNDARY_PAUSE_SECONDS", raising=False)
    args = _parse_args(["--run-name", "test"])
    assert args.gate_boundary_pause_seconds == 30.0


def test_gate_pause_begins_at_floor_and_repeats_afterward() -> None:
    args = Namespace(
        minimum_terminal_iteration=5,
        gate_boundary_pause_seconds=30.0,
    )
    assert _gate_boundary_pause_seconds(args, completed_iteration=4) == 0.0
    assert _gate_boundary_pause_seconds(args, completed_iteration=5) == 30.0
    assert _gate_boundary_pause_seconds(args, completed_iteration=15) == 30.0


def test_teal_auxiliary_override_accepts_effective_env_reason(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE",
        "receipt_backed_teal_auxiliary_head_rebalance_v1",
    )
    args = _parse_args(
        [
            "--run-name",
            "teal-r52-boundary",
            "--tactical-outcome-loss-weight-override",
            "0.01",
        ]
    )
    assert _effective_boundary_design_migration_reason(args) == (
        "receipt_backed_teal_auxiliary_head_rebalance_v1"
    )
