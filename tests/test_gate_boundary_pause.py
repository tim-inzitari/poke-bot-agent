from argparse import Namespace

from scripts.train_pure_rl import _gate_boundary_pause_seconds, _parse_args


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
