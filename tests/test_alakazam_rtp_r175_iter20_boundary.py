from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-iter20-boundary.service"
FINALIZER_UNIT = ROOT / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-iter20-finalize.service"
R192_UNIT = ROOT / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-r192-inactive-boundary.service"


def test_r175_boundary_stops_exactly_after_committed_iteration_20() -> None:
    text = UNIT.read_text(encoding="utf-8")

    assert "--run-dir /home/inzi/poke-bot-agent/outputs/pure_rl/final_format_alakazam_rtp_r175_i_v6_8k" in text
    assert "--unit pokebot-final-format-alakazam-rtp-r175-rl.service" in text
    assert "--completed-iteration 20" in text
    assert "--poll-seconds 0.10" in text
    assert "--next-unit pokebot-final-format-alakazam-rtp-r175-iter20-finalize.service" in text
    assert "Restart=on-failure" in text


def test_r175_boundary_uses_checksum_verifying_committed_watcher() -> None:
    text = UNIT.read_text(encoding="utf-8")
    watcher = (ROOT / "scripts/pause_at_committed_iteration.py").read_text(
        encoding="utf-8"
    )

    assert "scripts/pause_at_committed_iteration.py" in text
    assert "if loop_state != commit:" in watcher
    assert "_sha256(checkpoint) != digest" in watcher
    assert "_systemctl(\"stop\", unit)" in watcher
    assert "iteration {next_iteration} already committed" in watcher
    assert "status=\"paused_successor_start_requested\"" in watcher
    assert watcher.index("status=\"paused_successor_start_requested\"") < watcher.index(
        '_systemctl("start", "--no-block", next_unit)'
    )


def test_r175_boundary_successor_freezes_only_the_verified_terminal_checkpoint() -> None:
    text = FINALIZER_UNIT.read_text(encoding="utf-8")

    assert "scripts/finalize_alakazam_rtp_r175_iter20.py --check" in text
    assert "scripts/finalize_alakazam_rtp_r175_iter20.py --run-dir" in text
    assert "--boundary /home/inzi/poke-bot-agent/outputs/state/final-format-alakazam-rtp-r175-iter20-boundary-r193.json" in text
    assert "--unit pokebot-final-format-alakazam-rtp-r175-rl.service" in text
    assert "specialist_runtime_registry_h10_r175_iter20_terminal.json" in text
    assert "final-format-alakazam-rtp-r175-iter20-registration-v1.json" in text
    assert "final-format-alakazam-rtp-r175-iter20-completion-v1.json" in text
    assert "Restart=" not in text


def test_r192_inactive_boundary_is_exactly_post_iteration_16() -> None:
    text = R192_UNIT.read_text(encoding="utf-8")

    assert "scripts/pause_at_committed_iteration.py" in text
    assert "--unit pokebot-final-format-alakazam-rtp-r175-rl.service" in text
    assert "--completed-iteration 16" in text
    assert "--poll-seconds 0.10" in text
    assert "--next-unit" not in text
    assert "alakazam-marnie-splusplus-r192-inactive-boundary.json" in text
