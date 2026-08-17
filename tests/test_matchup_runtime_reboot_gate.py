from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_enabled_production_is_reboot_gated_until_boundary_receipt() -> None:
    guard = (ROOT / ".staging/99-stop-for-v31-sync.conf").read_text(
        encoding="utf-8"
    )
    ready = (
        "/home/pokebot/poke-bot-agent/outputs/state/"
        "matchup-runtime-v31-production-ready.json"
    )
    assert "RefuseManualStop=no" in guard
    assert "assert_matchup_runtime_production_ready.py" in guard
    assert f"--receipt {ready}" in guard


def test_finalizer_publishes_receipt_before_starting_production() -> None:
    script = (ROOT / "scripts/finalize_matchup_runtime_v31.sh").read_text(
        encoding="utf-8"
    )
    publish = script.index("status publishing_boundary_learner")
    install = script.index("status installing_production_v31")
    ready = script.index("status publishing_production_ready")
    start = script.index('systemctl --user start "$PRODUCTION"')
    assert publish < install < ready < start
    assert "poke_bot.matchup_runtime_production_ready/v1" in script


def test_finalizer_restarts_registered_bert_job_without_bootout_race() -> None:
    script = (ROOT / "scripts/finalize_matchup_runtime_v31.sh").read_text(
        encoding="utf-8"
    )
    bundle_step = script.split(
        "status installing_remote_runtime_bundles", 1
    )[1].split("status verifying_remote_runtime", 1)[0]
    assert 'launchctl kickstart -k \\\"\\$target\\\"' in bundle_step
    assert 'elmo_marker_tmp="/tmp/matchup-runtime-activation-$stamp.json"' in bundle_step
    assert 'bert_marker_tmp="/tmp/matchup-runtime-activation-$stamp.json"' in bundle_step
    assert "launchctl bootout" not in bundle_step
    assert "launchctl bootstrap" not in bundle_step


def test_ready_validator_is_the_systemd_exec_condition() -> None:
    validator = (
        ROOT / "scripts/assert_matchup_runtime_production_ready.py"
    ).read_text(encoding="utf-8")
    assert 'SCHEMA = "poke_bot.matchup_runtime_production_ready/v1"' in validator
    assert "set(artifacts) == set(EXPECTED_PATHS)" in validator
    assert '"Environment=POKEBOT_MATCHUP_ADAPTER_RUNTIME=1" in dropin' in validator


def test_bert_runtime_bundle_uses_supervised_kickstart() -> None:
    script = (ROOT / "scripts/finalize_matchup_runtime_v31.sh").read_text(
        encoding="utf-8"
    )
    line = next(
        item for item in script.splitlines()
        if item.startswith('ssh bert.local "set -e; domain=')
    )
    assert 'launchctl print \\"\\$target\\" >/dev/null' in line
    assert 'nohup launchctl kickstart -k \\"\\$target\\"' in line
    assert '>/dev/null 2>&1 </dev/null &' in line
    assert "launchctl bootout" not in line
