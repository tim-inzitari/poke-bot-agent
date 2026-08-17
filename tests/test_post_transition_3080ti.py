from pathlib import Path

from scripts.enable_3080ti_after_alakazam_iteration import (
    audit_cuda_reports,
    select_service_loop_state,
    simulator_report_is_production_eligible,
    write_drop_in,
)


def test_partial_cuda_rule_slice_never_claims_production() -> None:
    assert not simulator_report_is_production_eligible(
        {
            "status": "complete",
            "production_eligible": True,
            "full_seeded_game_parity": False,
            "full_engine_transition_coverage": False,
        }
    )


def test_full_cuda_engine_must_pass_every_gate() -> None:
    assert simulator_report_is_production_eligible(
        {
            "status": "complete",
            "production_eligible": True,
            "full_seeded_game_parity": True,
            "full_engine_transition_coverage": True,
        }
    )


def test_audit_defaults_to_fleet_fallback_without_full_report(tmp_path: Path) -> None:
    report = tmp_path / "partial-cuda.json"
    report.write_text(
        '{"status":"complete","production_eligible":false,'
        '"full_seeded_game_parity":false}\n',
        encoding="utf-8",
    )
    result = audit_cuda_reports([str(tmp_path / "*.json")])
    assert result["production_eligible"] is False
    assert result["reports"][0]["accepted"] is False


def test_drop_in_enables_bounded_3080ti_leaf_farm(tmp_path: Path) -> None:
    path = tmp_path / "service.d" / "3080.conf"
    write_drop_in(path, gpu0_replicas=10, gpu0_fraction=0.30)
    text = path.read_text(encoding="utf-8")
    assert "PURE_RL_LEAF_GPU0_REPLICAS=10" in text
    assert "PURE_RL_LEAF_GPU0_FRAC=0.3000" in text
    assert "PURE_RL_GPU0_CLIENT_FRAC=0.3800" in text
    assert "POKEBOT_LIVE_POOL_MAX_LEAF_GPU0=12" in text


def test_loop_discovery_follows_service_run_not_stale_mtime(
    tmp_path: Path, monkeypatch
) -> None:
    stale = tmp_path / "v1" / "loop_state.json"
    current = tmp_path / "v4" / "loop_state.json"
    stale.parent.mkdir()
    current.parent.mkdir()
    stale.write_text(
        '{"run_name":"pure_rl_alakazam_public64k_v1_20260720",'
        '"last_completed_iteration":0}\n',
        encoding="utf-8",
    )
    current.write_text(
        '{"run_name":"pure_rl_alakazam_public64k_v4_20260720",'
        '"last_completed_iteration":-1}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.enable_3080ti_after_alakazam_iteration.service_properties",
        lambda _name: {
            "ExecStart": "launch_pure_rl.py --run-name "
            "pure_rl_alakazam_public64k_v4_20260720 --resume auto"
        },
    )
    path, loop, rows = select_service_loop_state(
        service="pokebot-pure-rl-alakazam.service",
        paths=[],
        patterns=[str(tmp_path / "*" / "loop_state.json")],
    )
    assert path == current
    assert loop["last_completed_iteration"] == -1
    assert len(rows) == 2


def test_loop_discovery_fails_closed_without_service_match(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "loop_state.json"
    ledger.write_text(
        '{"run_name":"abandoned-run","last_completed_iteration":9}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.enable_3080ti_after_alakazam_iteration.service_properties",
        lambda _name: {"ExecStart": "launch_pure_rl.py --run-name current-run"},
    )
    path, loop, rows = select_service_loop_state(
        service="pokebot-pure-rl-alakazam.service",
        paths=[ledger],
        patterns=[],
    )
    assert path is None
    assert loop == {}
    assert rows[0]["matches_service_exec_start"] is False
