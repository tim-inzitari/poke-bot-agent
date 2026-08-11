"""Static safety checks for the isolated r241 specialist-corpus finalizer."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/elmo/run_r241_exact20_specialist_finalizer.sh"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _launch_receipt_program(script: str) -> str:
    match = re.search(
        r"write_launch_receipt\(\) \{\n  python3 - <<'PY'\n(?P<program>.*?)\nPY\n\}",
        script,
        flags=re.DOTALL,
    )
    assert match is not None, "launch receipt program is absent"
    return match.group("program")


def _docker_environment_keys(script: str) -> set[str]:
    return set(re.findall(r"^  -e ([A-Z0-9_]+)=", script, flags=re.MULTILINE))


def _environment_keys_read(program: str) -> set[str]:
    parsed = ast.parse(program)
    result: set[str] = set()
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Subscript):
            continue
        if not (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            continue
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            result.add(node.slice.value)
    return result


def test_r241_exact20_finalizer_is_syntax_valid_and_seals_its_identity() -> None:
    checked = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr

    script = _script()
    assert 'readonly R241_CANDIDATE_ID="alakazam-new-list-direct-policy-r241"' in script
    assert (
        'readonly R241_ARCHIVE_RECEIPT_SHA256='
        '"sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"'
    ) in script
    assert 'readonly R241_WINDOW_START="2026-07-22"' in script
    assert 'readonly R241_WINDOW_END="2026-08-10"' in script
    assert 'readonly R241_WINDOW_DAYS="20"' in script
    assert 'readonly R241_DATASET_SCHEMA="6"' in script
    assert 'readonly R241_FEATURE_SCHEMA="5"' in script
    assert (
        'readonly R241_EXPANDED_TARGET_SCHEMA='
        '"poke_bot.expanded_strategic_targets/v2"'
    ) in script
    assert (
        'readonly R241_EXPANDED_TARGET_DIGEST='
        '"sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f"'
    ) in script
    assert 'POKEBOT_SOURCE' in script
    assert 'r241 exact20 finalizer forbids $override overrides' in script


def test_r241_exact20_finalizer_uses_only_the_three_canonical_schema6_ranges() -> None:
    script = _script()

    assert (
        'readonly SOURCE="/home/admin/pokebot-expert-src-v6-strategic"'
    ) in script
    assert (
        'readonly EARLY_FEATURES="/mnt/Main/main/poke-bot-agent/archive/'
        'expert-r241-derived/daily/roster18-v6-strategic-20260722-23"'
    ) in script
    assert (
        'readonly MID_FEATURES="/mnt/Main/main/poke-bot-agent/archive/'
        'expert-latest20-derived/daily/'
        'roster18-v6-strategic-2026-07-14_2026-08-02"'
    ) in script
    assert (
        'readonly NEW_FEATURES="/mnt/Main/main/poke-bot-agent/archive/'
        'expert-r241-derived/daily/roster18-v6-strategic"'
    ) in script
    assert "roster18-v5" not in script
    assert "may not fall back to older v5" in script

    assert 'wait_for_marker /input/early/MISSING_DAYS_READY.json "official-r236 Jul22-Jul23"' in script
    assert 'wait_for_marker /input/mid/MISSING_DAYS_READY.json "schema6 Jul24-Aug02"' in script
    assert 'wait_for_marker /input/new/R241_MISSING_DAYS_READY.json "r241 Aug03-Aug10"' in script
    assert "EARLY_DATES = EXPECTED_DATES[:2]" in script
    assert "MID_DATES = EXPECTED_DATES[2:12]" in script
    assert "TAIL_DATES = EXPECTED_DATES[12:]" in script
    assert "expected_root_by_date" in script
    assert "selected a noncanonical root" in script
    assert "select_sources(receipt, [ROOTS[\"official_r236_edge_days\"], ROOTS[\"schema6_mid_days\"], ROOTS[\"r241_tail_days\"]])" in script


def test_r241_exact20_finalizer_fails_closed_on_receipts_and_expanded_v2_provenance() -> None:
    script = _script()

    assert "r241 archive receipt digest changed" in script
    assert "r241 archive receipt window policy changed" in script
    assert "r241 archive receipt dates changed" in script
    assert "r241 archive receipt validation changed" in script
    assert "marker dates changed" in script
    assert "completed dates changed" in script
    assert "source dataset schema is not r241 schema6" in script
    assert "source feature schema is not r241 schema5" in script
    assert "expanded target schema changed" in script
    assert "expanded target digest changed" in script
    assert "merge_expanded_strategic_coverages((expanded,))" in script
    assert "expanded head inventory drifted" in script
    assert "expanded decisions drifted" in script
    assert "R241_EXACT20_SPECIALIST_FINALIZER_LAUNCH.json" in script
    assert "R241_EXACT20_SPECIALIST_FINALIZER_PREFLIGHT.json" in script
    assert "os.replace(temporary, target)" in script
    assert "existing r241 preflight receipt identity changed" in script


def test_r241_exact20_finalizer_container_isolated_and_never_removes_prior_evidence() -> None:
    script = _script()

    assert 'readonly NAME="pokebot-r241-exact20-specialist-finalizer-a2"' in script
    assert "--network none" in script
    assert "--read-only" in script
    assert "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g" in script
    assert "--cpus 12" in script
    assert "--memory 32g" in script
    assert "--memory-swap 32g" in script
    assert "--pids-limit 2048" in script
    assert '-v "$SOURCE:/workspace:ro"' in script
    assert '-v "$archive_parent:/input/archive:ro"' in script
    assert '-v "$EARLY_FEATURES:/input/early:ro"' in script
    assert '-v "$MID_FEATURES:/input/mid:ro"' in script
    assert '-v "$NEW_FEATURES:/input/new:ro"' in script
    assert '-v "$OUTPUT:/output"' in script
    assert '-v "$MAIN:$MAIN"' not in script
    assert "docker rm" not in script
    assert "docker stop" not in script
    assert "kill" not in script
    assert "MCTS" not in script
    assert "RTP" not in script


def test_r241_launch_receipt_env_reads_are_provided_to_the_docker_container(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = _script()
    program = _launch_receipt_program(script)
    docker_environment = _docker_environment_keys(script)
    reads = _environment_keys_read(program)

    assert reads <= docker_environment
    assert reads == {
        "R241_ARCHIVE_RECEIPT_SHA256",
        "R241_CANDIDATE_ID",
        "R241_CONTAINER_NAME",
        "R241_HOST_ARCHIVE_RECEIPT",
        "R241_HOST_OFFICIAL_R236_EDGE_DAYS",
        "R241_HOST_R241_TAIL_DAYS",
        "R241_HOST_SCHEMA6_MID_DAYS",
        "R241_HOST_SOURCE",
        "R241_WINDOW_DAYS",
        "R241_WINDOW_END",
        "R241_WINDOW_START",
    }

    for key in docker_environment:
        monkeypatch.setenv(key, f"test-{key}")
    monkeypatch.setenv("R241_WINDOW_DAYS", "20")
    monkeypatch.setenv("R241_TEST_OUTPUT", str(tmp_path))
    executable = program.replace(
        'Path("/output")', 'Path(os.environ["R241_TEST_OUTPUT"])'
    )
    namespace: dict[str, object] = {}
    exec(compile(executable, "r241-launch-receipt", "exec"), namespace)

    receipt = tmp_path / "R241_EXACT20_SPECIALIST_FINALIZER_LAUNCH.json"
    assert receipt.is_file()
    assert "test-R241_HOST_OFFICIAL_R236_EDGE_DAYS" in receipt.read_text(
        encoding="utf-8"
    )


def test_r241_exact20_finalizer_rejects_identity_override_before_touching_host_state() -> None:
    environment = dict(os.environ)
    environment["POKEBOT_SOURCE"] = "/not-the-schema6-r241-source"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 2
    assert "r241 exact20 finalizer forbids POKEBOT_SOURCE overrides" in result.stderr
