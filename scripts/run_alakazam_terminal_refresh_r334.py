#!/usr/bin/env python3
"""Run only the owner-authorized terminal expert refresh from iter 13.

This wrapper reuses the exact managed derivative trainer environment and argv,
but closes the ledger at iteration 14 and makes that terminal boundary a
single-epoch rehearsal.  The trainer therefore executes no collection loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


UNIT = "pokebot-alakazam-rule-derivative-g5-rl.service"
RUN_DIR = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "alakazam_rule_derivative_g5_r12"
)
PARENT = RUN_DIR / "checkpoints/iter_00013.pt"
EXPECTED_PARENT = (
    "60b3b4b697f203698de5a580fe65da376f1be4b97fc47c473d214cb7ec25331d"
)


def _show(prop: str) -> str:
    return subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, f"--property={prop}", "--value"],
        text=True,
    ).strip()


def _replace_arg(argv: list[str], flag: str, value: str) -> None:
    try:
        index = argv.index(flag)
    except ValueError as exc:
        raise RuntimeError(f"managed trainer lacks required {flag}") from exc
    if index + 1 >= len(argv):
        raise RuntimeError(f"managed trainer has malformed {flag}")
    argv[index + 1] = value


def main() -> int:
    if _show("ActiveState") != "inactive":
        raise RuntimeError("managed derivative trainer is not inactive")
    if not PARENT.is_file():
        raise RuntimeError("committed iteration-13 parent is missing")
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != EXPECTED_PARENT:
        raise RuntimeError("committed iteration-13 parent digest drifted")
    if (RUN_DIR / "collection_receipts/iter_00014.json").exists():
        raise RuntimeError("iteration-14 collection already has a receipt")
    state = json.loads((RUN_DIR / "loop_state.json").read_text(encoding="utf-8"))
    if int(state.get("next_iteration", -1)) != 14:
        raise RuntimeError("ledger is not exactly at the iteration-14 boundary")
    learner = dict(state.get("learner") or {})
    if str(learner.get("digest") or "").removeprefix("sha256:") != EXPECTED_PARENT:
        raise RuntimeError("iteration-14 ledger parent is not committed iteration 13")

    raw_exec = _show("ExecStart")
    match = re.search(r"argv\[\]=(.*?) ; ignore_errors=", raw_exec)
    if match is None:
        raise RuntimeError("unable to parse managed trainer ExecStart")
    argv = shlex.split(match.group(1))
    _replace_arg(argv, "--iterations", "14")
    _replace_arg(argv, "--expert-rehearsal-every", "14")
    _replace_arg(argv, "--expert-rehearsal-epochs", "1")
    _replace_arg(
        argv,
        "--boundary-design-migration-reason",
        "owner_r334_terminal_one_epoch_expert_refresh",
    )

    env = dict(os.environ)
    for assignment in shlex.split(_show("Environment")):
        name, separator, value = assignment.partition("=")
        if not separator:
            raise RuntimeError("malformed managed trainer environment")
        env[name] = value
    env["PYTHONPATH"] = os.getcwd()
    env["PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE"] = (
        "owner_r334_terminal_one_epoch_expert_refresh"
    )
    print(
        "R334_TERMINAL_REFRESH_EXEC "
        f"parent=sha256:{EXPECTED_PARENT} epochs=1 collection_games=0 "
        "migration_reason="
        f"{argv[argv.index('--boundary-design-migration-reason') + 1]}",
        flush=True,
    )
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    raise SystemExit(main())
