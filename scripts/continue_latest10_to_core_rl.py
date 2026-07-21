#!/usr/bin/env python
"""Wait for the latest-ten bootstrap, validate it, then start core pure RL.

This is a one-shot lineage handoff.  It never stops a healthy bootstrap and it
never starts RL after a failed bootstrap.  The seed is weights-only and rebuilt
with current trusted stateless-policy provenance, then checked on the complete
17-deck core-ladder suite before the distributed RL service is enabled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint, paths
from scripts.activate_top_ladder_hotstart import (
    _atomic_json,
    _evaluate,
    _stage_weights_only_seed,
)
from scripts.train_pure_rl import _checkpoint_contract


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-service", default="pokemon-latest10-bootstrap.service"
    )
    parser.add_argument(
        "--bootstrap-run", default="state_core_top_ladder_latest10_20260719"
    )
    parser.add_argument(
        "--rl-run",
        default="pure_rl_core_top_ladder_10day_hotstart_v1_20260719",
    )
    parser.add_argument(
        "--rl-service",
        default="pokebot-pure-rl-top-ladder-10day-hotstart.service",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--wait-seconds", type=int, default=4 * 60 * 60)
    parser.add_argument("--eval-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args(argv)


def _run(
    command: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _system_properties(unit: str) -> dict[str, str]:
    completed = _run(
        [
            "systemctl",
            "show",
            unit,
            "--property=ActiveState,SubState,Result,ExecMainStatus,MainPID",
        ]
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            values[key] = value
    return values


def _wait_for_success(unit: str, wait_seconds: int, poll_seconds: float) -> dict:
    deadline = time.monotonic() + max(0, int(wait_seconds))
    while True:
        status = _system_properties(unit)
        if status.get("ActiveState") not in {
            "active",
            "activating",
            "reloading",
        }:
            if (
                status.get("Result") == "success"
                and status.get("ExecMainStatus") == "0"
            ):
                return status
            raise RuntimeError(f"bootstrap did not finish successfully: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {unit}: {status}")
        time.sleep(max(1.0, float(poll_seconds)))


def _activation_path(run_name: str) -> Path:
    return paths.OUTPUTS_DIR / "hotstart" / f"{run_name}.activation.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    activation_path = _activation_path(args.rl_run)
    activation: dict[str, Any] = {
        "schema": "poke_bot.latest10_core_activation/v1",
        "bootstrap_service": args.bootstrap_service,
        "bootstrap_run": args.bootstrap_run,
        "rl_run": args.rl_run,
        "rl_service": args.rl_service,
        "deck_suite": "core-ladder",
        "status": "waiting_for_bootstrap",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(activation_path, activation)

    try:
        activation["bootstrap_result"] = _wait_for_success(
            args.bootstrap_service,
            int(args.wait_seconds),
            float(args.poll_seconds),
        )
        best_path = checkpoint.best_path(args.bootstrap_run)
        if not best_path.is_file():
            raise FileNotFoundError(f"bootstrap best checkpoint missing: {best_path}")

        activation["status"] = "staging_trusted_core_seed"
        activation["best_checkpoint"] = str(best_path)
        activation["best_digest"] = checkpoint.checkpoint_digest(best_path)
        _atomic_json(activation_path, activation)

        seed_path = _stage_weights_only_seed(best_path, args.rl_run)
        seed_contract = _checkpoint_contract(seed_path, smoke=False)
        activation.update(
            {
                "seed_checkpoint": str(seed_path),
                "seed_digest": checkpoint.checkpoint_digest(seed_path),
                "seed_contract": seed_contract,
                "status": "deck_agnostic_pre_rl_eval",
            }
        )
        _atomic_json(activation_path, activation)

        eval_args = argparse.Namespace(
            eval_games_per_opponent=34,
            eval_workers=int(args.eval_workers),
            seed=int(args.seed),
        )
        eval_path = (
            paths.OUTPUTS_DIR / "eval" / f"{args.rl_run}.pre_rl.json"
        )
        report = _evaluate(seed_path, eval_args, eval_path)
        gate = dict(report.get("deck_agnostic_gate") or {})
        if gate.get("suite") != "core-ladder" or gate.get("deck_count") != 17:
            raise RuntimeError(f"pre-RL evaluation used wrong deck suite: {gate}")
        activation["pre_rl_eval"] = {
            "path": str(eval_path),
            "pooled_formal": report.get("pooled_formal"),
            "matchups": report.get("matchups"),
            "deck_count": gate.get("deck_count"),
            "roster": [row.get("deck_id") for row in gate.get("roster", [])],
        }

        activation["status"] = "starting_core_rl"
        _atomic_json(activation_path, activation)
        _run(["systemctl", "--user", "enable", "--now", args.rl_service])
        time.sleep(10)
        active = _run(
            ["systemctl", "--user", "is-active", args.rl_service],
            check=False,
        )
        if active.returncode != 0:
            raise RuntimeError(
                f"core RL service failed to stay active: {active.stdout.strip()}"
            )

        # The completed finite bootstrap must not race the persistent RL service
        # after a host reboot.  Passwordless sudo is a host preflight invariant.
        _run(
            ["sudo", "-n", "systemctl", "disable", args.bootstrap_service],
            check=False,
        )
        activation.update(
            {
                "status": "core_rl_active",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(activation_path, activation)
        print(json.dumps(activation, indent=2), flush=True)
        return 0
    except Exception as exc:
        activation.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(activation_path, activation)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
