#!/usr/bin/env python
"""Canary and atomically switch the persistent trainer to a hot-start lineage."""

from __future__ import annotations

import argparse
import json
import os
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
from poke_bot.train import load_model_from_checkpoint


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotstart-run", required=True)
    parser.add_argument("--new-rl-run", required=True)
    parser.add_argument(
        "--old-service", default="pokebot-pure-rl-core.service"
    )
    parser.add_argument(
        "--new-service", default="pokebot-pure-rl-top-ladder-hotstart.service"
    )
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--eval-games-per-opponent",
        type=int,
        default=34,
        help=(
            "Must be a multiple of 34 so all 17 core decks run once in each "
            "seat per official opponent."
        ),
    )
    parser.add_argument("--eval-workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _wait_ready(path: Path, wait_seconds: int, poll_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            status = str(payload.get("status") or "")
            if status == "ready_for_rl":
                return payload
            if status == "failed":
                raise RuntimeError(f"hot-start training failed: {payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"hot-start checkpoint was not ready: {path}")
        time.sleep(max(1.0, poll_seconds))


def _run_systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _stage_weights_only_seed(best_path: Path, new_run: str) -> Path:
    import torch

    from poke_bot.checkpoint import atomic_torch_save, build_checkpoint

    run_dir = paths.OUTPUTS_DIR / "pure_rl" / new_run
    state_path = run_dir / "loop_state.json"
    seed_path = run_dir / "checkpoints" / "seed.pt"
    if state_path.is_file():
        # The append-only ledger owns the lineage after first activation.
        return seed_path

    model = load_model_from_checkpoint(best_path, device=torch.device("cpu"))
    source = checkpoint.load_checkpoint(best_path, map_location="cpu")
    source_extra = dict(source.get("extra") or {})
    seed_extra = {
        "pure_rl": True,
        "smoke": False,
        "model_profile": source_extra.get("model_profile"),
        "top_ladder_hot_start_seed": {
            "weights_from": str(best_path),
            "weights_from_digest": checkpoint.checkpoint_digest(best_path),
            "optimizer_state_copied": False,
            "source": source_extra.get("top_ladder_hot_start"),
        },
    }
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        build_checkpoint(
            model=model,
            step=0,
            epoch=0,
            model_config=getattr(model, "cfg", None),
            extra=seed_extra,
        ),
        seed_path,
    )
    from scripts.train_pure_rl import _checkpoint_contract

    _checkpoint_contract(seed_path, smoke=False)
    return seed_path


def _evaluate(seed_path: Path, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    from scripts.train_pure_rl import OFFICIAL_BASELINE_IDS

    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "eval_vs_baselines.py"),
        "--checkpoint",
        str(seed_path),
        "--deck-suite",
        "core-ladder",
        "--games-per-opp",
        str(int(args.eval_games_per_opponent)),
        "--min-games-per-opp",
        str(int(args.eval_games_per_opponent)),
        "--workers",
        str(int(args.eval_workers)),
        "--agent-mode",
        "policy",
        "--gate",
        "0",
        "--out",
        str(out),
        "--only",
        *OFFICIAL_BASELINE_IDS,
        "--seed",
        str(int(args.seed)),
        "--leaf-eval",
        "gpu-server",
        "--leaf-gpu",
        "cuda:0",
        "--leaf-max-batch",
        "192",
        "--leaf-coalesce-ms",
        "0",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["POKEBOT_GAME_TIMEOUT_S"] = "300"
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if not out.is_file():
        raise RuntimeError(
            f"pre-activation evaluation produced no report (rc={completed.returncode})"
        )
    report = json.loads(out.read_text(encoding="utf-8"))
    if completed.returncode != 0 or not bool(report.get("valid", False)):
        raise RuntimeError(
            f"pre-activation evaluation invalid rc={completed.returncode}: "
            f"{report.get('invalid_reasons')}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    hotstart_meta_path = (
        paths.OUTPUTS_DIR / "hotstart" / f"{args.hotstart_run}.json"
    )
    activation_path = (
        paths.OUTPUTS_DIR / "hotstart" / f"{args.new_rl_run}.activation.json"
    )
    activation: dict[str, Any] = {
        "schema": "poke_bot.top_ladder_hotstart_activation/v1",
        "hotstart_run": args.hotstart_run,
        "new_rl_run": args.new_rl_run,
        "old_service": args.old_service,
        "new_service": args.new_service,
        "status": "waiting_for_hotstart",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(activation_path, activation)

    hotstart = _wait_ready(
        hotstart_meta_path, int(args.wait_seconds), float(args.poll_seconds)
    )
    best_path = Path(str(hotstart.get("best_checkpoint") or "")).resolve()
    from scripts.train_pure_rl import _checkpoint_contract

    _checkpoint_contract(best_path, smoke=False)
    seed_path = _stage_weights_only_seed(best_path, args.new_rl_run)
    activation.update(
        {
            "status": "stopping_old_trainer",
            "best_checkpoint": str(best_path),
            "seed_checkpoint": str(seed_path),
            "seed_digest": checkpoint.checkpoint_digest(seed_path),
        }
    )
    _atomic_json(activation_path, activation)

    old_was_active = (
        _run_systemctl("is-active", args.old_service, check=False).returncode == 0
    )
    if old_was_active:
        _run_systemctl("stop", args.old_service)

    try:
        activation["status"] = "pre_activation_eval"
        _atomic_json(activation_path, activation)
        eval_path = paths.OUTPUTS_DIR / "eval" / f"{args.new_rl_run}.pre_rl.json"
        report = _evaluate(seed_path, args, eval_path)
        activation["pre_activation_eval"] = {
            "path": str(eval_path),
            "pooled_formal": report.get("pooled_formal"),
            "matchups": report.get("matchups"),
        }
    except Exception as exc:
        activation.update(
            {
                "status": "eval_failed_old_restored",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(activation_path, activation)
        if old_was_active:
            _run_systemctl("start", args.old_service, check=False)
        raise

    try:
        _run_systemctl("disable", args.old_service, check=False)
        _run_systemctl("daemon-reload")
        _run_systemctl("enable", "--now", args.new_service)
        time.sleep(5)
        active = _run_systemctl("is-active", args.new_service, check=False)
        if active.returncode != 0:
            raise RuntimeError(active.stdout.strip() or "new service is not active")
    except Exception as exc:
        activation.update(
            {
                "status": "new_service_failed_old_restored",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(activation_path, activation)
        if old_was_active:
            _run_systemctl("enable", "--now", args.old_service, check=False)
        raise RuntimeError("new trainer failed; old trainer restored") from exc

    activation.update(
        {
            "status": "new_rl_active",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(activation_path, activation)
    print(json.dumps(activation, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
