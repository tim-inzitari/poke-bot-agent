#!/usr/bin/env python
"""Wait for a gated top-ladder dataset and behavior-clone the pure-RL core.

This is an unattended handoff helper, not a second simulator.  It resolves the
source run's current *champion* only after the replay artifact passes its
checksums and coverage gates, then starts a fresh optimizer on the exact small
pure-RL architecture.  A restart resumes only this isolated hot-start run.
"""

from __future__ import annotations

import argparse
import hashlib
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

from poke_bot import archetypes, checkpoint, paths
from poke_bot.ladder_deck_mix import load_ladder_deck_mix


DEFAULT_DATASET = paths.DATA_DIR / "bootstrap" / "top_ladder_all_2026-07-12.jsonl"
DEFAULT_MIX = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mix", type=Path, default=DEFAULT_MIX)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-run",
        help="Pure-RL run whose current champion initializes this hot start.",
    )
    source.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Explicit compatible checkpoint used for weights-only initialization.",
    )
    source.add_argument(
        "--init-hotstart-run",
        help="Wait for another hot-start run and initialize from its best checkpoint.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--games-per-batch", type=int, default=8)
    parser.add_argument("--max-decisions-per-batch", type=int, default=1024)
    parser.add_argument("--min-decisions", type=int, default=500_000)
    parser.add_argument(
        "--allow-partial-family-coverage",
        action="store_true",
        help=(
            "Allow a follow-up shard to omit rare families. The classifier "
            "contract and all represented labels are still validated."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _wait_for_dataset(dataset: Path, wait_seconds: int, poll_seconds: float) -> dict[str, Any]:
    meta_path = dataset.with_suffix(".meta.json")
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        if dataset.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if bool((meta.get("quality_gates") or {}).get("passed", False)):
                    return meta
            except (OSError, json.JSONDecodeError):
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"gated dataset did not appear: {dataset}")
        time.sleep(max(1.0, poll_seconds))


def _validate_dataset(
    dataset: Path,
    meta: dict[str, Any],
    mix_path: Path,
    *,
    require_all_families: bool = True,
) -> None:
    declared = str(meta.get("output_sha256") or "")
    actual = _sha256(dataset)
    if declared != actual:
        raise RuntimeError(
            f"top-ladder dataset digest mismatch: declared={declared} actual={actual}"
        )
    mix = load_ladder_deck_mix(mix_path)
    classifier = dict(meta.get("classifier") or {})
    if classifier.get("mix_artifact_sha256") != mix.artifact_sha256:
        raise RuntimeError("top-ladder dataset was classified against another mix")
    active = set(classifier.get("active_deck_ids") or [])
    expected = {entry.deck_id for entry in mix.decks}
    if active != expected:
        raise RuntimeError(
            f"active ladder families drifted: missing={sorted(expected-active)} "
            f"extra={sorted(active-expected)}"
        )
    represented = set(
        ((meta.get("stats") or {}).get("record_archetypes") or {}).keys()
    )
    missing = expected - represented
    unexpected = represented - (expected | {archetypes.UNKNOWN})
    if (require_all_families and missing) or unexpected or not represented:
        raise RuntimeError(
            f"hot-start corpus does not cover every active family: "
            f"missing={sorted(missing)} extra={sorted(unexpected)}"
        )


def _source_champion(source_run: str) -> dict[str, str]:
    state_path = paths.OUTPUTS_DIR / "pure_rl" / source_run / "loop_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    champion = dict(state.get("champion") or {})
    path = Path(str(champion.get("path") or "")).expanduser().resolve()
    digest = str(champion.get("digest") or "")
    if not path.is_file() or checkpoint.checkpoint_digest(path) != digest:
        raise RuntimeError(f"source champion identity is invalid: {champion!r}")
    return {"path": str(path), "digest": digest}


def _explicit_checkpoint(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"initial checkpoint does not exist: {resolved}")
    return {
        "path": str(resolved),
        "digest": checkpoint.checkpoint_digest(resolved),
    }


def _wait_for_hotstart(
    run_name: str, wait_seconds: int, poll_seconds: float
) -> dict[str, str]:
    meta_path = paths.OUTPUTS_DIR / "hotstart" / f"{run_name}.json"
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status = str(meta.get("status") or "")
            if status == "ready_for_rl":
                return _explicit_checkpoint(Path(str(meta["best_checkpoint"])))
            if status == "failed":
                raise RuntimeError(f"prerequisite hot start failed: {run_name}")
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"hot-start checkpoint did not appear: {run_name}")
        time.sleep(max(1.0, poll_seconds))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset = Path(args.dataset).expanduser().resolve()
    meta = _wait_for_dataset(
        dataset, int(args.wait_seconds), float(args.poll_seconds)
    )
    _validate_dataset(
        dataset,
        meta,
        Path(args.mix),
        require_all_families=not bool(args.allow_partial_family_coverage),
    )
    if args.source_run:
        initialization = _source_champion(str(args.source_run))
        initialization["kind"] = "pure_rl_champion"
        initialization["source_run"] = str(args.source_run)
    elif args.init_hotstart_run:
        initialization = _wait_for_hotstart(
            str(args.init_hotstart_run),
            int(args.wait_seconds),
            float(args.poll_seconds),
        )
        initialization["kind"] = "hotstart_checkpoint"
        initialization["source_run"] = str(args.init_hotstart_run)
    else:
        initialization = _explicit_checkpoint(Path(args.init_checkpoint))
        initialization["kind"] = "explicit_checkpoint"

    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_bootstrap.py"),
        "--jsonl",
        str(dataset),
        "--archetype",
        "top-ladder-core",
        "--run-name",
        str(args.run_name),
        "--model-profile",
        "pure-rl",
        "--epochs",
        str(int(args.epochs)),
        "--lr",
        str(float(args.learning_rate)),
        "--games-per-batch",
        str(int(args.games_per_batch)),
        "--max-decisions-per-batch",
        str(int(args.max_decisions_per_batch)),
        "--val-frac",
        "0.10",
        "--split-by-episode",
        "--patience",
        "2",
        "--aux-loss-weight",
        "0",
        "--opp-hand-loss-weight",
        "0",
        "--opp-remainder-loss-weight",
        "0",
        "--min-usable-record-frac",
        "0.98",
        "--min-decisions",
        str(int(args.min_decisions)),
        "--seed",
        str(int(args.seed)),
    ]
    latest = checkpoint.latest_path(args.run_name)
    if latest.is_file():
        command.extend(("--resume", "auto"))
    else:
        command.extend(
            ("--resume", "0", "--init-checkpoint", initialization["path"])
        )

    run_meta_path = paths.OUTPUTS_DIR / "hotstart" / f"{args.run_name}.json"
    run_meta = {
        "schema": "poke_bot.top_ladder_hotstart_run/v1",
        "run_name": args.run_name,
        "source_run": args.source_run or args.init_hotstart_run,
        "source_champion": initialization,
        "dataset": str(dataset),
        "dataset_sha256": str(meta["output_sha256"]),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training",
        "command": command,
    }
    _atomic_json(run_meta_path, run_meta)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        run_meta.update(
            {
                "status": "failed",
                "returncode": completed.returncode,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(run_meta_path, run_meta)
        return int(completed.returncode)

    result_path = paths.OUTPUTS_DIR / "train" / f"{args.run_name}_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    best_path = Path(str(result.get("best_path") or ""))
    if not best_path.is_file():
        raise RuntimeError("hot-start training completed without a best checkpoint")
    from scripts.train_pure_rl import _checkpoint_contract

    contract = _checkpoint_contract(best_path, smoke=False)
    run_meta.update(
        {
            "status": "ready_for_rl",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_checkpoint": str(best_path),
            "best_metric": result.get("best_metric"),
            "checkpoint_contract": contract,
        }
    )
    _atomic_json(run_meta_path, run_meta)
    print(json.dumps(run_meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
