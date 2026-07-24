#!/usr/bin/env python3
"""Run the exact 8x250 Alakazam strong-public gate plus 4x250 controls."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "ops/alakazam_gate_program_v1.json",
    )
    parser.add_argument("--mode", choices=("specialist",), default="specialist")
    parser.add_argument("--specialist-archetype", default="alakazam")
    parser.add_argument("--measurement-decks", default="alakazam")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--game-timeout-s", type=int, default=600)
    parser.add_argument(
        "--remote-worker-endpoints",
        default="192.168.1.143:8765,bert.local:8766",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "alakazam_strong_public_gate_result.json"
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/strong_public_gate/history"
        ),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "alakazam_strong_public_gate.lock"
        ),
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _assert_production_stopped() -> None:
    service = subprocess.run(
        ["systemctl", "--user", "is-active", "pokebot-pure-rl-alakazam.service"],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: list[str] = []
    if service.stdout.strip() == "active":
        rows.append("systemd service pokebot-pure-rl-alakazam.service is active")
    # Match real Python argv entries only; the launching shell contains the
    # command text too and is not a production process.
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for cmdline in proc_root.glob("[0-9]*/cmdline"):
            try:
                argv = [
                    item.decode(errors="replace")
                    for item in cmdline.read_bytes().split(b"\0")
                    if item
                ]
            except (OSError, PermissionError):
                continue
            if not argv or "python" not in Path(argv[0]).name.lower():
                continue
            if not any(Path(arg).name == "launch_pure_rl.py" for arg in argv[1:]):
                continue
            if not any("pure_rl_alakazam" in arg for arg in argv[1:]):
                continue
            rows.append(f"pid={cmdline.parent.name} argv={' '.join(argv)}")
    if rows:
        raise RuntimeError(
            "continuous Alakazam production is still active; the exact gate "
            "requires exclusive checkpoint publication:\n" + "\n".join(rows)
        )


def main() -> int:
    args = _args()
    _assert_production_stopped()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = args.lock.open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another strong-public gate owns the lock") from exc

    os.environ.setdefault(
        "PURE_RL_PROGRESS_LOG",
        "/home/inzi/poke-bot-agent/outputs/logs/"
        "alakazam_strong_public_gate.progress.log",
    )
    os.environ["POKEBOT_PRIMARY_ARCHETYPE"] = args.specialist_archetype
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    os.environ["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"

    import torch

    from poke_bot.baselines_runtime import (
        baseline_content_digest,
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.checkpoint import checkpoint_digest
    from poke_bot.pure_rl.hardware import full_hardware_profile
    from poke_bot.pure_rl.multi_env_self_play import pure_rl_leaf_coalesce_ms
    from poke_bot.pure_rl.strong_public_gate import (
        build_strong_public_gate_result,
        load_active_gate_contract,
        verify_roster_content,
    )
    from poke_bot.remote_jobs import RemoteWorkerFarm
    from poke_bot.remote_sim_jobs import (
        remote_play_job,
        remote_self_play_job,
    )
    from scripts.train_pure_rl import (
        _LeafFarm,
        _hard_gate_publish_weights,
        _heldout_eval,
        _our_decks,
        _remote_heldout_capability_audit,
        _select_measurement_decks,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = checkpoint_digest(checkpoint)
    contract = load_active_gate_contract(args.contract.expanduser().resolve())
    gate = contract["next_gate"]
    configured_output = Path(str(gate.get("exact_result_pointer") or ""))
    if configured_output != args.output:
        raise RuntimeError(
            f"output pointer mismatch: contract={configured_output} cli={args.output}"
        )

    ensure_baselines_installed()
    loadable, failed = filter_loadable_baselines(load_manifest())
    if failed:
        print(f"[strong_gate] ignored unrelated unloadable baselines={len(failed)}")
    by_id = {spec.id: spec for spec in loadable}
    gate_ids = tuple(str(row["opponent_id"]) for row in gate["roster"])
    research_ids = tuple(
        str(row["opponent_id"]) for row in gate["research_measurements"]
    )
    missing = [key for key in (*gate_ids, *research_ids) if key not in by_id]
    if missing:
        raise RuntimeError(f"required gate packages are unavailable: {missing}")
    gate_specs = [by_id[key] for key in gate_ids]
    research_specs = [by_id[key] for key in research_ids]
    verify_roster_content(
        gate,
        {spec.id: baseline_content_digest(spec.path) for spec in gate_specs},
    )

    decks = _our_decks(args.mode, args.specialist_archetype)
    measurement_decks = _select_measurement_decks(decks, args.measurement_decks)
    if [name for name, _deck in measurement_decks] != ["alakazam"]:
        raise RuntimeError("strong-public gate must use only the Alakazam deck")

    hw = full_hardware_profile()
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hw.validate_or_raise(visible_gpu_count=visible)
    workers = int(args.workers) if int(args.workers) > 0 else int(hw.sim_workers)
    endpoints = [
        value.strip()
        for value in str(args.remote_worker_endpoints).split(",")
        if value.strip()
    ]
    leaf = _LeafFarm()
    remote_farm = None
    publish_proof: dict = {}
    started = time.time()
    try:
        leaf.start(
            ckpt=checkpoint,
            digest=digest,
            leaf_devices=hw.leaf_cuda_devices(),
            n_workers=workers,
            max_batch=None,
            coalesce_ms=pure_rl_leaf_coalesce_ms(default=0.0),
        )
        if endpoints:
            timeout = max(120.0, float(args.game_timeout_s) + 90.0)
            remote_farm = RemoteWorkerFarm(endpoints, timeout_s=timeout)
            infos = remote_farm.connect(require_all=False)
            connected = [str(info.endpoint) for info in infos]
            if remote_farm.clients:
                capability = _remote_heldout_capability_audit(
                    remote_farm, required_endpoints=connected
                )
                if capability.get("passed") is not True:
                    raise RuntimeError(
                        "connected formal-eval remote failed capability audit: "
                        + json.dumps(capability, sort_keys=True)
                    )
            else:
                remote_farm.close()
                remote_farm = None
                connected = []
        else:
            connected = []
        publish_proof = _hard_gate_publish_weights(
            leaf=leaf,
            remote_farm=remote_farm,
            ckpt=checkpoint,
            digest=digest,
            version=int(time.time()),
            required_endpoints=connected,
            reload_local=False,
        )
        gate_seed = 19_000_000 + int(args.iteration) * 100_000
        research_seed = gate_seed + 50_000
        gate_rows, gate_audit = _heldout_eval(
            ckpt=checkpoint,
            digest=digest,
            n_games=int(gate["evaluation"]["games_total"]),
            decks=measurement_decks,
            official_specs=gate_specs,
            seed=gate_seed,
            game_timeout_s=int(args.game_timeout_s),
            n_workers=workers,
            leaf_channel=leaf.remote_channel,
            remote_farm=remote_farm,
            worker_play=remote_play_job,
            worker_self_play=remote_self_play_job,
            mode=args.mode,
            allow_remote_play=remote_farm is not None,
            iteration=int(args.iteration),
            gate_wr=float(gate["pass_criteria"]["skill_weighted_win_rate"]),
            opponent_ids=gate_ids,
            stage_label="heldout:strong_public_gate",
        )
        research_rows, research_audit = _heldout_eval(
            ckpt=checkpoint,
            digest=digest,
            n_games=sum(
                int(row["games"]) for row in gate["research_measurements"]
            ),
            decks=measurement_decks,
            official_specs=research_specs,
            seed=research_seed,
            game_timeout_s=int(args.game_timeout_s),
            n_workers=workers,
            leaf_channel=leaf.remote_channel,
            remote_farm=remote_farm,
            worker_play=remote_play_job,
            worker_self_play=remote_self_play_job,
            mode=args.mode,
            allow_remote_play=remote_farm is not None,
            iteration=int(args.iteration),
            gate_wr=float(
                gate["pass_criteria"]["accepted_official_holdout_non_regression"]
            ),
            opponent_ids=research_ids,
            stage_label="measure:research_controls",
        )
        result = build_strong_public_gate_result(
            contract=contract,
            checkpoint=str(checkpoint),
            checkpoint_digest=digest,
            iteration=int(args.iteration),
            gate_rows=gate_rows,
            gate_audit=gate_audit,
            research_rows=research_rows,
            research_audit=research_audit,
            gate_seed=gate_seed,
            research_seed=research_seed,
            bootstrap_resamples=int(args.bootstrap_resamples),
        )
        result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_sec"] = time.time() - started
        result["checkpoint_publish_proof"] = publish_proof
        history = args.history_dir / (
            f"iter_{int(args.iteration):05d}_{digest.split(':', 1)[-1][:12]}.json"
        )
        if history.exists():
            existing = json.loads(history.read_text(encoding="utf-8"))
            if existing != result:
                raise FileExistsError(f"immutable gate history already differs: {history}")
        else:
            _atomic_json(history, result)
        _atomic_json(args.output, result)
        print(
            "[strong_gate] COMPLETE "
            f"iteration={args.iteration} weighted_wr={result['skill_weighted_wr']:.4f} "
            f"lower={result['confidence_lower']:.4f} "
            f"research_wr={result['research_controls']['pooled_wr']:.4f} "
            f"passed={result['passed']} output={args.output}",
            flush=True,
        )
        return 0
    finally:
        leaf.stop()
        if remote_farm is not None:
            try:
                remote_farm.close()
            except Exception:
                pass
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
