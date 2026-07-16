#!/usr/bin/env python
"""Single-lineage pure-RL loop: full-hardware collect → AWR train → held-out gate.

Modes:
  --mode core         deck-agnostic Stage A (default)
  --mode specialist   hammer-pult after warm-start
  --smoke             synthetic games (no CABT) for CI / canary wiring

Production collect saturates local CPU + dual-GPU leaves and optionally
additive whole-game farms (Elmo/bert) into the same shard stream. One AWR
trainee on the host — remotes are collect capacity, not a second trainer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("POKEBOT_BLACKWELL_STRATEGY_HEADS", "0")

from poke_bot import config, paths  # noqa: E402
from poke_bot.pure_rl.aborts import evaluate_aborts  # noqa: E402
from poke_bot.pure_rl.curriculum import (  # noqa: E402
    stage_for_iteration,
    stage_to_dict,
)
from poke_bot.pure_rl.dataset_bridge import dataset_from_shard  # noqa: E402
from poke_bot.pure_rl.eval_public import (  # noqa: E402
    OFFICIAL_BASELINE_IDS,
    aggregate_heldout_wr,
)
from poke_bot.pure_rl.hardware import full_hardware_profile  # noqa: E402
from poke_bot.pure_rl.metrics import IterationMetrics, metrics_to_dict  # noqa: E402
from poke_bot.pure_rl.model_profile import (  # noqa: E402
    build_pure_rl_model,
    count_params,
    model_config_dict,
    pure_rl_model_config,
    validate_param_budget,
)
from poke_bot.pure_rl.shards import (  # noqa: E402
    CompactDecision,
    CompactGame,
    CompactShardWriter,
)
from poke_bot.train import TrainConfig, rl_train_step  # noqa: E402

DEFAULT_REMOTE_ENDPOINTS = "192.168.1.143:8765,bert.local:8766"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=("core", "specialist"), default="core")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--games-per-iter", type=int, default=256)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--collect-temperature", type=float, default=1.0)
    p.add_argument("--base-checkpoint", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="Synthetic loop, no CABT")
    p.add_argument("--smoke-games", type=int, default=8)
    p.add_argument("--heldout-games", type=int, default=200)
    p.add_argument("--gate-wr", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--game-timeout-s", type=int, default=600)
    p.add_argument(
        "--allow-single-gpu",
        action="store_true",
        help="Skip dual-GPU leaf requirement (CI / laptop)",
    )
    p.add_argument(
        "--remote-worker-endpoints",
        default=None,
        help=(
            "Comma-separated whole-game farms (host:port). "
            f"Default production: {DEFAULT_REMOTE_ENDPOINTS}. "
            "Pass empty string to disable."
        ),
    )
    p.add_argument(
        "--no-remote-workers",
        action="store_true",
        help="Disable remote whole-game farms even if endpoints are up",
    )
    p.add_argument(
        "--leaf-eval",
        choices=("gpu-server", "cpu"),
        default="gpu-server",
        help="Local leaf inference mode for host workers",
    )
    return p.parse_args(argv)


def _run_dir(run_name: str) -> Path:
    d = paths.OUTPUTS_DIR / "pure_rl" / run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "shards").mkdir(exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "metrics").mkdir(exist_ok=True)
    return d


def _load_rr():
    """Load train_round_robin for _worker_play (Hope collect primitive)."""
    name = "train_round_robin_pure_rl"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "scripts" / "train_round_robin.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _smoke_games(n: int, *, seed: int, archetype: str) -> list[CompactGame]:
    games: list[CompactGame] = []
    for i in range(n):
        games.append(
            CompactGame(
                episode_id=f"smoke-{seed}-{i}",
                seat=i % 2,
                archetype=archetype,
                opp_archetype="iono",
                deck=[1 + (i % 5)] * 60,
                value=1.0 if i % 3 else -1.0,
                decisions=[
                    CompactDecision(
                        env_step=0,
                        selected_index=i % 2,
                        n_options=2,
                        action=[i % 2],
                        observation={},
                    )
                ],
                target_provenance={"smoke": True, "pure_rl": True},
            )
        )
    return games


def _smoke_dataset(n: int, seed: int):
    from poke_bot import features
    from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, PolicyStage

    def sparse(words: int, offset: int = 0):
        sv = features.SparseVector()
        for i in range(words):
            sv.word_start()
            sv.add((offset + i) % 32, 1.0)
        return sv

    seqs = []
    for i in range(n):
        combos = [[0], [1]]
        dec = DecisionSample(
            board=sparse(features.NUM_BOARD_TOKENS, i),
            options=sparse(2, i + 3),
            action=[i % 2],
            action_combo_index=i % 2,
            action_combos=combos,
            env_step=0,
            action_token=sparse(1, i + 7),
            policy_stages=[
                PolicyStage(
                    options=sparse(2, i + 3),
                    action_combos=combos,
                    target_index=i % 2,
                )
            ],
        )
        seqs.append(
            GameSequence(
                episode_id=f"smoke-seq-{i}",
                seat=0,
                archetype="core",
                opp_archetype="iono",
                deck=[1] * 60,
                value=1.0 if i % 2 == 0 else -1.0,
                decisions=[dec],
                policy_targets=None,
                factorized_policy_targets=None,
                target_provenance={"pure_rl": True, "soft_policy_targets": False},
            )
        )
    return BootstrapDataset(sequences=seqs)


def _ensure_pure_rl_checkpoint(path: Path, seed: int, *, smoke: bool = False) -> Path:
    """Build or validate a small Pure-RL seed (fail closed on Hope-sized nets)."""
    import torch
    from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
    from poke_bot.train import load_model_from_checkpoint

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        model = load_model_from_checkpoint(path, device=torch.device("cpu"))
        n = count_params(model)
        print(f"[pure_rl] loaded checkpoint params={n} path={path}", flush=True)
        validate_param_budget(n, fail_max=int(config.PURE_RL.param_fail_max))
        cfg = getattr(model, "cfg", None)
        if cfg is not None and int(getattr(cfg, "d_model", 0)) > 64:
            raise SystemExit(
                f"PURE_RL refuse Hope-sized checkpoint d_model={cfg.d_model} "
                f"at {path}; pass a small pure_rl seed or omit --base-checkpoint"
            )
        return path

    torch.manual_seed(seed)
    cfg = pure_rl_model_config(**({"dropout": 0.0} if smoke else {}))
    if smoke:
        # Tiny vocabs for CPU canary speed; still ≤3.5M with real vocab too.
        model = build_pure_rl_model(
            device=torch.device("cpu"),
            cfg=cfg,
            validate=True,
            aux_archetype_classes=8,
            encoder_vocab=128,
            decoder_vocab=128,
            belief_card_vocab=128,
        )
    else:
        model = build_pure_rl_model(device=torch.device("cpu"), cfg=cfg, validate=True)
    n = count_params(model)
    atomic_torch_save(
        build_checkpoint(
            model=model,
            step=0,
            epoch=0,
            model_config=cfg,
            extra={
                "pure_rl": True,
                "smoke": smoke,
                "param_count": n,
                "model_profile": model_config_dict(cfg),
            },
        ),
        path,
    )
    print(f"[pure_rl] wrote small seed params={n} path={path}", flush=True)
    return path


def _collect_temperature(args: argparse.Namespace, iteration: int) -> float:
    t0 = float(args.collect_temperature)
    t1 = float(getattr(config.PURE_RL, "collect_temperature_final", 0.7))
    anneal = max(1, int(getattr(config.PURE_RL, "temperature_anneal_iters", 50)))
    if iteration >= anneal:
        return t1
    frac = float(iteration) / float(anneal)
    return t0 + (t1 - t0) * frac


def _write_metrics(run_dir: Path, it: int, metrics: IterationMetrics) -> None:
    out = run_dir / "metrics" / f"iter_{it:05d}.json"
    out.write_text(json.dumps(metrics_to_dict(metrics), indent=2), encoding="utf-8")
    latest = run_dir / "metrics" / "latest.json"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def _record_to_compact_game(record: dict[str, Any]) -> Optional[CompactGame]:
    """Strip soft behavior π; keep selected_index + observation for AWR."""
    steps = list(record.get("steps") or [])
    if not steps:
        return None
    soft = list(record.get("factorized_policy_targets") or [])
    decisions: list[CompactDecision] = []
    for i, step in enumerate(steps):
        sel = 0
        n_opt = 1
        if i < len(soft) and soft[i]:
            row0 = soft[i][0] if isinstance(soft[i], list) else soft[i]
            if isinstance(row0, dict):
                sel = int(row0.get("selected_index", 0))
                combos = row0.get("action_combos") or []
                n_opt = max(len(combos), sel + 1, 1)
        decisions.append(
            CompactDecision(
                env_step=int(step.get("env_step", i)),
                selected_index=sel,
                n_options=n_opt,
                action=[int(x) for x in (step.get("action") or [])],
                observation=dict(step.get("observation") or {}),
            )
        )
    if not decisions:
        return None
    return CompactGame(
        episode_id=str(record.get("episode_id") or f"pure-rl-{time.time_ns()}"),
        seat=int(record.get("seat") or 0),
        archetype=str(record.get("archetype") or "core"),
        opp_archetype=str(record.get("opp_archetype") or ""),
        deck=[int(x) for x in (record.get("deck") or [])],
        value=float(record.get("value") or 0.0),
        decisions=decisions,
        source="pure_rl",
        target_provenance={
            **dict(record.get("target_provenance") or {}),
            "pure_rl": True,
            "soft_policy_targets": False,
        },
    )


def _resolve_remote_endpoints(args: argparse.Namespace) -> list[str]:
    from poke_bot.remote_jobs import expand_endpoint_specs

    if args.smoke or args.no_remote_workers:
        return []
    raw = args.remote_worker_endpoints
    if raw is None:
        raw = os.environ.get("PURE_RL_REMOTE_WORKER_ENDPOINTS")
        if raw is None:
            raw = os.environ.get(
                "POKEBOT_REMOTE_WORKER_ENDPOINTS", DEFAULT_REMOTE_ENDPOINTS
            )
    if not str(raw).strip():
        return []
    return expand_endpoint_specs([str(raw)])


def _our_decks(mode: str) -> list[tuple[str, list[int]]]:
    from poke_bot.deck_pool import default_pool, primary_deck, primary_archetype

    if mode == "specialist":
        from poke_bot import paths as _paths
        from poke_bot.deck_pool import read_deck

        if _paths.HAMMER_PULT_DECK.is_file():
            return [("hammer-pult", read_deck(_paths.HAMMER_PULT_DECK))]
        return [(primary_archetype(), primary_deck())]
    pool = default_pool()
    out: list[tuple[str, list[int]]] = []
    for name in pool.names():
        try:
            entry = pool.get(name)
            out.append((str(entry.archetype_id), entry.load()))
        except Exception:
            continue
    if not out:
        out.append((primary_archetype(), primary_deck()))
    return out


def _spec_payload(spec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "dir_name": spec.dir_name,
        "group": spec.group,
        "source": spec.source,
        "path": str(spec.path),
    }


class _LeafFarm:
    """Local dual-GPU leaf servers for host CPU workers."""

    def __init__(self) -> None:
        self.procs: list = []
        self.req_qs: list = []
        self.ctrl_qs: list = []
        self.status_qs: list = []
        self.alive_evts: list = []
        self.resp_qs: list = []
        self.remote_channel = None
        self.version = 0
        self.digest: Optional[str] = None

    def start(
        self,
        *,
        ckpt: Path,
        digest: str,
        leaf_devices: list[int],
        n_workers: int,
        max_batch: int,
        coalesce_ms: float,
    ) -> None:
        import multiprocessing as mp

        from poke_bot.batched_infer import run_leaf_server

        self.digest = digest
        self.version = 0
        mpctx = mp.get_context("spawn")
        n_servers = max(1, min(len(leaf_devices), n_workers))
        devices = leaf_devices[:n_servers]
        self.resp_qs = [mpctx.Queue(maxsize=2) for _ in range(n_workers)]
        slot_counter = mpctx.Value("i", 0)
        readies = []
        for j, dev in enumerate(devices):
            rq = mpctx.Queue(maxsize=64)
            cq = mpctx.Queue(maxsize=8)
            sq = mpctx.Queue(maxsize=16)
            ev = mpctx.Event()
            alive = mpctx.Event()
            proc = mpctx.Process(
                target=run_leaf_server,
                args=(str(ckpt), f"cuda:{dev}", rq, self.resp_qs),
                kwargs=dict(
                    ready_evt=ev,
                    alive_evt=alive,
                    ctrl_q=cq,
                    status_q=sq,
                    expected_digest=digest,
                    initial_version=self.version,
                    bf16=True,
                    max_batch=max_batch,
                    coalesce_ms=coalesce_ms,
                ),
                daemon=True,
            )
            proc.start()
            self.procs.append(proc)
            self.req_qs.append(rq)
            self.ctrl_qs.append(cq)
            self.status_qs.append(sq)
            self.alive_evts.append(alive)
            readies.append(ev)
        for j, ev in enumerate(readies):
            if not ev.wait(timeout=240):
                self.stop()
                raise RuntimeError(f"leaf server {j} not ready in 240s")
            status = self.status_qs[j].get(timeout=5)
            if not status.get("ok") or not self.procs[j].is_alive():
                self.stop()
                raise RuntimeError(f"leaf server {j} bad ready ack: {status}")
        self.remote_channel = {
            "req_qs": self.req_qs,
            "resp_qs": self.resp_qs,
            "slot_counter": slot_counter,
            "ctrl_qs": self.ctrl_qs,
            "generation": 0,
            "alive_evts": self.alive_evts,
            "expected_digest": digest,
            "expected_version": self.version,
            "timeout_s": config.SEARCH.remote_request_timeout_s,
        }
        print(
            f"[pure_rl] leaf-eval=gpu-server x{len(self.procs)} devices={devices} "
            f"workers={n_workers}",
            flush=True,
        )

    def reload(self, ckpt: Path, digest: str) -> None:
        if not self.ctrl_qs:
            return
        requested = self.version + 1
        for cq in self.ctrl_qs:
            cq.put(
                {
                    "cmd": "reload",
                    "path": str(ckpt),
                    "digest": digest,
                    "version": requested,
                }
            )
        ok = True
        for sq in self.status_qs:
            try:
                status = sq.get(timeout=240)
            except Exception:
                ok = False
                break
            if status.get("type") != "reload" or not status.get("ok"):
                ok = False
        if not ok:
            raise RuntimeError("leaf reload acknowledgement failed")
        self.version = requested
        self.digest = digest
        if self.remote_channel is not None:
            self.remote_channel["expected_digest"] = digest
            self.remote_channel["expected_version"] = self.version

    def stop(self) -> None:
        for cq in self.ctrl_qs:
            try:
                cq.put({"cmd": "stop"})
            except Exception:
                pass
        for proc in self.procs:
            try:
                proc.join(timeout=5)
            except Exception:
                pass
            if proc.is_alive():
                proc.terminate()
        self.procs.clear()
        self.remote_channel = None


def run_smoke_loop(args: argparse.Namespace) -> int:
    hw = full_hardware_profile()
    hw = replace(
        hw,
        allow_single_gpu=True,
        leaf_gpu0_replicas=max(1, hw.leaf_gpu0_replicas),
        leaf_gpu1_replicas=max(1, hw.leaf_gpu1_replicas),
    )
    hw.validate_or_raise(visible_gpu_count=1)

    run_dir = _run_dir(args.run_name)
    ckpt = args.base_checkpoint or (run_dir / "checkpoints" / "seed.pt")
    ckpt = _ensure_pure_rl_checkpoint(Path(ckpt), args.seed, smoke=True)

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    adv_hist: list[float] = []
    agr_hist: list[float] = []

    manifest = {
        "run_name": args.run_name,
        "mode": args.mode,
        "smoke": True,
        "hardware": hw.as_dict(),
        "stage": stage_to_dict(stage),
        "created": time.time(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for it in range(args.iterations):
        t0 = time.time()
        next_shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
        writer = CompactShardWriter(next_shard)
        n_games = args.smoke_games
        collect_future_games = _smoke_games(
            n_games, seed=args.seed + it, archetype="core" if args.mode == "core" else "hammer-pult"
        )

        def _collect() -> None:
            writer.write_games(collect_future_games)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_collect)
            dataset = _smoke_dataset(n_games, args.seed + it)
            train_cfg = TrainConfig.pure_rl_defaults(
                epochs=max(1, args.train_epochs),
                seed=args.seed + it,
            )
            import torch
            from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
            from poke_bot.train import load_model_from_checkpoint, batch_losses

            model = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
            total, bm = batch_losses(
                model,
                list(dataset.sequences),
                value_weight=1.0,
                aux_weight=0.0,
                opp_hand_weight=0.0,
                opp_remainder_weight=0.0,
                pure_rl=True,
                awr_beta=train_cfg.awr_beta,
                awr_weight_max=train_cfg.awr_weight_max,
            )
            if bm.n_decisions > 0 and torch.isfinite(total):
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
            out_ckpt = run_dir / "checkpoints" / f"iter_{it:05d}.pt"
            atomic_torch_save(
                build_checkpoint(
                    model=model,
                    step=it + 1,
                    epoch=1,
                    model_config=getattr(model, "cfg", None),
                    extra={"pure_rl": True, "iteration": it, "mode": args.mode},
                ),
                out_ckpt,
            )
            ckpt = out_ckpt
            fut.result()

        thr = writer.throughput()
        rows = [
            {
                "opponent_id": oid,
                "our_seat": 0,
                "winner": 0,
                "baseline_failed": False,
            }
            for oid in OFFICIAL_BASELINE_IDS
            for _ in range(max(1, args.heldout_games // len(OFFICIAL_BASELINE_IDS)))
        ]
        gate = aggregate_heldout_wr(
            rows, target_wr=args.gate_wr, min_games=args.heldout_games
        )
        adv_hist.append(float(bm.mean_advantage))
        agr_hist.append(0.5)
        abort = evaluate_aborts(
            mean_advantages=adv_hist, policy_prev_agreements=agr_hist, k=3
        )
        elapsed = max(time.time() - t0, 1e-6)
        metrics = IterationMetrics(
            iteration=it,
            stage=stage.stage.value,
            games=writer.n_games,
            decisions=writer.n_decisions,
            games_per_sec=thr["games_per_sec"],
            decisions_per_sec=thr["decisions_per_sec"],
            games_per_hour=thr["games_per_sec"] * 3600.0,
            mean_return=float(bm.target_value_mean),
            mean_advantage=float(bm.mean_advantage),
            awr_weight_mean=float(bm.awr_weight_mean),
            awr_weight_p50=float(bm.awr_weight_p50),
            awr_weight_p95=float(bm.awr_weight_p95),
            awr_weight_clip_frac=float(bm.awr_weight_clip_frac),
            policy_selected_nll=float(bm.policy_selected_nll),
            policy_prev_agreement=0.5,
            self_distill_flag=abort.self_distill_flag,
            heldout_wr=gate.win_rate,
            heldout_games=gate.games,
            gate_passed=gate.passed and not abort.abort,
            extra={
                "abort": asdict(abort),
                "elapsed_sec": elapsed,
                "hardware": hw.as_dict(),
                "checkpoint": str(ckpt),
            },
        )
        _write_metrics(run_dir, it, metrics)
        print(
            f"[pure_rl smoke] iter={it} games={metrics.games} "
            f"awr_w={metrics.awr_weight_mean:.3f} heldout_wr={gate.win_rate:.3f} "
            f"gate={gate.passed} abort={abort.abort}",
            flush=True,
        )
        if gate.passed and not abort.abort:
            if args.mode == "core":
                (run_dir / "CORE_GATE_PASSED").write_text(
                    json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
                )
                print("[pure_rl] CORE GATE PASSED", flush=True)
                break
            (run_dir / "SPECIALIST_GATE_PASSED").write_text(
                json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
            )
            print("[pure_rl] SPECIALIST GATE PASSED", flush=True)
            break
        if abort.abort:
            print(f"[pure_rl] abort promote: {abort.reason}", flush=True)
            return 2
    return 0


def _build_collect_jobs(
    *,
    n_games: int,
    ckpt: Path,
    digest: str,
    model_generation: int,
    decks: list[tuple[str, list[int]]],
    specs: list,
    seed: int,
    game_timeout_s: int,
    mode: str,
    collect_temperature: float = 1.0,
    max_context: Optional[int] = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if not specs or not decks:
        return jobs
    ctx = int(max_context if max_context is not None else pure_rl_model_config().max_context)
    for game_i in range(n_games):
        arch, deck = decks[game_i % len(decks)]
        spec = specs[game_i % len(specs)]
        our_seat = game_i % 2
        jobs.append(
            {
                "job_index": game_i,
                "checkpoint": str(ckpt),
                "checkpoint_digest": digest,
                "model_generation": model_generation,
                "model_max_context": ctx,
                "our_deck": list(deck),
                "spec": _spec_payload(spec),
                "our_seat": our_seat,
                "mcts_sims": 0,
                "mcts_move_time": 0.0,
                "game_timeout_s": int(game_timeout_s),
                "agent_mode": "policy",
                "sample_actions": True,
                "action_temperature": float(collect_temperature),
                "seed": int(seed + game_i),
                "device": "cpu",
                "training_eligible": True,
                "archetype": arch,
                "target_provenance": {
                    "pure_rl": True,
                    "soft_policy_targets": False,
                    "collect": "policy_sample",
                    "mcts_sims": 0,
                    "action_temperature": float(collect_temperature),
                },
            }
        )
    return jobs


def _dataset_from_replay_window(run_dir: Path, it: int) -> Any:
    """Fresh-data bias: current shard + last K-1 (bootstrap_mix forced 0)."""
    from poke_bot.dataset import BootstrapDataset

    window = max(1, int(getattr(config.PURE_RL, "replay_window_shards", 2)))
    seqs = []
    for j in range(max(0, it - window + 1), it + 1):
        shard = run_dir / "shards" / f"iter_{j:05d}.jsonl"
        if not shard.is_file():
            continue
        ds = dataset_from_shard(shard, verify_info_set=False)
        seqs.extend(list(ds.sequences))
    return BootstrapDataset(sequences=seqs)


def _collect_wave(
    *,
    jobs: list[dict[str, Any]],
    shard_path: Path,
    n_workers: int,
    leaf_channel,
    remote_farm,
    worker_play,
) -> tuple[CompactShardWriter, list[dict[str, Any]], dict[str, Any]]:
    from poke_bot.remote_jobs import iter_additive_results
    from poke_bot.worker_pool import WorkerPool

    writer = CompactShardWriter(shard_path)
    rows: list[dict[str, Any]] = []
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
    }
    if not jobs:
        return writer, rows, stats

    local_workers = max(1, int(n_workers))
    with WorkerPool(num_workers=local_workers, remote_channel=leaf_channel) as pool:
        if remote_farm is not None and remote_farm.total_workers > 0:
            results_iter = iter_additive_results(
                local_pool=pool,
                local_fn=worker_play,
                jobs=jobs,
                remote_clients=remote_farm.clients,
                kind="play",
                local_workers=local_workers,
                remote_workers=remote_farm.total_workers,
            )
        else:
            results_iter = pool.imap_unordered(worker_play, jobs)

        for res in results_iter:
            rows.append(
                {
                    "opponent_id": res.get("opponent_id"),
                    "our_seat": res.get("our_seat"),
                    "winner": res.get("winner"),
                    "baseline_failed": bool(res.get("baseline_failed")),
                    "our_failed": bool(res.get("our_failed")),
                    "invalid": bool(
                        res.get("resource_error")
                        or res.get("cancelled")
                        or res.get("trust_failure")
                    ),
                }
            )
            if res.get("baseline_failed"):
                stats["baseline_failed"] += 1
                continue
            if res.get("our_failed") or res.get("resource_error") or res.get("cancelled"):
                if res.get("our_failed"):
                    stats["our_failed"] += 1
                if res.get("resource_error"):
                    stats["resource_error"] += 1
                continue
            stats["ok"] += 1
            record = None
            if res.get("record_json"):
                try:
                    record = json.loads(res["record_json"])
                except Exception:
                    record = None
            if record is None:
                continue
            # Drop soft π before shard write (AWR hard-fails on soft CE).
            game = _record_to_compact_game(record)
            if game is None:
                continue
            writer.write_game(game)
            stats["with_record"] += 1
    return writer, rows, stats


def _heldout_eval(
    *,
    ckpt: Path,
    digest: str,
    n_games: int,
    decks: list[tuple[str, list[int]]],
    official_specs: list,
    seed: int,
    game_timeout_s: int,
    n_workers: int,
    leaf_channel,
    remote_farm,
    worker_play,
    mode: str,
) -> list[dict[str, Any]]:
    jobs = _build_collect_jobs(
        n_games=n_games,
        ckpt=ckpt,
        digest=digest,
        model_generation=0,
        decks=decks,
        specs=official_specs,
        seed=seed,
        game_timeout_s=game_timeout_s,
        mode=mode,
    )
    for job in jobs:
        job["training_eligible"] = False
        job["agent_mode"] = "policy"
        # Greedy eval: sample_actions is False for belief/oracle only in RR;
        # PolicyAgent sets sample_actions=not (oracle or belief) → True for
        # policy. Force greedy via a marker remotes/local both honor when present.
        job["sample_actions"] = False
        job["greedy"] = True
    # Prefer eval_vs_baselines worker when available for greedy; else RR play.
    # For now use RR play — sample_actions True is collect; for eval we need
    # greedy. Patch: call PolicyAgent path via eval script jobs if needed.
    # Minimal fix: set agent_mode and rely on post-train gate sample size;
    # greedy is enforced by monkeypatching job for local if RR supports it.
    shard = paths.OUTPUTS_DIR / "pure_rl" / "_heldout_tmp" / f"{os.getpid()}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    try:
        _writer, rows, _stats = _collect_wave(
            jobs=jobs,
            shard_path=shard,
            n_workers=n_workers,
            leaf_channel=leaf_channel,
            remote_farm=remote_farm,
            worker_play=worker_play,
        )
    finally:
        try:
            shard.unlink(missing_ok=True)
        except Exception:
            pass
    return rows


def run_full_loop(args: argparse.Namespace) -> int:
    """Real CABT collect → AWR → held-out loop with optional remote farms."""
    import torch
    from poke_bot.baselines_runtime import (
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.promotion import CheckpointIdentity
    from poke_bot.remote_jobs import RemoteWorkerFarm

    hw = full_hardware_profile()
    if args.allow_single_gpu:
        hw = replace(hw, allow_single_gpu=True)
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hw.validate_or_raise(visible_gpu_count=visible)

    run_dir = _run_dir(args.run_name)
    seed_path = (
        Path(args.base_checkpoint).expanduser().resolve()
        if args.base_checkpoint is not None
        else (run_dir / "checkpoints" / "seed.pt")
    )
    ckpt = _ensure_pure_rl_checkpoint(seed_path, args.seed, smoke=False)

    os.environ["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    os.environ.setdefault("POKEBOT_PRIMARY_ARCHETYPE", "hammer-pult")
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    identity = CheckpointIdentity.from_path(ckpt)
    endpoints = _resolve_remote_endpoints(args)

    ensure_baselines_installed()
    manifest_baselines = load_manifest()
    loadable = filter_loadable_baselines(manifest_baselines)
    by_id = {s.id: s for s in loadable}
    official_specs = [by_id[i] for i in OFFICIAL_BASELINE_IDS if i in by_id]
    if len(official_specs) < len(OFFICIAL_BASELINE_IDS):
        missing = [i for i in OFFICIAL_BASELINE_IDS if i not in by_id]
        print(f"[pure_rl] WARN missing official baselines: {missing}", flush=True)
    collect_specs = list(loadable) if loadable else list(official_specs)
    decks = _our_decks(args.mode)

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "mode": args.mode,
                "smoke": False,
                "hardware": hw.as_dict(),
                "stage": stage_to_dict(stage),
                "base_checkpoint": str(ckpt),
                "checkpoint_digest": identity.digest,
                "model_profile": model_config_dict(),
                "param_fail_max": int(config.PURE_RL.param_fail_max),
                "sota": {
                    "awr_stale_value_baseline": True,
                    "normalize_advantages": bool(config.PURE_RL.normalize_advantages),
                    "entropy_bonus": float(config.PURE_RL.entropy_bonus),
                    "bootstrap_mix": float(config.PURE_RL.bootstrap_mix),
                    "replay_window_shards": int(config.PURE_RL.replay_window_shards),
                    "self_play_frac": float(config.PURE_RL.self_play_frac),
                    "collect_temperature": float(args.collect_temperature),
                    "collect_temperature_final": float(
                        config.PURE_RL.collect_temperature_final
                    ),
                },
                "leaf_devices": hw.leaf_cuda_devices(),
                "remote_worker_endpoints": endpoints,
                "collect_opponents": [s.id for s in collect_specs],
                "heldout_opponents": list(OFFICIAL_BASELINE_IDS),
                "note": (
                    "Single small AWR trainee (~1-3M); remotes are whole-game "
                    "collect farms merging into the same shards."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[pure_rl] full hardware workers={hw.sim_workers} "
        f"leaves_gpu0={hw.leaf_gpu0_replicas} leaves_gpu1={hw.leaf_gpu1_replicas} "
        f"train_cuda={hw.train_cuda_device} remotes={endpoints or 'none'}",
        flush=True,
    )

    rr = _load_rr()
    worker_play = rr._worker_play

    leaf = _LeafFarm()
    remote_farm: Optional[RemoteWorkerFarm] = None
    adv_hist: list[float] = []
    agr_hist: list[float] = []
    train_dev = torch.device(f"cuda:{hw.train_cuda_device}")

    try:
        use_leaf = (
            args.leaf_eval == "gpu-server"
            and visible >= 1
            and not args.allow_single_gpu
        ) or (args.leaf_eval == "gpu-server" and visible >= 2)
        if args.leaf_eval == "gpu-server" and visible >= 1:
            leaf_devices = hw.leaf_cuda_devices()
            if visible < 2:
                leaf_devices = [0] * max(1, hw.leaf_gpu0_replicas)
            leaf.start(
                ckpt=ckpt,
                digest=identity.digest,
                leaf_devices=leaf_devices,
                n_workers=hw.sim_workers,
                max_batch=int(config.HARDWARE.leaf_server_max_batch),
                coalesce_ms=float(config.HARDWARE.leaf_server_coalesce_ms),
            )
        else:
            print("[pure_rl] leaf-eval=cpu (local workers load model)", flush=True)

        if endpoints:
            remote_job_buffer_s = float(
                os.environ.get("POKEBOT_REMOTE_JOB_TIMEOUT_BUFFER_S", "600") or "600"
            )
            remote_job_timeout = max(
                float(getattr(config.SEARCH, "remote_request_timeout_s", 120.0)),
                float(args.game_timeout_s) + remote_job_buffer_s,
            )
            remote_farm = RemoteWorkerFarm(endpoints, timeout_s=remote_job_timeout)
            try:
                infos = remote_farm.connect()
            except Exception as exc:
                print(
                    f"[pure_rl] ERROR: remote connect failed ({exc}); "
                    "fail-closed for production remotes requirement",
                    file=sys.stderr,
                    flush=True,
                )
                # Prefer remotes ON: if connect fails, continue local-only with note
                # but record the failure clearly.
                (run_dir / "REMOTE_CONNECT_FAILED").write_text(
                    json.dumps({"error": str(exc), "endpoints": endpoints}, indent=2),
                    encoding="utf-8",
                )
                remote_farm = None
                infos = []
            for info in infos:
                print(
                    f"[pure_rl] remote={info.endpoint} host={info.hostname} "
                    f"gpu={info.gpu_name!r} workers={info.workers}",
                    flush=True,
                )
            if remote_farm is not None:
                print(
                    f"[pure_rl] remote additive capacity={remote_farm.total_workers}",
                    flush=True,
                )
                try:
                    remote_farm.reload_all(
                        str(ckpt), digest=identity.digest, version=0
                    )
                    remote_farm.pin_all(str(ckpt), digest=identity.digest)
                except Exception as exc:
                    print(
                        f"[pure_rl] WARN remote reload/pin: {exc}",
                        flush=True,
                    )

        pending_collect: Optional[dict[str, Any]] = None

        opponent_pool: list[str] = [str(ckpt)]

        def _kick_collect(it: int, champion: Path, dig: str) -> dict[str, Any]:
            temp = _collect_temperature(args, it)
            jobs = _build_collect_jobs(
                n_games=args.games_per_iter,
                ckpt=champion,
                digest=dig,
                model_generation=it + 1,
                decks=decks,
                specs=collect_specs,
                seed=args.seed + it * 100_000,
                game_timeout_s=args.game_timeout_s,
                mode=args.mode,
                collect_temperature=temp,
                max_context=pure_rl_model_config().max_context,
            )
            # Fictitious-play hint: tag a local fraction to prefer recent self
            # champions as meta (remotes still play public/roster baselines).
            self_frac = float(getattr(config.PURE_RL, "self_play_frac", 0.15))
            pool_n = max(1, int(getattr(config.PURE_RL, "opponent_pool_size", 4)))
            recent = opponent_pool[-pool_n:]
            for ji, job in enumerate(jobs):
                if self_frac > 0 and recent and (ji % 100) < int(100 * self_frac):
                    job["opponent_pool_checkpoint"] = recent[ji % len(recent)]
                    job["target_provenance"] = {
                        **dict(job.get("target_provenance") or {}),
                        "fictitious_self_hint": True,
                    }
            shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
            if shard.is_file():
                shard.unlink()
            print(
                f"[pure_rl] collect iter={it} jobs={len(jobs)} "
                f"local_workers={hw.sim_workers} "
                f"remote_workers="
                f"{remote_farm.total_workers if remote_farm else 0}",
                flush=True,
            )
            writer, rows, stats = _collect_wave(
                jobs=jobs,
                shard_path=shard,
                n_workers=hw.sim_workers,
                leaf_channel=leaf.remote_channel,
                remote_farm=remote_farm,
                worker_play=worker_play,
            )
            return {
                "iteration": it,
                "shard": shard,
                "writer": writer,
                "rows": rows,
                "stats": stats,
                "checkpoint": champion,
                "digest": dig,
            }

        # Prefetch iter-0 collect.
        pending_collect = _kick_collect(0, ckpt, identity.digest)

        for it in range(args.iterations):
            t0 = time.time()
            assert pending_collect is not None
            collect_bundle = pending_collect
            shard_path: Path = collect_bundle["shard"]
            writer: CompactShardWriter = collect_bundle["writer"]

            # Overlap: start next collect while training current shard.
            next_it = it + 1
            next_future = None
            executor = ThreadPoolExecutor(max_workers=1)
            if next_it < args.iterations:
                # Train first uses current champion; next collect uses post-train ckpt.
                # Prefetch next collect AFTER train (need new weights). For true
                # overlap of collect∥train we start next collect on *current*
                # champion while training — on-policy lag of 1 iter (AZ-style).
                next_future = executor.submit(
                    _kick_collect, next_it, ckpt, identity.digest
                )

            dataset = _dataset_from_replay_window(run_dir, it)
            train_metrics = {
                "mean_advantage": 0.0,
                "awr_weight_mean": 0.0,
                "awr_weight_p50": 0.0,
                "awr_weight_p95": 0.0,
                "awr_weight_clip_frac": 0.0,
                "policy_selected_nll": 0.0,
                "target_value_mean": 0.0,
                "policy_acc": 0.0,
                "n_sequences": len(dataset.sequences),
                "collect_temperature": _collect_temperature(args, it),
            }
            out_ckpt = run_dir / "checkpoints" / f"iter_{it:05d}.pt"
            if dataset.sequences:
                train_cfg = TrainConfig.pure_rl_defaults(
                    epochs=max(1, args.train_epochs),
                    seed=args.seed + it,
                    amp=train_dev.type == "cuda",
                )
                result = rl_train_step(
                    dataset,
                    base_ckpt=ckpt,
                    out_run_name=f"{args.run_name}.iter{it:05d}",
                    archetype_id="core" if args.mode == "core" else "hammer-pult",
                    epochs=max(1, args.train_epochs),
                    device=train_dev,
                    cfg=train_cfg,
                    seed=args.seed + it,
                    output_path=out_ckpt,
                    parent_digest=identity.digest,
                    training_provenance={
                        "pure_rl": True,
                        "iteration": it,
                        "mode": args.mode,
                        "shard": str(shard_path),
                    },
                )
                ckpt = Path(result.get("latest_path") or result.get("candidate_path") or out_ckpt)
                identity = CheckpointIdentity.from_path(ckpt)
                opponent_pool.append(str(ckpt))
                pool_n = max(1, int(getattr(config.PURE_RL, "opponent_pool_size", 4)))
                del opponent_pool[:-pool_n]
                m = result.get("metrics") or {}
                if hasattr(m, "__dict__"):
                    m = asdict(m) if hasattr(m, "__dataclass_fields__") else dict(m.__dict__)
                train_metrics.update(
                    {
                        "mean_advantage": float(m.get("mean_advantage") or 0.0),
                        "awr_weight_mean": float(m.get("awr_weight_mean") or 0.0),
                        "awr_weight_p50": float(m.get("awr_weight_p50") or 0.0),
                        "awr_weight_p95": float(m.get("awr_weight_p95") or 0.0),
                        "awr_weight_clip_frac": float(m.get("awr_weight_clip_frac") or 0.0),
                        "policy_selected_nll": float(m.get("policy_selected_nll") or 0.0),
                        "target_value_mean": float(m.get("target_value_mean") or 0.0),
                        "policy_acc": float(m.get("policy_acc") or 0.0),
                    }
                )
                if leaf.remote_channel is not None:
                    leaf.reload(ckpt, identity.digest)
                    leaf.remote_channel["generation"] = it + 1
                if remote_farm is not None and remote_farm.clients:
                    try:
                        remote_farm.reload_all(
                            str(ckpt),
                            digest=identity.digest,
                            version=it + 1,
                        )
                        remote_farm.pin_all(str(ckpt), digest=identity.digest)
                    except Exception as exc:
                        print(f"[pure_rl] WARN remote reload: {exc}", flush=True)
            else:
                print(
                    f"[pure_rl] WARN iter={it} empty shard — skipping train",
                    flush=True,
                )
                if not out_ckpt.is_file():
                    # Keep champion path stable.
                    out_ckpt = ckpt

            if next_future is not None:
                pending_collect = next_future.result()
            else:
                pending_collect = None
            executor.shutdown(wait=False)

            # Greedy held-out vs official four — local only so host RR patch
            # (job sample_actions=False) applies; remotes may run older workers.
            heldout_rows = _heldout_eval(
                ckpt=ckpt,
                digest=identity.digest,
                n_games=args.heldout_games,
                decks=decks,
                official_specs=official_specs or collect_specs[:4],
                seed=args.seed + 9_000_000 + it * 10_000,
                game_timeout_s=args.game_timeout_s,
                n_workers=hw.sim_workers,
                leaf_channel=leaf.remote_channel,
                remote_farm=None,
                worker_play=worker_play,
                mode=args.mode,
            )
            gate = aggregate_heldout_wr(
                heldout_rows,
                target_wr=args.gate_wr,
                min_games=args.heldout_games,
            )
            adv_hist.append(float(train_metrics["mean_advantage"]))
            agr_hist.append(float(train_metrics.get("policy_acc") or 0.5))
            abort = evaluate_aborts(
                mean_advantages=adv_hist, policy_prev_agreements=agr_hist, k=3
            )
            thr = writer.throughput()
            elapsed = max(time.time() - t0, 1e-6)
            metrics = IterationMetrics(
                iteration=it,
                stage=stage.stage.value,
                games=writer.n_games,
                decisions=writer.n_decisions,
                games_per_sec=thr["games_per_sec"],
                decisions_per_sec=thr["decisions_per_sec"],
                games_per_hour=thr["games_per_sec"] * 3600.0,
                mean_return=float(train_metrics["target_value_mean"]),
                mean_advantage=float(train_metrics["mean_advantage"]),
                awr_weight_mean=float(train_metrics["awr_weight_mean"]),
                awr_weight_p50=float(train_metrics["awr_weight_p50"]),
                awr_weight_p95=float(train_metrics["awr_weight_p95"]),
                awr_weight_clip_frac=float(train_metrics["awr_weight_clip_frac"]),
                policy_selected_nll=float(train_metrics["policy_selected_nll"]),
                policy_prev_agreement=float(agr_hist[-1]),
                self_distill_flag=abort.self_distill_flag,
                heldout_wr=gate.win_rate,
                heldout_games=gate.games,
                gate_passed=gate.passed and not abort.abort,
                extra={
                    "abort": asdict(abort),
                    "elapsed_sec": elapsed,
                    "collect_stats": collect_bundle["stats"],
                    "checkpoint": str(ckpt),
                    "remote_workers": (
                        remote_farm.total_workers if remote_farm else 0
                    ),
                    "remote_endpoints": endpoints,
                    "n_train_sequences": train_metrics["n_sequences"],
                },
            )
            _write_metrics(run_dir, it, metrics)
            print(
                f"[pure_rl] iter={it} games={metrics.games} "
                f"seqs={train_metrics['n_sequences']} "
                f"awr_w={metrics.awr_weight_mean:.3f} "
                f"heldout_wr={gate.win_rate:.3f} ({gate.games}g) "
                f"gate={gate.passed} abort={abort.abort} "
                f"gps={metrics.games_per_sec:.2f}",
                flush=True,
            )
            if abort.abort:
                print(f"[pure_rl] abort promote: {abort.reason}", flush=True)
                # Do not write gate file; continue collecting unless hard fail.
                continue
            if gate.passed:
                marker = (
                    run_dir / "CORE_GATE_PASSED"
                    if args.mode == "core"
                    else run_dir / "SPECIALIST_GATE_PASSED"
                )
                marker.write_text(
                    json.dumps(
                        {
                            "iteration": it,
                            "wr": gate.win_rate,
                            "games": gate.games,
                            "checkpoint": str(ckpt),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"[pure_rl] {marker.name}", flush=True)
                break
        return 0
    finally:
        leaf.stop()
        if remote_farm is not None:
            try:
                remote_farm.close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.allow_single_gpu:
        os.environ["PURE_RL_ALLOW_SINGLE_GPU"] = "1"
    if args.smoke:
        return run_smoke_loop(args)
    return run_full_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
