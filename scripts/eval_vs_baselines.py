#!/usr/bin/env python
"""Strict draw-aware evaluation vs every expected baseline.

The trusted formal mode is the realized-history policy. Legacy single-world
MCTS is an explicit oracle diagnostic only. Runtime/baseline failures, missing
opponents, and fail-closed play invalidate results.

The default evaluates one configured deck.  ``--deck-suite core-ladder`` is the
deck-agnostic gate: it evaluates every pinned top-ladder representative with
exact deck/seat balance and reports both pooled and per-deck results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from poke_bot import checkpoint, config, deck_pool, paths
from poke_bot.agent import PolicyAgent, play_game
from poke_bot.baselines_runtime import (
    ensure_baselines_installed,
    filter_loadable_baselines,
    load_baseline_agent,
    load_manifest,
)
from poke_bot.device import leaf_eval_device
from poke_bot.eval_metrics import FieldReport
from poke_bot.train import load_model_from_checkpoint
from poke_bot.worker_pool import WorkerPool
from poke_bot.promotion import CheckpointIdentity


_WORKER_STATE: dict = {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--our-deck",
        type=Path,
        action="append",
        default=[],
        help=(
            "Explicit 60-card deck CSV; repeat for a custom multi-deck suite. "
            "Cannot be combined with --deck-suite core-ladder."
        ),
    )
    p.add_argument(
        "--deck-suite",
        choices=("primary", "core-ladder"),
        default="primary",
        help=(
            "Deck coverage contract. core-ladder uses all 17 pinned modal "
            "top-ladder representatives."
        ),
    )
    p.add_argument("--games-per-opp", type=int, default=8, help="Must be even for balanced seats.")
    p.add_argument(
        "--min-games-per-opp",
        type=int,
        default=0,
        help="Formal minimum sample per opponent (0 = --games-per-opp).",
    )
    p.add_argument("--workers", type=int, default=config.HARDWARE.sim_workers)
    p.add_argument("--mcts-sims", type=int, default=32)
    p.add_argument(
        "--agent-mode",
        choices=("policy", "oracle-mcts"),
        default="policy",
        help="Trusted history policy (default) or untrusted oracle diagnostic.",
    )
    p.add_argument("--greedy-ablation", action="store_true", help="Also run a separate paired greedy ablation.")
    p.add_argument("--greedy-games-per-opp", type=int, default=2)
    p.add_argument("--gate", type=float, default=0.55)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--only", nargs="+", help="Subset of baseline ids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--leaf-eval", choices=("gpu-server", "cpu"), default="gpu-server")
    p.add_argument("--leaf-gpu", default="auto")
    p.add_argument(
        "--leaf-max-batch",
        type=int,
        default=config.HARDWARE.leaf_server_max_batch,
    )
    p.add_argument(
        "--leaf-coalesce-ms",
        type=float,
        default=config.HARDWARE.leaf_server_coalesce_ms,
    )
    return p.parse_args(argv)


def _deck_digest(cards: list[int]) -> str:
    """Stable digest of the exact ordered 60-card engine input."""
    payload = ",".join(str(int(card_id)) for card_id in cards).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _resolve_our_decks(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Resolve and validate the requested deck coverage contract."""
    explicit_paths = [Path(path) for path in args.our_deck]
    if args.deck_suite == "core-ladder" and explicit_paths:
        raise ValueError(
            "--our-deck cannot be combined with --deck-suite core-ladder"
        )

    if args.deck_suite == "core-ladder":
        from poke_bot.ladder_deck_mix import (
            load_ladder_deck_mix,
            load_ladder_deck_representatives,
        )

        mix_path = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
        representatives_path = (
            ROOT
            / "data"
            / "training_mixes"
            / "top_ladder_representatives.v1.json"
        )
        mix = load_ladder_deck_mix(mix_path)
        representatives = load_ladder_deck_representatives(
            representatives_path
        )
        bound = representatives.bind(mix)
        decks = [
            {
                "deck_id": item.bucket.deck_id,
                "cards": list(item.card_ids),
                "path": None,
                "source": "pinned_top_ladder_modal_representative",
                "sha256": _deck_digest(list(item.card_ids)),
                "canonical_multiset_sha256": item.canonical_multiset_sha256,
                "source_rank": item.bucket.source_rank,
                "train_weight": item.bucket.train_weight,
            }
            for item in bound
        ]
        return decks, {
            "suite": "core-ladder",
            "deck_agnostic": True,
            "mix_path": str(mix_path.resolve()),
            "representatives_path": str(representatives_path.resolve()),
            "contract": representatives.contract(mix),
        }

    if explicit_paths:
        missing = [path for path in explicit_paths if not path.is_file()]
        if missing:
            raise ValueError(f"missing deck(s): {', '.join(map(str, missing))}")
        seen: dict[str, int] = {}
        decks = []
        for path in explicit_paths:
            resolved = path.resolve()
            base_id = resolved.stem
            seen[base_id] = seen.get(base_id, 0) + 1
            deck_id = (
                base_id if seen[base_id] == 1 else f"{base_id}-{seen[base_id]}"
            )
            cards = deck_pool.read_deck(resolved)
            decks.append(
                {
                    "deck_id": deck_id,
                    "cards": cards,
                    "path": str(resolved),
                    "source": "explicit",
                    "sha256": _deck_digest(cards),
                }
            )
        return decks, {
            "suite": "explicit",
            "deck_agnostic": len(decks) > 1,
        }

    cards = deck_pool.primary_deck()
    return [
        {
            "deck_id": deck_pool.primary_archetype(),
            "cards": cards,
            "path": None,
            "source": "configured_primary",
            "sha256": _deck_digest(cards),
        }
    ], {"suite": "primary", "deck_agnostic": False}


def _game_job(payload: dict) -> dict:
    """Worker entry: strict game with explicit failure attribution."""
    import random
    import signal

    from poke_bot import batched_infer
    from poke_bot import config as _config
    from poke_bot.agent import PolicyAgent, install_quiet_stdout, play_game
    from poke_bot.baselines_runtime import BaselineSpec, load_baseline_agent
    from poke_bot.train import load_model_from_checkpoint

    # Silence baseline/libcg stdout unless POKEBOT_AGENT_VERBOSE=1.
    install_quiet_stdout(_config.agent_verbose())

    base = {
        "opponent_id": payload["spec"]["id"],
        "our_deck_id": payload["our_deck_id"],
        "our_seat": int(payload["our_seat"]),
        "winner": 2,
        "steps": 0,
        "is_mirror": False,
        "mcts_on": not bool(payload.get("ablation", False)),
        "pair_id": None,
        "seed": int(payload["seed"]),
        "valid": False,
        "failure_attribution": None,
        "error": None,
    }
    try:
        import torch

        seed = int(payload["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        leaf_backend = batched_infer.remote_leaf_backend_from_worker()
        ckpt = payload["checkpoint"]
        device = payload.get("device", "cpu")
        if leaf_backend is not None:
            model = None
        else:
            key = f"{ckpt}|{device}"
            if _WORKER_STATE.get("model_key") != key:
                _WORKER_STATE["model"] = load_model_from_checkpoint(
                    ckpt, device=torch.device(device)
                )
                _WORKER_STATE["model_key"] = key
            model = _WORKER_STATE["model"]
        our_deck = payload["our_deck"]
        spec_d = dict(payload["spec"])
        spec_d["path"] = Path(spec_d["path"])
        spec = BaselineSpec(**spec_d)
        try:
            opp_fn, opp_deck = load_baseline_agent(spec)
        except Exception as exc:
            return {
                **base,
                "failure_attribution": "baseline",
                "error": f"load: {type(exc).__name__}: {exc}",
            }
        our_seat = int(payload["our_seat"])
        use_mcts = bool(payload["use_mcts"])
        agent = PolicyAgent(
            model=model,
            deck=our_deck,
            opponent_deck=opp_deck if use_mcts else None,
            use_mcts=use_mcts,
            oracle_mode=use_mcts,
            max_sims=int(payload["mcts_sims"]),
            rng=random.Random(seed),
            leaf_backend=leaf_backend,
            device=torch.device("cpu") if leaf_backend is not None else None,
            strict_runtime=True,
        )
        agent.reset_game()

        def _timeout(_signum, _frame):
            raise TimeoutError(f"eval game exceeded {payload['timeout_s']}s")

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(int(payload["timeout_s"]))
        try:
            if our_seat == 0:
                result = play_game(agent, opp_fn, our_deck, opp_deck)
            else:
                result = play_game(opp_fn, agent, opp_deck, our_deck)
        finally:
            if had_alarm:
                signal.alarm(0)
        if result.get("failed_seat") is not None:
            failed = int(result["failed_seat"])
            return {
                **base,
                "steps": int(result.get("steps", 0)),
                "is_mirror": sorted(our_deck) == sorted(opp_deck),
                "failure_attribution": "ours" if failed == our_seat else "baseline",
                "error": result.get("error"),
            }
        if result.get("incomplete"):
            return {
                **base,
                "steps": int(result.get("steps", 0)),
                "is_mirror": sorted(our_deck) == sorted(opp_deck),
                "failure_attribution": "infrastructure",
                "error": "game reached max_steps without a terminal result",
            }
        if agent.fail_closed_count:
            return {
                **base,
                "failure_attribution": "ours",
                "error": f"fail_closed_count={agent.fail_closed_count}",
            }
        return {
            **base,
            "valid": True,
            "winner": int(result["winner"]),
            "steps": int(result["steps"]),
            "is_mirror": sorted(our_deck) == sorted(opp_deck),
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            **base,
            "failure_attribution": "infrastructure",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _legacy_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    config.apply_runtime_perf()  # TF32 / cuDNN benchmark / thread pins.
    if args.games_per_opp % 2 != 0:
        print("ERROR: --games-per-opp must be even (seat-swap)", file=sys.stderr)
        return 2
    if not args.checkpoint.is_file():
        print(f"ERROR: missing checkpoint {args.checkpoint}", file=sys.stderr)
        return 2

    specs = ensure_baselines_installed(load_manifest())
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.id in wanted]
    specs, _dropped = filter_loadable_baselines(specs)
    if not specs:
        print("ERROR: no loadable baselines", file=sys.stderr)
        return 2
    try:
        our_deck = _resolve_our_decks(args)[0][0]["cards"]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    device = str(leaf_eval_device(prefer_name=config.HARDWARE.leaf_gpu_name))

    pairs = args.games_per_opp // 2
    jobs: list[dict] = []
    seed = args.seed
    for spec in specs:
        for pair_i in range(pairs):
            for our_seat in (0, 1):
                jobs.append(
                    {
                        "checkpoint": str(args.checkpoint),
                        "our_deck": our_deck,
                        "our_deck_id": "legacy",
                        "spec": {
                            "id": spec.id,
                            "name": spec.name,
                            "dir_name": spec.dir_name,
                            "group": spec.group,
                            "source": spec.source,
                            "path": str(spec.path),
                        },
                        "our_seat": our_seat,
                        "use_mcts": True,
                        "mcts_sims": args.mcts_sims,
                        "seed": seed,
                        "device": device,
                    }
                )
                seed += 1
                if args.greedy_ablation and pair_i == 0 and our_seat == 0:
                    # One greedy game per opp for ablation signal (cheap).
                    jobs.append(
                        {
                            "checkpoint": str(args.checkpoint),
                            "our_deck": our_deck,
                            "our_deck_id": "legacy",
                            "spec": {
                                "id": spec.id,
                                "name": spec.name,
                                "dir_name": spec.dir_name,
                                "group": spec.group,
                                "source": spec.source,
                                "path": str(spec.path),
                            },
                            "our_seat": 0,
                            "use_mcts": False,
                            "mcts_sims": 0,
                            "seed": seed,
                            "device": device,
                        }
                    )
                    seed += 1

    print(
        f"== eval_vs_baselines opps={len(specs)} jobs={len(jobs)} "
        f"games_per_opp={args.games_per_opp} workers={args.workers} "
        f"sims={args.mcts_sims} device={device} "
        f"leaf_batch={config.leaf_batch_for_device(device)}",
        flush=True,
    )
    print(
        f"   N note: {args.games_per_opp} games/opp (seat-swapped) × {len(specs)} "
        f"opponents = {args.games_per_opp * len(specs)} MCTS games "
        f"(+ greedy ablation extras if enabled).",
        flush=True,
    )

    report = FieldReport(gate_threshold=args.gate)
    # Group jobs by opponent for nested progress UX.
    by_opp: dict[str, list[dict]] = {}
    for j in jobs:
        by_opp.setdefault(j["spec"]["id"], []).append(j)

    # Serial-per-batch via WorkerPool for all jobs with a flat tqdm is simpler
    # and still shows live WR; nested bars over pool results are awkward.
    results: list[dict] = []
    with WorkerPool(num_workers=args.workers) as pool:
        bar = tqdm(total=len(jobs), desc="eval games", unit="game")
        wins = 0.0
        n = 0
        for res in pool.imap_unordered(_game_job, jobs):
            results.append(res)
            report.merge_game(
                res["opponent_id"],
                our_seat=res["our_seat"],
                winner=res["winner"],
                is_mirror=res["is_mirror"],
                mcts_on=res["mcts_on"],
            )
            n += 1
            if res["winner"] == 2:
                wins += 0.5
            elif res["winner"] == res["our_seat"]:
                wins += 1.0
            bar.update(1)
            opp_st = report.get(res["opponent_id"])
            bar.set_postfix(
                wr=f"{wins / max(n, 1):.1%}",
                n=n,
                last=(
                    f"{res['opponent_id'][:14]}="
                    f"{opp_st.win_rate:.0%}"
                    f"({opp_st.wins + 0.5 * opp_st.draws:g}/{opp_st.games})"
                ),
                pass_=len(report.opponents_passing()),
            )
        bar.close()

    summary = report.summary()
    out = args.out or (paths.OUTPUTS_DIR / "eval" / "vs_baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f">> wrote {out}", flush=True)
    print(
        f">> evaluated={summary['n_evaluated']} passing_wilson>={args.gate}: "
        f"{summary['n_passing_wilson']} all_pass={summary['all_pass']}",
        flush=True,
    )
    for row in summary["matchups"]:
        print(
            f"   {row['opponent_id']:32} games={row['games']:3} "
            f"wr={row['wr']:.1%} wilson_lo={row['wilson_lo']:.1%}",
            flush=True,
        )
    return 0 if summary["all_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    config.apply_runtime_perf()
    try:
        our_decks, deck_contract = _resolve_our_decks(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: invalid deck suite: {exc}", file=sys.stderr)
        return 2
    deck_seat_block = 2 * len(our_decks)
    min_games = int(args.min_games_per_opp or args.games_per_opp)
    if (
        args.games_per_opp < deck_seat_block
        or args.games_per_opp % deck_seat_block
        or min_games < deck_seat_block
        or min_games % deck_seat_block
        or (args.greedy_ablation and (
            args.greedy_games_per_opp < deck_seat_block
            or args.greedy_games_per_opp % deck_seat_block
        ))
    ):
        print(
            "ERROR: evaluation sample counts must be multiples of "
            f"{deck_seat_block} (one game per deck in each seat)",
            file=sys.stderr,
        )
        return 2
    if min_games > args.games_per_opp:
        print("ERROR: --min-games-per-opp exceeds scheduled games", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("ERROR: --workers must be >= 1", file=sys.stderr)
        return 2
    if not args.checkpoint.is_file():
        print(f"ERROR: missing checkpoint {args.checkpoint}", file=sys.stderr)
        return 2
    if args.agent_mode == "policy" and args.greedy_ablation:
        print(
            "ERROR: --greedy-ablation is only meaningful for explicit "
            "--agent-mode oracle-mcts diagnostics",
            file=sys.stderr,
        )
        return 2
    if args.agent_mode == "policy":
        try:
            checkpoint.assert_trusted_policy_checkpoint(args.checkpoint)
        except Exception as exc:
            print(f"ERROR: untrusted policy checkpoint: {exc}", file=sys.stderr)
            return 2
    elif not config.SEARCH.allow_oracle_deck:
        print(
            "ERROR: oracle-mcts requires explicit POKEBOT_ALLOW_ORACLE_DECK=1",
            file=sys.stderr,
        )
        return 2

    identity = CheckpointIdentity.from_path(args.checkpoint)
    manifest_specs = load_manifest()
    manifest_by_id = {s.id: s for s in manifest_specs}
    expected_ids = set(args.only or manifest_by_id)
    selected_specs = [
        manifest_by_id[oid] for oid in sorted(expected_ids) if oid in manifest_by_id
    ]
    unavailable: list[dict] = [
        {"opponent_id": oid, "error": "not present in manifest"}
        for oid in sorted(expected_ids - set(manifest_by_id))
    ]
    installed = []
    for spec in selected_specs:
        if spec.main_py.is_file() and spec.deck_csv.is_file():
            installed.append(spec)
        else:
            unavailable.append(
                {"opponent_id": spec.id, "error": "missing main.py or deck.csv"}
            )
    specs, dropped = filter_loadable_baselines(installed)
    unavailable.extend(
        {"opponent_id": sid, "error": error} for sid, error in dropped
    )

    report = FieldReport(
        gate_threshold=args.gate,
        expected_opponents=set(expected_ids),
        min_games_per_opponent=min_games,
    )
    per_deck_min_games = min_games // len(our_decks)
    deck_reports = {
        entry["deck_id"]: FieldReport(
            gate_threshold=args.gate,
            expected_opponents=set(expected_ids),
            min_games_per_opponent=per_deck_min_games,
        )
        for entry in our_decks
    }
    for item in unavailable:
        reason = (
            f"expected opponent unavailable: {item['opponent_id']}: "
            f"{item['error']}"
        )
        report.mark_invalid(reason)
        for deck_report in deck_reports.values():
            deck_report.mark_invalid(reason)
    jobs: list[dict] = []
    seed = int(args.seed)

    def _spec_payload(spec) -> dict:
        return {
            "id": spec.id,
            "name": spec.name,
            "dir_name": spec.dir_name,
            "group": spec.group,
            "source": spec.source,
            "path": str(spec.path),
        }

    for spec in specs:
        for game_i in range(args.games_per_opp):
            game_seed = seed
            seed += 1
            deck_entry = our_decks[(game_i // 2) % len(our_decks)]
            jobs.append(
                {
                    "checkpoint": identity.path,
                    "our_deck": deck_entry["cards"],
                    "our_deck_id": deck_entry["deck_id"],
                    "spec": _spec_payload(spec),
                    "our_seat": game_i % 2,
                    "use_mcts": args.agent_mode == "oracle-mcts",
                    "ablation": False,
                    "mcts_sims": args.mcts_sims,
                    "seed": game_seed,
                    "pair_id": None,
                    "device": "cpu",
                    "timeout_s": int(
                        os.environ.get("POKEBOT_GAME_TIMEOUT_S", "180")
                    ),
                }
            )
        if args.greedy_ablation:
            for game_i in range(args.greedy_games_per_opp):
                game_seed = seed
                seed += 1
                deck_entry = our_decks[(game_i // 2) % len(our_decks)]
                jobs.append(
                    {
                        "checkpoint": identity.path,
                        "our_deck": deck_entry["cards"],
                        "our_deck_id": deck_entry["deck_id"],
                        "spec": _spec_payload(spec),
                        "our_seat": game_i % 2,
                        "use_mcts": False,
                        "ablation": True,
                        "mcts_sims": 0,
                        "seed": game_seed,
                        "pair_id": None,
                        "device": "cpu",
                        "timeout_s": int(
                            os.environ.get("POKEBOT_GAME_TIMEOUT_S", "180")
                        ),
                    }
                )

    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    remote_channel = None
    server = req_q = ctrl_q = status_q = alive_evt = None
    resp_qs = []
    startup_error: str | None = None
    if args.leaf_eval == "gpu-server" and jobs:
        import multiprocessing as mp

        from poke_bot.batched_infer import run_leaf_server

        ctx = mp.get_context("spawn")
        req_q = ctx.Queue()
        ctrl_q = ctx.Queue()
        status_q = ctx.Queue()
        ready_evt = ctx.Event()
        alive_evt = ctx.Event()
        resp_qs = [ctx.Queue() for _ in range(args.workers)]
        slot_counter = ctx.Value("i", 0)
        leaf_device = (
            str(leaf_eval_device(prefer_name=config.HARDWARE.leaf_gpu_name))
            if args.leaf_gpu == "auto"
            else str(args.leaf_gpu)
        )
        server = ctx.Process(
            target=run_leaf_server,
            args=(identity.path, leaf_device, req_q, resp_qs),
            kwargs={
                "ready_evt": ready_evt,
                "alive_evt": alive_evt,
                "ctrl_q": ctrl_q,
                "status_q": status_q,
                "expected_digest": identity.digest,
                "initial_version": 0,
                "max_batch": args.leaf_max_batch,
                "coalesce_ms": args.leaf_coalesce_ms,
            },
            daemon=True,
        )
        server.start()
        if not ready_evt.wait(timeout=240):
            startup_error = "leaf server ready timeout"
        else:
            try:
                ready = status_q.get(timeout=5)
            except Exception as exc:
                ready = {"ok": False, "error": str(exc)}
            if (
                not ready.get("ok")
                or ready.get("checkpoint_digest") != identity.digest
                or not server.is_alive()
                or not alive_evt.is_set()
            ):
                startup_error = f"invalid leaf server ready ack: {ready}"
        if startup_error is None:
            remote_channel = {
                "req_qs": [req_q],
                "resp_qs": resp_qs,
                "slot_counter": slot_counter,
                "generation": 1,
                "alive_evts": [alive_evt],
                "expected_digest": identity.digest,
                "expected_version": 0,
                "timeout_s": config.SEARCH.remote_request_timeout_s,
            }
        else:
            report.mark_invalid(startup_error)
            for deck_report in deck_reports.values():
                deck_report.mark_invalid(startup_error)

    print(
        f"== strict eval expected={len(expected_ids)} available={len(specs)} "
        f"our_decks={len(our_decks)} suite={deck_contract['suite']} "
        f"formal_games/opp={args.games_per_opp} min={min_games} "
        f"jobs={len(jobs)} checkpoint={identity.digest[:23]}",
        flush=True,
    )

    results: list[dict] = []
    failures: list[dict] = []
    try:
        if startup_error is None and jobs:
            with WorkerPool(
                num_workers=args.workers,
                remote_channel=remote_channel,
            ) as pool:
                for res in tqdm(
                    pool.imap_unordered(_game_job, jobs),
                    total=len(jobs),
                    desc="eval games",
                    unit="game",
                ):
                    results.append(res)
                    if not res.get("valid"):
                        failures.append(res)
                        reason = (
                            f"{res.get('failure_attribution')} failure for "
                            f"{res['our_deck_id']} vs {res['opponent_id']}: "
                            f"{res.get('error')}"
                        )
                        report.mark_invalid(reason)
                        deck_reports[res["our_deck_id"]].mark_invalid(reason)
                        continue
                    report.merge_game(
                        res["opponent_id"],
                        our_seat=res["our_seat"],
                        winner=res["winner"],
                        is_mirror=res["is_mirror"],
                        mcts_on=res["mcts_on"],
                        pair_id=None,
                    )
                    deck_reports[res["our_deck_id"]].merge_game(
                        res["opponent_id"],
                        our_seat=res["our_seat"],
                        winner=res["winner"],
                        is_mirror=res["is_mirror"],
                        mcts_on=res["mcts_on"],
                        pair_id=None,
                    )
        if len(results) != (len(jobs) if startup_error is None else 0):
            report.mark_invalid(
                f"completed result count {len(results)}/{len(jobs)}"
            )
        if server is not None and startup_error is None:
            if not server.is_alive() or not alive_evt.is_set():
                report.mark_invalid("leaf server died during evaluation")
                for deck_report in deck_reports.values():
                    deck_report.mark_invalid("leaf server died during evaluation")
    finally:
        if server is not None:
            try:
                if ctrl_q is not None:
                    ctrl_q.put_nowait({"cmd": "stop"})
                if req_q is not None:
                    req_q.put_nowait(None)
                server.join(timeout=10)
                if server.is_alive():
                    server.terminate()
                    server.join(timeout=5)
            except Exception:
                pass
        for queue_obj in (req_q, ctrl_q, status_q, *resp_qs):
            if queue_obj is None:
                continue
            try:
                queue_obj.cancel_join_thread()
            except Exception:
                pass
            try:
                queue_obj.close()
            except Exception:
                pass

    summary = report.summary()
    deck_metadata = [
        {key: value for key, value in entry.items() if key != "cards"}
        for entry in our_decks
    ]
    per_deck_summaries = []
    for metadata in deck_metadata:
        deck_summary = deck_reports[metadata["deck_id"]].summary()
        per_deck_summaries.append({**metadata, "report": deck_summary})
    deck_reports_valid = all(
        row["report"]["valid"] for row in per_deck_summaries
    )
    deck_reports_pass = all(
        row["report"]["all_pass"] for row in per_deck_summaries
    )
    summary["valid"] = bool(summary["valid"] and deck_reports_valid)
    summary["all_pass"] = bool(summary["all_pass"] and deck_reports_pass)
    summary.update(
        {
            "checkpoint": identity.as_dict(),
            "expected_opponents": sorted(expected_ids),
            "unavailable_opponents": unavailable,
            "failures": failures,
            "scheduled_jobs": len(jobs),
            "completed_jobs": len(results),
            "formal_mode": args.agent_mode,
            "trusted_formal": args.agent_mode == "policy",
            "engine_seedable": False,
            "pairing_claimed": False,
            "greedy_is_ablation_only": args.agent_mode == "oracle-mcts",
            "our_deck": deck_metadata[0] if len(deck_metadata) == 1 else None,
            "deck_agnostic_gate": {
                **deck_contract,
                "deck_count": len(our_decks),
                "exact_deck_seat_balance": True,
                "games_per_deck_per_opponent": (
                    args.games_per_opp // len(our_decks)
                ),
                "minimum_games_per_deck_per_opponent": per_deck_min_games,
                "roster": deck_metadata,
                "all_deck_reports_valid": deck_reports_valid,
                "all_deck_reports_pass": deck_reports_pass,
                "per_deck": per_deck_summaries,
            },
        }
    )
    if args.agent_mode != "policy":
        summary["all_pass"] = False
        summary["promotion_eligible"] = False
        summary["diagnostic_warning"] = (
            "single-world oracle MCTS is not trusted formal evidence"
        )
    else:
        summary["promotion_eligible"] = bool(
            summary["valid"] and summary["all_pass"]
        )
    out = args.out or (paths.OUTPUTS_DIR / "eval" / "vs_baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f">> wrote {out} valid={summary['valid']} all_pass={summary['all_pass']} "
        f"pooled_formal={summary['pooled_formal']} "
        f"greedy_ablation={summary['greedy_ablation']}",
        flush=True,
    )
    for row in summary["matchups"]:
        wr = "n/a" if row["games"] == 0 else f"{row['wr']:.1%}"
        print(
            f"   {row['opponent_id']:32} games={row['games']:3} "
            f"wr={wr} draw_aware_lo="
            f"{row['draw_aware_score_interval']['lower']:.1%}",
            flush=True,
        )
    if len(per_deck_summaries) > 1:
        for row in per_deck_summaries:
            pooled = row["report"]["pooled_formal"]
            wr = "n/a" if pooled["wr"] is None else f"{pooled['wr']:.1%}"
            print(
                f"   deck={row['deck_id']:28} games={pooled['games']:3} "
                f"wr={wr} valid={row['report']['valid']} "
                f"pass={row['report']['all_pass']}",
                flush=True,
            )
    if not summary["valid"]:
        return 3
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
