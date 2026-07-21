#!/usr/bin/env python3
"""Benchmark model-driven remote self-play throughput on one endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.checkpoint import checkpoint_digest
from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint
from scripts.train_pure_rl import _our_decks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=910_000)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only a compact completion line; the full report still goes to --json-out.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Atomically write the final report, including gameplay fingerprints.",
    )
    parser.add_argument(
        "--deck",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit 60-card CSV for the benchmark (repeat for two or more). "
            "Avoids depending on a checkout's default deck registry."
        ),
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Use the endpoint's current model identity without a reload request.",
    )
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    digest = checkpoint_digest(checkpoint_path)
    host, port = parse_endpoint(args.endpoint)
    games = max(1, int(args.games))
    concurrency = max(1, min(int(args.concurrency), games))
    if args.deck:
        from poke_bot.deck_pool import read_deck

        decks = [
            (f"benchmark:{path.stem}", read_deck(path.expanduser().resolve()))
            for path in args.deck
        ]
    else:
        decks = _our_decks("core")
    if len(decks) < 2:
        raise RuntimeError("benchmark requires at least two core decks")

    clients: list[RemoteJobClient] = []
    for _ in range(concurrency):
        client = RemoteJobClient(host, port, timeout_s=float(args.timeout + 120))
        client.connect()
        clients.append(client)
    if not args.no_reload:
        clients[0].reload_checkpoint(str(checkpoint_path), digest=digest, version=91_000)
    clients[0].pin_checkpoint(str(checkpoint_path), digest=digest)
    before = clients[0].health()

    pending: queue.Queue[int] = queue.Queue()
    for index in range(games):
        pending.put(index)
    rows: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()

    def consume(client: RemoteJobClient) -> None:
        while True:
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            our_arch, our_deck = decks[index % len(decks)]
            opp_arch, opp_deck = decks[(index + 1) % len(decks)]
            job = {
                "job_index": index,
                "checkpoint": str(checkpoint_path),
                "checkpoint_digest": digest,
                "opponent_checkpoint": str(checkpoint_path),
                "opponent_checkpoint_digest": digest,
                "opponent_id": "benchmark:self",
                "model_generation": 91_000,
                "model_max_context": 320,
                "our_deck": list(our_deck),
                "opp_deck": list(opp_deck),
                "our_seat": index % 2,
                "mcts_sims": 0,
                "mcts_move_time": 0.0,
                "game_timeout_s": int(args.timeout),
                "agent_mode": "policy",
                "sample_actions": False,
                "action_temperature": 1.0,
                "seed": int(args.seed + index),
                "device": "cpu",
                "training_eligible": True,
                "archetype": our_arch,
                "opp_archetype": opp_arch,
                "collect_both_seats": True,
                "target_provenance": {
                    "pure_rl": True,
                    "throughput_benchmark": True,
                    "soft_policy_targets": False,
                },
            }
            try:
                row = client.submit_job(job, kind="self_play")
                with lock:
                    rows.append(row)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append(f"job={index} {type(exc).__name__}: {exc}")
            finally:
                pending.task_done()

    started = time.perf_counter()
    threads = [threading.Thread(target=consume, args=(client,)) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    after = clients[0].health()
    for client in clients:
        client.close()

    valid = [
        row
        for row in rows
        if not row.get("error")
        and not row.get("resource_error")
        and not row.get("our_failed")
    ]
    trajectories = sum(
        len(row.get("record_jsons") or []) or int(bool(row.get("record_json")))
        for row in valid
    )
    decisions = sum(int(row.get("n_decisions") or 0) for row in valid)
    game_fingerprints: dict[str, str] = {}
    game_summaries: dict[str, dict] = {}
    for row in valid:
        action_traces = []
        for encoded in list(row.get("record_jsons") or []):
            try:
                record = json.loads(encoded) if isinstance(encoded, str) else encoded
            except (TypeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            action_traces.append(
                [
                    list(step.get("action") or [])
                    for step in list(record.get("steps") or [])
                    if isinstance(step, dict)
                ]
            )
        index = str(int(row.get("job_index") or 0))
        semantic = {
            "job_index": int(index),
            "winner": row.get("winner"),
            "steps": row.get("steps"),
            "action_traces": action_traces,
        }
        encoded_semantic = json.dumps(
            semantic, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        game_fingerprints[index] = hashlib.sha256(encoded_semantic).hexdigest()
        game_summaries[index] = {
            "winner": row.get("winner"),
            "steps": row.get("steps"),
            "n_decisions": row.get("n_decisions"),
            "trajectory_count": len(action_traces),
        }

    report = {
        "endpoint": args.endpoint,
        "checkpoint_digest": digest,
        "games_requested": games,
        "games_completed": len(valid),
        "concurrency": concurrency,
        "elapsed_s": elapsed,
        "games_per_s": len(valid) / max(elapsed, 1e-9),
        "decisions_per_s": decisions / max(elapsed, 1e-9),
        "trajectories_per_s": trajectories / max(elapsed, 1e-9),
        "mean_decisions_per_game": decisions / max(len(valid), 1),
        "usable_game_fraction": len(valid) / games,
        "errors": errors,
        "health_jobs_delta": int(after.get("jobs_completed") or 0)
        - int(before.get("jobs_completed") or 0),
        "tree_rss_gb": after.get("tree_rss_gb"),
        "free_ram_gb": after.get("free_ram_gb"),
        "leaf_alive": after.get("leaf_alive"),
        "game_fingerprints": game_fingerprints,
        "game_summaries": game_summaries,
    }
    if args.json_out is not None:
        output_path = args.json_out.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_path)
    if args.quiet:
        print(
            json.dumps(
                {
                    "games_completed": report["games_completed"],
                    "games_per_s": report["games_per_s"],
                    "decisions_per_s": report["decisions_per_s"],
                    "elapsed_s": report["elapsed_s"],
                    "errors": len(report["errors"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if len(valid) == games and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
