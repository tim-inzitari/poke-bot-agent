#!/usr/bin/env python3
"""Measure an isolated remote worker with exact direct-policy self-play jobs."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.checkpoint import checkpoint_digest
from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint
from scripts.train_pure_rl import _our_decks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=36)
    parser.add_argument("--concurrency", type=int, default=36)
    parser.add_argument("--pack", type=int, choices=(1, 4), default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=286_000)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    digest = checkpoint_digest(checkpoint)
    decks = _our_decks("alakazam")
    if not decks:
        raise RuntimeError("Alakazam deck resolver returned no decks")
    archetype, deck = decks[0]
    host, port = parse_endpoint(args.endpoint)
    games = max(1, int(args.games))
    pack = int(args.pack)
    if games % pack:
        raise ValueError("games must be divisible by pack")

    def job(index: int) -> dict:
        return {
            "job_index": index,
            "checkpoint": str(checkpoint),
            "checkpoint_digest": digest,
            "opponent_checkpoint": str(checkpoint),
            "opponent_id": "r286-elmo-throughput-canary",
            "model_generation": 0,
            "model_max_context": 320,
            "our_deck": list(deck),
            "opp_deck": list(deck),
            "our_seat": index % 2,
            "mcts_sims": 0,
            "mcts_move_time": 0.0,
            "game_timeout_s": int(args.timeout),
            "agent_mode": "policy",
            "sample_actions": False,
            "action_temperature": 1.0,
            "seed": int(args.seed) + index,
            "device": "cpu",
            "training_eligible": False,
            "archetype": archetype,
            "opp_archetype": archetype,
            "own_deck_ledger_enabled": True,
            "target_provenance": {
                "remote_canary": True,
                "training_eligible": False,
                "behavior_checkpoint_digest": digest,
                "soft_policy_targets": False,
            },
        }

    groups = [list(range(start, start + pack)) for start in range(0, games, pack)]

    def run_group(indices: list[int]) -> dict:
        started = time.perf_counter()
        client = RemoteJobClient(
            host,
            port,
            timeout_s=float(args.timeout + 120),
            connect_timeout_s=30,
            control_timeout_s=120,
        )
        try:
            client.connect()
            if pack == 1:
                results = [client.submit_job(job(indices[0]), kind="self_play")]
            else:
                results = client.submit_self_play_multi([job(i) for i in indices])
            return {
                "indices": indices,
                "elapsed_s": time.perf_counter() - started,
                "games": len(results),
                "trajectories": sum(
                    len(result.get("record_jsons") or ())
                    or int(bool(result.get("record_json")))
                    for result in results
                ),
                "decisions": sum(int(result.get("n_decisions") or 0) for result in results),
                "errors": [str(result.get("error")) for result in results if result.get("error")],
            }
        finally:
            client.close()

    started = time.perf_counter()
    results: list[dict] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max(1, int(args.concurrency)), len(groups))
    ) as pool:
        futures = [pool.submit(run_group, group) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started
    completed = sum(int(row["games"]) for row in results)
    payload = {
        "endpoint": args.endpoint,
        "checkpoint_digest": digest,
        "pack": pack,
        "requested_games": games,
        "completed_games": completed,
        "failed_groups": len(failures),
        "failures": failures,
        "wall_seconds": elapsed,
        "games_per_second": completed / elapsed if elapsed else 0.0,
        "trajectories": sum(int(row["trajectories"]) for row in results),
        "decisions": sum(int(row["decisions"]) for row in results),
        "group_elapsed_min": min((float(row["elapsed_s"]) for row in results), default=None),
        "group_elapsed_max": max((float(row["elapsed_s"]) for row in results), default=None),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if completed == games and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
