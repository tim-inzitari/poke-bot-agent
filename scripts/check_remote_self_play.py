#!/usr/bin/env python3
"""End-to-end remote self-play canary with two independently mapped checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.checkpoint import checkpoint_digest
from poke_bot.remote_jobs import (
    RemoteJobClient,
    cache_exact_resident_remote_checkpoint,
    parse_endpoint,
    prepare_remote_play_job,
)
from scripts.train_pure_rl import _our_decks


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoints",
        default="192.168.1.143:8765,bert.local:8766",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=73_001)
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Sequential real games per endpoint (default: 1)",
    )
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="Verify reload + active-digest pin without playing a game.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help=(
            "Reload even when every live leaf already proves the exact healthy "
            "checkpoint digest. Intended only for explicit reload testing."
        ),
    )
    return parser.parse_args()


def _exact_resident_checkpoint(
    health: dict[str, object],
    digest: str,
) -> bool:
    """Return true only for a complete, healthy proof of the resident digest."""

    leaves = list(health.get("leaves") or [])
    pinned = {str(value) for value in health.get("pinned_digests") or ()}
    return bool(
        health.get("ok") is True
        and health.get("controller_healthy") is True
        and health.get("leaf_alive") is True
        and health.get("leaf_identity_ok") is True
        and health.get("checkpoint_digest") == digest
        and digest in pinned
        and leaves
        and all(
            isinstance(row, dict)
            and row.get("healthy") is True
            and row.get("checkpoint_digest") == digest
            for row in leaves
        )
    )


def main() -> int:
    args = _args()
    current = args.checkpoint.expanduser().resolve()
    opponent = args.opponent_checkpoint.expanduser().resolve()
    for path in (current, opponent):
        if not path.is_file():
            raise FileNotFoundError(path)
    current_digest = checkpoint_digest(current)
    opponent_digest = checkpoint_digest(opponent)
    decks = _our_decks("core")
    if len(decks) < 2:
        raise RuntimeError("remote canary needs two valid local decks")

    failures = []
    for endpoint_i, endpoint in enumerate(
        item.strip() for item in args.endpoints.split(",") if item.strip()
    ):
        host, port = parse_endpoint(endpoint)
        client = RemoteJobClient(host, port, timeout_s=float(args.timeout + 120))
        try:
            info = client.connect()
            control_t0 = time.perf_counter()
            resident_health = client.health()
            reload_skipped_exact_health = bool(
                not args.force_reload
                and _exact_resident_checkpoint(
                    resident_health,
                    current_digest,
                )
            )
            if not reload_skipped_exact_health and not args.force_reload:
                raise RuntimeError(
                    "resident checkpoint health is incomplete or mismatched; "
                    "refusing an implicit multi-minute reload (use "
                    "--force-reload only for an intentional reload test): "
                    + json.dumps(
                        {
                            "checkpoint_digest": resident_health.get(
                                "checkpoint_digest"
                            ),
                            "checkpoint_version": resident_health.get(
                                "checkpoint_version"
                            ),
                            "controller_healthy": resident_health.get(
                                "controller_healthy"
                            ),
                            "leaf_alive": resident_health.get("leaf_alive"),
                            "leaf_identity_ok": resident_health.get(
                                "leaf_identity_ok"
                            ),
                            "pinned_digests": resident_health.get(
                                "pinned_digests"
                            ),
                        },
                        sort_keys=True,
                    )
                )
            if reload_skipped_exact_health:
                # Seed the process-local digest-addressed mapping from the
                # strict leaf proof. This prevents prepare_remote_play_job from
                # re-reading the already verified SMB object for ~141 seconds.
                cache_exact_resident_remote_checkpoint(
                    host,
                    str(current),
                    digest=current_digest,
                )
            if reload_skipped_exact_health:
                reload_reply = {
                    "ok": True,
                    "checkpoint_digest": current_digest,
                    "version": resident_health.get("checkpoint_version"),
                }
                pin_reply = {"ok": True, "checkpoint_digest": current_digest}
                reload_s = time.perf_counter() - control_t0
                pin_s = 0.0
            else:
                reload_reply = client.reload_checkpoint(
                    str(current), digest=current_digest, version=90_000 + endpoint_i
                )
                reload_s = time.perf_counter() - control_t0
                pin_t0 = time.perf_counter()
                pin_reply = client.pin_checkpoint(
                    str(current), digest=current_digest
                )
                pin_s = time.perf_counter() - pin_t0
            our_arch, our_deck = decks[endpoint_i % len(decks)]
            opp_arch, opp_deck = decks[(endpoint_i + 1) % len(decks)]
            job = {
                "job_index": endpoint_i,
                "checkpoint": str(current),
                "checkpoint_digest": current_digest,
                "opponent_checkpoint": str(opponent),
                "opponent_id": f"canary:{opponent.name}",
                "model_generation": int(
                    reload_reply.get("version") or 90_000 + endpoint_i
                ),
                "model_max_context": 320,
                "our_deck": list(our_deck),
                "opp_deck": list(opp_deck),
                "our_seat": endpoint_i % 2,
                "mcts_sims": 0,
                "mcts_move_time": 0.0,
                "game_timeout_s": int(args.timeout),
                "agent_mode": "policy",
                "sample_actions": False,
                "action_temperature": 1.0,
                "seed": int(args.seed + endpoint_i),
                "device": "cpu",
                "training_eligible": True,
                "archetype": our_arch,
                "opp_archetype": opp_arch,
                "target_provenance": {
                    "pure_rl": True,
                    "remote_canary": True,
                    "behavior_checkpoint_digest": current_digest,
                    "opponent_checkpoint_digest": opponent_digest,
                    "soft_policy_targets": False,
                },
            }
            mapped = prepare_remote_play_job(host, job)
            if args.control_only:
                health = client.health()
                summary = {
                    "endpoint": endpoint,
                    "hostname": info.hostname,
                    "reload_digest": reload_reply.get("checkpoint_digest"),
                    "pin_digest": pin_reply.get("checkpoint_digest"),
                    "reload_s": reload_s,
                    "pin_s": pin_s,
                    "reload_skipped_exact_health": reload_skipped_exact_health,
                    "mapped_checkpoint": mapped.get("checkpoint"),
                    "mapped_opponent_checkpoint": mapped.get(
                        "opponent_checkpoint"
                    ),
                    "health_digest": health.get("checkpoint_digest"),
                    "leaf_alive": health.get("leaf_alive"),
                }
                print(json.dumps(summary, sort_keys=True), flush=True)
                if (
                    reload_reply.get("checkpoint_digest") != current_digest
                    or pin_reply.get("checkpoint_digest") != current_digest
                    or health.get("checkpoint_digest") != current_digest
                    or not health.get("leaf_alive")
                ):
                    failures.append(summary)
                continue
            # Keep one connection and one verified leaf identity while driving
            # enough real tasks to exercise maxtasksperchild recycling. This is
            # intentionally sequential: it validates replacement generations
            # without conflating lifecycle bugs with socket concurrency.
            n_games = max(1, int(args.games))
            completed = 0
            trajectory_count = 0
            winners: dict[str, int] = {}
            result: dict = {}
            game_failed = False
            for game_i in range(n_games):
                game_job = {
                    **job,
                    "job_index": endpoint_i * n_games + game_i,
                    "our_seat": (endpoint_i + game_i) % 2,
                    "seed": int(args.seed + endpoint_i * n_games + game_i),
                }
                # submit_job performs the authoritative per-host mapping itself;
                # ``mapped`` is retained only for the diagnostic summary.
                result = client.submit_job(game_job, kind="self_play")
                record_rows = list(result.get("record_jsons") or [])
                if not record_rows and result.get("record_json"):
                    record_rows = [result["record_json"]]
                completed += 1
                trajectory_count += len(record_rows)
                winner = str(result.get("winner"))
                winners[winner] = winners.get(winner, 0) + 1
                if (
                    not record_rows
                    or bool(result.get("our_failed"))
                    or bool(result.get("resource_error"))
                    or result.get("error")
                ):
                    game_failed = True
                    break
            health = client.health()
            summary = {
                "endpoint": endpoint,
                "hostname": info.hostname,
                "job_kinds": list(info.job_kinds),
                "reload_digest": reload_reply.get("checkpoint_digest"),
                "pin_digest": pin_reply.get("checkpoint_digest"),
                "reload_s": reload_s,
                "pin_s": pin_s,
                "reload_skipped_exact_health": reload_skipped_exact_health,
                "mapped_checkpoint": mapped.get("checkpoint"),
                "mapped_opponent_checkpoint": mapped.get("opponent_checkpoint"),
                "winner": result.get("winner"),
                "winners": winners,
                "steps": result.get("steps"),
                "n_decisions": result.get("n_decisions"),
                "games_requested": n_games,
                "games_completed": completed,
                "trajectory_count": trajectory_count,
                "our_failed": bool(result.get("our_failed")),
                "resource_error": bool(result.get("resource_error")),
                "error": result.get("error"),
                "health": {
                    key: health.get(key)
                    for key in (
                        "checkpoint_digest",
                        "checkpoint_version",
                        "jobs_completed",
                        "jobs_failed",
                        "leaf_alive",
                    )
                },
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
            if (
                reload_reply.get("checkpoint_digest") != current_digest
                or pin_reply.get("checkpoint_digest") != current_digest
                or completed != n_games
                or trajectory_count < n_games
                or game_failed
                or summary["our_failed"]
                or summary["resource_error"]
                or summary["error"]
            ):
                failures.append(summary)
        except Exception as exc:  # noqa: BLE001 - canary must report every endpoint
            failure = {
                "endpoint": endpoint,
                "exception": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(json.dumps(failure, sort_keys=True), flush=True)
        finally:
            client.close()
    if failures:
        print(json.dumps({"ok": False, "failures": failures}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, "endpoints": args.endpoints}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
