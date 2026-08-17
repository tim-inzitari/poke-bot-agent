#!/usr/bin/env python
"""Generate exact hidden-hand/remainder targets with the private engine fork.

Every policy input remains the normal masked competition observation.  A
read-only training ABI snapshots the opponent hand/deck/prizes from the same
native state and writes those cards only under ``steps[].aux_labels``.

Output is resumable: each shard and sidecar are atomically published, and the
final manifest records source checkpoint, engine, deck-roster, and SHA-256
lineage.  The private engine is never a submission artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine-lib", type=Path, required=True)
    parser.add_argument(
        "--engine-source-dir",
        type=Path,
        default=None,
        help=(
            "Private engine source used for this build. Its deterministic tree "
            "digest is recorded so native binaries from different CPU "
            "architectures can be consolidated without losing provenance."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--games-per-shard", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--multi-env", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--game-timeout-s", type=int, default=600)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    """Hash a source tree independent of mtimes, owners, and host paths."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"engine source tree is empty: {path}")
    for item in files:
        rel = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _record_label_stats(record: dict[str, Any]) -> tuple[int, int]:
    labeled = 0
    decisions = 0
    seat = int(record["seat"])
    for step in record.get("steps") or []:
        decisions += 1
        obs = step.get("observation") or {}
        current = obs.get("current") or {}
        if int(current.get("yourIndex", seat)) != seat:
            raise RuntimeError("record contains a policy observation for wrong seat")
        players = current.get("players") or []
        if len(players) != 2 or players[1 - seat].get("hand") is not None:
            raise RuntimeError("privileged opponent hand leaked into policy input")
        aux = dict(step.get("aux_labels") or {})
        for key in ("opp_hand", "opp_deck_order", "opp_prizes"):
            cards = aux.get(key)
            if not isinstance(cards, list) or not all(
                isinstance(card, int) and not isinstance(card, bool)
                for card in cards
            ):
                raise RuntimeError(f"missing exact privileged label {key}")
        if (
            len(aux["opp_hand"])
            + len(aux["opp_deck_order"])
            + len(aux["opp_prizes"])
            > 60
        ):
            raise RuntimeError("privileged hidden zones exceed a legal deck")
        if aux.get("privileged_label_source") != "training_fork_exact_same_state":
            raise RuntimeError("untrusted privileged label source")
        labeled += 1
    return decisions, labeled


def _collect_shard(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["POKEBOT_LIBCG_PATH"] = str(payload["engine_lib"])
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"

    import torch

    torch.set_num_threads(1)
    from poke_bot.pure_rl.multi_env_self_play import run_self_play_multi

    output = Path(payload["output"])
    sidecar = output.with_suffix(".meta.json")
    expected_lineage = str(payload["lineage_digest"])
    if output.is_file() and sidecar.is_file():
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            existing.get("sha256") == _sha256(output)
            and existing.get("lineage_digest") == expected_lineage
            and int(existing.get("games", -1)) == len(payload["jobs"])
        ):
            return existing
        raise RuntimeError(f"existing shard failed lineage validation: {output}")

    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    decisions = 0
    labeled = 0
    records = 0
    games = 0
    t0 = time.monotonic()
    try:
        with partial.open("x", encoding="utf-8") as handle:
            multi_env = max(1, int(payload["multi_env"]))
            jobs = list(payload["jobs"])
            for offset in range(0, len(jobs), multi_env):
                batch = jobs[offset : offset + multi_env]
                results = run_self_play_multi(batch)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"multi-env returned {len(results)}/{len(batch)} games"
                    )
                for result in results:
                    if (
                        result.get("error")
                        or result.get("our_failed")
                        or result.get("resource_error")
                    ):
                        raise RuntimeError(
                            f"privileged self-play game failed: {result}"
                        )
                    rows = list(result.get("record_jsons") or [])
                    if len(rows) != 2:
                        raise RuntimeError(
                            "same-policy privileged game must emit both seats"
                        )
                    for raw in rows:
                        record = json.loads(raw)
                        n_decisions, n_labeled = _record_label_stats(record)
                        if n_decisions != n_labeled:
                            raise RuntimeError("not every decision has hidden targets")
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                        decisions += n_decisions
                        labeled += n_labeled
                        records += 1
                    games += 1
            handle.flush()
            os.fsync(handle.fileno())
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass

    meta = {
        "schema": "poke_bot.privileged_belief_shard/v1",
        "path": str(output.resolve()),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "lineage_digest": expected_lineage,
        "checkpoint_digest": payload["checkpoint_digest"],
        "engine_digest": payload["engine_digest"],
        "engine_source_digest": payload["engine_source_digest"],
        "hidden_export_digest": payload["hidden_export_digest"],
        "hidden_snapshot_abi": 1,
        "shard_index": int(payload["shard_index"]),
        "games": games,
        "records": records,
        "decisions": decisions,
        "hand_labeled_decisions": labeled,
        "elapsed_seconds": time.monotonic() - t0,
        "host": os.uname().nodename,
    }
    _atomic_json(sidecar, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.games < 1 or args.games_per_shard < 1:
        raise ValueError("game counts must be positive")
    if args.workers < 1 or args.multi_env < 1:
        raise ValueError("worker counts must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    engine_lib = args.engine_lib.expanduser().resolve()
    if not checkpoint_path.is_file() or not engine_lib.is_file():
        raise FileNotFoundError("checkpoint and hidden engine library are required")
    checkpoint.assert_trusted_policy_checkpoint(checkpoint_path)

    from scripts.train_pure_rl import _core_ladder_decks

    decks, mix, _representatives, deck_contract = _core_ladder_decks()
    checkpoint_digest = checkpoint.checkpoint_digest(checkpoint_path)
    engine_digest = _sha256(engine_lib)
    engine_source_digest = None
    if args.engine_source_dir is not None:
        engine_source_dir = args.engine_source_dir.expanduser().resolve()
        if not engine_source_dir.is_dir():
            raise FileNotFoundError(
                f"engine source directory does not exist: {engine_source_dir}"
            )
        engine_source_digest = _sha256_tree(engine_source_dir)
    hidden_export_digest = _sha256(ROOT / "engine_patches" / "HiddenExport.cpp")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "schema": "poke_bot.privileged_belief_lineage/v1",
        "checkpoint_digest": checkpoint_digest,
        "engine_digest": engine_digest,
        "engine_source_digest": engine_source_digest,
        "hidden_export_digest": hidden_export_digest,
        "hidden_snapshot_abi": 1,
        "deck_contract": deck_contract,
        "seed": int(args.seed),
        "temperature": float(args.temperature),
    }
    lineage_digest = "sha256:" + hashlib.sha256(
        json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    tasks = []
    shard_count = math.ceil(args.games / args.games_per_shard)
    for shard_index in range(shard_count):
        start = shard_index * args.games_per_shard
        stop = min(args.games, start + args.games_per_shard)
        jobs = []
        for game_index in range(start, stop):
            our_i = game_index % len(decks)
            round_i = game_index // len(decks)
            opp_i = (our_i + 1 + round_i % (len(decks) - 1)) % len(decks)
            our_arch, our_deck = decks[our_i]
            opp_arch, opp_deck = decks[opp_i]
            jobs.append(
                {
                    "job_index": game_index,
                    "checkpoint": str(checkpoint_path),
                    "opponent_checkpoint": str(checkpoint_path),
                    "checkpoint_digest": checkpoint_digest,
                    "our_deck": list(our_deck),
                    "opp_deck": list(opp_deck),
                    "our_seat": game_index % 2,
                    "opponent_id": opp_arch,
                    "archetype": our_arch,
                    "opp_archetype": opp_arch,
                    "seed": int(args.seed + game_index),
                    "device": "cpu",
                    "agent_mode": "policy",
                    "sample_actions": True,
                    "action_temperature": float(args.temperature),
                    "training_eligible": True,
                    "collect_both_seats": True,
                    "collect_privileged_belief": True,
                    "game_timeout_s": int(args.game_timeout_s),
                    "target_provenance": {
                        "privileged_belief": True,
                        "lineage_digest": lineage_digest,
                    },
                }
            )
        tasks.append(
            {
                "shard_index": shard_index,
                "output": str(output_dir / f"shard_{shard_index:05d}.jsonl"),
                "jobs": jobs,
                "multi_env": int(args.multi_env),
                "engine_lib": str(engine_lib),
                "engine_digest": engine_digest,
                "engine_source_digest": engine_source_digest,
                "hidden_export_digest": hidden_export_digest,
                "checkpoint_digest": checkpoint_digest,
                "lineage_digest": lineage_digest,
            }
        )

    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {executor.submit(_collect_shard, task): task for task in tasks}
        for future in as_completed(futures):
            meta = future.result()
            completed.append(meta)
            print(
                f"[privileged] shard={meta['shard_index']} "
                f"games={meta['games']} decisions={meta['decisions']} "
                f"seconds={meta['elapsed_seconds']:.1f}",
                flush=True,
            )

    completed.sort(key=lambda row: int(row["shard_index"]))
    manifest = {
        **lineage,
        "lineage_digest": lineage_digest,
        "output_dir": str(output_dir),
        "requested_games": int(args.games),
        "workers": int(args.workers),
        "multi_env": int(args.multi_env),
        "totals": {
            "games": sum(int(row["games"]) for row in completed),
            "records": sum(int(row["records"]) for row in completed),
            "decisions": sum(int(row["decisions"]) for row in completed),
            "hand_labeled_decisions": sum(
                int(row["hand_labeled_decisions"]) for row in completed
            ),
            "bytes": sum(int(row["bytes"]) for row in completed),
        },
        "shards": completed,
    }
    if manifest["totals"]["games"] != int(args.games):
        raise RuntimeError("completed game count does not match request")
    _atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["totals"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
