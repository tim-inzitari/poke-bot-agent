#!/usr/bin/env python3
"""Bert mesh canary: sync/reload, pin, belief-MCTS play, parallel slots, remap.

Owns ``bert.local:8766`` for diagnostics. Does **not** kill trainers.
Light load only (few games, ≤3 parallel slots). Optional brief Elmo hello
for multi-host remap path coverage — never soak Elmo.

Usage (training box)::

  /home/inzi/miniconda3/envs/poke-bot-agent/bin/python \\
    scripts/canary_bert_sync.py --endpoint bert.local:8766 --cycles 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poke_bot.checkpoint import checkpoint_digest  # noqa: E402
from poke_bot.deck_pool import hammer_pult_deck  # noqa: E402
from poke_bot.remote_jobs import (  # noqa: E402
    RemoteJobClient,
    parse_endpoint,
    prepare_remote_play_job,
    resolve_remote_checkpoint_path,
    resolve_remote_workdir_path,
)

DEFAULT_A = (
    REPO_ROOT
    / "outputs/checkpoints"
    / "blackwell_hammer_belief_v3_20260715T171515Z.evaluated.d2f4601fb653fab1.pt"
)
DEFAULT_B = (
    REPO_ROOT
    / "outputs/checkpoints"
    / "core_kernel_3080ti_trusted_20260714T2000Z.hammer_search_rl.evaluated.e32ce60b893086fd.pt"
)
DEFAULT_SPEC = REPO_ROOT / "baselines" / "official" / "iono"
ELMO_ENDPOINT = "192.168.1.143:8765"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="bert.local:8766")
    p.add_argument("--checkpoint-a", type=Path, default=DEFAULT_A)
    p.add_argument("--checkpoint-b", type=Path, default=DEFAULT_B)
    p.add_argument("--spec-path", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--move-time", type=float, default=12.0)
    p.add_argument("--games", type=int, default=1, help="play jobs per digest step")
    p.add_argument("--parallel-slots", type=int, default=2)
    p.add_argument("--cycles", type=int, default=3, help="full mesh cycles")
    p.add_argument("--job-timeout-s", type=float, default=900.0)
    p.add_argument("--game-timeout-s", type=int, default=600)
    p.add_argument(
        "--skip-elmo-touch",
        action="store_true",
        help="skip brief Elmo hello/remap (default: one health+path check)",
    )
    p.add_argument(
        "--skip-promotion-job",
        action="store_true",
        help="only dual-pin+health (skip expensive promotion game)",
    )
    p.add_argument(
        "--restore-digest",
        choices=("a", "pre", "none"),
        default="a",
        help="which digest to leave on Bert after the canary",
    )
    p.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="optional path to write full JSON report",
    )
    return p.parse_args(argv)


def _log(msg: str) -> None:
    print(f"[bert-sync] {msg}", flush=True)


def _check(name: str, ok: bool, **detail: Any) -> dict[str, Any]:
    row = {"pass": bool(ok), **detail}
    status = "PASS" if ok else "FAIL"
    _log(f"  {name}: {status}" + (f" {detail}" if detail and not ok else ""))
    return row


def _reload(
    client: RemoteJobClient,
    path: Path,
    *,
    digest: str,
    version: int,
) -> dict[str, Any]:
    remote_path = resolve_remote_checkpoint_path(client.host, str(path))
    _log(f"staged {path.name} → {remote_path}")
    reply = client.reload_checkpoint(str(path), digest=digest, version=version)
    if not reply.get("ok", False):
        raise RuntimeError(f"reload failed: {reply!r}")
    return reply


def _health_digest(client: RemoteJobClient) -> tuple[str, int, dict[str, Any]]:
    health = client.health()
    if not health.get("ok") or not health.get("leaf_alive"):
        raise RuntimeError(f"unhealthy: {health!r}")
    digest = str(health.get("checkpoint_digest") or "")
    version = int(health.get("checkpoint_version", -1))
    return digest, version, health


def _play_job_dict(
    *,
    digest: str,
    checkpoint: Path,
    spec_path: Path,
    sims: int,
    move_time: float,
    game_timeout_s: int,
    seed: int,
    job_index: int = 0,
) -> dict[str, Any]:
    # Match trainer belief-MCTS shape. training_eligible gates collect_targets
    # in _worker_play — must stay True or leaf traffic is invisible / search
    # targets are not recorded (and canary cannot prove leaf_remote sims).
    return {
        "job_index": int(job_index),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "model_generation": 1,
        "our_deck": hammer_pult_deck(),
        "spec": {
            "id": "iono",
            "name": "iono",
            "dir_name": "iono",
            "group": "roster",
            "source": "canary",
            "path": str(spec_path),
        },
        "our_seat": int(job_index) % 2,
        "pair_id": None,
        "mcts_sims": int(sims),
        "mcts_move_time": float(move_time),
        "game_timeout_s": int(game_timeout_s),
        "expected_search_decisions": 16,
        "agent_mode": "belief-mcts",
        "seed": int(seed),
        "device": "cpu",
        "archetype": "hammer-pult",
        "training_eligible": True,
        "target_provenance": {
            "target_source": "belief_mcts",
            "trusted": True,
            "model_generation": 1,
            "incumbent_checkpoint": {"digest": digest, "path": str(checkpoint)},
        },
        "preserve_stdout": False,
    }


def _assert_play_ok(result: dict[str, Any], label: str) -> None:
    err = result.get("error")
    failed = bool(
        result.get("our_failed")
        or result.get("resource_error")
        or result.get("trust_failure")
        or int(result.get("fail_closed") or 0) > 0
        or (err and "version mismatch" in str(err))
        or (err and "digest mismatch" in str(err))
        or (err and "RemoteLeafTimeout" in str(err))
        or (err and "FAIL-CLOSED" in str(err))
        or (err and "stale search target" in str(err))
    )
    leaf_remote = bool(result.get("leaf_remote"))
    leaf_evals = int(result.get("leaf_evaluations") or 0)
    remote_reqs = int(result.get("remote_leaf_requests") or 0)
    _log(
        f"{label} winner={result.get('winner')} steps={result.get('steps')} "
        f"leaf_remote={leaf_remote} leaf_evals={leaf_evals} "
        f"remote_reqs={remote_reqs} fail_closed={result.get('fail_closed')} "
        f"failed={failed} err={err!r}"
    )
    if failed:
        raise RuntimeError(f"{label} play failed: {result!r}")
    if not leaf_remote:
        raise RuntimeError(f"{label} expected leaf_remote=True, got {result!r}")
    if leaf_evals <= 0 and remote_reqs <= 0:
        raise RuntimeError(
            f"{label} leaf_remote set but no leaf traffic: {result!r}"
        )
    if int(result.get("n_decisions") or 0) <= 0:
        raise RuntimeError(f"{label} expected n_decisions>0: {result!r}")
    if int(result.get("mcts_sims_run_total") or 0) < 128:
        raise RuntimeError(
            f"{label} expected mcts_sims_run_total>=128: {result!r}"
        )

def _play_jobs(
    client: RemoteJobClient,
    *,
    digest: str,
    checkpoint: Path,
    spec_path: Path,
    sims: int,
    move_time: float,
    game_timeout_s: int,
    games: int,
    seed0: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(games):
        job = _play_job_dict(
            digest=digest,
            checkpoint=checkpoint,
            spec_path=spec_path,
            sims=sims,
            move_time=move_time,
            game_timeout_s=game_timeout_s,
            seed=seed0 + i,
            job_index=i,
        )
        t0 = time.perf_counter()
        result = client.submit_job(job, kind="play")
        dt = time.perf_counter() - t0
        _log(f"play job#{i} wall={dt:.1f}s")
        _assert_play_ok(result, f"play#{i}")
        rows.append(result)
    return rows


def _baselines_remap_check(host: str, spec_path: Path, checkpoint: Path) -> dict[str, Any]:
    local_spec = str(spec_path.resolve())
    remapped = resolve_remote_workdir_path(host, local_spec)
    job = prepare_remote_play_job(
        host,
        {
            "checkpoint": str(checkpoint),
            "spec": {"path": local_spec, "id": "iono"},
        },
    )
    ok = (
        remapped.startswith("/Users/tsinzitari/workspace/poke-bot-agent/")
        and str(job["spec"]["path"]).startswith(
            "/Users/tsinzitari/workspace/poke-bot-agent/"
        )
        and "/home/inzi/" not in str(job["spec"]["path"])
        and "/home/inzi/" not in str(job["checkpoint"])
    )
    return _check(
        "baselines_path_remap",
        ok,
        local=local_spec,
        remapped=remapped,
        job_spec=job["spec"]["path"],
        job_ckpt=job["checkpoint"],
    )


def _elmo_brief_touch() -> dict[str, Any]:
    """One hello+health+path remap on Elmo — no jobs (do not starve Core)."""
    host, port = parse_endpoint(ELMO_ENDPOINT)
    t0 = time.perf_counter()
    try:
        with RemoteJobClient(host, port, timeout_s=10.0) as client:
            health = client.health()
            local = str(DEFAULT_SPEC.resolve())
            remapped = resolve_remote_workdir_path(host, local)
            ok = bool(health.get("ok")) and remapped.startswith("/workspace/")
            return _check(
                "elmo_brief_touch",
                ok,
                rtt_s=round(time.perf_counter() - t0, 3),
                leaf_alive=health.get("leaf_alive"),
                remapped=remapped,
                jobs_completed=health.get("jobs_completed"),
            )
    except Exception as exc:  # noqa: BLE001
        return _check(
            "elmo_brief_touch",
            False,
            error=f"{type(exc).__name__}: {exc}",
            note="non-blocking for Bert GO if Bert-only mesh PASSes",
        )


def _parallel_play(
    host: str,
    port: int,
    *,
    digest: str,
    checkpoint: Path,
    spec_path: Path,
    sims: int,
    move_time: float,
    game_timeout_s: int,
    n_slots: int,
    seed0: int,
    timeout_s: float,
) -> dict[str, Any]:
    n_slots = max(1, min(3, int(n_slots)))

    def _one(slot: int) -> dict[str, Any]:
        job = _play_job_dict(
            digest=digest,
            checkpoint=checkpoint,
            spec_path=spec_path,
            sims=sims,
            move_time=move_time,
            game_timeout_s=game_timeout_s,
            seed=seed0 + slot,
            job_index=slot,
        )
        with RemoteJobClient(host, port, timeout_s=timeout_s) as client:
            t0 = time.perf_counter()
            result = client.submit_job(job, kind="play")
            result["_wall_s"] = time.perf_counter() - t0
            result["_slot"] = slot
            return result

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_slots) as ex:
        futs = [ex.submit(_one, i) for i in range(n_slots)]
        results = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    for r in results:
        _assert_play_ok(r, f"parallel_slot{r.get('_slot')}")
    return _check(
        "parallel_slots",
        True,
        slots=n_slots,
        wall_s=round(wall, 1),
        per_slot_s=[round(float(r.get("_wall_s", 0)), 1) for r in results],
        leaf_remote_all=all(bool(r.get("leaf_remote")) for r in results),
    )


def _promotion_dual_pin(
    client: RemoteJobClient,
    *,
    path_a: Path,
    path_b: Path,
    digest_a: str,
    digest_b: str,
    sims: int,
    move_time: float,
    skip_job: bool,
) -> dict[str, Any]:
    # Ensure A primary, pin B (promotion-style dual resident).
    _reload(client, path_a, digest=digest_a, version=900)
    pin_reply = client.pin_checkpoint(str(path_b), digest=digest_b)
    got, ver, health = _health_digest(client)
    pinned = {str(x) for x in (health.get("pinned_digests") or [])}
    both = digest_a in pinned and digest_b in pinned and got == digest_a
    checks: dict[str, Any] = {
        "dual_pin_health": _check(
            "dual_pin_health",
            both,
            primary=got[:23],
            version=ver,
            pinned=[p[:23] for p in sorted(pinned)],
            pin_ok=pin_reply.get("ok"),
        )
    }
    if not both:
        raise RuntimeError(f"dual pin failed: {checks}")

    # Play on primary while secondary remains pinned.
    _play_jobs(
        client,
        digest=digest_a,
        checkpoint=path_a,
        spec_path=DEFAULT_SPEC,
        sims=sims,
        move_time=move_time,
        game_timeout_s=600,
        games=1,
        seed0=7000,
    )
    checks["play_while_dual_pinned"] = _check("play_while_dual_pinned", True)

    if skip_job:
        checks["promotion_job"] = _check(
            "promotion_job",
            True,
            skipped=True,
            note="dual-pin+play covered; promotion game skipped (--skip-promotion-job)",
        )
        return checks

    promo = {
        "job_index": 0,
        "candidate_checkpoint": str(path_b),
        "parent_checkpoint": str(path_a),
        "candidate_digest": digest_b,
        "parent_digest": digest_a,
        "deck": hammer_pult_deck(),
        "candidate_seat": 0,
        "mcts_sims": int(sims),
        "mcts_move_time": float(move_time),
        "timeout_s": 600,
        "expected_search_decisions": 16,
        "agent_mode": "belief-mcts",
        "seed": 8001,
        "device": "cpu",
        "model_generation": 1,
    }
    t0 = time.perf_counter()
    result = client.submit_job(promo, kind="promotion")
    dt = time.perf_counter() - t0
    ok = bool(result.get("valid")) and not result.get("error")
    leafish = bool(
        result.get("leaf_remote")
        or result.get("candidate_search_decisions")
        or result.get("parent_search_decisions")
    )
    checks["promotion_job"] = _check(
        "promotion_job",
        ok and leafish,
        wall_s=round(dt, 1),
        valid=result.get("valid"),
        winner=result.get("winner"),
        steps=result.get("steps"),
        error=result.get("error"),
        leaf_remote=result.get("leaf_remote"),
        cand_sd=result.get("candidate_search_decisions"),
        parent_sd=result.get("parent_search_decisions"),
    )
    if not (ok and leafish):
        raise RuntimeError(f"promotion job failed: {result!r}")
    return checks


def run_cycle(
    client: RemoteJobClient,
    *,
    host: str,
    port: int,
    path_a: Path,
    path_b: Path,
    digest_a: str,
    digest_b: str,
    spec_path: Path,
    sims: int,
    move_time: float,
    game_timeout_s: int,
    games: int,
    parallel_slots: int,
    job_timeout_s: float,
    skip_promotion_job: bool,
    cycle: int,
) -> dict[str, Any]:
    matrix: dict[str, Any] = {"cycle": cycle, "checks": {}}

    matrix["checks"]["baselines_path_remap"] = _baselines_remap_check(
        host, spec_path, path_a
    )
    if not matrix["checks"]["baselines_path_remap"]["pass"]:
        raise RuntimeError("baselines remap failed")

    # A: sync → reload → health → belief-mcts play
    reply_a = _reload(client, path_a, digest=digest_a, version=10 + cycle * 10)
    got_a, ver_a, health_a = _health_digest(client)
    ok_a = got_a == digest_a
    matrix["checks"]["sync_a_health"] = _check(
        "sync_a_health",
        ok_a,
        expected=digest_a[:23],
        got=got_a[:23],
        version=ver_a,
        reload_version=reply_a.get("version"),
        pinned=health_a.get("pinned_digests"),
    )
    if not ok_a:
        raise RuntimeError(f"health digest A mismatch: {got_a} != {digest_a}")
    _play_jobs(
        client,
        digest=digest_a,
        checkpoint=path_a,
        spec_path=spec_path,
        sims=sims,
        move_time=move_time,
        game_timeout_s=game_timeout_s,
        games=games,
        seed0=1000 + cycle * 100,
    )
    matrix["checks"]["play_a_belief_mcts"] = _check(
        "play_a_belief_mcts", True, games=games, sims=sims, move_time=move_time
    )

    # B: sync → reload → health → play (must not stay on stale A)
    reply_b = _reload(client, path_b, digest=digest_b, version=11 + cycle * 10)
    got_b, ver_b, health_b = _health_digest(client)
    ok_b = got_b == digest_b and got_b != digest_a
    matrix["checks"]["sync_b_health"] = _check(
        "sync_b_health",
        ok_b,
        expected=digest_b[:23],
        got=got_b[:23],
        version=ver_b,
        reload_version=reply_b.get("version"),
        pinned=health_b.get("pinned_digests"),
    )
    if not ok_b:
        raise RuntimeError(
            f"health digest B mismatch/stale: got={got_b} want={digest_b}"
        )
    _play_jobs(
        client,
        digest=digest_b,
        checkpoint=path_b,
        spec_path=spec_path,
        sims=sims,
        move_time=move_time,
        game_timeout_s=game_timeout_s,
        games=games,
        seed0=2000 + cycle * 100,
    )
    matrix["checks"]["play_b_belief_mcts"] = _check(
        "play_b_belief_mcts", True, games=games, sims=sims
    )

    # Explicit multi-digest pin: primary B, pin A, both resident
    pin_a = client.pin_checkpoint(str(path_a), digest=digest_a)
    _, _, health_pin = _health_digest(client)
    pinned = {str(x) for x in (health_pin.get("pinned_digests") or [])}
    multi_ok = digest_a in pinned and digest_b in pinned
    matrix["checks"]["multi_digest_pin"] = _check(
        "multi_digest_pin",
        multi_ok and bool(pin_a.get("ok")),
        pinned=[p[:23] for p in sorted(pinned)],
    )
    if not multi_ok:
        raise RuntimeError(f"multi-digest pin failed: {pinned}")

    # Version lockstep: N then N+1
    v_n = int(ver_b)
    reply_n = _reload(client, path_b, digest=digest_b, version=v_n)
    # Re-pin A after reload (worker should restore, but assert)
    try:
        client.pin_checkpoint(str(path_a), digest=digest_a)
    except Exception as exc:  # noqa: BLE001
        _log(f"re-pin after reload warn: {exc}")
    reply_n1 = _reload(client, path_b, digest=digest_b, version=v_n + 1)
    got, ver_final, health_ver = _health_digest(client)
    ok_ver = (
        got == digest_b
        and int(reply_n.get("version", -1)) == v_n
        and int(reply_n1.get("version", -1)) == v_n + 1
        and ver_final == v_n + 1
    )
    matrix["checks"]["version_lockstep"] = _check(
        "version_lockstep",
        ok_ver,
        v_n=v_n,
        reload_n=reply_n.get("version"),
        reload_n1=reply_n1.get("version"),
        health_version=ver_final,
        restored_pins=health_ver.get("pinned_digests"),
    )
    if not ok_ver:
        raise RuntimeError(
            f"version lockstep failed: {matrix['checks']['version_lockstep']}"
        )

    _play_jobs(
        client,
        digest=digest_b,
        checkpoint=path_b,
        spec_path=spec_path,
        sims=sims,
        move_time=move_time,
        game_timeout_s=game_timeout_s,
        games=1,
        seed0=3000 + cycle * 100,
    )
    matrix["checks"]["play_after_version_bump"] = _check(
        "play_after_version_bump", True
    )

    # Parallel few slots (separate TCP clients)
    matrix["checks"]["parallel_slots"] = _parallel_play(
        host,
        port,
        digest=digest_b,
        checkpoint=path_b,
        spec_path=spec_path,
        sims=sims,
        move_time=move_time,
        game_timeout_s=game_timeout_s,
        n_slots=parallel_slots,
        seed0=4000 + cycle * 100,
        timeout_s=job_timeout_s,
    )

    # Promotion-style dual pin (+ optional one promo game)
    promo_checks = _promotion_dual_pin(
        client,
        path_a=path_a,
        path_b=path_b,
        digest_a=digest_a,
        digest_b=digest_b,
        sims=sims,
        move_time=move_time,
        skip_job=skip_promotion_job,
    )
    matrix["checks"].update(promo_checks)

    matrix["pass"] = all(c.get("pass") for c in matrix["checks"].values())
    return matrix


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    path_a = args.checkpoint_a.expanduser().resolve()
    path_b = args.checkpoint_b.expanduser().resolve()
    for label, path in (("A", path_a), ("B", path_b)):
        if not path.is_file():
            print(f"[bert-sync] missing checkpoint {label}: {path}", file=sys.stderr)
            return 2
    if int(args.sims) < 128:
        print("[bert-sync] belief-mcts requires --sims >= 128", file=sys.stderr)
        return 2
    if float(args.move_time) <= 0:
        print(
            "[bert-sync] --move-time must be > 0 (0 caused 0.2s RemoteLeafTimeout)",
            file=sys.stderr,
        )
        return 2
    digest_a = checkpoint_digest(path_a)
    digest_b = checkpoint_digest(path_b)
    if digest_a == digest_b:
        print("[bert-sync] A and B digests must differ", file=sys.stderr)
        return 2
    host, port = parse_endpoint(args.endpoint)
    _log(f"endpoint={host}:{port}")
    _log(f"digest_a={digest_a[:23]}… ({path_a.name})")
    _log(f"digest_b={digest_b[:23]}… ({path_b.name})")
    _log(
        f"sims={args.sims} move_time={args.move_time} "
        f"parallel={args.parallel_slots} cycles={args.cycles}"
    )

    report: dict[str, Any] = {
        "endpoint": f"{host}:{port}",
        "digest_a": digest_a,
        "digest_b": digest_b,
        "sims": int(args.sims),
        "move_time": float(args.move_time),
        "cycles": [],
        "pass": False,
        "root_causes": [],
    }
    pre_digest = None
    try:
        if not args.skip_elmo_touch:
            report["elmo_brief_touch"] = _elmo_brief_touch()

        with RemoteJobClient(host, port, timeout_s=args.job_timeout_s) as client:
            assert client.info is not None
            pre_digest, pre_ver, pre_health = _health_digest(client)
            report["pre"] = {
                "digest": pre_digest,
                "version": pre_ver,
                "jobs_completed": pre_health.get("jobs_completed"),
                "pinned": pre_health.get("pinned_digests"),
                "device": pre_health.get("device"),
                "workers": pre_health.get("workers"),
            }
            _log(
                f"pre health digest={pre_digest[:23]}… version={pre_ver} "
                f"jobs_completed={pre_health.get('jobs_completed')} "
                f"pinned={len(pre_health.get('pinned_digests') or [])}"
            )
            for cycle in range(1, int(args.cycles) + 1):
                _log(f"===== cycle {cycle}/{args.cycles} =====")
                row = run_cycle(
                    client,
                    host=host,
                    port=port,
                    path_a=path_a,
                    path_b=path_b,
                    digest_a=digest_a,
                    digest_b=digest_b,
                    spec_path=args.spec_path,
                    sims=int(args.sims),
                    move_time=float(args.move_time),
                    game_timeout_s=int(args.game_timeout_s),
                    games=int(args.games),
                    parallel_slots=int(args.parallel_slots),
                    job_timeout_s=float(args.job_timeout_s),
                    skip_promotion_job=bool(args.skip_promotion_job),
                    cycle=cycle,
                )
                report["cycles"].append(row)
                _log(f"cycle {cycle} PASS={row['pass']}")

            if args.restore_digest == "a":
                _reload(client, path_a, digest=digest_a, version=pre_ver + 100)
                _log("restored primary to digest A")
            elif args.restore_digest == "pre" and pre_digest == digest_a:
                _reload(client, path_a, digest=digest_a, version=pre_ver + 100)
                _log("restored primary to pre (A)")
            elif args.restore_digest == "pre" and pre_digest == digest_b:
                _reload(client, path_b, digest=digest_b, version=pre_ver + 100)
                _log("restored primary to pre (B)")
            final_digest, final_ver, final_health = _health_digest(client)
            report["final"] = {
                "digest": final_digest,
                "version": final_ver,
                "ok": final_health.get("ok"),
                "leaf_alive": final_health.get("leaf_alive"),
                "jobs_completed": final_health.get("jobs_completed"),
                "jobs_failed": final_health.get("jobs_failed"),
                "pinned": final_health.get("pinned_digests"),
            }
            report["pass"] = all(c.get("pass") for c in report["cycles"]) and bool(
                final_health.get("ok")
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["root_causes"].append(report["error"])
        _log(f"FAILED: {report['error']}")
        print(json.dumps(report, indent=2, default=str))
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2, default=str))
        return 1

    print(json.dumps(report, indent=2, default=str))
    print("\n=== PASS/FAIL MATRIX ===", flush=True)
    for cycle in report["cycles"]:
        for name, check in cycle["checks"].items():
            status = "PASS" if check.get("pass") else "FAIL"
            print(f"cycle{cycle['cycle']}.{name}: {status}", flush=True)
    if report.get("elmo_brief_touch"):
        et = report["elmo_brief_touch"]
        print(
            f"elmo_brief_touch: {'PASS' if et.get('pass') else 'FAIL'}",
            flush=True,
        )
    print(f"OVERALL: {'PASS' if report['pass'] else 'FAIL'}", flush=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
