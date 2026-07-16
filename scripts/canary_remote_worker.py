#!/usr/bin/env python3
"""Canary client for a remote TrueNAS (or LAN) poke-bot worker.

Does **not** restart or attach to overnight local trainers. Use this to verify
TCP reachability + health, optionally fail-closed on simulator_version /
checkpoint digest mismatches before a boundary cutover.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint  # noqa: E402


def _local_simulator_version() -> Optional[str]:
    try:
        from poke_bot.belief import simulator_version

        return simulator_version()
    except Exception as exc:  # noqa: BLE001
        print(f"[canary] WARN local simulator_version unavailable: {exc}", flush=True)
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "endpoint",
        nargs="?",
        default="192.168.1.143:8765",
        help="host:port of remote worker (default: 192.168.1.143:8765)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="connect / idle timeout seconds",
    )
    p.add_argument(
        "--require-match-local",
        action="store_true",
        help="Fail closed if remote simulator_version != local (version storm)",
    )
    p.add_argument(
        "--expect-simulator-version",
        default="",
        help="Fail closed if remote simulator_version != this exact string",
    )
    p.add_argument(
        "--submit-ping-only",
        action="store_true",
        default=True,
        help="Only hello+ping+health (default)",
    )
    p.add_argument(
        "--no-submit-ping-only",
        action="store_false",
        dest="submit_ping_only",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    host, port = parse_endpoint(args.endpoint)
    print(f"[canary] connecting to {host}:{port} …", flush=True)
    t0 = time.perf_counter()
    try:
        with RemoteJobClient(host, port, timeout_s=args.timeout) as client:
            info = client.info
            assert info is not None
            health = client.health()
            remote_sim = getattr(info, "simulator_version", None) or health.get(
                "simulator_version"
            )
            print(
                f"[canary] hello_ok hostname={info.hostname} "
                f"gpu={info.gpu_name!r} device={info.device} "
                f"workers={info.workers} leaf_servers={info.leaf_servers} "
                f"digest={(info.checkpoint_digest or '')[:12]} "
                f"simulator_version={(remote_sim or '')[:48]} "
                f"free_ram_gb={info.free_ram_gb}",
                flush=True,
            )
            pong = client.ping()
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[canary] ping ok rtt_ms≈{rtt_ms:.1f} pong={pong}", flush=True)
            print(
                f"[canary] health ok={health.get('ok')} "
                f"leaf_alive={health.get('leaf_alive')}",
                flush=True,
            )
            payload: dict[str, Any] = {
                "info": info.__dict__,
                "health": health,
                "remote_simulator_version": remote_sim,
            }
            expect = (args.expect_simulator_version or "").strip()
            if args.require_match_local:
                local = _local_simulator_version()
                payload["local_simulator_version"] = local
                if not local:
                    print(
                        "[canary] FAILED: --require-match-local but local "
                        "simulator_version unavailable",
                        file=sys.stderr,
                    )
                    print(json.dumps(payload, indent=2))
                    return 3
                if not remote_sim:
                    print(
                        "[canary] FAILED: remote did not report simulator_version "
                        "(old image?). Rebuild/redeploy Elmo/bert worker first.",
                        file=sys.stderr,
                    )
                    print(json.dumps(payload, indent=2))
                    return 4
                if remote_sim != local:
                    print(
                        "[canary] FAILED: simulator_version mismatch "
                        f"(version storm risk)\n  local ={local}\n  remote={remote_sim}",
                        file=sys.stderr,
                    )
                    print(json.dumps(payload, indent=2))
                    return 5
                print("[canary] simulator_version MATCH local", flush=True)
            if expect:
                if remote_sim != expect:
                    print(
                        "[canary] FAILED: expected simulator_version "
                        f"{expect!r}, got {remote_sim!r}",
                        file=sys.stderr,
                    )
                    print(json.dumps(payload, indent=2))
                    return 6
            print(json.dumps(payload, indent=2))
            if not health.get("ok"):
                return 2
    except Exception as exc:
        print(f"[canary] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nAuth / deploy blockers are expected until TrueNAS accepts SSH or\n"
            "the worker container is started. Stage assets under SMB:\n"
            "  //truenas.local/main/poke-bot-agent/\n"
            "then boundary-cutover compose and re-run this canary.\n"
            "See docs/REMOTE_WORKER_CUTOVER.md",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
