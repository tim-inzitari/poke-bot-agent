#!/usr/bin/env python3
"""Canary client for a remote TrueNAS (or LAN) poke-bot worker.

Does **not** restart or attach to overnight local trainers. Use this to verify
TCP reachability + health, optionally submit one cheap policy smoke job once
SSH/Docker on TrueNAS is up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "endpoint",
        nargs="?",
        default="truenas.local:8765",
        help="host:port of remote worker (default: truenas.local:8765)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="connect / idle timeout seconds",
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
            print(
                f"[canary] hello_ok hostname={info.hostname} "
                f"gpu={info.gpu_name!r} device={info.device} "
                f"workers={info.workers} leaf_servers={info.leaf_servers} "
                f"digest={(info.checkpoint_digest or '')[:12]} "
                f"free_ram_gb={info.free_ram_gb}",
                flush=True,
            )
            pong = client.ping()
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[canary] ping ok rtt_ms≈{rtt_ms:.1f} pong={pong}", flush=True)
            health = client.health()
            print(
                f"[canary] health ok={health.get('ok')} leaf_alive={health.get('leaf_alive')}",
                flush=True,
            )
            print(json.dumps({"info": info.__dict__, "health": health}, indent=2))
            if not health.get("ok"):
                return 2
    except Exception as exc:
        print(f"[canary] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nAuth / deploy blockers are expected until TrueNAS accepts SSH or\n"
            "the Custom App is started. Stage assets under SMB:\n"
            "  //truenas.local/main/poke-bot-agent/\n"
            "then start the compose stack and re-run this canary.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
