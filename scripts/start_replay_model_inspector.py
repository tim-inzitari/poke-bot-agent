#!/usr/bin/env python3
"""Launch or validate the standalone read-only Replay Model Inspector.

This wrapper deliberately owns only its own Python process.  It does not call
systemctl, SSH, Docker, training code, or remote checkpoint reload APIs.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Running this file directly sets ``sys.path[0]`` to ``scripts/`` rather than
# the repository root.  Add only this checksum-deployed source tree so the
# documented ``python scripts/start_replay_model_inspector.py`` entry point
# resolves the sibling ``replay_inspector`` package without relying on a
# caller-provided PYTHONPATH.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_config_path() -> Path:
    configured = os.environ.get("REPLAY_MODEL_INSPECTOR_CONFIG")
    return Path(configured) if configured else ROOT / "replay_inspector" / "config.json"


def _is_loopback_host(value: str) -> bool:
    host = str(value).strip()
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_config(config_path: Path, host_override: str | None) -> None:
    """Load the canonical flat config and enforce this tool's network boundary."""

    from replay_inspector.config import InspectorConfig

    config = InspectorConfig.load(config_path)
    effective_host = host_override or config.bind_host
    if not _is_loopback_host(effective_host):
        raise ValueError(
            "Replay Model Inspector is loopback-only; bind_host must be "
            "127.0.0.1, ::1, or localhost"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="JSON config path (default: $REPLAY_MODEL_INSPECTOR_CONFIG or replay_inspector/config.json)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="loopback-only bind override; never accepts a LAN address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port override",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate config and inspector roots without binding a listening socket",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = args.config.expanduser().resolve()
    _validate_config(config_path, args.host)
    if args.port is not None and not 1 <= args.port <= 65535:
        raise ValueError("--port must be in 1..65535")

    try:
        from replay_inspector.server import main as server_main
    except ImportError as exc:
        raise RuntimeError(
            "Replay Model Inspector server is unavailable; install the standalone "
            "replay_inspector package before launching it"
        ) from exc

    result = server_main(
        config_path=config_path,
        host=args.host,
        port=args.port,
        check=bool(args.check),
    )
    return 0 if result is None else int(result)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"replay-model-inspector: {exc}", file=sys.stderr)
        raise SystemExit(2)
