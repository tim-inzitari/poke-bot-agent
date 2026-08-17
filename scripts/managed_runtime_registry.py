#!/usr/bin/env python3
"""Resolve the exact runtime registry selected by a managed systemd service."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess


_REGISTRY_ARGUMENT = re.compile(r"(?:^|\s)--registry(?:=|\s+)([^\s;]+)")


def registry_from_managed_service(service: str) -> Path:
    """Return the one registry named by the service's effective ExecStart."""

    service = str(service or "").strip()
    if (
        not service.startswith("pokebot-")
        or not service.endswith(".service")
        or any(char.isspace() for char in service)
    ):
        raise RuntimeError("unsafe managed service identity")
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "--property=ExecStart",
            "--value",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "could not resolve managed runtime registry: "
            + completed.stdout.strip()
            + completed.stderr.strip()
        )
    matches = _REGISTRY_ARGUMENT.findall(completed.stdout)
    if len(matches) != 1:
        raise RuntimeError("managed service does not select exactly one registry")
    registry = Path(matches[0]).expanduser().resolve()
    if not registry.is_file():
        raise RuntimeError(f"managed runtime registry is missing: {registry}")
    return registry

