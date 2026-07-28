#!/usr/bin/env python3
"""Activate the checksum-identical V6 loader fleet at a stopped boundary."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_bot.dormant_adapter_compat import (
    LOADER_RUNTIME_FILES,
    loader_source_contract,
)
from poke_bot.matchup_adapters_v6 import load_slot_registry, registry_digest
from poke_bot.pure_rl.model_registry import sha256
from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint


RECEIPT_SCHEMA = "poke_bot.matchup_adapter_v6_fleet_activation/v1"


def _run(
    argv: list[str],
    *,
    timeout: float = 180.0,
    cwd: Path | None = None,
) -> str:
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        cwd=cwd,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed.stdout


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _service_is_active(service: str) -> bool:
    return (
        subprocess.run(
            ["/usr/bin/systemctl", "--user", "is-active", "--quiet", service],
            check=False,
        ).returncode
        == 0
    )


def _remote_digests(host: str, root: Path) -> dict[str, str]:
    files = [str(root / relative) for relative in LOADER_RUNTIME_FILES]
    output = _run(
        ["ssh", "-o", "BatchMode=yes", host, "sha256sum", *files],
        timeout=30,
    )
    by_path: dict[str, str] = {}
    for row in output.splitlines():
        digest, _, raw_path = row.partition("  ")
        if len(digest) == 64 and raw_path:
            by_path[raw_path] = "sha256:" + digest
    return {
        relative: by_path.get(str(root / relative), "")
        for relative in LOADER_RUNTIME_FILES
    }


def _wait_endpoint(
    endpoint: str,
    expected_workers: int,
    expected_leaves: int,
) -> dict[str, Any]:
    host, port = parse_endpoint(endpoint)
    deadline = time.monotonic() + 180.0
    last_error = ""
    while time.monotonic() < deadline:
        client = RemoteJobClient(
            host,
            port,
            timeout_s=5,
            connect_timeout_s=5,
            control_timeout_s=10,
        )
        try:
            info = client.connect()
            health = client.health()
            leaves = int(health.get("leaf_servers") or 0)
            healthy = (
                health.get("ok") is True
                and health.get("controller_healthy") is True
                and health.get("leaf_alive") is True
                and health.get("leaf_identity_ok") is True
            )
            if (
                int(info.workers) == int(expected_workers)
                and leaves == int(expected_leaves)
                and healthy
            ):
                return {
                    "endpoint": endpoint,
                    "workers": int(info.workers),
                    "leaf_servers": leaves,
                    "job_kinds": list(info.job_kinds),
                    "health": health,
                }
            last_error = (
                f"workers={info.workers}/{expected_workers} "
                f"leaves={leaves}/{expected_leaves} healthy={healthy}"
            )
        except Exception as exc:  # boundary polling reports the final cause
            last_error = f"{type(exc).__name__}: {exc}"
        finally:
            client.close()
        time.sleep(2)
    raise RuntimeError(f"V6 fleet endpoint did not become ready: {endpoint}: {last_error}")


def _expected_contract(source_root: Path) -> dict[str, str]:
    return loader_source_contract(source_root)


def validate_fleet(
    *,
    config: dict[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    receipt = _read(receipt_path)
    source_root = Path(str(config["source_root"])).expanduser().resolve()
    expected = _expected_contract(source_root)
    bert = dict(config["bert"])
    elmo = dict(config["elmo"])
    bert_observed = _remote_digests(
        str(bert["host"]),
        Path(str(bert["runtime_root"])),
    )
    elmo_output = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            str(elmo["host"]),
            "sudo",
            "-n",
            "docker",
            "exec",
            str(elmo["container"]),
            "sha256sum",
            *[
                "/workspace/" + relative
                for relative in LOADER_RUNTIME_FILES
            ],
        ],
        timeout=30,
    )
    elmo_observed: dict[str, str] = {}
    for row in elmo_output.splitlines():
        digest, _, raw_path = row.partition("  ")
        if raw_path.startswith("/workspace/"):
            elmo_observed[raw_path.removeprefix("/workspace/")] = (
                "sha256:" + digest
            )
    elmo_image = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            str(elmo["host"]),
            "sudo",
            "-n",
            "docker",
            "inspect",
            str(elmo["container"]),
            "--format={{.Config.Image}}",
        ],
        timeout=20,
    ).strip()
    bert_health = _wait_endpoint(
        str(bert["endpoint"]),
        int(bert["expected_workers"]),
        int(bert["expected_leaves"]),
    )
    elmo_health = _wait_endpoint(
        str(elmo["endpoint"]),
        int(elmo["expected_workers"]),
        int(elmo["expected_leaves"]),
    )
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "active"
        or receipt.get("loader_source_contract") != expected
        or bert_observed != expected
        or elmo_observed != expected
        or elmo_image != str(elmo["image"])
        or int(dict(receipt.get("bert") or {}).get("workers") or 0)
        != int(bert["expected_workers"])
        or int(dict(receipt.get("bert") or {}).get("leaf_servers") or 0)
        != int(bert["expected_leaves"])
        or int(dict(receipt.get("elmo") or {}).get("workers") or 0)
        != int(elmo["expected_workers"])
        or int(dict(receipt.get("elmo") or {}).get("leaf_servers") or 0)
        != int(elmo["expected_leaves"])
    ):
        raise RuntimeError("V6 fleet activation receipt or loader parity changed")
    return {
        **receipt,
        "current_health": {
            "bert": bert_health,
            "elmo": elmo_health,
        },
    }


def activate_fleet(
    *,
    config: dict[str, Any],
    training_service: str,
    receipt_path: Path,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    if _service_is_active(training_service):
        raise RuntimeError("V6 fleet activation requires a stopped trainer boundary")
    if receipt_path.is_file():
        return validate_fleet(config=config, receipt_path=receipt_path)

    source_root = Path(str(config["source_root"])).expanduser().resolve()
    registry_path = Path(str(config["registry"])).expanduser().resolve()
    expected = _expected_contract(source_root)
    registry = load_slot_registry(registry_path)
    bert = dict(config["bert"])
    elmo = dict(config["elmo"])

    relative_sources = list(LOADER_RUNTIME_FILES)
    _run(
        [
            "rsync",
            "-aR",
            *relative_sources,
            f"{bert['host']}:{bert['runtime_root']}/",
        ],
        timeout=120,
        cwd=source_root,
    )
    uid = _run(
        ["ssh", "-o", "BatchMode=yes", str(bert["host"]), "id", "-u"],
        timeout=20,
    ).strip()
    _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            str(bert["host"]),
            (
                "launchctl kickstart -k "
                f"gui/{uid}/{bert['service_label']} "
                ">/dev/null 2>&1 &"
            ),
        ],
        timeout=20,
    )

    compose_files = [
        str(Path(value).expanduser()) for value in elmo["compose_files"]
    ]
    compose_command = ["sudo", "-n", "docker", "compose"]
    for path in compose_files:
        compose_command.extend(["-f", path])
    compose_command.extend(["up", "-d", "--force-recreate", str(elmo["service"])])
    _run(
        ["ssh", "-o", "BatchMode=yes", str(elmo["host"]), *compose_command],
        timeout=240,
    )

    bert_health = _wait_endpoint(
        str(bert["endpoint"]),
        int(bert["expected_workers"]),
        int(bert["expected_leaves"]),
    )
    elmo_health = _wait_endpoint(
        str(elmo["endpoint"]),
        int(elmo["expected_workers"]),
        int(elmo["expected_leaves"]),
    )
    bert_observed = _remote_digests(
        str(bert["host"]),
        Path(str(bert["runtime_root"])),
    )
    if bert_observed != expected:
        raise RuntimeError("Bert V6 loader deployment lacks checksum parity")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "active",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_service": training_service,
        "training_service_was_stopped": True,
        "source_root": str(source_root),
        "loader_source_contract": expected,
        "registry": str(registry_path),
        "registry_digest": registry_digest(registry),
        "bert": bert_health,
        "elmo": elmo_health,
        "elmo_image": str(elmo["image"]),
        "managed_service_activation_only": True,
    }
    _atomic_json(receipt_path, receipt)
    return validate_fleet(config=config, receipt_path=receipt_path)


__all__ = ["RECEIPT_SCHEMA", "activate_fleet", "validate_fleet"]
