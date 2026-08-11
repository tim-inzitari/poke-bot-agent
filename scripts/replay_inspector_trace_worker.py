#!/usr/bin/env python3
"""Execute Replay Model Inspector reconstruction under an exact runtime root.

The parent inspector supplies a checksum-verified runtime as the first
``PYTHONPATH`` entry and an immutable replay address on stdin.  Legacy
``trace`` mode executes exactly one raw trace.  Game mode streams bounded
NDJSON raw traces back to its owning parent, which alone assembles the final
HTTP rows and decides whether a temporary cache can retain them.  This worker
does not bind a socket or control any managed service.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from replay_inspector.config import InspectorConfig
from replay_inspector.server import (
    GAME_MATERIALIZATION_WORKER_PROTOCOL,
    InspectorApplication,
)

_GAME_HEARTBEAT_SECONDS = 10.0


def _path(value: Any) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def _write_game_record(
    record: dict[str, Any], *, output_lock: threading.Lock
) -> None:
    """Write one flushed protocol frame without interleaving heartbeats."""

    encoded = json.dumps(record, allow_nan=False, separators=(",", ":"))
    with output_lock:
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()


def _game_addresses(request: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    raw_addresses = request.get("addresses")
    if not isinstance(raw_addresses, list):
        raise TypeError("game worker addresses are required")
    try:
        addresses = tuple(
            (int(item[0]), int(item[1]))
            for item in raw_addresses
            if isinstance(item, list) and len(item) == 2
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("game worker addresses are malformed") from exc
    if len(addresses) != len(raw_addresses) or len(set(addresses)) != len(addresses):
        raise ValueError("game worker addresses are malformed")
    return addresses


def main() -> int:
    request = json.load(sys.stdin)
    raw = request.get("config")
    if not isinstance(raw, dict):
        raise TypeError("config is required")
    defaults = InspectorConfig()
    config = InspectorConfig(
        bind_host=str(raw["bind_host"]),
        port=int(raw["port"]),
        replay_root=Path(raw["replay_root"]),
        rollout_root=Path(raw["rollout_root"]),
        provenance_manifest=_path(raw.get("provenance_manifest")),
        training_recipe_registry=_path(raw.get("training_recipe_registry")),
        artifact_roots=tuple(Path(value) for value in raw.get("artifact_roots", [])),
        runtime_source_root=Path(raw["runtime_source_root"]),
        web_root=Path(raw["web_root"]),
        torch_threads=int(raw["torch_threads"]),
        max_parameter_slice=int(raw["max_parameter_slice"]),
        max_tensor_values=int(raw["max_tensor_values"]),
        verify_digests=bool(raw["verify_digests"]),
        game_trace_cache_root=(
            _path(raw.get("game_trace_cache_root"))
            or defaults.game_trace_cache_root
        ),
        game_trace_cache_enabled=bool(raw.get("game_trace_cache_enabled", True)),
        game_trace_cache_max_bytes=int(
            raw.get("game_trace_cache_max_bytes", defaults.game_trace_cache_max_bytes)
        ),
        game_trace_cache_max_game_bytes=int(
            raw.get(
                "game_trace_cache_max_game_bytes",
                defaults.game_trace_cache_max_game_bytes,
            )
        ),
        game_trace_cache_max_entry_bytes=int(
            raw.get(
                "game_trace_cache_max_entry_bytes",
                defaults.game_trace_cache_max_entry_bytes,
            )
        ),
        game_trace_cache_min_free_bytes=int(
            raw.get(
                "game_trace_cache_min_free_bytes",
                defaults.game_trace_cache_min_free_bytes,
            )
        ),
    )
    application = InspectorApplication(config)
    mode = request.get("mode", "trace")
    if mode not in {"trace", "game"}:
        raise ValueError("unsupported worker mode")
    submission_id = int(request["submission_id"])
    episode_id = int(request["episode_id"])
    step_index = int(request["step_index"])
    factorized_stage = int(request["factorized_stage"])
    if mode == "game":
        if request.get("head_scales"):
            raise ValueError("game worker accepts baseline traces only")
        output_lock = threading.Lock()
        requested_addresses = _game_addresses(request)
        if not application._imported_runtime_is(config.runtime_source_root):
            _write_game_record(
                {
                    "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                    "kind": "error",
                    "code": "isolated_game_worker_runtime_import_mismatch",
                },
                output_lock=output_lock,
            )
            return 2
        prepared = application._prepare_baseline_game_materialization(
            submission_id,
            episode_id,
            step_index,
            factorized_stage,
            include_setup_model_forward=bool(
                request.get("allow_setup_prompt_model_forward", False)
            ),
        )
        if prepared is None:
            _write_game_record(
                {
                    "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                    "kind": "error",
                    "code": "game_materialization_preparation_unavailable",
                },
                output_lock=output_lock,
            )
            return 2
        _identity_key, _identity, addresses = prepared
        if set(requested_addresses) != set(addresses):
            raise ValueError("game worker address set differs from replay timeline")
        _write_game_record(
            {
                "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                "kind": "start",
                "address_count": len(addresses),
            },
            output_lock=output_lock,
        )
        stopped = threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(_GAME_HEARTBEAT_SECONDS):
                _write_game_record(
                    {
                        "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                        "kind": "heartbeat",
                    },
                    output_lock=output_lock,
                )

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name="replay-inspector-game-worker-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            raw_traces = application._materialize_baseline_game_backend_traces(
                submission_id,
                episode_id,
                addresses,
                include_setup_model_forward=bool(
                    request.get("allow_setup_prompt_model_forward", False)
                ),
            )
        finally:
            stopped.set()
            heartbeat_thread.join(timeout=1)
        if not isinstance(raw_traces, dict):
            _write_game_record(
                {
                    "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                    "kind": "error",
                    "code": "exact_game_trace_reconstruction_failed",
                },
                output_lock=output_lock,
            )
            return 2
        for step, stage in addresses:
            payload = raw_traces.pop((step, stage), None)
            if not isinstance(payload, dict):
                _write_game_record(
                    {
                        "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                        "kind": "error",
                        "code": "exact_game_trace_missing_address",
                    },
                    output_lock=output_lock,
                )
                return 2
            _write_game_record(
                {
                    "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                    "kind": "trace",
                    "step_index": step,
                    "factorized_stage": stage,
                    "payload": payload,
                },
                output_lock=output_lock,
            )
        _write_game_record(
            {
                "protocol": GAME_MATERIALIZATION_WORKER_PROTOCOL,
                "kind": "complete",
                "address_count": len(addresses),
            },
            output_lock=output_lock,
        )
        return 0

    payload = application._trace_payload_uncached(
        int(request["submission_id"]),
        int(request["episode_id"]),
        int(request["step_index"]),
        int(request["factorized_stage"]),
        head_scales=request.get("head_scales") or None,
        include_setup_model_forward=bool(
            request.get("allow_setup_prompt_model_forward", False)
        ),
    )
    result = {
        key: payload.get(key)
        for key in (
            "model",
            "heads",
            "fusion",
            "decision_influence",
            "guide_shadow",
            "legal_options",
            "provenance",
            "reproduction_status",
            "warnings",
        )
    }
    json.dump(result, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
