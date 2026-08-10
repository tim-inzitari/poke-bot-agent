#!/usr/bin/env python3
"""Execute one Replay Model Inspector trace under an exact runtime root.

The parent inspector supplies a checksum-verified runtime as the first
``PYTHONPATH`` entry and one immutable replay address on stdin.  This worker
does not bind a socket, write artifacts, or control any managed service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from replay_inspector.config import InspectorConfig
from replay_inspector.server import InspectorApplication


def _path(value: Any) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def main() -> int:
    request = json.load(sys.stdin)
    raw = request.get("config")
    if not isinstance(raw, dict):
        raise TypeError("config is required")
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
    )
    application = InspectorApplication(config)
    payload = application.trace_payload(
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
