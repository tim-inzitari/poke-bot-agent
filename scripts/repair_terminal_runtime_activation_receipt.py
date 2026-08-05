#!/usr/bin/env python3
"""Bind an already runtime-enabled incumbent to a successor activation receipt.

This is intentionally a narrow repair for a terminal handoff whose checkpoint
already contains the complete serving-enabled Fusion-v3 contract but whose
bootstrap receipt was lost.  It never changes a checkpoint, selector, or
trainer state.  Every identity in the receipt is read and checksum-verified
from the run manifest, incumbent registry, and initial checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from poke_bot import checkpoint


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _fusion_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model = dict(payload.get("model_config") or {})
    fusion = dict((payload.get("provenance") or {}).get("decision_fusion") or {})
    if not (
        model.get("decision_fusion_enabled") is True
        and model.get("decision_fusion_runtime_enabled") is True
        and fusion.get("schema") == "poke_bot.causal_decision_fusion/v3"
        and fusion.get("runtime_enabled") is True
    ):
        raise RuntimeError("checkpoint is not a complete runtime-enabled Fusion-v3 learner")
    heads = list(fusion.get("required_heads") or [])
    routes = dict(fusion.get("dedicated_routes") or {})
    if not heads or routes.get("enabled") is not True or routes.get("runtime_enabled") is not True:
        raise RuntimeError("checkpoint Fusion-v3 route contract is incomplete")
    return {
        "schema": fusion["schema"],
        "required": True,
        "required_heads": heads,
        "runtime_enabled": True,
        "typed_output_centered_routes": True,
        "route_reliability_bounds": list(fusion.get("reliability_bounds") or [0.25, 4.0]),
        "action_type_reliability_cap": float(fusion.get("action_type_reliability_cap", 0.25)),
        "dedicated_routes": routes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--specialist-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    manifest = _load(run / "manifest.json")
    initial = dict(manifest.get("initial_learner_checkpoint") or {})
    initial_path = Path(str(initial.get("path") or "")).expanduser().resolve()
    initial_digest = str(initial.get("digest") or "")
    if not initial_path.is_file() or not initial_digest.startswith("sha256:"):
        raise RuntimeError("run manifest lacks an initial learner checkpoint")
    if _sha(initial_path) != initial_digest:
        raise RuntimeError("initial learner digest does not match manifest")
    initial_payload = checkpoint.load_checkpoint(initial_path, map_location="cpu")
    fusion = _fusion_from_payload(initial_payload)
    if str(initial_payload.get("archetype_id") or "").casefold() != args.specialist_id.casefold():
        raise RuntimeError("initial checkpoint archetype does not match specialist")
    registry = _load(args.registry)
    row = dict((registry.get("specialists") or {}).get(args.specialist_id) or {})
    reg_fusion = dict(row.get("decision_fusion") or {})
    if reg_fusion.get("schema") != fusion["schema"] or reg_fusion.get("runtime_enabled") is not True:
        raise RuntimeError("runtime registry does not bind the same Fusion-v3 contract")
    registry_digest = _sha(args.registry)
    receipt = {
        "schema": "poke_bot.specialist_rl_activation/v2",
        "status": "ready",
        "mode": "terminal_incumbent_runtime_binding",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": {
            "specialist_id": args.specialist_id,
            "run_dir": str(run),
            "manifest": str(run / "manifest.json"),
            "manifest_sha256": _sha(run / "manifest.json"),
            "runtime_registry": str(args.registry.resolve()),
            "runtime_registry_sha256": registry_digest,
            "next_specialist_bootstrap": {
                "specialist_id": args.specialist_id,
                "checkpoint_digest": initial_digest,
                "checkpoint": str(initial_path),
                "decision_fusion": fusion,
            },
            "runtime_registration": {
                "specialist_id": args.specialist_id,
                "runtime_registry": str(args.registry.resolve()),
                "runtime_row": {
                    "initial_checkpoint_sha256": initial_digest.removeprefix("sha256:"),
                    "decision_fusion": {
                        **fusion,
                        "runtime_enabled": True,
                    },
                },
            },
        },
        "guide": {
            "retired": True,
            "shadow_only": True,
            "loss_weight": 0.0,
            "action_influence": False,
        },
        "authority": "terminal_owner_ceiling_handoff_only",
        "checkpoint_unchanged": True,
    }
    args.output = args.output.resolve()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite immutable receipt: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{__import__('os').getpid()}.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"receipt": str(args.output), "sha256": _sha(args.output), "status": "ready"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
