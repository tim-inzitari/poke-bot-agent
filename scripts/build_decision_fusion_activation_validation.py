#!/usr/bin/env python3
"""Combine immutable fusion audits into one fail-closed activation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.model import DECISION_FUSION_REQUIRED_HEADS  # noqa: E402


SCHEMA = "poke_bot.causal_decision_fusion_activation_validation/v1"
AUDIT_SCHEMA = "poke_bot.causal_decision_fusion_checkpoint_audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _protocol_limits(protocol_path: Path) -> tuple[float, int, bool]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    acceptance = protocol["specialist_training"]["decision_fusion"][
        "activation"
    ]["performance_acceptance"]
    return (
        float(acceptance["minimum_fused_decisions_per_second"]),
        int(acceptance["maximum_additional_peak_allocated_bytes"]),
        bool(acceptance["oom_allowed"]),
    )


def _numerical_parity(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    signatures = [
        dict(payload.get("deterministic_signature") or {})
        for _path, payload in records
    ]
    shapes = [
        tuple(int(value) for value in row.get("shape") or ())
        for row in signatures
    ]
    values = [
        [float(value) for value in row.get("values") or ()]
        for row in signatures
    ]
    if (
        not shapes
        or len(set(shapes)) != 1
        or len(shapes[0]) != 2
        or any(len(row) != math.prod(shapes[0]) for row in values)
    ):
        raise RuntimeError("local/remote deterministic signature shapes differ")
    reference = values[0]
    max_abs = max(
        (
            abs(left - right)
            for row in values[1:]
            for left, right in zip(reference, row)
        ),
        default=0.0,
    )
    columns = shapes[0][1]

    def decisions(row: list[float]) -> tuple[int, ...]:
        return tuple(
            max(
                range(columns),
                key=lambda column: row[offset + column],
            )
            for offset in range(0, len(row), columns)
        )

    decision_rows = [decisions(row) for row in values]
    decision_exact = len(set(decision_rows)) == 1
    if not math.isfinite(max_abs) or max_abs > absolute_tolerance:
        raise RuntimeError(
            "local/remote deterministic logits exceed float32 tolerance: "
            f"max_abs={max_abs:.9g} tolerance={absolute_tolerance:.9g}"
        )
    if not decision_exact:
        raise RuntimeError("local/remote deterministic greedy decisions differ")
    return {
        "absolute_tolerance": float(absolute_tolerance),
        "maximum_absolute_logit_delta": float(max_abs),
        "greedy_decisions_exact": True,
        "shape": list(shapes[0]),
    }


def _validate_audit(
    payload: dict[str, Any], *, digest: str, source: Path
) -> None:
    influence = dict(payload.get("influence") or {})
    causal = dict(payload.get("causal_contract") or {})
    signature = dict(payload.get("deterministic_signature") or {})
    if not (
        payload.get("schema") == AUDIT_SCHEMA
        and payload.get("checkpoint_digest") == digest
        and influence.get("required_head_count")
        == len(DECISION_FUSION_REQUIRED_HEADS)
        and influence.get("every_required_head_nonzero") is True
        and set(
            (influence.get("per_head_max_abs_ablation_delta") or {}).keys()
        )
        == set(DECISION_FUSION_REQUIRED_HEADS)
        and all(
            float(value) > 0.0
            for value in (
                influence.get("per_head_max_abs_ablation_delta") or {}
            ).values()
        )
        and signature.get("repeat_bit_exact") is True
        and causal.get("training_labels_enter_policy_observation") is False
        and causal.get("hidden_or_future_information_enter_policy_observation")
        is False
        and causal.get("matchup_adapter_route_handled_upstream") is True
        and causal.get("absent_deck_guide_exact_bypass") is True
    ):
        raise RuntimeError(f"incomplete fusion audit: {source}")


def build(
    *,
    checkpoint_path: Path,
    parity_audits: list[Path],
    performance_audit: Path,
    protocol_path: Path,
    output: Path,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    parity_audits = [
        value.expanduser().resolve() for value in parity_audits
    ]
    performance_audit = performance_audit.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("fusion activation validation is immutable")
    if len(parity_audits) < 2:
        raise RuntimeError("local/remote parity requires at least two audits")

    digest = checkpoint.checkpoint_digest(checkpoint_path)
    audit_records: list[tuple[Path, dict[str, Any]]] = []
    for path in parity_audits:
        payload = _read(path)
        _validate_audit(payload, digest=digest, source=path)
        audit_records.append((path, payload))
    hosts = {str(payload.get("host") or "") for _, payload in audit_records}
    if "" in hosts or len(hosts) < 2:
        raise RuntimeError("parity audits must come from distinct named hosts")
    parity = _numerical_parity(audit_records)

    performance = _read(performance_audit)
    _validate_audit(performance, digest=digest, source=performance_audit)
    perf = dict(performance.get("performance") or {})
    minimum_fused_dps, maximum_added_bytes, oom_allowed = _protocol_limits(
        protocol_path
    )
    measured_regression = float(perf.get("measured_regression_percent", math.inf))
    fused_dps = float(perf.get("fused_decisions_per_second") or 0.0)
    added_bytes = int(perf.get("additional_peak_allocated_bytes") or 0)
    performance_passed = (
        math.isfinite(fused_dps)
        and fused_dps >= minimum_fused_dps
        and added_bytes <= maximum_added_bytes
        and (bool(perf.get("oom")) is False or oom_allowed)
    )
    if not performance_passed:
        raise RuntimeError(
            "fusion rollout performance did not satisfy the serving floor: "
            f"dps={fused_dps:.3f}/{minimum_fused_dps:.3f} "
            f"added_bytes={added_bytes}/{maximum_added_bytes} "
            f"oom={bool(perf.get('oom'))}/{oom_allowed}"
        )

    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_digest": digest,
        "passed": True,
        "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
        "checks": {
            "trained_nonzero_fusion": True,
            "every_required_head_influence_nonzero": True,
            "deterministic_local_remote_logit_parity": True,
            "causal_information_audit": True,
            "rollout_throughput_and_memory": True,
        },
        "parity": {
            "hosts": sorted(hosts),
            "mode": "float32_numerical_logits_and_exact_greedy_decisions",
            **parity,
            "audits": [
                {"path": str(path), "digest": _sha256(path)}
                for path, _ in audit_records
            ],
        },
        "performance": {
            "host": performance.get("host"),
            "device": performance.get("device"),
            "audit": str(performance_audit),
            "audit_digest": _sha256(performance_audit),
            "measured_regression_percent": measured_regression,
            "relative_microbenchmark_regression_diagnostic_only": True,
            "fused_decisions_per_second": fused_dps,
            "minimum_fused_decisions_per_second": minimum_fused_dps,
            "oom": bool(perf.get("oom")),
            "additional_peak_allocated_bytes": added_bytes,
            "maximum_additional_peak_allocated_bytes": maximum_added_bytes,
        },
        "protocol": {
            "path": str(protocol_path),
            "digest": _sha256(protocol_path),
        },
    }
    _exclusive_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument(
        "--parity-audit", dest="parity_audits", type=Path, action="append",
        required=True,
    )
    parser.add_argument("--performance-audit", type=Path, required=True)
    parser.add_argument(
        "--protocol", dest="protocol_path", type=Path,
        default=ROOT / "config/rl_protocol.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
