"""Deterministic r195-first / r274-additions composite warm start.

This module only assembles checkpoint state in memory.  It does not load a
runtime, start training, mutate either parent, or write an artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
SCHEMA = "poke_bot.alakazam_r195_r274_composite_warmstart/v1"


class CompositeWarmstartError(RuntimeError):
    pass


def _shape(value: Any) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise CompositeWarmstartError("checkpoint state contains a non-tensor value")
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise CompositeWarmstartError("tensor shape is malformed") from exc


def _clone(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    clone = getattr(value, "clone", None)
    if not callable(clone):
        raise CompositeWarmstartError("checkpoint tensor is not cloneable")
    return clone()


def _matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


@dataclass(frozen=True)
class CompositeWarmstart:
    state_dict: dict[str, Any]
    source_map: dict[str, str]
    receipt: dict[str, Any]


def assemble_composite_warmstart(
    *,
    target_state: Mapping[str, Any],
    r195_state: Mapping[str, Any],
    r274_state: Mapping[str, Any],
    new_parameter_prefixes: Sequence[str],
    r195_checkpoint_sha256: str = R195_CHECKPOINT_SHA256,
    r274_checkpoint_sha256: str,
) -> CompositeWarmstart:
    """Fill the target deterministically using the revision-7 source order.

    Exact target-name and shape matches from r195 win.  r274 may fill only a
    name absent from r195.  Names absent from both must be explicitly declared
    genuinely new.  A same-name r195 shape mismatch is never silently replaced
    by r274.
    """

    if not target_state:
        raise CompositeWarmstartError("target state is empty")
    if not r195_checkpoint_sha256.startswith("sha256:") or not r274_checkpoint_sha256.startswith("sha256:"):
        raise CompositeWarmstartError("parent checkpoint identity is malformed")
    if len(set(new_parameter_prefixes)) != len(tuple(new_parameter_prefixes)):
        raise CompositeWarmstartError("new-parameter prefix inventory contains duplicates")

    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    counts = {"r195": 0, "r274_architecture_addition": 0, "new_initialization": 0}
    for name in sorted(target_state):
        target = target_state[name]
        target_shape = _shape(target)
        if name in r195_state:
            if _shape(r195_state[name]) != target_shape:
                raise CompositeWarmstartError(f"r195 tensor shape drift: {name}")
            merged[name] = _clone(r195_state[name])
            sources[name] = "r195"
            counts["r195"] += 1
            continue
        if name in r274_state:
            if _shape(r274_state[name]) != target_shape:
                raise CompositeWarmstartError(f"r274 addition tensor shape drift: {name}")
            merged[name] = _clone(r274_state[name])
            sources[name] = "r274_architecture_addition"
            counts["r274_architecture_addition"] += 1
            continue
        if not _matches_prefix(name, new_parameter_prefixes):
            raise CompositeWarmstartError(f"unclassified target tensor: {name}")
        merged[name] = _clone(target)
        sources[name] = "new_initialization"
        counts["new_initialization"] += 1

    extra_r195 = sorted(set(r195_state) - set(target_state))
    source_map_digest = "sha256:" + hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema": SCHEMA,
        "status": "assembled_in_memory_not_training_active",
        "r195_checkpoint_sha256": r195_checkpoint_sha256,
        "r274_checkpoint_sha256": r274_checkpoint_sha256,
        "source_precedence": ["r195_exact_name_shape", "r274_name_absent_from_r195", "explicit_new_initialization"],
        "tensor_count": len(merged),
        "source_counts": counts,
        "per_tensor_source_map_sha256": source_map_digest,
        "unused_r195_tensor_names": extra_r195,
        "all_target_tensors_classified_exactly_once": len(merged) == len(target_state) == len(sources),
        "parents_mutated": False,
        "training_authority": False,
    }
    return CompositeWarmstart(merged, sources, receipt)


__all__ = [
    "CompositeWarmstart",
    "CompositeWarmstartError",
    "R195_CHECKPOINT_SHA256",
    "SCHEMA",
    "assemble_composite_warmstart",
]
